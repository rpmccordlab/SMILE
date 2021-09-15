# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 19:23:52 2021

@author: Yang Xu
"""

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import io
from scipy.sparse import csr_matrix

from SMILE import SMILE
from SMILE.SMILE import PairedSMILE_trainer

import torch
import torch.nn.functional as F

import anndata
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

import seaborn as sns

def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform(m.weight.data)
        m.bias.data.zero_()
        
##-----------------------------------------------------------------------------
##1. load datasets from both modalities
##here we use mouse skin data we used to reproduce results in Figure 3e

##both RNA-seq and ATAC-seq (gene activity) data, including cell-type annotation can be found at
##https://figshare.com/articles/dataset/Mouse_skin_data_by_SHARE-seq/16620367
rna = "D:/GEO/SMILE/shareseq/GSM4156608_skin.late.anagen.rna.counts.txt.gz"
rna = pd.read_csv(rna,header=0,index_col=0,sep="\t")
rna = rna.T

dna = "D:/GEO/SMILE/shareseq/skin_gene_act.mtx"
dna = io.mmread(dna)
dna= csr_matrix(dna)
dna = dna.T

atac_genes = "D:/GEO/SMILE/shareseq/skin_atac_gene.txt"
atac_genes = pd.read_csv(atac_genes,sep="\t",index_col=False,header=None)
atac_genes = atac_genes[0].values.tolist()
dna.columns = atac_genes

dna_y = "D:/GEO/SMILE/shareseq/GSM4156597_skin.late.anagen.barcodes.txt.gz"
dna_y = pd.read_csv(dna_y,header=None,index_col=0,sep=",")
new_index = [i.replace(',','.') for i in rna.index]
rna.index = new_index

##get the common cells in both modalities
commons=[]
for i in rna.index:
    if i in dna_y.index:
        commons.append(True)
    else:
        commons.append(False)
commons = np.array(commons)
rna = rna[commons]
rna = rna.sort_index(ascending=False)

##cell-type label
rna_y = "D:/GEO/SMILE/shareseq/GSM4156597_skin_celltype.txt.gz"
rna_y = pd.read_csv(rna_y,header=0,index_col=0,sep="\t")
rna_y = rna_y.sort_index(ascending=False)

rna = rna[rna_y['celltype'].values!="Mix"]

adata = anndata.AnnData(X=rna)
adata.obs['CellType']=rna_y['celltype'].values[rna_y['celltype'].values!="Mix"]
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, min_mean=0.15, max_mean=5, min_disp=0.75,subset=True)##
hvg = adata.var.index[adata.var['highly_variable'].values]

sc.tl.rank_genes_groups(adata, 'CellType', method='wilcoxon')##identify differential genes
markers_table = pd.DataFrame(adata.uns['rank_genes_groups']['names'])
rna_X = adata.X[:,adata.var['highly_variable'].values]
cells = rna_y['celltype'].values


adata= anndata.AnnData(X=dna)
sc.pp.highly_variable_genes(adata, min_mean=0.15, max_mean=5, min_disp=0.75)
dna_X = adata.X[:,adata.var['highly_variable'].values].todense()
hvg_atac = np.array(atac_genes)[adata.var['highly_variable'].values]
dna_X = pd.DataFrame(dna_X)
dna_X.columns = hvg_atac
dna_X.index = dna_y.index

dna_X = dna_X.sort_index(ascending=False)
dna_X = dna_X[cells!="Mix"]

adata= anndata.AnnData(X=dna_X)
cells = cells[cells!="Mix"]
adata.obs['CellType']=cells

sc.tl.rank_genes_groups(adata, 'CellType', method='wilcoxon')##identify differential gene activities
markers_atac_table = pd.DataFrame(adata.uns['rank_genes_groups']['names'])

##scale data
scaler = StandardScaler()
dna_X = scaler.fit_transform(dna_X.values)
scaler = StandardScaler()
rna_X = scaler.fit_transform(rna_X)

##-----------------------------------------------------------------------------
##2. Use mpSMILE for integration
clf_out = 25
net = SMILE.Paired_SMILE(input_dim_a=rna_X.shape[1],
                         input_dim_b=dna_X.shape[1],clf_out=clf_out)
net.apply(weights_init)##initialize weights
PairedSMILE_trainer(X_a = rna_X, X_b = dna_X, model = net, num_epoch=20)##training for 20 epochs
        
##-----------------------------------------------------------------------------
##3. Screen genes that contribute to coembedding
##for example here we focus on high CD34 bulge cells
test_rna = rna_X[cells=="ahighCD34+ bulge",:].copy()
test_dna = dna_X[cells=="ahighCD34+ bulge",:].copy()

X_all_tensor_a = torch.tensor(test_rna).float()
X_all_tensor_b = torch.tensor(test_dna).float()

##forward the original data through the trained SMILE model
net.to(torch.device("cpu"))
y_pred_a = net.encoder_a(X_all_tensor_a)
y_pred_a = F.normalize(y_pred_a, dim=1,p=2)
y_pred_a = torch.Tensor.cpu(y_pred_a).detach().numpy()
y_pred_b = net.encoder_b(X_all_tensor_b)
y_pred_b = F.normalize(y_pred_b, dim=1,p=2)
y_pred_b = torch.Tensor.cpu(y_pred_b).detach().numpy()

y_pred = np.concatenate((y_pred_a, y_pred_b),axis=0)
pca = PCA(n_components=2)
pca.fit(np.concatenate((y_pred_a,y_pred_b),0))
y_pred_b = pca.transform(y_pred_b)
y_pred_a = pca.transform(y_pred_a)

##we use PCA-reduced space to calculate distance between two modalities
dist_ori= np.linalg.norm(y_pred_a-y_pred_b,axis=1)
ori_dis= np.mean(dist_ori)


##------------------
##code below to show how to screen genes in ATAC-seq that contribute coembedding
##the same approach can also be applied to RNA-seq

##the top 150 differentil gene activities in highCD34+ bulge cells
markers_atac = markers_atac_table.values[:150,20].flatten()#.astype(int)##ahighCD34+ bulge
imp_genes_atac = []
for i in hvg_atac:
    if i in markers_atac:
        imp_genes_atac.append(True)
    else:
        imp_genes_atac.append(False)

disrupt_dna = test_dna.copy()
disrupt_dna[:,np.array(imp_genes_atac)]=0##supress values to 0
##forward disrupted atac-seq data through model
X_all_tensor_b = torch.tensor(disrupt_dna).float()
y_pred_b = net.encoder_b(X_all_tensor_b)
y_pred_b = F.normalize(y_pred_b, dim=1,p=2)
y_pred_b = torch.Tensor.cpu(y_pred_b).detach().numpy()
y_pred_b = pca.transform(y_pred_b)
dist_diff = np.linalg.norm(y_pred_a-y_pred_b,axis=1)

##screen genes one by one
mean_dis = []
for i in range(test_dna.shape[1]):
    disrupt_dna = test_dna.copy()
    disrupt_dna[:,i]=0## one gene is supressed to 0
    X_all_tensor_b = torch.tensor(disrupt_dna).float()
    y_pred_b = net.encoder_b(X_all_tensor_b)
    y_pred_b = F.normalize(y_pred_b, dim=1,p=2)
    y_pred_b = torch.Tensor.cpu(y_pred_b).detach().numpy()
    y_pred_b = np.nan_to_num(y_pred_b)
    y_pred_b = pca.transform(y_pred_b)
    mean_dis.append(np.mean(np.linalg.norm(y_pred_a-y_pred_b,axis=1))-ori_dis)
mean_dis = np.array(mean_dis)

##we keep genes in the top 95 percentile and calculate their collective disruption again
disrupt_dna = test_dna.copy()
disrupt_dna[:,mean_dis>=np.quantile(mean_dis,0.95)]=0
X_all_tensor_b = torch.tensor(disrupt_dna).float()
y_pred_b = net.encoder_b(X_all_tensor_b)
y_pred_b = F.normalize(y_pred_b, dim=1,p=2)
y_pred_b = torch.Tensor.cpu(y_pred_b).detach().numpy()
y_pred_b = pca.transform(y_pred_b)
screen_out = np.linalg.norm(y_pred_a-y_pred_b,axis=1)

##-----------------------------------------------------------------------------
##4. Plot our outcome
dist1 = pd.DataFrame(dist_ori)
dist1['group']="1.Original data"
dist1.columns = ['x','group']

##disruption by random 150 genes
disrupt_dna = test_dna.copy()
index = np.random.permutation(test_dna.shape[1])[:150]
disrupt_dna[:,index]=0
X_all_tensor_b = torch.tensor(disrupt_dna).float()
y_pred_b = net.encoder_b(X_all_tensor_b)
y_pred_b = F.normalize(y_pred_b, dim=1,p=2)
y_pred_b = torch.Tensor.cpu(y_pred_b).detach().numpy()
y_pred_b = pca.transform(y_pred_b)
dist2 = np.linalg.norm(y_pred_a-y_pred_b,axis=1)
dist2 = pd.DataFrame(dist2)
dist2['group']="2.Random genes"
dist2.columns = ['x','group']

##differential genes
dist3 = pd.DataFrame(dist_diff)
dist3['group']="3.Top diffirential genes"
dist3.columns = ['x','group']

##screened genes
dist4 = pd.DataFrame(screen_out)
dist4['group']="4.Top screened genes"
dist4.columns = ['x','group']

dist = pd.concat((dist1,dist2,dist3,dist4),axis=0)
dist.columns = ['Distance','group']
dist = dist.sort_values(by=['group'])
f = sns.catplot(x="group", y="Distance",hue="group",data=dist, kind="box",showfliers=False)
f.fig.set_figwidth(8)#32
f.fig.set_figheight(6)
#f.savefig("Skin_distance_by_gene_atac_DermalFibroblast.pdf",dpi=1200)