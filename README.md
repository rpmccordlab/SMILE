# SMILE
SMILE: Mutual Information Learning for Integration of Single Cell Omics Data 

<img src="https://github.com/rpmccordlab/SMILE/blob/main/SMILE_logo.jpg" width="696" height="331">

# Requirements
###
* numpy
* scipy
* pandas
* scikit-learn
* scanpy
* anndata
* pytorch

## Update 05/09/2022
    ##rna_X: RNA-seq data; dna_X: ATAC-seq data, rna_X and dna_X are paired data
    ##rna_X_unpaired: RNA-seq data; dna_X_unpaired: ATAC-seq data, rna_X_unpaired and dna_X_unpaired are unpaired data. We could integrate unpaired data
    ##Both rna_X and dna_X are matrices in which each row represents one cell while each column stands for a feature
    ##each row in rna_X and dna_X is paired for training purpose
    ##Within modality, for example rna_X and rna_X_unpaired, data share the same feature space
    
    from SMILE import SMILE
    from SMILE.SMILE import ReferenceSMILE_trainer
    net = SMILE.Paired_SMILE(input_dim_a=rna_X.shape[1],input_dim_b=dna_X.shape[1],clf_out=25)
    ReferenceSMILE_trainer(X_a = rna_X, X_b = dna_X,X_a_unpaired = rna_X_unpaired, X_b_unpaired = dna_X_unpaired, model = net, num_epoch=50)

# Tutorial

### For quick start
    ##rna_X: RNA-seq data; dna_X: ATAC-seq data
    ##Both rna_X and dna_X are matrices in which each row represents one cell while each column stands for a feature
    ##each row in rna_X and dna_X is paired for training purpose 
    
    from SMILE import SMILE
    from SMILE.SMILE import PairedSMILE_trainer
    net = SMILE.Paired_SMILE(input_dim_a=rna_X.shape[1],input_dim_b=dna_X.shape[1],clf_out=25)
    PairedSMILE_trainer(X_a = rna_X, X_b = dna_X, model = net, num_epoch=10)

### For detail
    For use of SMILE in multi-source single-cell transcriptome data

    |----SMILE_MouserCortex_RNAseq_example.ipynb

    For use of SMILE in integration of multimodal single-cell data

    |----SMILE_Celllines_RNA-ATAC-integration_example.ipynb
    
### For screening key genes that contribute co-embedding

    |----screen_factors_for_coembedding.py
    Processed data can be found at https://figshare.com/articles/dataset/Mouse_skin_data_by_SHARE-seq/16620367
