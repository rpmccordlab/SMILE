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

# Tutorial

### For quick start
    from SMILE import SMILE
    from SMILE.SMILE import PairedSMILE_trainer
    net = SMILE.Paired_SMILE(input_dim_a=rna_X.shape[1],input_dim_b=dna_X.shape[1],clf_out=25)##rna_X: RNA-seq data; dna_X: ATAC-seq data
    PairedSMILE_trainer(X_a = rna_X, X_b = dna_X, model = net, num_epoch=10)

### For detail
    For use of SMILE in multi-source single-cell transcriptome data

    |----SMILE_MouserCortex_RNAseq_example.ipynb

    For use of SMILE in integration of multimodal single-cell data

    |----SMILE_Celllines_RNA-ATAC-integration_example.ipynb
