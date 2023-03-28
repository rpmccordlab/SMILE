# SMILE
SMILE: Mutual Information Learning for Integration of Single Cell Omics Data

<img src="https://github.com/rpmccordlab/SMILE/blob/main/SMILE_logo.jpg" width="696" height="331">

## Citation
Xu et al. "SMILE: mutual information learning for integration of single-cell omics data". <a href="https://academic.oup.com/bioinformatics/article-abstract/38/2/476/6384571">Bioinformatics</a>

## Requirements
###
* numpy
* scipy
* pandas
* scikit-learn
* scanpy
* pytorch

## Updates

### Update 06/30/2022 (Identifying shared signatures across modalities)

<img src="https://github.com/rpmccordlab/SMILE/blob/main/Tutorial/littleSMILE.jpg" width="780" height="540">

To identify shared signatures across modalities via SMILE, please see tutorial 

    |----SMILE_identify_shared_signatures_across_modalities.ipynb

### Update 05/09/2022 (Using joint-profiling data as reference for integration)
    
    ##rna_X: RNA-seq data; dna_X: ATAC-seq data, rna_X and dna_X are paired data
    ##rna_X_unpaired: RNA-seq data; dna_X_unpaired: ATAC-seq data, rna_X_unpaired and dna_X_unpaired are unpaired data, and we wish to integrate unpaired data
    ##Both rna_X and dna_X are matrices in which each row represents one cell while each column stands for a feature
    ##each row in rna_X and dna_X is paired for training purpose
    ##Within modality, for example rna_X and rna_X_unpaired, data share the same feature space
    
    ## Proecessed data could be found at https://doi.org/10.5281/zenodo.7776066
    
    from SMILE import littleSMILE
    from SMILE import ReferenceSMILE_trainer
    integrater = littleSMILE(input_dim_a=rna_X.shape[1],input_dim_b=dna_X.shape[1],clf_out=20)
    ReferenceSMILE_trainer(rna_X,dna_X,rna_X_unpaired,dna_X_unpaired, integrater, train_epoch=1000)

Integrate RNA-seq/ATAC-seq from Granja et al 2019 with 10X multiome PBMC data as reference
    
    |----SMILE_data_integration_withReference.ipynb

## Tutorial

### For quick start
    ##rna_X: RNA-seq data; dna_X: ATAC-seq data
    ##Both rna_X and dna_X are matrices in which each row represents one cell while each column stands for a feature
    ##each row in rna_X and dna_X is paired for training purpose 
    
    from SMILE import SMILE
    from SMILE import PairedSMILE_trainer
    net = littleSMILE(input_dim_a=rna_X.shape[1],input_dim_b=dna_X.shape[1],clf_out=30)
    ReferenceSMILE_trainer(X_rna_paired,X_dna_paired,X_rna_unpaired,X_dna_unpaired,net,batch_size=1024, f_temp = 0.2)

### For detail
    For use of SMILE in multi-source single-cell transcriptome data

    |----SMILE_MouserCortex_RNAseq_example.ipynb

    For use of SMILE in integration of multimodal single-cell data

    |----SMILE_Celllines_RNA-ATAC-integration_example.ipynb
    
### For screening key genes that contribute co-embedding

    |----screen_factors_for_coembedding.py
    Processed data can be found at https://figshare.com/articles/dataset/Mouse_skin_data_by_SHARE-seq/16620367
