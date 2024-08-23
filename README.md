## Cell Shape Prediction Proof of Concept
This repository is going to be divided into two parts: 
1) Data preprocessing  
   * Filtering out compounds with low number of repeats
   * Plate-wise Z-scoring parameter values against average values of control samples 
   * Random split and Tanimoto similarity-based stratified split of the data for model training
   

2) Model training (adding soon)
    * One-hot encoding of SMILES
    * Encoding SMIlES into latent vectors (using [NYAN encoder](https://github.com/Chokyotager/NotYetAnotherNightshade/blob/main/README.md))
    * Random forest training module
    * Deep learning training module