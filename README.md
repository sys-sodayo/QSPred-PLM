# IQSPred-PLM: An Interpretable Quorum Sensing Peptides Prediction Model Based on Protein Language Model

Quorum sensing regulates cooperative behaviors in bacteria through the accumulation and detection of signaling molecules. This process plays a crucial role in various biological functions, including biofilm formation, antibiotic production, regulation of virulence factors, and immune modulation.Within this framework, quorum sensing peptides (QSPs) serve as essential signaling molecules that mediate bacterial communication both within and between species, making them critical to understanding quorum sensing and its regulatory functions. Here, we propose IQSPred-PLM, a robust and interpretable deep learning model for accurate QSP prediction.

![新流程图](https://github.com/user-attachments/assets/f41ff016-9dc2-4af3-86b2-c01cfaa07b74)


IQSPred-PLM relies on a large-scale pre-trained protein language models: ESM-2. For detailed guidance on generating protein embedding representations, please refer to the official documentation available at the following websites:

- ESM-2:https://github.com/facebookresearch/esm

## Package requirement
```
pytorch==2.6.0+cu126  
scikit-learn==1.3.0
pandas==2.0.3
numpy==1.24.3
```

## Test on the model

### 1. Prepare Test Data and Labels

Ensure your test data and corresponding labels are ready and match the required input format for the model. You can set them up in the `getdata.py` script.

### 2. Download the Model Weights

Our model can be download from : https://drive.google.com/file/d/1-PLw9vhfQprivpGTy9nE_EeFkV1hQyO5/view?usp=drive_link.

### 3. Run the Test Script
To test the model, run the following command:
```bash
python test.py
```
