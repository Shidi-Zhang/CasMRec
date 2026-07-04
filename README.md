# CaSMRec
This is the data and code for our paper `CaSMRec: Causal Substructure-aligned Medication Recommendation`.

## Prerequisites
Make sure your local environment has the following installed:

* `torch = 2.0.1`
* `python == 3.8.17`
* `dill == 1.22.3`
* `pandas == 2.0.2`
* `numpy == 1.22.3`
* `cdt == 0.6.0`
* `dowhy == 0.10.1`
* `statsmodels == 0.14.0`

## Datasets

We provide the dataset in the [data](data/) folder.

| Data      | Source                                                   | Description                                                  |
| --------- | -------------------------------------------------------- | ------------------------------------------------------------ |
| MIMIC-III | [This link](https://physionet.org/content/mimiciii/1.4/) | MIMIC-III is freely-available database from 2001 to 2012, which is associated with over forty thousand patients who stayed in critical care units |
| MIMIC-IV  | [This link](https://physionet.org/content/mimiciv/2.2/)  | MIMIC-IV is freely-available database between 2008 - 2019, which is associated with 299,712 patients who stayed in critical care units |

## Documentation


* src
    * `modules/`:Code for model definition
    * `utils.py`: Code for data preparation and indicator calculation
    * `training.py`:Code for functions that train and evaluate the model
    * `main.py`:Code for training and evaluating the model
  
* `datas/`
    * `input/`
        * `drug-atc.csv`: drug to atc code mapping file.
        * `ndc2atc_level4.csv`: NDC to RXCUI mapping file.
        * `ndc2rxnorm_mapping.txt`: NDC to RXCUI mapping file.
        * `idx2ndc.pkl`：ACT-4 to rxnorm mapping file.
        * `idx2drug.pkl`：ACT-4 to SMILES mapping file.
    * `output/`
        * `ddi_A_final.pkl`：ddi adjacency matrix
        * `records_final.pkl`: the final diagnosis-procedure-medication EHR records of each patient, used for train/val/test split on MIMIC_III dataset
        * `voc_final.pkl`：diag/prod/med index to code dictionary on MIMIC_III dataset
        * `ddi_matrix_H.pkl`：H mask structure
    * `graphs/`
        * `causal_graph.pkl`：causal graph
        * `Diag_Med_causal_effect.pkl`,`Proc_Med_causal_effect.pkl`：causal effectss of med and diag/proc
    * `ddi_mask_H.py`：the python code for generate `ddi_mask_H.pkl` and `substrucure_smiles.pkl`.
    * `processing.py:`:the python code for generage `voc_final.pkl`,`records_final.pkl`,`ddi_A_final.pkl`.


## Step 1: Data Processing 

* Download the MIMIC-III/MIMIC-IV dataset from [MIMIC-III link](https://physionet.org/content/mimiciii/1.4/) / [MIMIC-IV link](https://physionet.org/content/mimiciv/1.4/)

* Extract three main files(PROCEDURES_ICD.csv.gz, PRESCRIPTIONS.csv.gz, DIAGNOSES_ICD.csv.gz), and change the path comply with current state:
```
# med_file = '/data/mimic-iii/PRESCRIPTIONS.csv'
# diag_file = '/data/mimic-iii/DIAGNOSES_ICD.csv'
# procedure_file = '/data/mimic-iii/PROCEDURES_ICD.csv'
```
## Step 2: Package Dependency
First, install the [conda](https://www.anaconda.com/)

## Step 4: data processing
```
python data/processing.py
python data/ddi_mask_H.py
python src/Relevance_construction.py
```
## Step 3: run the code
```
python main.py
```

### Acknowledge

Partial credit to previous reprostories:

- https://github.com/sjy1203/GAMENet
- https://github.com/ycq091044/SafeDrug
- https://github.com/yangnianzu0515/MoleRec
- https://github.com/lixiang-222/CIDGMed
