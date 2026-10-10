# Multimodal Omics Missing-Data Imputation

This repository contains a cleaned portfolio version of an MSc Biostatistics practicum analysis comparing missing-data imputation methods for high-dimensional multimodal omics data.

## Project question

How accurately can missing plasma RNA values be recovered, and does adding information from other omics modalities improve imputation performance?

The benchmark injects artificial MCAR missingness into plasma features and compares five approaches:

- mean imputation
- median imputation
- K-nearest-neighbors (KNN) imputation
- scikit-learn `IterativeImputer`
- ChainImputer, a neural-network iterative imputation method

Performance is evaluated only on artificially masked plasma cells using MAE and RMSE across repeated random seeds.

## Data

The analysis uses public multimodal osteoarthritis datasets from the `DeepLearning_KOA` repository maintained by Divya Sharma:

https://github.com/divya031090/DeepLearning_KOA

The source repository contains `RNA_plasma.csv`, `RNA_synovial.csv`, `RNA_urine.csv`, and `metabolites.csv`.

This portfolio repository does **not** redistribute those datasets. By default, the script reads them directly from the public source repository. You can alternatively download the four CSV files yourself and pass their directory with `--data-dir`.

## Methods

For each evaluation seed, the workflow:

1. aligns participants across modalities using ID columns when available;
2. injects a prespecified percentage of MCAR missingness into plasma features only;
3. standardizes masked plasma using statistics calculated from its observed cells;
4. standardizes the complete plasma truth using those same masked-data statistics;
5. optionally adds metabolomics or synovial RNA predictors;
6. imputes the missing values;
7. evaluates only the artificially masked plasma entries.

The main benchmark evaluates 10%, 30%, and 50% missingness across 20 random seeds.

## Predictor configurations

- plasma only
- plasma + metabolites
- plasma + synovial RNA

## Reproducibility

### 1. Clone this repository

```bash
git clone <YOUR-REPOSITORY-URL>
cd multimodal-omics-imputation
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run a quick demonstration

```bash
python multimodal_omics_imputation.py --quick
```

For a quicker run without the neural-network method:

```bash
python multimodal_omics_imputation.py --quick --skip-chain
```

### 5. Run the full benchmark

```bash
python multimodal_omics_imputation.py
```

Outputs are written to `results/` and `figures/`.


## Example outputs

The results and figures included in this repository were generated using the
`--quick` option and are provided as a demonstration that the analysis pipeline
runs successfully. These example outputs use 10% missingness, the plasma-only
predictor set, and two evaluation seeds (11 and 12).

The full analysis implemented in the script evaluates 10%, 30%, and 50%
missingness across 20 seeds (11–30) and multiple predictor configurations.
Because the full benchmark is computationally intensive, its outputs are not
included in this repository.


## Technical skills demonstrated

- Python
- pandas / NumPy
- scikit-learn
- PyTorch
- high-dimensional omics preprocessing
- multimodal data integration
- missing-data simulation
- imputation
- repeated-seed evaluation
- MAE/RMSE model comparison
- reproducible analysis workflows

## ChainImputer attribution

The `ChainImputer` class included in this repository is adapted from:

**Seo W, Khairulov T, Baek H-J, Lee J. ChainImputer: A Neural Network-Based Iterative Imputation Method Using Cumulative Features. Symmetry. 2025;17(6):869. doi:10.3390/sym17060869**

Source code:

https://github.com/KhrTim/ChainImputer

The upstream implementation is distributed under the MIT License. The required notice is reproduced in `THIRD_PARTY_NOTICES.md`.

## Notes

`IterativeImputer` is used as a single completed iterative chained-regression imputation procedure in this benchmark. It is therefore described as `IterativeImputer` rather than as full multiple imputation.

The parameter values stored in `BEST_PARAMS` were selected during the original practicum tuning workflow. The public script keeps them fixed for final repeated-seed evaluation so that the portfolio version remains concise and easy to review.
