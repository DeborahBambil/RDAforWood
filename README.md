 # RDAforWood

**Authors:** Deborah Bambil and Julia Sonsin-Oliveira

## Statistical Approach: Redundancy Analysis (RDA)

This workflow implements **Redundancy Analysis (RDA)** to assess how wood anatomical features (cell dimensions and tissue proportions) explain variations in physical properties (density, shrinkage, and anisotropy)

### Key Features:
* **Multivariate Integration:** Combines PCA with multiple linear regression to determine the variance proportion explained by predictors
* **Data Pre-processing:**
* **CLR Transformation:** Applied to compositional data (tissue proportions) to handle unit sum constraints
* **Standardization:** All variables are Z-score scaled for comparability across different units
* **Tech Stack:** Built with **Python** using `Scikit-learn`, `Pandas`, `Numpy`, and `Seaborn`

## Installation

1. **Install Dependencies on Win:**
   $ py -m pip install pandas numpy matplotlib seaborn scikit-learn statsmodels scikit-bio

2. **Start:**
    $ RDAforWood.py

 ![name-of-you-image](https://github.com/DeborahBambil/figs/blob/main/rda_VS.png)



