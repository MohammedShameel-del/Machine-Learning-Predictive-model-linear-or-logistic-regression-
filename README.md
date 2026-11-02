# Machine-Learning-Predictive-model-linear-or-logistic-regression-
Predicts diamond prices from 53,940 records using linear regression. Data was cleaned of duplicates and invalid dimensions, cut, color and clarity were ordinal-encoded, and price and carat were log-transformed. The model reaches R² ≈ 0.93 with a ~$463 MAE. Carat is the dominant driver, followed by clarity, color and cut.
# Diamond Price Prediction

Predicts diamond prices from physical and quality attributes using linear regression, with a full preprocessing and evaluation pipeline in a Jupyter Notebook.

## Data

- **Source file:** `diamon.txt` (CSV format), the classic diamonds dataset
- **Size:** 53,940 rows × 10 columns (53,772 rows after cleaning)
- **Target:** `price` (USD)

| Column | Type | Description |
|---|---|---|
| carat | numeric | Diamond weight |
| cut | ordinal | Fair < Good < Very Good < Premium < Ideal |
| color | ordinal | J (worst) → D (best) |
| clarity | ordinal | I1 < SI2 < SI1 < VS2 < VS1 < VVS2 < VVS1 < IF |
| depth | numeric | Total depth % = z / mean(x, y) |
| table | numeric | Width of top of diamond relative to widest point |
| x, y, z | numeric | Length, width, depth in mm |
| price | numeric | Price in US dollars (target) |

**Data quality notes:** no null values, but 146 duplicate rows, zero values in x/y/z (physically impossible), and 3 extreme outliers (e.g. y = 58.9 mm) were found and removed.

## Methodology

1. **Objective:** predict `price` (regression).
2. **Cleaning:** removed duplicates, rows with zero dimensions, and dimension outliers.
3. **Encoding:** ordinal encoding for `cut`, `color`, `clarity` using their true quality order.
4. **Transformation:** log-transformed `price` (skewed) and `carat` (curved relationship with price).
5. **Scaling:** `StandardScaler` on numeric features, inside a scikit-learn `Pipeline` to prevent data leakage.
6. **Feature selection:**
   - Kept: `carat`, `cut`, `color`, `clarity`
   - Dropped: `x`, `y`, `z` (correlated 0.95–0.98 with carat, so redundant)
   - Dropped: `depth`, `table` (near-zero effect once carat is included)
7. **Split:** 80% train / 20% test (`random_state=42`).
8. **Model:** `LinearRegression`.
9. **Evaluation:** MAE, RMSE, R² on the test set, with predictions converted back to dollars.

## Results

| Metric | Test set |
|---|---|
| MAE | ≈ $463 |
| RMSE | ≈ $1,044 |
| R² (dollars) | ≈ 0.93 |
| R² (log scale) | ≈ 0.98 |
| Median error | ≈ 9% |

## Key Findings

- **Carat is the dominant driver** of price, followed by clarity, color, and then cut.
- The model explains about 93% of price variation, with a typical error of about 9%.
- Log-transforming price and carat was essential for a good linear fit.

## Limitations

- Errors grow for higher-priced diamonds.
- Ordinal encoding assumes equal value gaps between grades.
- Does not capture interactions or price jumps at round carat sizes.
- Reflects one dataset snapshot; correlation, not causation.

## Future Work

- Compare with Random Forest / Gradient Boosting
- One-hot encoding and interaction terms
- Cross-validation and hyperparameter tuning
- Optional classification task (expensive vs. not) with logistic regression

## Tech Stack

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · Jupyter Notebook

## How to Run

```bash
git clone <your-repo-url>
cd <repo-name>
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook
```
Open the notebook and run all cells.
