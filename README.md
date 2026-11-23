# MTH-686: Nonlinear Regression Project

This project studies a dataset of 60 observations, \((t, y(t))\), by fitting and comparing three regression models. The work covers exploratory data analysis, least-squares parameter estimation, residual diagnostics, approximate Fisher-information-based confidence intervals, and model comparison.

The repository is based on the Jupyter Notebook and the accompanying six-page project report.

## Project objectives

- Explore the response variable and check the dataset for missing values, duplicate rows, and potential outliers.
- Fit three candidate models to the observed relationship between time \(t\) and response \(y(t)\).
- Estimate model parameters and residual variance.
- Construct approximate 95% confidence intervals using the Jacobian and Fisher information matrix.
- Assess residual behaviour using residual-versus-time plots, histograms, normal Q-Q plots, and formal normality tests.
- Compare the candidate models using residual sum of squares (RSS) and estimated error variance.

## Project files and input data

A suggested repository layout is:

```text
MTH686-Nonlinear-Regression/
├── MTH686_PROJECT.ipynb   # Main analysis and model-fitting notebook
├── Report.pdf             # Written project report
└── set-71.csv              # Input dataset (must be added separately)
```

**Important:** `set-71.csv` was referenced by the notebook but was not included with the two files supplied for this README. Add the original dataset to the repository or upload it to the notebook runtime before running the analysis.

### Dataset format

The notebook expects a CSV file with **two columns and no header row**:

| Column | Meaning |
|---|---|
| 1 | Time/input variable, `t` |
| 2 | Observed response, `y(t)` |

The supplied notebook output reports 60 observations, with `t` ranging from 1 to 60. Several cells read the file using `header=None` and assign the column names `t` and `y(t)`.

## Environment and dependencies

The notebook uses Python and the following libraries:

- NumPy
- pandas
- SciPy
- Matplotlib
- Jupyter Notebook or JupyterLab

Install the main dependencies with:

```bash
pip install numpy pandas scipy matplotlib jupyter
```

## Running the notebook

1. Clone or download this repository.
2. Place `set-71.csv` in the location expected by the notebook.
3. Install the dependencies listed above.
4. Open `MTH686_PROJECT.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
5. Run the notebook cells **from top to bottom**, because later cells reuse variables such as `t`, `y`, `t_data`, and `y_data` created earlier.

### Dataset path

The notebook was written with Google Colab paths such as:

```python
DATA_PATH = "/content/set-71.csv"
```

For a local run, change the dataset paths throughout the notebook to the actual location of the CSV. For example, when the CSV is in the same directory as the notebook:

```python
DATA_PATH = "./set-71.csv"
```

The path is repeated in multiple model-specific cells, so update each occurrence rather than changing only the first data-loading cell.

## Exploratory data analysis

The notebook performs the following initial checks and visualizations:

- Dataset dimensions, missing-value counts, duplicate-row count, and descriptive statistics.
- Scatter plot of `y` against `t`.
- Time-series plot of `y(t)` over `t`.
- Histogram of the response values.
- Boxplot of the response values.
- Outlier detection using the 1.5 × IQR rule.

### Reported EDA results

| Statistic | `t` | `y(t)` |
|---|---:|---:|
| Number of observations | 60 | 60 |
| Mean | 30.500000 | 1.364557 |
| Standard deviation | 17.464249 | 0.370148 |
| Minimum | 1.000000 | 0.915051 |
| Median | 30.500000 | 1.274801 |
| Maximum | 60.000000 | 3.034956 |

No missing values or duplicate rows were reported. The IQR procedure identified four high observations, corresponding to `t = 1, 2, 3, 4`. The notebook reports these points but does **not** remove them before fitting the models.

## Candidate regression models

### Model 1: Exponential Components Model

The fitted model is:

\[
 y(t) = \alpha_0 + \alpha_1 e^{\beta_1 t}
             + \alpha_2 e^{\beta_2 t} + \varepsilon(t),
 \qquad \varepsilon(t) \sim N(0,\sigma^2).
\]

This is a separable nonlinear regression model: for fixed \(\beta_1\) and \(\beta_2\), the model is linear in \(\alpha_0, \alpha_1, \alpha_2\).

**Implementation in the notebook**

1. The notebook uses visually selected starting values \((\beta_1,\beta_2)=(-0.5,-0.1)\). It records that an attempted Prony-based approach produced negative roots and was not used for the final fit.
2. For each trial pair of decay parameters, it constructs the design matrix with columns \(1\), \(e^{\beta_1t}\), and \(e^{\beta_2t}\).
3. It estimates the linear coefficients using the normal-equation expression \((X^T X)^{-1}X^T y\).
4. It minimizes the profiled RSS over the decay parameters with SciPy's `minimize` using the `L-BFGS-B` method and negative bounds for both decay rates.
5. It computes fitted values, residuals, an estimated residual variance, confidence intervals, and diagnostic plots.

The report describes this as separable nonlinear regression / variable projection. Note that the actual nonlinear optimizer in the corresponding notebook cell is **L-BFGS-B**, not a Gauss-Newton iteration.

**Reported estimates**

| Parameter | Estimate |
|---|---:|
| \(\alpha_0\) | 1.160876 |
| \(\alpha_1\) | 2.831736 |
| \(\alpha_2\) | 0.144015 |
| \(\beta_1\) | -0.415545 |
| \(\beta_2\) | -0.008583 |

- RSS: **2.070353**
- Estimated error variance, \(\hat\sigma^2\): **0.037643**

**Reported 95% confidence intervals**

| Parameter | 95% confidence interval |
|---|---|
| \(\alpha_0\) | [-6.821550, 9.143302] |
| \(\alpha_1\) | [2.019457, 3.644015] |
| \(\alpha_2\) | [-7.497974, 7.786004] |
| \(\beta_1\) | [-0.626823, -0.204267] |
| \(\beta_2\) | [-0.646254, 0.629148] |

### Model 2: Nonlinear Ratio Model

The model **implemented in the notebook** is:

\[
 y(t) = \frac{\alpha_0 + \alpha_1 t}{\beta_0 + \beta_1 t}
             + \varepsilon(t),
 \qquad \varepsilon(t) \sim N(0,\sigma^2).
\]

**Implementation in the notebook**

1. A two-dimensional grid search is performed over candidate values of \(\beta_0\) and \(\beta_1\). The configured ranges are 0.1 to 5.0 for \(\beta_0\), and -1.0 to 1.0 for \(\beta_1\), with 60 grid points in each range.
2. For each grid pair, the model is written as a linear least-squares problem in \(\alpha_0\) and \(\alpha_1\). The notebook uses `numpy.linalg.lstsq` to obtain those coefficients and selects the pair with the smallest RSS.
3. Starting from the best grid point, a Gauss-Newton procedure updates all four parameters.
4. A denominator safeguard rejects or reduces steps that would make \(\beta_0+\beta_1t\) too close to zero.
5. The notebook calculates the approximate covariance matrix, standard errors, 95% confidence intervals, and fitted/residual values. It uses a pseudoinverse when the Jacobian cross-product matrix is ill-conditioned.

**Reported estimates**

| Parameter | Estimate |
|---|---:|
| \(\alpha_0\) | 1.429014 |
| \(\alpha_1\) | 0.495448 |
| \(\beta_0\) | 0.182164 |
| \(\beta_1\) | 0.424622 |

- RSS: **2.3787376**
- Estimated error variance, \(\hat\sigma^2\): **0.0424775**
- Gauss-Newton iterations reported: **17**
- Condition number of \(J^T J\) reported in the confidence-interval calculation: approximately **3.38 × 10¹⁶**; a pseudoinverse is used.

**Reported 95% confidence intervals**

| Parameter | 95% confidence interval |
|---|---|
| \(\alpha_0\) | [1.279897, 1.578131] |
| \(\alpha_1\) | [0.189439, 0.801457] |
| \(\beta_0\) | [-0.052764, 0.417092] |
| \(\beta_1\) | [0.184325, 0.664919] |

### Model 3: Fourth-Degree Polynomial Regression Model

The fitted model is:

\[
 y(t) = \beta_0 + \beta_1t + \beta_2t^2
             + \beta_3t^3 + \beta_4t^4 + \varepsilon(t),
 \qquad \varepsilon(t) \sim N(0,\sigma^2).
\]

**Implementation in the notebook**

1. The design matrix contains the columns \(1,t,t^2,t^3,t^4\).
2. The five polynomial coefficients are estimated using ordinary least squares and the normal-equation expression \((X^T X)^{-1}X^T y\).
3. The notebook calculates fitted values, residuals, the Fisher information matrix, approximate covariance, standard errors, confidence intervals, and diagnostic plots.
4. The confidence-interval calculation checks the condition number of \(X^T X\) and uses a pseudoinverse because the matrix is ill-conditioned.

**Reported estimates**

| Parameter | Estimate |
|---|---:|
| \(\beta_0\) | 2.9286835 |
| \(\beta_1\) | -0.2800173 |
| \(\beta_2\) | 0.0154571 |
| \(\beta_3\) | -0.00033854 |
| \(\beta_4\) | 0.000002538545 |

- RSS: approximately **2.6322818**
- Estimated error variance, \(\hat\sigma^2\): **0.04785967**
- Condition number of \(X^T X\) reported in the confidence-interval calculation: approximately **6.42 × 10¹⁴**; a pseudoinverse is used.

**Reported 95% confidence intervals**

| Parameter | 95% confidence interval |
|---|---|
| \(\beta_0\) | [2.61485737, 3.24250959] |
| \(\beta_1\) | [-0.35004775, -0.20998687] |
| \(\beta_2\) | [0.01084275, 0.02007143] |
| \(\beta_3\) | [-0.00045175, -0.00022533] |
| \(\beta_4\) | [0.00000162, 0.00000346] |

## Model comparison

The report provides the following comparison:

| Model | RSS | Estimated \(\sigma^2\) |
|---|---:|---:|
| Model 1: Exponential components | 2.070353 | 0.037643 |
| Model 2: Nonlinear ratio | 2.3787376 | 0.0424775 |
| Model 3: Fourth-degree polynomial | 2.63228185 | 0.04785967 |

Using the **reported RSS and estimated error variance**, Model 1 has the smallest values among the three models. The report also contains a later sentence calling Model 3 the best based on \(R^2\), adjusted \(R^2\), and RMSE, but it does not provide those metric values. That statement is inconsistent with the RSS/variance comparison table, so the intended overall winner should be verified before making a definitive claim based on all criteria.

## Residual diagnostics

For each model, the notebook/report includes residual-versus-time plots, histograms, normal Q-Q plots, and observed-versus-fitted plots. The notebook additionally runs the D'Agostino-Pearson and Anderson-Darling normality tests.

### Shapiro-Wilk results

| Model | Test statistic | p-value | Interpretation at 5% |
|---|---:|---:|---|
| Model 1 | 0.973243 | 0.209693 | Fail to reject normality null hypothesis |
| Model 2 | 0.970864 | 0.160830 | Fail to reject normality null hypothesis |
| Model 3 | 0.987349 | 0.790216 | Fail to reject normality null hypothesis |

The reported Shapiro-Wilk p-values are all above 0.05. This means the tests do not provide significant evidence against residual normality at the 5% level; it does not prove that the errors are normally distributed.

The notebook's additional tests report:

- **Model 1:** D'Agostino-Pearson p-value = 0.370583; Anderson-Darling statistic = 0.501248. The Anderson-Darling statistic is below all listed critical values.
- **Model 2:** D'Agostino-Pearson p-value = 0.419111; Anderson-Darling statistic = 0.660270. It exceeds the 15% and 10% critical values, but not the 5% critical value (0.743), so it does not reject normality at the 5% level using the listed table.
- **Model 3:** D'Agostino-Pearson p-value = 0.992846; Anderson-Darling statistic = 0.225404. The Anderson-Darling statistic is below all listed critical values.

The report describes the residual plots as broadly centered around zero and the fitted curves as following the central trend of the observed data.

## Statistical estimation notes

For each model, the notebook estimates the residual variance using:

\[
\widehat{\sigma}^2 = \frac{RSS}{n-p},
\]

where \(n=60\) is the number of observations and \(p\) is the number of fitted parameters. The approximate covariance matrix is based on the Jacobian/design matrix and the inverse or pseudoinverse of its cross-product matrix. The reported 95% intervals use a Student-t critical value with \(n-p\) degrees of freedom.

## Implementation and reproducibility notes

The following details are worth checking when rerunning or extending the supplied notebook:

1. **Model 2 equation differs between the report and the code.** Page 3 of the report typesets the expression as \(\alpha_0 + \alpha_1 t/(\beta_0+\beta_1t)\). The notebook's `model_f` function and its Jacobian implement \((\alpha_0+\alpha_1t)/(\beta_0+\beta_1t)\). This README documents the model actually implemented in the code. Confirm the intended formula with the project specification.
2. **Model 1's \(\beta_2\) value is slightly inconsistent.** The optimizer output and report give approximately -0.008583, while the later confidence-interval cell hard-codes -0.008553. The reported interval is the interval printed by that inference cell.
3. **Model 3 data loading uses inconsistent CSV-header settings.** The main EDA and model-loading cells read the CSV with `header=None`, but the first OLS cell and one later inference cell call `pd.read_csv` without `header=None` and then look for columns named `t` and `y(t)`. The saved output shows a `KeyError: 't'` in those cells. The subsequent calculations can still print results because earlier notebook cells have already defined data variables in the interactive kernel. For a clean run, load the CSV consistently with `header=None` and assign the column names before accessing them.
4. **Normal-equation conditioning.** The polynomial model's \(X^T X\) and the ratio model's \(J^T J\) are reported as ill-conditioned. The notebook uses pseudoinverses for the corresponding uncertainty calculations, but Model 3's coefficient-estimation cell directly inverts \(X^T X\). For improved numerical stability, consider a QR/SVD-based least-squares solve (for example, `numpy.linalg.lstsq`) and, for polynomial fitting, scaling the time variable before forming high powers.
5. **Plot saving is not consistent.** Most EDA and diagnostic plots are displayed inline. Some filename variables and save-path messages are present, but not every corresponding plot is actually written to disk. In particular, the Model 2 plotting function calls `plt.show()` and prints a save message without calling `savefig` for the fit plot.
6. **Paths and notebook state.** The notebook uses absolute Google Colab paths and several later cells reuse variables set by earlier cells. Update all data paths and run the notebook from the first cell in order for reproducible execution.

## Conclusion

The project demonstrates three approaches to regression modelling on the same 60-point time-response dataset: a separable two-exponential model, a nonlinear ratio model, and a fourth-degree polynomial model. It compares parameter estimates and uncertainty intervals, examines residual behaviour, and reports RSS and residual-variance measures. Based strictly on the comparison table in the report, the exponential components model has the lowest RSS and estimated error variance. The report's separate claim favouring the polynomial model using other metrics should be reconciled with the underlying results before stating a final model selection.
