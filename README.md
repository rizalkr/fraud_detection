# From Transaction Data to Business Decisions
## Building a Fraud Detection Decision Engine

This project demonstrates how transaction-level fraud data can be transformed into practical operational decisions rather than stopping at binary fraud predictions.

Using the **IEEE-CIS Fraud Detection** dataset, the project covers the complete analytical workflow from business understanding and exploratory analysis to feature engineering, model development, model comparison, risk-based decision routing, and financial exposure analysis.

The central idea is:

> **Fraud detection is fundamentally a risk management problem, not only a classification problem.**

The machine learning model produces a transaction-level fraud risk score, while the Decision Engine translates that score into operational actions based on the trade-off between:

- fraud detection,
- customer friction,
- and manual review workload.

---

## Project Status

**Completed**

- [x] Business Understanding
- [x] Exploratory Data Analysis
- [x] Feature Engineering
- [x] Fraud Modeling
- [x] Model Comparison
- [x] Decision Engine
- [x] Financial Exposure Analysis
- [x] Final Report

---

## Project Resources

| Resource | Description | Link |
|---|---|---|
| Original Dataset | IEEE-CIS Fraud Detection source dataset | [View on Kaggle](https://www.kaggle.com/datasets/lnasiri007/ieeecis-fraud-detection) |
| Optimized Dataset | Preprocessed and memory-optimized modeling dataset (`data_optimized.parquet`) | [View on Kaggle](https://www.kaggle.com/datasets/rizalkurnia198/ieee-cis-fraud-detection-optimized) |
| Models & Project Artifacts | Trained models, preprocessing objects, probability outputs, predictions, and threshold-analysis artifacts | [View on Kaggle](https://www.kaggle.com/datasets/rizalkurnia198/fraud-detection-decision-engine-model-artifacts) |
| Final Report | Consolidated project findings, Decision Engine results, financial exposure analysis, limitations, and recommendations | [View Final Report](notebooks/06_Final_Report.ipynb) |

Large datasets and machine learning artifacts are hosted separately on Kaggle to keep this repository lightweight.

### Available Model Artifacts

The Kaggle artifact package contains the saved outputs generated throughout the modeling workflow, including:

- Logistic Regression model and preprocessing artifacts
- Random Forest model and probability outputs
- XGBoost model, predictions, and probability outputs
- LightGBM model and probability outputs
- Ordinal Encoder
- Feature-column definitions
- StandardScaler artifacts
- Threshold-analysis results

**Random Forest is the final model selected for the Decision Engine.**

The remaining artifacts are retained to support reproducibility and comparison across the complete modeling workflow.

---

## Project Workflow

```text
Business Understanding
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Modeling
        ↓
Model Comparison
        ↓
Decision Engine
        ↓
Financial Exposure Analysis
        ↓
Final Report
```

---

## Business Problem

Online payment fraud creates more than direct financial loss.

An ineffective fraud detection system may also create:

- unnecessary customer friction,
- excessive manual review workload,
- investigation costs,
- customer trust issues,
- and reputational risk.

This creates an important trade-off.

A fraud system should detect as much fraudulent activity as possible, but maximizing fraud detection without considering false positives can produce an unsustainable operational burden.

The project therefore focuses on the following business question:

> **How can transaction-level fraud risk be identified and translated into practical operational actions while balancing fraud detection, customer friction, and manual review workload?**

The intended outcome is not merely a fraud classifier, but a **decision-support framework**.

---

## Dataset Overview

The analysis uses the IEEE-CIS Fraud Detection dataset.

| Metric | Value |
|---|---:|
| Total Transactions | 590,540 |
| Legitimate Transactions | 569,877 |
| Fraud Transactions | 20,663 |
| Fraud Rate | 3.50% |

Fraud represents only a small proportion of the transaction population.

Because of this class imbalance, overall accuracy is not treated as the primary model-selection metric.

The project places greater emphasis on:

- **Recall** — how much observed fraud is identified
- **Precision** — how concentrated fraud is among flagged transactions
- **F1-Score** — technical balance between Precision and Recall
- **Average Precision (PR-AUC)** — ranking performance under class imbalance
- **ROC-AUC** — supporting measure of overall discrimination

---

## Key Exploratory Findings

Exploratory analysis showed that fraud is not distributed uniformly across transactions.

### Product Risk

`ProductCD` emerged as an important contextual variable.

- Product W represents the majority of transaction volume and has a relatively low fraud rate of approximately **2.04%**.
- Product C represents a much smaller transaction population but records a substantially higher fraud rate of approximately **11.69%**.

This reveals two different dimensions of fraud risk:

- **fraud concentration**
- **business exposure from transaction volume**

A segment with the highest fraud rate is not necessarily the segment creating the largest total exposure.

### Card Characteristics

Credit-card transactions show a higher observed fraud rate than debit-card transactions.

- Credit fraud rate: approximately **6.68%**
- Debit fraud rate: approximately **2.43%**

Within Product C, the difference becomes stronger:

- Product C credit: approximately **16.92%**
- Product C debit: approximately **8.16%**

Several individual card-related segments also show elevated fraud concentration, including:

- `card3 = 185`
- `card5 = 137`

Meanwhile, high-volume segments such as `card5 = 226` remain important from an exposure perspective.

### Device and Identity Information

Where device information is available:

- Mobile fraud rate: approximately **10.17%**
- Desktop fraud rate: approximately **6.52%**

Identity-related information provides complementary behavioral signals, but the analysis does not support using device attributes as standalone fraud rules.

### Missingness as Information

A substantial portion of identity information is missing.

The analysis suggests that missingness may encode differences in transaction or identity-collection behavior rather than representing only a data-quality problem.

This finding influenced the missing-value strategy used during feature engineering.

### Temporal Behavior

Fraud rates change across the observation period and are not directly explained by transaction volume alone.

However, the project does not claim a causal explanation for these temporal changes.

---

## Feature Engineering & Data Preparation

After removing `TransactionID`, the modeling dataset contained:

- **401 numerical features**
- **32 categorical features**
- **1 target variable**

The final predictor matrix contains **433 features**.

### Missing Value Strategy

Missing values were handled according to feature type:

| Feature Group | Treatment |
|---|---|
| Transaction numerical features | Median imputation |
| Transaction categorical features | `"Missing"` category |
| Identity numerical features | Sentinel value `-999` |
| Identity categorical features | `"Missing"` category |
| Extremely sparse features | Evaluated individually |

After preprocessing, no missing values remained in the modeling dataset.

### Memory Optimization

The initial dataset required approximately:

**2,060.63 MB**

After datatype optimization:

**922.55 MB**

This represents approximately:

**55.23% memory reduction**

The optimized dataset was saved as:

```text
data_optimized.parquet
```

---

## Modeling Approach

A stratified 80/20 train-test split was used.

| Dataset | Shape |
|---|---:|
| Training Set | 472,432 × 433 |
| Test Set | 118,108 × 433 |

Configuration:

```text
test_size = 0.20
random_state = 42
stratify = isFraud
```

### Categorical Encoding

The 32 categorical features were transformed using `OrdinalEncoder`.

Unknown categories are represented using:

```text
-1
```

This allows previously unseen categorical values to be handled during inference.

### Scaling

`StandardScaler` was applied only to the numerical features used by Logistic Regression.

Tree-based models were trained using encoded but unscaled features.

---

## Candidate Models

Four standalone models were developed:

1. Logistic Regression
2. Random Forest
3. XGBoost
4. LightGBM

Additional soft-voting ensemble combinations were evaluated during model comparison:

- Random Forest + XGBoost
- Random Forest + XGBoost + LightGBM

---

## Model Comparison

Each candidate was evaluated across the same threshold range to make the comparison more meaningful than relying only on the default threshold of `0.50`.

### Best Technical Operating Points

| Model | Best Threshold | Precision | Recall | F1-Score | ROC-AUC | Average Precision |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.85 | 47.27% | 39.34% | 42.94% | 0.8595 | 0.4178 |
| Random Forest | 0.20 | 75.46% | 64.87% | 69.76% | 0.9388 | 0.7365 |
| XGBoost | 0.25 | 77.60% | 56.59% | 65.45% | 0.9333 | 0.6909 |
| LightGBM | 0.85 | 71.12% | 54.34% | 61.61% | 0.9354 | 0.6468 |
| RF + XGBoost | 0.20 | 76.00% | 64.29% | 69.66% | 0.9460 | 0.7391 |
| RF + XGBoost + LightGBM | 0.40 | 73.69% | 62.74% | 67.77% | 0.9442 | 0.7158 |

### Why Random Forest?

Random Forest was selected as the final risk-scoring model.

At its technical operating point:

- Precision: **75.46%**
- Recall: **64.87%**
- F1-Score: **69.76%**
- ROC-AUC: **0.9388**
- Average Precision: **0.7365**

The RF + XGBoost ensemble achieved slightly higher ROC-AUC and Average Precision, but the improvement was marginal.

Random Forest was preferred because it provided:

- slightly stronger fraud Recall,
- competitive Precision,
- the strongest F1-Score among the evaluated candidates,
- strong ranking performance,
- and lower implementation complexity.

### Operational Comparison

| Model | Threshold | True Positives | False Negatives | False Positives | Flagged Transactions |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.20 | 2,703 | 1,430 | 915 | 3,618 |
| RF + XGBoost + LightGBM | 0.40 | 2,593 | 1,540 | 926 | 3,519 |

Random Forest identified:

- **110 additional observed fraud cases**
- with **11 fewer false positives**

than the three-model ensemble at the evaluated operating points.

---

## Final Decision Engine

The selected Random Forest model produces a transaction-level fraud risk score.

Instead of converting this score into a single binary prediction, the Decision Engine uses three operational risk bands.

| Fraud Risk Score | Action |
|---|---|
| `< 0.20` | **Approve** |
| `0.20 ≤ score < 0.50` | **Additional Authentication** |
| `≥ 0.50` | **Manual Review** |

The thresholds represent **business routing boundaries**, not merely model classification thresholds.

### Routing Results

| Action | Transactions | Fraud Cases | Transaction Share | Fraud Rate | Fraud Coverage |
|---|---:|---:|---:|---:|---:|
| Approve | 114,490 | 1,430 | 96.94% | 1.25% | 34.60% |
| Additional Authentication | 1,832 | 1,017 | 1.55% | 55.51% | 24.61% |
| Manual Review | 1,786 | 1,686 | 1.51% | 94.40% | 40.79% |

The two intervention bands together:

- affect approximately **3.06% of transactions**
- contain approximately **65.40% of observed fraud cases**

Meanwhile:

**96.94% of transactions remain in the low-friction Approve path.**

### Manual Review Concentration

Manual Review is limited to:

**1.51% of transactions**

while the observed fraud rate in this band reaches:

**94.40%**

This allows expensive analyst capacity to be concentrated on transactions with the strongest model risk signals.

The feasibility of this workload would still need to be validated against actual production transaction volume and analyst capacity.

---

## Financial Exposure Analysis

The original business objective included understanding financial impact.

However, the dataset does not contain:

- realized fraud losses,
- chargeback outcomes,
- fraud recovery,
- authentication costs,
- manual review costs,
- or post-intervention outcomes.

Therefore, this project does **not** estimate expected financial loss or financial savings.

Instead, `TransactionAmt` is used to analyze the transaction value associated with observed fraud cases.

### Fraud Amount Coverage

| Action | Fraud Case Coverage | Fraud Amount Coverage |
|---|---:|---:|
| Approve | 34.60% | 42.43% |
| Additional Authentication | 24.61% | 27.60% |
| Manual Review | 40.79% | 29.97% |

An important result emerges:

> **Fraud likelihood and financial severity are not the same risk dimension.**

Although the Approve band contains **34.60% of observed fraud cases**, those cases account for approximately **42.43% of observed fraud transaction amount**.

### Fraud Transaction Amount Profile

| Action | Average Fraud Amount | Median Fraud Amount | P90 Fraud Amount |
|---|---:|---:|---:|
| Approve | 189.65 | 100.00 | 420.25 |
| Additional Authentication | 173.42 | 77.00 | 445.00 |
| Manual Review | 113.62 | 56.49 | 280.00 |

Fraud transactions remaining in the Approve band have a higher average and median transaction value than those routed to Manual Review.

The current Decision Engine should therefore be interpreted as a:

> **fraud-risk prioritization framework**

rather than a:

> **financial-loss optimization system**

A future version could incorporate transaction value as a secondary decision dimension alongside fraud probability.

---

## Key Business Insights

### 1. Fraud Risk Is Concentrated

Fraud is highly concentrated within particular transaction contexts rather than being distributed uniformly.

### 2. High Fraud Rate and High Exposure Are Different

A smaller segment may have a high fraud rate, while a large-volume segment can still create substantial financial exposure despite a lower fraud rate.

### 3. Context Matters More Than Isolated Rules

Product, card, device, behavioral, and missingness signals become more informative when considered together.

This supports multivariate machine learning over isolated hard-coded fraud rules.

### 4. Intervention Can Be Concentrated

Only approximately **3.06% of transactions** receive additional intervention while those bands contain approximately **65.40% of observed fraud cases**.

### 5. Manual Review Can Be Focused

Only **1.51% of transactions** enter Manual Review, with an observed fraud rate of **94.40%**.

### 6. Residual Risk Is Intentional

The system does not attempt to maximize Recall at any operational cost.

Increasing intervention further would also increase:

- customer friction,
- legitimate transaction challenges,
- and operational workload.

### 7. Fraud Probability Is Not Financial Severity

The financial exposure analysis shows that the highest fraud probability does not necessarily correspond to the highest transaction value.

---

## Business Recommendations

### Use Risk-Based Routing

Retain the three-level Decision Engine:

```text
Low Risk
→ Approve

Medium Risk
→ Additional Authentication

High Risk
→ Manual Review
```

### Reserve Manual Review for High-Risk Transactions

Manual Review should remain concentrated on the highest-risk transactions rather than being applied broadly.

### Use Additional Authentication as a Lower-Cost Intervention

Medium-risk transactions should receive a lower-cost verification step before requiring analyst investigation.

The effectiveness of any authentication method would need to be validated using real intervention outcomes.

### Monitor Important Fraud Segments

Segments such as:

- Product C
- Product C credit
- `card3 = 185`
- `card5 = 137`
- high-volume Product W
- `card5 = 226`

can be useful for:

- fraud monitoring,
- reporting,
- targeted investigation,
- and model diagnostics.

They should not automatically be treated as hard-block rules without separate validation.

### Explore Amount-Aware Routing

Future versions of the Decision Engine should test whether selected high-value transactions deserve additional verification even when their fraud score remains below the current intervention threshold.

No amount threshold is recommended by the current project because it has not yet been validated.

---

## Important Limitations

The current results should be interpreted within several methodological limitations.

### Random Train-Test Split

The project uses a stratified random split rather than chronological out-of-time validation.

Fraud behavior can change over time, so production performance may differ.

### Threshold Selection

Thresholds were explored using held-out test labels.

This is appropriate for diagnostic portfolio analysis but may produce optimistic threshold-performance estimates.

A stronger design would use separate:

- training,
- validation,
- and untouched test sets.

### No Intervention Outcomes

The dataset does not show whether Additional Authentication or Manual Review would actually prevent fraud.

Fraud cases routed to intervention bands therefore represent:

**fraud coverage**

not:

**fraud prevented**

### No Real Financial Loss Data

`TransactionAmt` is not equivalent to realized fraud loss.

No claims are made about expected financial savings.

### Probability Calibration Was Not Tested

Random Forest `predict_proba` is treated as a model risk score.

A score of `0.70` should not automatically be interpreted as an exactly 70% real-world fraud probability.

### Operational Capacity Is Unknown

The project does not contain actual:

- analyst staffing,
- transaction throughput,
- review time,
- or service-level requirements.

### Dataset Interpretability

Many IEEE-CIS variables are anonymized or obfuscated.

Observed relationships should therefore not automatically be interpreted as causal.

### Drift Was Not Evaluated

Model drift and fraud concept drift were not tested in the current project.

---

## Future Improvements

Future development could include:

1. **Out-of-Time Validation**  
   Train on earlier transactions and evaluate on later periods.

2. **Independent Threshold Validation**  
   Separate threshold-selection data from the final test set.

3. **Probability Calibration**  
   Evaluate methods such as Platt scaling or isotonic regression.

4. **Amount-Aware Decision Routing**  
   Combine fraud likelihood with financial exposure.

5. **Business Cost Modeling**  
   Incorporate chargeback losses, authentication costs, manual review costs, and false-positive costs.

6. **Intervention Effectiveness Measurement**  
   Track authentication outcomes and analyst decisions.

7. **Fraud and Model Drift Monitoring**  
   Monitor score distributions, fraud rates, Precision, Recall, and action-band performance over time.

8. **Feature Ablation and Interaction Analysis**  
   Measure the incremental contribution of temporal, identity, card, product, and missingness feature groups.

9. **Operational Capacity Validation**  
   Translate routing percentages into real queue volumes and analyst workloads.

---

## Project Notebooks

| Notebook | Description |
|---|---|
| `01_Business_Understanding.ipynb` | Business problem, objectives, and success criteria |
| `02_Exploratory_Data_Analysis.ipynb` | Fraud behavior and transaction-pattern analysis |
| `03_Feature_Engineering.ipynb` | Missing-value treatment, feature preparation, and memory optimization |
| `04_Modeling.ipynb` | Logistic Regression, Random Forest, XGBoost, and LightGBM development |
| `04.5_Model_Comparison.ipynb` | Standalone model, ensemble, and threshold comparison |
| `05_Decision_Engine.ipynb` | Risk-band design and operational routing |
| `06_Final_Report.ipynb` | Consolidated findings, financial exposure analysis, recommendations, limitations, and conclusion |

---

## Technology Stack

The project primarily uses:

```text
Python
Pandas
NumPy
Scikit-learn
Random Forest
XGBoost
LightGBM
Matplotlib
Jupyter Notebook
Parquet
Joblib
```

---

## Reproducibility Notes

The optimized dataset and saved model artifacts are hosted separately on Kaggle because of their size.

The modeling workflow uses:

```text
test_size = 0.20
random_state = 42
stratify = isFraud
```

The saved probability artifacts correspond to the held-out test split used during model comparison.

Some Logistic Regression artifacts with similar purposes were retained from different stages of the modeling workflow rather than removed without verification.

---

## Final Takeaway

The main lesson from this project is that building a fraud classifier is only one part of the problem.

A machine learning model can estimate transaction risk, but the business still needs to answer:

> **What should be done with that risk score?**

This project connects:

**transaction data → fraud analysis → machine learning → risk scoring → operational decisions**

to demonstrate how fraud prediction can be transformed into a practical decision-support framework.

The final system does not attempt to eliminate all fraud at any cost.

Instead, it aims to:

- detect meaningful fraud risk,
- preserve a low-friction experience for most transactions,
- concentrate expensive intervention on the highest-risk cases,
- and provide a framework that can be extended toward financial-loss-aware decision making.
