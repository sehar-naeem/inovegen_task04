# 🏥 Inovegen Internship — Task 4: End-to-End Medical Insurance Cost Prediction

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-darkblue.svg)](https://pandas.pydata.org/)
[![Inovegen](https://img.shields.io/badge/Inovegen-AI%2FML%20Final%20Assessment-purple.svg)](https://inovegen.com)

---

## 📌 Executive Summary
Healthcare expenditure forecasting is a foundational problem in actuarial science and predictive analytics. Accurately predicting patient medical charges allows healthcare providers and insurers to price risk equitably, maintain financial solvency, and identify high-risk populations for preventative health interventions.

This project represents the **Final Capstone Assessment (Task 4)** for the **Inovegen AI/ML Internship**. It develops a complete, production-grade Supervised Regression pipeline that predicts individual medical charges (`charges` in USD) from patient demographic and biometric risk factors. 

By discovering clinical non-linear interactions during Exploratory Data Analysis (EDA) and engineering a custom **Obese-Smoker Interaction Feature**, the primary linear regression model achieved an exceptional **$R^2$ score of 90.56%** on unseen holdout test data.

---

## 🗂️ Dataset Source & Architecture
- **Dataset:** US Medical Insurance Cost Dataset (Brett Lantz / Machine Learning with R / Kaggle)
- **Observations:** `1,338` patient records across `7` attributes (1 duplicate removed $\rightarrow$ `1,337` clean records)
- **Target Variable:** `charges` (Continuous individual medical costs billed by health insurance in USD)
- **Features:** `age`, `sex`, `bmi`, `children`, `smoker`, `region`
- **Data Source Reference:** [Machine Learning with R Repository](https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv)

---

## 🧹 Data Cleaning & Preprocessing Pipeline
1. **Audit & Duplicate Remediation:** Screened for missing values (0 nulls found). Detected and eliminated 1 exact duplicate record (Row 581, duplicate of Row 195) to preserve sample independence.
2. **Clinical Feature Engineering (Requirement 3):**
   - Discovered during EDA that the combination of Obesity ($\text{BMI} \ge 30$) and Smoking triggers an exponential, non-linear cost multiplier.
   - Formulated `is_obese = (bmi >= 30).astype(int)`
   - Formulated `obese_smoker = is_obese * smoker` (Captures the extreme risk subgroup).
   - Formulated `age_squared = age ** 2` (Captures compounding late-career healthcare costs).
3. **Categorical Encoding:** One-Hot Encoded `region`, `sex`, and binary mapped `smoker` using `pd.get_dummies(..., drop_first=True)` to prevent multicollinearity, resulting in 11 total features.
4. **Train-Test Partitioning:** Split into an 80% Training Set (`1,069` patients) and 20% Holdout Test Set (`268` patients) with `random_state=42`.
5. **Feature Standardization:** Scaled continuous numerical attributes (`age`, `bmi`, `children`, `age_squared`) using `StandardScaler` fitted strictly on `X_train` to prevent data leakage.

---

## 📈 Exploratory Data Analysis (Key Visual Insights)

1. **Target Distribution & The Smoker Multiplier:**
   - Medical charges are strongly right-skewed with a median of **$9,382** and a maximum of **$63,770**.
   - Smokers incur a median cost of **$34,400** vs **$7,300** for non-smokers—a **4.7× cost multiplier**.
2. **Age Progression & Parallel Cost Strata:**
   - Scatter plots reveal three distinct upward-sloping cost strata.
   - Baseline medical charges rise predictably by approximately **~$250 to ~$280 per year of age**.
3. **The Clinical Obesity Cliff ($\text{BMI} \ge 30$):**
   - For non-smokers, high BMI yields only a modest cost increase.
   - For smokers, crossing the clinical threshold of $\text{BMI} \ge 30$ causes medical charges to explode from ~$15,000 to **over $40,000–$60,000+**!

---

## 🤖 Model Benchmark & Comparative Results (Requirements 4 & 5)

Three distinct regression architectures were evaluated on the 20% unseen holdout test set (`268` patients):

| Model Architecture | Train $R^2$ | Test $R^2$ | Test MAE ($) | Test RMSE ($) |
| :--- | :---: | :---: | :---: | :---: |
| **Ridge Regression (Linear + Interactions)** 🏆 | **0.8522** | **0.9056** | **$2,398.04** | **$4,164.60** |
| **Gradient Boosting Regressor** | 0.8915 | 0.9036 | $2,447.86 | $4,209.84 |
| **Random Forest Regressor** | 0.8706 | 0.9023 | $2,396.99 | $4,237.92 |

---

## 🏆 Evidence-Based Model Selection (Requirement 7)
**Ridge Regression (with engineered interaction terms)** is selected as the Champion Model:
1. **Highest Predictive Accuracy:** Achieved the highest $R^2$ score (**0.9056 / 90.56%**) and lowest Root Mean Squared Error (**$4,164.60**) on unseen holdout patients.
2. **Robust Generalization:** Closely matched Train $R^2$ (0.8522) and Test $R^2$ (0.9056), demonstrating zero overfitting.
3. **Actuarial Interpretability:** In regulated healthcare insurance, linear coefficients provide auditable dollar weights:
   - **Obese Smoker Interaction Penalty:** **+$19,298.10**
   - **Base Smoker Surcharge:** **+$13,271.62**
   - **Total Combined Smoker + Obesity Impact:** **+$32,569.72**

---

## 📉 Residual & Error Diagnostics (Requirement 6)
- **Actual vs Predicted Fit:** Test predictions cluster tightly along the 45-degree identity reference line ($y = x$).
- **Residual Distribution:** Prediction errors form a near-normal bell curve centered at **$0.00** (Mean residual error = -$12.38).
- **Residual Variance:** 75% of all holdout predictions fall within **$2,600** of actual billed charges. The small number of higher-residual outliers correspond to real-world unobserved medical emergencies (e.g. trauma, acute surgeries).

---

## 💡 6 Key Clinical & Business Findings
1. **The Compounding Smoker Penalty:** Smoking is the primary driver of medical bills, adding a baseline +$13,271 surcharge.
2. **The Obesity Interaction Multiplier:** Obesity ($\text{BMI} \ge 30$) combined with smoking creates an additional +$19,298 surcharge, bringing total risk surcharge past **+$32,569**.
3. **Age Progression:** Healthcare costs rise steadily by ~$250–$280 per year of age.
4. **Demographic Invariance:** Gender and geographic region account for less than 1–2% of cost variance.
5. **Feature Engineering Impact:** Adding the clinical interaction term boosted linear model explanatory power from ~75% to **over 90.5%**.
6. **Emergency Noise:** Outliers in the residuals reflect natural real-world unobserved medical trauma that cannot be modeled from demographic data alone.

---

## ⚠️ Limitations & Realistic Future Improvements (Requirement 8)
### Limitations:
- Lacks diagnostic history (e.g., ICD-10 codes, chronic illness history, prescription drug records).
- Lacks hospital and provider network pricing context (in-network vs. out-of-network rates).
- 1 duplicate row cleaned during preprocessing.

### Realistic Improvements:
- **Generalized Linear Models (GLM):** Train Tweedie or Gamma regression to naturally model the skewed, non-negative distribution of medical expenses.
- **Wearable Biometric Ingestion:** Incorporate real-time wearable activity data (resting heart rate, steps) for dynamic wellness pricing.
- **Preventative Health Subsidies:** Subsidize smoking cessation and weight management; preventing a smoker from crossing $\text{BMI} \ge 30$ saves up to **$19,000 annually per patient**.

---

## 🚀 How to Run the Notebook
1. Open Google Colab or Jupyter Notebook:
   ```bash
   jupyter notebook Inovegen_Task4_Medical_Insurance_Cost_Prediction.ipynb
