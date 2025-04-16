# 📍 Mapping Rural Realities: A Visual and Statistical Exploration of MGNREGA Employment and Economic Patterns

This project presents an in-depth exploratory and predictive analysis of the **Mahatma Gandhi National Rural Employment Guarantee Act (MGNREGA)** dataset. It leverages Python-based data science tools to uncover trends in rural employment, budget allocation, demographic participation, and expenditure efficiency.

---

## 📌 Project Objectives

- Analyze wage distribution and employment trends across Indian districts and financial years.
- Explore demographic inclusion in employment (SC, ST, Women).
- Identify outliers, trends, and disparities through visual analysis.
- Evaluate budget utilization and job completion performance.
- Build a **Linear Regression model** to predict employment outcomes based on financial inputs.

---

## 📂 Dataset

- **Source:** [data.gov.in](https://data.gov.in)
- **Size:** ~200,000 rows
- **Format:** CSV
- **Key Columns:**
  - `Total_Exp`, `Approved_Labour_Budget`, `Wages`
  - `Average_Wage_rate_per_day_per_person`
  - `Total_Individuals_Worked`, `Total_Households_Worked`
  - `SC/ST/Women_Persondays`, `Number_of_Completed_Works`
  - `percentage_payments_gererated_within_15_days`, and more

---

## 🛠 Tools & Libraries Used

- Python (Pandas, NumPy)
- Seaborn & Matplotlib (Data Visualization)
- Scikit-learn (Linear Regression, Cross-Validation)
- Statsmodels (VIF and Diagnostics)

---

## 📈 Analysis Highlights

### 🔍 Visual Insights

- **Histograms + KDE Plots:**  
  Revealed right-skewed distributions for wages and expenditures.
  
- **Line Plot:**  
  Identified a wage spike in 2021–22 with policy-linked fluctuations.

- **Correlation Heatmap:**  
  Showed strong positive correlation between `Wages`, `Total_Exp`, and `Total_Individuals_Worked`.

- **Stacked Bar Charts:**  
  Illustrated top 10 states for SC/ST/Women employment contributions.

- **Box Plots:**  
  Confirmed outliers in wage rates and employment numbers.

- **Scatter Plot:**  
  Suggested an inverse relation—lower average wages often link with more individuals employed.

- **Gender Distribution Bars:**  
  Compared inferred men vs. women persondays in top employment states.

---

## 📉 Regression Model

- **Model Type:** Linear Regression  
- **Target Variable:** `Total_Individuals_Worked`  
- **Predictors:** `Total_Exp`, `Approved_Labour_Budget`, `Exp_Budget_Interaction`
- **Performance:**
  - **R² Score (Test Set):** 0.83
  - **Mean CV R² Score (5-fold):** 0.77
  - **MSE:** ~1.47 Billion

---
