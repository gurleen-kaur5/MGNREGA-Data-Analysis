import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#Load dataset

df = pd.read_csv("C:/Users/Gurleen Kaur/Downloads/distdata.csv")

print(df)

# Exploring dataset 1------
print("\nDataset Info:",df.info())
print("Illustration of dataset",df.describe())
print("\nFirst 5 rows:", df.head())
print("\nLast 5 rows:", df.tail())
print("\ndatatypes of Dataset:", df.dtypes)
print("\nDataset Shape:", df.shape)

#statistical analysis 1---
corr_matrix = df.corr(numeric_only=True)
print("Correlation matrix: ",corr_matrix)

# Check for missing values 2------
print("\nMissing Values:",df.isnull().sum())

# Drop columns with all null values (e.g., 'Remarks') 2------
df.dropna(axis=1, how='all', inplace=True)


# Fill missing values in numeric columns (if any) using mean           
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Final check 2-------
print("\nMissing Values After Cleaning:")
print(df.isnull().sum().sum())
print(df)


# Handle duplicate rows 3------
duplicates = df.duplicated().sum()
print(f"\nDuplicate Rows: {duplicates}")
df.drop_duplicates(inplace=True)  #removing duplicates

# Unique values in states and districts  3-----
print("\nUnique States:", df['state_name'].nunique())
print("Unique Districts:", df['district_name'].nunique())



#VISUALS



# 1️⃣ Histogram + KDE
cols = ['Average_Wage_rate_per_day_per_person', 'Average_days_of_employment_provided_per_Household', 'Total_Exp']
for col in cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col].dropna(), kde=True, bins=50)
    plt.title(f'Distribution of {col}')
    plt.show()
    
trend = df.groupby('fin_year')['Average_Wage_rate_per_day_per_person'].mean().reset_index()
plt.figure(figsize=(8, 4))
sns.lineplot(data=trend, x='fin_year', y='Average_Wage_rate_per_day_per_person', marker='o')
plt.title('Wage Trends Over Financial Years')
plt.xticks(rotation=45)
plt.show()

# Select relevant numerical columns including more dimensions
num_cols = [
    'Average_Wage_rate_per_day_per_person',
    'Wages',
    'Total_Exp',
    'Total_Individuals_Worked',
    'No_of_Households_Completed',
    'No_of_Households_Registered',
    'persondays_generated_total',
    'persondays_generated_women',
    'percent_of_Category_B_Works'
]

# Filter out columns that might not exist to avoid KeyErrors
num_cols = [col for col in num_cols if col in df.columns]

# Plot enhanced heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5, square=True)
plt.title('Enhanced Correlation Matrix of Economic and Employment Factors')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Aggregate key demographic data
grouped = df.groupby('state_name')[['SC_persondays', 'ST_persondays', 'Women_Persondays', 'Total_Individuals_Worked']].sum()
top_states = grouped.sort_values('Total_Individuals_Worked', ascending=False).head(10)

# Plot
top_states[['SC_persondays', 'ST_persondays', 'Women_Persondays']].plot(
    kind='bar', stacked=True, figsize=(12, 6), colormap='viridis'
)
plt.title("Top 10 States by Employment (SC, ST & Women Persondays)")
plt.xlabel("State")
plt.ylabel("Persondays")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# Set a clean style
sns.set(style="whitegrid")

# Select columns to visualize
cols_to_plot = ['Average_Wage_rate_per_day_per_person', 'Total_Individuals_Worked']

plt.figure(figsize=(12, 6))

for i, col in enumerate(cols_to_plot, 1):
    plt.subplot(1, 2, i)
    sns.boxplot(y=df[col], color='skyblue', width=0.4)
    plt.title(f'Box Plot of {col.replace("_", " ")}')
    plt.ylabel(col.replace("_", " "))
    plt.xlabel('')

plt.tight_layout()
plt.show()
 

df_filtered = df[
    df['Average_Wage_rate_per_day_per_person'].notnull() &
    df['Total_Individuals_Worked'].notnull() &
    df['state_name'].notnull()
]

# Optional: Filter to remove extremely high wages (to reduce skew)
df_filtered = df_filtered[df_filtered['Average_Wage_rate_per_day_per_person'] < 5000]

# Step 2: Plot
plt.figure(figsize=(12, 6))
sns.scatterplot(
    data=df_filtered,
    x='Average_Wage_rate_per_day_per_person',
    y='Total_Individuals_Worked',
    hue='state_name',
    palette='husl',
    alpha=0.7,
    edgecolor='black'
)

plt.title('Wage vs Employment by District')
plt.xlabel('Average Wage per Day (INR)')
plt.ylabel('Total Individuals Worked')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='State')
plt.tight_layout()
plt.show()

gender = df.groupby('state_name')[['Women_Persondays', 'Total_Individuals_Worked']].sum().reset_index()
gender['Men'] = gender['Total_Individuals_Worked'] - gender['Women_Persondays']
gender_top = gender.nlargest(10, 'Total_Individuals_Worked')

plt.figure(figsize=(10, 5))
plt.bar(gender_top['state_name'], gender_top['Women_Persondays'], label='Women')
plt.bar(gender_top['state_name'], gender_top['Men'], bottom=gender_top['Women_Persondays'], label='Men')
plt.title('Gender Distribution of Employment')
plt.xticks(rotation=45)
plt.legend()
plt.show()    


#Linear Regression



from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# ------------------------- Define Features & Target -------------------------
# We choose 'Total_Exp' and 'Approved_Labour_Budget' as features to predict 'Total_Individuals_Worked'.
features = ['Total_Exp', 'Approved_Labour_Budget']
X = df[features]
y = df['Total_Individuals_Worked']

# ------------------------- Train-Test Split -------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ------------------------- Model Training -------------------------
lm = LinearRegression()
lm.fit(X_train, y_train)

# ------------------------- Model Prediction -------------------------
y_pred = lm.predict(X_test)

# ------------------------- Evaluation Metrics -------------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Linear Regression Model Evaluation:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.2f}\n")

print("Model Coefficients:")
for feature, coef in zip(features, lm.coef_):
    print(f"  {feature}: {coef:.4f}")
print(f"Intercept: {lm.intercept_:.4f}\n")

# ------------------------- Cross-Validation Testing -------------------------
# Use 5-fold cross-validation to test the accuracy of the model
cv_scores = cross_val_score(lm, X, y, cv=5, scoring='r2')
print("5-Fold Cross-Validation R2 Scores:", cv_scores)
print("Mean CV R2 Score:", cv_scores.mean())

# ------------------------- Visualization: Actual vs. Predicted -------------------------
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.6, color='steelblue', label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal Fit')
plt.xlabel('Actual Total Individuals Worked')
plt.ylabel('Predicted Total Individuals Worked')
plt.title('Linear Regression: Actual vs. Predicted')
plt.legend()
plt.show()




























             