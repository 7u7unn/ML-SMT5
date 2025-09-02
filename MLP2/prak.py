# %%
import pandas as pd
import requests
import matplotlib.pyplot as plt

# %%
"""
### Load Dataset
"""

# %%
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
df = pd.read_csv(url ,                delim_whitespace=True,  # Use whitespace as delimiter
                 header=None,            # No header in the file
                 na_values='?')          # Replace '?' with NaN)
df.describe()

# %%
attribute = {
    'mpg': 'continuous',
    'cylinders': 'multi-valued discrete',
    'displacement': 'continuous',
    'horsepower': 'continuous',
    'weight': 'continuous',
    'acceleration': 'continuous',
    'model year': 'multi-valued discrete',
    'origin': 'multi-valued discrete',
    'car name': 'string (unique for each instance)'
}

df.columns = list(attribute.keys())

# %%
df.info()

# %%
"""
### Describing : Mean, Max, Min, Modus
"""

# %%
df.describe()

# %%
for col in df.columns:
    print(f"\nModus {col}: ")
    print(df[col].mode())
    print("-----------")

# %%
"""
### Missing Value Diagnose
"""

# %%
df.head()

# %%
for i in df.columns:
    print(f"\n{i}:")
    print(f"sum null = {df[i].isna().sum()}")

# %%
"""
### Diagnosa Outlier
"""

# %%
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

plt.figure(figsize=(15, 8))

n_cols = 3
n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

for idx, col in enumerate(numeric_cols, 1):
    plt.subplot(n_rows, n_cols, idx)
    df.boxplot(column=col)
    plt.title(f'Boxplot of {col}')
    plt.xticks([1], [col], rotation=45)

plt.tight_layout()  
plt.show()

# %%
def analyze_boxplot(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    outliers = df[(df[column] < Q1 - 1.5*IQR) | (df[column] > Q3 + 1.5*IQR)][column]
    print(f"{column} outliers: {len(outliers)}")

# Analyze each numerical column
for col in numeric_cols:
    analyze_boxplot(df, col)

# %%
"""
### Penanganan Missing Value
"""

# %%
df['horsepower'].fillna(df.horsepower.mode()[0], inplace=True)
df['horsepower']
df.describe()


# %%
df.info()