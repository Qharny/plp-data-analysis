# Data Analysis and Visualization Assignment

## Overview
This Jupyter notebook demonstrates the process of loading, exploring, analyzing, and visualizing the Iris dataset. The Iris dataset is a classic dataset in machine learning and statistics that contains measurements of different iris flowers.

## Task 1: Load and Explore the Dataset

Let's start by importing the necessary libraries and loading the data:

```python
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Display the first few rows
print("First 5 rows of the dataset:")
df.head()
```

### Dataset Structure

Let's examine the dataset structure:

```python
# Check the shape of the dataset
print(f"Dataset dimensions: {df.shape}")

# Check column data types
print("\nData types:")
print(df.dtypes)

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())
```

The Iris dataset is clean and doesn't contain any missing values. If it did, we would handle them like this:

```python
# Example of handling missing values (not needed for Iris dataset)
# df_clean = df.fillna(df.mean())  # Fill numerical missing values with column means
```

## Task 2: Basic Data Analysis

Let's compute some basic statistics for our dataset:

```python
# Generate descriptive statistics
print("Basic statistics:")
df.describe()
```

Now let's group the data by species and calculate means for each group:

```python
# Group by species and compute means
species_means = df.groupby('species').mean()
print("Mean measurements by species:")
species_means
```

Let's also look at correlations between features:

```python
# Calculate correlation matrix
correlation = df.iloc[:, :-1].corr()
print("Correlation matrix:")
correlation
```

### Key Findings:
- The dataset contains 150 samples with 4 numeric features and 1 categorical feature (species).
- There are 50 samples for each of the three species: setosa, versicolor, and virginica.
- Petal length and petal width have the strongest correlation (0.96).
- Setosa species has the shortest petals but larger sepals compared to its overall size.
- Virginica species generally has the largest measurements across all features.

## Task 3: Data Visualization

Let's create several visualizations to better understand the data:

### 1. Line Chart: Average Measurements by Species

```python
# Line chart of average measurements by species
plt.figure(figsize=(12, 6))
species_means.T.plot(marker='o')
plt.title('Average Measurements by Species')
plt.xlabel('Measurements')
plt.ylabel('Value (cm)')
plt.grid(True)
plt.legend(title='Species')
plt.tight_layout()
plt.show()
```

### 2. Bar Chart: Comparison of Sepal Length Across Species

```python
# Bar chart of sepal length by species
plt.figure(figsize=(10, 6))
sns.barplot(x='species', y='sepal length (cm)', data=df)
plt.title('Average Sepal Length by Species')
plt.xlabel('Species')
plt.ylabel('Sepal Length (cm)')
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()
```

### 3. Histogram: Distribution of Petal Length

```python
# Histogram of petal length distribution by species
plt.figure(figsize=(10, 6))
for species in df['species'].unique():
    subset = df[df['species'] == species]
    plt.hist(subset['petal length (cm)'], alpha=0.5, bins=10, label=species)

plt.title('Distribution of Petal Lengths')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Frequency')
plt.legend(title='Species')
plt.grid(True)
plt.tight_layout()
plt.show()
```

### 4. Scatter Plot: Relationship Between Sepal and Petal Length

```python
# Scatter plot of sepal length vs petal length
plt.figure(figsize=(10, 6))
for species in df['species'].unique():
    subset = df[df['species'] == species]
    plt.scatter(subset['sepal length (cm)'], subset['petal length (cm)'], 
                label=species, alpha=0.7, s=70)

plt.title('Sepal Length vs. Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.grid(True)
plt.tight_layout()
plt.show()
```

### 5. Bonus: Correlation Heatmap

```python
# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
            linewidths=0.5, square=True)
plt.title('Correlation Matrix of Iris Features')
plt.tight_layout()
plt.show()
```

## Conclusion

Through our analysis of the Iris dataset, we found several interesting patterns:

1. **Clear Species Differentiation**: The three iris species show distinct measurement patterns, particularly in petal dimensions.

2. **Feature Correlations**: Petal length and petal width are highly correlated, suggesting they may provide similar information for species classification.

3. **Iris Setosa Distinction**: The setosa species is clearly separated from the other two species, particularly in petal measurements.

4. **Distribution Patterns**: The histograms show that the distributions of measurements differ across species, with some showing more variance than others.

5. **Size Relationships**: Flowers with longer sepals tend to have longer petals, showing a general size relationship across measurements.

These visualizations help us understand the underlying structure of the data and could inform further machine learning applications like classification models.
