#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Analysis and Visualization Assignment

This script demonstrates loading, analyzing, and visualizing the Iris dataset.
It covers data exploration, basic analysis, and creating various types of visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Set the style for our plots
plt.style.use('seaborn-v0_8-whitegrid')

def load_data():
    """Load and prepare the Iris dataset."""
    try:
        # Load iris dataset from sklearn
        iris = load_iris()
        
        # Create a DataFrame
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        
        # Add the target column (species)
        df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
        
        print("Data loaded successfully!")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def explore_data(df):
    """Explore the dataset structure and content."""
    print("\n===== DATA EXPLORATION =====")
    
    # Display the first 5 rows
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    
    # Dataset information
    print("\nDataset info:")
    print(f"- Shape: {df.shape}")
    print(f"- Columns: {df.columns.tolist()}")
    
    # Check data types
    print("\nData types:")
    print(df.dtypes)
    
    # Check for missing values
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # If there were missing values, we would handle them here
    # Since Iris dataset is complete, we'll just display a message
    if df.isnull().sum().sum() == 0:
        print("No missing values found. The dataset is complete.")
    else:
        # Example of handling missing values
        print("Handling missing values...")
        df_clean = df.fillna(df.mean())
        print("Missing values filled with column means.")

def analyze_data(df):
    """Perform basic data analysis on the dataset."""
    print("\n===== BASIC DATA ANALYSIS =====")
    
    # Basic statistics for numerical columns
    print("\nBasic statistics:")
    print(df.describe())
    
    # Group by species and compute means
    print("\nMean measurements by species:")
    species_means = df.groupby('species').mean()
    print(species_means)
    
    # Find correlations between features
    print("\nCorrelation matrix:")
    correlation = df.iloc[:, :-1].corr()
    print(correlation)
    
    # Additional analysis: Min and max values for each species
    print("\nMinimum values by species:")
    print(df.groupby('species').min())
    
    print("\nMaximum values by species:")
    print(df.groupby('species').max())
    
    return species_means, correlation

def visualize_data(df, species_means, correlation):
    """Create various visualizations of the dataset."""
    print("\n===== DATA VISUALIZATION =====")
    print("Creating visualizations... (close each plot to see the next one)")
    
    # Set figure size for all plots
    plt.figure(figsize=(10, 6))
    
    # 1. Line chart: Average measurements by species
    plt.figure(figsize=(12, 6))
    species_means.T.plot(marker='o')
    plt.title('Average Measurements by Species')
    plt.xlabel('Measurements')
    plt.ylabel('Value (cm)')
    plt.grid(True)
    plt.legend(title='Species')
    plt.tight_layout()
    plt.savefig('line_chart_by_species.png')
    plt.show()
    
    # 2. Bar chart: Comparison of a feature across species
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='species', y='sepal length (cm)', data=df)
    plt.title('Average Sepal Length by Species')
    plt.xlabel('Species')
    plt.ylabel('Sepal Length (cm)')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('bar_chart_sepal_length.png')
    plt.show()
    
    # 3. Histogram: Distribution of petal length
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
    plt.savefig('histogram_petal_length.png')
    plt.show()
    
    # 4. Scatter plot: Relationship between sepal and petal length
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
    plt.savefig('scatter_plot_sepal_vs_petal.png')
    plt.show()
    
    # 5. Bonus: Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                linewidths=0.5, square=True)
    plt.title('Correlation Matrix of Iris Features')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.show()

def main():
    """Main function to execute the data analysis pipeline."""
    print("Starting Iris Dataset Analysis...")
    
    # Load the data
    df = load_data()
    if df is None:
        print("Exiting due to data loading error.")
        return
    
    # Explore the data
    explore_data(df)
    
    # Analyze the data
    species_means, correlation = analyze_data(df)
    
    # Visualize the data
    visualize_data(df, species_means, correlation)
    
    print("\nAnalysis complete! All visualizations have been created and saved.")

if __name__ == "__main__":
    main()
