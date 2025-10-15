# =====================================================================
# Project 4: Customer Segmentation using Clustering (K-Means + PCA)
# =====================================================================

# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

import pickle
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# Step 2: Load Dataset
# =====================================================================

# You can replace this with your own dataset file (Mall_Customers.csv)
# Example dataset can be downloaded from Kaggle: "Mall Customer Segmentation Data"
try:
    df = pd.read_csv("Mall_Customers.csv")
except FileNotFoundError:
    # Create synthetic dataset if not found
    np.random.seed(42)
    df = pd.DataFrame({
        'CustomerID': range(1, 301),
        'Gender': np.random.choice(['Male', 'Female'], 300),
        'Age': np.random.randint(18, 70, 300),
        'Annual Income (k$)': np.random.randint(15, 150, 300),
        'Spending Score (1-100)': np.random.randint(1, 100, 300)
    })

print("? Dataset Loaded Successfully")
print(df.head())

# =====================================================================
# Step 3: Data Exploration
# =====================================================================
print("\n--- Dataset Info ---")
print(df.info())
print("\n--- Missing Values ---")
print(df.isnull().sum())

# Quick visualizations
sns.pairplot(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
plt.show()

# =====================================================================
# Step 4: Data Cleaning
# =====================================================================
df.dropna(inplace=True)  # remove missing rows if any
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
if 'CustomerID' in df.columns:
    df.drop('CustomerID', axis=1, inplace=True)

print("\n? Data Cleaning Completed")
print(df.head())

# =====================================================================
# Step 5: Feature Selection and Scaling
# =====================================================================
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print("\n? Feature Scaling Done")
print(X_scaled.head())

# =====================================================================
# Step 6: Determine Optimal Number of Clusters
# =====================================================================
inertia = []
silhouette = []

for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    silhouette.append(silhouette_score(X_scaled, kmeans.labels_))

plt.figure(figsize=(6,4))
plt.plot(range(2,10), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show()

plt.figure(figsize=(6,4))
plt.plot(range(2,10), silhouette, marker='o', color='green')
plt.title('Silhouette Score')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Score')
plt.show()

# =====================================================================
# Step 7: Apply K-Means Clustering (Assume k=5)
# =====================================================================
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print("\n? Clustering Completed")
print(df.head())

# =====================================================================
# Step 8: Visualize Clusters
# =====================================================================
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)',
                hue='Cluster', palette='tab10', s=100)
plt.title('Customer Segments Based on Income and Spending')
plt.show()

# =====================================================================
# Step 9: PCA for 2D Visualization
# =====================================================================
pca = PCA(n_components=2)
pca_data = pca.fit_transform(X_scaled)
df['PCA1'] = pca_data[:, 0]
df['PCA2'] = pca_data[:, 1]

plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='tab10', s=100)
plt.title('Customer Segmentation (PCA Visualization)')
plt.show()

# =====================================================================
# Step 10: Cluster Analysis
# =====================================================================
cluster_summary = df.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()
print("\n--- Cluster Summary ---")
print(cluster_summary)

sns.countplot(x='Cluster', data=df, palette='tab10')
plt.title('Number of Customers in Each Cluster')
plt.show()

# Step 12: Save Final Dataset
# =====================================================================
df.to_csv('Customer_Segmentation_Result.csv', index=False)
print("\n? Final Dataset Saved as 'Customer_Segmentation_Result.csv'")