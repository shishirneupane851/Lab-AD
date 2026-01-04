import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("housing_price_dataset.csv")
df = df.dropna()

# Select only numeric features (excluding target)
X = df.select_dtypes(include=[np.number]).drop(columns=["Price"])

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Output
print("UNSUPERVISED LEARNING")
print(df.head())

print("\nCluster Distribution:")
print(df["Cluster"].value_counts())

# MATPLOTLIB VISUALIZATION

# Scatter plot using first two features
plt.figure()
plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=df["Cluster"]
)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-Means Clustering (Unsupervised Learning)")
plt.show()
