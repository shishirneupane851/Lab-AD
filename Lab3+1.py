import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("housing_price_dataset.csv")
df = df.dropna()

# Numeric features (no target)
X = df.select_dtypes(include=["number"]).drop(columns=["Price"])

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

print("\nUNSUPERVISED LEARNING")
print(df["Cluster"].value_counts())

# Graph: Clusters 
plt.figure()
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=df["Cluster"])
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Unsupervised Learning: K-Means Clustering")
plt.show()
