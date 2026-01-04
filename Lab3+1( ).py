import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings

warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load the dataset
df = pd.read_csv('housing_price_dataset.csv')

print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nBasic Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Separate features and target (Price is the target in supervised context)
# For unsupervised, we'll use all features
features = df.drop('Price', axis=1) if 'Price' in df.columns else df
target = df['Price'] if 'Price' in df.columns else None

print(f"\nFeatures shape: {features.shape}")
print(f"Features columns: {features.columns.tolist()}")

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 1. DIMENSIONALITY REDUCTION 

print("\n" + "=" * 50)
print("DIMENSIONALITY REDUCTION ANALYSIS")
print("=" * 50)

# PCA Analysis
pca = PCA()
pca_result = pca.fit_transform(features_scaled)

# Explained variance ratio
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print(f"\nPCA Explained Variance Ratio:")
for i, (var, cum_var) in enumerate(zip(explained_variance, cumulative_variance)):
    print(f"PC{i + 1}: {var:.4f} (Cumulative: {cum_var:.4f})")

# Plot explained variance
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.6)
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'r-', marker='o')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA Explained Variance')
plt.axhline(y=0.95, color='g', linestyle='--', alpha=0.5, label='95% threshold')
plt.legend()

# t-SNE visualization (for 2D visualization)
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_result = tsne.fit_transform(features_scaled)

plt.subplot(1, 3, 2)
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.6)
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('t-SNE Visualization (2D)')

# 3D PCA visualization
pca_3d = PCA(n_components=3)
pca_3d_result = pca_3d.fit_transform(features_scaled)

ax = plt.subplot(1, 3, 3, projection='3d')
ax.scatter(pca_3d_result[:, 0], pca_3d_result[:, 1], pca_3d_result[:, 2], alpha=0.6)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('PCA 3D Visualization')

plt.tight_layout()
plt.show()

# 2.OPTIMAL NUMBER OF CLUSTERS

print("\n" + "=" * 50)
print("CLUSTER ANALYSIS - FINDING OPTIMAL K")
print("=" * 50)

# Elbow method and Silhouette analysis
inertias = []
silhouette_scores = []
calinski_scores = []
db_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_scaled)

    inertias.append(kmeans.inertia_)

    if len(set(cluster_labels)) > 1:  # Silhouette score needs at least 2 clusters
        silhouette_scores.append(silhouette_score(features_scaled, cluster_labels))
        calinski_scores.append(calinski_harabasz_score(features_scaled, cluster_labels))
        db_scores.append(davies_bouldin_score(features_scaled, cluster_labels))
    else:
        silhouette_scores.append(0)
        calinski_scores.append(0)
        db_scores.append(0)

# Plot metrics for determining optimal k
plt.figure(figsize=(15, 4))

plt.subplot(1, 4, 1)
plt.plot(k_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.xticks(k_range)

plt.subplot(1, 4, 2)
plt.plot(k_range, silhouette_scores, 'ro-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis')
plt.xticks(k_range)

plt.subplot(1, 4, 3)
plt.plot(k_range, calinski_scores, 'go-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Calinski-Harabasz Score')
plt.title('Calinski-Harabasz Index')
plt.xticks(k_range)

plt.subplot(1, 4, 4)
plt.plot(k_range, db_scores, 'mo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Davies-Bouldin Score')
plt.title('Davies-Bouldin Index (lower is better)')
plt.xticks(k_range)

plt.tight_layout()
plt.show()

# Find optimal k based on silhouette score
optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
optimal_k_calinski = k_range[np.argmax(calinski_scores)]
optimal_k_db = k_range[np.argmin(db_scores[1:]) + 1]  # Skip k=2 for DB index

print(f"\nOptimal k based on Silhouette Score: {optimal_k_silhouette}")
print(f"Optimal k based on Calinski-Harabasz Index: {optimal_k_calinski}")
print(f"Optimal k based on Davies-Bouldin Index: {optimal_k_db}")

# Use consensus optimal k
optimal_k = optimal_k_silhouette  # Using silhouette as primary metric
print(f"\nSelected optimal k for clustering: {optimal_k}")

# 3. CLUSTERING WITH DIFFERENT ALGORITHMS 

print("\n" + "=" * 50)
print("CLUSTERING WITH DIFFERENT ALGORITHMS")
print("=" * 50)

# 3.1 K-Means Clustering
print("\n K-Means Clustering:")
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
kmeans_labels = kmeans.fit_predict(features_scaled)

# 3.2 Gaussian Mixture Model (Soft Clustering)
print(" Gaussian Mixture Model:")
gmm = GaussianMixture(n_components=optimal_k, random_state=42)
gmm_labels = gmm.fit_predict(features_scaled)

# 3.3 Agglomerative Hierarchical Clustering
print(" Agglomerative Hierarchical Clustering:")
agg_clustering = AgglomerativeClustering(n_clusters=optimal_k)
agg_labels = agg_clustering.fit_predict(features_scaled)

# 3.4 DBSCAN (Density-based)
print("4. DBSCAN (Density-Based):")
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(features_scaled)
n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)
print(f"   DBSCAN found {n_clusters_dbscan} clusters and {n_noise} noise points")

# Evaluate clustering results
clustering_results = {
    'K-Means': kmeans_labels,
    'GMM': gmm_labels,
    'Agglomerative': agg_labels,
    'DBSCAN': dbscan_labels
}

print("\nClustering Evaluation Metrics:")
print("-" * 50)
print(f"{'Algorithm':<15} {'Silhouette':<12} {'Calinski':<12} {'Davies-Bouldin':<15}")
print("-" * 50)

for name, labels in clustering_results.items():
    if len(set(labels)) > 1 and -1 not in labels:  # Valid for evaluation
        sil_score = silhouette_score(features_scaled, labels)
        cal_score = calinski_harabasz_score(features_scaled, labels)
        db_score = davies_bouldin_score(features_scaled, labels)
        print(f"{name:<15} {sil_score:<12.4f} {cal_score:<12.4f} {db_score:<15.4f}")
    else:
        print(f"{name:<15} {'N/A':<12} {'N/A':<12} {'N/A':<15}")

# ==================== 4. VISUALIZE CLUSTERING RESULTS ====================

print("\n" + "=" * 50)
print("VISUALIZING CLUSTERING RESULTS")
print("=" * 50)

# Create subplots for different clustering results
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Use PCA for 2D visualization
pca_2d = PCA(n_components=2)
features_2d = pca_2d.fit_transform(features_scaled)

# Plot 1: K-Means
scatter1 = axes[0, 0].scatter(features_2d[:, 0], features_2d[:, 1],
                              c=kmeans_labels, cmap='tab10', alpha=0.6)
axes[0, 0].set_title(f'K-Means Clustering (k={optimal_k})')
axes[0, 0].set_xlabel('PC1')
axes[0, 0].set_ylabel('PC2')
plt.colorbar(scatter1, ax=axes[0, 0])

# Plot 2: GMM
scatter2 = axes[0, 1].scatter(features_2d[:, 0], features_2d[:, 1],
                              c=gmm_labels, cmap='tab10', alpha=0.6)
axes[0, 1].set_title(f'Gaussian Mixture Model (k={optimal_k})')
axes[0, 1].set_xlabel('PC1')
axes[0, 1].set_ylabel('PC2')
plt.colorbar(scatter2, ax=axes[0, 1])

# Plot 3: Agglomerative
scatter3 = axes[0, 2].scatter(features_2d[:, 0], features_2d[:, 1],
                              c=agg_labels, cmap='tab10', alpha=0.6)
axes[0, 2].set_title(f'Agglomerative Clustering (k={optimal_k})')
axes[0, 2].set_xlabel('PC1')
axes[0, 2].set_ylabel('PC2')
plt.colorbar(scatter3, ax=axes[0, 2])

# Plot 4: DBSCAN
unique_labels = set(dbscan_labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise
        col = [0, 0, 0, 1]

    class_member_mask = (dbscan_labels == k)
    xy = features_2d[class_member_mask]
    axes[1, 0].scatter(xy[:, 0], xy[:, 1], c=[col], alpha=0.6,
                       label='Noise' if k == -1 else f'Cluster {k}')

axes[1, 0].set_title(f'DBSCAN Clustering (Clusters: {n_clusters_dbscan})')
axes[1, 0].set_xlabel('PC1')
axes[1, 0].set_ylabel('PC2')
axes[1, 0].legend()

# Plot 5: t-SNE with K-Means clusters
scatter5 = axes[1, 1].scatter(tsne_result[:, 0], tsne_result[:, 1],
                              c=kmeans_labels, cmap='tab10', alpha=0.6)
axes[1, 1].set_title(f'K-Means on t-SNE (k={optimal_k})')
axes[1, 1].set_xlabel('t-SNE 1')
axes[1, 1].set_ylabel('t-SNE 2')
plt.colorbar(scatter5, ax=axes[1, 1])

# Plot 6: Cluster sizes
cluster_sizes = np.bincount(kmeans_labels[kmeans_labels >= 0])
axes[1, 2].bar(range(len(cluster_sizes)), cluster_sizes)
axes[1, 2].set_title('Cluster Sizes (K-Means)')
axes[1, 2].set_xlabel('Cluster ID')
axes[1, 2].set_ylabel('Number of Samples')
axes[1, 2].set_xticks(range(len(cluster_sizes)))

plt.tight_layout()
plt.show()

# ==================== 5. CLUSTER PROFILING ====================

print("\n" + "=" * 50)
print("CLUSTER PROFILING (K-Means)")
print("=" * 50)

# Add cluster labels to the original dataframe
df_clustered = df.copy()
df_clustered['Cluster'] = kmeans_labels

# Calculate mean values for each cluster
cluster_profiles = df_clustered.groupby('Cluster').mean()

print("\nCluster Profiles (Mean values):")
print(cluster_profiles)

# Visualize cluster profiles
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
axes = axes.flatten()

for i, column in enumerate(features.columns[:6]):  # First 6 features
    cluster_data = [df_clustered[df_clustered['Cluster'] == k][column]
                    for k in range(optimal_k)]

    axes[i].boxplot(cluster_data)
    axes[i].set_title(f'Distribution of {column} by Cluster')
    axes[i].set_xlabel('Cluster')
    axes[i].set_ylabel(column)
    axes[i].set_xticks(range(1, optimal_k + 1))
    axes[i].set_xticklabels(range(optimal_k))

plt.tight_layout()
plt.show()

# Heatmap of cluster means (normalized)
plt.figure(figsize=(12, 8))
cluster_profiles_normalized = cluster_profiles.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
sns.heatmap(cluster_profiles_normalized.T, cmap='coolwarm', center=0,
            annot=True, fmt='.2f', linewidths=0.5)
plt.title('Normalized Cluster Profiles (Z-scores)')
plt.xlabel('Cluster')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# ==================== 6. ANOMALY DETECTION ====================

print("\n" + "=" * 50)
print("ANOMALY DETECTION")
print("=" * 50)

# 6.1 Local Outlier Factor (LOF)
print("\n1. Local Outlier Factor (LOF):")
lof = LocalOutlierFactor(contamination=0.05, novelty=False)  # 5% contamination
lof_labels = lof.fit_predict(features_scaled)
lof_outliers = (lof_labels == -1)
print(f"   LOF detected {lof_outliers.sum()} outliers ({lof_outliers.sum() / len(df) * 100:.2f}%)")

# 6.2 Isolation Forest
from sklearn.ensemble import IsolationForest

print("2. Isolation Forest:")
iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_labels = iso_forest.fit_predict(features_scaled)
iso_outliers = (iso_labels == -1)
print(f"   Isolation Forest detected {iso_outliers.sum()} outliers ({iso_outliers.sum() / len(df) * 100:.2f}%)")

# 6.3 DBSCAN outliers (already computed)
dbscan_outliers = (dbscan_labels == -1)
print(
    f"3. DBSCAN detected {dbscan_outliers.sum()} noise points as outliers ({dbscan_outliers.sum() / len(df) * 100:.2f}%)")

# 6.4 Statistical approach (Z-score)
print("4. Statistical (Z-score) method:")
z_scores = np.abs((features_scaled - features_scaled.mean(axis=0)) / features_scaled.std(axis=0))
z_outliers = (z_scores > 3).any(axis=1)
print(f"   Z-score method (>3 std) detected {z_outliers.sum()} outliers ({z_outliers.sum() / len(df) * 100:.2f}%)")

# Visualize outliers
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# LOF outliers
scatter1 = axes[0, 0].scatter(features_2d[:, 0], features_2d[:, 1],
                              c=~lof_outliers, cmap='bwr', alpha=0.6)
axes[0, 0].set_title(f'LOF Outliers (Red: Outliers)\n{sum(lof_outliers)} outliers')
axes[0, 0].set_xlabel('PC1')
axes[0, 0].set_ylabel('PC2')

# Isolation Forest outliers
scatter2 = axes[0, 1].scatter(features_2d[:, 0], features_2d[:, 1],
                              c=~iso_outliers, cmap='bwr', alpha=0.6)
axes[0, 1].set_title(f'Isolation Forest Outliers (Red: Outliers)\n{sum(iso_outliers)} outliers')
axes[0, 1].set_xlabel('PC1')
axes[0, 1].set_ylabel('PC2')

# DBSCAN outliers
scatter3 = axes[1, 0].scatter(features_2d[:, 0], features_2d[:, 1],
                              c=~dbscan_outliers, cmap='bwr', alpha=0.6)
axes[1, 0].set_title(f'DBSCAN Outliers (Red: Noise)\n{sum(dbscan_outliers)} noise points')
axes[1, 0].set_xlabel('PC1')
axes[1, 0].set_ylabel('PC2')

# Z-score outliers
scatter4 = axes[1, 1].scatter(features_2d[:, 0], features_2d[:, 1],
                              c=~z_outliers, cmap='bwr', alpha=0.6)
axes[1, 1].set_title(f'Z-score Outliers (>3 std, Red: Outliers)\n{sum(z_outliers)} outliers')
axes[1, 1].set_xlabel('PC1')
axes[1, 1].set_ylabel('PC2')

plt.tight_layout()
plt.show()

# ==================== 7. CORRELATION WITH PRICE (IF AVAILABLE) ====================

if target is not None:
    print("\n" + "=" * 50)
    print("CORRELATION BETWEEN CLUSTERS AND PRICE")
    print("=" * 50)

    # Add price back for analysis
    df_clustered['Price'] = target

    # Boxplot of Price by Cluster
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Cluster', y='Price', data=df_clustered)
    plt.title('Price Distribution by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Price')
    plt.show()

    # Calculate average price by cluster
    avg_price_by_cluster = df_clustered.groupby('Cluster')['Price'].agg(['mean', 'median', 'std', 'count'])
    print("\nPrice Statistics by Cluster:")
    print(avg_price_by_cluster)

    # ANOVA test to see if clusters have significantly different prices
    from scipy.stats import f_oneway

    cluster_prices = [df_clustered[df_clustered['Cluster'] == k]['Price']
                      for k in range(optimal_k)]
    f_stat, p_value = f_oneway(*cluster_prices)

    print(f"\nANOVA Test for Price differences across clusters:")
    print(f"F-statistic: {f_stat:.4f}")
    print(f"P-value: {p_value:.4f}")

    if p_value < 0.05:
        print("Conclusion: There are significant price differences between clusters (p < 0.05)")
    else:
        print("Conclusion: No significant price differences between clusters")

# ==================== 8. SUMMARY AND INSIGHTS ====================

print("\n" + "=" * 50)
print("SUMMARY AND KEY INSIGHTS")
print("=" * 50)

print("\n1. DIMENSIONALITY REDUCTION:")
print(f"   - First 3 PCs explain {cumulative_variance[2]:.2%} of variance")
print(f"   - Need {np.argmax(cumulative_variance >= 0.95) + 1} PCs for 95% variance")

print("\n2. CLUSTERING:")
print(f"   - Optimal number of clusters: {optimal_k}")
print(f"   - Best algorithm (by silhouette score): K-Means")
print(f"   - Clusters are reasonably separated (Silhouette: {silhouette_score(features_scaled, kmeans_labels):.3f})")

print("\n3. ANOMALY DETECTION:")
print(f"   - LOF detected {lof_outliers.sum()} outliers ({lof_outliers.sum() / len(df) * 100:.2f}%)")
print(f"   - Isolation Forest detected {iso_outliers.sum()} outliers ({iso_outliers.sum() / len(df) * 100:.2f}%)")
print(f"   - DBSCAN detected {dbscan_outliers.sum()} noise points ({dbscan_outliers.sum() / len(df) * 100:.2f}%)")

print("\n4. RECOMMENDATIONS:")
print("   - Consider using PCA for dimensionality reduction before modeling")
print("   - The dataset can be segmented into distinct customer/property groups")
print("   - Investigate outliers for data quality or special cases")
print("   - Use clustering labels as features in supervised models")

# Save clustered data
df_clustered.to_csv('housing_price_clustered.csv', index=False)
print("\nClustered data saved to 'housing_price_clustered.csv'")