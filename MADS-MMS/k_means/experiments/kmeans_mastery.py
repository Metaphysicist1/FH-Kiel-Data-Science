"""
K-means Clustering Mastery Tutorial
==================================
This script provides a comprehensive guide to understanding and implementing K-means clustering
with multiple real-world scenarios and examples.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import silhouette_score
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

def plot_clusters(X, y_pred, centers=None, title="Clustering Results"):
    """Utility function to plot clustering results"""
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.6)
    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, linewidth=3, label='Centroids')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    if centers is not None:
        plt.legend()
    plt.show()

# Example 1: Basic K-means with Well-Separated Clusters
print("\n=== Example 1: Basic K-means with Well-Separated Clusters ===")
# Generate synthetic data
X_basic, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

# Fit K-means
kmeans_basic = KMeans(n_clusters=4, random_state=42)
y_pred_basic = kmeans_basic.fit_predict(X_basic)

# Plot results
plot_clusters(X_basic, y_pred_basic, kmeans_basic.cluster_centers_, 
             "Basic K-means with Well-Separated Clusters")

# Example 2: Finding Optimal K using Elbow Method
print("\n=== Example 2: Finding Optimal K using Elbow Method ===")
distortions = []
silhouette_scores = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_basic)
    distortions.append(kmeans.inertia_)
    if k > 1:  # Silhouette score requires at least 2 clusters
        silhouette_scores.append(silhouette_score(X_basic, kmeans.labels_))

# Plot Elbow Method
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(K_range, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('Elbow Method')

# Plot Silhouette Scores
plt.subplot(1, 2, 2)
plt.plot(list(K_range)[1:], silhouette_scores, 'rx-')
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis')
plt.tight_layout()
plt.show()

# Example 3: K-means with Different Cluster Shapes
print("\n=== Example 3: K-means with Different Cluster Shapes ===")

# Generate different shaped clusters
X_moons, _ = make_moons(n_samples=200, noise=0.05, random_state=42)
X_circles, _ = make_circles(n_samples=200, noise=0.05, random_state=42)

# Fit K-means on moons
kmeans_moons = KMeans(n_clusters=2, random_state=42)
y_pred_moons = kmeans_moons.fit_predict(X_moons)

# Fit K-means on circles
kmeans_circles = KMeans(n_clusters=2, random_state=42)
y_pred_circles = kmeans_circles.fit_predict(X_circles)

# Plot results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plot_clusters(X_moons, y_pred_moons, kmeans_moons.cluster_centers_, "K-means on Moon-shaped Clusters")
plt.subplot(1, 2, 2)
plot_clusters(X_circles, y_pred_circles, kmeans_circles.cluster_centers_, "K-means on Circular Clusters")
plt.tight_layout()
plt.show()

# Example 4: K-means with Feature Scaling
print("\n=== Example 4: K-means with Feature Scaling ===")

# Generate data with different scales
X_unscaled = np.random.randn(300, 2)
X_unscaled[:, 0] = X_unscaled[:, 0] * 10  # Make first feature have larger scale

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_unscaled)

# Fit K-means on both scaled and unscaled data
kmeans_unscaled = KMeans(n_clusters=3, random_state=42)
kmeans_scaled = KMeans(n_clusters=3, random_state=42)

y_pred_unscaled = kmeans_unscaled.fit_predict(X_unscaled)
y_pred_scaled = kmeans_scaled.fit_predict(X_scaled)

# Plot results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plot_clusters(X_unscaled, y_pred_unscaled, kmeans_unscaled.cluster_centers_, 
             "K-means without Scaling")
plt.subplot(1, 2, 2)
plot_clusters(X_scaled, y_pred_scaled, kmeans_scaled.cluster_centers_, 
             "K-means with Scaling")
plt.tight_layout()
plt.show()

# Example 5: Real-world Scenario - Customer Segmentation
print("\n=== Example 5: Real-world Scenario - Customer Segmentation ===")

# Generate synthetic customer data
n_customers = 500
np.random.seed(42)

# Generate customer features
age = np.random.normal(40, 15, n_customers)
income = np.random.normal(60000, 20000, n_customers)
spending_score = np.random.normal(50, 25, n_customers)

# Create customer dataset
customer_data = np.column_stack([age, income, spending_score])
customer_data_scaled = StandardScaler().fit_transform(customer_data)

# Fit K-means
kmeans_customers = KMeans(n_clusters=4, random_state=42)
customer_segments = kmeans_customers.fit_predict(customer_data_scaled)

# Create visualization
fig = plt.figure(figsize=(15, 5))

# Plot Age vs Income
plt.subplot(1, 3, 1)
plt.scatter(age, income, c=customer_segments, cmap='viridis')
plt.xlabel('Age')
plt.ylabel('Income')
plt.title('Age vs Income Segments')

# Plot Age vs Spending Score
plt.subplot(1, 3, 2)
plt.scatter(age, spending_score, c=customer_segments, cmap='viridis')
plt.xlabel('Age')
plt.ylabel('Spending Score')
plt.title('Age vs Spending Score Segments')

# Plot Income vs Spending Score
plt.subplot(1, 3, 3)
plt.scatter(income, spending_score, c=customer_segments, cmap='viridis')
plt.xlabel('Income')
plt.ylabel('Spending Score')
plt.title('Income vs Spending Score Segments')

plt.tight_layout()
plt.show()

# Print segment analysis
print("\nCustomer Segment Analysis:")
segments_df = pd.DataFrame({
    'Age': age,
    'Income': income,
    'Spending_Score': spending_score,
    'Segment': customer_segments
})

print("\nSegment Statistics:")
print(segments_df.groupby('Segment').mean().round(2))

# Example 6: Interactive Function for Experimentation
def experiment_kmeans(n_samples=300, n_clusters=3, cluster_std=0.60, random_state=42):
    """
    Interactive function to experiment with different K-means parameters
    """
    # Generate data
    X, _ = make_blobs(n_samples=n_samples, centers=n_clusters, 
                      cluster_std=cluster_std, random_state=random_state)
    
    # Fit K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    y_pred = kmeans.fit_predict(X)
    
    # Plot results
    plot_clusters(X, y_pred, kmeans.cluster_centers_,
                 f"K-means with {n_clusters} clusters\nSamples: {n_samples}, Std: {cluster_std}")
    
    return kmeans.inertia_

print("\n=== Example 6: Interactive Experimentation ===")
print("Try calling experiment_kmeans() with different parameters!")
print("Example usage:")
print("experiment_kmeans(n_samples=500, n_clusters=5, cluster_std=0.8)")

# Run an example
experiment_kmeans() 