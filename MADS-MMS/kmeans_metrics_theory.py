"""
Understanding K-means Clustering Metrics: Theory and Visualization
==============================================================
This script provides in-depth explanation of clustering evaluation metrics
with theoretical background and visual interpretations.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from scipy.spatial.distance import cdist

# Set style and random seed
plt.style.use('seaborn')
np.random.seed(42)

def create_example_clusters(n_samples=300, noise=0.6):
    """Create example datasets for demonstration"""
    # Well-separated clusters
    X1, y1 = make_blobs(n_samples=n_samples, centers=3, cluster_std=noise, random_state=42)
    
    # Overlapping clusters
    X2, y2 = make_blobs(n_samples=n_samples, centers=3, cluster_std=noise*2, random_state=42)
    
    # Non-globular clusters
    X3, y3 = make_moons(n_samples=n_samples, noise=noise*0.5, random_state=42)
    
    return (X1, y1), (X2, y2), (X3, y3)

def plot_inertia_explanation():
    """Visualize what inertia means"""
    X, y = make_blobs(n_samples=100, centers=3, cluster_std=0.7, random_state=42)
    kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Points and centroids
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                marker='x', s=200, linewidths=3, color='r', label='Centroids')
    plt.title('Clusters and Centroids')
    plt.legend()
    
    # Plot 2: Distance visualization
    plt.subplot(1, 2, 2)
    for idx in range(len(X)):
        center = kmeans.cluster_centers_[kmeans.labels_[idx]]
        plt.plot([X[idx, 0], center[0]], [X[idx, 1], center[1]], 
                'gray', alpha=0.1)
    
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                marker='x', s=200, linewidths=3, color='r', label='Centroids')
    plt.title('Inertia: Sum of Distances to Centroids')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_silhouette_analysis():
    """Visualize silhouette analysis"""
    X, y = make_blobs(n_samples=150, centers=3, cluster_std=0.5, random_state=42)
    kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
    
    # Calculate silhouette scores
    silhouette_vals = silhouette_samples(X, kmeans.labels_)
    
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Clusters
    plt.subplot(1, 3, 1)
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
    plt.title('Clustered Data')
    
    # Plot 2: Silhouette plot
    plt.subplot(1, 3, 2)
    y_lower, y_upper = 0, 0
    
    for i in range(3):
        cluster_silhouette_vals = silhouette_vals[kmeans.labels_ == i]
        cluster_silhouette_vals.sort()
        y_upper += len(cluster_silhouette_vals)
        
        plt.fill_betweenx(np.arange(y_lower, y_upper),
                         0, cluster_silhouette_vals,
                         alpha=0.7)
        y_lower += len(cluster_silhouette_vals)
        
    plt.axvline(x=silhouette_score(X, kmeans.labels_), color="red", linestyle="--")
    plt.title('Silhouette Plot')
    plt.xlabel('Silhouette coefficient')
    
    # Plot 3: Distance matrix
    plt.subplot(1, 3, 3)
    distances = cdist(X, X)
    plt.imshow(distances, cmap='viridis')
    plt.title('Pairwise Distances')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()

def plot_homogeneity_completeness():
    """Visualize homogeneity and completeness concepts"""
    # Create dataset with known structure
    X, y_true = make_blobs(n_samples=150, centers=3, cluster_std=0.5, random_state=42)
    
    # Create different clustering scenarios
    kmeans_good = KMeans(n_clusters=3, random_state=42).fit(X)
    kmeans_over = KMeans(n_clusters=5, random_state=42).fit(X)
    kmeans_under = KMeans(n_clusters=2, random_state=42).fit(X)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('Homogeneity and Completeness Visualization', fontsize=16)
    
    # True labels
    axes[0, 0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis')
    axes[0, 0].set_title('True Labels')
    
    # Good clustering
    axes[0, 1].scatter(X[:, 0], X[:, 1], c=kmeans_good.labels_, cmap='viridis')
    axes[0, 1].set_title('Good Clustering\nHigh Homogeneity & Completeness')
    
    # Over-clustering
    axes[1, 0].scatter(X[:, 0], X[:, 1], c=kmeans_over.labels_, cmap='viridis')
    axes[1, 0].set_title('Over-clustering\nHigh Homogeneity, Lower Completeness')
    
    # Under-clustering
    axes[1, 1].scatter(X[:, 0], X[:, 1], c=kmeans_under.labels_, cmap='viridis')
    axes[1, 1].set_title('Under-clustering\nLower Homogeneity, High Completeness')
    
    plt.tight_layout()
    plt.show()

def plot_ari_ami_comparison():
    """Visualize differences between ARI and AMI"""
    # Create datasets with different characteristics
    datasets = create_example_clusters()
    titles = ['Well-separated', 'Overlapping', 'Non-globular']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('ARI and AMI Comparison Across Different Cluster Types', fontsize=16)
    
    for idx, ((X, y_true), title) in enumerate(zip(datasets, titles)):
        kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
        axes[idx].scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
        axes[idx].set_title(f'{title} Clusters')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("=== Theoretical Understanding of Clustering Metrics ===\n")
    
    print("1. Inertia (Within-cluster Sum of Squares)")
    print("-------------------------------------------")
    print("• Mathematical definition: Σ(x - μc)²")
    print("• Measures compactness of clusters")
    print("• Lower values indicate tighter clusters")
    plot_inertia_explanation()
    
    print("\n2. Silhouette Analysis")
    print("----------------------")
    print("• Measures how similar points are to their own cluster vs other clusters")
    print("• Range: [-1, 1] where higher is better")
    print("• Calculated as: (b - a) / max(a, b)")
    print("  where a = mean intra-cluster distance")
    print("        b = mean nearest-cluster distance")
    plot_silhouette_analysis()
    
    print("\n3. Homogeneity and Completeness")
    print("-------------------------------")
    print("• Homogeneity: Each cluster contains only members of a single class")
    print("• Completeness: All members of a class are assigned to the same cluster")
    print("• V-measure: Harmonic mean of homogeneity and completeness")
    plot_homogeneity_completeness()
    
    print("\n4. ARI and AMI")
    print("-------------")
    print("• ARI (Adjusted Rand Index):")
    print("  - Based on pair counting")
    print("  - Measures agreement between two partitions")
    print("• AMI (Adjusted Mutual Information):")
    print("  - Based on information theory")
    print("  - Measures mutual information between true and predicted labels")
    plot_ari_ami_comparison() 