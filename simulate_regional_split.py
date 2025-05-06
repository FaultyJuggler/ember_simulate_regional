#!/usr/bin/env python3
import os
import pickle
import numpy as np
import json
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import Counter


def analyze_features(X, feature_dim=None):
    """
    Analyze features in the dataset and print statistics.

    Args:
        X: Feature matrix
        feature_dim: Expected feature dimension (optional)
    """
    print("\n===== Feature Analysis =====")

    if X is None or len(X) == 0:
        print("No features to analyze.")
        return

    # Basic statistics
    num_samples = X.shape[0]
    actual_feature_dim = X.shape[1]
    print(f"Total samples: {num_samples}")
    print(f"Feature dimension: {actual_feature_dim}")

    if feature_dim is not None and actual_feature_dim != feature_dim:
        print(f"WARNING: Feature dimension {actual_feature_dim} doesn't match expected {feature_dim}")

    # Check for NaN or infinite values
    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    print(f"NaN values: {nan_count}")
    print(f"Infinite values: {inf_count}")

    # Check for zero features
    zero_features = (X == 0).all(axis=0).sum()
    print(f"Features that are all zeros: {zero_features} ({zero_features / actual_feature_dim * 100:.2f}%)")

    # Check for constant features
    constant_features = 0
    for i in range(actual_feature_dim):
        if len(np.unique(X[:, i])) == 1:
            constant_features += 1
    print(f"Constant features: {constant_features} ({constant_features / actual_feature_dim * 100:.2f}%)")

    # Feature value ranges
    min_values = X.min(axis=0)
    max_values = X.max(axis=0)
    mean_values = X.mean(axis=0)
    std_values = X.std(axis=0)

    print(f"Feature value ranges:")
    print(f"  Min: {min_values.min():.2f}")
    print(f"  Max: {max_values.max():.2f}")
    print(f"  Mean: {mean_values.mean():.2f}")
    print(f"  Std: {std_values.mean():.2f}")

    # Top features with highest variance
    feature_variance = np.var(X, axis=0)
    top_n = 10
    top_var_indices = np.argsort(-feature_variance)[:top_n]
    print(f"\nTop {top_n} features with highest variance:")
    for i, idx in enumerate(top_var_indices):
        print(
            f"  {i + 1}. Feature {idx}: Variance = {feature_variance[idx]:.2f}, Range = [{min_values[idx]:.2f}, {max_values[idx]:.2f}]")

    print("=============================\n")


def split_ember_dataset(data_dir, output_dir, num_regions=3, feature_ranges=None):
    """
    Splits the EMBER dataset by feature characteristics across regions, with benign samples in separate folder.

    Args:
        data_dir: Directory containing the prepared EMBER dataset (.pkl files)
        output_dir: Directory where to save the regional datasets
        num_regions: Number of regions to create (defaults to 3 - US, JP, EU)
        feature_ranges: Optional dictionary defining feature ranges for each region.
                       If None, uses K-means clustering to determine regions.

    Returns:
        Dictionary with metadata about the regional datasets
    """
    # Hardcode the number of regions to 3
    num_regions = 3
    region_names = ["US", "JP", "EU"]

    # Check if the prepared files exist
    x_train_path = os.path.join(data_dir, "X_train.pkl")
    y_train_path = os.path.join(data_dir, "y_train.pkl")

    required_files = [x_train_path, y_train_path]
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        print(f"Error: Required files not found at {data_dir}")
        print("Make sure you've downloaded the EMBER dataset files")
        print(f"Expected files: {', '.join(required_files)}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Files in directory: {os.listdir(data_dir)}")
        return None

    # Load the dataset
    try:
        with open(x_train_path, 'rb') as f:
            X_train = pickle.load(f)
        with open(y_train_path, 'rb') as f:
            y_train = pickle.load(f)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

    # Analyze features before splitting
    analyze_features(X_train)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get indices of benign and malware samples
    benign_indices = np.where(y_train == 0)[0]
    malware_indices = np.where(y_train == 1)[0]

    print(f"Total samples: {len(X_train)}")
    print(f"Benign samples: {len(benign_indices)}")
    print(f"Malware samples: {len(malware_indices)}")

    # Create a separate folder for benign samples
    benign_dir = os.path.join(output_dir, "benign")
    os.makedirs(benign_dir, exist_ok=True)

    # Save benign samples
    X_benign = X_train[benign_indices]
    y_benign = y_train[benign_indices]

    with open(os.path.join(benign_dir, "X_train.pkl"), 'wb') as f:
        pickle.dump(X_benign, f)

    with open(os.path.join(benign_dir, "y_train.pkl"), 'wb') as f:
        pickle.dump(y_benign, f)

    # Check if we have malware samples
    if len(malware_indices) == 0:
        print("No malware samples found. Creating synthetic malware samples for demonstration.")

        # Randomly select some benign samples to convert to synthetic malware
        num_synthetic_malware = min(len(benign_indices) // 10, 10000)  # 10% of benign or max 10,000
        np.random.seed(42)  # For reproducibility
        synthetic_malware_indices = np.random.choice(benign_indices, num_synthetic_malware, replace=False)

        # Create synthetic malware by perturbing benign samples
        X_synthetic_malware = X_train[synthetic_malware_indices].copy()

        # Apply some perturbation to make them "malware-like" (increase certain feature values)
        # This is just for demonstration purposes - not actual malware generation
        perturb_scale = 0.2
        X_synthetic_malware += np.random.normal(0, perturb_scale, X_synthetic_malware.shape)

        # Create labels for synthetic malware
        y_synthetic_malware = np.ones(len(synthetic_malware_indices))

        print(f"Generated {num_synthetic_malware} synthetic malware samples for demonstration")

        # Use these samples for clustering instead
        X_malware = X_synthetic_malware
        is_synthetic = True
    else:
        # Extract malware samples for clustering
        X_malware = X_train[malware_indices]
        is_synthetic = False

    # Visualize malware samples using PCA (for 2D projection)
    try:
        print("Generating PCA visualization of malware samples...")
        pca = PCA(n_components=2)
        X_malware_pca = pca.fit_transform(X_malware)

        plt.figure(figsize=(10, 8))
        plt.scatter(X_malware_pca[:, 0], X_malware_pca[:, 1], alpha=0.5)
        plt.title('PCA Projection of Malware Samples')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')

        # Save the visualization
        viz_path = os.path.join(output_dir, "malware_pca_visualization.png")
        plt.savefig(viz_path)
        print(f"PCA visualization saved to {viz_path}")
    except Exception as e:
        print(f"Error generating PCA visualization: {e}")

    # Use K-means clustering to split malware samples by feature characteristics
    print(f"Clustering malware samples into {num_regions} regions based on features...")
    kmeans = KMeans(n_clusters=num_regions, random_state=42, n_init=10)
    malware_clusters = kmeans.fit_predict(X_malware)

    # Split malware indices by cluster
    malware_splits = [[] for _ in range(num_regions)]
    for i, cluster_id in enumerate(malware_clusters):
        malware_splits[cluster_id].append(i)  # Store indices into X_malware, not original indices

    # Convert lists to numpy arrays
    malware_splits = [np.array(indices) for indices in malware_splits]

    # Generate cluster statistics
    print("\n===== Cluster Statistics =====")
    for i, region_name in enumerate(region_names):
        cluster_size = len(malware_splits[i])
        percentage = (cluster_size / len(X_malware)) * 100
        print(f"Cluster {i} ({region_name}): {cluster_size} samples ({percentage:.2f}%)")

        # Get cluster centroid features
        centroid = kmeans.cluster_centers_[i]

        # Find top features that differentiate this cluster
        if len(malware_splits) > 1:  # Only if we have more than one cluster
            other_indices = np.concatenate([malware_splits[j] for j in range(num_regions) if j != i])
            this_cluster = X_malware[malware_splits[i]]
            other_clusters = X_malware[other_indices]

            this_mean = np.mean(this_cluster, axis=0)
            other_mean = np.mean(other_clusters, axis=0)

            # Calculate feature importance as the absolute difference in means
            feature_importance = np.abs(this_mean - other_mean)

            # Get top 5 most distinctive features
            top_features = np.argsort(-feature_importance)[:5]
            print("  Top distinctive features:")
            for idx in top_features:
                print(f"    Feature {idx}: This cluster = {this_mean[idx]:.2f}, Others = {other_mean[idx]:.2f}")

    print("============================\n")

    # Try to visualize the clusters if we have matplotlib
    try:
        print("Generating cluster visualization...")
        plt.figure(figsize=(10, 8))

        for i, region_name in enumerate(region_names):
            cluster_points = X_malware_pca[malware_splits[i]]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], alpha=0.5, label=f"Cluster {i} ({region_name})")

        # Add centroids to the plot
        centroids_pca = pca.transform(kmeans.cluster_centers_)
        plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], s=100, c='black', marker='X')

        plt.title('Malware Samples Clustered by Region')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()

        # Save the visualization
        viz_path = os.path.join(output_dir, "malware_clusters_visualization.png")
        plt.savefig(viz_path)
        print(f"Cluster visualization saved to {viz_path}")
    except Exception as e:
        print(f"Error generating cluster visualization: {e}")

    # Create regional datasets for malware
    metadata = {
        "total_samples": len(X_train),
        "num_regions": num_regions,
        "feature_dim": X_train.shape[1],
        "benign_samples": len(benign_indices),
        "malware_samples": len(malware_indices) if not is_synthetic else len(X_malware),
        "is_synthetic_malware": is_synthetic,
        "regions": {}
    }

    for region_id in range(num_regions):
        # Get malware indices for this region
        region_cluster_indices = malware_splits[region_id]

        if is_synthetic:
            # For synthetic malware, just use the clustered synthetic samples directly
            X_region = X_malware[region_cluster_indices]
            y_region = np.ones(len(region_cluster_indices))
        else:
            # For real malware, map back to the original indices
            region_original_indices = malware_indices[region_cluster_indices]
            X_region = X_train[region_original_indices]
            y_region = y_train[region_original_indices]

        # Use region names for directories
        region_name = region_names[region_id]
        region_dir = os.path.join(output_dir, f"region_{region_name}")
        os.makedirs(region_dir, exist_ok=True)

        with open(os.path.join(region_dir, "X_train.pkl"), 'wb') as f:
            pickle.dump(X_region, f)

        with open(os.path.join(region_dir, "y_train.pkl"), 'wb') as f:
            pickle.dump(y_region, f)

        # Save metadata for this region
        metadata["regions"][region_name] = {
            "num_samples": len(region_cluster_indices),
            "num_malware": len(region_cluster_indices),
            "feature_centroid": kmeans.cluster_centers_[region_id].tolist() if hasattr(kmeans,
                                                                                       'cluster_centers_') else None
        }

    # Save overall metadata
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    return metadata


if __name__ == "__main__":
    # Use current working directory as base path
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, "data", "ember2018")

    # Check if directory exists
    if not os.path.exists(data_dir):
        # Try another common location
        data_dir = os.path.join(current_dir, "ember", "data", "ember2018")
        if not os.path.exists(data_dir):
            # If that doesn't exist either, check if EMBER files are in the current directory
            files = os.listdir(current_dir)
            if any(f.endswith('.pkl') for f in files):
                data_dir = current_dir
            else:
                print("Could not find EMBER dataset directory.")
                print(f"Looked in: {os.path.join(current_dir, 'data', 'ember2018')}")
                print(f"And in: {os.path.join(current_dir, 'ember', 'data', 'ember2018')}")
                print(f"And in current directory: {current_dir}")
                print("Please provide the correct path as first argument")
                import sys

                if len(sys.argv) > 1:
                    data_dir = sys.argv[1]
                else:
                    sys.exit(1)

    output_dir = os.path.join(current_dir, "regional_data")
    print(f"Using EMBER data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    metadata = split_ember_dataset(data_dir, output_dir)

    if metadata:
        print(f"Created regional datasets with {metadata['total_samples']} total samples")
        print(f"Benign samples: {metadata['benign_samples']} (in 'benign' folder)")
        print("Region details (malware only):")
        for region_id, region_info in metadata['regions'].items():
            print(f"  Region {region_id}: {region_info['num_samples']} malware samples")