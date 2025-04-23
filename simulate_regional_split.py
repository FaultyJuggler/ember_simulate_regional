import os
import sys
import numpy as np
import json
import pickle
import ember
from sklearn.model_selection import train_test_split


def split_ember_into_regions(ember_data_path, output_dir, year='2018'):
    """
    Split EMBER dataset into three "regional" datasets based on feature distributions.

    This function directly loads the pre-processed EMBER data files rather than
    trying to vectorize the raw features.
    """
    print(f"Starting to process EMBER dataset from {ember_data_path}...")

    # Convert to absolute paths if needed
    ember_data_path = os.path.abspath(ember_data_path)
    output_dir = os.path.abspath(output_dir)

    # Define paths to the numpy arrays that contain the vectorized features
    X_train_path = os.path.join(ember_data_path, f"X_train.pkl")
    y_train_path = os.path.join(ember_data_path, f"y_train.pkl")

    # Check if files exist
    if not os.path.exists(X_train_path) or not os.path.exists(y_train_path):
        print(f"Error: Required files not found at {ember_data_path}")
        print(f"Make sure you've downloaded the EMBER dataset files")
        print(f"Expected files: {X_train_path}, {y_train_path}")
        print(f"Current directory: {os.getcwd()}")
        print(
            f"Files in directory: {os.listdir(ember_data_path) if os.path.exists(ember_data_path) else 'Directory not found'}")
        return None

    # Load the pre-processed data directly
    print("Loading pre-processed EMBER data...")
    try:
        with open(X_train_path, 'rb') as f:
            X_train = pickle.load(f)
        with open(y_train_path, 'rb') as f:
            y_train = pickle.load(f)

        print(f"Successfully loaded data: X_train shape {X_train.shape}, y_train shape {y_train.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    # Create directory structure
    print(f"Creating directory structure in {output_dir}...")
    regions = ['US', 'BR', 'JP']
    for region in regions:
        # Create region directory if it doesn't exist
        region_dir = os.path.join(output_dir, region)
        os.makedirs(region_dir, exist_ok=True)

        # Create malware directory if it doesn't exist
        malware_dir = os.path.join(region_dir, 'malware')
        os.makedirs(malware_dir, exist_ok=True)

    # Create goodware directory if it doesn't exist
    goodware_dir = os.path.join(output_dir, 'goodware')
    os.makedirs(goodware_dir, exist_ok=True)

    print("Filtering and preparing samples...")

    # Get malware samples (label=1) where the label is not -1 (unlabeled)
    malware_indices = np.where((y_train == 1) & (y_train != -1))[0]
    malware_features = X_train[malware_indices]

    # Get goodware samples (label=0) where the label is not -1 (unlabeled)
    goodware_indices = np.where((y_train == 0) & (y_train != -1))[0]
    goodware_features = X_train[goodware_indices]

    print(f"Found {len(malware_indices)} malware samples and {len(goodware_indices)} goodware samples")

    # Split malware samples into three regions based on some feature
    # Choose a feature that would create meaningful splits
    feature_index = 10  # You might need to adjust this

    # Sort malware by this feature value
    print("Splitting malware samples by feature distribution...")
    sorted_indices = np.argsort(malware_features[:, feature_index])
    sorted_malware = malware_features[sorted_indices]

    # Split into three roughly equal parts
    split1 = len(sorted_malware) // 3
    split2 = 2 * split1

    # Create "regional" characteristics by adding small variations
    print("Creating region-specific variations...")
    region_malware = {
        'US': sorted_malware[:split1].copy(),
        'BR': sorted_malware[split1:split2].copy(),
        'JP': sorted_malware[split2:].copy()
    }

    # Add region-specific variations to make datasets distinct
    # For US: Increase importance of some section features
    region_malware['US'][:, 20:30] *= 1.2

    # For BR: Modify import features
    region_malware['BR'][:, 50:60] *= 1.3

    # For JP: Increase string features
    region_malware['JP'][:, 70:80] *= 1.5

    print(f"Saving datasets to {output_dir}...")

    # Save datasets in a format compatible with your project
    sample_counts = {}
    for region in regions:
        count = min(len(region_malware[region]), 1000)  # Limit to 1000 samples per region to keep it manageable
        sample_counts[region] = count
        print(f"Saving {count} samples for {region}...")

        for j in range(count):
            filename = f"{region}_malware_{j}.npz"
            filepath = os.path.join(output_dir, region, 'malware', filename)
            np.savez(filepath, features=region_malware[region][j])

    # Save goodware samples - use a subset equal to total malware
    goodware_count = min(len(goodware_features), sum(sample_counts.values()))
    print(f"Saving {goodware_count} goodware samples...")

    for j in range(goodware_count):
        if j >= len(goodware_features):
            break
        filename = f"goodware_{j}.npz"
        filepath = os.path.join(output_dir, 'goodware', filename)
        np.savez(filepath, features=goodware_features[j])

    # Save metadata about the splits
    metadata = {
        'total_samples': sum(sample_counts.values()) + goodware_count,
        'malware_samples': sum(sample_counts.values()),
        'goodware_samples': goodware_count,
        'US_samples': sample_counts['US'],
        'BR_samples': sample_counts['BR'],
        'JP_samples': sample_counts['JP'],
        'feature_split_index': feature_index,
        'ember_year': year
    }

    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"Metadata saved to {metadata_path}")
    print(f"Successfully created regional datasets with {metadata['total_samples']} total samples")

    return metadata


# This is the critical part for macOS multiprocessing
if __name__ == "__main__":
    # Define paths - update these to match your actual paths
    # Use relative paths that match your directory structure
    ember_data_path = "./data/ember2018"
    output_dir = "./federated_malware_detection/data"

    # Run the function
    metadata = split_ember_into_regions(
        ember_data_path=ember_data_path,
        output_dir=output_dir,
        year='2018'
    )

    if metadata:
        print("Script completed successfully!")
    else:
        print("Script failed. Please check the paths and errors above.")