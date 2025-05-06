# !/usr/bin/env python3
import os
import sys
import argparse
from prepare_ember import prepare_ember_data
from simulate_regional_split import split_ember_dataset

# Parse command line arguments
parser = argparse.ArgumentParser(description='Process EMBER dataset and split it for regional federated learning.')
parser.add_argument('--data-dir', type=str, help='Directory containing the EMBER dataset')
parser.add_argument('--output-dir', type=str, default='/Users/felip/Developer/github/fed-reg-mal/ember_data/', help='Directory where to save the regional datasets')
args = parser.parse_args()

# Use current working directory as base path
current_dir = os.getcwd()

# Handle data directory
if args.data_dir:
    ember_data_dir = args.data_dir
else:
    ember_data_dir = os.path.join(current_dir, "data", "ember2018")

    # Check if directory exists
    if not os.path.exists(ember_data_dir):
        # Try another common location
        ember_data_dir = os.path.join(current_dir, "ember", "data", "ember2018")
        if not os.path.exists(ember_data_dir):
            # If that doesn't exist either, check if EMBER files are in the current directory
            files = os.listdir(current_dir)
            if any(f.endswith('.pkl') for f in files) or "X_train.dat" in files or any(
                    f.startswith("train_features") and f.endswith(".jsonl") for f in files):
                ember_data_dir = current_dir
            else:
                print("Could not find EMBER dataset directory.")
                print(f"Looked in: {os.path.join(current_dir, 'data', 'ember2018')}")
                print(f"And in: {os.path.join(current_dir, 'ember', 'data', 'ember2018')}")
                print(f"And in current directory: {current_dir}")
                print("Please specify the data directory with --data-dir")
                sys.exit(1)

# Handle output directory
if args.output_dir:
    output_dir = args.output_dir
else:
    output_dir = os.path.join(current_dir, "regional_data")

print(f"Using EMBER data directory: {ember_data_dir}")
print(f"Output directory will be: {output_dir}")

# Prepare the EMBER dataset if needed
print(f"Starting to process EMBER dataset from {ember_data_dir}...")
metadata = prepare_ember_data(ember_data_dir)

if not metadata:
    print("Error: Failed to prepare EMBER dataset")
    sys.exit(1)

# Split the dataset for regional federated learning
print("Splitting dataset into regions based on malware feature characteristics...")
print("All benign samples will be placed in a separate 'benign' folder.")
metadata = split_ember_dataset(ember_data_dir, output_dir)

if metadata:
    print(f"Created regional datasets with {metadata['total_samples']} total samples")
    print(f"Output directory: {output_dir}")
    print(f"Benign samples: {metadata['benign_samples']} (in 'benign' folder)")
    print("Region details (malware only):")
    for region_id, region_info in metadata['regions'].items():
        print(f"  Region {region_id}: {region_info['num_samples']} malware samples")
else:
    print("Error: Failed to create regional datasets")
    sys.exit(1)