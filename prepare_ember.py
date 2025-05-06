#!/usr/bin/env python3
import os
import pickle
import numpy as np
import json
import sys
from tqdm import tqdm


def extract_features_from_sample(sample):
    """
    Extract a feature vector from an EMBER sample record.
    Based on the sample record structure provided.

    Args:
        sample: Dictionary containing the raw features

    Returns:
        Numpy array of features
    """
    feature_vector = []

    # 1. Byte histogram features (256 values)
    if 'histogram' in sample and isinstance(sample['histogram'], list):
        feature_vector.extend(sample['histogram'])
    else:
        feature_vector.extend([0] * 256)

    # 2. Byte entropy histogram (256 values)
    if 'byteentropy' in sample and isinstance(sample['byteentropy'], list):
        feature_vector.extend(sample['byteentropy'])
    else:
        feature_vector.extend([0] * 256)

    # 3. String features
    if 'strings' in sample and isinstance(sample['strings'], dict):
        strings_features = [
            sample['strings'].get('numstrings', 0),
            sample['strings'].get('avlength', 0),
            sample['strings'].get('entropy', 0),
            sample['strings'].get('paths', 0),
            sample['strings'].get('urls', 0),
            sample['strings'].get('registry', 0),
            sample['strings'].get('MZ', 0),
            sample['strings'].get('printables', 0)
        ]
        # Add printabledist if available (typically 96 values)
        if 'printabledist' in sample['strings'] and isinstance(sample['strings']['printabledist'], list):
            strings_features.extend(sample['strings']['printabledist'])
        else:
            strings_features.extend([0] * 96)

        feature_vector.extend(strings_features)
    else:
        feature_vector.extend([0] * 104)  # 8 core string features + 96 printabledist

    # 4. General file info
    if 'general' in sample and isinstance(sample['general'], dict):
        general_features = [
            sample['general'].get('size', 0),
            sample['general'].get('vsize', 0),
            sample['general'].get('has_debug', 0),
            sample['general'].get('exports', 0),
            sample['general'].get('imports', 0),
            sample['general'].get('has_relocations', 0),
            sample['general'].get('has_resources', 0),
            sample['general'].get('has_signature', 0),
            sample['general'].get('has_tls', 0),
            sample['general'].get('symbols', 0)
        ]
        feature_vector.extend(general_features)
    else:
        feature_vector.extend([0] * 10)

    # 5. Section info
    section_features = []
    if 'section' in sample and isinstance(sample['section'], dict):
        # Count number of sections
        num_sections = len(sample['section'].get('sections', []))
        section_features.append(num_sections)

        # Get entry section name
        entry_section = sample['section'].get('entry', '')
        section_features.append(1 if entry_section else 0)

        # Extract section properties for up to 5 sections (adjust if needed)
        max_sections = 5
        for i in range(min(num_sections, max_sections)):
            section = sample['section']['sections'][i]
            section_features.extend([
                section.get('size', 0),
                section.get('entropy', 0),
                section.get('vsize', 0)
            ])

            # Add section property flags
            props = section.get('props', [])
            section_features.extend([
                1 if 'CNT_CODE' in props else 0,
                1 if 'CNT_INITIALIZED_DATA' in props else 0,
                1 if 'CNT_UNINITIALIZED_DATA' in props else 0,
                1 if 'MEM_EXECUTE' in props else 0,
                1 if 'MEM_READ' in props else 0,
                1 if 'MEM_WRITE' in props else 0
            ])

        # Pad for missing sections
        pad_sections = max_sections - num_sections
        for _ in range(pad_sections):
            section_features.extend([0] * 9)  # 3 numeric + 6 boolean features per section
    else:
        section_features = [0] * (1 + 1 + 5 * 9)  # num_sections + entry + (5 sections * 9 features)

    feature_vector.extend(section_features)

    # 6. Import features
    import_features = []
    if 'imports' in sample and isinstance(sample['imports'], dict):
        # Count total imports
        total_imports = sum(len(funcs) for funcs in sample['imports'].values())
        import_features.append(total_imports)

        # Count number of imported DLLs
        num_dlls = len(sample['imports'])
        import_features.append(num_dlls)

        # Check for specific common DLLs
        common_dlls = ['KERNEL32.dll', 'USER32.dll', 'GDI32.dll', 'SHELL32.dll',
                       'ADVAPI32.dll', 'COMCTL32.dll', 'ole32.dll', 'VERSION.dll']
        for dll in common_dlls:
            import_features.append(1 if dll in sample['imports'] else 0)

        # Count functions per common DLL
        for dll in common_dlls:
            import_features.append(len(sample['imports'].get(dll, [])))
    else:
        import_features = [0] * (2 + len(common_dlls) * 2)

    feature_vector.extend(import_features)

    # 7. Basic export features
    export_features = []
    if 'exports' in sample and isinstance(sample['exports'], list):
        num_exports = len(sample['exports'])
        export_features.append(num_exports)
    else:
        export_features.append(0)

    feature_vector.extend(export_features)

    # 8. Data directories
    datadirectory_features = []
    if 'datadirectories' in sample and isinstance(sample['datadirectories'], list):
        # Get count of non-zero entries
        num_directories = sum(1 for d in sample['datadirectories'] if d.get('size', 0) > 0)
        datadirectory_features.append(num_directories)

        # Add flags for specific directories
        directory_names = ['EXPORT_TABLE', 'IMPORT_TABLE', 'RESOURCE_TABLE', 'EXCEPTION_TABLE',
                           'CERTIFICATE_TABLE', 'BASE_RELOCATION_TABLE', 'DEBUG', 'ARCHITECTURE',
                           'GLOBAL_PTR', 'TLS_TABLE', 'LOAD_CONFIG_TABLE', 'BOUND_IMPORT',
                           'IAT', 'DELAY_IMPORT_DESCRIPTOR', 'CLR_RUNTIME_HEADER']

        # Create a dict for faster lookup
        directories = {d.get('name', ''): d for d in sample['datadirectories']}

        for name in directory_names:
            directory = directories.get(name, {})
            datadirectory_features.extend([
                1 if directory.get('size', 0) > 0 else 0,
                directory.get('size', 0),
                directory.get('virtual_address', 0)
            ])
    else:
        datadirectory_features = [0] * (1 + len(directory_names) * 3)

    feature_vector.extend(datadirectory_features)

    return np.array(feature_vector, dtype=np.float32)


def analyze_dataset_labels(y_train):
    """
    Analyze the distribution of labels in the dataset.

    Args:
        y_train: Array of labels

    Returns:
        Dictionary with statistics about the labels
    """
    total_samples = len(y_train)

    # Count label distribution
    benign_count = np.sum(y_train == 0)
    malware_count = np.sum(y_train == 1)
    unknown_count = np.sum((y_train != 0) & (y_train != 1))

    # Calculate percentages
    if total_samples > 0:
        benign_percent = (benign_count / total_samples) * 100
        malware_percent = (malware_count / total_samples) * 100
        unknown_percent = (unknown_count / total_samples) * 100
    else:
        benign_percent = malware_percent = unknown_percent = 0

    # Print detailed statistics
    print("\n===== Dataset Label Analysis =====")
    print(f"Total samples: {total_samples}")
    print(f"Benign samples: {benign_count} ({benign_percent:.2f}%)")
    print(f"Malware samples: {malware_count} ({malware_percent:.2f}%)")

    if unknown_count > 0:
        print(f"Unknown/other labels: {unknown_count} ({unknown_percent:.2f}%)")

    # Check if dataset is imbalanced
    if total_samples > 0:
        if malware_count == 0:
            print("\nWARNING: No malware samples detected in the dataset.")
            print("This may cause issues with the regional splitting that expects both classes.")
        elif benign_count == 0:
            print("\nWARNING: No benign samples detected in the dataset.")
        elif min(malware_percent, benign_percent) < 10:
            print(f"\nWARNING: Dataset is highly imbalanced.")
            print(f"The minority class represents only {min(malware_percent, benign_percent):.2f}% of the samples.")

    print("=====================================\n")

    return {
        "total_samples": total_samples,
        "benign_count": int(benign_count),
        "malware_count": int(malware_count),
        "unknown_count": int(unknown_count),
        "benign_percent": benign_percent,
        "malware_percent": malware_percent,
        "unknown_percent": unknown_percent
    }


def prepare_ember_data(ember_data_dir):
    """
    Prepares EMBER dataset by loading and processing the JSONL files

    Args:
        ember_data_dir: Directory containing the EMBER dataset files

    Returns:
        Dictionary with metadata about the prepared dataset
    """
    print(f"Starting to prepare EMBER dataset from {ember_data_dir}...")

    # Check if the directory exists
    if not os.path.exists(ember_data_dir):
        print(f"Error: Directory {ember_data_dir} does not exist")
        return None

    # List files in the directory
    files = os.listdir(ember_data_dir)
    print(f"Files in directory: {files}")

    try:
        # Check if pickle files already exist
        x_train_pkl = os.path.join(ember_data_dir, "X_train.pkl")
        y_train_pkl = os.path.join(ember_data_dir, "y_train.pkl")

        if os.path.exists(x_train_pkl) and os.path.exists(y_train_pkl):
            print("Pickle files already exist. Loading them...")
            with open(x_train_pkl, 'rb') as f:
                X_train = pickle.load(f)
            with open(y_train_pkl, 'rb') as f:
                y_train = pickle.load(f)
        else:
            # Try to load from .dat files first
            x_train_dat = os.path.join(ember_data_dir, "X_train.dat")
            y_train_dat = os.path.join(ember_data_dir, "y_train.dat")

            if os.path.exists(x_train_dat) and os.path.exists(y_train_dat):
                print("Loading from .dat files...")
                X_train = np.memmap(x_train_dat, dtype=np.float32, mode='r').copy()
                # Reshape the array - EMBER features are typically 2381 dimensions
                num_samples = X_train.shape[0] // 2381
                X_train = X_train.reshape(num_samples, 2381)
                print(f"Loaded X_train with shape: {X_train.shape}")

                y_train = np.memmap(y_train_dat, dtype=np.float32, mode='r').copy()
                print(f"Loaded y_train with shape: {y_train.shape}")
            else:
                # If .dat files don't exist, process the JSONL files
                print("Processing JSONL files...")

                # Find all training feature files
                train_feature_files = sorted(
                    [f for f in files if f.startswith("train_features_") and f.endswith(".jsonl")])

                if not train_feature_files:
                    print("Error: No training feature files found")
                    return None

                print(f"Found {len(train_feature_files)} training feature files: {train_feature_files}")

                # Process all files
                all_samples = []
                all_labels = []

                for file in train_feature_files:
                    file_path = os.path.join(ember_data_dir, file)
                    print(f"Processing {file}...")
                    with open(file_path, 'r') as f:
                        for line in tqdm(f, desc=f"Loading {file}"):
                            try:
                                sample = json.loads(line)
                                all_samples.append(sample)

                                # Extract label if available
                                if 'label' in sample:
                                    label = sample['label']
                                    all_labels.append(label)
                            except json.JSONDecodeError:
                                print(f"Error decoding JSON in {file_path}")

                print(f"Loaded {len(all_samples)} samples")

                # Check if we have labels for all samples
                if len(all_labels) != len(all_samples):
                    print("Warning: Number of labels doesn't match number of samples")
                    print(f"Samples: {len(all_samples)}, Labels: {len(all_labels)}")

                    # If no labels at all, create default labels (all benign)
                    if len(all_labels) == 0:
                        print("No labels found. Creating default labels (all benign).")
                        all_labels = [0] * len(all_samples)

                # Convert raw samples to feature vectors
                print("Converting samples to feature vectors...")
                feature_vectors = []
                for i, sample in enumerate(tqdm(all_samples, desc="Extracting features")):
                    try:
                        feature_vector = extract_features_from_sample(sample)
                        feature_vectors.append(feature_vector)
                    except Exception as e:
                        print(f"Error extracting features from sample {i}: {e}")
                        # Use a zero vector as fallback
                        feature_vectors.append(np.zeros(800, dtype=np.float32))

                # Convert to numpy arrays
                X_train = np.array(feature_vectors, dtype=np.float32)
                y_train = np.array(all_labels, dtype=np.float32)

                print(f"Created feature array with shape: {X_train.shape}")
                print(f"Created label array with shape: {y_train.shape}")

                # Check feature dimensions (expected to be uniform)
                if len(X_train) > 0:
                    feature_dim = X_train.shape[1]
                    print(f"Feature dimension: {feature_dim}")

                # Save as pickle files for future use
                print("Saving dataset as pickle files...")
                with open(x_train_pkl, 'wb') as f:
                    pickle.dump(X_train, f)

                with open(y_train_pkl, 'wb') as f:
                    pickle.dump(y_train, f)

        print("Dataset prepared successfully.")

        # Analyze label distribution
        label_stats = analyze_dataset_labels(y_train)

        # Return metadata
        metadata = {
            "total_samples": len(X_train),
            "feature_dim": X_train.shape[1] if len(X_train.shape) > 1 else 1,
            "X_train_shape": X_train.shape,
            "y_train_shape": y_train.shape,
            "benign_count": label_stats["benign_count"],
            "malware_count": label_stats["malware_count"],
            "unknown_count": label_stats.get("unknown_count", 0),
            "benign_percent": label_stats["benign_percent"],
            "malware_percent": label_stats["malware_percent"]
        }

        return metadata

    except Exception as e:
        print(f"Error preparing EMBER dataset: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) > 1:
        ember_data_dir = sys.argv[1]
    else:
        # Use current working directory as base path
        current_dir = os.getcwd()
        ember_data_dir = os.path.join(current_dir, "data", "ember2018")

        # Check if directory exists
        if not os.path.exists(ember_data_dir):
            # Try another common location
            ember_data_dir = os.path.join(current_dir, "ember", "data", "ember2018")
            if not os.path.exists(ember_data_dir):
                # Check current directory itself
                if os.path.exists(os.path.join(current_dir, "X_train.dat")) or any(
                        f.startswith("train_features_") for f in os.listdir(current_dir)):
                    ember_data_dir = current_dir
                else:
                    print("Could not find EMBER dataset directory.")
                    print(f"Looked in: {os.path.join(current_dir, 'data', 'ember2018')}")
                    print(f"And in: {os.path.join(current_dir, 'ember', 'data', 'ember2018')}")
                    print(f"And in current directory: {current_dir}")
                    print("Please provide the correct path as an argument:")
                    print("python prepare_ember.py /path/to/ember/data")
                    sys.exit(1)

    print(f"Using EMBER data directory: {ember_data_dir}")
    metadata = prepare_ember_data(ember_data_dir)

    if metadata:
        print(f"\nPrepared EMBER dataset with {metadata['total_samples']} samples")
        print(f"Benign: {metadata['benign_count']} ({metadata['benign_percent']:.2f}%)")
        print(f"Malware: {metadata['malware_count']} ({metadata['malware_percent']:.2f}%)")

        # Check if artificial malware generation is needed
        if metadata['malware_count'] == 0:
            print("\nWould you like to generate synthetic malware samples for demonstration purposes? (y/n)")
            response = input().lower()
            if response.startswith('y'):
                print("You can modify simulate_regional_split.py to create synthetic malware samples.")
                print("The script already has a fallback for when no malware is detected.")
    else:
        print("Failed to prepare EMBER dataset")
        sys.exit(1)