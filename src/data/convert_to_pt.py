"""
Convert WESAD_combined_time_series.pkl to PyTorch format and create train/test splits.
This script loads the PKL file, restructures the data, and saves it as a PyTorch tensor.
"""
import os
import pickle
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import argparse
from pathlib import Path

# Constants
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = PROJECT_ROOT / "data/processed"
TARGET_LENGTH = 30  # Fixed length of 30 time steps for BVP and EDA signals

def load_pkl_data(input_path):
    """
    Load data from pickle file.
    
    Args:
        input_path (str): Path to the pickle file
        
    Returns:
        list: List of sample dictionaries
    """
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loaded {len(data)} samples from {input_path}")
    return data


def reorganize_data(samples):
    """
    Reorganize data into a dictionary structure suitable for PyTorch models.
    
    Args:
        samples (list): List of sample dictionaries
        
    Returns:
        dict: Reorganized data with structure {data_type: {condition: samples}}
    """
    # Initialize data structure
    organized_data = {
        'EDA_series': {},
        'BVP_series': {},
        'HRV_features': {},
        'EDA_features': {}
    }
    
    # Group samples by condition/label
    for sample in samples:
        label = int(sample['label'])
        
        # Extract and organize EDA series
        if 'EDA_series' in sample and len(sample['EDA_series']) > 0:
            if label not in organized_data['EDA_series']:
                organized_data['EDA_series'][label] = []
            organized_data['EDA_series'][label].append(sample['EDA_series'])
        
        # Extract and organize BVP series
        if 'BVP_series' in sample and len(sample['BVP_series']) > 0:
            if label not in organized_data['BVP_series']:
                organized_data['BVP_series'][label] = []
            organized_data['BVP_series'][label].append(sample['BVP_series'])
        
        # Extract and organize HRV features
        if 'HRV_metrics' in sample and sample['HRV_metrics']:
            # Extract specific HRV metrics and convert to array
            hrv_features = np.array([
                sample['HRV_metrics'].get('RMSSD', 0),
                sample['HRV_metrics'].get('SDNN', 0),
                sample['HRV_metrics'].get('LF', 0),
                sample['HRV_metrics'].get('HF', 0),
                sample['HRV_metrics'].get('LF_HF_ratio', 0)
            ])
            
            if label not in organized_data['HRV_features']:
                organized_data['HRV_features'][label] = []
            organized_data['HRV_features'][label].append(hrv_features)
        
        # Extract and organize EDA features
        if 'EDA_features' in sample and sample['EDA_features']:
            # Convert EDA features to array
            eda_features = np.array([
                sample['EDA_features'].get('mean_EDA', 0),
                sample['EDA_features'].get('median_EDA', 0),
                sample['EDA_features'].get('SCR_count', 0)
            ])
            
            if label not in organized_data['EDA_features']:
                organized_data['EDA_features'][label] = []
            organized_data['EDA_features'][label].append(eda_features)
    
    # Convert feature lists to numpy arrays (these should have consistent shapes)
    for data_type in ['HRV_features', 'EDA_features']:
        for label in organized_data[data_type]:
            organized_data[data_type][label] = np.array(organized_data[data_type][label])
    
    return organized_data


def downsample_signals(data):
    """
    Downsample time series data to exactly TARGET_LENGTH points using linear interpolation.
    
    Args:
        data (dict): Organized data
        
    Returns:
        dict: Data with downsampled sequences
    """
    for data_type in ['EDA_series', 'BVP_series']:
        if data_type in data:
            for label in data[data_type]:
                downsampled_samples = []
                for sample in data[data_type][label]:
                    if len(sample) <= 1:  # Skip samples that are too short
                        continue
                        
                    # Create new indices evenly spaced
                    original_indices = np.arange(len(sample))
                    new_indices = np.linspace(0, len(sample)-1, TARGET_LENGTH)
                    
                    # Resample to TARGET_LENGTH points
                    downsampled = np.interp(new_indices, original_indices, sample)
                    downsampled_samples.append(downsampled)
                
                if downsampled_samples:
                    data[data_type][label] = np.array(downsampled_samples)
                else:
                    # If no valid samples, remove this label from the data type
                    del data[data_type][label]
    
    return data


def reshape_for_model(data):
    """
    Reshape data for model input. Time series data should be 3D (samples, time_steps, features).
    
    Args:
        data (dict): Organized data
        
    Returns:
        dict: Reshaped data
    """
    for data_type in ['EDA_series', 'BVP_series']:
        if data_type in data:
            for label in data[data_type]:
                # For time series, reshape to (samples, time_steps, 1)
                data[data_type][label] = data[data_type][label].reshape(
                    data[data_type][label].shape[0], 
                    data[data_type][label].shape[1], 
                    1
                )
    
    return data


def convert_to_pytorch(data):
    """
    Convert numpy arrays to PyTorch tensors.
    
    Args:
        data (dict): Organized and reshaped data
        
    Returns:
        dict: Data as PyTorch tensors
    """
    pt_data = {}
    for data_type in data:
        pt_data[data_type] = {}
        for label in data[data_type]:
            pt_data[data_type][label] = torch.tensor(data[data_type][label], dtype=torch.float32)
    
    return pt_data


def create_train_test_split(data, test_size=0.2, random_state=42):
    """
    Create train/test split for the data.
    
    Args:
        data (dict): PyTorch data
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_data, test_data)
    """
    train_data = {}
    test_data = {}
    
    # Get all conditions/labels
    all_labels = set()
    for data_type in data:
        all_labels.update(data[data_type].keys())
    
    # For each condition, perform stratified split
    for label in all_labels:
        # Get sample indices for this condition
        indices = {}
        sample_counts = {}
        
        for data_type in data:
            if label in data[data_type]:
                sample_counts[data_type] = len(data[data_type][label])
        
        # Use the data type with the most samples for splitting
        if sample_counts:
            main_data_type = max(sample_counts, key=sample_counts.get)
            num_samples = sample_counts[main_data_type]
            
            # Create indices for train/test split
            train_idx, test_idx = train_test_split(
                range(num_samples),
                test_size=test_size,
                random_state=random_state
            )
            
            # Split each data type using these indices
            for data_type in data:
                if label in data[data_type]:
                    if data_type not in train_data:
                        train_data[data_type] = {}
                    if data_type not in test_data:
                        test_data[data_type] = {}
                    
                    # Make sure we don't try to access more indices than we have samples
                    valid_train_idx = [i for i in train_idx if i < len(data[data_type][label])]
                    valid_test_idx = [i for i in test_idx if i < len(data[data_type][label])]
                    
                    train_data[data_type][label] = data[data_type][label][valid_train_idx]
                    test_data[data_type][label] = data[data_type][label][valid_test_idx]
    
    return train_data, test_data


def main(input_path, output_dir, test_size=0.2, random_state=42):
    """
    Main function to convert data and create train/test splits.
    
    Args:
        input_path (str): Path to the input PKL file
        output_dir (str): Directory to save the output files
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    samples = load_pkl_data(input_path)
    
    # Process data
    print("Reorganizing data...")
    data = reorganize_data(samples)
    
    print(f"Downsampling signals to {TARGET_LENGTH} time steps...")
    data = downsample_signals(data)
    
    print("Reshaping data for model input...")
    data = reshape_for_model(data)
    
    print("Converting to PyTorch tensors...")
    data = convert_to_pytorch(data)
    
    # Print data shapes
    print("\nData shapes after downsampling:")
    for data_type in data:
        print(f"\n{data_type}:")
        for label in data[data_type]:
            print(f"  Label {label}: {data[data_type][label].shape}")
    
    # Create train/test split
    print("\nCreating train/test split...")
    train_data, test_data = create_train_test_split(data, test_size, random_state)
    
    # Save data
    train_path = os.path.join(output_dir, "train_data.pt")
    test_path = os.path.join(output_dir, "test_data.pt")
    full_path = os.path.join(output_dir, "physio_data.pt")
    
    print("\nSaving data...")
    torch.save(train_data, train_path)
    torch.save(test_data, test_path)
    torch.save(data, full_path)
    
    print(f"\nData saved to:")
    print(f"  Train data: {train_path}")
    print(f"  Test data: {test_path}")
    print(f"  Full data: {full_path}")
    print(f"\nSignals have been downsampled to {TARGET_LENGTH} time steps for easier GAN training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert WESAD data to PyTorch format')
    parser.add_argument('--input', type=str, 
                        default=str(DATA_PROCESSED / 'WESAD_combined_time_series.pkl'),
                        help='Input PKL file')
    parser.add_argument('--output', type=str, 
                        default=str(DATA_PROCESSED),
                        help='Output directory')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data to use for testing')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--target_length', type=int, default=TARGET_LENGTH,
                        help='Target length for downsampled signals')
    
    args = parser.parse_args()
    
    # Update TARGET_LENGTH if provided via command line
    if args.target_length != TARGET_LENGTH:
        TARGET_LENGTH = args.target_length
        print(f"Using custom target length: {TARGET_LENGTH}")
    
    main(args.input, args.output, args.test_size, args.random_state) 