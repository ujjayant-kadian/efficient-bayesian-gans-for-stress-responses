# Data Processing for Bayesian GAN Models

This directory contains scripts for data processing, specifically for converting the WESAD physiological data from pickle format to PyTorch tensors.

## Data Files Structure

The raw data comes from the WESAD dataset (Wearable Stress and Affect Detection), which contains physiological measurements collected from subjects under different stress conditions.

The processed data is stored in the following structure:

- `WESAD_combined_time_series.pkl` - Combined pickle file with all subjects' data
- `physio_data.pt` - PyTorch tensor file containing all data organized by data type and condition
- `train_data.pt` - PyTorch tensor file containing the training split
- `test_data.pt` - PyTorch tensor file containing the test split

## Data Format

The PyTorch data is organized as a nested dictionary with the following structure:

```
{
    'EDA_series': {
        1: tensor of shape [n_samples, time_steps, 1],
        2: tensor of shape [n_samples, time_steps, 1],
        3: tensor of shape [n_samples, time_steps, 1]
    },
    'BVP_series': {
        1: tensor of shape [n_samples, time_steps, 1],
        2: tensor of shape [n_samples, time_steps, 1],
        3: tensor of shape [n_samples, time_steps, 1]
    },
    'HRV_features': {
        1: tensor of shape [n_samples, 5],
        2: tensor of shape [n_samples, 5],
        3: tensor of shape [n_samples, 5]
    },
    'EDA_features': {
        1: tensor of shape [n_samples, 3],
        2: tensor of shape [n_samples, 3],
        3: tensor of shape [n_samples, 3]
    }
}
```

Where:
- The outer keys are data types ('EDA_series', 'BVP_series', 'HRV_features', 'EDA_features')
- The inner keys are condition labels (1: baseline, 2: stress, 3: amusement)
- The values are PyTorch tensors containing the actual data

## Converting PKL to PT

To convert the pickle data to PyTorch format and create train/test splits, run:

```bash
python -m src.data.convert_to_pt
```

This will:
1. Load the data from `data/processed/WESAD_combined_time_series.pkl`
2. Reorganize the data into a dictionary structure
3. Pad sequences to ensure consistent dimensions
4. Reshape the data for model input
5. Convert to PyTorch tensors
6. Create train/test splits
7. Save the data to PT files

### Optional Arguments

- `--input` - Path to the input PKL file (default: `data/processed/WESAD_combined_time_series.pkl`)
- `--output` - Output directory (default: `data/processed`)
- `--test_size` - Proportion of data to use for testing (default: 0.2)
- `--random_state` - Random seed for reproducibility (default: 42)

## Using the Data in Models

You can load the data in your model as follows:

```python
import torch

# Load all data
data = torch.load("data/processed/physio_data.pt")

# Load train data
train_data = torch.load("data/processed/train_data.pt")

# Load test data
test_data = torch.load("data/processed/test_data.pt")
``` 