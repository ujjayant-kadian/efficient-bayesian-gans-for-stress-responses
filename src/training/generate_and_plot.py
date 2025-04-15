#!/usr/bin/env python
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import logging
import sys
import glob # Added for finding files
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# --- Model Imports ---
# Attempt to import necessary model classes, handle potential ImportError
try:
    from src.models.base_gan import (TimeSeriesGAN, EDATimeSeriesGAN,
                                     EDAMultiScaleTimeSeriesGAN, FeatureGAN,
                                     BVPTimeSeriesGAN, BVPMultiScaleTimeSeriesGAN)
    from src.models.dropout_bayes_gan import (BayesianTimeSeriesGAN,
                                              BayesianEDATimeSeriesGAN,
                                              BayesianEDAMultiScaleTimeSeriesGAN,
                                              BayesianBVPTimeSeriesGAN,
                                              BayesianBVPMultiScaleTimeSeriesGAN,
                                              BayesianFeatureGAN)
    # Add imports for variational Bayesian models
    from src.models.variational_bayes_gan import (BayesTimeSeriesGAN,
                                                 BayesEDATimeSeriesGAN,
                                                 BayesEDAMultiScaleTimeSeriesGAN,
                                                 BayesBVPTimeSeriesGAN,
                                                 BayesBVPMultiScaleTimeSeriesGAN,
                                                 BayesFeatureGAN)
                                                 
    # Define a mapping from inferred parameters to model classes
    # This helps centralize the model selection logic
    MODEL_CLASS_MAP = {
        # Format: (approach_name, model_type, signal_type, use_multiscale): Class
        
        # Baseline models
        ('baseline', 'time_series', 'EDA', True): EDAMultiScaleTimeSeriesGAN,
        ('baseline', 'time_series', 'EDA', False): EDATimeSeriesGAN,
        ('baseline', 'time_series', 'BVP', True): BVPMultiScaleTimeSeriesGAN,
        ('baseline', 'time_series', 'BVP', False): BVPTimeSeriesGAN,
        ('baseline', 'features', 'EDA', False): FeatureGAN, # Multiscale N/A for features
        ('baseline', 'features', 'BVP', False): FeatureGAN, # Signal type ignored for features
        
        # Dropout-based Bayesian models
        ('dropout', 'time_series', 'EDA', True): BayesianEDAMultiScaleTimeSeriesGAN,
        ('dropout', 'time_series', 'EDA', False): BayesianEDATimeSeriesGAN,
        ('dropout', 'time_series', 'BVP', True): BayesianBVPMultiScaleTimeSeriesGAN,
        ('dropout', 'time_series', 'BVP', False): BayesianBVPTimeSeriesGAN,
        ('dropout', 'features', 'EDA', False): BayesianFeatureGAN,
        ('dropout', 'features', 'BVP', False): BayesianFeatureGAN,
        
        # Variational Bayesian models
        ('var-bayes', 'time_series', 'EDA', True): BayesEDAMultiScaleTimeSeriesGAN,
        ('var-bayes', 'time_series', 'EDA', False): BayesEDATimeSeriesGAN,
        ('var-bayes', 'time_series', 'BVP', True): BayesBVPMultiScaleTimeSeriesGAN,
        ('var-bayes', 'time_series', 'BVP', False): BayesBVPTimeSeriesGAN,
        ('var-bayes', 'features', 'EDA', False): BayesFeatureGAN,
        ('var-bayes', 'features', 'BVP', False): BayesFeatureGAN,
    }

    # Check if the specific uncertainty method exists in the relevant imported models
    for model_cls in [BayesianEDAMultiScaleTimeSeriesGAN, BayesianFeatureGAN, 
                      BayesEDAMultiScaleTimeSeriesGAN, BayesFeatureGAN]:
        if not hasattr(model_cls, 'generate_with_uncertainty_samples'):
            logging.warning(f"generate_with_uncertainty_samples method not found in {model_cls.__name__}.")

except ImportError as e:
    logging.error(f"Failed to import model classes from 'src.models'. "
                  f"Ensure the 'src' directory is in your Python path and contains the necessary model files/submodules. Error: {e}")
    sys.exit(1)
except AttributeError as e:
     logging.error(f"Attribute error during model import check, possibly model structure changed: {e}")
     sys.exit(1)


# --- Constants ---
CONDITION_MAP = {1: "Baseline", 2: "Stress", 3: "Amusement"}
DEFAULT_LATENT_DIM = 100
DEFAULT_HIDDEN_DIM = 256
DEFAULT_NUM_LAYERS = 2
DEFAULT_NUM_CLASSES = len(CONDITION_MAP)
DEFAULT_SEQ_LENGTH = 30
DEFAULT_DROPOUT_RATE = 0.2
HRV_FEATURES = ['RMSSD', 'SDNN', 'LF', 'HF', 'LF/HF']
EDA_FEATURES = ['Mean EDA', 'Median EDA', 'SCR Count']

# Define prior types and their default parameters
PRIOR_TYPES = {
    'gaussian': {'sigma': 1.0},
    'laplace': {'b': 1.0},
    'scaled-mixture-gaussian': {'sigma1': 0.5, 'sigma2': 2.0, 'pi': 0.5}
}

# --- Helper Functions ---

def infer_parameters_from_dir(model_dir_path: str) -> Dict[str, Any]:
    """
    Infers model parameters (type, signal, bayesian, multiscale) from the directory name.

    Args:
        model_dir_path (str): The path to the model directory.

    Returns:
        Dict[str, Any]: A dictionary containing inferred parameters:
                        'model_type', 'signal_type', 'use_bayesian',
                        'approach_name', 'use_multiscale', 'prior_type', 'prior_params'.

    Raises:
        ValueError: If the directory name does not match the expected pattern
                    or contains ambiguous information.
    """
    if not os.path.isdir(model_dir_path):
        raise ValueError(f"Provided model directory path is not a valid directory: {model_dir_path}")

    dir_name = os.path.basename(os.path.normpath(model_dir_path)) # Get the last part of the path
    logging.info(f"Inferring parameters from directory name: {dir_name}")

    parts = dir_name.lower().split('-')

    if len(parts) < 3:
        raise ValueError(f"Directory name '{dir_name}' has too few parts to infer parameters. Expected format like '<approach>-<version>-<signal/features>[-extra]'.")

    # Initialize variables with defaults
    prior_type = 'gaussian'  # Default prior type
    prior_params = PRIOR_TYPES['gaussian']
    
    # Infer approach name and Bayesian status
    approach_name = parts[0]
    
    if approach_name == 'baseline':
        use_bayesian = False
        approach_name = 'baseline'
    elif approach_name == 'dropout' or 'dropout' in approach_name:
        use_bayesian = True
        approach_name = 'dropout'
    elif approach_name == 'var-bayes' or 'var' in approach_name:
        use_bayesian = True
        approach_name = 'var-bayes'
        
        # For var-bayes, look for prior type in the directory name
        for prior in PRIOR_TYPES.keys():
            if prior in dir_name.lower():
                prior_type = prior
                prior_params = PRIOR_TYPES[prior]
                logging.info(f"Inferred prior type: {prior_type} with parameters: {prior_params}")
                break
    elif 'bayes' in approach_name:
        # Generic Bayesian model - default to var-bayes unless explicitly stated
        use_bayesian = True
        approach_name = 'var-bayes'
        logging.warning(f"Found 'bayes' in approach name but no specific type. Defaulting to 'var-bayes'.")
    else:
        # Defaulting to non-Bayesian if approach is unknown, with a warning
        logging.warning(f"Unknown approach '{approach_name}' in directory name. Assuming non-Bayesian baseline.")
        use_bayesian = False
        approach_name = 'baseline'

    # For var-bayes models, the directory structure might be approach-version-signal-prior
    # Need a more flexible parsing approach
    model_type = None
    signal_type = None
    
    # Check each part for signal/model type identifiers
    for part in parts:
        if part in ['eda', 'bvp']:
            model_type = 'time_series'
            signal_type = part.upper()
            logging.info(f"Inferred model_type: time_series, signal_type: {signal_type}")
            break
        elif part == 'features':
            model_type = 'features'
            signal_type = 'EDA'  # Default signal type for features
            logging.info("Inferred model_type: features")
            break
    
    # If we couldn't determine the model_type and signal_type, raise an error
    if model_type is None or signal_type is None:
        raise ValueError(f"Cannot determine model/signal type from directory name: {dir_name}. Expected 'eda', 'bvp', or 'features'.")

    # Infer Multiscale status (Default True unless 'not_multiscale' is present)
    use_multiscale = 'not_multiscale' not in dir_name.lower()
    if model_type == 'features':
         use_multiscale = False # Multiscale doesn't apply to feature models
    logging.info(f"Inferred use_multiscale: {use_multiscale}")
    logging.info(f"Inferred use_bayesian: {use_bayesian}")
    logging.info(f"Inferred approach_name: {approach_name}")

    return {
        'model_type': model_type,
        'signal_type': signal_type,
        'use_bayesian': use_bayesian,
        'use_multiscale': use_multiscale,
        'approach_name': approach_name, # Store for output path structuring
        'prior_type': prior_type,
        'prior_params': prior_params
    }

def find_best_checkpoint(model_dir: str) -> str:
    """
    Finds the checkpoint file starting with 'best_' in the given directory.

    Args:
        model_dir (str): The directory to search within.

    Returns:
        str: The full path to the best checkpoint file.

    Raises:
        FileNotFoundError: If no file starting with 'best_' and ending with '.pt'
                           or '.pth' is found, or if multiple are found.
    """
    search_patterns = [os.path.join(model_dir, 'best_*.pt'), os.path.join(model_dir, 'best_*.pth')]
    best_files = []
    for pattern in search_patterns:
        found = glob.glob(pattern)
        if found:
            best_files.extend(found)

    if not best_files:
        raise FileNotFoundError(f"No 'best_*.pt' or 'best_*.pth' checkpoint file found in directory: {model_dir}")
    if len(best_files) > 1:
        logging.warning(f"Multiple 'best_*.pt/.pth' files found in {model_dir}: {best_files}. Using the first one: {best_files[0]}")
        # Alternatively, raise an error:
        # raise FileNotFoundError(f"Multiple 'best_*.pt/.pth' files found in {model_dir}: {best_files}. Please ensure only one exists.")

    logging.info(f"Found best checkpoint file: {best_files[0]}")
    return best_files[0]


def load_config_from_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """
    Attempts to load configuration from a companion config file or the checkpoint itself.
    Looks for 'config.json' in the same directory as the checkpoint.

    Args:
        checkpoint_path (str): Path to the model checkpoint.

    Returns:
        dict: Configuration dictionary. Returns default values if config is not found.

    Raises:
        FileNotFoundError: If the checkpoint file itself does not exist.
    """
    if not os.path.exists(checkpoint_path):
        # This check might be redundant if find_best_checkpoint is called first, but good practice.
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint_dir = os.path.dirname(checkpoint_path) # Directory containing the checkpoint
    config_path = os.path.join(checkpoint_dir, 'config.json')

    if os.path.exists(config_path):
        logging.info(f"Loading configuration from companion file: {config_path}")
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from {config_path}: {e}")
        except IOError as e:
            logging.error(f"Error reading config file {config_path}: {e}")
        # Fall through if errors occur

    # If no config file or error reading it, check the checkpoint itself
    logging.info("Config file not found or failed to load. Checking checkpoint file for embedded config.")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            logging.info("Loading configuration embedded in the checkpoint file.")
            if isinstance(checkpoint['config'], dict):
                 return checkpoint['config']
            else:
                 logging.warning("Embedded 'config' in checkpoint is not a dictionary. Using defaults.")
        else:
            logging.warning("No 'config.json' found and no 'config' key in checkpoint dictionary.")

    except Exception as e:
        logging.error(f"Error loading checkpoint file {checkpoint_path} to read config: {e}")

    # If we can't find config info, return default values
    logging.warning("Using default configuration values as no valid configuration was found.")
    return {
        'latent_dim': DEFAULT_LATENT_DIM,
        'hidden_dim': DEFAULT_HIDDEN_DIM,
        'num_layers': DEFAULT_NUM_LAYERS,
        'num_classes': DEFAULT_NUM_CLASSES,
        'seq_length': DEFAULT_SEQ_LENGTH,
        'dropout_rate': DEFAULT_DROPOUT_RATE
    }

def load_model(checkpoint_path: str, model_type: str, use_multiscale: bool,
               signal_type: str, use_bayesian: bool, approach_name: str, 
               prior_type: str, prior_params: Dict):
    """
    Load the appropriate model based on checkpoint path and **inferred** parameters.

    Args:
        checkpoint_path (str): Path to the specific checkpoint file (e.g., best_*.pt).
        model_type (str): Inferred type of model ('time_series' or 'features').
        use_multiscale (bool): Inferred whether to use multi-scale generator models.
        signal_type (str): Inferred type of signal ('EDA' or 'BVP').
        use_bayesian (bool): Inferred whether to use Bayesian models.
        approach_name (str): Inferred approach name ('baseline', 'dropout', 'var-bayes').
        prior_type (str): The prior distribution type for variational Bayesian models.
        prior_params (Dict): Parameters for the prior distribution.

    Returns:
        tuple: (loaded model, configuration dict).

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        ValueError: If the inferred parameters lead to an unknown model class.
        RuntimeError: If loading the state dictionary fails.
        KeyError: If the required model class is not found in MODEL_CLASS_MAP.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load configuration using the specific checkpoint path
    config = load_config_from_checkpoint(checkpoint_path)

    # Extract model parameters from config or use defaults
    latent_dim = config.get('latent_dim', DEFAULT_LATENT_DIM)
    hidden_dim = config.get('hidden_dim', DEFAULT_HIDDEN_DIM)
    num_layers = config.get('num_layers', DEFAULT_NUM_LAYERS)
    num_classes = config.get('num_classes', DEFAULT_NUM_CLASSES)
    seq_length = config.get('seq_length', DEFAULT_SEQ_LENGTH)
    dropout_rate = config.get('dropout_rate', DEFAULT_DROPOUT_RATE)

    # --- Model Instantiation based on inferred parameters ---
    model_key = (approach_name, model_type, signal_type, use_multiscale)
    # Adjust key for feature models where multiscale is not applicable
    if model_type == 'features':
         model_key = (approach_name, model_type, signal_type, False) # Force multiscale=False for lookup

    logging.info(f"Looking for model class with key: {model_key}")

    try:
        model_class = MODEL_CLASS_MAP[model_key]
    except KeyError:
        logging.error(f"No model class found in MODEL_CLASS_MAP for parameters: {model_key}")
        logging.error(f"Available keys: {list(MODEL_CLASS_MAP.keys())}")
        raise KeyError(f"Could not find a model class matching the inferred parameters: {model_key}")


    model_args = {
        'latent_dim': latent_dim,
        'hidden_dim': hidden_dim,
        'num_classes': num_classes,
    }
    if model_type == "time_series":
        # Adjust num_layers for BVP multiscale if needed (as per original logic)
        actual_num_layers = num_layers
        if signal_type == "BVP" and use_multiscale:
             actual_num_layers = num_layers + 1
             logging.info(f"Adjusting num_layers to {actual_num_layers} for BVP MultiScale model.")
        model_args.update({'seq_length': seq_length, 'num_layers': actual_num_layers})
    
    # Add specific parameters based on the approach
    if use_bayesian:
        if approach_name == 'dropout':
            model_args['dropout_rate'] = dropout_rate
        elif approach_name == 'var-bayes':
            # Import PriorType enum for variational Bayesian models
            try:
                from src.models.variational_bayes_gan import PriorType
                # Convert string prior_type to enum value
                if prior_type == 'gaussian':
                    model_args['prior_type'] = PriorType.GAUSSIAN
                elif prior_type == 'laplace':
                    model_args['prior_type'] = PriorType.LAPLACE
                elif prior_type == 'scaled-mixture-gaussian':
                    model_args['prior_type'] = PriorType.SCALED_MIXTURE_GAUSSIAN
                else:
                    logging.warning(f"Unknown prior type '{prior_type}'. Defaulting to GAUSSIAN.")
                    model_args['prior_type'] = PriorType.GAUSSIAN
                
                model_args['prior_params'] = prior_params
                logging.info(f"Using prior type: {prior_type} with parameters: {prior_params}")
            except ImportError:
                logging.warning("Could not import PriorType from variational_bayes_gan. Using default parameters.")


    logging.info(f"Instantiating model: {model_class.__name__} with args: {model_args}")
    try:
        model = model_class(**model_args)
    except Exception as e:
        logging.error(f"Error instantiating model {model_class.__name__}: {e}")
        raise # Re-raise the exception after logging

    # --- Load Checkpoint State ---
    try:
        logging.info(f"Loading checkpoint state from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Handle different checkpoint formats robustly
        state_dict = None
        if isinstance(checkpoint, dict):
            # Prioritize specific keys often used in training scripts
            if 'generator_state_dict' in checkpoint:
                 state_dict = checkpoint['generator_state_dict']
                 logging.info("Loaded state_dict from 'generator_state_dict' key.")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                logging.info("Loaded state_dict from 'state_dict' key.")
            elif 'model_state_dict' in checkpoint: # Another common key
                 state_dict = checkpoint['model_state_dict']
                 logging.info("Loaded state_dict from 'model_state_dict' key.")
            else:
                 logging.warning("Checkpoint lacks common state_dict keys. Attempting to load dictionary directly.")
                 state_dict = checkpoint
        else:
             state_dict = checkpoint
             logging.info("Loaded state_dict directly from checkpoint file (assumed not a dictionary).")

        if state_dict is None:
             raise ValueError("Could not extract a valid state dictionary from the checkpoint.")

        # Load the state dictionary
        model.load_state_dict(state_dict)
        logging.info("Successfully loaded model state dictionary.")

    except FileNotFoundError:
        # Should have been caught earlier by find_best_checkpoint
        logging.error(f"Checkpoint file disappeared unexpectedly: {checkpoint_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading state dict from checkpoint {checkpoint_path}: {e}")
        raise RuntimeError(f"Failed to load model state from checkpoint: {e}") from e

    model = model.to(device)
    model.eval() # Set model to evaluation mode

    return model, config

# --- Physiological Signal Generation and Validation ---
# (validate_and_fix_signal, validate_and_fix_bvp_signal, generate_physiological_signal functions remain unchanged)
# ... (Keep the existing functions here) ...
def validate_and_fix_signal(sample_np: np.ndarray) -> np.ndarray:
    """
    Validates and fixes generated EDA signals for physiological plausibility.
    Handles potential numerical issues.

    Args:
        sample_np (np.ndarray): The raw generated signal.

    Returns:
        np.ndarray: The validated and potentially fixed signal.
    """
    if not isinstance(sample_np, np.ndarray) or sample_np.ndim != 1 or len(sample_np) < 3:
        logging.warning(f"Invalid input signal shape {sample_np.shape if isinstance(sample_np, np.ndarray) else type(sample_np)} for validation/fixing. Returning as is.")
        return sample_np

    # Replace NaNs or Infs if they somehow occur
    if np.isnan(sample_np).any() or np.isinf(sample_np).any():
        logging.warning("NaN or Inf detected in generated signal. Replacing with zeros.")
        sample_np = np.nan_to_num(sample_np, nan=0.0, posinf=0.0, neginf=0.0)

    # Smooth out extreme jumps
    diff = np.abs(np.diff(sample_np))
    # Use nanstd to be safe, handle case where diff is empty or all same
    std_diff = np.nanstd(diff) if len(diff) > 0 else 0
    threshold = std_diff * 3 if std_diff > 1e-6 else 0.1 # Avoid threshold being zero

    for i in range(1, len(sample_np)):
        # Check index bounds for diff
        if i-1 < len(diff) and diff[i-1] > threshold:
            # Apply smoothing cautiously
            sample_np[i] = sample_np[i-1] * 0.9 + sample_np[i] * 0.1

    # Ensure no flat lines longer than 2 steps (adjust threshold based on signal scale)
    signal_std = np.std(sample_np)
    flat_threshold = max(signal_std * 0.01, 1e-5) # Relative threshold
    for i in range(2, len(sample_np)):
        if abs(sample_np[i] - sample_np[i-1]) < flat_threshold and \
           abs(sample_np[i-1] - sample_np[i-2]) < flat_threshold:
            # Add small noise relative to signal std dev
            noise_std = max(signal_std * 0.005, 1e-6)
            sample_np[i] += np.random.normal(0, noise_std)

    return sample_np

def validate_and_fix_bvp_signal(sample_np: np.ndarray) -> np.ndarray:
    """
    Validates and fixes generated BVP signals for physiological plausibility.
    Handles potential numerical issues and enhances oscillations if needed.

    Args:
        sample_np (np.ndarray): The raw generated signal.

    Returns:
        np.ndarray: The validated and potentially fixed signal.
    """
    if not isinstance(sample_np, np.ndarray) or sample_np.ndim != 1 or len(sample_np) < 10: # Need more points for BVP analysis
        logging.warning(f"Invalid input signal shape {sample_np.shape if isinstance(sample_np, np.ndarray) else type(sample_np)} for BVP validation/fixing. Returning as is.")
        return sample_np

    # Replace NaNs or Infs
    if np.isnan(sample_np).any() or np.isinf(sample_np).any():
        logging.warning("NaN or Inf detected in generated BVP signal. Replacing with zeros.")
        sample_np = np.nan_to_num(sample_np, nan=0.0, posinf=0.0, neginf=0.0)

    # Verify there's enough oscillation (not too flat)
    std_val = np.std(sample_np)
    if std_val < 0.1: # Increased threshold slightly
        logging.warning("BVP signal seems too flat (std < 0.1). Attempting to increase amplitude.")
        mean_val = np.mean(sample_np)
        # Avoid division by zero or very small numbers
        scale_factor = 0.5 / max(std_val, 0.01)
        sample_np = (sample_np - mean_val) * scale_factor + mean_val

    # Smooth out extreme jumps
    diff = np.abs(np.diff(sample_np))
    std_diff = np.nanstd(diff) if len(diff) > 0 else 0
    threshold = std_diff * 3 if std_diff > 1e-6 else 0.5 # Higher base threshold for BVP

    for i in range(1, len(sample_np)):
         # Check index bounds for diff
        if i-1 < len(diff) and diff[i-1] > threshold:
             # More aggressive smoothing for BVP jumps
            sample_np[i] = sample_np[i-1] * 0.7 + sample_np[i] * 0.3

    # Enhance oscillatory pattern if needed (using SciPy)
    try:
        # Defer SciPy import until needed
        from scipy import signal as sp_signal

        # Ensure sufficient length for Welch
        nperseg = min(len(sample_np), 10) # Needs at least 10 points for reasonable PSD
        if len(sample_np) >= nperseg:
            freqs, psd = sp_signal.welch(sample_np, fs=1.0, nperseg=nperseg, scaling='density') # Use density scaling

            if len(psd) > 1:
                # Find peak frequency excluding DC (index 0)
                valid_psd = psd[1:]
                if len(valid_psd) > 0:
                    peak_idx_offset = np.argmax(valid_psd)
                    peak_idx = peak_idx_offset + 1 # Actual index in freqs/psd
                    if peak_idx < len(freqs): # Ensure index is valid
                        peak_freq = freqs[peak_idx]
                        # Target frequency range (approx 0.23-0.27 Hz for 7-8 cycles in 30s -> adjusted for fs=1.0)
                        target_low, target_high = 0.19, 0.29 # Corresponds to ~5.7 to 8.7 bpm if fs=1Hz, adjust if fs is different

                        # If dominant frequency is outside target range, try to enhance target range
                        if peak_freq < target_low or peak_freq > target_high:
                            logging.info(f"Dominant frequency {peak_freq:.2f}Hz outside target [{target_low}-{target_high}]Hz. Applying bandpass filter.")
                            # Ensure filter order is less than signal length
                            filter_order = min(2, (len(sample_np) // 2) - 1) # Max order for stability
                            if filter_order > 0:
                                # Use try-except for filter design robustness
                                try:
                                    # Define Nyquist frequency (fs/2)
                                    nyquist = 0.5 # Since fs=1.0
                                    # Normalize frequencies
                                    low_norm = target_low / nyquist
                                    high_norm = target_high / nyquist
                                    # Ensure frequencies are valid (0 < f < 1)
                                    if 0 < low_norm < high_norm < 1.0:
                                        b, a = sp_signal.butter(filter_order, [low_norm, high_norm], btype='bandpass')
                                        # Use filtfilt for zero-phase filtering, check for stability issues
                                        oscillatory_component = sp_signal.filtfilt(b, a, sample_np)
                                        # Combine cautiously
                                        sample_np = sample_np * 0.7 + oscillatory_component * 0.3
                                    else:
                                        logging.warning(f"Invalid normalized frequencies [{low_norm:.2f}, {high_norm:.2f}] for bandpass filter.")

                                except ValueError as filter_err:
                                    logging.warning(f"Could not apply bandpass filter (ValueError): {filter_err}")
                            else:
                                logging.warning("Signal too short to apply stable bandpass filter.")
                else:
                    logging.warning("PSD calculation resulted in insufficient data points for peak finding.")
        else:
            logging.warning(f"Signal length {len(sample_np)} too short for Welch analysis with nperseg={nperseg}.")


    except ImportError:
        logging.warning("SciPy not found. Cannot perform frequency analysis for BVP validation.")
    except Exception as e:
        logging.warning(f"Error during BVP frequency analysis/filtering: {e}")

    return sample_np

def generate_physiological_signal(model: torch.nn.Module, z_base: Optional[torch.Tensor],
                                  condition: torch.Tensor, signal_type: str,
                                  num_attempts: int) -> np.ndarray:
    """
    Generates the best physiological signal out of multiple attempts using the specified model.

    Args:
        model: The GAN model (generator part).
        z_base: A base latent vector (optional, if None, random vectors are used).
        condition: The condition label tensor (batch size should be 1).
        signal_type: The type of signal ('EDA' or 'BVP').
        num_attempts: Number of generation attempts.

    Returns:
        numpy.ndarray: The best generated signal found.

    Raises:
        ValueError: If generation fails repeatedly or input is invalid.
    """
    device = next(model.parameters()).device
    if condition.size(0) != 1:
         raise ValueError("generate_physiological_signal expects batch size of 1 for condition.")
    latent_dim = getattr(model, 'latent_dim', DEFAULT_LATENT_DIM) # Get latent_dim safely

    best_sample_np = None
    min_zeros = float('inf') # For EDA
    max_oscillations = 0 # For BVP
    best_frequency_match = float('inf') # For BVP

    # Use more attempts for BVP as it's harder to get good oscillations
    actual_attempts = max(num_attempts * 2 if signal_type == "BVP" else num_attempts, 1)
    logging.info(f"Attempting to generate best {signal_type} signal from {actual_attempts} attempts...")

    for attempt in range(actual_attempts):
        # Generate a new latent vector for each attempt if z_base is None or after first try
        z_attempt = z_base if attempt == 0 and z_base is not None else torch.randn(1, latent_dim, device=device)

        try:
            with torch.no_grad(): # Ensure no gradients are computed
                # Assume model.generate exists and returns a tensor
                sample = model.generate(z_attempt, condition)

            # --- Process Generated Sample ---
            # Detach, move to CPU, convert to NumPy
            sample_np = sample.detach().cpu().numpy()

            # Handle potential variations in output shape [batch, seq, feat] or [batch, feat, seq] etc.
            # Aim for a 1D array [seq_len]
            if sample_np.shape[0] == 1: # Remove batch dim if it exists
                 sample_np = sample_np.squeeze(0)
            # If still > 1D, try to squeeze singleton dimensions
            if sample_np.ndim > 1:
                 try:
                      sample_np = sample_np.squeeze() # Squeeze all singleton dimensions
                 except ValueError: # Cannot squeeze if multiple non-singleton dimensions exist
                      logging.warning(f"Generated sample has ambiguous shape {sample.shape} -> {sample_np.shape} after initial squeeze. Trying to select first feature.")
                      # Attempt common formats like [seq, feat] or [feat, seq]
                      if sample_np.shape[0] > 1 and sample_np.shape[1] == 1: # [seq, 1]
                           sample_np = sample_np[:, 0]
                      elif sample_np.shape[0] == 1 and sample_np.shape[1] > 1: # [1, seq]
                            sample_np = sample_np[0, :]
                      # Add more heuristics if other shapes are common, e.g., [seq, feat, other]
                      else:
                           # Fallback: take the first slice along the last dimension if shape is complex
                           sample_np = sample_np[..., 0]


            # Ensure it's 1D after processing
            if sample_np.ndim != 1:
                 # If still not 1D, flatten as a last resort, though this might lose structure
                 logging.warning(f"Generated sample still not 1D (shape: {sample_np.shape}). Flattening array.")
                 sample_np = sample_np.flatten()
                 # Could raise error here instead if flattening is undesirable
                 # raise ValueError(f"Could not reduce generated sample to 1D array. Original shape: {sample.shape}, processed shape: {sample_np.shape}")


            # --- Basic Validation Metrics ---
            zero_count = np.sum(np.abs(sample_np) < 0.01) # Count near-zero values

            # --- BVP Specific Evaluation ---
            if signal_type == "BVP":
                # Need SciPy for frequency analysis
                oscillation_count = 0
                freq_match = float('inf')
                try:
                    # Defer import
                    from scipy import signal as sp_signal
                    # Zero-crossings around mean
                    mean_val = np.mean(sample_np)
                    oscillation_count = np.sum(np.diff(np.signbit(sample_np - mean_val)))

                    # Frequency match
                    nperseg = min(len(sample_np), 10)
                    if len(sample_np) >= nperseg:
                        freqs, psd = sp_signal.welch(sample_np, fs=1.0, nperseg=nperseg, scaling='density')
                        if len(psd) > 1:
                            valid_psd = psd[1:]
                            if len(valid_psd) > 0:
                                peak_idx_offset = np.argmax(valid_psd)
                                peak_idx = peak_idx_offset + 1
                                if peak_idx < len(freqs):
                                    peak_freq = freqs[peak_idx]
                                    target_low, target_high = 0.19, 0.29
                                    # Calculate distance to target range (0 if inside)
                                    if target_low <= peak_freq <= target_high:
                                         freq_match = 0
                                    else:
                                         freq_match = min(abs(peak_freq - target_low), abs(peak_freq - target_high))

                except ImportError:
                     logging.warning("SciPy not found. BVP quality check based only on zero-crossings.")
                except Exception as e:
                     logging.warning(f"Error during BVP quality check (attempt {attempt+1}): {e}")


                # Selection criteria for BVP: Prioritize frequency match (lower is better), then oscillations (higher is better)
                is_better = False
                if best_sample_np is None:
                    is_better = True
                # Prefer samples within the target frequency range (freq_match == 0)
                elif freq_match == 0 and best_frequency_match > 0:
                     is_better = True # New sample is in range, old wasn't
                elif freq_match == 0 and best_frequency_match == 0:
                     # Both in range, prefer more oscillations
                     if oscillation_count > max_oscillations:
                          is_better = True
                # If neither is in range, prefer closer frequency match
                elif freq_match < best_frequency_match:
                     is_better = True
                # If frequency match is similar, prefer more oscillations
                elif np.isclose(freq_match, best_frequency_match) and oscillation_count > max_oscillations:
                     is_better = True

                if is_better:
                    best_frequency_match = freq_match
                    max_oscillations = oscillation_count
                    best_sample_np = sample_np.copy() # Store a copy

            # --- EDA Specific Evaluation ---
            else: # EDA
                # Selection criteria for EDA: Minimize near-zero values
                if best_sample_np is None or zero_count < min_zeros:
                    min_zeros = zero_count
                    best_sample_np = sample_np.copy() # Store a copy

        except AttributeError as e:
             # Catch if model doesn't have 'generate' method
             logging.error(f"Model object does not have a 'generate' method. Error: {e}")
             raise # Re-raise as this is a fundamental issue
        except Exception as e:
            logging.error(f"Error during signal generation attempt {attempt+1}: {e}")
            # Continue to next attempt if possible

    # Check if any sample was successfully generated
    if best_sample_np is None:
        raise ValueError(f"Failed to generate any valid signal after {actual_attempts} attempts.")

    # Apply final validation and fixing to the best sample found
    logging.info(f"Selected best sample (Quality - BVP FreqMatch: {best_frequency_match:.2f}, Osc: {max_oscillations} | EDA Zeros: {min_zeros}). Applying final validation/fixing for {signal_type}.")
    if signal_type == "EDA":
        final_sample = validate_and_fix_signal(best_sample_np)
    else: # BVP
        final_sample = validate_and_fix_bvp_signal(best_sample_np)

    return final_sample

# --- Plotting and Saving Functions ---
# (plot_physiological_signal, plot_physiological_signal_with_uncertainty,
#  plot_feature_distributions, save_feature_data, print_feature_table, print_feature_stats
#  functions remain unchanged)
# ... (Keep the existing functions here) ...
def plot_physiological_signal(sample_np: np.ndarray, signal_type: str,
                              condition_name: str, output_path: str):
    """Creates and saves a plot of a single physiological signal."""
    if not isinstance(sample_np, np.ndarray) or sample_np.ndim != 1:
         logging.error(f"Cannot plot signal: Invalid data type or shape ({type(sample_np)}, {sample_np.shape if isinstance(sample_np, np.ndarray) else 'N/A'}). Expected 1D NumPy array.")
         return
    try:
        plt.figure(figsize=(12, 4))
        plt.style.use('seaborn-v0_8-whitegrid')
        line_color = '#ff7f0e' if signal_type == "EDA" else '#1f77b4' # Orange for EDA, Blue for BVP

        plt.plot(sample_np, '-', color=line_color, label=signal_type,
                 markersize=4, linewidth=1.5) # Removed '-o' for cleaner look on dense data

        plt.grid(True, linestyle='-', alpha=0.2)
        plt.title(f"Generated {signal_type} Signal - {condition_name}", pad=10)
        plt.xlabel("Time (samples)") # Changed label slightly
        plt.ylabel("Amplitude (z-score normalized)")

        # Add dynamic Y-axis limits for better visualization
        y_min, y_max = np.min(sample_np), np.max(sample_np)
        y_range = y_max - y_min
        y_margin = y_range * 0.1 if y_range > 1e-6 else 0.1 # Add margin, handle flat signals
        plt.ylim(y_min - y_margin, y_max + y_margin)

        plt.legend(loc='best') # Use 'best' location
        plt.tight_layout()
        # Ensure parent directory exists before saving
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close() # Close the figure to free memory
        logging.info(f"Saved plot: {output_path}")
    except Exception as e:
        logging.error(f"Error plotting/saving signal to {output_path}: {e}")

def plot_physiological_signal_with_uncertainty(samples: List[np.ndarray], signal_type: str,
                                               condition_name: str, output_path: str):
    """Creates and saves a plot showing multiple signal samples for uncertainty."""
    if not samples or not all(isinstance(s, np.ndarray) and s.ndim == 1 for s in samples):
         logging.warning(f"No valid samples provided for uncertainty plot (found {len(samples)} items). Skipping.")
         return
    # Ensure all samples have the same length for mean/std calculation
    try:
        seq_len = len(samples[0])
        if not all(len(s) == seq_len for s in samples):
            logging.warning("Samples for uncertainty plot have different lengths. Skipping plot.")
            return
    except IndexError: # Handle case where samples list might be empty after filtering
        logging.warning("Samples list is empty for uncertainty plot. Skipping.")
        return


    try:
        plt.figure(figsize=(12, 4))
        plt.style.use('seaborn-v0_8-whitegrid')
        base_color = '#ff7f0e' if signal_type == "EDA" else '#1f77b4'

        # Calculate mean and std dev safely
        plot_bands = False
        if len(samples) > 1: # Need more than 1 sample for std dev
            try:
                 mean_signal = np.mean(samples, axis=0)
                 std_signal = np.std(samples, axis=0)
                 plot_bands = True
            except Exception as calc_e:
                 logging.warning(f"Could not calculate mean/std for uncertainty plot: {calc_e}. Plotting individual lines only.")
        elif len(samples) == 1:
             mean_signal = samples[0] # If only one sample, mean is the sample itself
             logging.info("Only one sample provided for uncertainty plot. Plotting single line.")
        else: # Should have been caught earlier, but as safeguard
             return


        # Plot individual samples with transparency first (behind mean/bands)
        for i, sample in enumerate(samples):
            plt.plot(sample, '-', color=base_color, alpha=0.1, linewidth=1) # Make very transparent

        # Plot mean and std deviation band if calculated
        if plot_bands:
             plt.plot(mean_signal, '-', color=base_color, alpha=0.9, linewidth=2.0, label=f'{signal_type} (Mean)')
             plt.fill_between(range(len(mean_signal)), mean_signal - std_signal, mean_signal + std_signal,
                              color=base_color, alpha=0.2, label='Mean ± Std Dev') # More descriptive label
        elif len(samples) == 1:
             plt.plot(mean_signal, '-', color=base_color, alpha=0.9, linewidth=2.0, label=f'{signal_type} (Single Sample)')


        plt.grid(True, linestyle='-', alpha=0.2)
        plt.title(f"Generated {signal_type} Signal (with Uncertainty/Variability) - {condition_name}", pad=10)
        plt.xlabel("Time (samples)")
        plt.ylabel("Amplitude (z-score normalized)")

        # Adjust Y limits based on all data
        all_values = np.concatenate([s.flatten() for s in samples])
        y_min, y_max = np.min(all_values), np.max(all_values)
        y_range = y_max - y_min
        y_margin = y_range * 0.1 if y_range > 1e-6 else 0.1
        plt.ylim(y_min - y_margin, y_max + y_margin)

        plt.legend(loc='best')
        plt.tight_layout()
        # Ensure parent directory exists before saving
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved uncertainty plot: {output_path}")
    except Exception as e:
        logging.error(f"Error plotting/saving uncertainty signal to {output_path}: {e}")

def plot_feature_distributions(feature_samples: List[np.ndarray], feature_names: List[str],
                               title_suffix: str, output_path: str):
    """Creates box plots for feature distributions."""
    if not feature_samples or not all(isinstance(s, np.ndarray) for s in feature_samples):
        logging.warning("No valid feature samples provided for distribution plot. Skipping.")
        return
    if not feature_names:
         logging.warning("No feature names provided for distribution plot. Skipping.")
         return

    # Check consistency of feature count in samples
    num_features = len(feature_names)
    valid_samples = [s for s in feature_samples if len(s) == num_features]
    if len(valid_samples) < len(feature_samples):
         logging.warning(f"Removed {len(feature_samples) - len(valid_samples)} samples with incorrect feature count for distribution plot.")
    if not valid_samples:
         logging.warning("No samples with correct feature count remain. Skipping distribution plot.")
         return

    try:
        plt.figure(figsize=(max(8, num_features * 1.2), 6)) # Dynamic width
        plt.style.use('seaborn-v0_8-whitegrid')

        # Data needs to be transposed for boxplot: list where each element contains all values for one feature
        data_for_plot = [[] for _ in feature_names]
        for sample in valid_samples:
             for i, val in enumerate(sample):
                 data_for_plot[i].append(val)

        plt.boxplot(data_for_plot, labels=feature_names, showfliers=True) # Show outliers by default

        plt.title(f"Generated Feature Distributions - {title_suffix}")
        plt.xticks(rotation=45, ha='right') # Improve label readability
        plt.ylabel("Value (z-score normalized)") # Assume normalized
        plt.grid(True, linestyle='--', alpha=0.3) # Lighter grid
        plt.tight_layout()
        # Ensure parent directory exists before saving
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved feature distribution plot: {output_path}")
    except Exception as e:
        logging.error(f"Error plotting/saving feature distributions to {output_path}: {e}")

def save_feature_data(data: Dict, output_path: str):
    """Saves feature data (single sample or stats) to a JSON file."""
    try:
        # Ensure numpy types are converted to standard Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            # Use abstract base classes for broader compatibility (NumPy 1.x & 2.x)
            elif isinstance(obj, np.integer): # Catches all int types (int8, int16, int32, int64, uint8, ...)
                return int(obj)
            elif isinstance(obj, np.floating): # Catches all float types (float16, float32, float64, ...)
                return float(obj)
            elif isinstance(obj, np.complexfloating): # Catches complex types (complex64, complex128)
                return {'real': obj.real, 'imag': obj.imag}
            elif isinstance(obj, (np.bool_, bool)): # Check for np.bool_ and standard bool
                return bool(obj)
            elif isinstance(obj, np.void): # Handle void types (often structured arrays)
                return None # Or convert appropriately if structure is known
            # Recurse into dictionaries and lists
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            # Handle standard Python types that are already JSON serializable
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            # Log a warning for unhandled types
            else:
                logging.warning(f"Unhandled type for JSON conversion: {type(obj)}. Converting to string.")
                return str(obj)


        serializable_data = convert_numpy(data)

        # Ensure parent directory exists before saving
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(serializable_data, f, indent=4)
        logging.info(f"Saved feature data to: {output_path}")
    except IOError as e:
        logging.error(f"Error writing feature data to JSON file {output_path}: {e}")
    except TypeError as e:
         # This might still occur if convert_numpy misses a type or recursion fails
         logging.error(f"Error serializing feature data to JSON: {e}. Processed data type: {type(serializable_data)}")


def print_feature_table(title: str, values: np.ndarray, names: List[str]):
    """Prints a simple table of feature values."""
    if values is None or values.size == 0:
         print(f"\n{title}: No data available.")
         return
    print(f"\n{title}:")
    print("-" * 30)
    print(f"{'Feature':<15} {'Value':>10}")
    print("-" * 30)
    for i, name in enumerate(names):
        if i < len(values):
            print(f"{name:<15} {values[i]:>10.4f}")
        else:
             # This case should ideally not happen if lengths match
             print(f"{name:<15} {'MISSING':>10}")
    print("-" * 30)

def print_feature_stats(title: str, stats: Dict, names: List[str]):
     """Prints a table of feature statistics (mean, std)."""
     if not stats:
          print(f"\n{title}: No stats available.")
          return
     print(f"\n{title}:")
     print("-" * 40)
     print(f"{'Feature':<15} {'Mean':>10} {'Std Dev':>10}")
     print("-" * 40)
     for name in names:
          feature_stats = stats.get(name)
          if feature_stats and isinstance(feature_stats, dict):
               mean = feature_stats.get('mean', float('nan'))
               std = feature_stats.get('std', float('nan'))
               print(f"{name:<15} {mean:>10.4f} {std:>10.4f}")
          else:
               # Handle case where feature might be missing from stats dict
               print(f"{name:<15} {'N/A':>10} {'N/A':>10}")
     print("-" * 40)


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(
        description="Generate physiological time series or features using a trained GAN model, "
                    "inferring parameters from the model directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # --- Argument Parsing (Simplified) ---
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to the directory containing the 'best_*.pt' model checkpoint and 'config.json'. "
                             "Parameters (type, signal, bayesian, multiscale) will be inferred from this directory's name.")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of independent samples (plots/tables) to generate.")
    parser.add_argument("--condition", type=int, default=1, choices=CONDITION_MAP.keys(),
                        help=f"Condition label: {', '.join([f'{k}={v}' for k, v in CONDITION_MAP.items()])}.")
    parser.add_argument("--output_dir", type=str, default="generated_outputs",
                        help="Base directory to save the generated outputs.")
    # Removed: --model_type, --signal_type, --checkpoint, --use_multiscale, --use_bayesian
    parser.add_argument("--attempts", type=int, default=5,
                        help="Number of generation attempts per sample to find the 'best' signal (time_series only).")
    parser.add_argument("--plot_uncertainty", action="store_true",
                        help="Generate multiple outputs from the *same* latent vector (if Bayesian with uncertainty method) "
                             "or multiple latent vectors (otherwise) to visualize uncertainty/variability.")
    parser.add_argument("--uncertainty_samples", type=int, default=10,
                        help="Number of outputs for uncertainty visualization (used with --plot_uncertainty).")

    args = parser.parse_args()

    # --- Parameter Inference and Validation ---
    try:
        if not os.path.isdir(args.model_dir):
             raise FileNotFoundError(f"Model directory not found: {args.model_dir}")

        inferred_params = infer_parameters_from_dir(args.model_dir)
        model_type = inferred_params['model_type']
        signal_type = inferred_params['signal_type']
        use_bayesian = inferred_params['use_bayesian']
        use_multiscale = inferred_params['use_multiscale']
        approach_name = inferred_params['approach_name']
        prior_type = inferred_params.get('prior_type', 'gaussian')
        prior_params = inferred_params.get('prior_params', PRIOR_TYPES['gaussian'])

        best_checkpoint_path = find_best_checkpoint(args.model_dir)

    except (ValueError, FileNotFoundError) as e:
        logging.error(f"Error during setup: {e}")
        sys.exit(1)

    # --- Input Validation (Remaining Args) ---
    if args.num_samples <= 0:
        logging.error("--num_samples must be a positive integer.")
        sys.exit(1)
    if args.uncertainty_samples <= 0:
        logging.error("--uncertainty_samples must be a positive integer.")
        sys.exit(1)
    if args.attempts <= 0:
         logging.warning("--attempts should be positive; setting to 1.")
         args.attempts = 1
    # Add warning if plot_uncertainty is used without a Bayesian model being inferred
    if args.plot_uncertainty and not use_bayesian:
         logging.warning("Using --plot_uncertainty with a non-Bayesian model inferred from directory name. "
                         "This will show variability across different latent vectors, not model uncertainty.")

    # --- Load Model ---
    try:
        model, config = load_model(
            best_checkpoint_path, # Use the found best checkpoint
            model_type,
            use_multiscale,
            signal_type,
            use_bayesian,
            approach_name,  # Pass the approach_name to load_model
            prior_type,
            prior_params
        )
    except (FileNotFoundError, ValueError, RuntimeError, TypeError, AttributeError, ImportError, KeyError) as e:
        logging.error(f"Failed to load model: {e}")
        sys.exit(1)

    # --- Setup Output Path ---
    condition_name = CONDITION_MAP.get(args.condition, f"Condition_{args.condition}")
    # Create a structured output subdirectory: base_output_dir / model_dir_name / condition_name
    model_dir_basename = os.path.basename(os.path.normpath(args.model_dir))
    structured_output_dir = os.path.join(args.output_dir, model_dir_basename, condition_name)
    try:
        os.makedirs(structured_output_dir, exist_ok=True)
        logging.info(f"Output will be saved to: {structured_output_dir}")
    except OSError as e:
        logging.error(f"Could not create structured output directory {structured_output_dir}: {e}")
        # Fallback to base output directory? Or exit? Let's fallback with warning.
        logging.warning(f"Falling back to saving in base output directory: {args.output_dir}")
        structured_output_dir = args.output_dir


    # Get necessary parameters
    latent_dim = getattr(model, 'latent_dim', config.get('latent_dim', DEFAULT_LATENT_DIM))
    device = next(model.parameters()).device


    logging.info(f"--- Starting Generation ---")
    logging.info(f"Model Directory: {args.model_dir}")
    logging.info(f"Inferred - Type: {model_type}, Signal: {signal_type}, "
                f"Bayesian: {use_bayesian}, Approach: {approach_name}, "
                f"Multiscale: {use_multiscale}")
    if approach_name == 'var-bayes':
        logging.info(f"Variational Bayes Prior: {prior_type} with parameters {prior_params}")
    logging.info(f"Condition: {args.condition} ({condition_name})")
    logging.info(f"Generating {args.num_samples} sample(s).")
    if args.plot_uncertainty:
         logging.info(f"Uncertainty plotting enabled ({args.uncertainty_samples} outputs per sample).")


    # --- Generation Loop ---
    for i in range(args.num_samples):
        sample_idx = i + 1
        logging.info(f"\n--- Generating Sample {sample_idx}/{args.num_samples} ---")

        # Generate a base latent vector for this sample
        z_base = torch.randn(1, latent_dim, device=device)
        condition_tensor = torch.tensor([args.condition], dtype=torch.long, device=device)

        # --- Time Series Generation ---
        if model_type == "time_series":
            if args.plot_uncertainty:
                samples_for_plot = []
                logging.info(f"Generating {args.uncertainty_samples} outputs for uncertainty plot...")
                try:
                    has_uncertainty_method = use_bayesian and hasattr(model, 'generate_with_uncertainty_samples')
                    if has_uncertainty_method:
                         logging.info("Using model's 'generate_with_uncertainty_samples' method.")
                         samples_list = model.generate_with_uncertainty_samples(
                             z_base, condition_tensor, n_samples=args.uncertainty_samples
                         )
                         if isinstance(samples_list, list):
                             logging.info(f"Received {len(samples_list)} samples from uncertainty method.")
                             for sample_tensor in samples_list:
                                 if isinstance(sample_tensor, torch.Tensor):
                                     sample_np = sample_tensor.detach().cpu().numpy().squeeze()
                                     if sample_np.ndim == 1:
                                         validated_sample = validate_and_fix_signal(sample_np) if signal_type == "EDA" else validate_and_fix_bvp_signal(sample_np)
                                         samples_for_plot.append(validated_sample)
                                     else: logging.warning(f"Skipping uncertainty sample due to unexpected shape: {sample_np.shape}")
                                 else: logging.warning(f"Item in samples list is not a tensor: {type(sample_tensor)}. Skipping.")
                         elif isinstance(samples_list, torch.Tensor):
                              logging.warning("'generate_with_uncertainty_samples' returned Tensor, expected List. Processing as stacked.")
                              if samples_list.size(0) == args.uncertainty_samples:
                                   for s_idx in range(samples_list.size(0)):
                                        sample_np = samples_list[s_idx].detach().cpu().numpy().squeeze()
                                        if sample_np.ndim == 1:
                                             validated_sample = validate_and_fix_signal(sample_np) if signal_type == "EDA" else validate_and_fix_bvp_signal(sample_np)
                                             samples_for_plot.append(validated_sample)
                                        else: logging.warning(f"Skipping uncertainty sample {s_idx} due to shape: {sample_np.shape}")
                              else: logging.error(f"Uncertainty method returned tensor with wrong batch size.")
                         else: logging.error(f"Unexpected return type from uncertainty method: {type(samples_list)}.")
                    else:
                         logging.info("Generating multiple independent samples for variability plot.")
                         for _ in range(args.uncertainty_samples):
                              sample_np = generate_physiological_signal(
                                  model, None, condition_tensor, signal_type, args.attempts
                              )
                              samples_for_plot.append(sample_np)

                    if samples_for_plot:
                        # Use structured output path
                        output_plot_path = os.path.join(structured_output_dir,
                            f"{signal_type}_uncertainty_sample_{sample_idx}.png") # Simplified filename
                        plot_physiological_signal_with_uncertainty(
                            samples_for_plot, signal_type, condition_name, output_plot_path
                        )
                    else: logging.warning(f"No valid samples generated for uncertainty plot {sample_idx}.")
                except Exception as e:
                    logging.error(f"Error generating uncertainty samples for sample {sample_idx}: {e}", exc_info=True)

            else: # Single time series sample
                try:
                    sample_np = generate_physiological_signal(
                        model, z_base, condition_tensor, signal_type, args.attempts
                    )
                    # Use structured output path
                    output_plot_path = os.path.join(structured_output_dir,
                        f"{signal_type}_sample_{sample_idx}.png") # Simplified filename
                    plot_physiological_signal(sample_np, signal_type, condition_name, output_plot_path)
                except (ValueError, RuntimeError) as e:
                    logging.error(f"Failed to generate time series signal for sample {sample_idx}: {e}")


        # --- Feature Generation ---
        elif model_type == "features":
            try:
                has_uncertainty_method = use_bayesian and hasattr(model, 'generate_with_uncertainty_samples')
                if args.plot_uncertainty and has_uncertainty_method:
                    logging.info(f"Generating {args.uncertainty_samples} feature sets for uncertainty analysis...")
                    samples_output = model.generate_with_uncertainty_samples(
                        z_base, condition_tensor, n_samples=args.uncertainty_samples
                    )
                    hrv_samples, eda_samples = [], []
                    if isinstance(samples_output, dict):
                         hrv_tensor_list = samples_output.get('HRV_features', [])
                         eda_tensor_list = samples_output.get('EDA_features', [])
                         if isinstance(hrv_tensor_list, list): hrv_samples = [s.detach().cpu().numpy().squeeze() for s in hrv_tensor_list if isinstance(s, torch.Tensor)]
                         if isinstance(eda_tensor_list, list): eda_samples = [s.detach().cpu().numpy().squeeze() for s in eda_tensor_list if isinstance(s, torch.Tensor)]
                    elif isinstance(samples_output, list):
                         for sample_dict in samples_output:
                              if isinstance(sample_dict, dict):
                                   hrv_tensor = sample_dict.get('HRV_features')
                                   eda_tensor = sample_dict.get('EDA_features')
                                   if isinstance(hrv_tensor, torch.Tensor): hrv_samples.append(hrv_tensor.detach().cpu().numpy().squeeze())
                                   if isinstance(eda_tensor, torch.Tensor): eda_samples.append(eda_tensor.detach().cpu().numpy().squeeze())
                    else: logging.error(f"Unexpected output type from feature uncertainty method: {type(samples_output)}")

                    hrv_stats, eda_stats = {}, {}
                    if hrv_samples:
                        hrv_samples = [s for s in hrv_samples if len(s) == len(HRV_FEATURES)]
                        if hrv_samples:
                            hrv_stats = {f: {'mean': float(np.mean([s[j] for s in hrv_samples])), 'std': float(np.std([s[j] for s in hrv_samples])), 'samples': [float(s[j]) for s in hrv_samples]} for j, f in enumerate(HRV_FEATURES)}
                            output_plot_path_hrv = os.path.join(structured_output_dir, f"feature_dist_hrv_sample_{sample_idx}.png")
                            plot_feature_distributions(hrv_samples, HRV_FEATURES, f"{condition_name} - HRV (Uncertainty)", output_plot_path_hrv)
                        else: logging.warning("No valid HRV samples found.")
                    if eda_samples:
                         eda_samples = [s for s in eda_samples if len(s) == len(EDA_FEATURES)]
                         if eda_samples:
                            eda_stats = {f: {'mean': float(np.mean([s[j] for s in eda_samples])), 'std': float(np.std([s[j] for s in eda_samples])), 'samples': [float(s[j]) for s in eda_samples]} for j, f in enumerate(EDA_FEATURES)}
                            output_plot_path_eda = os.path.join(structured_output_dir, f"feature_dist_eda_sample_{sample_idx}.png")
                            plot_feature_distributions(eda_samples, EDA_FEATURES, f"{condition_name} - EDA (Uncertainty)", output_plot_path_eda)
                         else: logging.warning("No valid EDA samples found.")

                    table_data = {"HRV Features (Stats)": hrv_stats, "EDA Features (Stats)": eda_stats}
                    print(f"\nGenerated Feature Stats (Sample {sample_idx} - {condition_name})")
                    if hrv_stats: print_feature_stats("HRV Features", hrv_stats, HRV_FEATURES)
                    if eda_stats: print_feature_stats("EDA Features", eda_stats, EDA_FEATURES)

                else: # Single feature set or variability plot
                    all_hrv_samples, all_eda_samples = [], []
                    num_gen = args.uncertainty_samples if args.plot_uncertainty else 1
                    gen_mode = "Variability" if args.plot_uncertainty else "Single"
                    logging.info(f"Generating {num_gen} feature set(s) ({gen_mode} mode).")
                    for iter_num in range(num_gen):
                         gen_z = torch.randn(1, latent_dim, device=device) if args.plot_uncertainty else z_base
                         with torch.no_grad(): sample_dict = model.generate(gen_z, condition_tensor)
                         hrv_tensor, eda_tensor = sample_dict.get('HRV_features'), sample_dict.get('EDA_features')
                         if isinstance(hrv_tensor, torch.Tensor):
                              hrv_sample = hrv_tensor.detach().cpu().numpy().squeeze()
                              if len(hrv_sample) == len(HRV_FEATURES): all_hrv_samples.append(hrv_sample)
                              else: logging.warning(f"HRV sample {iter_num+1} length mismatch: {len(hrv_sample)}")
                         if isinstance(eda_tensor, torch.Tensor):
                              eda_sample = eda_tensor.detach().cpu().numpy().squeeze()
                              if len(eda_sample) == len(EDA_FEATURES): all_eda_samples.append(eda_sample)
                              else: logging.warning(f"EDA sample {iter_num+1} length mismatch: {len(eda_sample)}")

                    if args.plot_uncertainty:
                         if all_hrv_samples:
                              output_plot_path_hrv = os.path.join(structured_output_dir, f"feature_dist_hrv_sample_{sample_idx}.png")
                              plot_feature_distributions(all_hrv_samples, HRV_FEATURES, f"{condition_name} - HRV ({gen_mode})", output_plot_path_hrv)
                         if all_eda_samples:
                              output_plot_path_eda = os.path.join(structured_output_dir, f"feature_dist_eda_sample_{sample_idx}.png")
                              plot_feature_distributions(all_eda_samples, EDA_FEATURES, f"{condition_name} - EDA ({gen_mode})", output_plot_path_eda)
                         table_data = {f"HRV Features ({gen_mode} Samples)": [dict(zip(HRV_FEATURES, map(float, s))) for s in all_hrv_samples],
                                       f"EDA Features ({gen_mode} Samples)": [dict(zip(EDA_FEATURES, map(float, s))) for s in all_eda_samples]}
                         print(f"\nGenerated {num_gen} sets of features (Sample {sample_idx}, {gen_mode}) - see JSON.")
                    else: # Single generation
                         hrv_sample = all_hrv_samples[0] if all_hrv_samples else np.array([])
                         eda_sample = all_eda_samples[0] if all_eda_samples else np.array([])
                         table_data = {"HRV Features": dict(zip(HRV_FEATURES, map(float, hrv_sample))),
                                       "EDA Features": dict(zip(EDA_FEATURES, map(float, eda_sample)))}
                         print(f"\nGenerated Features (Sample {sample_idx} - {condition_name})")
                         if hrv_sample.size > 0: print_feature_table("HRV Features", hrv_sample, HRV_FEATURES)
                         else: print("HRV Features: N/A")
                         if eda_sample.size > 0: print_feature_table("EDA Features", eda_sample, EDA_FEATURES)
                         else: print("EDA Features: N/A")

                # Save feature data to JSON using structured path
                output_json_path = os.path.join(structured_output_dir,
                    f"features_sample_{sample_idx}.json") # Simplified filename
                save_feature_data(table_data, output_json_path)

            except Exception as e:
                logging.error(f"Failed to generate features for sample {sample_idx}: {e}", exc_info=True)

    logging.info("\n--- Generation Complete ---")


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass # Allow SystemExit from argparse or explicit calls
    except ImportError as e:
         logging.critical(f"Critical dependency missing: {e}. Please install required packages.", exc_info=True)
         sys.exit(1)
    except Exception as e:
        logging.critical(f"An unexpected critical error occurred: {e}", exc_info=True)
        sys.exit(1)
