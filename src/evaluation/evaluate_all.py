import argparse
import torch
import numpy as np
import os
import json
import logging
import sys
import random
from typing import Dict, Tuple, List, Optional, Any

# --- Scikit-learn Imports (NEW) ---
from sklearn.model_selection import cross_val_score, StratifiedKFold # Changed from train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler # Good practice for classifiers
from sklearn.pipeline import Pipeline # Useful for combining scaling and classifier

# --- Custom Module Imports ---
# Ensure 'src' directory is in the Python path or adjust imports accordingly
try:
    # Functions to load model and infer parameters
    from src.training.generate_and_plot import (
        infer_parameters_from_dir,
        find_best_checkpoint,
        load_model
    )
    # Metric calculation functions
    from src.evaluation.metrics import (
        calculate_rmse,
        calculate_mmd,
        calculate_wasserstein_distance,
        calculate_feature_statistics,
        calculate_fid_like_score
    )
    # Uncertainty evaluation
    from src.evaluation.uncertainty_eval import uncertainty_evaluation_summary
except ImportError as e:
    logging.error(f"Failed to import necessary modules from 'src'. "
                  f"Ensure 'src' is in PYTHONPATH. Error: {e}")
    sys.exit(1)

# --- Constants ---
CONDITION_MAP = {1: "Baseline", 2: "Stress", 3: "Amusement"}

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# --- Helper Functions ---

def match_sample_counts(real_array: np.ndarray, gen_array: np.ndarray, random_subset: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ensures real and generated arrays have the same number of samples (axis 0).

    Args:
        real_array: NumPy array of real data samples.
        gen_array: NumPy array of generated data samples.
        random_subset: If True, randomly samples from the larger array.
                       If False, takes the first min_n samples.

    Returns:
        Tuple containing the adjusted real and generated arrays.
    """
    real_n = real_array.shape[0]
    gen_n = gen_array.shape[0]
    min_n = min(real_n, gen_n)

    if real_n == gen_n:
        return real_array, gen_array

    logging.info(f"Matching sample counts: Real={real_n}, Generated={gen_n}. Using {min_n} samples.")

    if real_n > min_n:
        if random_subset:
            idx = np.random.choice(real_n, min_n, replace=False)
            real_array = real_array[idx]
        else:
            real_array = real_array[:min_n]

    if gen_n > min_n:
        if random_subset:
            idx = np.random.choice(gen_n, min_n, replace=False)
            gen_array = gen_array[idx]
        else:
            gen_array = gen_array[:min_n]

    return real_array, gen_array

def prepare_real_data(data_path: str, condition: int, model_type: str, signal_type: str) -> Dict[str, np.ndarray]:
    """
    Loads real test data for a specific condition and formats it for metric functions.

    Args:
        data_path: Path to the .pt file containing test data.
        condition: The condition label (1, 2, or 3).
        model_type: 'time_series' or 'features'.
        signal_type: 'EDA' or 'BVP' (used for time_series).

    Returns:
        Dictionary formatted for metric functions (e.g., {'EDA_series': array}).
    """
    try:
        full_data = torch.load(data_path)
        logging.info(f"Loaded real data from {data_path}")
    except FileNotFoundError:
        logging.error(f"Real data file not found: {data_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading real data: {e}")
        raise

    real_data_dict = {}
    if model_type == 'time_series':
        data_key = f"{signal_type}_series"
        if data_key in full_data and condition in full_data[data_key]:
            data_tensor = full_data[data_key][condition]
            real_data_dict[data_key] = data_tensor.cpu().numpy()
            logging.info(f"Extracted real {data_key} data for condition {condition}, shape: {real_data_dict[data_key].shape}")
        else:
            logging.warning(f"No real data found for {data_key}, condition {condition} in {data_path}")
            raise ValueError(f"Missing real data for {data_key} condition {condition}")

    elif model_type == 'features':
        hrv_key = 'HRV_features'
        eda_key = 'EDA_features'
        found_hrv = False
        found_eda = False
        if hrv_key in full_data and condition in full_data[hrv_key]:
            data_tensor = full_data[hrv_key][condition]
            real_data_dict[hrv_key] = data_tensor.cpu().numpy()
            logging.info(f"Extracted real {hrv_key} data for condition {condition}, shape: {real_data_dict[hrv_key].shape}")
            found_hrv = True
        else:
             logging.warning(f"No real data found for {hrv_key}, condition {condition} in {data_path}")

        if eda_key in full_data and condition in full_data[eda_key]:
             data_tensor = full_data[eda_key][condition]
             real_data_dict[eda_key] = data_tensor.cpu().numpy()
             logging.info(f"Extracted real {eda_key} data for condition {condition}, shape: {real_data_dict[eda_key].shape}")
             found_eda = True
        else:
             logging.warning(f"No real data found for {eda_key}, condition {condition} in {data_path}")

        if not found_hrv or not found_eda:
             raise ValueError(f"Missing real feature data for condition {condition}")

    else:
        raise ValueError(f"Invalid model_type: {model_type}")

    return real_data_dict


def generate_samples(model: torch.nn.Module, num_samples: int, condition: int,
                     latent_dim: int, device: torch.device, model_type: str,
                     signal_type: str, is_bayesian: bool) -> Dict[str, np.ndarray]:
    """
    Generates samples from the model (one sample per unique latent vector z),
    formatted for metric functions. Suitable for distribution comparison metrics
    (MMD, Wasserstein) and standard RMSE/FID calculation.

    Args:
        model: The loaded GAN model.
        num_samples: Number of samples to generate.
        condition: Condition label (1, 2, or 3).
        latent_dim: Dimension of the latent space.
        device: Torch device (cpu or cuda).
        model_type: 'time_series' or 'features'.
        signal_type: 'EDA' or 'BVP'.
        is_bayesian: Whether the model is Bayesian (used to set train mode for dropout).

    Returns:
        Dictionary formatted for metric functions.
    """
    model.eval() # Set to evaluation mode for generation by default
    if is_bayesian and hasattr(model, 'dropout_rate'): # Specific check for Dropout Bayesian
        logging.info("Setting Bayesian Dropout model to train() mode for generation to enable dropout.")
        model.train() # Ensure dropout is active for Bayesian Dropout models

    generated_data_list = {key: [] for key in ['EDA_series', 'BVP_series', 'HRV_features', 'EDA_features']}
    condition_tensor = torch.tensor([condition], dtype=torch.long, device=device)

    logging.info(f"Generating {num_samples} individual samples (one per z) for condition {condition}...")

    for i in range(num_samples):
        z = torch.randn(1, latent_dim, device=device) # Generate one sample at a time with a new z

        with torch.no_grad():
            # Always generate a single sample for this z
            # For Bayesian models, this uses one sample from the posterior weights
            generated_output = model.generate(z, condition_tensor.expand(z.size(0))) # Expand condition for batch size 1

            if model_type == 'time_series':
                data_key = f"{signal_type}_series"
                if isinstance(generated_output, torch.Tensor):
                    # Squeeze batch dim and convert to numpy
                    generated_data_list[data_key].append(generated_output.squeeze(0).cpu().numpy())
                else:
                     logging.warning(f"Unexpected output type {type(generated_output)} from model.generate. Skipping sample {i}.")

            elif model_type == 'features':
                if isinstance(generated_output, dict):
                    for key in ['HRV_features', 'EDA_features']:
                        if key in generated_output and isinstance(generated_output[key], torch.Tensor):
                            # Squeeze batch dim and convert to numpy
                            generated_data_list[key].append(generated_output[key].squeeze(0).cpu().numpy())
                        else:
                             logging.warning(f"Missing or invalid data for key '{key}' in model.generate output. Skipping sample {i}.")
                             # Ensure key exists even if empty
                             if key not in generated_data_list: generated_data_list[key] = []
                else:
                     logging.warning(f"Unexpected output type {type(generated_output)} from feature model.generate. Skipping sample {i}.")

    # Combine samples into NumPy arrays
    generated_data_dict = {}
    if model_type == 'time_series':
        data_key = f"{signal_type}_series"
        if generated_data_list[data_key]:
            generated_data_dict[data_key] = np.stack(generated_data_list[data_key], axis=0)
            logging.info(f"Stacked generated {data_key} data shape: {generated_data_dict[data_key].shape}")
        else:
             raise ValueError(f"No valid {data_key} samples were generated.")
    elif model_type == 'features':
        for key in ['HRV_features', 'EDA_features']:
             if generated_data_list[key]:
                  generated_data_dict[key] = np.stack(generated_data_list[key], axis=0)
                  logging.info(f"Stacked generated {key} data shape: {generated_data_dict[key].shape}")
             else:
                  raise ValueError(f"No valid {key} samples were generated.")

    # Restore eval mode if model was set to train
    model.eval()

    return generated_data_dict


def generate_uncertainty_samples(model: torch.nn.Module, condition: int,
                                 latent_dim: int, device: torch.device,
                                 model_type: str, signal_type: str,
                                 num_latent_vecs: int = 10,
                                 samples_per_z: int = 10) -> Dict[str, List[np.ndarray]]:
    """
    Generates multiple samples from the same latent vectors for Bayesian models.
    This is used specifically for uncertainty evaluation.
    Ensures time series samples retain their channel dimension (T, C).
    Ensures feature samples are consistently shaped (e.g., 1D array (F,)).

    Args:
        model: The loaded GAN model (should have generate_with_uncertainty_samples).
        condition: Condition label (1, 2, or 3).
        latent_dim: Dimension of the latent space.
        device: Torch device (cpu or cuda).
        model_type: 'time_series' or 'features'.
        signal_type: 'EDA' or 'BVP' (used for time_series).
        num_latent_vecs: Number of distinct latent vectors (z) to sample from.
        samples_per_z: Number of output samples to generate for each z.

    Returns:
        Dictionary where keys are data types ('EDA_series', 'HRV_features', etc.)
        and values are flat lists of numpy arrays representing all generated
        samples across all latent vectors.
        e.g., {'EDA_series': [sample1_np, sample2_np, ..., sampleM_np]}
              where M = num_latent_vecs * samples_per_z
              and each sampleX_np has shape (T, C) for time series
              or (F,) for features.
    """
    # Ensure we have a Bayesian model with the uncertainty sampling method
    if not hasattr(model, 'generate_with_uncertainty_samples'):
        logging.warning("Model doesn't have 'generate_with_uncertainty_samples' method. Cannot generate uncertainty samples.")
        return {} # Return empty dict

    # Prepare model and track samples by latent vector
    if hasattr(model, 'dropout_rate'):  # Specific check for Dropout Bayesian
        model.train()  # Enable dropout for MC Dropout sampling
        logging.info("Enabled train() mode for Dropout model uncertainty sampling.")
    else:
        model.eval()  # For other Bayesian methods that don't need train mode

    # Stores results temporarily grouped by z:
    # { data_key: [ [z1_s1, z1_s2,...], [z2_s1, z2_s2,...] ] }
    samples_by_z: Dict[str, List[List[np.ndarray]]] = {}

    condition_tensor = torch.tensor([condition], dtype=torch.long, device=device)

    logging.info(f"Generating uncertainty samples from {num_latent_vecs} latent vectors with {samples_per_z} samples each...")

    # Generate samples from multiple latent vectors
    for z_idx in range(num_latent_vecs):
        # Create a fixed latent vector for this iteration
        z_base = torch.randn(1, latent_dim, device=device)

        try:
            # Generate multiple samples from the same latent vector
            # Assume model returns list of tensors or a stacked tensor
            # Use torch.no_grad() for inference efficiency if dropout is handled by model.train()
            with torch.no_grad() if not hasattr(model, 'dropout_rate') else torch.enable_grad(): # Context manager for grad
                 samples_output: Union[List[torch.Tensor], torch.Tensor, Dict[str, Any]] = \
                     model.generate_with_uncertainty_samples(
                        z_base, condition_tensor, n_samples=samples_per_z
                     )

            # --- Process based on model_type ---
            if model_type == 'time_series':
                data_key = f"{signal_type}_series"
                if data_key not in samples_by_z:
                    samples_by_z[data_key] = []

                z_samples_np_list = [] # Store numpy arrays for this specific z

                # --- Handle different output formats ---
                if isinstance(samples_output, list): # Expected: list of tensors
                    for sample_tensor in samples_output:
                        if isinstance(sample_tensor, torch.Tensor):
                            sample_np = sample_tensor.detach().cpu().numpy()
                            # Ensure shape is (T, C) - remove potential batch dim (dim 0) ONLY
                            if sample_np.shape[0] == 1 and sample_np.ndim > 1: # Check for batch dim=1
                                sample_np = sample_np.squeeze(0)

                            # Ensure channel dim exists (should be 2D: T, C)
                            if sample_np.ndim == 1: # If squeeze removed C=1, add it back
                                sample_np = np.expand_dims(sample_np, axis=-1) # Becomes (T, 1)

                            if sample_np.ndim == 2: # Should be (T, C) now
                                z_samples_np_list.append(sample_np)
                            else:
                                logging.warning(f"Unexpected sample shape {sample_np.shape} after processing time series. Skipping.")
                        else:
                            logging.warning(f"Item in time series samples_output list is not a Tensor: {type(sample_tensor)}")

                elif isinstance(samples_output, torch.Tensor): # Alt: returns tensor (samples_per_z, T, C)
                    samples_tensor = samples_output.detach().cpu().numpy()
                    # Handle potential variations in output tensor dimensions
                    if samples_tensor.ndim == 3: # Expected (M, T, C)
                        for s_idx in range(samples_tensor.shape[0]):
                            sample_np = samples_tensor[s_idx, :, :] # Shape (T, C)
                            if sample_np.ndim == 1: # Ensure channel dim
                                sample_np = np.expand_dims(sample_np, axis=-1)
                            z_samples_np_list.append(sample_np)
                    elif samples_tensor.ndim == 4 and samples_tensor.shape[1] == 1 : # Check for (M, 1, T, C)
                         logging.warning(f"Sample tensor has 4 dims {samples_tensor.shape}, attempting to squeeze dim 1.")
                         samples_tensor = samples_tensor.squeeze(1) # Remove singleton dimension
                         if samples_tensor.ndim == 3: # Should be (M, T, C) now
                            for s_idx in range(samples_tensor.shape[0]):
                                sample_np = samples_tensor[s_idx, :, :] # Shape (T, C)
                                if sample_np.ndim == 1: # Ensure channel dim
                                    sample_np = np.expand_dims(sample_np, axis=-1)
                                z_samples_np_list.append(sample_np)
                         else:
                             logging.warning(f"Could not reduce 4D tensor to 3D after squeeze. Final Shape: {samples_tensor.shape}")
                    else:
                         logging.warning(f"Unexpected Tensor shape from model: {samples_tensor.shape}. Expected 3 dims (M, T, C) or 4 dims (M, 1, T, C).")
                else:
                    logging.warning(f"Unexpected output type for time series from model: {type(samples_output)}")

                # Add processed numpy samples for this z if any were valid
                if z_samples_np_list:
                    samples_by_z[data_key].append(z_samples_np_list)

            elif model_type == 'features':
                # Process feature samples (expecting dict or list of dicts)
                hrv_key = 'HRV_features'
                eda_key = 'EDA_features'
                hrv_z_samples = []
                eda_z_samples = []

                # --- Handle different output formats ---
                if isinstance(samples_output, list): # List of dicts?
                    for item in samples_output:
                        if isinstance(item, dict):
                             hrv_tensor = item.get(hrv_key)
                             eda_tensor = item.get(eda_key)
                             if isinstance(hrv_tensor, torch.Tensor):
                                 hrv_np = hrv_tensor.detach().cpu().numpy().flatten() # Ensure 1D (F,)
                                 hrv_z_samples.append(hrv_np)
                             if isinstance(eda_tensor, torch.Tensor):
                                 eda_np = eda_tensor.detach().cpu().numpy().flatten() # Ensure 1D (F,)
                                 eda_z_samples.append(eda_np)
                        else:
                             logging.warning(f"Item in feature list is not a dict: {type(item)}")

                elif isinstance(samples_output, dict): # Dict of lists or tensors?
                     hrv_data = samples_output.get(hrv_key)
                     eda_data = samples_output.get(eda_key)

                     # Process HRV data
                     if isinstance(hrv_data, list): # List of tensors
                         for hrv_tensor in hrv_data:
                             if isinstance(hrv_tensor, torch.Tensor):
                                 hrv_np = hrv_tensor.detach().cpu().numpy().flatten()
                                 hrv_z_samples.append(hrv_np)
                     elif isinstance(hrv_data, torch.Tensor): # Stacked tensor (M, F)
                         hrv_tensor_all = hrv_data.detach().cpu().numpy()
                         if hrv_tensor_all.ndim == 2:
                             for i in range(hrv_tensor_all.shape[0]):
                                 hrv_z_samples.append(hrv_tensor_all[i, :]) # Append each row (F,)
                         else:
                             logging.warning(f"Unexpected HRV tensor shape in dict: {hrv_tensor_all.shape}")

                     # Process EDA data similarly
                     if isinstance(eda_data, list):
                         for eda_tensor in eda_data:
                             if isinstance(eda_tensor, torch.Tensor):
                                 eda_np = eda_tensor.detach().cpu().numpy().flatten()
                                 eda_z_samples.append(eda_np)
                     elif isinstance(eda_data, torch.Tensor):
                         eda_tensor_all = eda_data.detach().cpu().numpy()
                         if eda_tensor_all.ndim == 2:
                             for i in range(eda_tensor_all.shape[0]):
                                 eda_z_samples.append(eda_tensor_all[i, :])
                         else:
                             logging.warning(f"Unexpected EDA tensor shape in dict: {eda_tensor_all.shape}")
                else:
                     logging.warning(f"Unexpected output type for features from model: {type(samples_output)}")


                # Add collected samples for this z
                if hrv_z_samples:
                    if hrv_key not in samples_by_z: samples_by_z[hrv_key] = []
                    samples_by_z[hrv_key].append(hrv_z_samples) # Append list of (F,) arrays
                if eda_z_samples:
                    if eda_key not in samples_by_z: samples_by_z[eda_key] = []
                    samples_by_z[eda_key].append(eda_z_samples) # Append list of (F,) arrays

        except Exception as e:
            # Log error with traceback for easier debugging
            logging.error(f"Error generating/processing uncertainty samples for latent vector {z_idx}: {e}", exc_info=True)


    # --- Flatten the samples_by_z structure ---
    # Input: { key: [ [z1_s1, z1_s2,...], [z2_s1, z2_s2,...] ] }
    # Output: { key: [ s1, s2, s3, ... ] } (flat list of all numpy samples)
    uncertainty_samples_dict: Dict[str, List[np.ndarray]] = {}
    for key, list_of_z_sample_lists in samples_by_z.items():
        if not list_of_z_sample_lists:
            logging.warning(f"No samples were collected for key '{key}' during uncertainty generation.")
            continue
        # Flatten the list of lists into a single list containing all numpy arrays
        flat_list_of_samples = [sample_np for z_list in list_of_z_sample_lists for sample_np in z_list]
        uncertainty_samples_dict[key] = flat_list_of_samples
        logging.info(f"Successfully collected {len(flat_list_of_samples)} samples for key '{key}'.")


    # --- Final Report and Shape Check ---
    for key, final_samples_list in uncertainty_samples_dict.items():
        logging.info(f"Generated {len(final_samples_list)} total uncertainty samples for {key}")
        if final_samples_list:
            # Log the shape of the first numpy array in the final flattened list
            logging.info(f"Shape of first uncertainty sample numpy array for {key}: {final_samples_list[0].shape}")
        else:
             logging.warning(f"Final sample list for key '{key}' is empty.")

    # Restore model state to eval mode if it was changed
    model.eval()
    logging.info("Set model back to eval() mode.")

    return uncertainty_samples_dict

# --- NEW: Classifier Test Function (Using Cross-Validation) ---
def calculate_classifier_accuracy(
    real_data: Dict[str, np.ndarray],
    gen_data: Dict[str, np.ndarray],
    model_type: str,
    signal_type: Optional[str] = None, # Only needed for time_series
    n_splits: int = 5, # Number of folds for cross-validation
    random_state: Optional[int] = None
) -> Dict[str, float]:
    """
    Trains a classifier using k-fold cross-validation to distinguish
    real vs. generated data and returns the mean accuracy. More robust for
    smaller datasets than a single train/test split.

    Args:
        real_data: Dictionary containing real data arrays (already matched count).
        gen_data: Dictionary containing generated data arrays (already matched count).
        model_type: 'time_series' or 'features'.
        signal_type: 'EDA' or 'BVP' (required if model_type is 'time_series').
        n_splits: Number of folds for cross-validation.
        random_state: Random seed for reproducibility of folds and classifier.

    Returns:
        Dictionary containing the mean classifier accuracy score(s) across folds.
        e.g., {'Classifier_Mean_Accuracy': 0.65, 'Classifier_Std_Accuracy': 0.05}
    """
    accuracies = {}
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=random_state, max_iter=1000, solver='liblinear')) # Added solver for robustness
    ])
    # Define the cross-validation strategy
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    if model_type == 'time_series':
        if not signal_type:
            raise ValueError("signal_type is required for model_type 'time_series'")
        data_key = f"{signal_type}_series"
        if data_key not in real_data or data_key not in gen_data:
            logging.warning(f"Missing '{data_key}' in real or generated data for classifier test. Skipping.")
            return accuracies

        real_samples = real_data[data_key]
        gen_samples = gen_data[data_key]
        n_real = real_samples.shape[0]
        n_gen = gen_samples.shape[0]

        # Flatten the time series data
        real_flat = real_samples.reshape(n_real, -1)
        gen_flat = gen_samples.reshape(n_gen, -1)

        # Combine data and create labels
        X = np.vstack((real_flat, gen_flat))
        y = np.concatenate((np.zeros(n_real), np.ones(n_gen))) # 0 = real, 1 = generated

        try:
            # Perform cross-validation
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy', n_jobs=-1) # Use n_jobs for potential speedup
            mean_accuracy = np.mean(scores)
            std_accuracy = np.std(scores)
            metric_key_mean = f'{data_key}_Classifier_Mean_Accuracy'
            metric_key_std = f'{data_key}_Classifier_Std_Accuracy'
            accuracies[metric_key_mean] = mean_accuracy
            accuracies[metric_key_std] = std_accuracy
            logging.info(f"Cross-validated classifier test accuracy for {data_key}: "
                         f"Mean={mean_accuracy:.4f}, Std={std_accuracy:.4f} ({n_splits} folds)")
        except Exception as e:
             logging.error(f"Error during cross-validated classifier test for {data_key}: {e}", exc_info=True)


    elif model_type == 'features':
        hrv_key = 'HRV_features'
        eda_key = 'EDA_features'

        if hrv_key not in real_data or eda_key not in real_data or \
           hrv_key not in gen_data or eda_key not in gen_data:
            logging.warning("Missing HRV or EDA features in real or generated data for classifier test. Skipping combined test.")
            return accuracies

        real_hrv = real_data[hrv_key]
        real_eda = real_data[eda_key]
        gen_hrv = gen_data[hrv_key]
        gen_eda = gen_data[eda_key]

        assert real_hrv.shape[0] == real_eda.shape[0] == gen_hrv.shape[0] == gen_eda.shape[0], \
            "Feature sets have mismatched sample counts!"

        real_combined = np.hstack((real_hrv, real_eda))
        gen_combined = np.hstack((gen_hrv, gen_eda))
        n_real = real_combined.shape[0]
        n_gen = gen_combined.shape[0]

        X = np.vstack((real_combined, gen_combined))
        y = np.concatenate((np.zeros(n_real), np.ones(n_gen)))

        try:
            # Perform cross-validation
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
            mean_accuracy = np.mean(scores)
            std_accuracy = np.std(scores)
            metric_key_mean = 'Features_Combined_Classifier_Mean_Accuracy'
            metric_key_std = 'Features_Combined_Classifier_Std_Accuracy'
            accuracies[metric_key_mean] = mean_accuracy
            accuracies[metric_key_std] = std_accuracy
            logging.info(f"Cross-validated classifier test accuracy for combined features: "
                         f"Mean={mean_accuracy:.4f}, Std={std_accuracy:.4f} ({n_splits} folds)")

            # Optional: Separate cross-validation for HRV/EDA could go here if needed
            # Remember to handle potential errors individually

        except Exception as e:
             logging.error(f"Error during cross-validated classifier test for features: {e}", exc_info=True)

    else:
        logging.warning(f"Classifier test not implemented for model_type: {model_type}")

    return accuracies

# --- Main Evaluation Function ---
def main(args):
    """Runs the evaluation pipeline."""
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- Load Model and Config ---
    try:
        logging.info(f"--- Loading Model from: {args.model_dir} ---")
        inferred_params = infer_parameters_from_dir(args.model_dir)
        best_checkpoint_path = find_best_checkpoint(args.model_dir)
        model, config = load_model(
            checkpoint_path=best_checkpoint_path,
            **inferred_params # Pass inferred params directly
        )
        model.to(device)
        logging.info(f"Model loaded successfully. Inferred params: {inferred_params}")
        latent_dim = config.get('latent_dim', 100) # Get latent dim from loaded config or default
    except (FileNotFoundError, ValueError, RuntimeError, TypeError, AttributeError, ImportError, KeyError) as e:
        logging.error(f"Failed to load model: {e}")
        sys.exit(1)

    # --- Prepare Data ---
    try:
        logging.info(f"--- Loading Real Data ---")
        real_data_dict = prepare_real_data(
            args.test_data_path,
            args.condition,
            inferred_params['model_type'],
            inferred_params['signal_type']
        )
        # Determine number of samples needed based on real data
        first_key = list(real_data_dict.keys())[0]
        num_real_samples = real_data_dict[first_key].shape[0]
        # Use the smaller of user-requested samples or available real samples
        num_gen_samples = args.num_gen_samples if args.num_gen_samples > 0 else num_real_samples
        logging.info(f"Number of real samples for condition {args.condition}: {num_real_samples}")
        logging.info(f"Number of generated samples to create: {num_gen_samples}")

    except (FileNotFoundError, ValueError, KeyError) as e:
        logging.error(f"Failed to load or prepare real data: {e}")
        sys.exit(1)

    # --- Generate Samples ---
    try:
        logging.info(f"--- Generating Synthetic Data ---")
        # Generate individual samples (one per z)
        generated_data_dict = generate_samples(
            model=model,
            num_samples=num_gen_samples,
            condition=args.condition,
            latent_dim=latent_dim,
            device=device,
            model_type=inferred_params['model_type'],
            signal_type=inferred_params['signal_type'],
            is_bayesian=inferred_params['use_bayesian']
            # No mc_samples or gen_mode needed here
        )
    except (ValueError, RuntimeError) as e:
         logging.error(f"Failed during sample generation: {e}")
         sys.exit(1)
    
    # --- Generate Uncertainty Samples for Bayesian Models ---
    uncertainty_samples_dict = {}
    if inferred_params['use_bayesian'] and hasattr(model, 'generate_with_uncertainty_samples'):
        try:
            logging.info(f"--- Generating Uncertainty Samples for Bayesian Model ---")
            
            # Calculate number of latent vectors needed to match standard sample count
            # We want approximately num_gen_samples total uncertainty samples
            samples_per_z = args.uncertainty_samples
            
            # Calculate how many latent vectors we need to get close to num_gen_samples total samples
            # We want: num_latent_vectors * samples_per_z ≈ num_gen_samples
            num_latent_vectors = max(1, round(num_gen_samples / samples_per_z))
            
            logging.info(f"Using {num_latent_vectors} latent vectors with {samples_per_z} samples each")
            logging.info(f"This will generate approximately {num_latent_vectors * samples_per_z} total uncertainty samples")
            logging.info(f"(compared to {num_gen_samples} standard samples)")
            
            uncertainty_samples_dict = generate_uncertainty_samples(
                model=model,
                condition=args.condition,
                latent_dim=latent_dim,
                device=device,
                model_type=inferred_params['model_type'],
                signal_type=inferred_params['signal_type'],
                num_latent_vecs=num_latent_vectors,
                samples_per_z=samples_per_z
            )
            
            # Check if we got uncertainty samples
            if not uncertainty_samples_dict:
                logging.warning("No uncertainty samples were generated. Skipping uncertainty evaluation.")
        except Exception as e:
            logging.error(f"Failed to generate uncertainty samples: {e}")
            logging.warning("Continuing with standard evaluation metrics only.")
    else:
        if not inferred_params['use_bayesian']:
            logging.info("Model is not Bayesian. Skipping uncertainty evaluation.")
        else:
            logging.info("Bayesian model does not support uncertainty sampling. Skipping uncertainty evaluation.")

    # --- Match Sample Counts and Calculate Metrics ---
    logging.info(f"--- Calculating Metrics ---")
    all_metrics = {}
    try:
        if inferred_params['model_type'] == 'time_series':
            data_key = f"{inferred_params['signal_type']}_series"
            # Ensure generated dict has the key before matching
            if data_key not in generated_data_dict:
                 raise KeyError(f"Generated data dictionary missing key: {data_key}")
            if data_key not in real_data_dict:
                 raise KeyError(f"Real data dictionary missing key: {data_key}")

            real_arr, gen_arr = match_sample_counts(real_data_dict[data_key], generated_data_dict[data_key])
            real_data_matched = {data_key: real_arr}
            gen_data_matched = {data_key: gen_arr}

            logging.info(f"Calculating RMSE for {data_key}...")
            rmse_res = calculate_rmse(real_data_matched, gen_data_matched, feature_type='time_series')
            logging.info(f"Calculating MMD for {data_key}...")
            mmd_res = calculate_mmd(real_data_matched, gen_data_matched, feature_type='time_series')
            logging.info(f"Calculating Wasserstein for {data_key}...")
            wd_res = calculate_wasserstein_distance(real_data_matched, gen_data_matched, feature_type='time_series')

            # Calculate Stats and FID-like for time series
            logging.info(f"Calculating feature statistics and FID-like score for {data_key}...")
            real_stats = calculate_feature_statistics(real_data_matched, feature_type='time_series')
            gen_stats = calculate_feature_statistics(gen_data_matched, feature_type='time_series')
            fid_ts = calculate_fid_like_score(real_stats, gen_stats, data_key)

            all_metrics.update({
                f"{data_key}_RMSE": rmse_res.get(data_key),
                f"{data_key}_MMD": mmd_res.get(data_key),
                f"{data_key}_Wasserstein": wd_res.get(data_key),
                f"{data_key}_FID_like": fid_ts,
                f"{data_key}_Real_Stats": real_stats,
                f"{data_key}_Generated_Stats": gen_stats
            })

            # --- NEW: Calculate Classifier Accuracy ---
            logging.info(f"Calculating Classifier Accuracy for {data_key}...")
            classifier_acc = calculate_classifier_accuracy(
                real_data_matched,
                gen_data_matched,
                model_type=inferred_params['model_type'],
                signal_type=inferred_params['signal_type'],
                random_state=args.seed # Use the main seed
            )
            all_metrics.update(classifier_acc)
            # --- End NEW ---
            
            # --- Calculate uncertainty metrics if available ---
            if inferred_params['use_bayesian'] and data_key in uncertainty_samples_dict and uncertainty_samples_dict[data_key]:
                logging.info(f"Calculating uncertainty metrics for {data_key}...")
                try:
                    # Create a dict with the same structure as expected by uncertainty_evaluation_summary
                    uncertainty_data = {data_key: uncertainty_samples_dict[data_key]}
                    
                    # Count the total uncertainty samples
                    num_uncertainty_samples = len(uncertainty_samples_dict[data_key])
                    logging.info(f"Total uncertainty samples for {data_key}: {num_uncertainty_samples}")
                    
                    # For standard metrics we matched sample counts, but for uncertainty evaluation
                    # we need to make sure the real and generated data have exactly the same shape
                    # We'll use the real_data_matched but potentially need to subsample or oversample
                    if num_uncertainty_samples != real_data_matched[data_key].shape[0]:
                        logging.info(f"Matching real data count ({real_data_matched[data_key].shape[0]}) to uncertainty sample count ({num_uncertainty_samples})")
                        
                        if real_data_matched[data_key].shape[0] > num_uncertainty_samples:
                            # Randomly select subset of real data
                            indices = np.random.choice(real_data_matched[data_key].shape[0], num_uncertainty_samples, replace=False)
                            real_subset = real_data_matched[data_key][indices]
                        else:
                            # Need more real samples - repeat some
                            indices = np.random.choice(real_data_matched[data_key].shape[0], num_uncertainty_samples, replace=True)
                            real_subset = real_data_matched[data_key][indices]
                    else:
                        # Counts already match
                        real_subset = real_data_matched[data_key]
                        logging.info(f"Real data count already matches uncertainty sample count: {num_uncertainty_samples}")
                    
                    real_matched_for_uncertainty = {data_key: real_subset}
                    
                    # Create a directory for calibration plots if needed
                    uncertainty_plots_dir = None
                    if args.output_dir:
                        uncertainty_plots_dir = os.path.join(args.output_dir, 'uncertainty_plots')
                        os.makedirs(uncertainty_plots_dir, exist_ok=True)
                    
                    # Calculate uncertainty metrics with matched real data
                    uncertainty_metrics = uncertainty_evaluation_summary(
                        real_data=real_matched_for_uncertainty,
                        generated_samples=uncertainty_data,
                        feature_type='time_series',
                        alpha=0.1,  # For 90% prediction intervals
                        save_path=uncertainty_plots_dir,
                        sample_strategy='random'
                    )
                    
                    # Add to overall metrics
                    all_metrics[f"{data_key}_Uncertainty"] = uncertainty_metrics
                    logging.info(f"Added uncertainty metrics for {data_key}")
                    
                    # Log key uncertainty metrics
                    if data_key in uncertainty_metrics.get('coverage', {}):
                        coverage_info = uncertainty_metrics['coverage'][data_key]
                        logging.info(f"Mean coverage probability: {coverage_info.get('mean_coverage', 'N/A')}")
                        logging.info(f"Expected coverage: {coverage_info.get('expected_coverage', 'N/A')}")
                    
                    if data_key in uncertainty_metrics.get('negative_log_likelihood', {}):
                        nll = uncertainty_metrics['negative_log_likelihood'][data_key]
                        if not isinstance(nll, dict):  # If it's a scalar value, not a warning
                            logging.info(f"Negative log-likelihood: {nll}")
                    
                except Exception as e:
                    logging.error(f"Error calculating uncertainty metrics for {data_key}: {e}")
                    logging.warning("Continuing without uncertainty metrics.")

        elif inferred_params['model_type'] == 'features':
            # Ensure keys exist
            if 'HRV_features' not in generated_data_dict or 'EDA_features' not in generated_data_dict:
                 raise KeyError("Generated data dictionary missing feature keys.")
            if 'HRV_features' not in real_data_dict or 'EDA_features' not in real_data_dict:
                 raise KeyError("Real data dictionary missing feature keys.")

            # Match counts for each feature type separately
            real_hrv, gen_hrv = match_sample_counts(real_data_dict['HRV_features'], generated_data_dict['HRV_features'])
            real_eda, gen_eda = match_sample_counts(real_data_dict['EDA_features'], generated_data_dict['EDA_features'])
            real_data_matched = {'HRV_features': real_hrv, 'EDA_features': real_eda}
            gen_data_matched = {'HRV_features': gen_hrv, 'EDA_features': gen_eda}

            logging.info("Calculating metrics for features...")
            rmse_res = calculate_rmse(real_data_matched, gen_data_matched, feature_type='features')
            mmd_res = calculate_mmd(real_data_matched, gen_data_matched, feature_type='features')
            wd_res = calculate_wasserstein_distance(real_data_matched, gen_data_matched, feature_type='features')

            # Calculate Stats and FID-like
            real_stats = calculate_feature_statistics(real_data_matched, feature_type='features')
            gen_stats = calculate_feature_statistics(gen_data_matched, feature_type='features')
            fid_hrv = calculate_fid_like_score(real_stats, gen_stats, 'HRV_features')
            fid_eda = calculate_fid_like_score(real_stats, gen_stats, 'EDA_features')

            all_metrics.update({
                "HRV_RMSE": rmse_res.get('HRV_features'),
                "EDA_Features_RMSE": rmse_res.get('EDA_features'),
                "HRV_MMD": mmd_res.get('HRV_features'),
                "EDA_Features_MMD": mmd_res.get('EDA_features'),
                "HRV_Wasserstein": wd_res.get('HRV_features'),
                "EDA_Features_Wasserstein": wd_res.get('EDA_features'),
                "HRV_FID_like": fid_hrv,
                "EDA_Features_FID_like": fid_eda,
                "Real_Stats": real_stats, # Include stats for reference
                "Generated_Stats": gen_stats
            })

            # --- NEW: Calculate Classifier Accuracy ---
            logging.info("Calculating Classifier Accuracy for features...")
            classifier_acc = calculate_classifier_accuracy(
                real_data_matched,
                gen_data_matched,
                model_type=inferred_params['model_type'],
                random_state=args.seed # Use the main seed
            )
            all_metrics.update(classifier_acc)
            # --- End NEW ---
            
            # --- Calculate uncertainty metrics for features if available ---
            if inferred_params['use_bayesian'] and 'HRV_features' in uncertainty_samples_dict and 'EDA_features' in uncertainty_samples_dict:
                logging.info("Calculating uncertainty metrics for features...")
                try:
                    # Check if we have samples to use
                    hrv_samples = uncertainty_samples_dict['HRV_features']
                    eda_samples = uncertainty_samples_dict['EDA_features']
                    
                    if hrv_samples and eda_samples:
                        # Count the total uncertainty samples 
                        num_hrv_uncertainty_samples = len(hrv_samples)
                        num_eda_uncertainty_samples = len(eda_samples)
                        logging.info(f"Total uncertainty samples - HRV: {num_hrv_uncertainty_samples}, EDA: {num_eda_uncertainty_samples}")
                        
                        # For uncertainty evaluation, need to match real and generated sample counts exactly
                        # Match HRV samples
                        if num_hrv_uncertainty_samples != real_data_matched['HRV_features'].shape[0]:
                            logging.info(f"Matching real HRV data count ({real_data_matched['HRV_features'].shape[0]}) to uncertainty sample count ({num_hrv_uncertainty_samples})")
                            
                            if real_data_matched['HRV_features'].shape[0] > num_hrv_uncertainty_samples:
                                indices = np.random.choice(real_data_matched['HRV_features'].shape[0], num_hrv_uncertainty_samples, replace=False)
                                real_hrv_subset = real_data_matched['HRV_features'][indices]
                            else:
                                indices = np.random.choice(real_data_matched['HRV_features'].shape[0], num_hrv_uncertainty_samples, replace=True)
                                real_hrv_subset = real_data_matched['HRV_features'][indices]
                        else:
                            real_hrv_subset = real_data_matched['HRV_features']
                            logging.info(f"Real HRV data count already matches uncertainty sample count: {num_hrv_uncertainty_samples}")
                        
                        # Match EDA samples
                        if num_eda_uncertainty_samples != real_data_matched['EDA_features'].shape[0]:
                            logging.info(f"Matching real EDA data count ({real_data_matched['EDA_features'].shape[0]}) to uncertainty sample count ({num_eda_uncertainty_samples})")
                            
                            if real_data_matched['EDA_features'].shape[0] > num_eda_uncertainty_samples:
                                indices = np.random.choice(real_data_matched['EDA_features'].shape[0], num_eda_uncertainty_samples, replace=False)
                                real_eda_subset = real_data_matched['EDA_features'][indices]
                            else:
                                indices = np.random.choice(real_data_matched['EDA_features'].shape[0], num_eda_uncertainty_samples, replace=True)
                                real_eda_subset = real_data_matched['EDA_features'][indices]
                        else:
                            real_eda_subset = real_data_matched['EDA_features']
                            logging.info(f"Real EDA data count already matches uncertainty sample count: {num_eda_uncertainty_samples}")
                        
                        # Create matched real data dictionary
                        real_matched_for_uncertainty = {
                            'HRV_features': real_hrv_subset,
                            'EDA_features': real_eda_subset
                        }
                        
                        # Create a directory for calibration plots if needed
                        uncertainty_plots_dir = None
                        if args.output_dir:
                            uncertainty_plots_dir = os.path.join(args.output_dir, 'uncertainty_plots')
                            os.makedirs(uncertainty_plots_dir, exist_ok=True)
                        
                        # Calculate uncertainty metrics with matched real data
                        uncertainty_metrics = uncertainty_evaluation_summary(
                            real_data=real_matched_for_uncertainty,
                            generated_samples={'HRV_features': hrv_samples, 'EDA_features': eda_samples},
                            feature_type='features',
                            alpha=0.1,  # For 90% prediction intervals
                            save_path=uncertainty_plots_dir,
                            sample_strategy='random'
                        )
                        
                        # Add to overall metrics
                        all_metrics["Features_Uncertainty"] = uncertainty_metrics
                        logging.info("Added uncertainty metrics for features")
                        
                        # Log key uncertainty metrics
                        for feature_type in ['HRV_features', 'EDA_features']:
                            if feature_type in uncertainty_metrics.get('coverage', {}):
                                coverage_info = uncertainty_metrics['coverage'][feature_type]
                                logging.info(f"{feature_type} mean coverage probability: {coverage_info.get('mean_coverage', 'N/A')}")
                                logging.info(f"{feature_type} expected coverage: {coverage_info.get('expected_coverage', 'N/A')}")
                            
                            if feature_type in uncertainty_metrics.get('negative_log_likelihood', {}):
                                nll = uncertainty_metrics['negative_log_likelihood'][feature_type]
                                if not isinstance(nll, dict):  # If it's a scalar value, not a warning
                                    logging.info(f"{feature_type} negative log-likelihood: {nll}")
                    else:
                        logging.warning("Insufficient uncertainty samples for features. Skipping uncertainty evaluation.")
                
                except Exception as e:
                    logging.error(f"Error calculating uncertainty metrics for features: {e}")
                    logging.warning("Continuing without uncertainty metrics for features.")

        logging.info("\n--- Evaluation Results ---")
        for key, value in all_metrics.items():
            if isinstance(value, (float, int, np.number)):
                logging.info(f"{key}: {value:.6f}")
            # Avoid logging the full stats dictionary to console
            elif key not in ["Real_Stats", "Generated_Stats"]:
                 logging.info(f"{key}: {value}") # Log other types directly

        # --- Save Results ---
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            model_dir_basename = os.path.basename(os.path.normpath(args.model_dir))
            condition_name = CONDITION_MAP.get(args.condition, f"Cond{args.condition}")
            # Add model type and signal type to filename for clarity
            model_desc = f"{inferred_params['model_type']}_{inferred_params['signal_type']}" if inferred_params['model_type'] == 'time_series' else "features"
            results_filename = f"metrics_{model_dir_basename}_{model_desc}_{condition_name}.json"
            results_path = os.path.join(args.output_dir, results_filename)

            # Convert numpy arrays in stats to lists for JSON serialization
            def convert_stats_to_json(stats_dict):
                json_stats = {}
                for key, stats_data in stats_dict.items(): # Iterate through 'HRV_features', 'EDA_features'
                    json_stats[key] = {}
                    if isinstance(stats_data, dict):
                         for stat_name, stat_val in stats_data.items(): # Iterate through 'mean', 'std' etc.
                              if isinstance(stat_val, np.ndarray):
                                   json_stats[key][stat_name] = stat_val.tolist()
                              elif isinstance(stat_val, np.number): # Handle numpy scalars
                                   json_stats[key][stat_name] = float(stat_val)
                              else:
                                   json_stats[key][stat_name] = stat_val # Keep other types as is
                    else:
                         json_stats[key] = stats_data # Keep non-dict values as is
                return json_stats
            
            # Prepare metrics for JSON saving
            def convert_value_to_json_safe(value):
                if isinstance(value, (np.ndarray)):
                    return value.tolist()
                elif isinstance(value, (np.number)):
                     # Handle numpy floats/ints, including potential NaN/Inf
                    if np.isnan(value): return None # Or "NaN" as string
                    if np.isinf(value): return None # Or "Infinity" as string
                    return value.item() # Convert to Python native type
                elif isinstance(value, (float, int, str, bool)) or value is None:
                    return value # Already JSON serializable
                elif isinstance(value, dict):
                     # Recursively convert dict values
                    return {k: convert_value_to_json_safe(v) for k, v in value.items()}
                elif isinstance(value, list):
                     # Recursively convert list items
                    return [convert_value_to_json_safe(item) for item in value]
                else:
                    # Fallback for unknown types
                    logging.warning(f"Converting unknown type {type(value)} to string for JSON.")
                    return str(value)

            # Prepare metrics for JSON saving
            json_metrics = {}
            for key, value in all_metrics.items():
                 # Special handling for stats dicts containing numpy arrays
                 if "Real_Stats" in key or "Generated_Stats" in key:
                      json_metrics[key] = convert_stats_to_json(value) # Use existing specialized func
                 else:
                      # Use the new general recursive converter for other metrics (incl. uncertainty)
                      json_metrics[key] = convert_value_to_json_safe(value)


            try:
                with open(results_path, 'w') as f:
                    json.dump(json_metrics, f, indent=4)
                logging.info(f"Metrics saved to: {results_path}")
            except Exception as e:
                logging.error(f"Failed to save metrics to JSON: {e}")

    except Exception as e:
        logging.error(f"An error occurred during metric calculation: {e}", exc_info=True)
        sys.exit(1)

    logging.info("--- Evaluation Finished ---")


# --- Argument Parser ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate GAN models using various metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to the directory containing the 'best_*.pt' model checkpoint and 'config.json'.")
    parser.add_argument("--test_data_path", type=str, required=True,
                        help="Path to the .pt file containing the real test dataset.")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="Directory to save the calculated metrics (JSON file).")
    parser.add_argument("--condition", type=int, default=2, choices=CONDITION_MAP.keys(),
                        help=f"Condition label to evaluate: {', '.join([f'{k}={v}' for k, v in CONDITION_MAP.items()])}. Default is Stress (2).")
    parser.add_argument("--num_gen_samples", type=int, default=0,
                        help="Number of samples to generate for evaluation. If 0, uses the number of real samples found for the condition.")
    parser.add_argument("--uncertainty_samples", type=int, default=10,
                        help="Number of uncertainty samples to generate per latent vector for Bayesian models.")
    parser.add_argument("--num_latent_vectors", type=int, default=10,
                        help="Number of latent vectors to use for uncertainty evaluation in Bayesian models.")
    # mc_samples is no longer directly used by generate_samples in this version
    # parser.add_argument("--mc_samples", type=int, default=10,
    #                     help="Number of Monte Carlo samples per input 'z' if using 'mean' generation mode for Bayesian models.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    parsed_args = parser.parse_args()
    # Add mc_samples back to args if needed elsewhere, even if not used by generate_samples
    if not hasattr(parsed_args, 'mc_samples'):
         parsed_args.mc_samples = 10 # Add a default value if needed by other parts

    main(parsed_args)
