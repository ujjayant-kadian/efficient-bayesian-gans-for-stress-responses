# Required imports for the new script
import argparse
import torch
import numpy as np
import os
import json
import logging
import sys
import random
from typing import Dict, Tuple, List, Optional, Any

# Imports for correlation calculation and plotting
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Import Functions from your Existing Modules/Script ---
# NOTE: Adjust these import paths based on your project structure.
try:
    # Assume functions related to model loading are here
    from src.training.generate_and_plot import (
        infer_parameters_from_dir,
        find_best_checkpoint,
        load_model
    )
    # Imports from src.evaluation.utils are removed as functions will be defined locally
except ImportError as e:
     logging.error(f"Failed to import necessary functions from src.training.generate_and_plot. "
                   f"Ensure relevant modules from 'src' are in PYTHONPATH. Error: {e}")
     sys.exit(1)


# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# --- Constants ---
CONDITION_MAP = {1: "Baseline", 2: "Stress", 3: "Amusement"}

# --- Locally Defined Helper Functions ---

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

def generate_samples(model: torch.nn.Module, num_samples: int, condition: int,
                     latent_dim: int, device: torch.device, model_type: str,
                     signal_type: Optional[str], is_bayesian: bool) -> Dict[str, np.ndarray]:
    """
    Generates samples from the model (one sample per unique latent vector z)
    for a SINGLE condition.
    (Copied from original script/utils for use by generate_samples_all_conditions)

    Args:
        model: The loaded GAN model.
        num_samples: Number of samples to generate for this condition.
        condition: Condition label (1, 2, or 3).
        latent_dim: Dimension of the latent space.
        device: Torch device (cpu or cuda).
        model_type: 'time_series' or 'features'.
        signal_type: 'EDA' or 'BVP' (or None for features).
        is_bayesian: Whether the model is Bayesian.

    Returns:
        Dictionary formatted for metric functions (e.g., {'HRV_features': ..., 'EDA_features': ...}).
    """
    # Ensure model_type is features for this script's context
    if model_type != 'features':
         raise ValueError("generate_samples called with incorrect model_type in correlation script.")

    model.eval() # Set to evaluation mode for generation by default
    if is_bayesian and hasattr(model, 'dropout_rate'): # Specific check for Dropout Bayesian
        logging.debug("Setting Bayesian Dropout model to train() mode for generation.")
        model.train() # Ensure dropout is active for Bayesian Dropout models

    # Initialize lists for features only
    generated_data_list = {key: [] for key in ['HRV_features', 'EDA_features']}
    condition_tensor = torch.tensor([condition], dtype=torch.long, device=device)

    logging.debug(f"Generating {num_samples} individual samples (one per z) for condition {condition}...")

    for i in range(num_samples):
        z = torch.randn(1, latent_dim, device=device) # Generate one sample at a time with a new z

        # Use torch.no_grad() for inference if not doing MC Dropout
        context = torch.enable_grad() if (is_bayesian and hasattr(model, 'dropout_rate')) else torch.no_grad()
        with context:
            generated_output = model.generate(z, condition_tensor.expand(z.size(0))) # Expand condition for batch size 1

            if isinstance(generated_output, dict):
                for key in ['HRV_features', 'EDA_features']:
                    if key in generated_output and isinstance(generated_output[key], torch.Tensor):
                        # Squeeze batch dim and convert to numpy
                        generated_data_list[key].append(generated_output[key].squeeze(0).detach().cpu().numpy()) # Added detach()
                    else:
                        logging.warning(f"Missing or invalid data for key '{key}' in model.generate output for condition {condition}. Skipping sample {i}.")
                        # Ensure key exists even if empty for this sample run
                        if key not in generated_data_list: generated_data_list[key] = []
            else:
                logging.warning(f"Unexpected output type {type(generated_output)} from feature model.generate for condition {condition}. Skipping sample {i}.")

    # Combine samples into NumPy arrays
    generated_data_dict = {}
    try:
        for key in ['HRV_features', 'EDA_features']:
              if generated_data_list[key]: # Check if list is not empty
                  generated_data_dict[key] = np.stack(generated_data_list[key], axis=0)
                  logging.debug(f"Stacked generated {key} data shape for condition {condition}: {generated_data_dict[key].shape}")
              else:
                  # If no valid samples were generated for a key, create an empty array with correct feature dimension if possible
                  # This requires knowing the feature dimension beforehand, which is tricky here.
                  # Let's raise an error or return empty for now.
                  logging.error(f"No valid {key} samples were generated for condition {condition}.")
                  # Return empty dict to signal failure for this condition in the calling function
                  model.eval() # Ensure model is back in eval mode
                  return {} # Indicate failure
    except ValueError as e:
        logging.error(f"Stacking failed for condition {condition}, likely due to inconsistent shapes: {e}")
        model.eval()
        return {} # Indicate failure


    # Restore eval mode if model was set to train
    model.eval()

    return generated_data_dict


def prepare_real_data_all_conditions(data_path: str, model_type: str) -> Dict[str, np.ndarray]:
    """
    Loads real test data for ALL conditions and concatenates features.

    Args:
        data_path: Path to the .pt file containing test data.
        model_type: Must be 'features'.

    Returns:
        Dictionary containing concatenated 'HRV_features' and 'EDA_features' arrays.
        Returns empty arrays if no feature data is found.
    """
    if model_type != 'features':
        raise ValueError("This function only supports model_type='features'")

    hrv_key = 'HRV_features'
    eda_key = 'EDA_features'
    all_real_hrv: List[np.ndarray] = []
    all_real_eda: List[np.ndarray] = []
    total_samples = 0

    try:
        # Load data onto CPU to avoid potential CUDA memory issues during loading large files
        full_data = torch.load(data_path, map_location='cpu')
        logging.info(f"Loaded full real data from {data_path}")
    except FileNotFoundError:
        logging.error(f"Real data file not found: {data_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading real data: {e}")
        raise

    for condition_label, condition_name in CONDITION_MAP.items():
        hrv_found = False
        eda_found = False
        condition_hrv = None
        condition_eda = None

        # Extract HRV
        if hrv_key in full_data and condition_label in full_data[hrv_key]:
            condition_hrv_tensor = full_data[hrv_key][condition_label]
            # Ensure it's a tensor before converting
            if isinstance(condition_hrv_tensor, torch.Tensor):
                condition_hrv = condition_hrv_tensor.numpy() # Already on CPU
                if condition_hrv.ndim == 2 and condition_hrv.shape[0] > 0: # Basic validation
                    hrv_found = True
                else:
                    logging.warning(f"Invalid or empty HRV data for condition {condition_name} ({condition_label}). Shape: {condition_hrv.shape}")
            else:
                 logging.warning(f"HRV data for condition {condition_name} is not a Tensor (Type: {type(condition_hrv_tensor)}). Skipping.")
        else:
            logging.warning(f"No real {hrv_key} data found for condition {condition_name} ({condition_label}) in {data_path}")

        # Extract EDA
        if eda_key in full_data and condition_label in full_data[eda_key]:
            condition_eda_tensor = full_data[eda_key][condition_label]
            if isinstance(condition_eda_tensor, torch.Tensor):
                condition_eda = condition_eda_tensor.numpy() # Already on CPU
                if condition_eda.ndim == 2 and condition_eda.shape[0] > 0: # Basic validation
                    eda_found = True
                else:
                    logging.warning(f"Invalid or empty EDA data for condition {condition_name} ({condition_label}). Shape: {condition_eda.shape}")
            else:
                 logging.warning(f"EDA data for condition {condition_name} is not a Tensor (Type: {type(condition_eda_tensor)}). Skipping.")
        else:
            logging.warning(f"No real {eda_key} data found for condition {condition_name} ({condition_label}) in {data_path}")

        # Check consistency and append
        if hrv_found and eda_found:
            if condition_hrv.shape[0] != condition_eda.shape[0]:
                logging.warning(f"HRV ({condition_hrv.shape[0]}) and EDA ({condition_eda.shape[0]}) sample count mismatch for condition {condition_name}. Skipping this condition.")
            else:
                all_real_hrv.append(condition_hrv)
                all_real_eda.append(condition_eda)
                logging.info(f"Loaded {condition_hrv.shape[0]} samples for condition {condition_name}.")
                total_samples += condition_hrv.shape[0]
        elif hrv_found or eda_found:
             logging.warning(f"Missing either HRV or EDA data for condition {condition_name}. Skipping this condition.")


    if not all_real_hrv or not all_real_eda:
        logging.error("Could not load sufficient feature data from any condition.")
        # Return empty arrays
        return {hrv_key: np.array([]), eda_key: np.array([])}

    # Concatenate data from all conditions
    combined_hrv = np.concatenate(all_real_hrv, axis=0)
    combined_eda = np.concatenate(all_real_eda, axis=0)
    logging.info(f"Total real samples loaded across all conditions: {total_samples}")
    logging.info(f"Combined Real HRV shape: {combined_hrv.shape}")
    logging.info(f"Combined Real EDA shape: {combined_eda.shape}")

    return {hrv_key: combined_hrv, eda_key: combined_eda}


def generate_samples_all_conditions(
    model: torch.nn.Module,
    total_num_samples: int,
    latent_dim: int,
    device: torch.device,
    model_type: str,
    is_bayesian: bool
) -> Dict[str, np.ndarray]:
    """
    Generates synthetic samples for ALL conditions and concatenates features.

    Args:
        model: The loaded generator model.
        total_num_samples: The total number of samples to generate across all conditions.
        latent_dim: Dimension of the latent space.
        device: Torch device.
        model_type: Must be 'features'.
        is_bayesian: Whether the model is Bayesian.

    Returns:
        Dictionary containing concatenated 'HRV_features' and 'EDA_features' arrays.
    """
    if model_type != 'features':
        raise ValueError("This function only supports model_type='features'")

    hrv_key = 'HRV_features'
    eda_key = 'EDA_features'
    all_gen_hrv: List[np.ndarray] = []
    all_gen_eda: List[np.ndarray] = []
    num_conditions = len(CONDITION_MAP)

    if num_conditions == 0:
        logging.error("CONDITION_MAP is empty. Cannot generate samples.")
        return {hrv_key: np.array([]), eda_key: np.array([])}

    # Distribute samples somewhat evenly across conditions
    samples_per_condition = total_num_samples // num_conditions
    remainder = total_num_samples % num_conditions
    samples_generated = 0

    logging.info(f"Generating approx. {samples_per_condition} samples per condition for {num_conditions} conditions (Total: {total_num_samples}).")

    for i, (condition_label, condition_name) in enumerate(CONDITION_MAP.items()):
        num_to_gen = samples_per_condition + (1 if i < remainder else 0)
        if num_to_gen == 0:
            continue

        logging.info(f"Generating {num_to_gen} samples for condition {condition_name} ({condition_label})...")
        try:
            # Call the original single-condition generator function (now defined locally)
            generated_dict = generate_samples( # Calls the local function
                model=model,
                num_samples=num_to_gen,
                condition=condition_label,
                latent_dim=latent_dim,
                device=device,
                model_type=model_type,
                signal_type=None, # Not used for features
                is_bayesian=is_bayesian
            )

            # Check if generation for the condition was successful
            if generated_dict and hrv_key in generated_dict and eda_key in generated_dict:
                 # Basic check for non-empty arrays
                 if generated_dict[hrv_key].size > 0 and generated_dict[eda_key].size > 0:
                    all_gen_hrv.append(generated_dict[hrv_key])
                    all_gen_eda.append(generated_dict[eda_key])
                    samples_generated += generated_dict[hrv_key].shape[0] # Count actual generated
                 else:
                     logging.warning(f"Generation for condition {condition_name} produced empty arrays. Skipping.")
            else:
                logging.warning(f"Generation for condition {condition_name} failed or did not produce expected feature keys. Skipping.")

        except Exception as e:
            logging.error(f"Error generating samples for condition {condition_name}: {e}", exc_info=True)
            # Decide whether to continue or stop if one condition fails

    if not all_gen_hrv or not all_gen_eda:
        logging.error("Could not generate sufficient feature data for any condition.")
        return {hrv_key: np.array([]), eda_key: np.array([])}

    # Concatenate data from all conditions
    combined_hrv = np.concatenate(all_gen_hrv, axis=0)
    combined_eda = np.concatenate(all_gen_eda, axis=0)
    logging.info(f"Total generated samples across all conditions: {samples_generated}")
    logging.info(f"Combined Generated HRV shape: {combined_hrv.shape}")
    logging.info(f"Combined Generated EDA shape: {combined_eda.shape}")

    return {hrv_key: combined_hrv, eda_key: combined_eda}


def calculate_and_plot_feature_correlation(
    real_data: Dict[str, np.ndarray],
    gen_data: Dict[str, np.ndarray],
    output_dir: Optional[str] = None,
    condition_name: str = "condition", # Will be "all_conditions" when called from main
    model_dir_basename: str = "model"
) -> Dict[str, float]:
    """
    Calculates and plots correlation matrices for real and generated features.
    (Defined locally)

    Assumes input dictionaries contain 'HRV_features' and 'EDA_features'.
    Combines these features for correlation analysis.

    Args:
        real_data: Dictionary with matched real feature arrays.
        gen_data: Dictionary with matched generated feature arrays.
        output_dir: Directory to save the heatmap plots. If None, plots aren't saved.
        condition_name: Name used in plot filenames (e.g., "all_conditions").
        model_dir_basename: Base name of the model directory for plot filenames.

    Returns:
        Dictionary containing metrics comparing the correlation matrices (e.g., MAE).
    """
    correlation_metrics = {}
    hrv_key = 'HRV_features'
    eda_key = 'EDA_features'

    # Check if data is present and valid
    if hrv_key not in real_data or eda_key not in real_data or \
       real_data[hrv_key].size == 0 or real_data[eda_key].size == 0:
        logging.warning("Missing or empty real HRV or EDA features for correlation analysis. Skipping.")
        return correlation_metrics
    if hrv_key not in gen_data or eda_key not in gen_data or \
       gen_data[hrv_key].size == 0 or gen_data[eda_key].size == 0:
        logging.warning("Missing or empty generated HRV or EDA features for correlation analysis. Skipping.")
        return correlation_metrics

    try:
        # --- Combine Features ---
        real_hrv = real_data[hrv_key]
        real_eda = real_data[eda_key]
        gen_hrv = gen_data[hrv_key]
        gen_eda = gen_data[eda_key]

        # Ensure consistent number of samples after potential matching
        if real_hrv.shape[0] != real_eda.shape[0] or gen_hrv.shape[0] != gen_eda.shape[0] or real_hrv.shape[0] != gen_hrv.shape[0]:
             logging.error(f"Sample count mismatch after matching: Real HRV {real_hrv.shape[0]}, Real EDA {real_eda.shape[0]}, Gen HRV {gen_hrv.shape[0]}, Gen EDA {gen_eda.shape[0]}. Cannot proceed.")
             return correlation_metrics

        real_combined = np.hstack((real_hrv, real_eda))
        gen_combined = np.hstack((gen_hrv, gen_eda))

        num_hrv_feats = real_hrv.shape[1]
        num_eda_feats = real_eda.shape[1]

        # --- Define Actual Feature Names ---
        # Based on the user-provided snippet
        hrv_feature_names_expected = ['RMSSD', 'SDNN', 'LF', 'HF', 'LF_HF_ratio']
        eda_feature_names_expected = ['mean_EDA', 'median_EDA', 'SCR_count']

        # Check if the actual number of features matches the expected names
        if num_hrv_feats == len(hrv_feature_names_expected):
            hrv_feature_names = hrv_feature_names_expected
        else:
            logging.warning(f"Mismatch between expected HRV features ({len(hrv_feature_names_expected)}) and actual ({num_hrv_feats}). Using generic names.")
            hrv_feature_names = [f'HRV_{i}' for i in range(num_hrv_feats)] # Fallback

        if num_eda_feats == len(eda_feature_names_expected):
            eda_feature_names = eda_feature_names_expected
        else:
            logging.warning(f"Mismatch between expected EDA features ({len(eda_feature_names_expected)}) and actual ({num_eda_feats}). Using generic names.")
            eda_feature_names = [f'EDA_{i}' for i in range(num_eda_feats)] # Fallback

        feature_names = hrv_feature_names + eda_feature_names
        # --- End Feature Name Definition ---


        # --- Create Pandas DataFrames ---
        real_df = pd.DataFrame(real_combined, columns=feature_names)
        gen_df = pd.DataFrame(gen_combined, columns=feature_names)

        # --- Calculate Correlation Matrices ---
        real_corr = real_df.corr()
        gen_corr = gen_df.corr()

        # Handle potential NaN values in correlation matrices (e.g., if a feature has zero variance)
        real_corr = real_corr.fillna(0)
        gen_corr = gen_corr.fillna(0)

        # --- Calculate Difference Metric (e.g., Mean Absolute Error) ---
        # Ensure shapes match before calculating difference
        if real_corr.shape == gen_corr.shape:
            corr_mae = np.mean(np.abs(real_corr.values - gen_corr.values))
            correlation_metrics['Feature_Correlation_MAE'] = corr_mae
            logging.info(f"Feature Correlation Matrix MAE (vs Real): {corr_mae:.4f}")
        else:
             logging.warning(f"Correlation matrix shapes differ: Real={real_corr.shape}, Gen={gen_corr.shape}. Cannot calculate MAE.")


        # --- Plot Heatmaps ---
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plot_base_filename = f"correlation_heatmap_{model_dir_basename}_{condition_name}" # condition_name is 'all_conditions' here

            # Plot Real Data Correlation
            plt.figure(figsize=(12, 10)) # Adjusted size slightly
            sns.heatmap(real_corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1,
                        xticklabels=real_corr.columns, yticklabels=real_corr.columns) # Use DataFrame columns for labels
            plt.title(f"Real Feature Correlation Matrix ({condition_name.replace('_', ' ').title()})")
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            real_plot_path = os.path.join(output_dir, f"{plot_base_filename}_real.png")
            try:
                plt.savefig(real_plot_path, dpi=150) # Added dpi
                logging.info(f"Saved real correlation heatmap to: {real_plot_path}")
            except Exception as e:
                logging.error(f"Failed to save real correlation heatmap: {e}")
            plt.close() # Close the figure to free memory

            # Plot Generated Data Correlation
            plt.figure(figsize=(12, 10))
            sns.heatmap(gen_corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1,
                        xticklabels=gen_corr.columns, yticklabels=gen_corr.columns) # Use DataFrame columns for labels
            plt.title(f"Generated Feature Correlation Matrix ({condition_name.replace('_', ' ').title()})")
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            gen_plot_path = os.path.join(output_dir, f"{plot_base_filename}_generated.png")
            try:
                plt.savefig(gen_plot_path, dpi=150)
                logging.info(f"Saved generated correlation heatmap to: {gen_plot_path}")
            except Exception as e:
                logging.error(f"Failed to save generated correlation heatmap: {e}")
            plt.close() # Close the figure

        else:
            logging.info("output_dir not specified, skipping saving correlation plots.")

    except Exception as e:
        logging.error(f"Error during correlation calculation or plotting: {e}", exc_info=True)

    return correlation_metrics


# --- Main Analysis Function ---
def main(args):
    """Runs the feature correlation analysis pipeline across ALL conditions."""
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        logging.info("CUDA available, using GPU.")
    else:
        logging.info("CUDA not available, using CPU.")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load Model and Config ---
    try:
        logging.info(f"--- Loading Model from: {args.model_dir} ---")
        inferred_params = infer_parameters_from_dir(args.model_dir)

        # !!! CRITICAL CHECK: Ensure it's a features model !!!
        if inferred_params.get('model_type') != 'features':
            logging.error(f"Model type is '{inferred_params.get('model_type')}', "
                          f"but correlation analysis is only supported for 'features' models.")
            sys.exit(1)

        # --- FIX: Remove the line that forces signal_type to None ---
        # Let load_model use the signal_type inferred from the directory/checkpoint
        # inferred_params['signal_type'] = None # <-- REMOVED THIS LINE
        # --- End FIX ---

        # Ensure use_bayesian flag is present, default to False if missing
        if 'use_bayesian' not in inferred_params:
             logging.warning("Could not infer 'use_bayesian' parameter, assuming False.")
             inferred_params['use_bayesian'] = False
        # Ensure signal_type is present after inference (needed for load_model key)
        if 'signal_type' not in inferred_params or inferred_params['signal_type'] is None:
             # Attempt to infer from checkpoint name if not found in dir name
             try:
                  ckpt_name = os.path.basename(find_best_checkpoint(args.model_dir))
                  if 'EDA' in ckpt_name: inferred_params['signal_type'] = 'EDA'
                  elif 'BVP' in ckpt_name: inferred_params['signal_type'] = 'BVP'
                  else: raise ValueError("Cannot determine signal_type")
                  logging.info(f"Inferred signal_type '{inferred_params['signal_type']}' from checkpoint name.")
             except Exception as ckpt_e:
                  logging.error(f"Could not infer 'signal_type' parameter needed for model loading: {ckpt_e}")
                  sys.exit(1)


        best_checkpoint_path = find_best_checkpoint(args.model_dir)
        # Now load_model receives the full inferred_params, including the inferred signal_type
        model, config = load_model(
            checkpoint_path=best_checkpoint_path,
            **inferred_params # Pass inferred params directly
        )
        model.to(device)
        logging.info(f"Model loaded successfully. Type: features, Signal Origin: {inferred_params.get('signal_type', 'N/A')}")
        latent_dim = config.get('latent_dim', 100) # Get latent dim from loaded config or default
    except (FileNotFoundError, ValueError, RuntimeError, TypeError, AttributeError, ImportError, KeyError) as e:
        logging.error(f"Failed to load model: {e}", exc_info=True)
        sys.exit(1)

    # --- Prepare Real Data (All Conditions) ---
    try:
        logging.info(f"--- Loading Real Data (All Conditions) ---")
        # Use the local function to load data combined across conditions
        real_data_dict = prepare_real_data_all_conditions(
            args.test_data_path,
            inferred_params['model_type'] # 'features'
        )
        # Determine number of samples needed based on real data
        if real_data_dict['HRV_features'].size == 0:
             logging.error("No real feature data loaded. Exiting.")
             sys.exit(1)

        num_real_samples = real_data_dict['HRV_features'].shape[0]
        # Use the smaller of user-requested samples or available real samples
        num_gen_samples = args.num_gen_samples if args.num_gen_samples > 0 else num_real_samples
        logging.info(f"Total number of real samples across all conditions: {num_real_samples}")
        logging.info(f"Total number of generated samples to create: {num_gen_samples}")

    except (FileNotFoundError, ValueError, KeyError) as e:
        logging.error(f"Failed to load or prepare real feature data: {e}", exc_info=True)
        sys.exit(1)

    # --- Generate Synthetic Samples (All Conditions) ---
    try:
        logging.info(f"--- Generating Synthetic Feature Data (All Conditions) ---")
        # Use the local function to generate samples across conditions
        generated_data_dict = generate_samples_all_conditions(
            model=model,
            total_num_samples=num_gen_samples,
            latent_dim=latent_dim,
            device=device,
            model_type=inferred_params['model_type'], # 'features'
            is_bayesian=inferred_params['use_bayesian'] # Pass Bayesian flag
        )
        if generated_data_dict['HRV_features'].size == 0:
             logging.error("No synthetic feature data generated. Exiting.")
             sys.exit(1)

    except (ValueError, RuntimeError) as e:
        logging.error(f"Failed during sample generation: {e}", exc_info=True)
        sys.exit(1)

    # --- Match Sample Counts ---
    # This step ensures consistency if generation didn't produce exact numbers
    try:
        logging.info("--- Matching Sample Counts ---")
        real_hrv, gen_hrv = match_sample_counts(real_data_dict['HRV_features'], generated_data_dict['HRV_features'])
        real_eda, gen_eda = match_sample_counts(real_data_dict['EDA_features'], generated_data_dict['EDA_features'])
        real_data_matched = {'HRV_features': real_hrv, 'EDA_features': real_eda}
        gen_data_matched = {'HRV_features': gen_hrv, 'EDA_features': gen_eda}
        logging.info(f"Final matched sample count: {real_hrv.shape[0]}")

    except (KeyError, ValueError) as e:
        logging.error(f"Failed during sample count matching: {e}", exc_info=True)
        sys.exit(1)


    # --- Calculate and Plot Feature Correlations (Combined Data) ---
    logging.info("--- Calculating and Plotting Feature Correlation (All Conditions Combined) ---")
    model_dir_basename = os.path.basename(os.path.normpath(args.model_dir))
    condition_name = "all_conditions" # Use specific name for output

    # Use the local plotting function, passing the combined data
    correlation_metrics = calculate_and_plot_feature_correlation(
        real_data_matched,
        gen_data_matched,
        output_dir=args.output_dir, # Pass the output directory
        condition_name=condition_name, # Indicate combined analysis in filename
        model_dir_basename=model_dir_basename
    )

    # --- Save Correlation Metrics (Optional) ---
    if args.output_dir and correlation_metrics:
        os.makedirs(args.output_dir, exist_ok=True)
        # Update filename to reflect combined analysis
        results_filename = f"correlation_metrics_{model_dir_basename}_{condition_name}.json"
        results_path = os.path.join(args.output_dir, results_filename)
        try:
            # Convert numpy types to native Python types for JSON
            json_metrics = {k: (float(v) if isinstance(v, (np.number, np.floating)) else v)
                            for k, v in correlation_metrics.items()}
            with open(results_path, 'w') as f:
                json.dump(json_metrics, f, indent=4)
            logging.info(f"Correlation metrics saved to: {results_path}")
        except Exception as e:
            logging.error(f"Failed to save correlation metrics to JSON: {e}")
    elif not correlation_metrics:
         logging.warning("No correlation metrics were calculated.")


    logging.info("--- Combined Correlation Analysis Finished ---")


# --- Argument Parser ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze feature correlation for trained GAN models across ALL conditions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to the directory containing the 'best_*.pt' model checkpoint and 'config.json'. Model MUST be of type 'features'.")
    parser.add_argument("--test_data_path", type=str, required=True,
                        help="Path to the .pt file containing the real test dataset (with HRV_features and EDA_features).")
    parser.add_argument("--output_dir", type=str, default="correlation_analysis_results",
                        help="Directory to save the correlation heatmap plots and metrics JSON file.")
    # --- Condition argument removed ---
    parser.add_argument("--num_gen_samples", type=int, default=0,
                        help="Total number of samples to generate across all conditions. If 0, uses the total number of real samples found.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    parsed_args = parser.parse_args()
    main(parsed_args)
