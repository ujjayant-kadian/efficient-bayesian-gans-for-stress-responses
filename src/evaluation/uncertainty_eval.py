import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import os
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

# --- Constants ---
EPSILON = 1e-6 # Small value to prevent division by zero or log(0)

# --- Helper Functions ---

def _calculate_percentiles(
    generated_samples_array: np.ndarray,
    alpha: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates the lower and upper percentile bounds."""
    lower_bound = np.percentile(generated_samples_array, (alpha / 2) * 100, axis=0)
    upper_bound = np.percentile(generated_samples_array, (1 - alpha / 2) * 100, axis=0)
    return lower_bound, upper_bound

def _calculate_coverage_probability(
    real_data_array: np.ndarray,
    generated_samples_array: np.ndarray,
    alpha: float
) -> float:
    """
    Calculates the mean coverage probability across all dimensions/time points.

    Args:
        real_data_array: Ground truth data (N, D) or (N, T, C=1).
        generated_samples_array: Stacked generated samples (M, D) or (M, T, C=1).
                                  M is total samples, N is number of real points.
        alpha: Significance level for prediction interval (e.g., 0.1 for 90%).

    Returns:
        Mean coverage probability.
    """
    if real_data_array.shape[0] == 0 or generated_samples_array.shape[0] == 0:
        logging.warning("Cannot calculate coverage with empty real or generated data.")
        return np.nan

    # Calculate prediction intervals from generated samples
    lower_bound, upper_bound = _calculate_percentiles(generated_samples_array, alpha)
    # lower/upper_bound shape is (D,) or (T, C=1)

    # Check if real data points fall within the intervals
    # We compare each real point against the *overall* interval derived from all generated samples
    if real_data_array.ndim == 2: # Features (N, D)
        is_covered = (real_data_array >= lower_bound) & (real_data_array <= upper_bound)
        # is_covered shape is (N, D)
        mean_coverage = np.mean(is_covered) # Average over samples and features

    elif real_data_array.ndim == 3: # Time Series (N, T, C=1)
        # Ensure bounds have the same ndim for broadcasting
        lower_bound_exp = lower_bound[np.newaxis, :, :] # Shape (1, T, C=1)
        upper_bound_exp = upper_bound[np.newaxis, :, :] # Shape (1, T, C=1)
        is_covered = (real_data_array >= lower_bound_exp) & (real_data_array <= upper_bound_exp)
        # is_covered shape is (N, T, C=1)
        mean_coverage = np.mean(is_covered) # Average over samples, time steps, and channels
    else:
        logging.error(f"Unsupported real_data_array ndim: {real_data_array.ndim}")
        return np.nan

    return mean_coverage


def _calculate_negative_log_likelihood(
    real_data_array: np.ndarray,
    generated_samples_array: np.ndarray
) -> float:
    """
    Calculates the average Negative Log-Likelihood assuming a Gaussian distribution
    fitted to the generated samples.

    Args:
        real_data_array: Ground truth data (N, D) or (N, T, C=1).
        generated_samples_array: Stacked generated samples (M, D) or (M, T, C=1).

    Returns:
        Mean Negative Log-Likelihood.
    """
    if real_data_array.shape[0] == 0 or generated_samples_array.shape[0] == 0:
        logging.warning("Cannot calculate NLL with empty real or generated data.")
        return np.nan

    # Estimate mean and std dev from generated samples for each dimension/time point
    gen_mean = np.mean(generated_samples_array, axis=0) # Shape (D,) or (T, C=1)
    gen_std = np.std(generated_samples_array, axis=0)   # Shape (D,) or (T, C=1)
    gen_std = np.maximum(gen_std, EPSILON) # Add epsilon for numerical stability

    try:
        # Calculate log probability density of real data under the fitted Gaussian
        log_pdfs = norm.logpdf(real_data_array, loc=gen_mean, scale=gen_std)
        # log_pdfs shape is (N, D) or (N, T, C=1)

        # Average NLL over all samples and dimensions/time points
        mean_nll = -np.mean(log_pdfs)
    except Exception as e:
        logging.error(f"Error during NLL calculation: {e}", exc_info=True)
        return np.nan

    return mean_nll


def _plot_calibration_curve(
    real_data_array: np.ndarray,
    generated_samples_array: np.ndarray,
    data_key: str,
    save_path: Optional[str] = None
) -> Optional[str]:
    """
    Generates and optionally saves a calibration plot.

    Args:
        real_data_array: Ground truth data (N, D) or (N, T, C=1).
        generated_samples_array: Stacked generated samples (M, D) or (M, T, C=1).
        data_key: Name of the data type (e.g., 'EDA_series').
        save_path: Directory to save the plot. If None, plot is not saved.

    Returns:
        Path to the saved plot file, or None.
    """
    if real_data_array.shape[0] == 0 or generated_samples_array.shape[0] == 0:
        logging.warning(f"Skipping calibration plot for {data_key} due to empty data.")
        return None

    expected_coverages = np.linspace(0.05, 0.95, 19) # e.g., 5%, 10%, ..., 95%
    observed_coverages = []

    for p in expected_coverages:
        alpha = 1.0 - p
        observed_coverage = _calculate_coverage_probability(
            real_data_array, generated_samples_array, alpha
        )
        observed_coverages.append(observed_coverage)

    # --- Plotting ---
    plt.figure(figsize=(6, 6))
    plt.plot(expected_coverages, observed_coverages, 'o-', label='Model Calibration')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration (y=x)')
    plt.xlabel("Expected Coverage Probability (1 - alpha)")
    plt.ylabel("Observed Coverage Probability")
    plt.title(f"Calibration Curve - {data_key}")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_filename = None
    if save_path:
        try:
            base_filename = f"calibration_{data_key}.png"
            counter = 1
            plot_filename = os.path.join(save_path, base_filename)
            
            # Check if file exists and increment counter until finding unused name
            while os.path.exists(plot_filename):
                base_filename = f"calibration_{data_key}_{counter}.png"
                plot_filename = os.path.join(save_path, base_filename)
                counter += 1
                
            plt.savefig(plot_filename)
            logging.info(f"Calibration plot saved to: {plot_filename}")
        except Exception as e:
            logging.error(f"Failed to save calibration plot to {plot_filename}: {e}")
            plot_filename = None # Indicate failure
        finally:
             plt.close() # Close the figure to free memory
    else:
        # If not saving, maybe show? Or just close. For automated scripts, closing is better.
        plt.close()

    return plot_filename


# --- Main Uncertainty Evaluation Function ---

def uncertainty_evaluation_summary(
    real_data: Dict[str, np.ndarray],
    generated_samples: Dict[str, List[np.ndarray]],
    feature_type: str,
    alpha: float = 0.1, # Default to 90% prediction intervals
    save_path: Optional[str] = None,
    sample_strategy: Optional[str] = None # Currently unused, data assumed matched
) -> Dict[str, Any]:
    """
    Calculates uncertainty metrics (coverage, NLL) and generates calibration plots
    for Bayesian model outputs.

    Args:
        real_data: Dictionary containing ground truth data arrays (matched count).
                   Keys like 'EDA_series', 'HRV_features'.
                   Values are NumPy arrays (N, ...) where N is sample count.
        generated_samples: Dictionary containing generated uncertainty samples.
                           Keys match real_data.
                           Values are lists of NumPy arrays [sample1, sample2, ...].
                           The total number of samples across the list for a key
                           should ideally match N in real_data for that key,
                           but we stack them anyway.
        feature_type: Type of data ('time_series' or 'features').
        alpha: Significance level for prediction intervals (e.g., 0.1 for 90%).
        save_path: Optional directory path to save calibration plots.
        sample_strategy: (Currently unused) Intended for future sampling logic.

    Returns:
        Dictionary containing calculated uncertainty metrics:
        {
            'coverage': { 'data_key': {'mean_coverage': float, 'expected_coverage': float}, ... },
            'negative_log_likelihood': { 'data_key': float, ... },
            'calibration_plot_paths': { 'data_key': str_or_None, ... }
        }
    """
    results = {
        'coverage': {},
        'negative_log_likelihood': {},
        'calibration_plot_paths': {}
    }
    expected_coverage_level = 1.0 - alpha

    data_keys = []
    if feature_type == 'time_series':
        data_keys = [k for k in ['EDA_series', 'BVP_series'] if k in real_data and k in generated_samples]
    elif feature_type == 'features':
        data_keys = [k for k in ['HRV_features', 'EDA_features'] if k in real_data and k in generated_samples]
    else:
        logging.error(f"Unsupported feature_type for uncertainty evaluation: {feature_type}")
        return results # Return empty results

    if not data_keys:
        logging.warning("No matching data keys found between real_data and generated_samples for uncertainty evaluation.")
        return results

    for key in data_keys:
        logging.info(f"--- Evaluating Uncertainty for: {key} ---")
        real_arr = real_data.get(key)
        gen_samples_list = generated_samples.get(key)

        if real_arr is None or gen_samples_list is None or len(gen_samples_list) == 0:
            logging.warning(f"Skipping uncertainty evaluation for '{key}' due to missing or empty data.")
            continue

        try:
            # Stack the list of generated samples into a single array
            # Assumes all samples in the list have the same shape
            gen_arr_stacked = np.stack(gen_samples_list, axis=0) # Shape (M, D) or (M, T, C=1)
            logging.info(f"Stacked generated samples shape for {key}: {gen_arr_stacked.shape}")
            logging.info(f"Real data shape for {key}: {real_arr.shape}")

            # 1. Calculate Coverage Probability
            mean_coverage = _calculate_coverage_probability(real_arr, gen_arr_stacked, alpha)
            results['coverage'][key] = {
                'mean_coverage': mean_coverage,
                'expected_coverage': expected_coverage_level
            }
            logging.info(f"[{key}] Mean Coverage: {mean_coverage:.4f} (Expected: {expected_coverage_level:.4f})")

            # 2. Calculate Negative Log-Likelihood
            nll = _calculate_negative_log_likelihood(real_arr, gen_arr_stacked)
            results['negative_log_likelihood'][key] = nll
            logging.info(f"[{key}] Negative Log-Likelihood: {nll:.4f}")

            # 3. Generate and Save Calibration Plot
            plot_path = _plot_calibration_curve(real_arr, gen_arr_stacked, key, save_path)
            results['calibration_plot_paths'][key] = plot_path

        except ValueError as ve:
             logging.error(f"ValueError during uncertainty evaluation for {key}: {ve}. "
                           "This might be due to inconsistent shapes in generated samples list.", exc_info=True)
        except Exception as e:
            logging.error(f"Unexpected error during uncertainty evaluation for {key}: {e}", exc_info=True)
            # Add placeholders to indicate failure for this key
            results['coverage'][key] = {'mean_coverage': np.nan, 'expected_coverage': expected_coverage_level}
            results['negative_log_likelihood'][key] = np.nan
            results['calibration_plot_paths'][key] = None

    return results