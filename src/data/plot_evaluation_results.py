# Import necessary libraries
import os
import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from sklearn.preprocessing import minmax_scale
import logging # Import logging library

# --- Configuration ---

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Define the directory containing the JSON files
data_dir = 'evaluation_results'
plots_output_dir = 'plots' # Define base output directory for plots
radar_output_subdir = 'radar_charts'
bar_output_subdir = 'bar_charts_uncertainty'

# Define the metrics for comparison
non_bayesian_metrics_template = {
    'RMSE': '{prefix}_RMSE',
    'MMD': '{prefix}_MMD',
    'Wasserstein': '{prefix}_Wasserstein',
    'FID-like': '{prefix}_FID_like',
    # Classifier accuracy key varies, handle separately in extraction
}
# REVISED uncertainty template
uncertainty_metrics_template = {
    # Use placeholders for top-level key part and sub-key part
    'Coverage': '{top_prefix}_Uncertainty.coverage.{sub_prefix}.mean_coverage',
    'NLL': '{top_prefix}_Uncertainty.negative_log_likelihood.{sub_prefix}',
}

# Metrics where lower is better (for normalization inversion)
# Note: Coverage error is handled separately in normalization, NLL is lower is better.
# ADDED 'Acc' to treat Classifier Accuracy as lower-is-better in this context
lower_is_better = ['RMSE', 'MMD', 'Wasserstein', 'FID-like', 'NLL', 'Acc']

# Target coverage for uncertainty metric
target_coverage = 0.9

# Define model types for plots
models_non_bayesian = ['Baseline', 'Dropout', 'Var-Gaussian'] # Compare these in non-Bayesian plots
models_uncertainty = ['Dropout', 'Var-Gaussian', 'Var-Laplace', 'Var-ScaledMix'] # Compare these in uncertainty plots

# Define data types/prefixes to iterate over
data_prefixes = {
    'EDA_series': 'EDA_series',
    'BVP_series': 'BVP_series',
    'HRV_features': 'HRV', # Prefix used in HRV metrics keys (for non-Bayesian)
    'EDA_features': 'EDA_Features' # Prefix used in EDA feature metrics keys (for non-Bayesian)
}
# Define specific sub-prefixes needed for uncertainty keys
uncertainty_sub_prefixes = {
    'HRV': 'HRV_features',
    'EDA': 'EDA_features',
    'EDA_series': 'EDA_series',
    'BVP_series': 'BVP_series'
}

# Special handling for combined features classifier accuracy key
features_classifier_key = 'Features_Combined_Classifier_Mean_Accuracy'
# Top prefix for features uncertainty
features_uncertainty_top_prefix = 'Features'

# --- Helper Functions ---

def parse_model_details(model_string):
    """Parses the model part of the filename (e.g., 'baseline-v1-features')"""
    model_type = 'Unknown'
    subtype = None

    if 'baseline' in model_string:
        model_type = 'Baseline'
    elif 'dropout' in model_string:
        model_type = 'Dropout'
    elif 'var-bayes' in model_string:
        model_type = 'Variational'
        if 'gaussian' in model_string:
             subtype = 'Gaussian'
        elif 'laplace' in model_string:
             subtype = 'Laplace'
        elif 'scaled-mixture' in model_string:
             subtype = 'ScaledMix'
        else:
             subtype = 'Gaussian' # Default assumption
    else:
        parts = model_string.split('-')
        if parts: model_type = parts[0].capitalize()

    full_model_name = f"Var-{subtype}" if model_type == 'Variational' and subtype else model_type
    return model_type, subtype, full_model_name


def parse_filename(filename):
    """Parses the filename to extract metadata using updated regex."""
    base_filename = os.path.basename(filename)
    features_match = re.match(r"metrics_([a-z0-9\-]+)_features_(Baseline|Stress)\.json", base_filename)
    if features_match:
        model_string, condition = features_match.groups()
        model_type, subtype, full_model_name = parse_model_details(model_string)
        return {'model_type': model_type, 'subtype': subtype, 'full_model_name': full_model_name,
                'data_category': 'features', 'series_feature_type': 'features',
                'data_type_key': 'Features', 'condition': condition, 'filename': filename}
    timeseries_match = re.match(r"metrics_([a-z0-9\-]+)_(time_series)_([A-Z]+)_(Baseline|Stress)\.json", base_filename)
    if timeseries_match:
        model_string, data_category, series_type, condition = timeseries_match.groups()
        model_type, subtype, full_model_name = parse_model_details(model_string)
        data_type_key = f"{series_type}_series"
        return {'model_type': model_type, 'subtype': subtype, 'full_model_name': full_model_name,
                'data_category': data_category, 'series_feature_type': series_type,
                'data_type_key': data_type_key, 'condition': condition, 'filename': filename}
    logging.warning(f"Could not parse filename structure: {base_filename}")
    return None

def safe_get(data, key_path, default=np.nan):
    """Safely gets a value from nested dictionary using dot notation path."""
    keys = key_path.split('.')
    value = data
    try:
        for key in keys:
            if isinstance(value, dict): value = value[key]
            else: return default
        if isinstance(value, list): return default
        return float(value) if value is not None else default
    except (KeyError, TypeError, IndexError): return default

def load_data(data_dir):
    """Loads all JSON files from the directory and parses them."""
    all_data = []
    if not os.path.isdir(data_dir):
        logging.error(f"Directory not found - {data_dir}")
        return pd.DataFrame()
    logging.info(f"Loading data from directory: {data_dir}")
    for filename in os.listdir(data_dir):
        if filename.endswith(".json") and filename.startswith("metrics_"):
            filepath = os.path.join(data_dir, filename)
            metadata = parse_filename(filepath)
            if metadata:
                try:
                    with open(filepath, 'r') as f:
                        metrics_data = json.load(f)
                        record = metadata.copy()
                        record['metrics_content'] = metrics_data
                        all_data.append(record)
                except json.JSONDecodeError: logging.warning(f"Could not decode JSON from {filename}")
                except Exception as e: logging.warning(f"Error processing file {filename}: {e}")
    df = pd.DataFrame(all_data)
    logging.info(f"Loaded data for {len(df)} experiments.")
    return df

def extract_metrics(df, data_type_key, metrics_template, is_uncertainty=False):
    """Extracts specific metrics for a given data type, handling dynamic keys."""
    extracted = pd.DataFrame(index=df.index)
    is_features = (data_type_key == 'Features')

    for metric_name_short, key_template in metrics_template.items():
        if is_features:
            # Handle Features (HRV and EDA)
            if is_uncertainty:
                top_prefix = features_uncertainty_top_prefix
                hrv_sub_prefix = uncertainty_sub_prefixes['HRV']
                eda_sub_prefix = uncertainty_sub_prefixes['EDA']
                hrv_key = key_template.format(top_prefix=top_prefix, sub_prefix=hrv_sub_prefix)
                eda_key = key_template.format(top_prefix=top_prefix, sub_prefix=eda_sub_prefix)
                # Use metric_name_short to create unique column names
                extracted[f'HRV_{metric_name_short}'] = df['metrics_content'].apply(lambda data: safe_get(data, hrv_key))
                extracted[f'EDA_{metric_name_short}'] = df['metrics_content'].apply(lambda data: safe_get(data, eda_key))
            else: # Non-Bayesian Features
                hrv_prefix = data_prefixes['HRV_features']
                eda_prefix = data_prefixes['EDA_features']
                hrv_key = key_template.format(prefix=hrv_prefix)
                eda_key = key_template.format(prefix=eda_prefix)
                extracted[f'HRV_{metric_name_short}'] = df['metrics_content'].apply(lambda data: safe_get(data, hrv_key))
                extracted[f'EDA_{metric_name_short}'] = df['metrics_content'].apply(lambda data: safe_get(data, eda_key))
        else:
            # Handle Time Series (EDA_series, BVP_series)
            if is_uncertainty:
                top_prefix = data_prefixes.get(data_type_key)
                sub_prefix = uncertainty_sub_prefixes.get(data_type_key)
                if top_prefix and sub_prefix:
                    key = key_template.format(top_prefix=top_prefix, sub_prefix=sub_prefix)
                    extracted[metric_name_short] = df['metrics_content'].apply(lambda data: safe_get(data, key))
                else:
                    logging.warning(f"Missing prefix info for uncertainty metric '{metric_name_short}' with data_key '{data_type_key}'")
            else: # Non-Bayesian Time Series
                prefix = data_prefixes.get(data_type_key)
                if prefix:
                    key = key_template.format(prefix=prefix)
                    extracted[metric_name_short] = df['metrics_content'].apply(lambda data: safe_get(data, key))
                else:
                     logging.warning(f"Unknown prefix for data_type_key: {data_type_key} in extract_metrics (non-Bayesian)")

     # Handle Classifier Accuracy separately (only for non-Bayesian extraction)
    if not is_uncertainty:
        if data_type_key == 'Features':
             extracted['Classifier_Acc'] = df['metrics_content'].apply(lambda data: safe_get(data, features_classifier_key))
        elif data_type_key in data_prefixes:
             prefix = data_prefixes[data_type_key]
             clf_key = f"{prefix}_Classifier_Mean_Accuracy"
             extracted['Classifier_Acc'] = df['metrics_content'].apply(lambda data: safe_get(data, clf_key))

    return extracted

def normalize_metrics(plot_data, metrics_present, title):
    """Normalizes metrics, handling lower-is-better and coverage."""
    normalized_data = plot_data.copy()
    logging.debug(f"--- Normalizing metrics for: {title} ---")
    for metric in metrics_present:
        # Skip normalization if column doesn't exist (shouldn't happen with metrics_present)
        if metric not in plot_data.columns: continue

        metric_std_dev = plot_data[metric].std()
        logging.debug(f"Metric: {metric}, StdDev: {metric_std_dev:.4f}")

        # Handle constant columns
        if metric_std_dev == 0 or pd.isna(metric_std_dev):
             # If all values are the same, scale to 0.5 (neutral)
             # Check if the constant value itself is NaN first
             if plot_data[metric].isnull().all():
                  scaled_values = np.full(plot_data[metric].shape, np.nan) # Keep NaNs if all are NaN
                  logging.warning(f"Plot '{title}': Metric '{metric}' is all NaN. Keeping NaN.")
             else:
                  scaled_values = np.full(plot_data[metric].shape, 0.5)
                  logging.warning(f"Plot '{title}': Metric '{metric}' has zero variance. Norm -> 0.5.")
        else:
             # MinMax scale across the models for this metric
             scaled_values = minmax_scale(plot_data[metric])
             logging.debug(f" Original: {plot_data[metric].values}")
             logging.debug(f" Scaled:   {scaled_values}")

        # Check if lower is better based on the metric name containing a keyword from lower_is_better
        is_lower_better = any(lb_name.lower() in metric.lower() for lb_name in lower_is_better)

        if 'Coverage' in metric: # Handle Coverage specifically
            # Calculate absolute error from target coverage
            error = np.abs(plot_data[metric] - target_coverage)
            error_std_dev = error.std()
            logging.debug(f" Coverage Error: {error.values}, StdDev: {error_std_dev:.4f}")
            # Scale the error (0 is best)
            if error_std_dev == 0 or pd.isna(error_std_dev):
                 # If error is constant, assign max score (1.0) if error is 0, else min score (0.0)
                 # Check first non-NaN error value
                 first_error = error.dropna().iloc[0] if not error.dropna().empty else 0
                 scaled_error = np.full(error.shape, 0.0) if np.isclose(first_error, 0) else np.full(error.shape, 1.0)
                 logging.warning(f"Plot '{title}': Coverage error for '{metric}' has zero variance. Scaled error -> {scaled_error[0]:.1f}.")
            else:
                 scaled_error = minmax_scale(error)
            # Invert so 0 error -> 1.0 score (best performance)
            normalized_data[metric] = 1.0 - scaled_error
            logging.debug(f" Coverage Norm (Inverted Error): {normalized_data[metric].values}")
        elif is_lower_better:
            # Invert the scale: 0 becomes 1 (best), 1 becomes 0 (worst)
            normalized_data[metric] = 1.0 - scaled_values
            logging.debug(f" Norm (Inverted): {normalized_data[metric].values}")
        else: # Higher is better
            normalized_data[metric] = scaled_values
            logging.debug(f" Norm (Direct):   {normalized_data[metric].values}")

    return normalized_data


def plot_radar_chart(data, metrics, models_to_plot, title, output_dir):
    """Creates a radar chart for the given data and metrics."""
    logging.info(f"Attempting to generate RADAR plot: '{title}' for models {models_to_plot}")

    if 'full_model_name' not in data.columns: logging.error(f"Plot '{title}': 'full_model_name' column missing."); return
    plot_data_filtered_models = data[data['full_model_name'].isin(models_to_plot)].copy()
    if plot_data_filtered_models.empty: logging.warning(f"Skipping plot '{title}': No data found for models {models_to_plot}."); return
    plot_data = plot_data_filtered_models.set_index('full_model_name')

    metrics_present = [m for m in metrics if m in plot_data.columns]
    if not metrics_present: logging.warning(f"Skipping plot '{title}': None of the specified metrics ({metrics}) found."); return
    plot_data = plot_data[metrics_present]
    plot_data.dropna(axis=0, how='all', inplace=True)
    if plot_data.empty: logging.warning(f"Skipping plot '{title}': No data left after dropping rows with all NaNs."); return

    # --- LOGGING: Print metrics before normalization ---
    logging.info(f"--- Metrics for Plot: {title} (Before Normalization) ---")
    try: logging.info("\n" + plot_data.to_string())
    except Exception as e: logging.error(f"Error converting metrics DF to string: {e}"); logging.info(plot_data)

    # --- Fill NaNs before normalization ---
    metrics_filled = []
    for col in plot_data.columns:
         if plot_data[col].isnull().any():
              mean_val = plot_data[col].mean()
              if pd.isna(mean_val): mean_val = 0; logging.warning(f"Plot '{title}': All values for metric '{col}' are NaN. Filling with 0.")
              else: logging.warning(f"Plot '{title}': Filling NaN in metric '{col}' with mean value {mean_val:.3f}")
              plot_data[col].fillna(mean_val, inplace=True)
              metrics_filled.append(col)

    num_vars = len(metrics_present)
    if num_vars < 3: # Radar charts need >= 3 axes
        logging.warning(f"Skipping RADAR plot '{title}': Need >= 3 metrics, found {num_vars}.")
        return

    # --- Normalization ---
    normalized_data = normalize_metrics(plot_data, metrics_present, title)

    # --- LOGGING: Print metrics AFTER normalization ---
    logging.info(f"--- Metrics for Plot: {title} (After Normalization) ---")
    try: logging.info("\n" + normalized_data.to_string())
    except Exception as e: logging.error(f"Error converting normalized metrics DF to string: {e}"); logging.info(normalized_data)

    # --- Plotting ---
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    tick_labels = metrics_present
    plt.xticks(angles[:-1], tick_labels, size=9)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels([f"{i:.1f}" for i in np.linspace(0, 1, 6)])
    ax.set_ylim(0, 1.05)

    for i, model_name in enumerate(normalized_data.index):
        values = normalized_data.loc[model_name].values.flatten().tolist()
        values += values[:1]
        line, = ax.plot(angles, values, linewidth=2, linestyle='solid', label=model_name)
        ax.fill(angles, values, color=line.get_color(), alpha=0.2)

    legend = plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize='small')
    plt.title(title, size=13, y=1.18)
    fig.subplots_adjust(top=0.85, right=0.8)

    # --- Saving Plot ---
    try:
        output_path = os.path.join(plots_output_dir, radar_output_subdir)
        os.makedirs(output_path, exist_ok=True)
        safe_filename = re.sub(r'[^\w\-]+', '_', title.lower()) + ".png"
        filepath = os.path.join(output_path, safe_filename)
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        logging.info(f"Saved plot: {filepath}")
    except Exception as e: logging.error(f"Failed to save plot '{title}' to {filepath}: {e}")
    finally: plt.close(fig)


def plot_bar_chart(data, metrics, models_to_plot, title, output_dir):
    """Creates a grouped bar chart for the given data and metrics."""
    logging.info(f"Attempting to generate BAR plot: '{title}' for models {models_to_plot}")

    if 'full_model_name' not in data.columns: logging.error(f"Plot '{title}': 'full_model_name' column missing."); return
    plot_data_filtered_models = data[data['full_model_name'].isin(models_to_plot)].copy()
    if plot_data_filtered_models.empty: logging.warning(f"Skipping plot '{title}': No data found for models {models_to_plot}."); return
    plot_data = plot_data_filtered_models.set_index('full_model_name')

    metrics_present = [m for m in metrics if m in plot_data.columns]
    if not metrics_present: logging.warning(f"Skipping plot '{title}': None of the specified metrics ({metrics}) found."); return
    plot_data = plot_data[metrics_present]
    plot_data.dropna(axis=0, how='all', inplace=True)
    if plot_data.empty: logging.warning(f"Skipping plot '{title}': No data left after dropping rows with all NaNs."); return

    # --- LOGGING: Print metrics before normalization ---
    logging.info(f"--- Metrics for Plot: {title} (Before Normalization) ---")
    try: logging.info("\n" + plot_data.to_string())
    except Exception as e: logging.error(f"Error converting metrics DF to string: {e}"); logging.info(plot_data)

    # --- Fill NaNs before normalization ---
    metrics_filled = []
    for col in plot_data.columns:
         if plot_data[col].isnull().any():
              mean_val = plot_data[col].mean()
              if pd.isna(mean_val): mean_val = 0; logging.warning(f"Plot '{title}': All values for metric '{col}' are NaN. Filling with 0.")
              else: logging.warning(f"Plot '{title}': Filling NaN in metric '{col}' with mean value {mean_val:.3f}")
              plot_data[col].fillna(mean_val, inplace=True)
              metrics_filled.append(col)

    # --- Normalization (using the same helper function) ---
    normalized_data = normalize_metrics(plot_data, metrics_present, title)

    # --- LOGGING: Print metrics AFTER normalization ---
    logging.info(f"--- Metrics for Plot: {title} (After Normalization) ---")
    try: logging.info("\n" + normalized_data.to_string())
    except Exception as e: logging.error(f"Error converting normalized metrics DF to string: {e}"); logging.info(normalized_data)

    # --- Plotting ---
    num_metrics = len(metrics_present)
    num_models = len(normalized_data.index)
    bar_width = 0.8 / num_models # Adjust bar width based on number of models
    index = np.arange(num_metrics) # x locations for the groups

    fig, ax = plt.subplots(figsize=(max(6, num_metrics * num_models * 0.5), 6)) # Adjust figure width

    for i, model_name in enumerate(normalized_data.index):
        # Calculate position for each model's bar within the group
        bar_positions = index + i * bar_width - (bar_width * (num_models -1) / 2)
        values = normalized_data.loc[model_name].values
        ax.bar(bar_positions, values, bar_width, label=model_name)

    ax.set_ylabel('Normalized Performance (Higher is Better)')
    ax.set_title(title, size=13)
    ax.set_xticks(index)
    ax.set_xticklabels(metrics_present, rotation=15, ha='right') # Rotate labels if needed
    ax.set_ylim(0, 1.1) # Y-axis from 0 to 1 (with slight margin)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0)) # Adjust legend position
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout to prevent legend overlap

    # --- Saving Plot ---
    try:
        output_path = os.path.join(plots_output_dir, bar_output_subdir)
        os.makedirs(output_path, exist_ok=True)
        safe_filename = re.sub(r'[^\w\-]+', '_', title.lower()) + ".png"
        filepath = os.path.join(output_path, safe_filename)
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        logging.info(f"Saved plot: {filepath}")
    except Exception as e: logging.error(f"Failed to save plot '{title}' to {filepath}: {e}")
    finally: plt.close(fig)



# --- Main Execution ---
if __name__ == "__main__": # Ensure code runs only when script is executed directly
    df_raw = load_data(data_dir)

    if df_raw.empty:
        logging.error("No data loaded. Exiting.")
    else:
        if 'full_model_name' not in df_raw.columns:
             logging.error("'full_model_name' column not created. Check parse_filename.")
        else:
             all_data_keys = set(df_raw['data_type_key'].unique())
             all_data_keys = {k for k in all_data_keys if pd.notna(k)}

             for data_key in sorted(list(all_data_keys)): # Sort for consistent order
                 for condition in ['Baseline', 'Stress']:
                     logging.info(f"--- Processing context: {data_key} - {condition} ---")
                     df_context = df_raw[(df_raw['data_type_key'] == data_key) & (df_raw['condition'] == condition)].copy()
                     if df_context.empty: logging.info(f"No data found. Skipping."); continue
                     if 'metrics_content' not in df_context.columns: logging.error(f"'metrics_content' missing."); continue

                     # --- Non-Bayesian Plot (Radar) ---
                     logging.debug(f"Extracting Non-Bayesian metrics")
                     metrics_nb_extracted = extract_metrics(df_context, data_key, non_bayesian_metrics_template, is_uncertainty=False)
                     if 'Classifier_Acc' not in metrics_nb_extracted.columns and data_key != 'Features' and data_key in data_prefixes: logging.warning(f"Classifier_Acc column missing after extraction for {data_key}")
                     df_plot_nb = pd.concat([df_context[['full_model_name']].reset_index(drop=True), metrics_nb_extracted.reset_index(drop=True)], axis=1)
                     current_nb_metrics_plot = []
                     if data_key == 'Features':
                          for m_short in non_bayesian_metrics_template.keys(): current_nb_metrics_plot.extend([f'HRV_{m_short}', f'EDA_{m_short}'])
                          current_nb_metrics_plot.append('Classifier_Acc')
                     elif data_key in data_prefixes: current_nb_metrics_plot.extend(list(non_bayesian_metrics_template.keys()) + ['Classifier_Acc'])
                     current_nb_metrics_plot = [m for m in current_nb_metrics_plot if m in df_plot_nb.columns and df_plot_nb[m].notna().any()]
                     logging.debug(f"Selected Non-Bayesian metrics: {current_nb_metrics_plot}")
                     if current_nb_metrics_plot:
                          # Still use radar for non-bayesian if >= 3 metrics
                          if len(current_nb_metrics_plot) >= 3:
                              plot_radar_chart(df_plot_nb, metrics=current_nb_metrics_plot, models_to_plot=models_non_bayesian,
                                               title=f"Non-Bayesian Metrics {data_key} {condition}", output_dir=plots_output_dir)
                          else:
                              logging.warning(f"Skipping Non-Bayesian RADAR plot for {data_key} {condition}: Found only {len(current_nb_metrics_plot)} metrics.")
                              # Optionally call plot_bar_chart here too if desired for < 3 metrics
                     else: logging.warning(f"Skipping Non-Bayesian plot: No valid metrics.")

                     # --- Uncertainty Plot (Bar Chart) ---
                     bayesian_models_present = df_context[df_context['model_type'].isin(['Dropout', 'Variational'])]
                     if not bayesian_models_present.empty:
                         logging.debug(f"Extracting Uncertainty metrics")
                         metrics_unc_extracted = extract_metrics(df_context, data_key, uncertainty_metrics_template, is_uncertainty=True)
                         df_plot_unc = pd.concat([df_context[['full_model_name']].reset_index(drop=True), metrics_unc_extracted.reset_index(drop=True)], axis=1)
                         current_unc_metrics_plot = []
                         if data_key == 'Features':
                             for m_short in uncertainty_metrics_template.keys(): current_unc_metrics_plot.extend([f'HRV_{m_short}', f'EDA_{m_short}'])
                         elif data_key in data_prefixes: current_unc_metrics_plot.extend(list(uncertainty_metrics_template.keys()))
                         current_unc_metrics_plot = [m for m in current_unc_metrics_plot if m in df_plot_unc.columns and df_plot_unc[m].notna().any()]
                         logging.debug(f"Selected Uncertainty metrics: {current_unc_metrics_plot}")
                         df_plot_unc_valid = df_plot_unc.dropna(subset=current_unc_metrics_plot, how='all').copy()
                         if not df_plot_unc_valid.empty:
                            valid_uncertainty_models = df_plot_unc_valid['full_model_name'].unique()
                            models_to_plot_unc_final = [m for m in models_uncertainty if m in valid_uncertainty_models]
                            logging.debug(f"Valid uncertainty models: {models_to_plot_unc_final}")
                            # Use Bar Chart for Uncertainty Metrics
                            if current_unc_metrics_plot and models_to_plot_unc_final:
                                 plot_bar_chart(df_plot_unc_valid, metrics=current_unc_metrics_plot, models_to_plot=models_to_plot_unc_final,
                                                  title=f"Uncertainty Metrics {data_key} {condition}", output_dir=plots_output_dir)
                            else: logging.warning(f"Skipping Uncertainty plot: No valid metrics or models.")
                         else: logging.info(f"Skipping Uncertainty plot: No models with valid uncertainty data.")
                     else: logging.info(f"Skipping Uncertainty plot: No Bayesian models found.")

        print("\n--- Plot generation finished ---")