import os
import pickle
import numpy as np
import pandas as pd
import scipy.signal as scisig
import torch
from pathlib import Path

# =============================================================================
#                          CONFIGURATIONS
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Sampling Frequencies
fs_dict = {
    'BVP': 64,   # Hz
    'EDA': 4,    # Hz
    'label': 700 # Hz (WESAD label annotation frequency)
}
WINDOW_IN_SECONDS = 30
# Overlap fraction (e.g., 0.5 = 50% overlap)
OVERLAP_FRAC = 0.5

DATA_RAW = PROJECT_ROOT / "data/raw"
DATA_PROCESSED = PROJECT_ROOT / "data/processed"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

# =============================================================================
#                        HELPER FUNCTIONS
# =============================================================================

def butter_lowpass_filter(data, cutoff, fs, order=5):
    """
    Applies a Butterworth low-pass filter to the input signal.
    Args:
        data (array-like): The 1D signal to be filtered.
        cutoff (float): Cutoff frequency in Hz.
        fs (float): Sampling frequency of the data in Hz.
        order (int): The order of the Butterworth filter.
    Returns:
        filtered_data (array-like): The filtered signal.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scisig.butter(order, normal_cutoff, btype='low', analog=False)
    return scisig.lfilter(b, a, data)

def remove_outliers(signal, threshold=3):
    """
    Clamps signal values exceeding ±(threshold * std).
    Args:
        signal (array-like): The signal to process.
        threshold (float): The number of standard deviations to use as cutoff.
    Returns:
        clipped (array-like): The signal after clipping outliers.
    """
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    lower_bound = mean_val - threshold * std_val
    upper_bound = mean_val + threshold * std_val
    clipped = np.clip(signal, lower_bound, upper_bound)
    return clipped

def zscore_normalize(signal):
    """
    Performs Z-score normalization on the input signal.
    Args:
        signal (array-like): The 1D signal to be normalized.
    Returns:
        norm_signal (array-like): Z-score normalized signal.
    """
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    if std_val < 1e-12:
        # Avoid division by zero if the signal is constant
        return signal
    return (signal - mean_val) / std_val

def extract_rr_and_time_domain(bvp_signal, fs=64):
    """
    Finds peaks in the BVP signal, computes RR intervals, and calculates 
    time-domain HRV metrics such as RMSSD and SDNN.
    Args:
        bvp_signal (array-like): BVP signal data.
        fs (float): Sampling frequency for BVP.
    Returns:
        rr_intervals (np.ndarray): Array of RR intervals in seconds.
        time_domain_metrics (dict): RMSSD and SDNN values.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Approximate minimum distance between peaks ~ 0.6s
    peaks, _ = scisig.find_peaks(bvp_signal, distance=int(fs * 0.6))

    rr_intervals = torch.diff(torch.tensor(peaks, device=device)) / fs  # in seconds
    if len(rr_intervals) > 1:
        rmssd = torch.sqrt(torch.mean(torch.diff(rr_intervals) ** 2)).cpu().item()
        sdnn = torch.std(rr_intervals).cpu().item()
    else:
        rmssd, sdnn = np.nan, np.nan
    
    return rr_intervals.cpu().numpy(), {"RMSSD": rmssd, "SDNN": sdnn}

def extract_freq_domain_hrv(rr_intervals):
    """
    Uses Welch's method to compute low-frequency (LF) and high-frequency (HF) power
    of RR intervals (resampled at ~4 Hz recommended). Then computes LF/HF ratio.
    Args:
        rr_intervals (np.ndarray): Array of RR intervals in seconds.
    Returns:
        freq_domain_metrics (dict): LF power, HF power, and LF/HF ratio.
    """
    if len(rr_intervals) < 4:
        # Not enough data for frequency analysis
        return {"LF": np.nan, "HF": np.nan, "LF_HF_ratio": np.nan}
    
    # 1) Convert RR intervals to an approximate tachogram (heart rate time series)
    #    or an evenly sampled signal. Let's do a simple approach:
    #    - Create time stamps for each R-peak in seconds
    times = np.cumsum(rr_intervals)
    #    - Resample at 4 Hz (or 2 Hz) to get an even sampling
    fs_resampled = 4.0
    if times[-1] < 1.0:
        # If total duration is extremely short, skip
        return {"LF": np.nan, "HF": np.nan, "LF_HF_ratio": np.nan}
    t_resampled = np.arange(0, times[-1], 1.0/fs_resampled)
    #    - Interpolate instantaneous heart rate
    #      instantaneous HR = 60 / RR_interval (BPM)
    #      For simplicity, let's assume each RR interval is the average over that segment
    hr_values = 60.0 / rr_intervals
    # Interpolate HR over time
    hr_resampled = np.interp(t_resampled, times, hr_values)  # times[:-1] because hr_values has one less element than the R-peaks

    # 2) Apply Welch's method to hr_resampled
    freqs, psd = scisig.welch(hr_resampled, fs=fs_resampled, nperseg=len(hr_resampled)//2)
    
    # 3) Define typical LF (0.04–0.15 Hz) and HF (0.15–0.4 Hz) bands
    lf_band = (0.04, 0.15)
    hf_band = (0.15, 0.4)
    
    lf_mask = (freqs >= lf_band[0]) & (freqs < lf_band[1])
    hf_mask = (freqs >= hf_band[0]) & (freqs < hf_band[1])
    
    lf_power = np.trapz(psd[lf_mask], freqs[lf_mask])
    hf_power = np.trapz(psd[hf_mask], freqs[hf_mask])
    
    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else np.nan
    return {"LF": lf_power, "HF": hf_power, "LF_HF_ratio": lf_hf_ratio}

def extract_eda_features(eda_signal, threshold=0.01):
    """
    Computes basic EDA features such as mean, median, and 
    counts of 'SCR events' above a specified threshold.
    Args:
        eda_signal (array-like): EDA signal data.
        threshold (float): Threshold to detect skin conductance responses (SCRs).
    Returns:
        eda_features (dict): {'mean_EDA', 'median_EDA', 'SCR_count'}.
    """
    if len(eda_signal) == 0:
        return {"mean_EDA": np.nan, "median_EDA": np.nan, "SCR_count": np.nan}
    
    mean_val = np.mean(eda_signal)
    median_val = np.median(eda_signal)
    
    # Simple threshold-based SCR detection
    # If the derivative is above threshold, we consider it an SCR onset
    # Note: This is a very simplistic approach
    eda_diff = np.diff(eda_signal)
    scr_count = np.sum(eda_diff > threshold)
    
    return {"mean_EDA": mean_val, "median_EDA": median_val, "SCR_count": scr_count}

def compute_features(e4_data_dict, labels):
    """
    1) Extract raw EDA & BVP signals from E4 dictionary.
    2) Lowpass-filter the EDA signal.
    3) Remove outliers & Z-score normalize EDA and BVP.
    4) Convert to Pandas DataFrame and align with label data.
    Args:
        e4_data_dict (dict): Dictionary containing 'EDA' and 'BVP' signals from the WESAD dataset.
        labels (array-like): Label array at 700 Hz describing stress states.
    Returns:
        combined_df (pandas.DataFrame): DataFrame with time index, columns: ['EDA', 'BVP', 'label'].
    """
    # 1) Raw signals
    eda_raw = np.array(e4_data_dict['EDA'], dtype=np.float32)
    bvp_raw = np.array(e4_data_dict['BVP'], dtype=np.float32)
    
    # 2) Lowpass filter EDA
    eda_filtered = butter_lowpass_filter(eda_raw, cutoff=1.0, fs=fs_dict['EDA'], order=6)
    
    # 3) Outlier removal & Z-score normalization
    eda_clipped = remove_outliers(eda_filtered)
    bvp_clipped = remove_outliers(bvp_raw)
    
    eda_norm = zscore_normalize(eda_clipped)
    bvp_norm = zscore_normalize(bvp_clipped)
    
    # Create DataFrames for each signal
    eda_df = pd.DataFrame(eda_norm, columns=['EDA'])
    bvp_df = pd.DataFrame(bvp_norm, columns=['BVP'])
    label_df = pd.DataFrame(labels, columns=['label'])
    
    # Indexing in real time (makes it easier to join):
    eda_df.index = pd.to_datetime([(1 / fs_dict['EDA']) * i for i in range(len(eda_df))], unit='s')
    bvp_df.index = pd.to_datetime([(1 / fs_dict['BVP']) * i for i in range(len(bvp_df))], unit='s')
    label_df.index = pd.to_datetime([(1 / fs_dict['label']) * i for i in range(len(label_df))], unit='s')
    
    # Outer join all signals, and forward-fill label
    combined_df = eda_df.join(bvp_df, how='outer').join(label_df, how='outer')
    combined_df['label'] = combined_df['label'].bfill()
    
    return combined_df

def get_samples(data, label):
    """
    Segments a DataFrame by time-based windows with overlap,
    extracts BVP and EDA features (time-domain, freq-domain HRV, basic EDA stats).
    Args:
        data (pandas.DataFrame): DataFrame with columns ['EDA', 'BVP', 'label'].
        label (float): The stress label (e.g. 1.0, 2.0, 3.0).
    Returns:
        samples (list): A list of dictionaries containing:
                        'EDA_series', 'BVP_series', 'RR_intervals',
                        'HRV_metrics', 'EDA_features', 'label', etc.
    """
    samples = []
    window_len = int(fs_dict['label'] * WINDOW_IN_SECONDS)   # total samples in a 30s window
    step_size = int(window_len * OVERLAP_FRAC)               # overlap-based step size
    
    start_idx = 0
    while start_idx + window_len <= len(data):
        # Extract the current window
        window_data = data.iloc[start_idx : start_idx + window_len]
        
        eda_series = window_data['EDA'].dropna().values
        bvp_series = window_data['BVP'].dropna().values
        
        # ---------------------- BVP Feature Extraction ----------------------
        if len(bvp_series) > 0:
            rr_intervals, time_metrics = extract_rr_and_time_domain(bvp_series, fs=fs_dict['BVP'])
            freq_metrics = extract_freq_domain_hrv(rr_intervals)
            # Combine the time and frequency domain HRV features
            hrv_metrics = {**time_metrics, **freq_metrics}
        else:
            rr_intervals, hrv_metrics = [], {
                'RMSSD': np.nan,
                'SDNN': np.nan,
                'LF': np.nan,
                'HF': np.nan,
                'LF_HF_ratio': np.nan
            }
        
        # ---------------------- EDA Feature Extraction ----------------------
        eda_features = extract_eda_features(eda_series, threshold=0.01)
        
        samples.append({
            'EDA_series': eda_series,
            'BVP_series': bvp_series,
            'RR_intervals': rr_intervals,
            'HRV_metrics': hrv_metrics,
            'EDA_features': eda_features,
            'label': label
        })
        
        # Advance by step_size to create overlapping windows
        start_idx += step_size
    
    return samples

def process_subject(subject_id):
    """
    Reads raw WESAD data for a subject, processes signals, segments the data,
    and saves the time-series samples for that subject.
    Args:
        subject_id (int): Subject ID to process (e.g., 2, 3, 4, etc.)
    """
    input_path = DATA_RAW / f"WESAD/S{subject_id}/S{subject_id}.pkl"  # Raw input data
    output_path = DATA_PROCESSED / f"WESAD_subject_{subject_id}_time_series.pkl"

    with open(input_path, 'rb') as file:
        data = pickle.load(file, encoding='latin1')
    
    e4_data_dict = data['signal']['wrist']
    labels = data['label']
    
    # Compute features & signals
    combined_data = compute_features(e4_data_dict, labels)
    
    # Filter for only the desired labels (1, 2, 3 in WESAD typically correspond to baseline, stress, meditation)
    combined_data = combined_data[combined_data['label'].isin([1.0, 2.0, 3.0])]
    
    # Group by label and segment
    grouped = combined_data.groupby('label')
    all_samples = []
    for lbl, group in grouped:
        all_samples.extend(get_samples(group, lbl))
    
    # Save the segmented time-series
    with open(output_path, 'wb') as f:
        pickle.dump(all_samples, f)
    
    print(f"Subject {subject_id} processed and saved at {output_path}.")

def combine_files(subject_ids):
    """
    Combines the per-subject time-series data into one file for easier model input.
    Args:
        subject_ids (list of int): The subject IDs that have been processed.
    """
    combined_data = []
    for subject_id in subject_ids:
        file_path = DATA_PROCESSED / f"WESAD_subject_{subject_id}_time_series.pkl"
        with open(file_path, 'rb') as f:
            combined_data.extend(pickle.load(f))
    
    output_path = DATA_PROCESSED / "WESAD_combined_time_series.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(combined_data, f)
    
    print(f"All subjects combined and saved at {output_path}.")

# =============================================================================
#                                 MAIN
# =============================================================================
if __name__ == '__main__':
    subject_ids = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    for subject_id in subject_ids:
        process_subject(subject_id)
    combine_files(subject_ids)
    print("Processing complete.")
