import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import wasserstein_distance
from sklearn.metrics import mean_squared_error

def calculate_rmse(real_data, generated_data, feature_type='time_series'):
    """
    Calculate Root Mean Squared Error between real and generated data.
    
    Args:
        real_data (dict or array): Real data samples
        generated_data (dict or array): Generated data samples
        feature_type (str): Type of data to compare ('time_series' or 'features')
        
    Returns:
        dict: RMSE values for different data components
    """
    results = {}
    
    if feature_type == 'time_series':
        # For time series data (EDA, BVP)
        if isinstance(real_data, dict) and isinstance(generated_data, dict):
            for key in ['EDA_series', 'BVP_series']:
                if key in real_data and key in generated_data:
                    real = real_data[key]
                    gen = generated_data[key]
                    
                    # Handle multi-sample case (Bayesian models)
                    if isinstance(gen, list):
                        # Average over samples
                        gen_mean = np.mean(np.array(gen), axis=0)
                        results[key] = np.sqrt(mean_squared_error(real.flatten(), gen_mean.flatten()))
                    else:
                        results[key] = np.sqrt(mean_squared_error(real.flatten(), gen.flatten()))
        else:
            # Handle array input
            results['overall'] = np.sqrt(mean_squared_error(
                np.array(real_data).flatten(), 
                np.array(generated_data).flatten()
            ))
    
    elif feature_type == 'features':
        # For feature data (HRV, EDA features)
        if isinstance(real_data, dict) and isinstance(generated_data, dict):
            for key in ['HRV_features', 'EDA_features']:
                if key in real_data and key in generated_data:
                    real = real_data[key]
                    gen = generated_data[key]
                    
                    # Handle multi-sample case (Bayesian models)
                    if isinstance(gen, list):
                        gen_mean = np.mean(np.array(gen), axis=0)
                        results[key] = np.sqrt(mean_squared_error(real.flatten(), gen_mean.flatten()))
                    else:
                        results[key] = np.sqrt(mean_squared_error(real.flatten(), gen.flatten()))
        else:
            # Handle array input
            results['overall'] = np.sqrt(mean_squared_error(
                np.array(real_data).flatten(), 
                np.array(generated_data).flatten()
            ))
    
    return results

def gaussian_kernel(x, y, sigma=1.0):
    """
    Compute Gaussian kernel between x and y.
    
    Args:
        x (torch.Tensor): First sample (n_samples_1, dim)
        y (torch.Tensor): Second sample (n_samples_2, dim)
        sigma (float): Kernel bandwidth
        
    Returns:
        torch.Tensor: Kernel matrix (n_samples_1, n_samples_2)
    """
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    
    x = x.unsqueeze(1)  # (n_samples_1, 1, dim)
    y = y.unsqueeze(0)  # (1, n_samples_2, dim)
    
    kernel_input = (x - y).pow(2).sum(2) / float(dim)
    return torch.exp(-kernel_input / (2 * sigma**2))

def maximum_mean_discrepancy(x, y, kernel=gaussian_kernel):
    """
    Compute Maximum Mean Discrepancy (MMD) between samples x and y.
    
    Args:
        x (torch.Tensor): First sample (n_samples_1, dim)
        y (torch.Tensor): Second sample (n_samples_2, dim)
        kernel (function): Kernel function to use
        
    Returns:
        float: MMD value
    """
    x_kernel = kernel(x, x)
    y_kernel = kernel(y, y)
    xy_kernel = kernel(x, y)
    
    n_x = x.size(0)
    n_y = y.size(0)
    
    # Calculate MMD
    mmd = x_kernel.sum() / (n_x * n_x) + y_kernel.sum() / (n_y * n_y) - 2 * xy_kernel.sum() / (n_x * n_y)
    
    return mmd.item()

def calculate_mmd(real_data, generated_data, feature_type='features', batch_size=64, sigma=1.0):
    """
    Calculate Maximum Mean Discrepancy between real and generated distributions.
    
    Args:
        real_data (numpy.ndarray or dict): Real data samples
        generated_data (numpy.ndarray or dict): Generated data samples
        feature_type (str): Type of data to compare ('time_series' or 'features')
        batch_size (int): Batch size for processing
        sigma (float): Kernel bandwidth
        
    Returns:
        dict: MMD values for different data components
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}
    
    if feature_type == 'features':
        # For feature data (HRV, EDA features)
        if isinstance(real_data, dict) and isinstance(generated_data, dict):
            for key in ['HRV_features', 'EDA_features']:
                if key in real_data and key in generated_data:
                    real = real_data[key]
                    gen = generated_data[key]
                    
                    # Handle multi-sample case (Bayesian models)
                    if isinstance(gen, list):
                        # Calculate MMD for each sample and average
                        mmd_values = []
                        for sample in gen:
                            real_tensor = torch.tensor(real, dtype=torch.float32).to(device)
                            gen_tensor = torch.tensor(sample, dtype=torch.float32).to(device)
                            mmd = maximum_mean_discrepancy(real_tensor, gen_tensor, 
                                                          lambda x, y: gaussian_kernel(x, y, sigma))
                            mmd_values.append(mmd)
                        results[key] = np.mean(mmd_values)
                    else:
                        real_tensor = torch.tensor(real, dtype=torch.float32).to(device)
                        gen_tensor = torch.tensor(gen, dtype=torch.float32).to(device)
                        results[key] = maximum_mean_discrepancy(real_tensor, gen_tensor, 
                                                               lambda x, y: gaussian_kernel(x, y, sigma))
        else:
            # Handle array input
            real_tensor = torch.tensor(real_data, dtype=torch.float32).to(device)
            gen_tensor = torch.tensor(generated_data, dtype=torch.float32).to(device)
            results['overall'] = maximum_mean_discrepancy(real_tensor, gen_tensor, 
                                                        lambda x, y: gaussian_kernel(x, y, sigma))
    
    elif feature_type == 'time_series':
        # For time series data (EDA, BVP) - we'll use a sliding window approach
        if isinstance(real_data, dict) and isinstance(generated_data, dict):
            for key in ['EDA_series', 'BVP_series']:
                if key in real_data and key in generated_data:
                    real = real_data[key]
                    gen = generated_data[key]
                    
                    # Handle multi-sample case (Bayesian models)
                    if isinstance(gen, list):
                        # Calculate MMD for each sample and average
                        mmd_values = []
                        for sample in gen:
                            # Extract fixed-length segments for comparison
                            window_size = min(100, min(real.shape[1], sample.shape[1]))
                            real_windows = extract_windows(real, window_size)
                            gen_windows = extract_windows(sample, window_size)
                            
                            real_tensor = torch.tensor(real_windows, dtype=torch.float32).to(device)
                            gen_tensor = torch.tensor(gen_windows, dtype=torch.float32).to(device)
                            
                            # Process in batches to avoid memory issues
                            mmd = batch_mmd_calculation(real_tensor, gen_tensor, batch_size, sigma)
                            mmd_values.append(mmd)
                        
                        results[key] = np.mean(mmd_values)
                    else:
                        # Extract fixed-length segments for comparison
                        window_size = min(100, min(real.shape[1], gen.shape[1]))
                        real_windows = extract_windows(real, window_size)
                        gen_windows = extract_windows(gen, window_size)
                        
                        real_tensor = torch.tensor(real_windows, dtype=torch.float32).to(device)
                        gen_tensor = torch.tensor(gen_windows, dtype=torch.float32).to(device)
                        
                        # Process in batches to avoid memory issues
                        results[key] = batch_mmd_calculation(real_tensor, gen_tensor, batch_size, sigma)
    
    return results

def extract_windows(data, window_size, stride=50):
    """
    Extract fixed-length windows from time series data.
    
    Args:
        data (numpy.ndarray): Time series data (batch_size, seq_length, features)
        window_size (int): Size of windows to extract
        stride (int): Stride between windows
        
    Returns:
        numpy.ndarray: Extracted windows (n_windows, window_size, features)
    """
    batch_size, seq_length, features = data.shape
    windows = []
    
    for b in range(batch_size):
        for i in range(0, seq_length - window_size + 1, stride):
            windows.append(data[b, i:i+window_size, :])
    
    return np.array(windows)

def batch_mmd_calculation(real_tensor, gen_tensor, batch_size, sigma):
    """
    Calculate MMD in batches to avoid memory issues.
    
    Args:
        real_tensor (torch.Tensor): Real data windows
        gen_tensor (torch.Tensor): Generated data windows
        batch_size (int): Batch size
        sigma (float): Kernel bandwidth
        
    Returns:
        float: MMD value
    """
    n_real = real_tensor.size(0)
    n_gen = gen_tensor.size(0)
    
    # Sample a subset if too large
    if n_real > 1000:
        indices = torch.randperm(n_real)[:1000]
        real_tensor = real_tensor[indices]
        n_real = 1000
    
    if n_gen > 1000:
        indices = torch.randperm(n_gen)[:1000]
        gen_tensor = gen_tensor[indices]
        n_gen = 1000
    
    # Flatten windows for MMD calculation
    real_flat = real_tensor.reshape(n_real, -1)
    gen_flat = gen_tensor.reshape(n_gen, -1)
    
    # Calculate MMD
    return maximum_mean_discrepancy(real_flat, gen_flat, lambda x, y: gaussian_kernel(x, y, sigma))

def calculate_wasserstein_distance(real_data, generated_data, feature_type='features'):
    """
    Calculate Wasserstein distance (Earth Mover's Distance) between real and generated distributions.
    
    Args:
        real_data (numpy.ndarray or dict): Real data samples
        generated_data (numpy.ndarray or dict): Generated data samples
        feature_type (str): Type of data to compare ('time_series' or 'features')
        
    Returns:
        dict: Wasserstein distance values for different data components
    """
    results = {}
    
    if feature_type == 'features':
        # For feature data (HRV, EDA features)
        if isinstance(real_data, dict) and isinstance(generated_data, dict):
            for key in ['HRV_features', 'EDA_features']:
                if key in real_data and key in generated_data:
                    real = real_data[key]
                    gen = generated_data[key]
                    
                    # Calculate per-feature Wasserstein distance
                    feature_distances = []
                    
                    # Handle multi-sample case (Bayesian models)
                    if isinstance(gen, list):
                        # Calculate distances for each sample and average
                        sample_distances = []
                        for sample in gen:
                            feature_distances = []
                            for i in range(real.shape[1]):
                                wd = wasserstein_distance(real[:, i].flatten(), sample[:, i].flatten())
                                feature_distances.append(wd)
                            sample_distances.append(np.mean(feature_distances))
                        results[key] = np.mean(sample_distances)
                    else:
                        for i in range(real.shape[1]):
                            wd = wasserstein_distance(real[:, i].flatten(), gen[:, i].flatten())
                            feature_distances.append(wd)
                        results[key] = np.mean(feature_distances)
        else:
            # Handle array input
            feature_distances = []
            for i in range(real_data.shape[1]):
                wd = wasserstein_distance(real_data[:, i].flatten(), generated_data[:, i].flatten())
                feature_distances.append(wd)
            results['overall'] = np.mean(feature_distances)
    
    elif feature_type == 'time_series':
        # For time series data (EDA, BVP)
        if isinstance(real_data, dict) and isinstance(generated_data, dict):
            for key in ['EDA_series', 'BVP_series']:
                if key in real_data and key in generated_data:
                    real = real_data[key]
                    gen = generated_data[key]
                    
                    # Handle multi-sample case (Bayesian models)
                    if isinstance(gen, list):
                        # Calculate for each sample and average
                        sample_distances = []
                        for sample in gen:
                            # Calculate distributions of values and their Wasserstein distance
                            wd = wasserstein_distance(real.flatten(), sample.flatten())
                            sample_distances.append(wd)
                        results[key] = np.mean(sample_distances)
                    else:
                        # Calculate distributions of values and their Wasserstein distance
                        results[key] = wasserstein_distance(real.flatten(), gen.flatten())
        else:
            # Handle array input
            results['overall'] = wasserstein_distance(
                np.array(real_data).flatten(), 
                np.array(generated_data).flatten()
            )
    
    return results

def calculate_feature_statistics(data, feature_type='features'):
    """
    Calculate basic statistics (mean, std, min, max) for data features.
    
    Args:
        data (numpy.ndarray or dict): Data samples
        feature_type (str): Type of data ('time_series' or 'features')
        
    Returns:
        dict: Statistics for data features
    """
    stats = {}
    
    if feature_type == 'features':
        # For feature data (HRV, EDA features)
        if isinstance(data, dict):
            for key in ['HRV_features', 'EDA_features']:
                if key in data:
                    feature_data = data[key]
                    
                    # Handle multi-sample case (Bayesian models)
                    if isinstance(feature_data, list):
                        # Calculate stats for each sample and average
                        sample_stats = []
                        for sample in feature_data:
                            sample_stats.append({
                                'mean': np.mean(sample, axis=0),
                                'std': np.std(sample, axis=0),
                                'min': np.min(sample, axis=0),
                                'max': np.max(sample, axis=0)
                            })
                        
                        # Average stats across samples
                        stats[key] = {
                            'mean': np.mean([s['mean'] for s in sample_stats], axis=0),
                            'std': np.mean([s['std'] for s in sample_stats], axis=0),
                            'min': np.mean([s['min'] for s in sample_stats], axis=0),
                            'max': np.mean([s['max'] for s in sample_stats], axis=0)
                        }
                    else:
                        stats[key] = {
                            'mean': np.mean(feature_data, axis=0),
                            'std': np.std(feature_data, axis=0),
                            'min': np.min(feature_data, axis=0),
                            'max': np.max(feature_data, axis=0)
                        }
        else:
            # Handle array input
            stats['overall'] = {
                'mean': np.mean(data, axis=0),
                'std': np.std(data, axis=0),
                'min': np.min(data, axis=0),
                'max': np.max(data, axis=0)
            }
    
    elif feature_type == 'time_series':
        # For time series data (EDA, BVP)
        if isinstance(data, dict):
            for key in ['EDA_series', 'BVP_series']:
                if key in data:
                    ts_data = data[key]
                    
                    # Handle multi-sample case (Bayesian models)
                    if isinstance(ts_data, list):
                        # Calculate stats for each sample and average
                        sample_stats = []
                        for sample in ts_data:
                            sample_stats.append({
                                'mean': np.mean(sample),
                                'std': np.std(sample),
                                'min': np.min(sample),
                                'max': np.max(sample),
                                # Add time series specific stats
                                'range': np.max(sample) - np.min(sample),
                                'median': np.median(sample)
                            })
                        
                        # Average stats across samples
                        stats[key] = {
                            'mean': np.mean([s['mean'] for s in sample_stats]),
                            'std': np.mean([s['std'] for s in sample_stats]),
                            'min': np.mean([s['min'] for s in sample_stats]),
                            'max': np.mean([s['max'] for s in sample_stats]),
                            'range': np.mean([s['range'] for s in sample_stats]),
                            'median': np.mean([s['median'] for s in sample_stats])
                        }
                    else:
                        stats[key] = {
                            'mean': np.mean(ts_data),
                            'std': np.std(ts_data),
                            'min': np.min(ts_data),
                            'max': np.max(ts_data),
                            # Add time series specific stats
                            'range': np.max(ts_data) - np.min(ts_data),
                            'median': np.median(ts_data)
                        }
        else:
            # Handle array input
            stats['overall'] = {
                'mean': np.mean(data),
                'std': np.std(data),
                'min': np.min(data),
                'max': np.max(data),
                'range': np.max(data) - np.min(data),
                'median': np.median(data)
            }
    
    return stats

def calculate_fid_like_score(real_stats, generated_stats, feature_key):
    """
    Calculate a FID-like score based on feature statistics.
    This is a simplified version inspired by Fréchet Inception Distance.
    
    Args:
        real_stats (dict): Statistics of real data
        generated_stats (dict): Statistics of generated data
        feature_key (str): Key to access specific feature statistics
        
    Returns:
        float: FID-like score (lower is better)
    """
    real_mean = real_stats[feature_key]['mean']
    real_std = real_stats[feature_key]['std']
    gen_mean = generated_stats[feature_key]['mean']
    gen_std = generated_stats[feature_key]['std']
    
    # Calculate squared mean difference
    mean_diff_sq = np.sum((real_mean - gen_mean) ** 2)
    
    # Calculate covariance term (simplified)
    cov_term = np.sum((real_std - gen_std) ** 2)
    
    # FID-like score
    fid_like = mean_diff_sq + cov_term
    
    return fid_like