import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import json
from datetime import datetime
import argparse
from tqdm import tqdm
import random
from copy import deepcopy

from src.models.dropout_bayes_gan import (BayesianTimeSeriesGAN, BayesianEDATimeSeriesGAN, 
                                      BayesianEDAMultiScaleTimeSeriesGAN, BayesianFeatureGAN,
                                      BayesianBVPTimeSeriesGAN, BayesianBVPMultiScaleTimeSeriesGAN)
from src.evaluation.metrics import calculate_mmd
from src.training.train_base_gan import prepare_data, historical_average

def evaluate_mmd_with_uncertainty(model, data_loader, n_samples=1000, mc_samples=10, data_type='time_series', signal_type='EDA'):
    """
    Evaluate Bayesian model using Maximum Mean Discrepancy (MMD) with uncertainty estimation.
    
    Args:
        model: The Bayesian GAN model to evaluate
        data_loader: DataLoader containing validation or test data
        n_samples (int): Number of samples to generate for evaluation
        mc_samples (int): Number of Monte Carlo samples for uncertainty estimation
        data_type (str): Type of data to evaluate ('time_series' or 'features')
        signal_type (str): Type of physiological signal ('EDA' or 'BVP')
        
    Returns:
        dict: MMD scores with uncertainty metrics (mean, std, min, max)
    """
    device = next(model.parameters()).device
    model.train()  # Keep in training mode for dropout
    
    # Collect real data
    real_data_list = []
    real_labels = []
    
    with torch.no_grad():
        if data_type == 'time_series':
            # For time series data (EDA or BVP)
            for data, labels in data_loader:
                real_data_list.append(data.to(device))
                real_labels.append(labels.to(device))
            
            # Concatenate batches
            real_data = torch.cat(real_data_list, dim=0)
            real_labels = torch.cat(real_labels, dim=0)
            
            # Generate fake data multiple times with Monte Carlo dropout
            batch_size = min(512, real_data.size(0))  # Avoid OOM issues
            z = torch.randn(batch_size, model.latent_dim, device=device)
            fake_labels = torch.randint(1, 4, (batch_size,), device=device)  # Random stress conditions
            
            # Generate multiple MC samples
            fake_data_samples = []
            for _ in range(mc_samples):
                fake_data = model.generate(z, fake_labels)
                fake_data_samples.append(fake_data)
            
            # Calculate MMD for each sample
            mmd_scores = []
            real_np = real_data.cpu().numpy()
            
            for fake_data in fake_data_samples:
                fake_np = fake_data.cpu().numpy()
                
                # Format for MMD calculation based on signal type
                if signal_type == 'EDA':
                    real_dict = {"EDA_series": real_np}
                    fake_dict = {"EDA_series": fake_np}
                else:  # BVP
                    real_dict = {"BVP_series": real_np}
                    fake_dict = {"BVP_series": fake_np}
                
                mmd_result = calculate_mmd(real_dict, fake_dict, feature_type='time_series')
                
                # Get appropriate result based on signal type
                if signal_type == 'EDA':
                    mmd_scores.append(mmd_result.get("EDA_series", float('inf')))
                else:  # BVP
                    mmd_scores.append(mmd_result.get("BVP_series", float('inf')))
            
            # Calculate statistics
            mmd_scores = np.array(mmd_scores)
            return {
                'mean': float(np.mean(mmd_scores)),
                'std': float(np.std(mmd_scores)),
                'min': float(np.min(mmd_scores)),
                'max': float(np.max(mmd_scores))
            }
            
        else:  # features
            # For feature data (HRV + EDA features)
            real_data = {}
            for data_batch, labels in data_loader():
                for k in data_batch:
                    if k not in real_data:
                        real_data[k] = []
                    real_data[k].append(data_batch[k].to(device))
                real_labels.append(labels.to(device))
            
            # Concatenate batches for each feature type
            for k in real_data:
                real_data[k] = torch.cat(real_data[k], dim=0)
            
            # Concatenate batches for labels
            real_labels = torch.cat(real_labels, dim=0)
            
            # Generate fake features with Monte Carlo dropout
            batch_size = min(512, real_labels.size(0))  # Avoid OOM issues
            z = torch.randn(batch_size, model.latent_dim, device=device)
            fake_labels = torch.randint(1, 4, (batch_size,), device=device)  # Random stress conditions
            
            # Generate multiple MC samples
            fake_feature_samples = []
            for _ in range(mc_samples):
                fake_features = model.generate(z, fake_labels)
                fake_feature_samples.append(fake_features)
            
            # Calculate MMD for each sample
            mmd_scores = []
            real_features = {k: v.cpu().numpy() for k, v in real_data.items()}
            
            for fake_features in fake_feature_samples:
                fake_features_np = {k: v.cpu().numpy() for k, v in fake_features.items()}
                mmd_result = calculate_mmd(real_features, fake_features_np, feature_type='features')
                mmd_scores.append(np.mean([v for k, v in mmd_result.items()]))
            
            # Calculate statistics
            mmd_scores = np.array(mmd_scores)
            return {
                'mean': float(np.mean(mmd_scores)),
                'std': float(np.std(mmd_scores)),
                'min': float(np.min(mmd_scores)),
                'max': float(np.max(mmd_scores))
            }

def train_bayesian_gan(model, dataloaders, config):
    """
    Train a Bayesian Dropout GAN model with the given data loaders and configuration.
    
    Args:
        model: The Bayesian GAN model to train
        dataloaders: Dictionary of data loaders for training, validation, and testing
        config: Dictionary containing training configuration
        
    Returns:
        dict: Training history and best model
    """
    # Training configuration
    lr_g = config.get('lr_g', 0.0002)  # Generator learning rate
    lr_d = config.get('lr_d', 0.0001)  # Discriminator learning rate (slower)
    beta1 = config.get('beta1', 0.5)
    beta2 = config.get('beta2', 0.999)
    n_epochs = config.get('n_epochs', 100)
    latent_dim = config.get('latent_dim', 100)
    save_interval = config.get('save_interval', 10)
    patience = config.get('patience', 20)
    model_dir = config.get('model_dir', 'models')
    data_type = config.get('data_type', 'time_series')
    signal_type = config.get('signal_type', 'EDA')
    dropout_rate = config.get('dropout_rate', 0.2)
    mc_samples = config.get('mc_samples', 10)  # Monte Carlo samples for evaluation
    
    # Instance noise parameters
    initial_noise = config.get('initial_noise', 0.15)  # Initial noise level
    noise_decay_epochs = n_epochs // 2  # Decay noise over first half of training
    
    os.makedirs(model_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create optimizers with different learning rates
    g_optimizer = optim.Adam(model.generator.parameters(), lr=lr_g, betas=(beta1, beta2))
    d_optimizer = optim.Adam(model.discriminator.parameters(), lr=lr_d, betas=(beta1, beta2))
    
    # Create EMA model for stable training
    ema_model = deepcopy(model)
    ema_model = ema_model.to(device)
    
    # Select appropriate data loader based on model type
    if data_type == 'time_series':
        # Handle time series data (EDA or BVP)
        data_key = f"{signal_type}_series"
        
        if data_key not in dataloaders:
            raise ValueError(f"No dataloaders found for {data_key}. Available keys: {list(dataloaders.keys())}")
            
        if dataloaders[data_key]['train'] is None:
            # Check if we should use a fallback
            if signal_type == 'BVP' and 'EDA_series' in dataloaders and dataloaders['EDA_series']['train'] is not None:
                print(f"WARNING: No {signal_type} data available. Using EDA data as fallback.")
                train_loader = dataloaders['EDA_series']['train']
                val_loader = dataloaders['EDA_series']['val']
            else:
                raise ValueError(f"No training data available for {data_key}")
        else:
            train_loader = dataloaders[data_key]['train']
            val_loader = dataloaders[data_key]['val']
            
        print(f"Using {data_key} data for training")
    else:  # features
        if dataloaders['features']['train'] is None:
            raise ValueError("No feature training data available")
            
        train_loader = dataloaders['features']['train']
        val_loader = dataloaders['features']['val']
        print("Using combined feature data for training")
    
    # Training history
    history = {
        'g_loss': [],
        'd_loss': [],
        'd_real': [],
        'd_fake': [],
        'val_mmd_mean': [],
        'val_mmd_std': [],
        'noise_level': [],
        'fm_loss': []
    }
    
    # Early stopping variables
    best_mmd = float('inf')
    best_epoch = 0
    best_model = None
    patience_counter = 0
    
    # Training loop
    for epoch in range(n_epochs):
        model.train()
        g_losses = []
        d_losses = []
        d_real_outputs = []
        d_fake_outputs = []
        
        # Calculate current noise level (linear decay)
        if epoch < noise_decay_epochs:
            noise_level = initial_noise * (1 - epoch / noise_decay_epochs)
        else:
            noise_level = 0.0
        
        history['noise_level'].append(noise_level)
        
        # Create progress bar for this epoch
        if data_type == 'time_series':
            pbar = tqdm(train_loader)
            for real_data, labels in pbar:
                batch_size = real_data.size(0)
                real_data = real_data.to(device)
                labels = labels.to(device)
                
                # ---------------------
                # Train Discriminator
                # ---------------------
                d_optimizer.zero_grad()
                
                # Add instance noise to real data
                real_data_noisy = real_data + noise_level * torch.randn_like(real_data)
                
                # Real data
                d_real_output = model.discriminate(real_data_noisy, labels)
                d_real_loss = -torch.mean(d_real_output)
                
                # Fake data
                z = torch.randn(batch_size, latent_dim, device=device)
                fake_data = model.generate(z, labels)
                
                # Add instance noise to fake data
                fake_data_noisy = fake_data + noise_level * torch.randn_like(fake_data)
                
                d_fake_output = model.discriminate(fake_data_noisy.detach(), labels)
                d_fake_loss = torch.mean(d_fake_output)
                
                # Gradient penalty (WGAN-GP style)
                alpha = torch.rand(batch_size, 1, 1, device=device)
                alpha = alpha.expand_as(real_data)
                interpolated = alpha * real_data + (1 - alpha) * fake_data.detach()
                interpolated.requires_grad_(True)
                
                d_interpolated = model.discriminate(interpolated, labels)
                
                gradients = torch.autograd.grad(
                    outputs=d_interpolated,
                    inputs=interpolated,
                    grad_outputs=torch.ones_like(d_interpolated),
                    create_graph=True,
                    retain_graph=True
                )[0]
                
                gradients = gradients.view(batch_size, -1)
                gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10.0
                
                # Total discriminator loss
                d_loss = d_real_loss + d_fake_loss + gradient_penalty
                d_loss.backward()
                d_optimizer.step()
                
                # ---------------------
                # Train Generator
                # ---------------------
                if epoch > 0 or len(d_losses) % 5 == 0:  # Train G less frequently at the start
                    g_optimizer.zero_grad()
                    
                    # Generate fake data
                    z = torch.randn(batch_size, latent_dim, device=device)
                    fake_data = model.generate(z, labels)
                    
                    # Calculate generator loss (no noise for generator training)
                    g_fake_output = model.discriminate(fake_data, labels)
                    g_loss = -torch.mean(g_fake_output)
                    
                    # Add feature matching loss
                    fm_loss = model.feature_matching_loss(real_data, fake_data, labels)
                    g_total_loss = g_loss + 0.1 * fm_loss  # Weight the feature matching loss
                    
                    g_total_loss.backward()
                    g_optimizer.step()
                    
                    g_losses.append(g_loss.item())
                    history['fm_loss'].append(fm_loss.item())  # Track feature matching loss
                
                # Record losses
                d_losses.append(d_loss.item())
                d_real_outputs.append(torch.mean(d_real_output).item())
                d_fake_outputs.append(torch.mean(d_fake_output).item())
                
                # Update EMA model
                historical_average(model, ema_model)
                
                # Update progress bar
                pbar.set_description(f"Epoch {epoch+1}/{n_epochs}, D: {d_loss.item():.4f}, G: {g_loss.item() if len(g_losses) > 0 else 0:.4f}, Noise: {noise_level:.4f}")
                
        else:  # features
            data_iter = train_loader()
            num_batches = min(100, len(list(train_loader())))  # Limit number of batches per epoch
            pbar = tqdm(range(num_batches))
            
            for _ in pbar:
                try:
                    features, labels = next(data_iter)
                except StopIteration:
                    data_iter = train_loader()
                    features, labels = next(data_iter)
                
                batch_size = labels.size(0)
                labels = labels.to(device)
                features = {k: v.to(device) for k, v in features.items()}
                
                # ---------------------
                # Train Discriminator
                # ---------------------
                d_optimizer.zero_grad()
                
                # Add instance noise to real features
                noisy_features = {}
                for k, v in features.items():
                    noisy_features[k] = v + noise_level * torch.randn_like(v)
                
                # Real data
                d_real_output = model.discriminate(noisy_features, labels)
                d_real_loss = -torch.mean(d_real_output)
                
                # Fake data
                z = torch.randn(batch_size, latent_dim, device=device)
                fake_features = model.generate(z, labels)
                
                # Add instance noise to fake features
                noisy_fake_features = {}
                for k, v in fake_features.items():
                    noisy_fake_features[k] = v + noise_level * torch.randn_like(v)
                
                d_fake_output = model.discriminate(noisy_fake_features, labels)
                d_fake_loss = torch.mean(d_fake_output)
                
                # Simple GP for feature data (features are already normalized)
                d_loss = d_real_loss + d_fake_loss + 0.1 * (d_real_output ** 2).mean() # Simplified regularization
                d_loss.backward()
                d_optimizer.step()
                
                # ---------------------
                # Train Generator
                # ---------------------
                if epoch > 0 or _ % 5 == 0:  # Train G less frequently at the start
                    g_optimizer.zero_grad()
                    
                    # Generate fake features
                    z = torch.randn(batch_size, latent_dim, device=device)
                    fake_features = model.generate(z, labels)
                    
                    # Calculate generator loss (no noise for generator training)
                    g_fake_output = model.discriminate(fake_features, labels)
                    g_loss = -torch.mean(g_fake_output)
                    
                    # Add feature matching loss
                    fm_loss = model.feature_matching_loss(features, fake_features, labels)
                    g_total_loss = g_loss + 0.1 * fm_loss  # Weight the feature matching loss
                    
                    g_total_loss.backward()
                    g_optimizer.step()
                    
                    g_losses.append(g_loss.item())
                    history['fm_loss'].append(fm_loss.item())  # Track feature matching loss
                
                # Record losses
                d_losses.append(d_loss.item())
                d_real_outputs.append(torch.mean(d_real_output).item())
                d_fake_outputs.append(torch.mean(d_fake_output).item())
                
                # Update EMA model
                historical_average(model, ema_model)
                
                # Update progress bar
                pbar.set_description(f"Epoch {epoch+1}/{n_epochs}, D: {d_loss.item():.4f}, G: {g_loss.item() if len(g_losses) > 0 else 0:.4f}, Noise: {noise_level:.4f}")
        
        # Record average losses for this epoch
        history['g_loss'].append(np.mean(g_losses))
        history['d_loss'].append(np.mean(d_losses))
        history['d_real'].append(np.mean(d_real_outputs))
        history['d_fake'].append(np.mean(d_fake_outputs))
        
        # Validation using MMD with uncertainty
        mmd_metrics = evaluate_mmd_with_uncertainty(
            ema_model, 
            val_loader, 
            data_type=data_type,
            signal_type=signal_type,
            mc_samples=mc_samples
        )
        
        history['val_mmd_mean'].append(mmd_metrics['mean'])
        history['val_mmd_std'].append(mmd_metrics['std'])
        
        print(f"Epoch {epoch+1}/{n_epochs} - Val MMD: {mmd_metrics['mean']:.6f} ± {mmd_metrics['std']:.6f} (min: {mmd_metrics['min']:.6f}, max: {mmd_metrics['max']:.6f})")
        
        # Check for improvement (using mean MMD)
        if mmd_metrics['mean'] < best_mmd:
            best_mmd = mmd_metrics['mean']
            best_epoch = epoch
            best_model = deepcopy(ema_model)
            patience_counter = 0
            
            # Save best model
            model_path = os.path.join(model_dir, f"best_bayesian_{data_type}_{signal_type}_gan.pt")
            torch.save(ema_model.state_dict(), model_path)
            print(f"New best model saved with MMD: {best_mmd:.6f}")
        else:
            patience_counter += 1
            
        # Save model at specified intervals
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(model_dir, f"bayesian_{data_type}_{signal_type}_gan_e{epoch+1}.pt")
            torch.save(model.state_dict(), checkpoint_path)
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs!")
            break
            
    # Save final model
    final_path = os.path.join(model_dir, f"final_bayesian_{data_type}_{signal_type}_gan.pt")
    torch.save(model.state_dict(), final_path)
    
    # Save training history
    history_path = os.path.join(model_dir, f"bayesian_{data_type}_{signal_type}_gan_history.json")
    with open(history_path, 'w') as f:
        json.dump({k: [float(i) for i in v] for k, v in history.items()}, f)
    
    return {
        'history': history,
        'best_model': best_model,
        'best_epoch': best_epoch,
        'best_mmd': best_mmd
    }

def plot_bayesian_history(history, title, save_path=None):
    """
    Plot training history metrics for Bayesian GAN.
    
    Args:
        history (dict): Training history containing metrics
        title (str): Title for the plot
        save_path (str, optional): Path to save the plot
        
    Returns:
        None
    """
    plt.figure(figsize=(16, 14))
    
    # Plot generator and discriminator losses
    plt.subplot(4, 2, 1)
    plt.plot(history['g_loss'], label='Generator Loss')
    plt.plot(history['d_loss'], label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('GAN Losses')
    
    # Plot discriminator outputs
    plt.subplot(4, 2, 2)
    plt.plot(history['d_real'], label='D(real)')
    plt.plot(history['d_fake'], label='D(fake)')
    plt.xlabel('Epoch')
    plt.ylabel('Discriminator Output')
    plt.legend()
    plt.title('Discriminator Outputs')
    
    # Plot validation MMD mean
    plt.subplot(4, 2, 3)
    plt.plot(history['val_mmd_mean'], label='Validation MMD (Mean)')
    plt.xlabel('Epoch')
    plt.ylabel('MMD')
    plt.legend()
    plt.title('Validation MMD Mean (lower is better)')
    
    # Plot validation MMD std
    plt.subplot(4, 2, 4)
    plt.plot(history['val_mmd_std'], label='Validation MMD (Std)')
    plt.xlabel('Epoch')
    plt.ylabel('MMD Std Dev')
    plt.legend()
    plt.title('Validation MMD Uncertainty')
    
    # Plot instance noise level
    if 'noise_level' in history:
        plt.subplot(4, 2, 5)
        plt.plot(history['noise_level'], label='Instance Noise (σ)')
        plt.xlabel('Epoch')
        plt.ylabel('Noise Level (σ)')
        plt.legend()
        plt.title('Instance Noise Level')
    
    # Plot feature matching loss
    if 'fm_loss' in history:
        plt.subplot(4, 2, 6)
        plt.plot(history['fm_loss'], label='Feature Matching Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Feature Matching Loss')
    
    # Plot MMD with uncertainty ranges
    plt.subplot(4, 2, 7)
    mean = np.array(history['val_mmd_mean'])
    std = np.array(history['val_mmd_std'])
    epochs = np.arange(len(mean))
    plt.plot(epochs, mean, 'k-', label='Mean MMD')
    plt.fill_between(epochs, mean - std, mean + std, alpha=0.3, label='±1 std dev')
    plt.xlabel('Epoch')
    plt.ylabel('MMD')
    plt.legend()
    plt.title('MMD with Uncertainty')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def main():
    """
    Main function to parse arguments and run Bayesian Dropout GAN training.
    """
    parser = argparse.ArgumentParser(description='Train Bayesian Dropout GAN models for physiological signals')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True, help='Path to preprocessed data (.pt file)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--test_split', type=float, default=0.1, help='Test split ratio')
    
    # Model arguments
    parser.add_argument('--latent_dim', type=int, default=100, help='Dimension of latent space')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers (for time series)')
    parser.add_argument('--model_type', type=str, default='all', choices=['all', 'EDA', 'BVP', 'features'], 
                        help='Type of model to train: EDA time series, BVP time series, features, or all')
    parser.add_argument('--use_multiscale', action='store_true', help='Use multi-scale generator for signals')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout probability for Bayesian layers')
    parser.add_argument('--mc_samples', type=int, default=10, help='Number of Monte Carlo samples for uncertainty')
    
    # Training arguments
    parser.add_argument('--lr_g', type=float, default=0.0002, help='Generator learning rate')
    parser.add_argument('--lr_d', type=float, default=0.0001, help='Discriminator learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta2 for Adam optimizer')
    parser.add_argument('--n_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--save_interval', type=int, default=10, help='Save model every N epochs')
    parser.add_argument('--patience', type=int, default=50, help='Patience for early stopping')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--initial_noise', type=float, default=0.15, help='Initial instance noise level')
    
    args = parser.parse_args()
    
    # Set up seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Prepare data
    print("Preparing data...")
    dataloaders = prepare_data(
        args.data_path,
        batch_size=args.batch_size,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed
    )
    
    # Create output directories
    os.makedirs(args.model_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = os.path.join(args.model_dir, f"bayesian_gan_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    config_path = os.path.join(experiment_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    # Training configuration
    train_config = {
        'lr_g': args.lr_g,
        'lr_d': args.lr_d,
        'beta1': args.beta1,
        'beta2': args.beta2,
        'n_epochs': args.n_epochs,
        'latent_dim': args.latent_dim,
        'save_interval': args.save_interval,
        'patience': args.patience,
        'model_dir': experiment_dir,
        'initial_noise': args.initial_noise,
        'dropout_rate': args.dropout_rate,
        'mc_samples': args.mc_samples
    }
    
    # Train models based on the specified model_type
    if args.model_type in ['all', 'EDA']:
        # Train EDA time series model
        print("\n===== Training Bayesian EDA Time Series Model =====")
        # Get a sample batch to determine data dimensions
        eda_sample_data, _ = next(iter(dataloaders['EDA_series']['train']))
        
        # Determine sequence length from data shape
        seq_length = max(eda_sample_data.shape[1], eda_sample_data.shape[2])
        
        # Choose between standard and multi-scale EDA model
        if args.use_multiscale:
            print("Using Bayesian multi-scale generator for EDA signals")
            eda_model = BayesianEDAMultiScaleTimeSeriesGAN(
                latent_dim=args.latent_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                num_classes=3,  # 3 stress conditions
                seq_length=seq_length,
                dropout_rate=args.dropout_rate
            )
        else:
            eda_model = BayesianEDATimeSeriesGAN(
                latent_dim=args.latent_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                num_classes=3,  # 3 stress conditions
                seq_length=seq_length,
                dropout_rate=args.dropout_rate
            )
        
        eda_config = {**train_config, 'data_type': 'time_series', 'signal_type': 'EDA'}
        eda_results = train_bayesian_gan(eda_model, dataloaders, eda_config)
        
        # Plot and save EDA training history
        plot_bayesian_history(
            eda_results['history'],
            'Bayesian EDA Time Series GAN Training',
            save_path=os.path.join(experiment_dir, 'bayesian_eda_series_history.png')
        )
    
    if args.model_type in ['all', 'BVP']:
        # Train BVP time series model
        print("\n===== Training Bayesian BVP Time Series Model =====")
        # Use BVP data shape for model initialization
        if 'BVP_series' in dataloaders and dataloaders['BVP_series']['train'] is not None:
            bvp_sample_data, _ = next(iter(dataloaders['BVP_series']['train']))
        else:
            # If no specific BVP data, use EDA data for shape determination
            bvp_sample_data, _ = next(iter(dataloaders['EDA_series']['train']))
            print("No BVP-specific data found, using EDA data shape information")
        
        # Determine sequence length from data shape
        seq_length = max(bvp_sample_data.shape[1], bvp_sample_data.shape[2])
        
        # Choose between standard and multi-scale BVP model
        if args.use_multiscale:
            print("Using Bayesian multi-scale generator for BVP signals")
            bvp_model = BayesianBVPMultiScaleTimeSeriesGAN(
                latent_dim=args.latent_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers + 1,  # Extra layer for oscillatory patterns
                num_classes=3,  # 3 stress conditions
                seq_length=seq_length,
                dropout_rate=args.dropout_rate
            )
        else:
            bvp_model = BayesianBVPTimeSeriesGAN(
                latent_dim=args.latent_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                num_classes=3,  # 3 stress conditions
                seq_length=seq_length,
                dropout_rate=args.dropout_rate
            )
        
        bvp_config = {**train_config, 'data_type': 'time_series', 'signal_type': 'BVP'}
        bvp_results = train_bayesian_gan(bvp_model, dataloaders, bvp_config)
        
        # Plot and save BVP training history
        plot_bayesian_history(
            bvp_results['history'],
            'Bayesian BVP Time Series GAN Training',
            save_path=os.path.join(experiment_dir, 'bayesian_bvp_series_history.png')
        )
    
    if args.model_type in ['all', 'features'] and dataloaders['features']['train'] is not None:
        # Train features model (HRV + EDA features)
        print("\n===== Training Bayesian Features Model (HRV + EDA) =====")
        feature_model = BayesianFeatureGAN(
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            num_classes=3,  # 3 stress conditions
            dropout_rate=args.dropout_rate
        )
        
        feature_config = {**train_config, 'data_type': 'features'}
        feature_results = train_bayesian_gan(feature_model, dataloaders, feature_config)
        
        # Plot and save feature training history
        plot_bayesian_history(
            feature_results['history'],
            'Bayesian Feature GAN Training (HRV + EDA)',
            save_path=os.path.join(experiment_dir, 'bayesian_feature_history.png')
        )
    elif args.model_type == 'features' and dataloaders['features']['train'] is None:
        print("Skipping feature model training - feature data not available.")
    
    model_type_str = "all models" if args.model_type == 'all' else f"{args.model_type} model"
    print(f"\nTraining completed for Bayesian {model_type_str}! Models saved in {experiment_dir}")

if __name__ == "__main__":
    main() 