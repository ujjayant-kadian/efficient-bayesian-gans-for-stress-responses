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

from src.models.variational_bayes_gan import (BayesTimeSeriesGAN, BayesEDATimeSeriesGAN, 
                                             BayesEDAMultiScaleTimeSeriesGAN, BayesFeatureGAN,
                                             BayesBVPTimeSeriesGAN, BayesBVPMultiScaleTimeSeriesGAN)
from src.models.variational_layers import PriorType, BayesLinear, BayesConv1d
from src.training.train_base_gan import prepare_data, evaluate_mmd, historical_average, plot_history

def train_gan(model, dataloaders, config):
    """
    Train a Bayesian GAN model with the given data loaders and configuration.
    Incorporates KL divergence from both generator and discriminator sides.
    
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
    
    # KL annealing parameters
    kl_weight_final = config.get('kl_weight', 0.001)  # Final KL weight value
    kl_annealing_start = config.get('kl_annealing_start', 35)  # Start annealing after this epoch
    kl_annealing_end = config.get('kl_annealing_end', 85)  # Complete annealing by this epoch
    
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
        'val_mmd': [],
        'noise_level': [],      # Track noise level over training
        'fm_loss': [],          # Track feature matching loss
        'g_kl_loss': [],        # Track generator KL loss
        'd_kl_loss': [],        # Track discriminator KL loss
        'total_kl_loss': [],    # Track total KL loss
        'kl_weight': [],        # Track KL weight throughout training
        'avg_logvar': []        # Track average logvar of Bayesian layers
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
        g_kl_losses = []
        d_kl_losses = []
        total_kl_losses = []

        # KL annealing: weight starts at 0, then linearly increases to final value
        if epoch < kl_annealing_start:
            kl_weight = 0.0  # No KL regularization in early training
        elif epoch < kl_annealing_end:
            # Linear annealing between start and end epochs
            progress = (epoch - kl_annealing_start) / (kl_annealing_end - kl_annealing_start)
            kl_weight = progress * kl_weight_final
        else:
            kl_weight = kl_weight_final  # Full KL weight after annealing period
        
        # Track the KL weight
        history['kl_weight'].append(kl_weight)
        
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
                
                # Calculate discriminator KL divergence
                d_kl = model.discriminator.kl_divergence()
                
                # Total discriminator loss (adding KL if needed)
                # Note: With non-Bayesian discriminator, d_kl will be zero
                d_loss = d_real_loss + d_fake_loss + gradient_penalty
                if d_kl.item() > 0 and kl_weight > 0:
                    d_loss = d_loss + kl_weight * d_kl
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
                    
                    # Calculate generator KL divergence
                    g_kl = model.generator.kl_divergence()
                    
                    # Total generator loss (adding KL)
                    g_total_loss = g_loss + 0.1 * fm_loss
                    if g_kl.item() > 0 and kl_weight > 0:
                        g_total_loss = g_total_loss + kl_weight * g_kl
                    
                    g_total_loss.backward()
                    g_optimizer.step()
                    
                    g_losses.append(g_loss.item())
                    g_kl_losses.append(g_kl.item())
                    history['fm_loss'].append(fm_loss.item())
                
                # Record losses
                d_losses.append(d_loss.item())
                d_kl_losses.append(d_kl.item())
                total_kl_losses.append((g_kl.item() if len(g_kl_losses) > 0 else 0) + d_kl.item())
                d_real_outputs.append(torch.mean(d_real_output).item())
                d_fake_outputs.append(torch.mean(d_fake_output).item())
                
                # Update EMA model
                historical_average(model, ema_model)
                
                # Update progress bar
                pbar.set_description(
                    f"Epoch {epoch+1}/{n_epochs}, D: {d_loss.item():.4f}, G: {g_loss.item() if len(g_losses) > 0 else 0:.4f}, "
                    f"KL: {total_kl_losses[-1]:.4f}, KL weight: {kl_weight:.6f}, Noise: {noise_level:.4f}"
                )
                
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
                gradient_penalty = 0.1 * (d_real_output ** 2).mean()  # Simplified regularization
                
                # Calculate discriminator KL divergence
                d_kl = model.discriminator.kl_divergence()
                
                # Total discriminator loss (adding KL if needed)
                # Note: With non-Bayesian discriminator, d_kl will be zero
                d_loss = d_real_loss + d_fake_loss + gradient_penalty
                if d_kl.item() > 0 and kl_weight > 0:
                    d_loss = d_loss + kl_weight * d_kl
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
                    
                    # Calculate generator KL divergence
                    g_kl = model.generator.kl_divergence()
                    
                    # Total generator loss (adding KL)
                    g_total_loss = g_loss + 0.1 * fm_loss
                    if g_kl.item() > 0 and kl_weight > 0:
                        g_total_loss = g_total_loss + kl_weight * g_kl
                    
                    g_total_loss.backward()
                    g_optimizer.step()
                    
                    g_losses.append(g_loss.item())
                    g_kl_losses.append(g_kl.item())
                    history['fm_loss'].append(fm_loss.item())
                
                # Record losses
                d_losses.append(d_loss.item())
                d_kl_losses.append(d_kl.item())
                total_kl_losses.append((g_kl.item() if len(g_kl_losses) > 0 else 0) + d_kl.item())
                d_real_outputs.append(torch.mean(d_real_output).item())
                d_fake_outputs.append(torch.mean(d_fake_output).item())
                
                # Update EMA model
                historical_average(model, ema_model)
                
                # Update progress bar
                pbar.set_description(
                    f"Epoch {epoch+1}/{n_epochs}, D: {d_loss.item():.4f}, G: {g_loss.item() if len(g_losses) > 0 else 0:.4f}, "
                    f"KL: {total_kl_losses[-1]:.4f}, KL weight: {kl_weight:.6f}, Noise: {noise_level:.4f}"
                )
        
        # Calculate average logvar of Bayesian layers
        avg_logvar = 0
        count = 0
        for module in model.generator.modules():
            if isinstance(module, (BayesLinear, BayesConv1d)):  # Add other Bayes types if used
                if hasattr(module, 'weight_logvar'):
                    avg_logvar += module.weight_logvar.data.mean().item()
                    count += 1
                if hasattr(module, 'bias_logvar'):
                    avg_logvar += module.bias_logvar.data.mean().item()
                    count += 1
        avg_logvar /= count if count > 0 else 1
        
        # Add avg_logvar to history tracking
        history['avg_logvar'].append(avg_logvar)
        
        # Record average losses for this epoch
        history['g_loss'].append(np.mean(g_losses))
        history['d_loss'].append(np.mean(d_losses))
        history['d_real'].append(np.mean(d_real_outputs))
        history['d_fake'].append(np.mean(d_fake_outputs))
        history['g_kl_loss'].append(np.mean(g_kl_losses))
        history['d_kl_loss'].append(np.mean(d_kl_losses))
        history['total_kl_loss'].append(np.mean(total_kl_losses))
        
        # Validation using MMD
        val_mmd = evaluate_mmd(
            ema_model, 
            val_loader, 
            data_type=data_type,
            signal_type=signal_type
        )
        history['val_mmd'].append(val_mmd)
        
        print(f"Epoch {epoch+1}/{n_epochs} - Val MMD: {val_mmd:.6f}, KL: {np.mean(total_kl_losses):.4f}, "
              f"KL weight: {kl_weight:.6f}, Noise: {noise_level:.4f}, D(real) - D(fake): {np.mean(d_real_outputs) - np.mean(d_fake_outputs):.4f}, "
              f"Avg LogVar: {avg_logvar:.4f}")
        
        # Check for improvement
        if val_mmd < best_mmd:
            best_mmd = val_mmd
            best_epoch = epoch
            best_model = deepcopy(ema_model)
            patience_counter = 0
            
            # Save best model
            model_path = os.path.join(model_dir, f"best_{data_type}_{signal_type}_bayes_gan.pt")
            torch.save(ema_model.state_dict(), model_path)
            print(f"New best model saved with MMD: {best_mmd:.6f}")
        else:
            patience_counter += 1
            
        # Save model at specified intervals
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(model_dir, f"{data_type}_{signal_type}_bayes_gan_e{epoch+1}.pt")
            torch.save(model.state_dict(), checkpoint_path)
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs!")
            break
            
    # Save final model
    final_path = os.path.join(model_dir, f"final_{data_type}_{signal_type}_bayes_gan.pt")
    torch.save(model.state_dict(), final_path)
    
    # Save training history
    history_path = os.path.join(model_dir, f"{data_type}_{signal_type}_bayes_gan_history.json")
    with open(history_path, 'w') as f:
        json.dump({k: [float(i) for i in v] for k, v in history.items()}, f)
    
    return {
        'history': history,
        'best_model': best_model,
        'best_epoch': best_epoch,
        'best_mmd': best_mmd
    }

def plot_bayes_gan_history(history, title, save_path=None):
    """
    Plot training history metrics specifically for Bayesian GANs.
    
    Args:
        history (dict): Training history containing metrics
        title (str): Title for the plot
        save_path (str, optional): Path to save the plot
        
    Returns:
        None
    """
    plt.figure(figsize=(16, 20))  # Increased figure size for the new plot
    
    # Plot generator and discriminator losses
    plt.subplot(5, 2, 1)  # Changed from 4,2 to 5,2 to add a row
    plt.plot(history['g_loss'], label='Generator Loss')
    plt.plot(history['d_loss'], label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('GAN Losses')
    
    # Plot discriminator outputs
    plt.subplot(5, 2, 2)
    plt.plot(history['d_real'], label='D(real)')
    plt.plot(history['d_fake'], label='D(fake)')
    plt.xlabel('Epoch')
    plt.ylabel('Discriminator Output')
    plt.legend()
    plt.title('Discriminator Outputs')
    
    # Plot validation MMD
    plt.subplot(5, 2, 3)
    plt.plot(history['val_mmd'], label='Validation MMD')
    plt.xlabel('Epoch')
    plt.ylabel('MMD')
    plt.legend()
    plt.title('Validation MMD (lower is better)')
    
    # Plot instance noise level
    if 'noise_level' in history:
        plt.subplot(5, 2, 4)
        plt.plot(history['noise_level'], label='Instance Noise (σ)')
        plt.xlabel('Epoch')
        plt.ylabel('Noise Level (σ)')
        plt.legend()
        plt.title('Instance Noise Level')
    
    # Plot feature matching loss
    if 'fm_loss' in history:
        plt.subplot(5, 2, 5)
        plt.plot(history['fm_loss'], label='Feature Matching Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Feature Matching Loss')
    
    # Plot KL divergence losses and KL weight
    plt.subplot(5, 2, 6)
    plt.plot(history['g_kl_loss'], label='Generator KL')
    plt.plot(history['d_kl_loss'], label='Discriminator KL')
    plt.plot(history['total_kl_loss'], label='Total KL')
    
    # Plot KL weight on a separate axis if available
    if 'kl_weight' in history:
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax2.plot(history['kl_weight'], 'r--', label='KL Weight')
        ax2.set_ylabel('KL Weight', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    else:
        plt.legend()
    
    plt.xlabel('Epoch')
    plt.ylabel('KL Divergence')
    plt.title('KL Divergence Losses and Weight')
    
    # Plot discriminator Wasserstein estimate
    plt.subplot(5, 2, 7)
    w_distance = np.array(history['d_real']) - np.array(history['d_fake'])
    plt.plot(w_distance, label='D(real) - D(fake)')
    plt.xlabel('Epoch')
    plt.ylabel('Wasserstein Estimate')
    plt.legend()
    plt.title('Wasserstein Distance Estimate')
    
    # Plot average logvar of Bayesian layers
    if 'avg_logvar' in history:
        plt.subplot(5, 2, 8)
        plt.plot(history['avg_logvar'], label='Avg LogVar')
        plt.xlabel('Epoch')
        plt.ylabel('Log Variance')
        plt.legend()
        plt.title('Average Log Variance of Bayesian Layers')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def main():
    """
    Main function to parse arguments and run training for Bayesian GANs.
    """
    parser = argparse.ArgumentParser(description='Train Variational Bayesian GAN models for physiological signals generation')
    
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
    
    # Bayesian model parameters
    parser.add_argument('--prior_type', type=str, default='gaussian', 
                        choices=['gaussian', 'scaled_mixture_gaussian', 'laplace'],
                        help='Type of prior distribution for Bayesian layers')
    parser.add_argument('--sigma', type=float, default=1.0, help='Standard deviation for Gaussian prior')
    parser.add_argument('--sigma1', type=float, default=1.0, help='Sigma1 for scaled mixture Gaussian prior')
    parser.add_argument('--sigma2', type=float, default=0.1, help='Sigma2 for scaled mixture Gaussian prior')
    parser.add_argument('--pi', type=float, default=0.5, help='Mixture coefficient for scaled mixture Gaussian prior')
    parser.add_argument('--b', type=float, default=1.0, help='Scale parameter for Laplace prior')
    parser.add_argument('--kl_weight', type=float, default=0.001, help='Final weight for KL divergence term')
    parser.add_argument('--kl_annealing_start', type=int, default=35, help='Epoch to start KL weight annealing')
    parser.add_argument('--kl_annealing_end', type=int, default=85, help='Epoch to end KL weight annealing')
    
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
    
    # Convert prior type string to enum
    prior_type = PriorType[args.prior_type.upper()]
    
    # Set up prior parameters based on prior type
    prior_params = {}
    if prior_type == PriorType.GAUSSIAN:
        prior_params = {'sigma': args.sigma}
    elif prior_type == PriorType.SCALED_MIXTURE_GAUSSIAN:
        prior_params = {'sigma1': args.sigma1, 'sigma2': args.sigma2, 'pi': args.pi}
    elif prior_type == PriorType.LAPLACE:
        prior_params = {'b': args.b}
    
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
    experiment_dir = os.path.join(args.model_dir, f"bayes_gan_{timestamp}")
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
        'kl_weight': args.kl_weight,
        'kl_annealing_start': args.kl_annealing_start,
        'kl_annealing_end': args.kl_annealing_end
    }
    
    # Train models based on the specified model_type
    if args.model_type in ['all', 'EDA']:
        # Train EDA time series model
        print("\n===== Training Bayesian EDA Time Series Model =====")
        # Get a sample batch to determine data dimensions
        eda_sample_data, _ = next(iter(dataloaders['EDA_series']['train']))
        
        # Determine sequence length from data shape - it's the dimension with largest value
        # If shape is [batch_size, seq_length, 1], then seq_length is the second dimension
        # If shape is [batch_size, 1, seq_length], then seq_length is the third dimension
        seq_length = max(eda_sample_data.shape[1], eda_sample_data.shape[2])
        
        # Choose between standard and multi-scale EDA model
        if args.use_multiscale:
            print("Using multi-scale generator for EDA signals")
            eda_model = BayesEDAMultiScaleTimeSeriesGAN(
                latent_dim=args.latent_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                num_classes=3,  # 3 stress conditions
                seq_length=seq_length,
                prior_type=prior_type,
                prior_params=prior_params
            )
        else:
            eda_model = BayesEDATimeSeriesGAN(
                latent_dim=args.latent_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                num_classes=3,  # 3 stress conditions
                seq_length=seq_length,
                prior_type=prior_type,
                prior_params=prior_params
            )
        
        eda_config = {**train_config, 'data_type': 'time_series', 'signal_type': 'EDA'}
        eda_results = train_gan(eda_model, dataloaders, eda_config)
        
        # Plot and save EDA training history
        plot_bayes_gan_history(
            eda_results['history'],
            'Bayesian EDA Time Series GAN Training',
            save_path=os.path.join(experiment_dir, 'bayes_eda_series_history.png')
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
            print("Using multi-scale generator for BVP signals")
            bvp_model = BayesBVPMultiScaleTimeSeriesGAN(
                latent_dim=args.latent_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers + 1,  # Extra layer for oscillatory patterns
                num_classes=3,  # 3 stress conditions
                seq_length=seq_length,
                prior_type=prior_type,
                prior_params=prior_params
            )
        else:
            bvp_model = BayesBVPTimeSeriesGAN(
                latent_dim=args.latent_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                num_classes=3,  # 3 stress conditions
                seq_length=seq_length,
                prior_type=prior_type,
                prior_params=prior_params
            )
        
        bvp_config = {**train_config, 'data_type': 'time_series', 'signal_type': 'BVP'}
        bvp_results = train_gan(bvp_model, dataloaders, bvp_config)
        
        # Plot and save BVP training history
        plot_bayes_gan_history(
            bvp_results['history'],
            'Bayesian BVP Time Series GAN Training',
            save_path=os.path.join(experiment_dir, 'bayes_bvp_series_history.png')
        )
    
    if args.model_type in ['all', 'features'] and dataloaders['features']['train'] is not None:
        # Train features model (HRV + EDA features)
        print("\n===== Training Bayesian Features Model (HRV + EDA) =====")
        feature_model = BayesFeatureGAN(
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            num_classes=3,  # 3 stress conditions
            prior_type=prior_type,
            prior_params=prior_params
        )
        
        feature_config = {**train_config, 'data_type': 'features'}
        feature_results = train_gan(feature_model, dataloaders, feature_config)
        
        # Plot and save feature training history
        plot_bayes_gan_history(
            feature_results['history'],
            'Bayesian Feature GAN Training (HRV + EDA)',
            save_path=os.path.join(experiment_dir, 'bayes_feature_history.png')
        )
    elif args.model_type == 'features' and dataloaders['features']['train'] is None:
        print("Skipping feature model training - feature data not available.")
    
    model_type_str = "all models" if args.model_type == 'all' else f"{args.model_type} model"
    print(f"\nTraining completed for Bayesian {model_type_str}! Models saved in {experiment_dir}")

if __name__ == "__main__":
    main()
