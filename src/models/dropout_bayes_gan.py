import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math

from src.models.base_gan import (TimeSeriesGAN, EDATimeSeriesGAN, 
                                 EDAMultiScaleTimeSeriesGAN, FeatureGAN,
                                 BVPTimeSeriesGAN, BVPMultiScaleTimeSeriesGAN,
                                 TransposeLayer)

class BayesianDropoutSignalGenerator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_layers, num_classes, seq_length, dropout_rate=0.2):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.dropout_rate = dropout_rate

        # Label embedding (add 1 to num_classes for padding idx 0)
        self.label_embedding = nn.Embedding(num_classes + 1, hidden_dim)

        # Initial noise processing with dropout
        self.noise_processor = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate)  # Bayesian dropout
        )

        # LSTM for temporal coherence with Bayesian dropout
        self.signal_lstm = nn.LSTM(
            input_size=hidden_dim * 2,  # Doubled for concatenated label
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate  # Bayesian dropout between LSTM layers
        )

        # Output network with dropout
        self.signal_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),  # Bayesian dropout
            nn.Linear(hidden_dim // 2, 1),
            nn.LeakyReLU(0.2)
        )

    def forward(self, z, labels):
        # Get label embeddings
        label_emb = self.label_embedding(labels)  # [batch_size, hidden_dim]

        # Process noise with label context
        combined_input = torch.cat([z, label_emb], dim=1)
        h = self.noise_processor(combined_input)

        # Prepare sequence input
        label_repeat = label_emb.unsqueeze(1).expand(-1, self.seq_length, -1)
        signal_in = h.unsqueeze(1).expand(-1, self.seq_length, -1)
        signal_in = torch.cat([signal_in, label_repeat], dim=2)

        # Generate temporal features
        signal_hidden, _ = self.signal_lstm(signal_in)
        
        # Generate signal
        signal_hidden = signal_hidden.reshape(-1, signal_hidden.size(-1))
        signal_series = self.signal_out(signal_hidden)
        signal_series = signal_series.view(-1, self.seq_length, 1)

        # Return in format [batch_size, seq_length, 1]
        return signal_series


class BayesianTimeSeriesDiscriminator(nn.Module):
    def __init__(self, seq_length, hidden_dim, num_classes, dropout_rate=0.2):
        super().__init__()
        
        # Label embedding (add 1 for padding idx 0)
        self.label_embedding = nn.Embedding(num_classes + 1, hidden_dim)
        
        # Signal processing with convolutional layers and dropout
        self.conv_layers = nn.Sequential(
            # Transpose from [batch, seq_len, 1] to [batch, 1, seq_len]
            TransposeLayer(1, 2),
            
            nn.Conv1d(1, hidden_dim // 4, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),  # Bayesian dropout
            
            nn.Conv1d(hidden_dim // 4, hidden_dim // 2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),  # Bayesian dropout
            
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate)   # Bayesian dropout
        )
        
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        # Classification layers
        self.flatten = nn.Flatten()
        
        # Dynamic first linear layer
        self.classifier_first_layer = None
        
        # Second classifier layer with dropout
        self.classifier_second_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),  # Bayesian dropout
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x, labels):
        # Get label embeddings
        label_emb = self.label_embedding(labels)
        
        # Process signal through convolutions
        conv_out = self.conv_layers(x)
        
        # Flatten and initialize first classifier layer if needed
        conv_flat = self.flatten(conv_out)
        
        if self.classifier_first_layer is None:
            input_features = conv_flat.shape[1]
            self.classifier_first_layer = nn.Sequential(
                nn.Linear(input_features + self.hidden_dim, self.hidden_dim),
                nn.Dropout(self.dropout_rate)  # Bayesian dropout
            ).to(conv_flat.device)
        
        # Concatenate with label and classify
        combined = torch.cat([conv_flat, label_emb], dim=1)
        hidden = self.classifier_first_layer(combined)
        
        # Final classification
        return self.classifier_second_layer(hidden)
    
    def extract_features(self, x, labels):
        """Extract intermediate features for feature matching loss."""
        # Get label embeddings
        label_emb = self.label_embedding(labels)
        
        # Process signal through convolutions
        conv_out = self.conv_layers(x)
        
        # Flatten and initialize first classifier layer if needed
        conv_flat = self.flatten(conv_out)
        
        if self.classifier_first_layer is None:
            input_features = conv_flat.shape[1]
            self.classifier_first_layer = nn.Sequential(
                nn.Linear(input_features + self.hidden_dim, self.hidden_dim),
                nn.Dropout(self.dropout_rate)  # Bayesian dropout
            ).to(conv_flat.device)
        
        # Concatenate with label and get hidden representation
        combined = torch.cat([conv_flat, label_emb], dim=1)
        
        # Get the output before dropout
        if isinstance(self.classifier_first_layer, nn.Sequential):
            # Get the Linear layer's output before the dropout
            hidden = self.classifier_first_layer[0](combined)
        else:
            hidden = self.classifier_first_layer(combined)
        
        return hidden


class BayesianMultiScaleSignalGenerator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_layers, num_classes, seq_length, dropout_rate=0.2):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        self.dropout_rate = dropout_rate
        
        # Label embedding (add 1 to num_classes for padding idx 0)
        self.label_embedding = nn.Embedding(num_classes + 1, hidden_dim)
        
        # Initial noise processing with dropout
        self.noise_processor = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate)  # Bayesian dropout
        )
        
        # Coarse branch (slow changes) - Larger hidden size, fewer layers
        self.lstm_coarse = nn.LSTM(
            input_size=hidden_dim * 2,  # Doubled for concatenated label
            hidden_size=hidden_dim,
            num_layers=max(1, num_layers-1),
            batch_first=True,
            dropout=dropout_rate  # Bayesian dropout
        )
        
        # Fine branch (fast oscillations) - Smaller hidden size, more layers
        self.lstm_fine = nn.LSTM(
            input_size=hidden_dim * 2,  # Doubled for concatenated label
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate  # Bayesian dropout
        )
        
        # Output processing with convolutional layers and dropout
        self.output_conv = nn.Sequential(
            # Transpose from [batch, seq_len, hidden_dim+hidden_dim//2] to [batch, hidden_dim+hidden_dim//2, seq_len]
            TransposeLayer(1, 2),
            nn.Conv1d(hidden_dim + hidden_dim // 2, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),  # Bayesian dropout
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),  # Bayesian dropout
            nn.Conv1d(hidden_dim // 2, 1, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            # Transpose back to [batch, seq_len, 1]
            TransposeLayer(1, 2)
        )
    
    def forward(self, z, labels):
        # Get label embeddings
        label_emb = self.label_embedding(labels)  # [batch_size, hidden_dim]
        
        # Process noise with label context
        combined_input = torch.cat([z, label_emb], dim=1)
        h = self.noise_processor(combined_input)
        
        # Prepare sequence input
        label_repeat = label_emb.unsqueeze(1).expand(-1, self.seq_length, -1)
        signal_in = h.unsqueeze(1).expand(-1, self.seq_length, -1)
        signal_in = torch.cat([signal_in, label_repeat], dim=2)  # [batch, seq_length, hidden_dim*2]
        
        # Generate temporal features through parallel paths
        coarse_out, _ = self.lstm_coarse(signal_in)  # [batch, seq_length, hidden_dim]
        fine_out, _ = self.lstm_fine(signal_in)      # [batch, seq_length, hidden_dim//2]
        
        # Combine coarse and fine features
        combined = torch.cat([coarse_out, fine_out], dim=2)  # [batch, seq_length, hidden_dim+hidden_dim//2]
        
        # Generate final signal
        signal_series = self.output_conv(combined)   # [batch, seq_length, 1]
        
        return signal_series


class BayesianFeatureGenerator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_classes, dropout_rate=0.2):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        # Label embedding
        self.label_embedding = nn.Embedding(num_classes + 1, hidden_dim)  # +1 for padding idx 0

        # Main feature processing network with dropout
        self.feature_processor = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),  # Bayesian dropout
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate)   # Bayesian dropout
        )

        # Output heads for different feature types with dropout
        self.hrv_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),  # Bayesian dropout
            nn.Linear(hidden_dim // 2, 5),  # 5 HRV features
            nn.SiLU()  # Smooth positive activation for physiological metrics
        )

        self.eda_features_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),  # Bayesian dropout
            nn.Linear(hidden_dim // 2, 3),  # 3 EDA features
            nn.SiLU()  # Smooth positive activation for physiological metrics
        )

    def forward(self, z, labels):
        # Get label embeddings
        label_emb = self.label_embedding(labels)

        # Combine noise and label information
        combined_input = torch.cat([z, label_emb], dim=1)
        
        # Process through main network
        features = self.feature_processor(combined_input)

        # Generate different feature types
        hrv_features = self.hrv_out(features)
        eda_features = self.eda_features_out(features)

        return {
            'HRV_features': hrv_features,
            'EDA_features': eda_features
        }

    def generate_with_uncertainty(self, z, labels, n_samples=10):
        """
        Generate multiple feature samples to estimate uncertainty.
        
        Args:
            z: Latent vector [batch_size, latent_dim]
            labels: Condition labels [batch_size]
            n_samples: Number of MC samples for uncertainty estimation
            
        Returns:
            mean_features: Dictionary of mean feature values
            std_features: Dictionary of standard deviations (uncertainty)
        """
        # Ensure model is in training mode to activate dropout
        training_mode = self.training
        self.train()
        
        hrv_samples = []
        eda_samples = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                features = self.generate(z, labels)
                hrv_samples.append(features['HRV_features'])
                eda_samples.append(features['EDA_features'])
        
        # Stack samples along a new dimension
        hrv_samples = torch.stack(hrv_samples, dim=0)  # [n_samples, batch_size, 5]
        eda_samples = torch.stack(eda_samples, dim=0)  # [n_samples, batch_size, 3]
        
        # Calculate mean and standard deviation
        hrv_mean = torch.mean(hrv_samples, dim=0)
        hrv_std = torch.std(hrv_samples, dim=0)
        
        eda_mean = torch.mean(eda_samples, dim=0)
        eda_std = torch.std(eda_samples, dim=0)
        
        # Restore original training mode
        self.train(training_mode)
        
        return {
            'mean': {
                'HRV_features': hrv_mean,
                'EDA_features': eda_mean
            },
            'std': {
                'HRV_features': hrv_std,
                'EDA_features': eda_std
            }
        }

    def generate_with_uncertainty_samples(self, z, labels, n_samples=10):
        """
        Generate multiple feature samples to visualize uncertainty.
        
        Args:
            z: Latent vector [batch_size, latent_dim]
            labels: Condition labels [batch_size]
            n_samples: Number of MC samples for uncertainty visualization
            
        Returns:
            dict: Dictionary with lists of samples for each feature type
                {
                    'HRV_features': List of n_samples HRV feature tensors,
                    'EDA_features': List of n_samples EDA feature tensors
                }
        """
        # Ensure model is in training mode to activate dropout
        training_mode = self.training
        self.train()
        
        hrv_samples = []
        eda_samples = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                features = self.generate(z, labels)
                hrv_samples.append(features['HRV_features'])
                eda_samples.append(features['EDA_features'])
        
        # Restore original training mode
        self.train(training_mode)
        
        return {
            'HRV_features': hrv_samples,
            'EDA_features': eda_samples
        }


class BayesianFeatureDiscriminator(nn.Module):
    def __init__(self, hidden_dim, num_classes, dropout_rate=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        # Label embedding
        self.label_embedding = nn.Embedding(num_classes + 1, hidden_dim)  # +1 for padding idx 0

        # Feature type-specific processing with dropout
        self.hrv_processor = nn.Sequential(
            nn.Linear(5, hidden_dim // 2),  # 5 HRV features
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate)  # Bayesian dropout
        )

        self.eda_processor = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),  # 3 EDA features
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate)  # Bayesian dropout
        )

        # Combined feature processing with dropout
        self.feature_processor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Combined features + label
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),  # Bayesian dropout
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate)   # Bayesian dropout
        )

        # Final classification
        self.classifier = nn.Linear(hidden_dim // 2, 1)  # Output raw logits

    def forward(self, features, labels):
        # Process each feature type
        hrv_encoded = self.hrv_processor(features['HRV_features'])
        eda_encoded = self.eda_processor(features['EDA_features'])

        # Get label embeddings
        label_emb = self.label_embedding(labels)

        # Combine all features
        combined = torch.cat([hrv_encoded, eda_encoded, label_emb], dim=1)

        # Process combined features
        features = self.feature_processor(combined)

        # Final classification (raw logits)
        return self.classifier(features)
        
    def extract_features(self, features, labels):
        """Extract intermediate features for feature matching loss."""
        # Process each feature type
        hrv_encoded = self.hrv_processor(features['HRV_features'])
        eda_encoded = self.eda_processor(features['EDA_features'])

        # Get label embeddings
        label_emb = self.label_embedding(labels)

        # Combine all features
        combined = torch.cat([hrv_encoded, eda_encoded, label_emb], dim=1)

        # Get the features before the last dropout layer
        partial_proc = self.feature_processor[:3]  # Process until before the dropout
        return partial_proc(combined)


# Bayesian Time Series GAN classes
class BayesianTimeSeriesGAN(TimeSeriesGAN):
    """Bayesian Dropout GAN for time series data with uncertainty estimation."""
    def __init__(self, 
                 latent_dim=100,
                 seq_length=30,
                 hidden_dim=128,
                 num_layers=2,
                 num_classes=3,
                 dropout_rate=0.2):
        super().__init__(
            latent_dim=latent_dim,
            seq_length=seq_length,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes
        )
        
        # Override generator and discriminator with Bayesian versions
        self.generator = BayesianDropoutSignalGenerator(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            seq_length=seq_length,
            dropout_rate=dropout_rate
        )
        
        self.discriminator = BayesianTimeSeriesDiscriminator(
            seq_length=seq_length,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )
        
        self.dropout_rate = dropout_rate
    
    def generate_with_uncertainty(self, z, labels, n_samples=10):
        """
        Generate multiple samples to estimate uncertainty.
        
        Args:
            z: Latent vector [batch_size, latent_dim]
            labels: Condition labels [batch_size]
            n_samples: Number of MC samples for uncertainty estimation
            
        Returns:
            mean: Mean of generated samples
            std: Standard deviation of generated samples (uncertainty)
        """
        # Ensure model is in training mode to activate dropout
        training_mode = self.training
        self.train()
        
        samples = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                sample = self.generate(z, labels)
                samples.append(sample)
        
        # Stack samples along a new dimension [n_samples, batch_size, seq_length, 1]
        samples = torch.stack(samples, dim=0)
        
        # Calculate mean and standard deviation
        mean = torch.mean(samples, dim=0)
        std = torch.std(samples, dim=0)
        
        # Restore original training mode
        self.train(training_mode)
        
        return mean, std

    def generate_with_uncertainty_samples(self, z, labels, n_samples=10):
        """
        Generate multiple samples to visualize uncertainty.
        
        Args:
            z: Latent vector [batch_size, latent_dim]
            labels: Condition labels [batch_size]
            n_samples: Number of MC samples for uncertainty visualization
            
        Returns:
            list: List of generated samples, each of shape [batch_size, seq_length, 1]
        """
        # Ensure model is in training mode to activate dropout
        training_mode = self.training
        self.train()
        
        samples = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                sample = self.generate(z, labels)
                samples.append(sample)
        
        # Restore original training mode
        self.train(training_mode)
        
        return samples


class BayesianEDAGenerator(nn.Module):
    """
    Enhanced Bayesian generator specialized for EDA signals.
    Includes components for tonic (slow baseline) and phasic (fast responses) components.
    """
    def __init__(self, latent_dim, hidden_dim, num_layers, num_classes, seq_length, dropout_rate=0.2):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.dropout_rate = dropout_rate

        # Label embedding (add 1 to num_classes for padding idx 0)
        self.label_embedding = nn.Embedding(num_classes + 1, hidden_dim)

        # Initial noise processing with dropout
        self.noise_processor = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate)  # Bayesian dropout
        )

        # LSTM for tonic component (slow baseline changes) with Bayesian dropout
        self.tonic_lstm = nn.LSTM(
            input_size=hidden_dim * 2,  # Doubled for concatenated label
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate  # Bayesian dropout between LSTM layers
        )

        # LSTM for phasic component (fast responses) with Bayesian dropout
        self.phasic_lstm = nn.LSTM(
            input_size=hidden_dim * 2,  # Doubled for concatenated label
            hidden_size=hidden_dim // 2,
            num_layers=num_layers + 1,  # Extra layer for detailed responses
            batch_first=True,
            dropout=dropout_rate  # Bayesian dropout between LSTM layers
        )

        # Output network with dropout for tonic component
        self.tonic_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),  # Bayesian dropout
            nn.Linear(hidden_dim // 2, 1),
            nn.LeakyReLU(0.2)  # EDA is always non-negative
        )

        # Output network with dropout for phasic component
        self.phasic_out = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),  # Bayesian dropout
            nn.Linear(hidden_dim // 4, 1),
            nn.LeakyReLU(0.2)  # EDA is always non-negative
        )
        
        # Condition-specific adaptation layers
        self.stress_layer = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=1, bias=False, groups=1),
        )
        self.stress_layer[0].weight.data.fill_(1.5)  # Increased amplitude in stress
        
        self.amusement_layer = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=5, padding=2, bias=True, groups=1),
        )
        # Initialize smoothing filter weights for amusement
        self.amusement_layer[0].weight.data.fill_(0)
        self.amusement_layer[0].weight.data[:, :, 0] = 0.1
        self.amusement_layer[0].weight.data[:, :, 1] = 0.2
        self.amusement_layer[0].weight.data[:, :, 2] = 0.4
        self.amusement_layer[0].weight.data[:, :, 3] = 0.2
        self.amusement_layer[0].weight.data[:, :, 4] = 0.1
        if hasattr(self.amusement_layer[0], 'bias'):
            self.amusement_layer[0].bias.data.fill_(-0.2)

    def forward(self, z, labels):
        batch_size = z.size(0)
        
        # Get label embeddings
        label_emb = self.label_embedding(labels)  # [batch_size, hidden_dim]

        # Process noise with label context
        combined_input = torch.cat([z, label_emb], dim=1)
        h = self.noise_processor(combined_input)

        # Prepare sequence input
        label_repeat = label_emb.unsqueeze(1).expand(-1, self.seq_length, -1)
        signal_in = h.unsqueeze(1).expand(-1, self.seq_length, -1)
        signal_in = torch.cat([signal_in, label_repeat], dim=2)

        # Generate tonic component (slow baseline changes)
        tonic_hidden, _ = self.tonic_lstm(signal_in)
        tonic_series = self.tonic_out(tonic_hidden)  # [batch, seq_length, 1]

        # Generate phasic component (fast responses)
        phasic_hidden, _ = self.phasic_lstm(signal_in)
        phasic_series = self.phasic_out(phasic_hidden)  # [batch, seq_length, 1]

        # Combine components with appropriate weighting
        # Tonic component (70%) provides the baseline
        # Phasic component (30%) adds the skin conductance responses
        combined_signal = 0.7 * tonic_series + 0.3 * phasic_series

        # Create output tensor for condition-specific outputs
        output_signal = torch.zeros_like(combined_signal)
        
        # Apply condition-specific transformations
        for i in range(batch_size):
            sample_signal = combined_signal[i:i+1]  # Keep batch dimension
            condition = labels[i].item()
            
            if condition == 1:  # Baseline - use as is
                output_signal[i:i+1] = sample_signal
            elif condition == 2:  # Stress - higher amplitude
                transposed = sample_signal.transpose(1, 2)  # [1, 1, seq_len]
                scaled = self.stress_layer(transposed)
                output_signal[i:i+1] = scaled.transpose(1, 2)
            elif condition == 3:  # Amusement - smoother
                transposed = sample_signal.transpose(1, 2)  # [1, 1, seq_len]
                smoothed = self.amusement_layer(transposed)
                output_signal[i:i+1] = smoothed.transpose(1, 2)
        
        return output_signal


class BayesianEDATimeSeriesGAN(BayesianTimeSeriesGAN):
    """Bayesian Dropout GAN specialized for EDA signals."""
    def __init__(self,
                 latent_dim=100,
                 seq_length=30,
                 hidden_dim=128,
                 num_layers=2,
                 num_classes=3,
                 dropout_rate=0.2):
        super().__init__(
            latent_dim=latent_dim,
            seq_length=seq_length,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )
        
        # Override with EDA-specific Bayesian generator
        self.generator = BayesianEDAGenerator(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            seq_length=seq_length,
            dropout_rate=dropout_rate
        )


class BayesianEDAMultiScaleGenerator(nn.Module):
    """
    Enhanced Bayesian multi-scale generator specialized for EDA signals.
    Handles both tonic (slow baseline) and phasic (fast responses) components.
    """
    def __init__(self, latent_dim, hidden_dim, num_layers, num_classes, seq_length, dropout_rate=0.2):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        self.dropout_rate = dropout_rate
        
        # Label embedding (add 1 to num_classes for padding idx 0)
        self.label_embedding = nn.Embedding(num_classes + 1, hidden_dim)
        
        # Initial noise processing with dropout
        self.noise_processor = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate)  # Bayesian dropout
        )
        
        # Coarse branch uses fewer layers than fine branch since tonic components
        # vary more slowly and need less processing depth.
        # Note: Use num_layers >= 3 to enable dropout in this branch.
        self.lstm_coarse = nn.LSTM(
            input_size=hidden_dim * 2,  # Doubled for concatenated label
            hidden_size=hidden_dim,
            num_layers=max(1, num_layers-1),
            batch_first=True,
            dropout=dropout_rate  # Bayesian dropout
        )
        
        # Fine branch (phasic component) with Bayesian dropout
        self.lstm_fine = nn.LSTM(
            input_size=hidden_dim * 2,  # Doubled for concatenated label
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate  # Bayesian dropout
        )
        
        # Output processing with convolutional layers and dropout
        self.output_conv = nn.Sequential(
            TransposeLayer(1, 2),
            nn.Conv1d(hidden_dim + hidden_dim // 2, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),  # Bayesian dropout
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),  # Bayesian dropout
            nn.Conv1d(hidden_dim // 2, 1, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),  # EDA is always non-negative
            TransposeLayer(1, 2)
        )
        
        # Condition-specific adaptation layers
        self.stress_layer = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=1, bias=False, groups=1),
        )
        self.stress_layer[0].weight.data.fill_(1.5)  # Increased amplitude in stress
        
        self.amusement_layer = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=5, padding=2, bias=True, groups=1),
        )
        # Initialize smoothing filter weights for amusement
        self.amusement_layer[0].weight.data.fill_(0)
        self.amusement_layer[0].weight.data[:, :, 0] = 0.1
        self.amusement_layer[0].weight.data[:, :, 1] = 0.2
        self.amusement_layer[0].weight.data[:, :, 2] = 0.4
        self.amusement_layer[0].weight.data[:, :, 3] = 0.2
        self.amusement_layer[0].weight.data[:, :, 4] = 0.1
        if hasattr(self.amusement_layer[0], 'bias'):
            self.amusement_layer[0].bias.data.fill_(-0.2)
    
    def forward(self, z, labels):
        batch_size = z.size(0)
        
        # Get label embeddings
        label_emb = self.label_embedding(labels)  # [batch_size, hidden_dim]
        
        # Process noise with label context
        combined_input = torch.cat([z, label_emb], dim=1)
        h = self.noise_processor(combined_input)
        
        # Prepare sequence input
        label_repeat = label_emb.unsqueeze(1).expand(-1, self.seq_length, -1)
        signal_in = h.unsqueeze(1).expand(-1, self.seq_length, -1)
        signal_in = torch.cat([signal_in, label_repeat], dim=2)  # [batch, seq_length, hidden_dim*2]
        
        # Generate temporal features through parallel paths
        coarse_out, _ = self.lstm_coarse(signal_in)  # [batch, seq_length, hidden_dim]
        fine_out, _ = self.lstm_fine(signal_in)      # [batch, seq_length, hidden_dim//2]
        
        # Combine coarse and fine features
        combined = torch.cat([coarse_out, fine_out], dim=2)  # [batch, seq_length, hidden_dim+hidden_dim//2]
        
        # Generate final signal
        signal_series = self.output_conv(combined)   # [batch, seq_length, 1]
        
        # Create output tensor for condition-specific outputs
        output_signal = torch.zeros_like(signal_series)
        
        # Apply condition-specific transformations
        for i in range(batch_size):
            sample_signal = signal_series[i:i+1]  # Keep batch dimension
            condition = labels[i].item()
            
            if condition == 1:  # Baseline - use as is
                output_signal[i:i+1] = sample_signal
            elif condition == 2:  # Stress - higher amplitude
                transposed = sample_signal.transpose(1, 2)  # [1, 1, seq_len]
                scaled = self.stress_layer(transposed)
                output_signal[i:i+1] = scaled.transpose(1, 2)
            elif condition == 3:  # Amusement - smoother
                transposed = sample_signal.transpose(1, 2)  # [1, 1, seq_len]
                smoothed = self.amusement_layer(transposed)
                output_signal[i:i+1] = smoothed.transpose(1, 2)
        
        return output_signal


class BayesianEDAMultiScaleTimeSeriesGAN(BayesianTimeSeriesGAN):
    """Bayesian Dropout GAN with multi-scale generator for EDA signals."""
    def __init__(self,
                 latent_dim=100,
                 seq_length=30,
                 hidden_dim=128,
                 num_layers=2,
                 num_classes=3,
                 dropout_rate=0.2):
        super().__init__(
            latent_dim=latent_dim,
            seq_length=seq_length,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )
        
        # Create a custom Bayesian multi-scale generator for EDA
        self.generator = BayesianEDAMultiScaleGenerator(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            seq_length=seq_length,
            dropout_rate=dropout_rate
        )


class BayesianFeatureGAN(FeatureGAN):
    """Bayesian Dropout GAN for generating physiological features with uncertainty."""
    def __init__(self, 
                 latent_dim=100,
                 hidden_dim=128,
                 num_classes=3,
                 dropout_rate=0.2):
        super().__init__(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes
        )
        
        # Override generator and discriminator with Bayesian versions
        self.generator = BayesianFeatureGenerator(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )
        
        self.discriminator = BayesianFeatureDiscriminator(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )
        
        self.dropout_rate = dropout_rate
    
    def generate_with_uncertainty(self, z, labels, n_samples=10):
        """
        Generate multiple feature samples to estimate uncertainty.
        
        Args:
            z: Latent vector [batch_size, latent_dim]
            labels: Condition labels [batch_size]
            n_samples: Number of MC samples for uncertainty estimation
            
        Returns:
            mean_features: Dictionary of mean feature values
            std_features: Dictionary of standard deviations (uncertainty)
        """
        # Ensure model is in training mode to activate dropout
        training_mode = self.training
        self.train()
        
        hrv_samples = []
        eda_samples = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                features = self.generate(z, labels)
                hrv_samples.append(features['HRV_features'])
                eda_samples.append(features['EDA_features'])
        
        # Stack samples along a new dimension
        hrv_samples = torch.stack(hrv_samples, dim=0)  # [n_samples, batch_size, 5]
        eda_samples = torch.stack(eda_samples, dim=0)  # [n_samples, batch_size, 3]
        
        # Calculate mean and standard deviation
        hrv_mean = torch.mean(hrv_samples, dim=0)
        hrv_std = torch.std(hrv_samples, dim=0)
        
        eda_mean = torch.mean(eda_samples, dim=0)
        eda_std = torch.std(eda_samples, dim=0)
        
        # Restore original training mode
        self.train(training_mode)
        
        return {
            'mean': {
                'HRV_features': hrv_mean,
                'EDA_features': eda_mean
            },
            'std': {
                'HRV_features': hrv_std,
                'EDA_features': eda_std
            }
        }

    def generate_with_uncertainty_samples(self, z, labels, n_samples=10):
        """
        Generate multiple feature samples to visualize uncertainty.
        
        Args:
            z: Latent vector [batch_size, latent_dim]
            labels: Condition labels [batch_size]
            n_samples: Number of MC samples for uncertainty visualization
            
        Returns:
            dict: Dictionary with lists of samples for each feature type
                {
                    'HRV_features': List of n_samples HRV feature tensors,
                    'EDA_features': List of n_samples EDA feature tensors
                }
        """
        # Ensure model is in training mode to activate dropout
        training_mode = self.training
        self.train()
        
        hrv_samples = []
        eda_samples = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                features = self.generate(z, labels)
                hrv_samples.append(features['HRV_features'])
                eda_samples.append(features['EDA_features'])
        
        # Restore original training mode
        self.train(training_mode)
        
        return {
            'HRV_features': hrv_samples,
            'EDA_features': eda_samples
        }


class BayesianBVPGenerator(nn.Module):
    """
    Enhanced Bayesian generator specialized for Blood Volume Pulse (BVP) signals.
    BVP signals are oscillatory with distinct cardiac cycles.
    Includes explicit oscillatory components and Bayesian dropout for uncertainty estimation.
    """
    def __init__(self, latent_dim, hidden_dim, num_layers, num_classes, seq_length, dropout_rate=0.2):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.dropout_rate = dropout_rate

        # Label embedding (add 1 to num_classes for padding idx 0)
        self.label_embedding = nn.Embedding(num_classes + 1, hidden_dim)

        # Initial noise processing with dropout
        self.noise_processor = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate)  # Bayesian dropout
        )

        # LSTM for cardiac rhythm generation with Bayesian dropout
        self.cardiac_lstm = nn.LSTM(
            input_size=hidden_dim * 2,  # Doubled for concatenated label
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate  # Bayesian dropout between LSTM layers
        )

        # Oscillation parameters network with dropout
        self.oscillation_params = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),  # Bayesian dropout
            nn.Linear(hidden_dim // 2, 3),  # Frequency, amplitude, phase
        )

        # Final output network with dropout
        self.signal_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),  # Bayesian dropout
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()  # Use tanh for oscillatory signals
        )
        
        # Condition-specific adaptation layers
        self.stress_layer = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=1, bias=False, groups=1),
        )
        self.stress_layer[0].weight.data.fill_(1.8)  # Increased amplitude in stress
        
        self.amusement_layer = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=True, groups=1),
        )
        # Initialize smoothing filter weights for amusement
        self.amusement_layer[0].weight.data.fill_(0)
        self.amusement_layer[0].weight.data[:, :, 0] = 0.15
        self.amusement_layer[0].weight.data[:, :, 1] = 0.7
        self.amusement_layer[0].weight.data[:, :, 2] = 0.15
        if hasattr(self.amusement_layer[0], 'bias'):
            self.amusement_layer[0].bias.data.fill_(-0.3)

    def forward(self, z, labels):
        batch_size = z.size(0)
        
        # Get label embeddings
        label_emb = self.label_embedding(labels)  # [batch_size, hidden_dim]

        # Process noise with label context
        combined_input = torch.cat([z, label_emb], dim=1)
        h = self.noise_processor(combined_input)

        # Prepare sequence input
        label_repeat = label_emb.unsqueeze(1).expand(-1, self.seq_length, -1)
        signal_in = h.unsqueeze(1).expand(-1, self.seq_length, -1)
        signal_in = torch.cat([signal_in, label_repeat], dim=2)

        # Generate temporal features
        cardiac_hidden, _ = self.cardiac_lstm(signal_in)
        
        # Generate explicit oscillatory pattern for each sample
        cardiac_oscillations = torch.zeros((batch_size, self.seq_length, 1), device=z.device)
        
        for i in range(batch_size):
            # Extract oscillation parameters from the hidden state
            h_avg = cardiac_hidden[i].mean(dim=0)
            osc_params = self.oscillation_params(h_avg)
            
            # Use parameters to generate appropriate oscillation frequency
            freq = 0.23 + 0.04 * torch.tanh(osc_params[0])  # Range 0.19-0.27 Hz
            amp = 2.0 + 1.5 * torch.tanh(osc_params[1])  # Range 0.5-3.5
            phase = math.pi * torch.sigmoid(osc_params[2])  # 0 to π
            
            # Generate time steps
            t = torch.linspace(0, 1, self.seq_length, device=z.device)
            
            # Create oscillatory signal
            osc = amp * torch.sin(2 * math.pi * freq * self.seq_length * t + phase)
            
            # Add to the batch
            cardiac_oscillations[i, :, 0] = osc
        
        # Generate base cardiac signal
        signal_series_list = []
        
        # Process each time step
        for t in range(self.seq_length):
            h_t = cardiac_hidden[:, t, :]
            out_t = self.signal_out(h_t)
            signal_series_list.append(out_t)
        
        # Stack time steps
        base_signal = torch.stack(signal_series_list, dim=1)  # [batch, seq_length, 1]
        
        # Combine base signal with cardiac oscillations
        enhanced_signal = 0.6 * base_signal + 0.4 * cardiac_oscillations
        
        # Create output tensor for condition-specific outputs
        output_signal = torch.zeros_like(enhanced_signal)
        
        # Apply condition-specific transformations
        for i in range(batch_size):
            sample_signal = enhanced_signal[i:i+1]  # Keep batch dimension
            condition = labels[i].item()
            
            if condition == 1:  # Baseline - use as is
                output_signal[i:i+1] = sample_signal
            elif condition == 2:  # Stress - higher amplitude
                transposed = sample_signal.transpose(1, 2)  # [1, 1, seq_len]
                scaled = self.stress_layer(transposed)
                output_signal[i:i+1] = scaled.transpose(1, 2)
            elif condition == 3:  # Amusement - smoother
                transposed = sample_signal.transpose(1, 2)  # [1, 1, seq_len]
                smoothed = self.amusement_layer(transposed)
                output_signal[i:i+1] = smoothed.transpose(1, 2)
        
        return output_signal


class BayesianBVPTimeSeriesGAN(BayesianTimeSeriesGAN):
    """Bayesian Dropout GAN specialized for BVP signals."""
    def __init__(self,
                 latent_dim=100,
                 seq_length=30,
                 hidden_dim=128,
                 num_layers=2,
                 num_classes=3,
                 dropout_rate=0.2):
        super().__init__(
            latent_dim=latent_dim,
            seq_length=seq_length,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )
        
        # Override with BVP-specific Bayesian generator
        self.generator = BayesianBVPGenerator(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            seq_length=seq_length,
            dropout_rate=dropout_rate
        )


class BayesianBVPMultiScaleGenerator(nn.Module):
    """
    Enhanced Bayesian multi-scale generator specialized for Blood Volume Pulse (BVP) signals.
    Handles both the fast oscillations of cardiac cycles and slower baseline variations.
    Includes explicit oscillatory components and Bayesian dropout for uncertainty estimation.
    """
    def __init__(self, latent_dim, hidden_dim, num_layers, num_classes, seq_length, dropout_rate=0.2):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        self.dropout_rate = dropout_rate
        
        # Label embedding (add 1 to num_classes for padding idx 0)
        self.label_embedding = nn.Embedding(num_classes + 1, hidden_dim)
        
        # Initial noise processing with dropout
        self.noise_processor = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate)  # Bayesian dropout
        )
        
        # Coarse branch (slow baseline changes) with Bayesian dropout
        self.lstm_coarse = nn.LSTM(
            input_size=hidden_dim * 2,  # Doubled for concatenated label
            hidden_size=hidden_dim,
            num_layers=max(1, num_layers-1),
            batch_first=True,
            dropout=dropout_rate  # Bayesian dropout
        )
        
        # Enhanced fine branch (cardiac oscillations) with Bayesian dropout
        self.lstm_fine = nn.LSTM(
            input_size=hidden_dim * 2,  # Doubled for concatenated label
            hidden_size=hidden_dim,  # Increased from hidden_dim//2
            num_layers=num_layers + 2,  # Add more layers for oscillatory patterns
            batch_first=True,
            dropout=dropout_rate  # Bayesian dropout
        )
        
        # Explicit oscillation generator with dropout
        self.oscillation_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),  # Bayesian dropout
            nn.Linear(hidden_dim, self.seq_length),
            nn.Tanh()  # Oscillatory component should have zero mean
        )
        
        # Cardiac parameter network with dropout
        self.cardiac_params = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),  # Bayesian dropout
            nn.Linear(hidden_dim // 2, 4),  # [frequency, amplitude, phase, baseline]
            nn.Sigmoid()  # Bound parameters to reasonable ranges
        )
        
        # Forced periodic component generator
        class SinusoidalEncoding(nn.Module):
            def __init__(self, seq_length, hidden_dim):
                super().__init__()
                self.frequencies = nn.Parameter(
                    torch.tensor([0.19, 0.21, 0.23, 0.25, 0.27, 0.29]), 
                    requires_grad=False
                )  # Typical BVP frequencies
                self.seq_length = seq_length
                self.projection = nn.Sequential(
                    nn.Linear(len(self.frequencies) * 2, hidden_dim // 2),
                    nn.Dropout(dropout_rate)  # Add Bayesian dropout
                )
            
            def forward(self, batch_size, device):
                t = torch.linspace(0, 1, self.seq_length).to(device)
                t = t.view(1, -1, 1)  # [1, seq_length, 1]
                t = t.repeat(batch_size, 1, 1)  # [batch_size, seq_length, 1]
                
                freqs = self.frequencies.view(1, 1, -1)  # [1, 1, num_frequencies]
                phases = 2 * math.pi * t * freqs * self.seq_length
                
                sin_features = torch.sin(phases)
                cos_features = torch.cos(phases)
                
                combined = torch.cat([sin_features, cos_features], dim=-1)
                return self.projection(combined)
        
        self.sinusoidal_encoding = SinusoidalEncoding(seq_length, hidden_dim)
        
        # Enhanced output processing with residual connections and dropout
        self.output_conv = nn.Sequential(
            TransposeLayer(1, 2),
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),  # Bayesian dropout
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),  # Bayesian dropout
            nn.Conv1d(hidden_dim // 2, 1, kernel_size=3, padding=1),
            nn.Tanh(),  # Use tanh for oscillatory signals
            TransposeLayer(1, 2)
        )
        
        # Condition-specific adaptation layers
        self.stress_layer = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=1, bias=False, groups=1),
        )
        self.stress_layer[0].weight.data.fill_(1.8)  # Increased amplitude in stress
        
        self.amusement_layer = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=True, groups=1),
        )
        # Initialize smoothing filter weights for amusement
        self.amusement_layer[0].weight.data.fill_(0)
        self.amusement_layer[0].weight.data[:, :, 0] = 0.15
        self.amusement_layer[0].weight.data[:, :, 1] = 0.7
        self.amusement_layer[0].weight.data[:, :, 2] = 0.15
        if hasattr(self.amusement_layer[0], 'bias'):
            self.amusement_layer[0].bias.data.fill_(-0.3)
    
    def forward(self, z, labels):
        batch_size = z.size(0)
        
        # Get label embeddings
        label_emb = self.label_embedding(labels)  # [batch_size, hidden_dim]
        
        # Process noise with label context
        combined_input = torch.cat([z, label_emb], dim=1)
        h = self.noise_processor(combined_input)
        
        # Prepare sequence input
        label_repeat = label_emb.unsqueeze(1).expand(-1, self.seq_length, -1)
        signal_in = h.unsqueeze(1).expand(-1, self.seq_length, -1)
        signal_in = torch.cat([signal_in, label_repeat], dim=2)  # [batch, seq_length, hidden_dim*2]
        
        # Generate temporal features through parallel paths
        coarse_out, _ = self.lstm_coarse(signal_in)  # [batch, seq_length, hidden_dim]
        fine_out, _ = self.lstm_fine(signal_in)      # [batch, seq_length, hidden_dim]
        
        # Generate explicit oscillatory pattern based on the noise vector
        oscillations = self.oscillation_generator(h)  # [batch_size, seq_length]
        oscillations = oscillations.unsqueeze(-1)     # [batch_size, seq_length, 1]
        
        # Generate sinusoidal encoding for forced periodicity
        sinusoidal_features = self.sinusoidal_encoding(batch_size, z.device)
        sinusoidal_features_reshaped = sinusoidal_features.mean(dim=2, keepdim=True)
        
        # Generate cardiac parameters for each sample
        cardiac_params = []
        for i in range(batch_size):
            # Combine coarse and fine features
            features = torch.cat([coarse_out[i].mean(dim=0), fine_out[i].mean(dim=0)])
            params = self.cardiac_params(features)
            cardiac_params.append(params)
        cardiac_params = torch.stack(cardiac_params)
        
        # Generate cardiac oscillations using parameters
        cardiac_oscillations = torch.zeros((batch_size, self.seq_length, 1), device=z.device)
        t = torch.linspace(0, 1, self.seq_length, device=z.device)
        
        for i in range(batch_size):
            freq = 0.19 + 0.1 * cardiac_params[i, 0]  # Range: 0.19-0.29 Hz
            amp = 1.0 + 2.0 * cardiac_params[i, 1]    # Range: 1.0-3.0
            phase = 2 * math.pi * cardiac_params[i, 2] # Range: 0-2π
            baseline = -0.5 + cardiac_params[i, 3]     # Range: -0.5-0.5
            
            osc = amp * torch.sin(2 * math.pi * freq * self.seq_length * t + phase) + baseline
            cardiac_oscillations[i, :, 0] = osc
        
        # Combine coarse and fine features
        combined = torch.cat([coarse_out, fine_out], dim=2)  # [batch, seq_length, hidden_dim*2]
        
        # Generate base signal
        base_signal = self.output_conv(combined)  # [batch, seq_length, 1]
        
        # Add the oscillatory components with appropriate weighting
        enhanced_signal = (
            0.1 * base_signal + 
            0.1 * oscillations +
            0.7 * cardiac_oscillations +
            0.1 * sinusoidal_features_reshaped
        )
        
        # Create output tensor for condition-specific outputs
        output_signal = torch.zeros_like(enhanced_signal)
        
        # Apply condition-specific transformations
        for i in range(batch_size):
            sample_signal = enhanced_signal[i:i+1]  # Keep batch dimension
            condition = labels[i].item()
            
            if condition == 1:  # Baseline - use as is
                output_signal[i:i+1] = sample_signal
            elif condition == 2:  # Stress - higher amplitude
                transposed = sample_signal.transpose(1, 2)  # [1, 1, seq_len]
                scaled = self.stress_layer(transposed)
                output_signal[i:i+1] = scaled.transpose(1, 2)
            elif condition == 3:  # Amusement - smoother
                transposed = sample_signal.transpose(1, 2)  # [1, 1, seq_len]
                smoothed = self.amusement_layer(transposed)
                output_signal[i:i+1] = smoothed.transpose(1, 2)
        
        return output_signal


class BayesianBVPMultiScaleTimeSeriesGAN(BayesianTimeSeriesGAN):
    """Bayesian Dropout GAN with multi-scale generator for BVP signals."""
    def __init__(self,
                 latent_dim=100,
                 seq_length=30,
                 hidden_dim=128,
                 num_layers=2,
                 num_classes=3,
                 dropout_rate=0.2):
        super().__init__(
            latent_dim=latent_dim,
            seq_length=seq_length,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )
        
        # Create a custom Bayesian multi-scale generator for BVP
        self.generator = BayesianBVPMultiScaleGenerator(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers + 2,  # Add more layers for oscillatory patterns
            num_classes=num_classes,
            seq_length=seq_length,
            dropout_rate=dropout_rate
        )
