import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math

class SignalGenerator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_layers, num_classes, seq_length):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_length = seq_length

        # Label embedding (add 1 to num_classes for padding idx 0)
        self.label_embedding = nn.Embedding(num_classes + 1, hidden_dim)

        # Initial noise processing
        self.noise_processor = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2)
        )

        # LSTM for temporal coherence
        self.signal_lstm = nn.LSTM(
            input_size=hidden_dim * 2,  # Doubled for concatenated label
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )

        # Output network for EDA signals (smoother and slower-varying)
        self.signal_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
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

        # Return in format [batch_size, seq_length, 1] - no transpose needed
        return signal_series

class TimeSeriesDiscriminator(nn.Module):
    def __init__(self, seq_length, hidden_dim, num_classes):
        super().__init__()
        
        # Label embedding (add 1 for padding idx 0)
        self.label_embedding = nn.Embedding(num_classes + 1, hidden_dim)
        
        # EDA signal processing with wider kernels for slower patterns
        self.conv_layers = nn.Sequential(
            # Transpose from [batch, seq_len, 1] to [batch, 1, seq_len]
            TransposeLayer(1, 2),
            
            nn.Conv1d(1, hidden_dim // 4, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(hidden_dim // 4, hidden_dim // 2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2)
        )
        
        # Calculate flattened size - we'll compute this dynamically in forward pass
        # instead of hardcoding dimensions
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        
        # Classification layers
        self.flatten = nn.Flatten()
        
        # Use a more adaptive approach with a first linear layer 
        # that adapts to input size
        self.classifier_first_layer = None  # Will be initialized in first forward pass
        
        # Second classifier layer (fixed size)
        self.classifier_second_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x, labels):
        # Get label embeddings
        label_emb = self.label_embedding(labels)
        
        # Process signal through convolutions
        # Input x shape: [batch_size, seq_length, 1]
        conv_out = self.conv_layers(x)
        
        # Flatten and initialize first classifier layer if needed
        conv_flat = self.flatten(conv_out)
        
        if self.classifier_first_layer is None:
            input_features = conv_flat.shape[1]
            self.classifier_first_layer = nn.Linear(input_features + self.hidden_dim, self.hidden_dim).to(conv_flat.device)
        
        # Concatenate with label and classify
        combined = torch.cat([conv_flat, label_emb], dim=1)
        hidden = self.classifier_first_layer(combined)
        
        # Final classification
        return self.classifier_second_layer(hidden)
    
    def extract_features(self, x, labels):
        """
        Extract intermediate features for feature matching loss.
        Returns activations from the penultimate layer.
        """
        # Get label embeddings
        label_emb = self.label_embedding(labels)
        
        # Process signal through convolutions
        conv_out = self.conv_layers(x)
        
        # Flatten and initialize first classifier layer if needed
        conv_flat = self.flatten(conv_out)
        
        if self.classifier_first_layer is None:
            input_features = conv_flat.shape[1]
            self.classifier_first_layer = nn.Linear(input_features + self.hidden_dim, self.hidden_dim).to(conv_flat.device)
        
        # Concatenate with label and get hidden representation
        combined = torch.cat([conv_flat, label_emb], dim=1)
        hidden = self.classifier_first_layer(combined)
        
        return hidden

class BVPTimeSeriesDiscriminator(nn.Module):
    """
    Specialized discriminator for Blood Volume Pulse (BVP) signals.
    Includes components for analyzing oscillatory patterns and cardiac rhythms.
    """
    def __init__(self, seq_length, hidden_dim, num_classes):
        super().__init__()
        
        # Label embedding (add 1 for padding idx 0)
        self.label_embedding = nn.Embedding(num_classes + 1, hidden_dim)
        
        # BVP signal processing with narrower kernels for fast oscillations
        self.temporal_conv_layers = nn.Sequential(
            # Transpose from [batch, seq_len, 1] to [batch, 1, seq_len]
            TransposeLayer(1, 2),
            
            # First layer with narrower kernel for cardiac oscillations
            nn.Conv1d(1, hidden_dim // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.LeakyReLU(0.2),
            
            # Second layer
            nn.Conv1d(hidden_dim // 4, hidden_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            
            # Third layer
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2)
        )
        
        # Spectral analysis path using 1D convolutions to simulate frequency analysis
        # This helps detect cardiac rhythms and periodicity in frequency domain
        self.spectral_conv_layers = nn.Sequential(
            # Transpose from [batch, seq_len, 1] to [batch, 1, seq_len]
            TransposeLayer(1, 2),
            
            # First layer with wide kernels to capture spectral properties
            nn.Conv1d(1, hidden_dim // 4, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.LeakyReLU(0.2),
            
            # Second layer
            nn.Conv1d(hidden_dim // 4, hidden_dim // 2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            
            # Max pooling to emphasize dominant frequencies
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Coefficients for rhythm regularity analysis
        self.rhythm_coeffs = nn.Parameter(torch.randn(hidden_dim // 2, hidden_dim // 4) * 0.1)
        
        # Store dimensions
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        
        # Flatten and feature integration
        self.flatten = nn.Flatten()
        
        # Feature integration layer (will be dynamically sized)
        self.feature_integration_layer = None
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x, labels):
        # Get label embeddings
        label_emb = self.label_embedding(labels)
        
        # Process signal through temporal convolutions
        # Input x shape: [batch_size, seq_length, 1]
        temporal_out = self.temporal_conv_layers(x)
        
        # Process through spectral convolutions
        spectral_out = self.spectral_conv_layers(x)
        
        # Flatten and prepare features
        temporal_flat = self.flatten(temporal_out)
        spectral_flat = self.flatten(spectral_out)
        
        # Initialize integration layer if needed
        if self.feature_integration_layer is None:
            temporal_features = temporal_flat.shape[1]
            spectral_features = spectral_flat.shape[1]
            total_features = temporal_features + spectral_features + self.hidden_dim
            self.feature_integration_layer = nn.Linear(total_features, self.hidden_dim).to(temporal_flat.device)
        
        # Concatenate all features with label
        combined = torch.cat([temporal_flat, spectral_flat, label_emb], dim=1)
        
        # Integrate features
        features = self.feature_integration_layer(combined)
        
        # Final classification
        return self.classifier(features)
    
    def extract_features(self, x, labels):
        """
        Extract intermediate features for feature matching loss.
        Returns activations from the penultimate layer.
        """
        # Get label embeddings
        label_emb = self.label_embedding(labels)
        
        # Process signal through temporal convolutions
        temporal_out = self.temporal_conv_layers(x)
        
        # Process through spectral convolutions
        spectral_out = self.spectral_conv_layers(x)
        
        # Flatten and prepare features
        temporal_flat = self.flatten(temporal_out)
        spectral_flat = self.flatten(spectral_out)
        
        # Initialize integration layer if needed
        if self.feature_integration_layer is None:
            temporal_features = temporal_flat.shape[1]
            spectral_features = spectral_flat.shape[1]
            total_features = temporal_features + spectral_features + self.hidden_dim
            self.feature_integration_layer = nn.Linear(total_features, self.hidden_dim).to(temporal_flat.device)
        
        # Concatenate all features with label
        combined = torch.cat([temporal_flat, spectral_flat, label_emb], dim=1)
        
        # Return integrated features
        return self.feature_integration_layer(combined)

# Add a custom transpose layer for clarity
class TransposeLayer(nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1
        
    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)

class MultiScaleSignalGenerator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_layers, num_classes, seq_length):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        
        # Label embedding (add 1 to num_classes for padding idx 0)
        self.label_embedding = nn.Embedding(num_classes + 1, hidden_dim)
        
        # Initial noise processing
        self.noise_processor = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2)
        )
        
        # Coarse branch (slow changes) - Larger hidden size, fewer layers
        self.lstm_coarse = nn.LSTM(
            input_size=hidden_dim * 2,  # Doubled for concatenated label
            hidden_size=hidden_dim,
            num_layers=max(1, num_layers-1),
            batch_first=True,
            dropout=0.1 if num_layers > 2 else 0
        )
        
        # Fine branch (fast oscillations) - Smaller hidden size, more layers
        self.lstm_fine = nn.LSTM(
            input_size=hidden_dim * 2,  # Doubled for concatenated label
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Output processing with convolutional layers
        self.output_conv = nn.Sequential(
            # Transpose from [batch, seq_len, hidden_dim+hidden_dim//2] to [batch, hidden_dim+hidden_dim//2, seq_len]
            TransposeLayer(1, 2),
            nn.Conv1d(hidden_dim + hidden_dim // 2, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
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

class TimeSeriesGAN(nn.Module):
    """
    Base GAN architecture for generating EDA physiological time series.
    """
    def __init__(self, 
                 latent_dim=100,
                 seq_length=30,    # 30 timesteps as per dataset
                 hidden_dim=128,
                 num_layers=2,     # Number of LSTM layers
                 num_classes=3):   # 3 stress conditions
        super().__init__()
        
        self.latent_dim = latent_dim
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Initialize generator and discriminator
        self.generator = SignalGenerator(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            seq_length=seq_length
        )
        
        self.discriminator = TimeSeriesDiscriminator(
            seq_length=seq_length,
            hidden_dim=hidden_dim,
            num_classes=num_classes
        )
    
    def generate(self, z, labels):
        """Generate time series from latent vector z and condition labels."""
        return self.generator(z, labels)
    
    def discriminate(self, x, labels):
        """Discriminate real vs. fake time series."""
        return self.discriminator(x, labels)
        
    def feature_matching_loss(self, real_data, fake_data, labels):
        """
        Calculate feature matching loss using discriminator's intermediate layers.
        This encourages generator to match the statistics of features in real data.
        
        Args:
            real_data: Real physiological signals
            fake_data: Generated signals
            labels: Condition labels for both datasets
            
        Returns:
            Feature matching loss (MSE between intermediate representations)
        """
        real_features = self.discriminator.extract_features(real_data, labels)
        fake_features = self.discriminator.extract_features(fake_data, labels)
        
        return F.mse_loss(fake_features, real_features)

class EDATimeSeriesGAN(TimeSeriesGAN):
    """
    Specialized GAN for generating EDA (Electrodermal Activity) signals.
    Optimized for the smooth, slowly varying characteristics of EDA signals.
    """
    def __init__(self,
                 latent_dim=100,
                 seq_length=30,
                 hidden_dim=128,
                 num_layers=2,
                 num_classes=3):
        super().__init__(
            latent_dim=latent_dim,
            seq_length=seq_length,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes
        )
        
        # Override or add EDA-specific components here if needed
        # By default, uses the SignalGenerator designed for EDA

class EDAMultiScaleTimeSeriesGAN(TimeSeriesGAN):
    """
    Specialized GAN for EDA signals that uses a multi-scale generator to better capture
    both tonic (slow baseline) and phasic (faster responses) components typical in EDA signals.
    """
    def __init__(self,
                 latent_dim=100,
                 seq_length=30,
                 hidden_dim=128,
                 num_layers=2,  # EDA typically requires fewer layers due to simpler patterns
                 num_classes=3):
        # Initialize the base class but don't use its generator
        super().__init__(
            latent_dim=latent_dim,
            seq_length=seq_length,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes
        )
        
        # Create a custom multi-scale generator with EDA-specific parameters
        self.generator = MultiScaleSignalGenerator(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            seq_length=seq_length
        )

class BVPGenerator(nn.Module):
    """
    Enhanced generator specialized for Blood Volume Pulse (BVP) signals.
    BVP signals are oscillatory with distinct cardiac cycles.
    Includes explicit oscillatory components to ensure realistic cardiac patterns.
    """
    def __init__(self, latent_dim, hidden_dim, num_layers, num_classes, seq_length):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_length = seq_length

        # Label embedding (add 1 to num_classes for padding idx 0)
        self.label_embedding = nn.Embedding(num_classes + 1, hidden_dim)

        # Initial noise processing
        self.noise_processor = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2)
        )

        # LSTM for cardiac rhythm generation
        self.cardiac_lstm = nn.LSTM(
            input_size=hidden_dim * 2,  # Doubled for concatenated label
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )

        # Oscillation parameters network (frequency, amplitude, phase)
        self.oscillation_params = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 3),  # Frequency, amplitude, phase
        )

        # Final output network with tanh activation for oscillation
        self.signal_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()  # Use tanh to enable both positive and negative values
        )
        
        # Condition-specific adaptation layers
        # Use Conv1d instead of Linear for all layers to properly handle the [1, 1, seq_len] shape
        self.stress_layer = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=1, bias=False, groups=1),
        )
        self.stress_layer[0].weight.data.fill_(1.8)  # Increased from 1.2 to 1.8 for higher amplitude in stress
        
        self.amusement_layer = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=True, groups=1),  # Added bias
        )
        # Initialize smoothing filter weights for amusement
        self.amusement_layer[0].weight.data.fill_(0)
        self.amusement_layer[0].weight.data[:, :, 0] = 0.15
        self.amusement_layer[0].weight.data[:, :, 1] = 0.7
        self.amusement_layer[0].weight.data[:, :, 2] = 0.15
        # Initialize bias to shift the baseline down slightly for amusement (as seen in plots)
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
        
        # Generate explicit oscillatory pattern for each sample - create cardiac-like oscillations
        # These will be combined with the learned features
        cardiac_oscillations = torch.zeros((batch_size, self.seq_length, 1), device=z.device)
        
        for i in range(batch_size):
            # Extract oscillation parameters from the hidden state
            h_avg = cardiac_hidden[i].mean(dim=0)
            osc_params = self.oscillation_params(h_avg)
            
            # Use parameters to generate appropriate oscillation frequency
            # Adjust for approximately 7-8 cycles in 30 time steps (0.23-0.27 Hz)
            freq = 0.23 + 0.04 * torch.tanh(osc_params[0])  # Range 0.19-0.27 Hz
            amp = 2.0 + 1.5 * torch.tanh(osc_params[1])  # Range 0.5-3.5 for z-score normalized data
            phase = math.pi * torch.sigmoid(osc_params[2])  # 0 to π
            
            # Generate time steps (normalized 0-1 range)
            t = torch.linspace(0, 1, self.seq_length, device=z.device)
            
            # Create oscillatory signal with adjusted frequency
            # Multiply by seq_length to scale frequency to match observed cycles per window
            osc = amp * torch.sin(2 * math.pi * freq * self.seq_length * t + phase)
            
            # Add to the batch
            cardiac_oscillations[i, :, 0] = osc
        
        # Generate base cardiac signal
        signal_series_list = []
        
        # Process each time step
        for t in range(self.seq_length):
            # Get current hidden state
            h_t = cardiac_hidden[:, t, :]
            
            # Generate output for this timestep
            out_t = self.signal_out(h_t)
            
            # Collect outputs
            signal_series_list.append(out_t)
        
        # Stack time steps
        base_signal = torch.stack(signal_series_list, dim=1)  # [batch, seq_length, 1]
        
        # Combine base signal with cardiac oscillations
        # Base signal (60%) provides the overall shape
        # Cardiac oscillations (40%) add the necessary oscillatory patterns - increased from 30% for more pronounced oscillations
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
                # Use dedicated layer for amplitude scaling
                transposed = sample_signal.transpose(1, 2)  # [1, 1, seq_len]
                scaled = self.stress_layer(transposed)
                output_signal[i:i+1] = scaled.transpose(1, 2)
            elif condition == 3:  # Amusement - smoother
                # Use dedicated convolutional layer for smoothing
                transposed = sample_signal.transpose(1, 2)  # [1, 1, seq_len]
                smoothed = self.amusement_layer(transposed)
                output_signal[i:i+1] = smoothed.transpose(1, 2)
        
        return output_signal

class BVPMultiScaleGenerator(nn.Module):
    """
    Enhanced multi-scale generator specialized for Blood Volume Pulse (BVP) signals.
    Handles both the fast oscillations of cardiac cycles and slower baseline variations.
    Includes explicit oscillatory components to ensure realistic cardiac patterns.
    """
    def __init__(self, latent_dim, hidden_dim, num_layers, num_classes, seq_length):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        
        # Label embedding (add 1 to num_classes for padding idx 0)
        self.label_embedding = nn.Embedding(num_classes + 1, hidden_dim)
        
        # Initial noise processing
        self.noise_processor = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2)
        )
        
        # Coarse branch (slow baseline changes)
        self.lstm_coarse = nn.LSTM(
            input_size=hidden_dim * 2,  # Doubled for concatenated label
            hidden_size=hidden_dim,
            num_layers=max(1, num_layers-1),
            batch_first=True,
            dropout=0.1 if num_layers > 2 else 0
        )
        
        # Enhanced fine branch (cardiac oscillations) - Increased capacity
        self.lstm_fine = nn.LSTM(
            input_size=hidden_dim * 2,  # Doubled for concatenated label
            hidden_size=hidden_dim,  # Increased from hidden_dim//2 to hidden_dim
            num_layers=num_layers + 2,  # Add more layers for oscillatory patterns
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0  # Increased dropout
        )
        
        # Explicit oscillation generator - creates high-frequency components
        self.oscillation_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, self.seq_length),
            nn.Tanh()  # Oscillatory component should have zero mean
        )
        
        # Cardiac parameter network for explicit control
        self.cardiac_params = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim // 2),  # Updated to match new dimensions
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 4),  # [frequency, amplitude, phase, baseline]
            nn.Sigmoid()  # Bound parameters to reasonable ranges
        )
        
        # Forced periodic component generator
        class SinusoidalEncoding(nn.Module):
            """Adds sinusoidal encoding to force periodic patterns"""
            def __init__(self, seq_length, hidden_dim):
                super().__init__()
                # Use exactly 6 frequencies to match the checkpoint's parameter count
                # while adjusting to observed BVP patterns (7-8 cycles in 30 time steps)
                self.frequencies = nn.Parameter(
                    torch.tensor([0.19, 0.21, 0.23, 0.25, 0.27, 0.29]), 
                    requires_grad=False
                )  # Typical BVP frequencies for your data
                self.seq_length = seq_length
                self.projection = nn.Linear(len(self.frequencies) * 2, hidden_dim // 2)
            
            def forward(self, batch_size, device):
                # Create time steps
                t = torch.linspace(0, 1, self.seq_length).to(device)
                t = t.view(1, -1, 1)  # [1, seq_length, 1]
                t = t.repeat(batch_size, 1, 1)  # [batch_size, seq_length, 1]
                
                # Generate sinusoidal components at different frequencies
                freqs = self.frequencies.view(1, 1, -1)  # [1, 1, num_frequencies]
                # Scale frequencies by seq_length to match the desired cycles per window
                phases = 2 * math.pi * t * freqs * self.seq_length  # [batch_size, seq_length, num_frequencies]
                
                # Create sin and cos features
                sin_features = torch.sin(phases)  # [batch_size, seq_length, num_frequencies]
                cos_features = torch.cos(phases)  # [batch_size, seq_length, num_frequencies]
                
                # Concatenate and project
                combined = torch.cat([sin_features, cos_features], dim=-1)  # [batch_size, seq_length, 2*num_frequencies]
                return self.projection(combined)  # [batch_size, seq_length, hidden_dim//2]
        
        self.sinusoidal_encoding = SinusoidalEncoding(seq_length, hidden_dim)
        
        # Enhanced output processing with residual connections
        self.output_conv = nn.Sequential(
            # Transpose from [batch, seq_len, hidden_dim*2] to [batch, hidden_dim*2, seq_len]
            TransposeLayer(1, 2),
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=5, padding=2),  # Updated dim
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim // 2, 1, kernel_size=3, padding=1),
            nn.Tanh(),  # Use tanh for oscillatory signals with positive and negative values
            # Transpose back to [batch, seq_len, 1]
            TransposeLayer(1, 2)
        )
        
        # Condition-specific adaptation layers
        # Use Conv1d instead of Linear for all layers to properly handle the [1, 1, seq_len] shape
        self.stress_layer = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=1, bias=False, groups=1),
        )
        self.stress_layer[0].weight.data.fill_(1.8)  # Increased from 1.2 to 1.8 for higher amplitude in stress

        self.amusement_layer = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=True, groups=1),  # Added bias
        )
        # Initialize smoothing filter weights for amusement (smoother transitions)
        self.amusement_layer[0].weight.data.fill_(0)
        self.amusement_layer[0].weight.data[:, :, 0] = 0.15
        self.amusement_layer[0].weight.data[:, :, 1] = 0.7
        self.amusement_layer[0].weight.data[:, :, 2] = 0.15
        # Initialize bias to shift the baseline down slightly for amusement (as seen in plots)
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
        
        # Ensure sinusoidal_features has the correct shape [batch, seq_length, hidden_dim//2]
        # We need to reshape it to [batch, seq_length, 1] to match other components
        sinusoidal_features_reshaped = sinusoidal_features.mean(dim=2, keepdim=True)
        
        # Generate cardiac parameters for each sample
        cardiac_params = []
        for i in range(batch_size):
            # Combine coarse and fine features
            features = torch.cat([coarse_out[i].mean(dim=0), fine_out[i].mean(dim=0)])
            params = self.cardiac_params(features)
            cardiac_params.append(params)
        cardiac_params = torch.stack(cardiac_params)
        
        # Extract parameters
        frequencies = cardiac_params[:, 0] * 0.08 + 0.21  # Range 0.21-0.29 Hz
        amplitudes = cardiac_params[:, 1] * 3.0 + 1.0    # Range 1.0-4.0 for z-score normalized data
        phases = cardiac_params[:, 2] * 2 * math.pi      # Range 0-2π
        baselines = cardiac_params[:, 3] * 3.0 - 1.5     # Range -1.5-1.5 for z-score normalized data
        
        # Condition-specific amplitude adjustments based on observed data
        for i in range(batch_size):
            condition = labels[i].item()
            if condition == 1:  # Baseline
                # No adjustment needed
                pass
            elif condition == 2:  # Stress - observed higher amplitude in stress condition
                amplitudes[i] = amplitudes[i] * 1.5  # Increase amplitude for stress
            elif condition == 3:  # Amusement - observed lower amplitude
                amplitudes[i] = amplitudes[i] * 0.7  # Decrease amplitude for amusement
        
        # Generate cardiac oscillations based on parameters
        cardiac_oscillations = torch.zeros((batch_size, self.seq_length, 1), device=z.device)
        
        for i in range(batch_size):
            # Time vector (normalized to 0-1)
            t = torch.linspace(0, 1, self.seq_length, device=z.device)
            # Generate cardiac oscillation with frequency, amplitude, phase
            # Multiply frequency by seq_length to match cycles per window
            oscillation = amplitudes[i] * torch.sin(2 * math.pi * frequencies[i] * self.seq_length * t + phases[i])
            # Add baseline shift
            oscillation = oscillation + baselines[i]
            # Add to batch
            cardiac_oscillations[i, :, 0] = oscillation
        
        # Combine features for the final output
        # 1. Coarse features provide the overall trend
        # 2. Fine features add detailed variations
        # 3. Explicit oscillations add higher-frequency components
        # 4. Sinusoidal features force cardiac-like periodicity
        # 5. Generated cardiac oscillations add physiologically plausible patterns
        
        # Combine coarse and fine features for base signal
        combined = torch.cat([coarse_out, fine_out], dim=2)  # [batch, seq_length, hidden_dim*2]
        
        # Generate base signal
        base_signal = self.output_conv(combined)  # [batch, seq_length, 1]
        
        # Add the oscillatory components with appropriate weighting
        # Base signal provides overall structure (60%)
        # Oscillations add learned high-frequency components (10%)
        # Cardiac oscillations add physiological patterns (20%)
        # Sinusoidal features ensure periodicity (10%)
        enhanced_signal = (
            0.1 * base_signal + 
            0.1 * oscillations +
            0.7 * cardiac_oscillations +
            0.1 * sinusoidal_features_reshaped  # Use the reshaped version with correct dimensions
        )
        
        # Create output tensor where we'll store the condition-specific outputs
        output_signal = torch.zeros_like(enhanced_signal)
        
        # Apply condition-specific transformations using dedicated layers
        for i in range(batch_size):
            sample_signal = enhanced_signal[i:i+1]  # Keep batch dimension
            condition = labels[i].item()
            
            if condition == 1:  # Baseline - use as is
                output_signal[i:i+1] = sample_signal
            elif condition == 2:  # Stress - higher amplitude and frequency
                # Use dedicated layer for amplitude scaling
                transposed = sample_signal.transpose(1, 2)  # [1, 1, seq_len]
                scaled = self.stress_layer(transposed)
                output_signal[i:i+1] = scaled.transpose(1, 2)
            elif condition == 3:  # Amusement - smoother
                # Use dedicated convolutional layer for smoothing
                transposed = sample_signal.transpose(1, 2)  # [1, 1, seq_len]
                smoothed = self.amusement_layer(transposed)
                output_signal[i:i+1] = smoothed.transpose(1, 2)
        
        return output_signal

class BVPTimeSeriesGAN(TimeSeriesGAN):
    """
    Specialized GAN for generating BVP (Blood Volume Pulse) signals.
    Optimized for the oscillatory characteristics of BVP signals.
    """
    def __init__(self,
                 latent_dim=100,
                 seq_length=30,
                 hidden_dim=128,
                 num_layers=2,
                 num_classes=3):
        # Initialize with parent class but we'll override components
        super().__init__(
            latent_dim=latent_dim,
            seq_length=seq_length,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes
        )
        
        # Override with BVP-specific generator
        self.generator = BVPGenerator(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            seq_length=seq_length
        )
        
        # Override with BVP-specific discriminator
        self.discriminator = BVPTimeSeriesDiscriminator(
            seq_length=seq_length,
            hidden_dim=hidden_dim,
            num_classes=num_classes
        )

class BVPMultiScaleTimeSeriesGAN(TimeSeriesGAN):
    """
    Specialized GAN for BVP signals that uses a multi-scale generator to better capture
    both cardiac cycles and slower baseline variations.
    """
    def __init__(self,
                 latent_dim=100,
                 seq_length=30,
                 hidden_dim=128,
                 num_layers=2,
                 num_classes=3):
        # Initialize the base class but don't use its generator or discriminator
        super().__init__(
            latent_dim=latent_dim,
            seq_length=seq_length,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes
        )
        
        # Create a custom multi-scale generator with BVP-specific parameters
        self.generator = BVPMultiScaleGenerator(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            seq_length=seq_length
        )
        
        # Use the BVP-specific discriminator
        self.discriminator = BVPTimeSeriesDiscriminator(
            seq_length=seq_length,
            hidden_dim=hidden_dim,
            num_classes=num_classes
        )

class FeatureGenerator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_classes):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Label embedding for condition (1: Baseline, 2: Stress, 3: Amusement)
        self.label_embedding = nn.Embedding(num_classes + 1, hidden_dim)  # +1 for padding idx 0

        # Main feature processing network
        self.feature_processor = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2)
        )

        # Output heads for different feature types
        # HRV features: RMSSD, SDNN, LF, HF, LF_HF_ratio
        self.hrv_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 5),  # 5 HRV features
            nn.SiLU()  # Smooth positive activation for physiological metrics
        )

        # EDA features: mean_EDA, median_EDA, SCR_count
        self.eda_features_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
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

class FeatureDiscriminator(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Label embedding for condition (1: Baseline, 2: Stress, 3: Amusement)
        self.label_embedding = nn.Embedding(num_classes + 1, hidden_dim)  # +1 for padding idx 0

        # Feature type-specific processing
        self.hrv_processor = nn.Sequential(
            nn.Linear(5, hidden_dim // 2),  # 5 HRV features
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2)
        )

        self.eda_processor = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),  # 3 EDA features
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2)
        )

        # Combined feature processing
        self.feature_processor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Combined features + label
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2)
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
        """
        Extract intermediate features for feature matching loss.
        Returns activations from the penultimate layer.
        """
        # Process each feature type
        hrv_encoded = self.hrv_processor(features['HRV_features'])
        eda_encoded = self.eda_processor(features['EDA_features'])

        # Get label embeddings
        label_emb = self.label_embedding(labels)

        # Combine all features
        combined = torch.cat([hrv_encoded, eda_encoded, label_emb], dim=1)

        # Process combined features - this is the intermediate representation we want
        return self.feature_processor(combined)

class FeatureGAN(nn.Module):
    """
    GAN architecture for generating correlated physiological features.
    Generates both HRV features (5-dim) and EDA features (3-dim) conditioned on stress state.
    """
    def __init__(self, 
                 latent_dim=100,
                 hidden_dim=128,
                 num_classes=3):    # 3 stress conditions (1: Baseline, 2: Stress, 3: Amusement)
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Initialize generator and discriminator
        self.generator = FeatureGenerator(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes
        )
        
        self.discriminator = FeatureDiscriminator(
            hidden_dim=hidden_dim,
            num_classes=num_classes
        )
    
    def generate(self, z, labels):
        """Generate HRV and EDA features from latent vector z and condition labels."""
        return self.generator(z, labels)
    
    def discriminate(self, features, labels):
        """Discriminate real vs. fake features. Returns raw logits for training stability."""
        return self.discriminator(features, labels)
    
    def discriminate_prob(self, features, labels):
        """Discriminate real vs. fake features. Returns probabilities."""
        return torch.sigmoid(self.discriminator(features, labels))
        
    def feature_matching_loss(self, real_features, fake_features, labels):
        """
        Calculate feature matching loss using discriminator's intermediate layers.
        This encourages generator to match the statistics of features in real data.
        
        Args:
            real_features: Real physiological features dictionary
            fake_features: Generated features dictionary
            labels: Condition labels for both datasets
            
        Returns:
            Feature matching loss (MSE between intermediate representations)
        """
        real_intermediate = self.discriminator.extract_features(real_features, labels)
        fake_intermediate = self.discriminator.extract_features(fake_features, labels)
        
        return F.mse_loss(fake_intermediate, real_intermediate)
