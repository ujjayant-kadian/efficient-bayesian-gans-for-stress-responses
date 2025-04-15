import torch
import torch.nn as nn
import torch.nn.functional as F

from .variational_layers import BayesLinear, BayesConv1d, BayesLSTM, PriorType

# Add a custom transpose layer for clarity (same as in base_gan.py)
class TransposeLayer(nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1
        
    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)

class BayesSignalGenerator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_layers, num_classes, seq_length,
                 prior_type=PriorType.GAUSSIAN, prior_params={'sigma': 1.0}):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.prior_type = prior_type
        self.prior_params = prior_params

        # Label embedding (add 1 to num_classes for padding idx 0)
        self.label_embedding = nn.Embedding(num_classes + 1, hidden_dim)

        # Initial noise processing with standard Linear layers
        self.noise_processor = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2)
        )

        # LSTM for temporal coherence - CONVERT TO BAYESIAN to model temporal uncertainty
        self.signal_lstm = BayesLSTM(
            input_size=hidden_dim * 2,  # Doubled for concatenated label
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0,
            prior_type=prior_type,
            prior_params=prior_params
        )

        # Output network for EDA signals - CONVERT TO STANDARD LAYERS
        self.signal_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1),  # Standard layer instead of Bayesian
            nn.LeakyReLU(0.2)
        )

    def forward(self, z, labels, sample=True):
        # Get label embeddings
        label_emb = self.label_embedding(labels)  # [batch_size, hidden_dim]

        # Process noise with label context
        combined_input = torch.cat([z, label_emb], dim=1)
        h = self.noise_processor(combined_input)

        # Prepare sequence input
        label_repeat = label_emb.unsqueeze(1).expand(-1, self.seq_length, -1)
        signal_in = h.unsqueeze(1).expand(-1, self.seq_length, -1)
        signal_in = torch.cat([signal_in, label_repeat], dim=2)

        # Generate temporal features - NOW USING BAYESIAN LSTM WITH SAMPLING
        signal_hidden, _ = self.signal_lstm(signal_in, None, sample)  # Fixed parameter order
        
        # Generate signal
        signal_hidden = signal_hidden.reshape(-1, signal_hidden.size(-1))
        
        # Process through signal_out with standard layers (no sampling)
        signal_series = self.signal_out(signal_hidden)
        
        signal_series = signal_series.view(-1, self.seq_length, 1)

        # Return in format [batch_size, seq_length, 1]
        return signal_series
    
    def kl_divergence(self):
        """Sum up the KL divergences from all Bayesian layers."""
        kl_total = 0.0
        from src.models.variational_layers import BayesLinear, BayesConv1d, BayesLSTM, BayesConv2d
        
        # Recursively check all modules to find Bayesian layers
        # This properly handles nested modules like Sequential
        for module in self.modules():
            # Check if module is a Bayesian layer itself (not a container)
            if isinstance(module, (BayesLinear, BayesConv1d, BayesLSTM, BayesConv2d)):
                kl_total += module.kl_divergence()
        
        return kl_total

class BayesTimeSeriesDiscriminator(nn.Module):
    def __init__(self, seq_length, hidden_dim, num_classes,
                 prior_type=PriorType.GAUSSIAN, prior_params={'sigma': 1.0}):
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
        
        # Store dimensions
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        
        # Flatten
        self.flatten = nn.Flatten()
        
        # First classifier layer will be initialized in first forward pass
        self.classifier_first_layer = None
        
        # Second classifier layer (fixed size)
        self.classifier_second_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )
        
        # Prior configuration
        self.prior_type = prior_type
        self.prior_params = prior_params
    
    def forward(self, x, labels, sample=True):
        # Get label embeddings
        label_emb = self.label_embedding(labels)
        
        # Process signal through convolutions
        # Input x shape: [batch_size, seq_length, 1]
        x = self.conv_layers(x)
        
        # Flatten
        conv_flat = self.flatten(x)
        
        # Initialize first classifier layer if needed
        if self.classifier_first_layer is None:
            input_features = conv_flat.shape[1]
            self.classifier_first_layer = nn.Linear(
                input_features + self.hidden_dim, 
                self.hidden_dim
            ).to(conv_flat.device)
        
        # Concatenate with label and classify
        combined = torch.cat([conv_flat, label_emb], dim=1)
        hidden = self.classifier_first_layer(combined)
        
        # Final classification
        output = self.classifier_second_layer(hidden)
        
        return output
    
    def extract_features(self, x, labels, sample=True):
        """
        Extract intermediate features for feature matching loss.
        Returns activations from the penultimate layer.
        """
        # Get label embeddings
        label_emb = self.label_embedding(labels)
        
        # Process signal through convolutions
        x = self.conv_layers(x)
        
        # Flatten
        conv_flat = self.flatten(x)
        
        # Initialize first classifier layer if needed
        if self.classifier_first_layer is None:
            input_features = conv_flat.shape[1]
            self.classifier_first_layer = nn.Linear(
                input_features + self.hidden_dim, 
                self.hidden_dim
            ).to(conv_flat.device)
        
        # Concatenate with label and get hidden representation
        combined = torch.cat([conv_flat, label_emb], dim=1)
        hidden = self.classifier_first_layer(combined)
        
        return hidden
    
    def kl_divergence(self):
        """Sum up the KL divergences from all Bayesian layers."""
        # No Bayesian layers left, return 0 as a tensor
        return torch.tensor(0.0, device=next(self.parameters()).device)

class BayesTimeSeriesGAN(nn.Module):
    """
    Bayesian GAN architecture for generating EDA physiological time series.
    """
    def __init__(self, 
                 latent_dim=100,
                 seq_length=30,
                 hidden_dim=128,
                 num_layers=2,
                 num_classes=3,
                 prior_type=PriorType.GAUSSIAN,
                 prior_params={'sigma': 1.0}):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.prior_type = prior_type
        self.prior_params = prior_params
        
        # Initialize generator and discriminator
        self.generator = BayesSignalGenerator(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            seq_length=seq_length,
            prior_type=prior_type,
            prior_params=prior_params
        )
        
        self.discriminator = BayesTimeSeriesDiscriminator(
            seq_length=seq_length,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            prior_type=prior_type,
            prior_params=prior_params
        )
    
    def generate(self, z, labels, sample=True):
        """Generate time series from latent vector z and condition labels."""
        return self.generator(z, labels, sample)
    
    def discriminate(self, x, labels, sample=True):
        """Discriminate real vs. fake time series."""
        return self.discriminator(x, labels, sample)
        
    def feature_matching_loss(self, real_data, fake_data, labels, sample=True):
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
        real_features = self.discriminator.extract_features(real_data, labels, sample)
        fake_features = self.discriminator.extract_features(fake_data, labels, sample)
        
        return F.mse_loss(fake_features, real_features)
    
    def kl_loss(self):
        """
        Sum up the KL divergences of remaining Bayesian layers (generator only).
        """
        # Only use KL divergence from the generator
        g_kl = self.generator.kl_divergence()
        return g_kl
        
    def generate_with_uncertainty(self, z, labels, n_samples=10):
        """
        Generate multiple time series samples to estimate uncertainty.
        
        Args:
            z: Latent vector [batch_size, latent_dim]
            labels: Condition labels [batch_size]
            n_samples: Number of MC samples for uncertainty estimation
            
        Returns:
            mean_series: Mean of generated time series [batch_size, seq_length, 1]
            std_series: Standard deviation (uncertainty) [batch_size, seq_length, 1]
        """
        # Store original evaluation/training mode
        training_mode = self.training
        
        # Use eval mode but we'll still be sampling weights
        self.eval()
        
        samples = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                # Generate sample with weight sampling
                generated = self.generate(z, labels, sample=True)
                samples.append(generated)
        
        # Stack samples along a new dimension
        samples = torch.stack(samples, dim=0)  # [n_samples, batch_size, seq_length, 1]
        
        # Calculate mean and standard deviation along the samples dimension
        mean_series = torch.mean(samples, dim=0)  # [batch_size, seq_length, 1]
        std_series = torch.std(samples, dim=0)    # [batch_size, seq_length, 1]
        
        # Restore original mode
        self.train(training_mode)
        
        return mean_series, std_series
    
    def generate_with_uncertainty_samples(self, z, labels, n_samples=10):
        """
        Generate multiple time series samples for uncertainty visualization.
        
        Args:
            z: Latent vector [batch_size, latent_dim]
            labels: Condition labels [batch_size]
            n_samples: Number of MC samples for uncertainty visualization
            
        Returns:
            List of n_samples time series tensors, each of shape [batch_size, seq_length, 1]
        """
        # Store original evaluation/training mode
        training_mode = self.training
        
        # Use eval mode but we'll still be sampling weights
        self.eval()
        
        samples = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                # Generate sample with weight sampling
                generated = self.generate(z, labels, sample=True)
                samples.append(generated)
        
        # Restore original mode
        self.train(training_mode)
        
        return samples

class BayesEDATimeSeriesGAN(BayesTimeSeriesGAN):
    """
    Specialized Bayesian GAN for generating EDA (Electrodermal Activity) signals.
    Optimized for the smooth, slowly varying characteristics of EDA signals.
    """
    def __init__(self,
                 latent_dim=100,
                 seq_length=30,
                 hidden_dim=128,
                 num_layers=2,
                 num_classes=3,
                 prior_type=PriorType.GAUSSIAN,
                 prior_params={'sigma': 1.0}):
        super().__init__(
            latent_dim=latent_dim,
            seq_length=seq_length,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            prior_type=prior_type,
            prior_params=prior_params
        )
        
        # Uses the base BayesSignalGenerator designed for EDA

class BayesMultiScaleSignalGenerator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_layers, num_classes, seq_length,
                prior_type=PriorType.GAUSSIAN, prior_params={'sigma': 1.0}):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        self.prior_type = prior_type
        self.prior_params = prior_params
        
        # Label embedding (add 1 to num_classes for padding idx 0)
        self.label_embedding = nn.Embedding(num_classes + 1, hidden_dim)
        
        # Initial noise processing with standard layer
        self.noise_processor = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2)
        )
        
        # Coarse branch (slow changes) - CONVERT TO BAYESIAN for temporal coherence
        self.lstm_coarse = BayesLSTM(
            input_size=hidden_dim * 2,  # Doubled for concatenated label
            hidden_size=hidden_dim,
            num_layers=max(1, num_layers-1),
            batch_first=True,
            dropout=0.1 if num_layers > 2 else 0,
            prior_type=prior_type,
            prior_params=prior_params
        )
        
        # Fine branch (fast oscillations) - keep as standard LSTM
        self.lstm_fine = nn.LSTM(
            input_size=hidden_dim * 2,  # Doubled for concatenated label
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Output processing with convolutional layers - convert to all standard layers
        self.output_conv = nn.Sequential(
            # Transpose from [batch, seq_len, hidden_dim+hidden_dim//2] to [batch, hidden_dim+hidden_dim//2, seq_len]
            TransposeLayer(1, 2),
            nn.Conv1d(hidden_dim + hidden_dim // 2, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim // 2, 1, kernel_size=3, padding=1), # Standard layer instead of Bayesian
            nn.LeakyReLU(0.2),
            # Transpose back to [batch, seq_len, 1]
            TransposeLayer(1, 2)
        )
    
    def forward(self, z, labels, sample=True):
        # Get label embeddings
        label_emb = self.label_embedding(labels)  # [batch_size, hidden_dim]
        
        # Process noise with label context
        combined_input = torch.cat([z, label_emb], dim=1)
        h = self.noise_processor(combined_input)
        
        # Prepare sequence input
        label_repeat = label_emb.unsqueeze(1).expand(-1, self.seq_length, -1)
        signal_in = h.unsqueeze(1).expand(-1, self.seq_length, -1)
        signal_in = torch.cat([signal_in, label_repeat], dim=2)  # [batch, seq_length, hidden_dim*2]
        
        # Generate temporal features through parallel paths - fix parameter order for BayesLSTM
        coarse_out, _ = self.lstm_coarse(signal_in, None, sample)  # Properly pass None for hx and sample parameter
        fine_out, _ = self.lstm_fine(signal_in)      # Standard LSTM
        
        # Combine coarse and fine features
        combined = torch.cat([coarse_out, fine_out], dim=2)  # [batch, seq_length, hidden_dim+hidden_dim//2]
        
        # Generate final signal with standard convolutional layers
        signal_series = self.output_conv(combined)
        
        return signal_series
    
    def kl_divergence(self):
        """Sum up the KL divergences from all Bayesian layers."""
        kl_total = 0.0
        from src.models.variational_layers import BayesLinear, BayesConv1d, BayesLSTM, BayesConv2d
        
        # Recursively check all modules to find Bayesian layers
        # This properly handles nested modules like Sequential
        for module in self.modules():
            # Check if module is a Bayesian layer itself (not a container)
            if isinstance(module, (BayesLinear, BayesConv1d, BayesLSTM, BayesConv2d)):
                kl_total += module.kl_divergence()
        
        return kl_total

class BayesEDAMultiScaleTimeSeriesGAN(BayesTimeSeriesGAN):
    """
    Specialized Bayesian GAN for EDA signals that uses a multi-scale generator to better capture
    both tonic (slow baseline) and phasic (faster responses) components typical in EDA signals.
    """
    def __init__(self,
                 latent_dim=100,
                 seq_length=30,
                 hidden_dim=128,
                 num_layers=2,  # EDA typically requires fewer layers due to simpler patterns
                 num_classes=3,
                 prior_type=PriorType.GAUSSIAN,
                 prior_params={'sigma': 1.0}):
        # Initialize the base class but don't use its generator
        super().__init__(
            latent_dim=latent_dim,
            seq_length=seq_length,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            prior_type=prior_type,
            prior_params=prior_params
        )
        
        # Create a custom multi-scale generator with EDA-specific parameters
        self.generator = BayesMultiScaleSignalGenerator(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            seq_length=seq_length,
            prior_type=prior_type,
            prior_params=prior_params
        )

class BayesBVPTimeSeriesDiscriminator(nn.Module):
    """
    Specialized discriminator for Blood Volume Pulse (BVP) signals.
    Includes components for analyzing oscillatory patterns and cardiac rhythms.
    """
    def __init__(self, seq_length, hidden_dim, num_classes,
                prior_type=PriorType.GAUSSIAN, prior_params={'sigma': 1.0}):
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
        
        # Prior configuration
        self.prior_type = prior_type
        self.prior_params = prior_params
    
    def forward(self, x, labels, sample=True):
        # Get label embeddings
        label_emb = self.label_embedding(labels)
        
        # Process signal through temporal convolutions
        # Input x shape: [batch_size, seq_length, 1]
        
        # Temporal path
        t = self.temporal_conv_layers(x)
        
        # Spectral path
        s = self.spectral_conv_layers(x)
        
        # Flatten and prepare features
        temporal_flat = self.flatten(t)
        spectral_flat = self.flatten(s)
        
        # Initialize integration layer if needed
        if self.feature_integration_layer is None:
            temporal_features = temporal_flat.shape[1]
            spectral_features = spectral_flat.shape[1]
            total_features = temporal_features + spectral_features + self.hidden_dim
            self.feature_integration_layer = nn.Linear(
                total_features, 
                self.hidden_dim
            ).to(temporal_flat.device)
        
        # Concatenate all features with label
        combined = torch.cat([temporal_flat, spectral_flat, label_emb], dim=1)
        
        # Integrate features
        features = self.feature_integration_layer(combined)
        
        # Final classification
        output = self.classifier(features)
        
        return output
    
    def extract_features(self, x, labels, sample=True):
        """
        Extract intermediate features for feature matching loss.
        Returns activations from the penultimate layer.
        """
        # Get label embeddings
        label_emb = self.label_embedding(labels)
        
        # Process signal through temporal convolutions
        
        # Temporal path
        t = self.temporal_conv_layers(x)
        
        # Spectral path
        s = self.spectral_conv_layers(x)
        
        # Flatten and prepare features
        temporal_flat = self.flatten(t)
        spectral_flat = self.flatten(s)
        
        # Initialize integration layer if needed
        if self.feature_integration_layer is None:
            temporal_features = temporal_flat.shape[1]
            spectral_features = spectral_flat.shape[1]
            total_features = temporal_features + spectral_features + self.hidden_dim
            self.feature_integration_layer = nn.Linear(
                total_features, 
                self.hidden_dim
            ).to(temporal_flat.device)
        
        # Concatenate all features with label
        combined = torch.cat([temporal_flat, spectral_flat, label_emb], dim=1)
        
        # Return integrated features
        return self.feature_integration_layer(combined)
    
    def kl_divergence(self):
        """Sum up the KL divergences from all Bayesian layers."""
        # No Bayesian layers left, return 0 as a tensor
        return torch.tensor(0.0, device=next(self.parameters()).device)

class BayesBVPGenerator(nn.Module):
    """
    Enhanced Bayesian generator specialized for Blood Volume Pulse (BVP) signals.
    BVP signals are oscillatory with distinct cardiac cycles.
    Includes explicit oscillatory components to ensure realistic cardiac patterns.
    """
    def __init__(self, latent_dim, hidden_dim, num_layers, num_classes, seq_length,
                prior_type=PriorType.GAUSSIAN, prior_params={'sigma': 1.0}):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.prior_type = prior_type
        self.prior_params = prior_params

        # Label embedding (add 1 to num_classes for padding idx 0)
        self.label_embedding = nn.Embedding(num_classes + 1, hidden_dim)

        # Initial noise processing with standard Linear
        self.noise_processor = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2)
        )

        # LSTM for cardiac rhythm generation - CONVERT TO BAYESIAN for temporal uncertainty
        self.cardiac_lstm = BayesLSTM(
            input_size=hidden_dim * 2,  # Doubled for concatenated label
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0,
            prior_type=prior_type,
            prior_params=prior_params
        )

        # Oscillation parameters network (frequency, amplitude, phase)
        # Convert to all standard layers
        self.oscillation_params = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 3),  # Standard layer instead of Bayesian
        )

        # Final output network with tanh activation for oscillation
        # Convert to all standard layers
        self.signal_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1),  # Standard layer instead of Bayesian
            nn.Tanh()  # Use tanh to enable both positive and negative values
        )
        
        # Condition-specific adaptation layers - convert to standard layers
        # Use Conv1d instead of Linear for all layers to properly handle the [1, 1, seq_len] shape
        self.stress_layer = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=1),
        )
        self.stress_layer[0].weight.data.fill_(1.8)  # Initialize to same value as base model
        
        self.amusement_layer = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=3, padding=1),
        )
        # Initialize smoothing filter weights for amusement
        self.amusement_layer[0].weight.data.fill_(0)
        self.amusement_layer[0].weight.data[:, :, 0] = 0.15
        self.amusement_layer[0].weight.data[:, :, 1] = 0.7
        self.amusement_layer[0].weight.data[:, :, 2] = 0.15
        # Initialize bias to shift the baseline down slightly for amusement
        if hasattr(self.amusement_layer[0], 'bias'):
            self.amusement_layer[0].bias.data.fill_(-0.3)

    def forward(self, z, labels, sample=True):
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

        # Generate temporal features - NOW USING BAYESIAN LSTM WITH SAMPLING
        cardiac_hidden, _ = self.cardiac_lstm(signal_in, None, sample)  # Fixed parameter order
        
        # Generate explicit oscillatory pattern for each sample - create cardiac-like oscillations
        # These will be combined with the learned features
        cardiac_oscillations = torch.zeros((batch_size, self.seq_length, 1), device=z.device)
        
        for i in range(batch_size):
            # Extract oscillation parameters from the hidden state
            h_avg = cardiac_hidden[i].mean(dim=0)
            
            # Process through oscillation_params with standard layers
            osc_params = self.oscillation_params(h_avg)
            
            # Use parameters to generate appropriate oscillation frequency
            # Adjust for approximately 7-8 cycles in 30 time steps (0.23-0.27 Hz)
            freq = 0.23 + 0.04 * torch.tanh(osc_params[0])  # Range 0.19-0.27 Hz
            amp = 2.0 + 1.5 * torch.tanh(osc_params[1])  # Range 0.5-3.5 for z-score normalized data
            phase = torch.pi * torch.sigmoid(osc_params[2])  # 0 to π
            
            # Generate time steps (normalized 0-1 range)
            t = torch.linspace(0, 1, self.seq_length, device=z.device)
            
            # Create oscillatory signal with adjusted frequency
            # Multiply by seq_length to scale frequency to match observed cycles per window
            osc = amp * torch.sin(2 * torch.pi * freq * self.seq_length * t + phase)
            
            # Add to the batch
            cardiac_oscillations[i, :, 0] = osc
        
        # Generate base cardiac signal
        signal_series_list = []
        
        # Process each time step
        for t in range(self.seq_length):
            # Get current hidden state
            h_t = cardiac_hidden[:, t, :]
            
            # Generate output for this timestep with standard layers
            out_t = self.signal_out(h_t)
            
            # Collect outputs
            signal_series_list.append(out_t)
        
        # Stack time steps
        base_signal = torch.stack(signal_series_list, dim=1)  # [batch, seq_length, 1]
        
        # Combine base signal with cardiac oscillations
        # Base signal (60%) provides the overall shape
        # Cardiac oscillations (40%) add the necessary oscillatory patterns
        enhanced_signal = 0.6 * base_signal + 0.4 * cardiac_oscillations
        
        # Create output tensor for condition-specific outputs
        output_signal = torch.zeros_like(enhanced_signal)
        
        # Apply condition-specific transformations with standard layers
        for i in range(batch_size):
            sample_signal = enhanced_signal[i:i+1]  # Keep batch dimension
            condition = labels[i].item()
            
            if condition == 1:  # Baseline - use as is
                output_signal[i:i+1] = sample_signal
            elif condition == 2:  # Stress - higher amplitude
                # Use standard layer for amplitude scaling
                transposed = sample_signal.transpose(1, 2)  # [1, 1, seq_len]
                scaled = self.stress_layer(transposed)
                output_signal[i:i+1] = scaled.transpose(1, 2)
            elif condition == 3:  # Amusement - smoother
                # Use standard convolutional layer for smoothing
                transposed = sample_signal.transpose(1, 2)  # [1, 1, seq_len]
                smoothed = self.amusement_layer(transposed)
                output_signal[i:i+1] = smoothed.transpose(1, 2)
        
        return output_signal
    
    def kl_divergence(self):
        """Sum up the KL divergences from all Bayesian layers."""
        kl_total = 0.0
        from src.models.variational_layers import BayesLinear, BayesConv1d, BayesLSTM, BayesConv2d
        
        # Recursively check all modules to find Bayesian layers
        # This properly handles nested modules like Sequential
        for module in self.modules():
            # Check if module is a Bayesian layer itself (not a container)
            if isinstance(module, (BayesLinear, BayesConv1d, BayesLSTM, BayesConv2d)):
                kl_total += module.kl_divergence()
        
        return kl_total

class BayesBVPTimeSeriesGAN(BayesTimeSeriesGAN):
    """
    Specialized Bayesian GAN for generating BVP (Blood Volume Pulse) signals.
    Optimized for the oscillatory characteristics of BVP signals.
    """
    def __init__(self,
                 latent_dim=100,
                 seq_length=30,
                 hidden_dim=128,
                 num_layers=2,
                 num_classes=3,
                 prior_type=PriorType.GAUSSIAN,
                 prior_params={'sigma': 1.0}):
        # Initialize with parent class but we'll override components
        super().__init__(
            latent_dim=latent_dim,
            seq_length=seq_length,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            prior_type=prior_type,
            prior_params=prior_params
        )
        
        # Override with BVP-specific generator
        self.generator = BayesBVPGenerator(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            seq_length=seq_length,
            prior_type=prior_type,
            prior_params=prior_params
        )
        
        # Override with BVP-specific discriminator
        self.discriminator = BayesBVPTimeSeriesDiscriminator(
            seq_length=seq_length,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            prior_type=prior_type,
            prior_params=prior_params
        )

class BayesBVPMultiScaleGenerator(nn.Module):
    """
    Enhanced multi-scale generator specialized for Blood Volume Pulse (BVP) signals.
    Handles both the fast oscillations of cardiac cycles and slower baseline variations.
    Includes explicit oscillatory components to ensure realistic cardiac patterns.
    """
    def __init__(self, latent_dim, hidden_dim, num_layers, num_classes, seq_length,
                prior_type=PriorType.GAUSSIAN, prior_params={'sigma': 1.0}):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        self.prior_type = prior_type
        self.prior_params = prior_params
        
        # Label embedding (add 1 to num_classes for padding idx 0)
        self.label_embedding = nn.Embedding(num_classes + 1, hidden_dim)
        
        # Initial noise processing with standard layer
        self.noise_processor = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2)
        )
        
        # Coarse branch (slow baseline changes) - CONVERT TO BAYESIAN for modeling slower variability
        self.lstm_coarse = BayesLSTM(
            input_size=hidden_dim * 2,  # Doubled for concatenated label
            hidden_size=hidden_dim,
            num_layers=max(1, num_layers-1),
            batch_first=True,
            dropout=0.1 if num_layers > 2 else 0,
            prior_type=prior_type,
            prior_params=prior_params
        )
        
        # Enhanced fine branch (cardiac oscillations) - keep as standard LSTM
        self.lstm_fine = nn.LSTM(
            input_size=hidden_dim * 2,  # Doubled for concatenated label
            hidden_size=hidden_dim,  # Increased from hidden_dim//2 to hidden_dim
            num_layers=num_layers + 2,  # Add more layers for oscillatory patterns
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0  # Increased dropout
        )
        
        # Explicit oscillation generator - convert to all standard layers
        self.oscillation_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, self.seq_length),  # Standard layer instead of Bayesian
            nn.Tanh()  # Oscillatory component should have zero mean
        )
        
        # Cardiac parameter network - convert to all standard layers
        self.cardiac_params = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 4),  # Standard layer instead of Bayesian
            nn.Sigmoid()  # Bound parameters to reasonable ranges
        )
        
        # Forced periodic component generator - keep projection as Bayesian
        class BayesSinusoidalEncoding(nn.Module):
            """Adds sinusoidal encoding to force periodic patterns"""
            def __init__(self, seq_length, hidden_dim, prior_type, prior_params):
                super().__init__()
                # Use exactly 6 frequencies to match the checkpoint's parameter count
                self.frequencies = nn.Parameter(
                    torch.tensor([0.19, 0.21, 0.23, 0.25, 0.27, 0.29]), 
                    requires_grad=False
                )  # Typical BVP frequencies for your data
                self.seq_length = seq_length
                # Keep projection as Bayesian to capture frequency uncertainty
                self.projection = BayesLinear(len(self.frequencies) * 2, hidden_dim // 2, 
                                            prior_type, prior_params)
                self.prior_type = prior_type
                self.prior_params = prior_params
            
            def forward(self, batch_size, device, sample=True):
                # Create time steps
                t = torch.linspace(0, 1, self.seq_length).to(device)
                t = t.view(1, -1, 1)  # [1, seq_length, 1]
                t = t.repeat(batch_size, 1, 1)  # [batch_size, seq_length, 1]
                
                # Generate sinusoidal components at different frequencies
                freqs = self.frequencies.view(1, 1, -1)  # [1, 1, num_frequencies]
                # Scale frequencies by seq_length to match the desired cycles per window
                phases = 2 * torch.pi * t * freqs * self.seq_length  # [batch_size, seq_length, num_frequencies]
                
                # Create sin and cos features
                sin_features = torch.sin(phases)  # [batch_size, seq_length, num_frequencies]
                cos_features = torch.cos(phases)  # [batch_size, seq_length, num_frequencies]
                
                # Concatenate and project with sampling
                combined = torch.cat([sin_features, cos_features], dim=-1)  # [batch_size, seq_length, 2*num_frequencies]
                return self.projection(combined, sample)  # [batch_size, seq_length, hidden_dim//2]
            
            def kl_divergence(self):
                """Sum up the KL divergences from all Bayesian layers."""
                kl_total = 0.0
                for module in self.modules():
                    if hasattr(module, 'kl_divergence'):
                        kl_total += module.kl_divergence()
                return kl_total
        
        # Keep BayesSinusoidalEncoding as is with Bayesian projection
        self.sinusoidal_encoding = BayesSinusoidalEncoding(seq_length, hidden_dim, prior_type, prior_params)
        
        # Enhanced output processing - convert to all standard layers
        self.output_conv = nn.Sequential(
            # Transpose from [batch, seq_len, hidden_dim*2] to [batch, hidden_dim*2, seq_len]
            TransposeLayer(1, 2),
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim // 2, 1, kernel_size=3, padding=1),  # Standard layer instead of Bayesian
            nn.Tanh(),  # Use tanh for oscillatory signals with positive and negative values
            # Transpose back to [batch, seq_len, 1]
            TransposeLayer(1, 2)
        )
        
        # Condition-specific adaptation layers - convert to standard layers
        self.stress_layer = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=1),
        )
        self.stress_layer[0].weight.data.fill_(1.8)  # Initialized like base version

        self.amusement_layer = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=3, padding=1),
        )
        # Initialize smoothing filter weights for amusement
        self.amusement_layer[0].weight.data.fill_(0)
        self.amusement_layer[0].weight.data[:, :, 0] = 0.15
        self.amusement_layer[0].weight.data[:, :, 1] = 0.7
        self.amusement_layer[0].weight.data[:, :, 2] = 0.15
        # Initialize bias to shift the baseline down slightly for amusement
        if hasattr(self.amusement_layer[0], 'bias'):
            self.amusement_layer[0].bias.data.fill_(-0.3)
    
    def forward(self, z, labels, sample=True):
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
        coarse_out, _ = self.lstm_coarse(signal_in, None, sample)  # Fixed parameter order
        fine_out, _ = self.lstm_fine(signal_in)      # Standard LSTM
        
        # Generate explicit oscillatory pattern with standard layers
        oscillations = self.oscillation_generator(h)
        oscillations = oscillations.unsqueeze(-1)     # [batch_size, seq_length, 1]
        
        # Generate sinusoidal encoding for forced periodicity - keep Bayesian
        sinusoidal_features = self.sinusoidal_encoding(batch_size, z.device, sample)
        
        # Ensure sinusoidal_features has the correct shape [batch, seq_length, hidden_dim//2]
        # We need to reshape it to [batch, seq_length, 1] to match other components
        sinusoidal_features_reshaped = sinusoidal_features.mean(dim=2, keepdim=True)
        
        # Generate cardiac parameters with standard layers
        cardiac_params = []
        for i in range(batch_size):
            # Combine coarse and fine features
            features = torch.cat([coarse_out[i].mean(dim=0), fine_out[i].mean(dim=0)])
            
            # Generate parameters with standard layers
            params = self.cardiac_params(features)
            
            cardiac_params.append(params)
        cardiac_params = torch.stack(cardiac_params)
        
        # Extract parameters
        frequencies = cardiac_params[:, 0] * 0.08 + 0.21  # Range 0.21-0.29 Hz
        amplitudes = cardiac_params[:, 1] * 3.0 + 1.0    # Range 1.0-4.0 for z-score normalized data
        phases = cardiac_params[:, 2] * 2 * torch.pi      # Range 0-2π
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
            oscillation = amplitudes[i] * torch.sin(2 * torch.pi * frequencies[i] * self.seq_length * t + phases[i])
            # Add baseline shift
            oscillation = oscillation + baselines[i]
            # Add to batch
            cardiac_oscillations[i, :, 0] = oscillation
        
        # Combine coarse and fine features for base signal
        combined = torch.cat([coarse_out, fine_out], dim=2)  # [batch, seq_length, hidden_dim*2]
        
        # Generate base signal with standard layers
        base_signal = self.output_conv(combined)
        
        # Add the oscillatory components with appropriate weighting (same as base model)
        enhanced_signal = (
            0.1 * base_signal + 
            0.1 * oscillations +
            0.7 * cardiac_oscillations +
            0.1 * sinusoidal_features_reshaped
        )
        
        # Create output tensor for condition-specific outputs
        output_signal = torch.zeros_like(enhanced_signal)
        
        # Apply condition-specific transformations with standard layers
        for i in range(batch_size):
            sample_signal = enhanced_signal[i:i+1]  # Keep batch dimension
            condition = labels[i].item()
            
            if condition == 1:  # Baseline - use as is
                output_signal[i:i+1] = sample_signal
            elif condition == 2:  # Stress - higher amplitude
                # Use standard layer for amplitude scaling
                transposed = sample_signal.transpose(1, 2)  # [1, 1, seq_len]
                scaled = self.stress_layer(transposed)
                output_signal[i:i+1] = scaled.transpose(1, 2)
            elif condition == 3:  # Amusement - smoother
                # Use standard convolutional layer for smoothing
                transposed = sample_signal.transpose(1, 2)  # [1, 1, seq_len]
                smoothed = self.amusement_layer(transposed)
                output_signal[i:i+1] = smoothed.transpose(1, 2)
        
        return output_signal
    
    def kl_divergence(self):
        """Sum up the KL divergences from all Bayesian layers."""
        kl_total = 0.0
        from src.models.variational_layers import BayesLinear, BayesConv1d, BayesLSTM, BayesConv2d
        
        # Recursively check all modules to find Bayesian layers
        # This properly handles nested modules like Sequential
        for module in self.modules():
            # Check if module is a Bayesian layer itself (not a container)
            if isinstance(module, (BayesLinear, BayesConv1d, BayesLSTM, BayesConv2d)):
                kl_total += module.kl_divergence()
        
        return kl_total

class BayesBVPMultiScaleTimeSeriesGAN(BayesTimeSeriesGAN):
    """
    Specialized Bayesian GAN for BVP signals that uses a multi-scale generator to better capture
    both cardiac cycles and slower baseline variations.
    """
    def __init__(self,
                 latent_dim=100,
                 seq_length=30,
                 hidden_dim=128,
                 num_layers=2,
                 num_classes=3,
                 prior_type=PriorType.GAUSSIAN,
                 prior_params={'sigma': 1.0}):
        # Initialize the base class but don't use its generator or discriminator
        super().__init__(
            latent_dim=latent_dim,
            seq_length=seq_length,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            prior_type=prior_type,
            prior_params=prior_params
        )
        
        # Create a custom Bayesian multi-scale generator with BVP-specific parameters
        self.generator = BayesBVPMultiScaleGenerator(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            seq_length=seq_length,
            prior_type=prior_type,
            prior_params=prior_params
        )
        
        # Use the BVP-specific Bayesian discriminator
        self.discriminator = BayesBVPTimeSeriesDiscriminator(
            seq_length=seq_length,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            prior_type=prior_type,
            prior_params=prior_params
        )

class BayesFeatureGenerator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_classes,
                prior_type=PriorType.GAUSSIAN, prior_params={'sigma': 1.0}):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.prior_type = prior_type
        self.prior_params = prior_params

        # Label embedding for condition (1: Baseline, 2: Stress, 3: Amusement)
        self.label_embedding = nn.Embedding(num_classes + 1, hidden_dim)  # +1 for padding idx 0

        # Main feature processing network with Bayesian internal layer to model feature relationships
        self.feature_processor = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            BayesLinear(hidden_dim, hidden_dim, prior_type, prior_params),  # CONVERT TO BAYESIAN to model feature uncertainties
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2)
        )

        # Output heads for different feature types - CONVERT TO STANDARD LAYERS
        # HRV features: RMSSD, SDNN, LF, HF, LF_HF_ratio
        self.hrv_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 5),  # Standard layer instead of Bayesian
            nn.SiLU()  # Smooth positive activation for physiological metrics
        )

        # EDA features: mean_EDA, median_EDA, SCR_count
        self.eda_features_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 3),  # Standard layer instead of Bayesian
            nn.SiLU()  # Smooth positive activation for physiological metrics
        )

    def forward(self, z, labels, sample=True):
        # Get label embeddings
        label_emb = self.label_embedding(labels)

        # Combine noise and label information
        combined_input = torch.cat([z, label_emb], dim=1)
        
        # Process through main network with sampling for Bayesian layer
        x = self.feature_processor[0](combined_input)
        x = self.feature_processor[1](x)
        x = self.feature_processor[2](x)
        features = self.feature_processor[3](x, sample)  # Bayesian layer
        features = self.feature_processor[4](features)
        features = self.feature_processor[5](features)

        # Generate different feature types with standard layers
        hrv_features = self.hrv_out(features)
        eda_features = self.eda_features_out(features)

        return {
            'HRV_features': hrv_features,
            'EDA_features': eda_features
        }
    
    def kl_divergence(self):
        """Sum up the KL divergences from all Bayesian layers."""
        kl_total = 0.0
        from src.models.variational_layers import BayesLinear, BayesConv1d, BayesLSTM, BayesConv2d
        
        # Recursively check all modules to find Bayesian layers
        # This properly handles nested modules like Sequential
        for module in self.modules():
            # Check if module is a Bayesian layer itself (not a container)
            if isinstance(module, (BayesLinear, BayesConv1d, BayesLSTM, BayesConv2d)):
                kl_total += module.kl_divergence()
        
        return kl_total

class BayesFeatureDiscriminator(nn.Module):
    def __init__(self, hidden_dim, num_classes,
                prior_type=PriorType.GAUSSIAN, prior_params={'sigma': 1.0}):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.prior_type = prior_type
        self.prior_params = prior_params

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

    def forward(self, features, labels, sample=True):
        # Process each feature type
        hrv_features = self.hrv_processor(features['HRV_features'])
        eda_features = self.eda_processor(features['EDA_features'])

        # Get label embeddings
        label_emb = self.label_embedding(labels)

        # Combine all features
        combined = torch.cat([hrv_features, eda_features, label_emb], dim=1)

        # Process combined features
        features = self.feature_processor(combined)

        # Final classification (raw logits)
        return self.classifier(features)
        
    def extract_features(self, features, labels, sample=True):
        """
        Extract intermediate features for feature matching loss.
        Returns activations from the penultimate layer.
        """
        # Process each feature type
        hrv_features = self.hrv_processor(features['HRV_features'])
        eda_features = self.eda_processor(features['EDA_features'])

        # Get label embeddings
        label_emb = self.label_embedding(labels)

        # Combine all features
        combined = torch.cat([hrv_features, eda_features, label_emb], dim=1)

        # Process combined features - this is the intermediate representation we want
        return self.feature_processor(combined)
    
    def kl_divergence(self):
        """Sum up the KL divergences from all Bayesian layers."""
        # No Bayesian layers left, return 0 as a tensor
        return torch.tensor(0.0, device=next(self.parameters()).device)

class BayesFeatureGAN(nn.Module):
    """
    Bayesian GAN architecture for generating correlated physiological features.
    Generates both HRV features (5-dim) and EDA features (3-dim) conditioned on stress state.
    """
    def __init__(self, 
                 latent_dim=100,
                 hidden_dim=128,
                 num_classes=3,    # 3 stress conditions (1: Baseline, 2: Stress, 3: Amusement)
                 prior_type=PriorType.GAUSSIAN,
                 prior_params={'sigma': 1.0}):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.prior_type = prior_type
        self.prior_params = prior_params
        
        # Initialize generator and discriminator
        self.generator = BayesFeatureGenerator(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            prior_type=prior_type,
            prior_params=prior_params
        )
        
        self.discriminator = BayesFeatureDiscriminator(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            prior_type=prior_type,
            prior_params=prior_params
        )
    
    def generate(self, z, labels, sample=True):
        """Generate HRV and EDA features from latent vector z and condition labels."""
        return self.generator(z, labels, sample)
    
    def discriminate(self, features, labels, sample=True):
        """Discriminate real vs. fake features. Returns raw logits for training stability."""
        return self.discriminator(features, labels, sample)
    
    def discriminate_prob(self, features, labels, sample=True):
        """Discriminate real vs. fake features. Returns probabilities."""
        return torch.sigmoid(self.discriminator(features, labels, sample))
        
    def feature_matching_loss(self, real_features, fake_features, labels, sample=True):
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
        real_intermediate = self.discriminator.extract_features(real_features, labels, sample)
        fake_intermediate = self.discriminator.extract_features(fake_features, labels, sample)
        
        return F.mse_loss(fake_intermediate, real_intermediate)
    
    def kl_loss(self):
        """
        Sum up the KL divergences of remaining Bayesian layers (generator only).
        """
        # Only use KL divergence from the generator
        g_kl = self.generator.kl_divergence()
        return g_kl
        
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
        # Store original evaluation/training mode
        training_mode = self.training
        
        # Use eval mode but we'll still be sampling weights
        self.eval()
        
        hrv_samples = []
        eda_samples = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                features = self.generate(z, labels, sample=True)
                hrv_samples.append(features['HRV_features'])
                eda_samples.append(features['EDA_features'])
        
        # Stack samples along a new dimension
        hrv_samples = torch.stack(hrv_samples, dim=0)  # [n_samples, batch_size, 5]
        eda_samples = torch.stack(eda_samples, dim=0)  # [n_samples, batch_size, 3]
        
        # Calculate mean and standard deviation
        hrv_mean = torch.mean(hrv_samples, dim=0)  # [batch_size, 5]
        hrv_std = torch.std(hrv_samples, dim=0)    # [batch_size, 5]
        
        eda_mean = torch.mean(eda_samples, dim=0)  # [batch_size, 3]
        eda_std = torch.std(eda_samples, dim=0)    # [batch_size, 3]
        
        # Restore original mode
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
        # Store original evaluation/training mode
        training_mode = self.training
        
        # Use eval mode but we'll still be sampling weights
        self.eval()
        
        hrv_samples = []
        eda_samples = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                features = self.generate(z, labels, sample=True)
                hrv_samples.append(features['HRV_features'])
                eda_samples.append(features['EDA_features'])
        
        # Restore original mode
        self.train(training_mode)
        
        return {
            'HRV_features': hrv_samples,
            'EDA_features': eda_samples
        }
