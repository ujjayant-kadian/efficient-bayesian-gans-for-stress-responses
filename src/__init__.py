"""
Bayesian GAN Stress Response Analysis

This package implements a Bayesian Generative Adversarial Network (GAN) 
for analyzing stress responses. It includes modules for data processing,
model architecture, training procedures, and evaluation metrics.

The package provides three main GAN architectures:
1. Base GANs (TimeSeriesGAN and FeatureGAN) for generating physiological signals and features
2. Bayesian GANs with different uncertainty estimation approaches
3. Evaluation tools for comparing model performance and uncertainty estimates
"""

__version__ = '0.1.0'
__author__ = 'kadianu'

# Import base models
from .models.base_gan import (
    TimeSeriesGAN, 
    FeatureGAN,
    EDATimeSeriesGAN,
    EDAMultiScaleTimeSeriesGAN,
    BVPTimeSeriesGAN,
    BVPMultiScaleTimeSeriesGAN
)

# Import Bayesian models
from .models.dropout_bayes_gan import (
    BayesianTimeSeriesGAN,
    BayesianEDATimeSeriesGAN,
    BayesianEDAMultiScaleTimeSeriesGAN,
    BayesianBVPTimeSeriesGAN,
    BayesianBVPMultiScaleTimeSeriesGAN,
    BayesianFeatureGAN
)
from .models.variational_layers import (
    PriorType,
    BayesLinear,
    BayesConv1d,
    BayesConv2d,
    BayesLSTM
)

from .models.variational_bayes_gan import (
    BayesTimeSeriesGAN,
    BayesEDATimeSeriesGAN,
    BayesEDAMultiScaleTimeSeriesGAN,
    BayesBVPTimeSeriesGAN,
    BayesBVPMultiScaleTimeSeriesGAN,
    BayesFeatureGAN
)

# Import training modules
from .training.train_base_gan import (
    train_gan,
    prepare_data,
    evaluate_mmd,
    plot_history
)
from .training.train_bdgan import train_bayesian_gan
from .training.train_var_bayes_gan import (
    train_gan as train_var_bayes_gan,
    plot_bayes_gan_history
)

# Import evaluation modules
from .evaluation.uncertainty_eval import uncertainty_evaluation_summary
from .evaluation.metrics import (
    calculate_feature_statistics,
    calculate_fid_like_score,
    calculate_mmd,
    calculate_rmse
)

# Define what should be exported with 'from src import *'
__all__ = [
    # Base Models
    'TimeSeriesGAN',
    'FeatureGAN',
    'EDATimeSeriesGAN',
    'EDAMultiScaleTimeSeriesGAN',
    'BVPTimeSeriesGAN',
    'BVPMultiScaleTimeSeriesGAN',
    
    # Bayesian Models - Dropout
    'BayesianTimeSeriesGAN',
    'BayesianEDATimeSeriesGAN',
    'BayesianEDAMultiScaleTimeSeriesGAN',
    'BayesianBVPTimeSeriesGAN',
    'BayesianBVPMultiScaleTimeSeriesGAN',
    'BayesianFeatureGAN',
    'VariationalBayesGAN',
    'BayesGAN',

    # Bayesian Models - Variational Bayes
    'BayesTimeSeriesGAN',
    'BayesEDATimeSeriesGAN',
    'BayesEDAMultiScaleTimeSeriesGAN',
    'BayesBVPTimeSeriesGAN',
    'BayesBVPMultiScaleTimeSeriesGAN',
    'BayesFeatureGAN',
     
    # Bayesian Layers
    'PriorType',
    'BayesLinear',
    'BayesConv1d',
    'BayesConv2d',
    'BayesLSTM',
    
    # Training functions
    'train_gan',
    'prepare_data',
    'train_bayesian_gan',
    'train_var_bayes_gan',
    'train_bayes_gan',
    # 'evaluate_mmd',
    # 'plot_history',
    
    # Evaluation functions
    'uncertainty_evaluation_summary',
    'calculate_feature_statistics',
    'calculate_fid_like_score',
    'calculate_mmd',
    'calculate_rmse'
]
