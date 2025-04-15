# Efficient Bayesian GANs for Simulating Physiological Responses to Virtual Stressors

This project develops and evaluates efficient Bayesian Generative Adversarial Networks (GANs) to simulate realistic physiological responses, such as Electrodermal Activity (EDA) and Blood Volume Pulse (BVP), to virtual stressors. It aims to overcome the limitations of standard GANs by improving simulation accuracy and providing essential uncertainty quantification, comparing approaches like Variational Bayesian GANs and Bayesian Dropout GANs.

## Table of Contents
- [Project Description](#project-description)
- [Project Structure](#project-structure)
- [Key Features](#key-features)
- [Visuals](#visuals)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
  - [Training Models](#training-models)
  - [Evaluating Models](#evaluating-models)
- [Development Strategies](#development-strategies)
- [Author](#author)
- [License](#license)

## Project Description

This project addresses the critical need for accurate and reliable simulation of human physiological responses, specifically Electrodermal Activity (EDA) and Blood Volume Pulse (BVP), particularly in reaction to virtual stressors. While standard Generative Adversarial Networks (GANs) have potential, they often struggle with instability, mode collapse, and crucially, lack the ability to quantify the uncertainty inherent in physiological signals and their generation process. To address these shortcomings, this research focuses on developing, implementing, and rigorously evaluating efficient Bayesian GAN architectures tailored for physiological time-series data.

The project investigates and compares multiple approaches:
- A standard GAN as a baseline
- A Variational Bayesian GAN leveraging variational inference techniques
- A computationally efficient Bayesian Dropout GAN (BDGAN) that utilizes Monte Carlo Dropout for approximate Bayesian inference

Key contributions include adapting these models for the specific characteristics of EDA and BVP signals and conducting a comparative analysis focusing on the trade-offs between generated signal fidelity, computational efficiency, and the quality of uncertainty quantification.

Ultimately, this work aims to provide more robust, reliable, and computationally feasible tools for physiological simulation, advancing research in stress modeling, human-computer interaction, and the creation of adaptive virtual environments.

## Project Structure

```
.
├── data/                  # Processed physiological data
├── src/                   # Source code
│   ├── models/            # Model implementations
│   │   ├── base_gan.py            # Standard GAN implementation
│   │   ├── variational_bayes_gan.py # Variational Bayesian GAN implementation
│   │   ├── dropout_bayes_gan.py   # Bayesian Dropout GAN implementation
│   │   └── variational_layers.py  # Custom variational layers
│   ├── training/          # Training scripts
│   │   ├── train_base_gan.py      # Training script for standard GAN
│   │   ├── train_var_bayes_gan.py # Training script for Variational Bayesian GAN
│   │   ├── train_bdgan.py         # Training script for Bayesian Dropout GAN
│   │   └── generate_and_plot.py   # Utilities for generating samples and plotting
│   ├── evaluation/        # Evaluation scripts
│   │   ├── metrics.py             # Evaluation metrics implementation
│   │   ├── correlation_evaluation.py # Correlation analysis
│   │   ├── uncertainty_eval.py    # Uncertainty evaluation
│   │   └── evaluate_all.py        # Script to run all evaluations
│   └── data/              # Data processing utilities
├── final_models/                # Saved model configuration files, and training history plots
│   ├── baseline-v1-eda/   # Standard GAN for EDA
│   ├── baseline-v1-bvp/   # Standard GAN for BVP
│   ├── var-bayes-v1-eda-gaussian/ # Variational Bayesian GAN with Gaussian prior for EDA
│   ├── dropout-v1-eda/    # Bayesian Dropout GAN for EDA
│   └── ...                # Other model variants
├── plots/                 # Generated plots and visualizations
├── generated_data/        # Generated physiological signals
├── evaluation_results/    # Evaluation metrics and results
├── correlation_analysis_results/ # Correlation analysis outputs
├── notebooks/             # Jupyter notebooks for analysis and visualization
├── requirements.txt       # Python dependencies
└── setup.py               # Package setup script
```

## Key Features

- **Multiple Bayesian GAN Architectures**: Implementation of standard GAN, Variational Bayesian GAN, and Bayesian Dropout GAN architectures
- **Physiological Signal Generation**: Specialized for EDA and BVP time-series data
- **Uncertainty Quantification**: Methods to estimate and evaluate the uncertainty in generated signals
- **Comparative Analysis Framework**: Tools to compare model performance across multiple metrics
- **Efficient Implementation**: Optimized for computational efficiency while maintaining Bayesian properties
- **Comprehensive Evaluation**: Metrics for signal fidelity, statistical properties, and uncertainty quality
- **Visualizations**: Extensive plotting capabilities for generated signals and evaluation results

## Visuals

The project includes various visualizations:

- **Generated Physiological Signals**: Available in the `generated_data/` directory
- **Training Progress**: Training history plots in each model's directory
- **Evaluation Results**: Detailed evaluation metrics and plots in `evaluation_results/`
- **Correlation Analysis**: Correlation plots in `correlation_analysis_results/`
- **Visualization Gallery**: Various visualizations in the `plots/` directory

## Prerequisites

- Python 3.8+
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- tqdm

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training Models

#### Standard GAN

```bash
python -m src.training.train_base_gan --data_type eda --output_dir models/baseline-v1-eda
```

Common training parameters:
- `--data_path`: Path to preprocessed data (.pt file) [required]
- `--batch_size`: Batch size for training (default: 64)
- `--val_split`: Validation split ratio (default: 0.2)
- `--test_split`: Test split ratio (default: 0.1)
- `--latent_dim`: Dimension of latent space (default: 100)
- `--hidden_dim`: Hidden dimension size (default: 256)
- `--num_layers`: Number of LSTM layers for time series models (default: 2)
- `--model_type`: Type of model to train: 'all', 'EDA', 'BVP', or 'features' (default: 'all')
- `--use_multiscale`: Use multi-scale generator for improved signal quality
- `--lr_g`: Generator learning rate (default: 0.0002)
- `--lr_d`: Discriminator learning rate (default: 0.0001)
- `--beta1`: Beta1 for Adam optimizer (default: 0.5)
- `--beta2`: Beta2 for Adam optimizer (default: 0.999)
- `--n_epochs`: Number of training epochs (default: 200)
- `--save_interval`: Save model every N epochs (default: 10)
- `--patience`: Patience for early stopping (default: 50)
- `--seed`: Random seed for reproducibility (default: 42)
- `--initial_noise`: Initial instance noise level for training stability (default: 0.15)

#### Variational Bayesian GAN

```bash
python -m src.training.train_var_bayes_gan --data_path data/processed_data.pt --model_type eda --output_dir models/var-bayes-v1-eda-gaussian --prior gaussian
```

Additional parameters for Variational Bayesian GAN:
- `--prior`: Prior distribution type: 'gaussian', 'laplace', or 'scaled_mixture' (default: 'gaussian')
- `--kl_weight`: Weight for KL divergence term in loss function (default: 0.01)
- `--prior_sigma`: Standard deviation for Gaussian prior (default: 1.0)
- `--prior_scale`: Scale parameter for Laplace prior (default: 1.0)
- `--mixture_scale1`: First scale parameter for scaled mixture prior (default: 0.5)
- `--mixture_scale2`: Second scale parameter for scaled mixture prior (default: 2.0)
- `--mixture_weight`: Mixing weight for scaled mixture prior (default: 0.5)

#### Bayesian Dropout GAN

```bash
python -m src.training.train_bdgan --data_path data/processed_data.pt --model_type eda --output_dir models/dropout-v1-eda
```

Additional parameters for Bayesian Dropout GAN:
- `--dropout_rate`: Dropout rate for Bayesian Dropout (default: 0.2)
- `--weight_decay`: Weight decay for L2 regularization (default: 1e-5)
- `--mc_samples`: Number of Monte Carlo samples during training (default: 5)

### Evaluating Models

#### Run All Evaluations

```bash
python -m src.evaluation.evaluate_all --model_dir models/baseline-v1-eda --test_data_path data/processed_data.pt
```

Evaluation parameters:
- `--model_dir`: Path to the model directory containing 'best_*.pt' model checkpoint and 'config.json' [required]
- `--test_data_path`: Path to the .pt file containing the real test dataset [required]
- `--output_dir`: Directory to save evaluation metrics (default: "evaluation_results")
- `--condition`: Condition label to evaluate: 1=Baseline, 2=Stress, 3=Amusement (default: 2)
- `--num_gen_samples`: Number of samples to generate for evaluation. If 0, uses the number of real samples for the condition (default: 0)
- `--uncertainty_samples`: Number of uncertainty samples per latent vector for Bayesian models (default: 10)
- `--num_latent_vectors`: Number of latent vectors for uncertainty evaluation in Bayesian models (default: 10)
- `--seed`: Random seed for reproducibility (default: 42)

#### Correlation Analysis

```bash
python -m src.evaluation.correlation_evaluation --model_dir models/baseline-v1-eda --data_type eda --test_data_path data/processed_data.pt
```

Correlation analysis parameters:
- `--model_dir`: Path to the model directory [required]
- `--data_type`: Type of data to evaluate: 'eda', 'bvp', or 'features' [required]
- `--test_data_path`: Path to the test data [required]
- `--output_dir`: Directory to save correlation results (default: "correlation_analysis_results")
- `--num_samples`: Number of samples to generate (default: 100)
- `--condition`: Condition to evaluate: 1=Baseline, 2=Stress, 3=Amusement (default: 2)
- `--save_plots`: Whether to save correlation plots (default: True)

## Development Strategies

For improving the existing models or developing new ones:

1. **Model Architecture Modifications**:
   - Modify network architectures in `src/models/`
   - Experiment with different layer configurations and activation functions
   - Consider implementing Stochastic Gradient Hamiltonian Monte Carlo (SGHMC) based GAN for improved uncertainty estimation and sampling efficiency

2. **Prior Distribution Selection**:
   - For Variational Bayesian GANs, experiment with different prior distributions in `src/models/variational_bayes_gan.py`
   - Options include Gaussian, Laplace, and scaled mixture priors

3. **Hyperparameter Tuning**:
   - Adjust learning rates, batch sizes, and training epochs
   - Modify dropout rates for Bayesian Dropout GANs

4. **Custom Metrics**:
   - Implement additional evaluation metrics in `src/evaluation/metrics.py`

5. **Data Collection and Processing**:
   - Explore alternative physiological datasets beyond WESAD, such as AffectiveROAD, DEAP, or AMIGOS
   - Consider collecting custom dataset with more diverse stressors and participant demographics
   - Enhance data preprocessing methods in `src/data/`
   - Experiment with different normalization and filtering techniques

## Author

Ujjayant Kadian

## License

This project is licensed under the MIT License - see the LICENSE file for details.

This work was completed as part of a dissertation project. 