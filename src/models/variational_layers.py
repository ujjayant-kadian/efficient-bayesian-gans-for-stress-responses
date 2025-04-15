# variational_layers.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum

class PriorType(Enum):
    GAUSSIAN = 'gaussian'
    SCALED_MIXTURE_GAUSSIAN = 'scaled_mixture_gaussian'
    LAPLACE = 'laplace'

class BayesLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_type=PriorType.GAUSSIAN, prior_params={'sigma': 1.0}):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_type = prior_type
        self.prior_params = prior_params
        
        # Mean and log-variance of approximate posterior
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_logvar = nn.Parameter(torch.zeros(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_logvar = nn.Parameter(torch.zeros(out_features))
        
        # Initialize the parameters (optional heuristics here)
        nn.init.xavier_normal_(self.weight_mu)
        nn.init.constant_(self.weight_logvar, -3.0)  # small initial variance

    def forward(self, x, sample=True):
        if sample:
            # Sample weights using reparameterization
            eps_w = torch.randn_like(self.weight_mu)
            eps_b = torch.randn_like(self.bias_mu)
            
            weight = self.weight_mu + torch.exp(0.5 * self.weight_logvar) * eps_w
            bias = self.bias_mu + torch.exp(0.5 * self.bias_logvar) * eps_b
        else:
            # Use mean only (for a deterministic forward pass)
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def kl_divergence(self):
        """KL(q(W|theta) || p(W)) for both weight and bias."""
        weight_var = torch.exp(self.weight_logvar)
        bias_var = torch.exp(self.bias_logvar)
        
        if self.prior_type == PriorType.GAUSSIAN:
            device = self.weight_mu.device
            dtype = self.weight_mu.dtype
            prior_sigma = self.prior_params.get('sigma', 1.0)
            prior_sigma_tensor = torch.tensor(prior_sigma**2, device=device, dtype=dtype)
            
            kl_weight = 0.5 * (
                weight_var / prior_sigma_tensor + 
                (self.weight_mu**2) / prior_sigma_tensor - 
                self.weight_logvar +
                torch.log(prior_sigma_tensor)
                - 1
            ).sum()
            
            kl_bias = 0.5 * (
                bias_var / prior_sigma_tensor + 
                (self.bias_mu**2) / prior_sigma_tensor - 
                self.bias_logvar +
                torch.log(prior_sigma_tensor)
                - 1
            ).sum()
            
        elif self.prior_type == PriorType.SCALED_MIXTURE_GAUSSIAN:
            # Scaled mixture of Gaussians prior (as in Blundell et al. 2015)
            device = self.weight_mu.device
            dtype = self.weight_mu.dtype
            
            sigma1 = self.prior_params.get('sigma1', 1.0)
            sigma2 = self.prior_params.get('sigma2', 0.1)
            pi = self.prior_params.get('pi', 0.5)  # Mixing coefficient
            
            # Convert to tensors
            sigma1_sq = torch.tensor(sigma1**2, device=device, dtype=dtype)
            sigma2_sq = torch.tensor(sigma2**2, device=device, dtype=dtype)
            pi_tensor = torch.tensor(pi, device=device, dtype=dtype)
            const_log_2pi = torch.log(torch.tensor(2 * torch.pi, device=device, dtype=dtype))
            const_e = torch.tensor(torch.e, device=device, dtype=dtype)
            
            # Log of mixture of Gaussians
            log_gaussian1_w = -0.5 * (const_log_2pi + 
                               torch.log(sigma1_sq) + 
                               self.weight_mu**2 / sigma1_sq)
            log_gaussian2_w = -0.5 * (const_log_2pi + 
                               torch.log(sigma2_sq) + 
                               self.weight_mu**2 / sigma2_sq)
            log_gaussian1_b = -0.5 * (const_log_2pi + 
                               torch.log(sigma1_sq) + 
                               self.bias_mu**2 / sigma1_sq)
            log_gaussian2_b = -0.5 * (const_log_2pi + 
                               torch.log(sigma2_sq) + 
                               self.bias_mu**2 / sigma2_sq)
            
            # Use logsumexp for numerical stability (optional)
            stacked_w = torch.stack([torch.log(pi_tensor) + log_gaussian1_w, 
                                    torch.log(1 - pi_tensor) + log_gaussian2_w], dim=0)
            stacked_b = torch.stack([torch.log(pi_tensor) + log_gaussian1_b, 
                                    torch.log(1 - pi_tensor) + log_gaussian2_b], dim=0)
                                    
            log_mix_w = torch.logsumexp(stacked_w, dim=0)
            log_mix_b = torch.logsumexp(stacked_b, dim=0)
            
            # Entropy of q(w|θ)
            entropy_w = 0.5 * (torch.log(2 * const_e * torch.pi) + self.weight_logvar).sum()
            entropy_b = 0.5 * (torch.log(2 * const_e * torch.pi) + self.bias_logvar).sum()
            
            # KL = E_q[-log p(w)] - entropy
            kl_weight = -log_mix_w.sum() - entropy_w
            kl_bias = -log_mix_b.sum() - entropy_b
            
        elif self.prior_type == PriorType.LAPLACE:
            # Laplace prior
            device = self.weight_mu.device
            dtype = self.weight_mu.dtype
            b_param = self.prior_params.get('b', 1.0)
            b = torch.tensor(b_param, device=device, dtype=dtype)
            log_2b = torch.log(2 * b)
            
            # For Laplace, KL doesn't have a simple closed form, approximate using MC
            n_samples = 10
            kl_weight = torch.tensor(0.0, device=device, dtype=dtype)
            kl_bias = torch.tensor(0.0, device=device, dtype=dtype)
            
            for _ in range(n_samples):
                eps_w = torch.randn_like(self.weight_mu)
                eps_b = torch.randn_like(self.bias_mu)
                
                w_sample = self.weight_mu + torch.exp(0.5 * self.weight_logvar) * eps_w
                b_sample = self.bias_mu + torch.exp(0.5 * self.bias_logvar) * eps_b
                
                log_q_w = -0.5 * (self.weight_logvar + (w_sample - self.weight_mu)**2 / weight_var)
                log_q_b = -0.5 * (self.bias_logvar + (b_sample - self.bias_mu)**2 / bias_var)
                
                log_p_w = -torch.abs(w_sample) / b - log_2b
                log_p_b = -torch.abs(b_sample) / b - log_2b
                
                kl_weight += (log_q_w - log_p_w).sum() / n_samples
                kl_bias += (log_q_b - log_p_b).sum() / n_samples
        
        return kl_weight + kl_bias


class BayesConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 prior_type=PriorType.GAUSSIAN, prior_params={'sigma': 1.0}):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.prior_type = prior_type
        self.prior_params = prior_params
        
        # Mean and log-variance for weight posterior
        self.weight_mu = nn.Parameter(torch.zeros(out_channels, in_channels, *self.kernel_size))
        self.weight_logvar = nn.Parameter(torch.zeros(out_channels, in_channels, *self.kernel_size))
        
        # Mean and log-variance for bias posterior
        self.bias_mu = nn.Parameter(torch.zeros(out_channels))
        self.bias_logvar = nn.Parameter(torch.zeros(out_channels))
        
        # Initialize the parameters
        nn.init.kaiming_normal_(self.weight_mu, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.weight_logvar, -3.0)
        
    def forward(self, x, sample=True):
        if sample:
            # Sample weights using reparameterization trick
            eps_w = torch.randn_like(self.weight_mu)
            eps_b = torch.randn_like(self.bias_mu)
            
            weight = self.weight_mu + torch.exp(0.5 * self.weight_logvar) * eps_w
            bias = self.bias_mu + torch.exp(0.5 * self.bias_logvar) * eps_b
        else:
            # Use mean only for deterministic pass
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.conv2d(x, weight, bias, stride=self.stride, padding=self.padding)
    
    def kl_divergence(self):
        """KL(q(W|theta) || p(W)) for weight and bias."""
        weight_var = torch.exp(self.weight_logvar)
        bias_var = torch.exp(self.bias_logvar)
        
        if self.prior_type == PriorType.GAUSSIAN:
            device = self.weight_mu.device
            dtype = self.weight_mu.dtype
            prior_sigma = self.prior_params.get('sigma', 1.0)
            prior_sigma_tensor = torch.tensor(prior_sigma**2, device=device, dtype=dtype)
            
            kl_weight = 0.5 * (
                weight_var / prior_sigma_tensor + 
                (self.weight_mu**2) / prior_sigma_tensor - 
                self.weight_logvar +
                torch.log(prior_sigma_tensor)
                - 1
            ).sum()
            
            kl_bias = 0.5 * (
                bias_var / prior_sigma_tensor + 
                (self.bias_mu**2) / prior_sigma_tensor - 
                self.bias_logvar +
                torch.log(prior_sigma_tensor)
                - 1
            ).sum()
            
        elif self.prior_type == PriorType.SCALED_MIXTURE_GAUSSIAN:
            device = self.weight_mu.device
            dtype = self.weight_mu.dtype
            
            sigma1 = self.prior_params.get('sigma1', 1.0)
            sigma2 = self.prior_params.get('sigma2', 0.1)
            pi = self.prior_params.get('pi', 0.5)
            
            # Convert to tensors
            sigma1_sq = torch.tensor(sigma1**2, device=device, dtype=dtype)
            sigma2_sq = torch.tensor(sigma2**2, device=device, dtype=dtype)
            pi_tensor = torch.tensor(pi, device=device, dtype=dtype)
            const_log_2pi = torch.log(torch.tensor(2 * torch.pi, device=device, dtype=dtype))
            const_e = torch.tensor(torch.e, device=device, dtype=dtype)
            
            log_gaussian1_w = -0.5 * (const_log_2pi + 
                               torch.log(sigma1_sq) + 
                               self.weight_mu**2 / sigma1_sq)
            log_gaussian2_w = -0.5 * (const_log_2pi + 
                               torch.log(sigma2_sq) + 
                               self.weight_mu**2 / sigma2_sq)
            log_gaussian1_b = -0.5 * (const_log_2pi + 
                               torch.log(sigma1_sq) + 
                               self.bias_mu**2 / sigma1_sq)
            log_gaussian2_b = -0.5 * (const_log_2pi + 
                               torch.log(sigma2_sq) + 
                               self.bias_mu**2 / sigma2_sq)
            
            # Use logsumexp for numerical stability
            stacked_w = torch.stack([torch.log(pi_tensor) + log_gaussian1_w, 
                                    torch.log(1 - pi_tensor) + log_gaussian2_w], dim=0)
            stacked_b = torch.stack([torch.log(pi_tensor) + log_gaussian1_b, 
                                    torch.log(1 - pi_tensor) + log_gaussian2_b], dim=0)
                                    
            log_mix_w = torch.logsumexp(stacked_w, dim=0)
            log_mix_b = torch.logsumexp(stacked_b, dim=0)
            
            entropy_w = 0.5 * (torch.log(2 * const_e * torch.pi) + self.weight_logvar).sum()
            entropy_b = 0.5 * (torch.log(2 * const_e * torch.pi) + self.bias_logvar).sum()
            
            kl_weight = -log_mix_w.sum() - entropy_w
            kl_bias = -log_mix_b.sum() - entropy_b
            
        elif self.prior_type == PriorType.LAPLACE:
            device = self.weight_mu.device
            dtype = self.weight_mu.dtype
            b_param = self.prior_params.get('b', 1.0)
            b = torch.tensor(b_param, device=device, dtype=dtype)
            log_2b = torch.log(2 * b)
            
            n_samples = 10
            kl_weight = torch.tensor(0.0, device=device, dtype=dtype)
            kl_bias = torch.tensor(0.0, device=device, dtype=dtype)
            
            for _ in range(n_samples):
                eps_w = torch.randn_like(self.weight_mu)
                eps_b = torch.randn_like(self.bias_mu)
                
                w_sample = self.weight_mu + torch.exp(0.5 * self.weight_logvar) * eps_w
                b_sample = self.bias_mu + torch.exp(0.5 * self.bias_logvar) * eps_b
                
                log_q_w = -0.5 * (self.weight_logvar + (w_sample - self.weight_mu)**2 / weight_var)
                log_q_b = -0.5 * (self.bias_logvar + (b_sample - self.bias_mu)**2 / bias_var)
                
                log_p_w = -torch.abs(w_sample) / b - log_2b
                log_p_b = -torch.abs(b_sample) / b - log_2b
                
                kl_weight += (log_q_w - log_p_w).sum() / n_samples
                kl_bias += (log_q_b - log_p_b).sum() / n_samples
        
        return kl_weight + kl_bias


class BayesLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, 
                 dropout=0, bidirectional=False, prior_type=PriorType.GAUSSIAN, 
                 prior_params={'sigma': 1.0}):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.prior_type = prior_type
        self.prior_params = prior_params
        
        self.num_directions = 2 if bidirectional else 1
        
        # Create parameters for each layer
        self.weight_ih_mu = nn.ParameterList()
        self.weight_ih_logvar = nn.ParameterList()
        self.weight_hh_mu = nn.ParameterList()
        self.weight_hh_logvar = nn.ParameterList()
        
        if bias:
            self.bias_ih_mu = nn.ParameterList()
            self.bias_ih_logvar = nn.ParameterList()
            self.bias_hh_mu = nn.ParameterList()
            self.bias_hh_logvar = nn.ParameterList()
        
        # Initialize parameters for each layer and direction
        for layer in range(num_layers):
            for direction in range(self.num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
                
                # Input-to-hidden weights (4 gates: input, forget, cell, output)
                weight_ih_mu = nn.Parameter(torch.zeros(4 * hidden_size, layer_input_size))
                weight_ih_logvar = nn.Parameter(torch.zeros(4 * hidden_size, layer_input_size))
                
                # Hidden-to-hidden weights (4 gates)
                weight_hh_mu = nn.Parameter(torch.zeros(4 * hidden_size, hidden_size))
                weight_hh_logvar = nn.Parameter(torch.zeros(4 * hidden_size, hidden_size))
                
                # Initialize weights
                nn.init.xavier_uniform_(weight_ih_mu)
                nn.init.orthogonal_(weight_hh_mu)
                nn.init.constant_(weight_ih_logvar, -3.0)
                nn.init.constant_(weight_hh_logvar, -3.0)
                
                self.weight_ih_mu.append(weight_ih_mu)
                self.weight_ih_logvar.append(weight_ih_logvar)
                self.weight_hh_mu.append(weight_hh_mu)
                self.weight_hh_logvar.append(weight_hh_logvar)
                
                if bias:
                    # Biases for input-to-hidden (4 gates)
                    bias_ih_mu = nn.Parameter(torch.zeros(4 * hidden_size))
                    bias_ih_logvar = nn.Parameter(torch.zeros(4 * hidden_size))
                    
                    # Biases for hidden-to-hidden (4 gates)
                    bias_hh_mu = nn.Parameter(torch.zeros(4 * hidden_size))
                    bias_hh_logvar = nn.Parameter(torch.zeros(4 * hidden_size))
                    
                    # Set forget gate bias to 1 (helps with vanishing gradients)
                    bias_ih_mu.data[hidden_size:2*hidden_size].fill_(1.0)
                    nn.init.constant_(bias_ih_logvar, -3.0)
                    nn.init.constant_(bias_hh_logvar, -3.0)
                    
                    self.bias_ih_mu.append(bias_ih_mu)
                    self.bias_ih_logvar.append(bias_ih_logvar)
                    self.bias_hh_mu.append(bias_hh_mu)
                    self.bias_hh_logvar.append(bias_hh_logvar)
    
    def forward(self, x, hx=None, sample=True):
        # Implementation note: For simplicity, we'll use the built-in LSTM
        # and just sample parameters before passing them in
        if self.batch_first:
            batch_size = x.size(0)
            seq_len = x.size(1)
        else:
            batch_size = x.size(1)
            seq_len = x.size(0)
            x = x.transpose(0, 1)  # Convert to batch_first for processing
        
        if hx is None:
            h_zeros = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, 
                                device=x.device)
            c_zeros = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size,
                                device=x.device)
            hx = (h_zeros, c_zeros)
        
        h_out = []
        hidden_states = []
        cell_states = []
        
        h_prev, c_prev = hx
        
        # For each layer
        for layer in range(self.num_layers):
            layer_h_out = []
            h_t = h_prev[layer:layer+self.num_directions]
            c_t = c_prev[layer:layer+self.num_directions]
            
            # For each time step
            for t in range(seq_len):
                if layer == 0:
                    x_t = x[:, t].unsqueeze(1)  # Get current time step
                else:
                    # For upper layers, use the output of the previous layer
                    if t == 0:
                        # For the first time step, we don't have previous output from this layer
                        # So we use the output from the lower layer at this time step
                        x_t = h_out[layer-1][:, t].unsqueeze(1)
                    else:
                        # For other time steps, use output from previous time step
                        x_t = layer_h_out[t-1].unsqueeze(1)
                
                h_next = []
                c_next = []
                
                # For each direction (forward or bidirectional)
                for direction in range(self.num_directions):
                    idx = layer * self.num_directions + direction
                    
                    # Sample weights
                    if sample:
                        eps_ih = torch.randn_like(self.weight_ih_mu[idx])
                        eps_hh = torch.randn_like(self.weight_hh_mu[idx])
                        
                        weight_ih = self.weight_ih_mu[idx] + torch.exp(0.5 * self.weight_ih_logvar[idx]) * eps_ih
                        weight_hh = self.weight_hh_mu[idx] + torch.exp(0.5 * self.weight_hh_logvar[idx]) * eps_hh
                        
                        if self.bias:
                            eps_bih = torch.randn_like(self.bias_ih_mu[idx])
                            eps_bhh = torch.randn_like(self.bias_hh_mu[idx])
                            
                            bias_ih = self.bias_ih_mu[idx] + torch.exp(0.5 * self.bias_ih_logvar[idx]) * eps_bih
                            bias_hh = self.bias_hh_mu[idx] + torch.exp(0.5 * self.bias_hh_logvar[idx]) * eps_bhh
                        else:
                            bias_ih = None
                            bias_hh = None
                    else:
                        # Use mean for deterministic pass
                        weight_ih = self.weight_ih_mu[idx]
                        weight_hh = self.weight_hh_mu[idx]
                        
                        if self.bias:
                            bias_ih = self.bias_ih_mu[idx]
                            bias_hh = self.bias_hh_mu[idx]
                        else:
                            bias_ih = None
                            bias_hh = None
                    
                    # LSTM Cell operation (similar to torch.nn.LSTMCell)
                    h_dir = h_t[direction]
                    c_dir = c_t[direction]
                    
                    gates = F.linear(x_t.squeeze(1), weight_ih, bias_ih) + F.linear(h_dir, weight_hh, bias_hh)
                    
                    # Split into the 4 gates
                    i, f, g, o = gates.chunk(4, 1)
                    
                    # Apply activations
                    i = torch.sigmoid(i)  # input gate
                    f = torch.sigmoid(f)  # forget gate
                    g = torch.tanh(g)     # cell gate
                    o = torch.sigmoid(o)  # output gate
                    
                    # Update cell state and hidden state
                    c_dir = f * c_dir + i * g
                    h_dir = o * torch.tanh(c_dir)
                    
                    h_next.append(h_dir)
                    c_next.append(c_dir)
                
                # Combine directions for this time step
                if self.num_directions == 2:
                    h_t_out = torch.cat(h_next, dim=-1)
                else:
                    h_t_out = h_next[0]
                
                layer_h_out.append(h_t_out)
                h_t = torch.stack(h_next)
                c_t = torch.stack(c_next)
            
            # Update for next layer
            h_out.append(torch.stack(layer_h_out, dim=1))  # [batch, seq_len, hidden]
            hidden_states.append(h_t)
            cell_states.append(c_t)
            
            # Apply dropout between layers
            if layer < self.num_layers - 1 and self.dropout > 0:
                h_out[-1] = F.dropout(h_out[-1], p=self.dropout, training=self.training)
        
        # Stack all layers
        output = h_out[-1]
        h_n = torch.cat(hidden_states, dim=0)
        c_n = torch.cat(cell_states, dim=0)
        
        if not self.batch_first:
            output = output.transpose(0, 1)  # Convert back to seq_len first
        
        return output, (h_n, c_n)
    
    def kl_divergence(self):
        """Compute KL divergence for all weights and biases across all layers and directions."""
        device = self.weight_ih_mu[0].device
        dtype = self.weight_ih_mu[0].dtype
        total_kl = torch.tensor(0.0, device=device, dtype=dtype)
        
        for idx in range(len(self.weight_ih_mu)):
            # Weight KL calculations
            weight_ih_var = torch.exp(self.weight_ih_logvar[idx])
            weight_hh_var = torch.exp(self.weight_hh_logvar[idx])
            
            if self.prior_type == PriorType.GAUSSIAN:
                prior_sigma = self.prior_params.get('sigma', 1.0)
                prior_sigma_tensor = torch.tensor(prior_sigma**2, device=device, dtype=dtype)
                
                kl_weight_ih = 0.5 * (
                    weight_ih_var / prior_sigma_tensor + 
                    (self.weight_ih_mu[idx]**2) / prior_sigma_tensor - 
                    self.weight_ih_logvar[idx] +
                    torch.log(prior_sigma_tensor)
                    - 1
                ).sum()
                
                kl_weight_hh = 0.5 * (
                    weight_hh_var / prior_sigma_tensor + 
                    (self.weight_hh_mu[idx]**2) / prior_sigma_tensor - 
                    self.weight_hh_logvar[idx] +
                    torch.log(prior_sigma_tensor)
                    - 1
                ).sum()
                
                total_kl += kl_weight_ih + kl_weight_hh
                
                # Bias KL calculations if applicable
                if self.bias:
                    bias_ih_var = torch.exp(self.bias_ih_logvar[idx])
                    bias_hh_var = torch.exp(self.bias_hh_logvar[idx])
                    
                    kl_bias_ih = 0.5 * (
                        bias_ih_var / prior_sigma_tensor + 
                        (self.bias_ih_mu[idx]**2) / prior_sigma_tensor - 
                        self.bias_ih_logvar[idx] +
                        torch.log(prior_sigma_tensor)
                        - 1
                    ).sum()
                    
                    kl_bias_hh = 0.5 * (
                        bias_hh_var / prior_sigma_tensor + 
                        (self.bias_hh_mu[idx]**2) / prior_sigma_tensor - 
                        self.bias_hh_logvar[idx] +
                        torch.log(prior_sigma_tensor)
                        - 1
                    ).sum()
                    
                    total_kl += kl_bias_ih + kl_bias_hh
            
            elif self.prior_type == PriorType.SCALED_MIXTURE_GAUSSIAN:
                # Implementation for scaled mixture of Gaussians prior
                sigma1 = self.prior_params.get('sigma1', 1.0)
                sigma2 = self.prior_params.get('sigma2', 0.1)
                pi = self.prior_params.get('pi', 0.5)
                
                # Convert to tensors
                sigma1_sq = torch.tensor(sigma1**2, device=device, dtype=dtype)
                sigma2_sq = torch.tensor(sigma2**2, device=device, dtype=dtype)
                pi_tensor = torch.tensor(pi, device=device, dtype=dtype)
                const_log_2pi = torch.log(torch.tensor(2 * torch.pi, device=device, dtype=dtype))
                const_e = torch.tensor(torch.e, device=device, dtype=dtype)
                
                # Weights (ih)
                log_gaussian1_w_ih = -0.5 * (const_log_2pi + 
                                   torch.log(sigma1_sq) + 
                                   self.weight_ih_mu[idx]**2 / sigma1_sq)
                log_gaussian2_w_ih = -0.5 * (const_log_2pi + 
                                   torch.log(sigma2_sq) + 
                                   self.weight_ih_mu[idx]**2 / sigma2_sq)
                
                stacked_w_ih = torch.stack([torch.log(pi_tensor) + log_gaussian1_w_ih, 
                                        torch.log(1 - pi_tensor) + log_gaussian2_w_ih], dim=0)
                log_mix_w_ih = torch.logsumexp(stacked_w_ih, dim=0)
                entropy_w_ih = 0.5 * (torch.log(2 * const_e * torch.pi) + self.weight_ih_logvar[idx]).sum()
                kl_weight_ih = -log_mix_w_ih.sum() - entropy_w_ih
                
                # Weights (hh)
                log_gaussian1_w_hh = -0.5 * (const_log_2pi + 
                                   torch.log(sigma1_sq) + 
                                   self.weight_hh_mu[idx]**2 / sigma1_sq)
                log_gaussian2_w_hh = -0.5 * (const_log_2pi + 
                                   torch.log(sigma2_sq) + 
                                   self.weight_hh_mu[idx]**2 / sigma2_sq)
                
                stacked_w_hh = torch.stack([torch.log(pi_tensor) + log_gaussian1_w_hh, 
                                        torch.log(1 - pi_tensor) + log_gaussian2_w_hh], dim=0)
                log_mix_w_hh = torch.logsumexp(stacked_w_hh, dim=0)
                entropy_w_hh = 0.5 * (torch.log(2 * const_e * torch.pi) + self.weight_hh_logvar[idx]).sum()
                kl_weight_hh = -log_mix_w_hh.sum() - entropy_w_hh
                
                total_kl += kl_weight_ih + kl_weight_hh
                
                # Biases if applicable
                if self.bias:
                    # Biases (ih)
                    log_gaussian1_b_ih = -0.5 * (const_log_2pi + 
                                       torch.log(sigma1_sq) + 
                                       self.bias_ih_mu[idx]**2 / sigma1_sq)
                    log_gaussian2_b_ih = -0.5 * (const_log_2pi + 
                                       torch.log(sigma2_sq) + 
                                       self.bias_ih_mu[idx]**2 / sigma2_sq)
                    
                    stacked_b_ih = torch.stack([torch.log(pi_tensor) + log_gaussian1_b_ih, 
                                            torch.log(1 - pi_tensor) + log_gaussian2_b_ih], dim=0)
                    log_mix_b_ih = torch.logsumexp(stacked_b_ih, dim=0)
                    entropy_b_ih = 0.5 * (torch.log(2 * const_e * torch.pi) + self.bias_ih_logvar[idx]).sum()
                    kl_bias_ih = -log_mix_b_ih.sum() - entropy_b_ih
                    
                    # Biases (hh)
                    log_gaussian1_b_hh = -0.5 * (const_log_2pi + 
                                       torch.log(sigma1_sq) + 
                                       self.bias_hh_mu[idx]**2 / sigma1_sq)
                    log_gaussian2_b_hh = -0.5 * (const_log_2pi + 
                                       torch.log(sigma2_sq) + 
                                       self.bias_hh_mu[idx]**2 / sigma2_sq)
                    
                    stacked_b_hh = torch.stack([torch.log(pi_tensor) + log_gaussian1_b_hh, 
                                            torch.log(1 - pi_tensor) + log_gaussian2_b_hh], dim=0)
                    log_mix_b_hh = torch.logsumexp(stacked_b_hh, dim=0)
                    entropy_b_hh = 0.5 * (torch.log(2 * const_e * torch.pi) + self.bias_hh_logvar[idx]).sum()
                    kl_bias_hh = -log_mix_b_hh.sum() - entropy_b_hh
                    
                    total_kl += kl_bias_ih + kl_bias_hh
                
            elif self.prior_type == PriorType.LAPLACE:
                # Implementation for Laplace prior
                b_param = self.prior_params.get('b', 1.0)
                b = torch.tensor(b_param, device=device, dtype=dtype)
                log_2b = torch.log(2 * b)
                
                n_samples_mc = 10
                
                # Monte Carlo for weights (ih)
                kl_w_ih_mc = torch.tensor(0.0, device=device, dtype=dtype)
                kl_w_hh_mc = torch.tensor(0.0, device=device, dtype=dtype)
                
                for _ in range(n_samples_mc):
                    # Sample weights
                    eps_w_ih = torch.randn_like(self.weight_ih_mu[idx])
                    eps_w_hh = torch.randn_like(self.weight_hh_mu[idx])
                    
                    w_ih_sample = self.weight_ih_mu[idx] + torch.exp(0.5 * self.weight_ih_logvar[idx]) * eps_w_ih
                    w_hh_sample = self.weight_hh_mu[idx] + torch.exp(0.5 * self.weight_hh_logvar[idx]) * eps_w_hh
                    
                    # Calculate log q(w|θ) for weights
                    log_q_w_ih = -0.5 * (self.weight_ih_logvar[idx] + 
                                (w_ih_sample - self.weight_ih_mu[idx])**2 / weight_ih_var)
                    log_q_w_hh = -0.5 * (self.weight_hh_logvar[idx] + 
                                (w_hh_sample - self.weight_hh_mu[idx])**2 / weight_hh_var)
                    
                    # Calculate log p(w) for weights under Laplace prior
                    log_p_w_ih = -torch.abs(w_ih_sample) / b - log_2b
                    log_p_w_hh = -torch.abs(w_hh_sample) / b - log_2b
                    
                    # Accumulate KL for weights
                    kl_w_ih_mc += (log_q_w_ih - log_p_w_ih).sum()
                    kl_w_hh_mc += (log_q_w_hh - log_p_w_hh).sum()
                
                # Average over samples
                kl_w_ih_mc /= n_samples_mc
                kl_w_hh_mc /= n_samples_mc
                
                total_kl += kl_w_ih_mc + kl_w_hh_mc
                
                # For biases if applicable
                if self.bias:
                    kl_b_ih_mc = torch.tensor(0.0, device=device, dtype=dtype)
                    kl_b_hh_mc = torch.tensor(0.0, device=device, dtype=dtype)
                    
                    bias_ih_var = torch.exp(self.bias_ih_logvar[idx])
                    bias_hh_var = torch.exp(self.bias_hh_logvar[idx])
                    
                    for _ in range(n_samples_mc):
                        # Sample biases
                        eps_b_ih = torch.randn_like(self.bias_ih_mu[idx])
                        eps_b_hh = torch.randn_like(self.bias_hh_mu[idx])
                        
                        b_ih_sample = self.bias_ih_mu[idx] + torch.exp(0.5 * self.bias_ih_logvar[idx]) * eps_b_ih
                        b_hh_sample = self.bias_hh_mu[idx] + torch.exp(0.5 * self.bias_hh_logvar[idx]) * eps_b_hh
                        
                        # Calculate log q(b|θ) for biases
                        log_q_b_ih = -0.5 * (self.bias_ih_logvar[idx] + 
                                    (b_ih_sample - self.bias_ih_mu[idx])**2 / bias_ih_var)
                        log_q_b_hh = -0.5 * (self.bias_hh_logvar[idx] + 
                                    (b_hh_sample - self.bias_hh_mu[idx])**2 / bias_hh_var)
                        
                        # Calculate log p(b) for biases under Laplace prior
                        log_p_b_ih = -torch.abs(b_ih_sample) / b - log_2b
                        log_p_b_hh = -torch.abs(b_hh_sample) / b - log_2b
                        
                        # Accumulate KL for biases
                        kl_b_ih_mc += (log_q_b_ih - log_p_b_ih).sum()
                        kl_b_hh_mc += (log_q_b_hh - log_p_b_hh).sum()
                    
                    # Average over samples
                    kl_b_ih_mc /= n_samples_mc
                    kl_b_hh_mc /= n_samples_mc
                    
                    total_kl += kl_b_ih_mc + kl_b_hh_mc
        
        return total_kl


class BayesConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 prior_type=PriorType.GAUSSIAN, prior_params={'sigma': 1.0}):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.prior_type = prior_type
        self.prior_params = prior_params
        
        # Mean and log-variance for weight posterior
        self.weight_mu = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size))
        self.weight_logvar = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size))
        
        # Mean and log-variance for bias posterior
        self.bias_mu = nn.Parameter(torch.zeros(out_channels))
        self.bias_logvar = nn.Parameter(torch.zeros(out_channels))
        
        # Initialize the parameters
        nn.init.kaiming_normal_(self.weight_mu, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.weight_logvar, -3.0)
        
    def forward(self, x, sample=True):
        if sample:
            # Sample weights using reparameterization trick
            eps_w = torch.randn_like(self.weight_mu)
            eps_b = torch.randn_like(self.bias_mu)
            
            weight = self.weight_mu + torch.exp(0.5 * self.weight_logvar) * eps_w
            bias = self.bias_mu + torch.exp(0.5 * self.bias_logvar) * eps_b
        else:
            # Use mean only for deterministic pass
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.conv1d(x, weight, bias, stride=self.stride, padding=self.padding)
    
    def kl_divergence(self):
        """KL(q(W|theta) || p(W)) for weight and bias."""
        weight_var = torch.exp(self.weight_logvar)
        bias_var = torch.exp(self.bias_logvar)
        
        if self.prior_type == PriorType.GAUSSIAN:
            device = self.weight_mu.device
            dtype = self.weight_mu.dtype
            prior_sigma = self.prior_params.get('sigma', 1.0)
            prior_sigma_tensor = torch.tensor(prior_sigma**2, device=device, dtype=dtype)
            
            kl_weight = 0.5 * (
                weight_var / prior_sigma_tensor + 
                (self.weight_mu**2) / prior_sigma_tensor - 
                self.weight_logvar +
                torch.log(prior_sigma_tensor)
                - 1
            ).sum()
            
            kl_bias = 0.5 * (
                bias_var / prior_sigma_tensor + 
                (self.bias_mu**2) / prior_sigma_tensor - 
                self.bias_logvar +
                torch.log(prior_sigma_tensor)
                - 1
            ).sum()
            
        elif self.prior_type == PriorType.SCALED_MIXTURE_GAUSSIAN:
            device = self.weight_mu.device
            dtype = self.weight_mu.dtype
            
            sigma1 = self.prior_params.get('sigma1', 1.0)
            sigma2 = self.prior_params.get('sigma2', 0.1)
            pi = self.prior_params.get('pi', 0.5)
            
            # Convert to tensors
            sigma1_sq = torch.tensor(sigma1**2, device=device, dtype=dtype)
            sigma2_sq = torch.tensor(sigma2**2, device=device, dtype=dtype)
            pi_tensor = torch.tensor(pi, device=device, dtype=dtype)
            const_log_2pi = torch.log(torch.tensor(2 * torch.pi, device=device, dtype=dtype))
            const_e = torch.tensor(torch.e, device=device, dtype=dtype)
            
            log_gaussian1_w = -0.5 * (const_log_2pi + 
                               torch.log(sigma1_sq) + 
                               self.weight_mu**2 / sigma1_sq)
            log_gaussian2_w = -0.5 * (const_log_2pi + 
                               torch.log(sigma2_sq) + 
                               self.weight_mu**2 / sigma2_sq)
            log_gaussian1_b = -0.5 * (const_log_2pi + 
                               torch.log(sigma1_sq) + 
                               self.bias_mu**2 / sigma1_sq)
            log_gaussian2_b = -0.5 * (const_log_2pi + 
                               torch.log(sigma2_sq) + 
                               self.bias_mu**2 / sigma2_sq)
            
            # Use logsumexp for numerical stability
            stacked_w = torch.stack([torch.log(pi_tensor) + log_gaussian1_w, 
                                    torch.log(1 - pi_tensor) + log_gaussian2_w], dim=0)
            stacked_b = torch.stack([torch.log(pi_tensor) + log_gaussian1_b, 
                                    torch.log(1 - pi_tensor) + log_gaussian2_b], dim=0)
                                    
            log_mix_w = torch.logsumexp(stacked_w, dim=0)
            log_mix_b = torch.logsumexp(stacked_b, dim=0)
            
            entropy_w = 0.5 * (torch.log(2 * const_e * torch.pi) + self.weight_logvar).sum()
            entropy_b = 0.5 * (torch.log(2 * const_e * torch.pi) + self.bias_logvar).sum()
            
            kl_weight = -log_mix_w.sum() - entropy_w
            kl_bias = -log_mix_b.sum() - entropy_b
            
        elif self.prior_type == PriorType.LAPLACE:
            device = self.weight_mu.device
            dtype = self.weight_mu.dtype
            b_param = self.prior_params.get('b', 1.0)
            b = torch.tensor(b_param, device=device, dtype=dtype)
            log_2b = torch.log(2 * b)
            
            n_samples = 10
            kl_weight = torch.tensor(0.0, device=device, dtype=dtype)
            kl_bias = torch.tensor(0.0, device=device, dtype=dtype)
            
            for _ in range(n_samples):
                eps_w = torch.randn_like(self.weight_mu)
                eps_b = torch.randn_like(self.bias_mu)
                
                w_sample = self.weight_mu + torch.exp(0.5 * self.weight_logvar) * eps_w
                b_sample = self.bias_mu + torch.exp(0.5 * self.bias_logvar) * eps_b
                
                log_q_w = -0.5 * (self.weight_logvar + (w_sample - self.weight_mu)**2 / weight_var)
                log_q_b = -0.5 * (self.bias_logvar + (b_sample - self.bias_mu)**2 / bias_var)
                
                log_p_w = -torch.abs(w_sample) / b - log_2b
                log_p_b = -torch.abs(b_sample) / b - log_2b
                
                kl_weight += (log_q_w - log_p_w).sum() / n_samples
                kl_bias += (log_q_b - log_p_b).sum() / n_samples
        
        return kl_weight + kl_bias
