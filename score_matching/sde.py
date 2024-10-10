import copy
import functools
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from ipdb import set_trace
from scipy import integrate

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# from networks.pointnets import PointNetfeat
# from networks.rotation_decoder import ConvOutPerRotHead
# from networks.translation_decoder import TransHead
# from utils.anypose_utils import get_pose_dim
# from utils.data_utils import transform_batch_pts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def marginal_prob_std(t, sigma):
    t = torch.tensor(t)
    return torch.sqrt((sigma ** (2 * t) - 1.) / 2. / np.log(sigma))

def marginal_prob_std_v2(t, sigma, sigma_min=0.01, sigma_max=90):
    t = torch.tensor(t)
    std = sigma_min * (sigma_max / sigma_min) ** t
    return std

def diffusion_coeff(t, sigma):
    return sigma ** t


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.dense(x)[..., None, None]


#----- VE SDE -----
#------------------
def ve_marginal_prob(x, t, sigma_min=0.01, sigma_max=90):
    std = sigma_min * (sigma_max / sigma_min) ** t
    mean = x
    return mean, std

def ve_sde(t, sigma_min=0.01, sigma_max=90):
    sigma = sigma_min * (sigma_max / sigma_min) ** t
    drift_coeff = torch.tensor(0)
    diffusion_coeff = sigma * torch.sqrt(torch.tensor(2 * (np.log(sigma_max) - np.log(sigma_min)), device=t.device))
    return drift_coeff, diffusion_coeff

def ve_prior(shape, sigma_min=0.01, sigma_max=90):
    return torch.randn(*shape) * sigma_max

#----- VP SDE -----
#------------------
def vp_marginal_prob(x, t, beta_0=0.1, beta_1=20):
    log_mean_coeff = -0.25 * t ** 2 * (beta_1 - beta_0) - 0.5 * t * beta_0
    mean = torch.exp(log_mean_coeff) * x
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return mean, std

def vp_sde(t, beta_0=0.1, beta_1=20):
    beta_t = beta_0 + t * (beta_1 - beta_0)
    drift_coeff = -0.5 * beta_t
    diffusion_coeff = torch.sqrt(beta_t)
    return drift_coeff, diffusion_coeff

def vp_prior(shape, beta_0=0.1, beta_1=20):
    return torch.randn(*shape)

#----- sub-VP SDE -----
#----------------------
def subvp_marginal_prob(x, t, beta_0, beta_1):
    log_mean_coeff = -0.25 * t ** 2 * (beta_1 - beta_0) - 0.5 * t * beta_0
    mean = torch.exp(log_mean_coeff) * x
    std = 1 - torch.exp(2. * log_mean_coeff)
    return mean, std

def subvp_sde(t, beta_0, beta_1):
    beta_t = beta_0 + t * (beta_1 - beta_0)
    drift_coeff = -0.5 * beta_t
    discount = 1. - torch.exp(-2 * beta_0 * t - (beta_1 - beta_0) * t ** 2)
    diffusion_coeff = torch.sqrt(beta_t * discount)
    return drift_coeff, diffusion_coeff

def subvp_prior(shape, beta_0=0.1, beta_1=20):
    return torch.randn(*shape)


def init_sde(sde_mode):
    # the SDE-related hyperparameters are copied from https://github.com/yang-song/score_sde_pytorch
    if sde_mode == 've':
        sigma_min = 0.01
        sigma_max = 20
        # sigma_max = 10
        eps = 1e-5
        prior_fn = functools.partial(ve_prior, sigma_min=sigma_min, sigma_max=sigma_max)
        marginal_prob_fn = functools.partial(ve_marginal_prob, sigma_min=sigma_min, sigma_max=sigma_max)
        sde_fn = functools.partial(ve_sde, sigma_min=sigma_min, sigma_max=sigma_max)
    elif sde_mode == 'vp':
        beta_0 = 0.1
        beta_1 = 20
        eps = 1e-3
        prior_fn = functools.partial(vp_prior, beta_0=beta_0, beta_1=beta_1)
        marginal_prob_fn = functools.partial(vp_marginal_prob, beta_0=beta_0, beta_1=beta_1)
        sde_fn = functools.partial(vp_sde, beta_0=beta_0, beta_1=beta_1)
    elif sde_mode == 'subvp':
        beta_0 = 0.1
        beta_1 = 20
        eps = 1e-3
        prior_fn = functools.partial(subvp_prior, beta_0=beta_0, beta_1=beta_1)
        marginal_prob_fn = functools.partial(subvp_marginal_prob, beta_0=beta_0, beta_1=beta_1)
        sde_fn = functools.partial(subvp_sde, beta_0=beta_0, beta_1=beta_1)
    else:
        raise NotImplementedError
    return prior_fn, marginal_prob_fn, sde_fn, eps



class ExponentialMovingAverage:
  """
  Maintains (exponential) moving average of a set of parameters.
  """

  def __init__(self, parameters, decay, use_num_updates=True):
    """
    Args:
      parameters: Iterable of `torch.nn.Parameter`; usually the result of
        `model.parameters()`.
      decay: The exponential decay.
      use_num_updates: Whether to use number of updates when computing
        averages.
    """
    if decay < 0.0 or decay > 1.0:
      raise ValueError('Decay must be between 0 and 1')
    self.decay = decay
    self.num_updates = 0 if use_num_updates else None
    self.shadow_params = [p.clone().detach()
                          for p in parameters if p.requires_grad]
    self.collected_params = []

  def update(self, parameters):
    """
    Update currently maintained parameters.

    Call this every time the parameters are updated, such as the result of
    the `optimizer.step()` call.

    Args:
      parameters: Iterable of `torch.nn.Parameter`; usually the same set of
        parameters used to initialize this object.
    """
    decay = self.decay
    if self.num_updates is not None:
      self.num_updates += 1
      decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
    one_minus_decay = 1.0 - decay
    with torch.no_grad():
      parameters = [p for p in parameters if p.requires_grad]
      for s_param, param in zip(self.shadow_params, parameters):
        s_param.sub_(one_minus_decay * (s_param - param)) # only update the ema-params

  def copy_to(self, parameters):
    """
    Copy current parameters into given collection of parameters.

    Args:
      parameters: Iterable of `torch.nn.Parameter`; the parameters to be
        updated with the stored moving averages.
    """
    parameters = [p for p in parameters if p.requires_grad]
    for s_param, param in zip(self.shadow_params, parameters):
      if param.requires_grad:
        param.data.copy_(s_param.data)

  def store(self, parameters):
    """
    Save the current parameters for restoring later.

    Args:
      parameters: Iterable of `torch.nn.Parameter`; the parameters to be
        temporarily stored.
    """
    self.collected_params = [param.clone() for param in parameters]

  def restore(self, parameters):
    """
    Restore the parameters stored with the `store` method.
    Useful to validate the model with EMA parameters without affecting the
    original optimization process. Store the parameters before the
    `copy_to` method. After validation (or model saving), use this to
    restore the former parameters.

    Args:
      parameters: Iterable of `torch.nn.Parameter`; the parameters to be
        updated with the stored parameters.
    """
    for c_param, param in zip(self.collected_params, parameters):
      param.data.copy_(c_param.data)

  def state_dict(self):
    return dict(decay=self.decay, num_updates=self.num_updates,
                shadow_params=self.shadow_params)

  def load_state_dict(self, state_dict):
    self.decay = state_dict['decay']
    self.num_updates = state_dict['num_updates']
    self.shadow_params = state_dict['shadow_params']
