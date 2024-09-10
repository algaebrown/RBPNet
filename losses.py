from typing import Mapping


import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

def MNLLLoss(logps, true_counts):
    """A loss function based on the multinomial negative log-likelihood.

    This loss function takes in a tensor of normalized log probabilities such
    that the sum of each row is equal to 1 (e.g. from a log softmax) and
    an equal sized tensor of true counts and returns the probability of
    observing the true counts given the predicted probabilities under a
    multinomial distribution. Can accept tensors with 2 or more dimensions
    and averages over all except for the last axis, which is the number
    of categories.

    Adapted from Alex Tseng.
    
    Parameters
    ----------
    logps: torch.tensor, shape=(n, ..., L)
        A tensor with `n` examples and `L` possible categories. 

    true_counts: torch.tensor, shape=(n, ..., L)
        A tensor with `n` examples and `L` possible categories.

    Returns
    -------
    loss: float
        The multinomial log likelihood loss of the true counts given the
        predicted probabilities, averaged over all examples and all other
        dimensions.
    """

    log_fact_sum = torch.lgamma(torch.sum(true_counts, dim=-1) + 1) # Computes the natural logarithm of the absolute value of the gamma function on input.
    log_prod_fact = torch.sum(torch.lgamma(true_counts + 1), dim=-1)
    log_prod_exp = torch.sum(true_counts * logps, dim=-1)
    return -log_fact_sum + log_prod_fact - log_prod_exp

def ln_sum_of_factorial_stirling_approx(n):
    # https://en.wikipedia.org/wiki/Stirling%27s_approximation
    values = n*torch.log(n)-n
    # when n=0, value is nan. but 0!=1, log(1) = 0, therefore replace with 0
    return torch.nan_to_num(values, nan = 0)
def sum_of_logs(start, end):
    ''' log(start)+log(start+1)...log(end) '''
    
    return torch.log(torch.arange(start, end+1)).sum()
def log_N_choose_x_approx(N,x):    
    return ln_sum_of_factorial_stirling_approx(N)-ln_sum_of_factorial_stirling_approx(N-x)-ln_sum_of_factorial_stirling_approx(x)
def log_N_choose_x(N,x):
    return sum_of_logs(x+1, N)-sum_of_logs(1, N-x)
def BinomNLLLoss(p, ip_read, total_read,pseudo = 1e-4):
    ''' pseudo is added to prevent inf/-inf when p is close to 0 and 1 '''
    ''' https://discuss.pytorch.org/t/how-to-solve-the-loss-become-nan-because-of-using-torch-log/54499 '''
    L=log_N_choose_x_approx(total_read, ip_read)+total_read*torch.log(p+pseudo)+(total_read-ip_read)*torch.log(1-p+pseudo)

    return -L
    

def rbpnet_loss(
    outputs: Mapping[str, torch.Tensor], 
    targets: Mapping[str, torch.Tensor],
    l: int =1.0,
    w: int =30.0
    ):
    y_clip, y_ctl = F.log_softmax(outputs["eCLIP_profile"], dim=-1), F.log_softmax(outputs["control_profile"], dim=-1)
    y_obs, y_obs_ctl = targets["signal"], targets["control"]
    clip_loss = MNLLLoss(y_clip, y_obs)
    ctl_loss = MNLLLoss(y_ctl, y_obs_ctl)

    # dlog odds binom NLL shit
    d_log_odds = outputs["d_log_odds"]
    basal_gc_odds = torch.log(targets['gc_fraction']/(1-targets['gc_fraction']))
    p = torch.sigmoid(basal_gc_odds+d_log_odds)
    ip_read = targets['n_IP']
    total_read = targets['n_IN']+targets['n_IP']
    dlog_loss = BinomNLLLoss(p, ip_read,total_read)

    # pi binom NLL
    p_pi = torch.sigmoid(outputs['mixing_coefficient'])
    pi_nll_loss = BinomNLLLoss(p_pi.squeeze(), ip_read,total_read)
    
    loss = clip_loss + l*ctl_loss + w*dlog_loss + w* pi_nll_loss
    if not torch.all(torch.isfinite(loss)):
        warnings.warn(f'Non-finite loss: {clip_loss=}; {ctl_loss=}; {dlog_loss=} {pi_nll_loss=}')
        print(outputs)
    return {
        "loss": loss, 
        "eCLIP_loss": clip_loss, 
        "control_loss": ctl_loss,
        "binom_loss": w*dlog_loss
    }
