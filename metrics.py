from typing import Mapping

import torch


def pearson_corr(arr1, arr2):
    """The Pearson correlation between two tensors across the last axis.

    Computes the Pearson correlation in the last dimension of `arr1` and `arr2`.
    `arr1` and `arr2` must be the same shape. For example, if they are both
    A x B x L arrays, then the correlation of corresponding L-arrays will be
    computed and returned in an A x B array.

    Parameters
    ----------
    arr1: torch.tensor
        One of the tensor to correlate.

    arr2: torch.tensor
        The other tensor to correlation.

    Returns
    -------
    correlation: torch.tensor
        The correlation for each element, calculated along the last axis.
    """

    mean1 = torch.mean(arr1, axis=-1).unsqueeze(-1)
    mean2 = torch.mean(arr2, axis=-1).unsqueeze(-1)
    dev1, dev2 = arr1 - mean1, arr2 - mean2

    sqdev1, sqdev2 = torch.square(dev1), torch.square(dev2)
    numer = torch.sum(dev1 * dev2, axis=-1)  # Covariance
    var1, var2 = torch.sum(sqdev1, axis=-1), torch.sum(sqdev2, axis=-1)  # Variances
    denom = torch.sqrt(var1 * var2)
   
    # Divide numerator by denominator, but use 0 where the denominator is 0
    correlation = torch.zeros_like(numer)
    correlation[denom != 0] = numer[denom != 0] / denom[denom != 0]
    return correlation


def dlog_odds_from_data(
    targets: Mapping[str, torch.Tensor],
    pseudo_nread:int=10
):
    obs_log_odds = torch.log((targets['n_IP']+pseudo_nread*targets['gc_fraction']
                               )/(targets['n_IN']+pseudo_nread*(1-targets['gc_fraction'])))
    obs_d_log_odds = obs_log_odds-targets['gc_fraction']

    return obs_log_odds, obs_d_log_odds
def rbpnet_metrics(
    outputs: Mapping[str, torch.Tensor], 
    targets: Mapping[str, torch.Tensor]
):
    y_clip, y_ctl = outputs["eCLIP_profile"], outputs["control_profile"]
    y_obs, y_obs_ctl = targets["signal"], targets["control"]
    
    
    # eCLIP counts
    y_total = torch.sum(y_obs, axis=1)
    y_total = y_total.unsqueeze(1)
    probs = torch.nn.functional.softmax(y_clip, dim=1)
    expected_counts = probs * y_total
    clip_profile_pearson = pearson_corr(expected_counts, y_obs)
    
    # Control counts
    y_total = torch.sum(y_obs_ctl, axis=1)
    y_total = y_total.unsqueeze(1)
    probs = torch.nn.functional.softmax(y_ctl, dim=1)
    expected_counts = probs * y_total
    ctl_profile_pearson = pearson_corr(expected_counts, y_obs_ctl)

    # dlog odds
    obs_log_odds, obs_d_log_odds = dlog_odds_from_data(targets)
    d_log_odds_pearson = pearson_corr(obs_d_log_odds, outputs["d_log_odds"])

    # pi pearson
    pi_pearson = pearson_corr(obs_log_odds, outputs["mixing_coefficient"].squeeze())
    
    return {
        "eCLIP_profile_pearson": clip_profile_pearson,
        "control_profile_pearson": ctl_profile_pearson,
        "d_log_odds_profile_pearson": d_log_odds_pearson,
        "pi_pearson": pi_pearson
    } 