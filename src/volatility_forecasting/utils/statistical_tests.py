"""
Statistical testing utilities for forecast comparison.

Implements:
- Holm correction for multiple testing
- Model Confidence Set (MCS)
- Diebold-Mariano test
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
from ..logger import logger


def holm_correction(pvalues: np.ndarray, alpha: float = 0.05) -> Dict[int, bool]:
    """
    Holm-Bonferroni correction for multiple testing.
    
    Sequentially tests hypotheses in order of increasing p-values,
    adjusting significance level down by 1/(m-i+1).
    
    Parameters
    ----------
    pvalues : np.ndarray
        Array of p-values
    alpha : float
        Initial significance level (default 0.05)
        
    Returns
    -------
    dict
        Mapping of test index to rejection decision (True = reject H0)
        
    References
    ----------
    Holm, S. (1979). A simple sequentially rejective multiple test procedure.
    Scandinavian Journal of Statistics, 6(2), 65-70.
    """
    m = len(pvalues)
    
    # Sort p-values and get original indices
    sorted_indices = np.argsort(pvalues)
    sorted_pvals = pvalues[sorted_indices]
    
    # Initialize decisions (all False)
    decisions_sorted = np.zeros(m, dtype=bool)
    
    # Sequential rejection
    for i in range(m):
        threshold = alpha / (m - i)
        if sorted_pvals[i] <= threshold:
            decisions_sorted[i:] = True  # Reject this and all subsequent
            break
    
    # Map back to original indices
    decisions = np.zeros(m, dtype=bool)
    decisions[sorted_indices] = decisions_sorted
    
    result = {i: bool(decisions[i]) for i in range(m)}
    
    logger.info(f"Holm correction applied: {sum(decisions)}/{m} rejections at α={alpha}")
    
    return result


def model_confidence_set(loss_matrix: pd.DataFrame, 
                         alpha: float = 0.10,
                         B: int = 1000) -> Dict:
    """
    Model Confidence Set (MCS) analysis.
    
    Identifies the set of "superior" models that statistically equal
    the best model, using the superior predictive ability (SPA) test.
    
    Parameters
    ----------
    loss_matrix : pd.DataFrame
        Shape (T, m): T time periods, m models
        Columns are model names, values are loss values (MSE, MAE, QLIKE, etc.)
    alpha : float
        Confidence level (default 0.10 = 90% MCS)
    B : int
        Bootstrap samples (default 1000)
        
    Returns
    -------
    dict
        Keys: 'mcs_models' (list of models in MCS),
              'elimination_order' (list of models eliminated),
              'spa_stat' (SPA test statistic),
              'correlation' (correlation matrix of losses)
        
    References
    ----------
    Hansen, P. R., Lunde, A., & Nason, J. M. (2011).
    The Model Confidence Set. Econometric Reviews, 30(2), 160-201.
    """
    
    # Ensure DataFrame
    if not isinstance(loss_matrix, pd.DataFrame):
        loss_matrix = pd.DataFrame(loss_matrix)
    
    T, m = loss_matrix.shape
    model_names = loss_matrix.columns.tolist()
    
    # Relative losses: d_ij = loss_i - loss_j (best model)
    best_loss = loss_matrix.min(axis=1)
    relative_losses = loss_matrix.sub(best_loss, axis=0)
    
    # Mean relative loss for each model
    mean_losses = relative_losses.mean(axis=0)
    
    # Variance of relative losses
    var_losses = relative_losses.var(axis=0, ddof=1)
    
    # Superior Predictive Ability statistic
    # SPA = max_i sqrt(T) * mean_loss_i / std_loss_i
    std_losses = np.sqrt(var_losses)
    spa_stats = (np.sqrt(T) * mean_losses / np.maximum(std_losses, 1e-8)).values
    
    # Bootstrap p-values
    mcs_models = set(model_names)
    elimination_order = []
    
    for _ in range(m):
        if len(mcs_models) == 1:
            break
        
        # Current best model among remaining
        current_losses = loss_matrix[list(mcs_models)]
        best_current = current_losses.min(axis=1)
        rel_losses_current = current_losses.sub(best_current, axis=0)
        
        # Bootstrap test
        mean_rel = rel_losses_current.mean(axis=0)
        std_rel = rel_losses_current.std(axis=0, ddof=1)
        
        # Relative t-stats
        t_stats = (np.sqrt(T) * mean_rel / np.maximum(std_rel, 1e-8)).values
        max_t_idx = np.argmax(t_stats)
        worst_model = list(mcs_models)[max_t_idx]
        
        # Bootstrap p-value (simplified)
        bootstrap_stats = []
        for _ in range(B):
            boot_sample = rel_losses_current.sample(n=T, replace=True)
            boot_mean = boot_sample.mean(axis=0)
            boot_std = boot_sample.std(axis=0, ddof=1)
            boot_t = np.sqrt(T) * boot_mean / np.maximum(boot_std, 1e-8)
            bootstrap_stats.append(boot_t.max())
        
        p_value = np.mean(np.array(bootstrap_stats) >= t_stats.max())
        
        if p_value < alpha:
            # Remove worst model
            mcs_models.discard(worst_model)
            elimination_order.append((worst_model, p_value))
        else:
            break
    
    result = {
        'mcs_models': list(mcs_models),
        'elimination_order': elimination_order,
        'spa_stat': spa_stats,
        'mean_losses': mean_losses.to_dict(),
        'correlation': loss_matrix.corr(),
        'alpha': alpha,
    }
    
    logger.info(f"MCS (alpha={alpha}): {len(mcs_models)} models in confidence set")
    logger.info(f"MCS Models: {', '.join(mcs_models)}")
    
    return result


def diebold_mariano_test(e1: np.ndarray, e2: np.ndarray, 
                        h: int = 1, loss_type: str = 'mse') -> Dict:
    """
    Diebold-Mariano test for forecast comparison.
    
    Tests H0: models produce equal loss (forecast accuracy).
    
    Parameters
    ----------
    e1 : np.ndarray
        Forecast errors from model 1
    e2 : np.ndarray
        Forecast errors from model 2
    h : int
        Forecast horizon (for HAC correction)
    loss_type : str
        'mse', 'mae', or 'qlike'
        
    Returns
    -------
    dict
        Keys: 'dm_stat' (test statistic),
              'p_value' (two-sided),
              'reject' (bool, reject H0 at 5% level)
    
    References
    ----------
    Diebold, F. X., & Mariano, R. S. (1995).
    Comparing predictive accuracy. Journal of Business & Economic Statistics, 13(3), 253-263.
    """
    
    e1 = np.asarray(e1).flatten()
    e2 = np.asarray(e2).flatten()
    
    if loss_type == 'mse':
        d = e1**2 - e2**2
    elif loss_type == 'mae':
        d = np.abs(e1) - np.abs(e2)
    elif loss_type == 'qlike':
        # QLIKE: log(sigma_hat^2) + RV/sigma_hat^2
        d = e1 - e2
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    
    d_mean = d.mean()
    d_var = d.var(ddof=1)
    
    # HAC covariance (Newey-West style)
    if h > 1:
        gamma = np.zeros(h)
        for j in range(h):
            gamma[j] = np.mean(d * np.roll(d, j))
        long_run_var = gamma[0] + 2 * np.sum(gamma[1:])
    else:
        long_run_var = d_var
    
    # DM test statistic
    dm_stat = d_mean / np.sqrt(long_run_var / len(e1))
    
    # Two-sided p-value
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    
    result = {
        'dm_stat': float(dm_stat),
        'p_value': float(p_value),
        'reject': p_value < 0.05,
        'mean_loss_diff': float(d_mean),
        'variance': float(d_var),
        'loss_type': loss_type,
    }
    
    return result


def compare_forecasts(forecasts: Dict[str, np.ndarray],
                     actual: np.ndarray,
                     loss_types: List[str] = None) -> pd.DataFrame:
    """
    Compare multiple forecasts using DM test and MCS.
    
    Parameters
    ----------
    forecasts : dict
        Mapping of model name to forecast array
    actual : np.ndarray
        Actual values
    loss_types : list
        Loss functions to use (default: ['mse', 'mae'])
        
    Returns
    -------
    pd.DataFrame
        Comparison matrix with ranks and test results
    """
    
    if loss_types is None:
        loss_types = ['mse', 'mae']
    
    results = []
    
    for model_name, forecast in forecasts.items():
        errors = actual - forecast
        
        row = {'Model': model_name}
        
        for loss_type in loss_types:
            if loss_type == 'mse':
                loss = np.mean(errors**2)
            elif loss_type == 'mae':
                loss = np.mean(np.abs(errors))
            elif loss_type == 'qlike':
                loss = np.mean(np.log(forecast**2) + actual / np.maximum(forecast**2, 1e-8))
            else:
                continue
            
            row[f'{loss_type}'] = loss
        
        results.append(row)
    
    df_results = pd.DataFrame(results)
    
    # Add ranks
    for loss_type in loss_types:
        df_results[f'{loss_type}_rank'] = df_results[loss_type].rank()
    
    return df_results
