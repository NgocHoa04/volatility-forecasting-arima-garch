"""
Volatility proxy validation and sanity checks.

This module provides functions to validate volatility proxy calculations,
detect outliers, and ensure data quality before modeling.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


def validate_proxy_levels(proxies: pd.DataFrame, 
                         proxy_names: list = None,
                         verbose: bool = True) -> Dict[str, Any]:
    """
    Validate volatility proxy statistics and identify issues.
    
    Parameters
    ----------
    proxies : pd.DataFrame
        DataFrame with volatility proxy columns
    proxy_names : list, optional
        Column names to validate (default: all columns)
    verbose : bool
        Print detailed statistics (default: True)
        
    Returns
    -------
    dict
        Validation results with statistics and warnings
    """
    if proxy_names is None:
        proxy_names = proxies.columns.tolist()
    
    results = {
        'valid': True,
        'warnings': [],
        'statistics': {},
        'outliers': {}
    }
    
    if verbose:
        print("\n" + "="*80)
        print("📊 VOLATILITY PROXY VALIDATION")
        print("="*80)
        print("\nDescriptive Statistics:")
        print(proxies[proxy_names].describe().T[['mean', 'std', 'min', 'max']].to_string())
        print()
    
    for col in proxy_names:
        vol_data = proxies[col]
        mean_val = vol_data.mean()
        std_val = vol_data.std()
        min_val = vol_data.min()
        max_val = vol_data.max()
        
        # Store statistics
        results['statistics'][col] = {
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
            'ratio': max_val / min_val if min_val > 0 else np.inf
        }
        
        # Checks
        if verbose:
            print(f"{col.upper()}:")
            print(f"   Mean:    {mean_val:.4%}, Std:     {std_val:.4%}")
            print(f"   Min-Max: {min_val:.4%} - {max_val:.4%}")
        
        # Warning 1: Very low mean
        if mean_val < 0.001:
            warning = f"{col}: Very low average volatility (< 0.1%)"
            results['warnings'].append(warning)
            results['valid'] = False
            if verbose:
                print(f"   ⚠️  {warning}")
        
        # Warning 2: Very high mean
        if mean_val > 0.10:
            warning = f"{col}: Very high average volatility (> 10%)"
            results['warnings'].append(warning)
            results['valid'] = False
            if verbose:
                print(f"   ⚠️  {warning}")
        
        # Warning 3: Extreme spike
        if max_val > 1.0:
            warning = f"{col}: Extreme spike detected ({max_val:.2%}) at {vol_data.idxmax().date()}"
            results['warnings'].append(warning)
            if verbose:
                print(f"   ⚠️  {warning}")
        
        # Warning 4: High dispersion
        if std_val > mean_val * 2:
            warning = f"{col}: High dispersion (std > 2*mean) - suggests outliers"
            results['warnings'].append(warning)
            if verbose:
                print(f"   ⚠️  {warning}")
        
        if verbose:
            if not any(w.startswith(col) for w in results['warnings'][-4:]):
                print(f"   ✓ Normal range")
        
        if verbose:
            print()
    
    return results


def detect_proxy_outliers(proxies: pd.DataFrame,
                         threshold_std: float = 3.0,
                         threshold_pct: float = 0.05,
                         verbose: bool = True) -> Dict[str, Any]:
    """
    Detect outliers in volatility proxies using multiple methods.
    
    Parameters
    ----------
    proxies : pd.DataFrame
        DataFrame with volatility proxy columns
    threshold_std : float
        Z-score threshold for outlier detection (default: 3.0)
    threshold_pct : float
        Percentage threshold (default: 0.05 = 5%)
    verbose : bool
        Print results (default: True)
        
    Returns
    -------
    dict
        Outlier detection results
    """
    results = {
        'total_outliers': 0,
        'by_proxy': {},
        'by_method': {'std_based': 0, 'pct_based': 0}
    }
    
    if verbose:
        print("\n" + "="*80)
        print("🚨 OUTLIER DETECTION")
        print("="*80 + "\n")
    
    for col in proxies.columns:
        vol_data = proxies[col]
        mean_val = vol_data.mean()
        std_val = vol_data.std()
        
        # Method 1: Z-score
        z_scores = np.abs((vol_data - mean_val) / std_val)
        outliers_std = z_scores > threshold_std
        n_outliers_std = outliers_std.sum()
        
        # Method 2: Percentage threshold
        outliers_pct = vol_data > threshold_pct
        n_outliers_pct = outliers_pct.sum()
        
        results['by_proxy'][col] = {
            'std_based': n_outliers_std,
            'pct_based': n_outliers_pct,
            'outlier_dates_std': vol_data[outliers_std].index.tolist(),
            'outlier_dates_pct': vol_data[outliers_pct].index.tolist()
        }
        
        results['total_outliers'] += n_outliers_std + n_outliers_pct
        results['by_method']['std_based'] += n_outliers_std
        results['by_method']['pct_based'] += n_outliers_pct
        
        if verbose:
            print(f"{col.upper()}:")
            if n_outliers_std > 0:
                print(f"   Z-score outliers (> {threshold_std}σ): {n_outliers_std}")
                for date in vol_data[outliers_std].index[:3]:  # Show first 3
                    val = vol_data[date]
                    z_score = (val - mean_val) / std_val
                    print(f"      {date.date()}: {val:.4%} (z={z_score:.2f})")
            
            if n_outliers_pct > 0:
                print(f"   Percentage outliers (> {threshold_pct:.0%}): {n_outliers_pct}")
                for date in vol_data[outliers_pct].index[:3]:  # Show first 3
                    print(f"      {date.date()}: {vol_data[date]:.4%}")
            
            if n_outliers_std == 0 and n_outliers_pct == 0:
                print(f"   ✓ No outliers detected")
            print()
    
    return results


def cap_proxy_outliers(proxies_train: pd.DataFrame,
                      proxies_test: pd.DataFrame,
                      cap_level: float = 0.05,
                      verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Cap extreme volatility proxy values at specified level.
    
    Parameters
    ----------
    proxies_train : pd.DataFrame
        Training set proxies
    proxies_test : pd.DataFrame
        Test set proxies
    cap_level : float
        Maximum allowed value (default: 0.05 = 5%)
    verbose : bool
        Print results (default: True)
        
    Returns
    -------
    tuple
        (proxies_train_capped, proxies_test_capped, capping_report)
    """
    proxies_train_capped = proxies_train.copy()
    proxies_test_capped = proxies_test.copy()
    
    report = {'capped_values': {}, 'original_max': {}}
    
    if verbose:
        print("\n" + "="*80)
        print("✅ APPLYING VOLATILITY PROXY CAPS")
        print("="*80 + "\n")
    
    for col in proxies_train.columns:
        original_max_train = proxies_train_capped[col].max()
        original_max_test = proxies_test_capped[col].max()
        original_max = max(original_max_train, original_max_test)
        
        n_capped_train = (proxies_train_capped[col] > cap_level).sum()
        n_capped_test = (proxies_test_capped[col] > cap_level).sum()
        n_capped = n_capped_train + n_capped_test
        
        # Apply cap
        proxies_train_capped[col] = np.minimum(proxies_train_capped[col], cap_level)
        proxies_test_capped[col] = np.minimum(proxies_test_capped[col], cap_level)
        
        report['original_max'][col] = original_max
        report['capped_values'][col] = n_capped
        
        if verbose:
            print(f"{col.upper()}:")
            print(f"   Original max: {original_max:.4f} ({original_max*100:.2f}%)")
            print(f"   Values capped: {n_capped} (train: {n_capped_train}, test: {n_capped_test})")
            if n_capped > 0:
                print(f"   ✓ Capped to {cap_level*100:.1f}%")
            else:
                print(f"   ✓ No capping needed")
            print()
    
    if verbose:
        print("="*80)
        print("📊 UPDATED PROXY STATISTICS (After Capping):")
        print("="*80 + "\n")
        updated_stats = proxies_train_capped.describe().T[['mean', 'std', 'min', 'max']]
        print(updated_stats.to_string())
        print()
    
    return proxies_train_capped, proxies_test_capped, report


def validate_proxy_correlation(proxies: pd.DataFrame,
                              min_corr: float = 0.80,
                              max_corr: float = 0.99,
                              verbose: bool = True) -> Dict[str, Any]:
    """
    Validate correlation between proxies.
    
    Parameters
    ----------
    proxies : pd.DataFrame
        DataFrame with volatility proxies
    min_corr : float
        Minimum expected correlation (default: 0.80)
    max_corr : float
        Maximum before redundancy concern (default: 0.99)
    verbose : bool
        Print results (default: True)
        
    Returns
    -------
    dict
        Correlation validation results
    """
    corr_matrix = proxies.corr()
    
    results = {
        'correlation_matrix': corr_matrix,
        'issues': [],
        'recommendations': []
    }
    
    if verbose:
        print("\n" + "="*80)
        print("🔗 PROXY CORRELATION ANALYSIS")
        print("="*80 + "\n")
        print(corr_matrix.round(4).to_string())
        print()
    
    # Check all pairs
    for i, col1 in enumerate(proxies.columns):
        for col2 in proxies.columns[i+1:]:
            corr_val = corr_matrix.loc[col1, col2]
            
            if corr_val > max_corr:
                issue = f"{col1} vs {col2}: {corr_val:.4f} (too similar - redundant)"
                results['issues'].append(issue)
                if verbose:
                    print(f"   ⚠️  {issue}")
            elif corr_val > min_corr:
                if verbose:
                    print(f"   ✓ {col1} vs {col2}: {corr_val:.4f} (good - capture similar signal)")
            else:
                if verbose:
                    print(f"   ✓ {col1} vs {col2}: {corr_val:.4f} (diverse perspectives)")
    
    if verbose:
        print()
    
    if results['issues']:
        results['recommendations'].append("Consider removing or combining highly correlated proxies")
    else:
        results['recommendations'].append("Proxy diversity is good")
    
    return results


def generate_proxy_report(proxies_train: pd.DataFrame,
                         proxies_test: pd.DataFrame,
                         cap_level: float = 0.05,
                         output_file: str = None) -> Dict[str, Any]:
    """
    Generate comprehensive proxy validation report.
    
    Parameters
    ----------
    proxies_train : pd.DataFrame
        Training set proxies
    proxies_test : pd.DataFrame
        Test set proxies
    cap_level : float
        Capping level (default: 0.05)
    output_file : str, optional
        Save report to CSV file
        
    Returns
    -------
    dict
        Complete validation report
    """
    report = {
        'train_validation': validate_proxy_levels(proxies_train, verbose=False),
        'train_outliers': detect_proxy_outliers(proxies_train, verbose=False),
        'correlation': validate_proxy_correlation(proxies_train, verbose=False),
        'capping_info': {
            'cap_level': cap_level,
            'data_before_capping': {
                'train': proxies_train.describe().to_dict(),
                'test': proxies_test.describe().to_dict()
            }
        }
    }
    
    # Apply capping
    proxies_train_capped, proxies_test_capped, capping_results = cap_proxy_outliers(
        proxies_train, proxies_test, cap_level=cap_level, verbose=False
    )
    
    report['capping_results'] = capping_results
    report['data_after_capping'] = {
        'train': proxies_train_capped.describe().to_dict(),
        'test': proxies_test_capped.describe().to_dict()
    }
    
    # Save report summary if requested
    if output_file:
        summary_df = pd.DataFrame({
            'Proxy': proxies_train.columns,
            'Mean': [proxies_train_capped[col].mean() for col in proxies_train.columns],
            'Std': [proxies_train_capped[col].std() for col in proxies_train.columns],
            'Min': [proxies_train_capped[col].min() for col in proxies_train.columns],
            'Max': [proxies_train_capped[col].max() for col in proxies_train.columns],
            'Values_Capped': [capping_results['capped_values'].get(col, 0) for col in proxies_train.columns]
        })
        summary_df.to_csv(output_file, index=False)
        logger.info(f"Proxy validation report saved to {output_file}")
    
    return report
