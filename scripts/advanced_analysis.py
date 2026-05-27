# -*- coding: utf-8 -*-
"""
Advanced Volatility Forecasting Analysis.

Demonstrates:
1. HAR-RV with improved volatility proxies (Parkinson, Garman-Klass, Yang-Zhang)
2. Holm correction for multiple testing
3. Model Confidence Set (MCS) analysis
4. Proxy sensitivity robustness check
"""

import sys
import io

# Set UTF-8 encoding for output on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import sys
from pathlib import Path
import yfinance as yf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.volatility_forecasting.models.har_model import HARModel
from src.volatility_forecasting.utils.volatility_proxies import compute_multiple_proxies
from src.volatility_forecasting.utils.statistical_tests import (
    holm_correction, model_confidence_set, diebold_mariano_test,
    compare_forecasts
)
from src.volatility_forecasting.analysis.proxy_validation import (
    validate_proxy_levels, detect_proxy_outliers, cap_proxy_outliers,
    validate_proxy_correlation, generate_proxy_report
)
from src.volatility_forecasting.logger import logger

# Setup
os.makedirs('results', exist_ok=True)
os.makedirs('report/figures', exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 120,
    'figure.facecolor': 'white',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.size': 10,
})

print("=" * 80)
print("  ADVANCED VOLATILITY FORECASTING ANALYSIS")
print("  1. Multiple Proxies (Parkinson, Garman-Klass, Yang-Zhang)")
print("  2. Holm Correction + Model Confidence Set")
print("  3. Proxy Robustness Check")
print("=" * 80)
print()

# ──────────────────────────────────────────────────────────────────────────
# SECTION 1: DOWNLOAD OHLC DATA FROM YAHOO FINANCE
# ──────────────────────────────────────────────────────────────────────────

print("SECTION 1: Data Collection")
print("-" * 80)
print("Loading price data...")

ticker = '^GSPC'  # S&P 500

# Load from existing CSV file
csv_path = 'dataset/price_data.csv'
if os.path.exists(csv_path):
    print(f"Loading from {csv_path}...")
    df_price = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    data = df_price.copy()
else:
    print(f"ERROR: {csv_path} not found!")
    sys.exit(1)

# Create synthetic OHLC data if not available
# (for demo purposes - use daily returns to estimate High/Low)
if 'Open' not in data.columns or 'High' not in data.columns or 'Low' not in data.columns:
    print("Creating synthetic OHLC data from Close price...")
    data['Open'] = data['Close'].shift(1)  # Open = previous Close
    
    # Estimate intraday volatility to create High/Low
    daily_vol = data['Log_Return'].rolling(20).std().fillna(data['Log_Return'].std())
    
    # Random High/Low within intraday range
    np.random.seed(42)
    high_low_ratio = np.random.uniform(1.002, 1.010, len(data))  # 0.2%-1% intraday range
    data['High'] = data['Close'] * high_low_ratio
    data['Low'] = data['Close'] / high_low_ratio
    
    # Fix any missing values
    data['Open'].fillna(method='bfill', inplace=True)

# Rename 'Close' to 'Adj Close' if needed for compatibility with proxy calculations
if 'Adj Close' not in data.columns and 'Close' in data.columns:
    data['Adj Close'] = data['Close']

print(f"Loaded {len(data)} observations for {ticker}")
print(f"Period: {data.index[0].date()} to {data.index[-1].date()}")
print(f"Columns: {list(data.columns)}")
print()

# Ensure log returns are available
if 'Log_Return' not in data.columns:
    data['Log_Return'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1)) * 100

data = data.dropna()

print(f"Log Return statistics:")
print(f"  Mean: {data['Log_Return'].mean():.4f}%")
print(f"  Std:  {data['Log_Return'].std():.4f}%")
print(f"  Min:  {data['Log_Return'].min():.4f}%")
print(f"  Max:  {data['Log_Return'].max():.4f}%")
print()

# Train/test split
split_idx = int(0.80 * len(data))
train_data = data.iloc[:split_idx]
test_data = data.iloc[split_idx:]

print(f"Train: {len(train_data)} obs ({train_data.index[0].date()} to {train_data.index[-1].date()})")
print(f"Test:  {len(test_data)} obs ({test_data.index[0].date()} to {test_data.index[-1].date()})")
print()

# ──────────────────────────────────────────────────────────────────────────
# SECTION 2: COMPUTE VOLATILITY PROXIES
# ──────────────────────────────────────────────────────────────────────────

print("SECTION 2: Volatility Proxy Estimation")
print("-" * 80)

print("Computing three volatility proxies...")
proxies_train = compute_multiple_proxies(train_data[['Open', 'High', 'Low', 'Adj Close']])
proxies_test = compute_multiple_proxies(test_data[['Open', 'High', 'Low', 'Adj Close']])

print(f"Proxy statistics (Training set):")
print(proxies_train.describe().to_string())
print()

# ─── VOLATILITY PROXY VALIDATION ────────────────────────────────────────
print("\n" + "="*80)
print("VOLATILITY PROXY VALIDATION & QUALITY CHECKS")
print("="*80)

# 1. Validate proxy levels
print("\n[1] PROXY LEVEL VALIDATION:")
validation_results = validate_proxy_levels(proxies_train, verbose=True)

# 2. Detect outliers
print("\n[2] OUTLIER DETECTION:")
outlier_results = detect_proxy_outliers(proxies_train, verbose=True)

# 3. Check correlation
print("\n[3] PROXY CORRELATION ANALYSIS:")
corr_results = validate_proxy_correlation(proxies_train, verbose=True)

# 4. Cap extreme values
print("\n4️⃣  APPLYING CAPPING TO OUTLIERS:")
proxies_train, proxies_test, capping_report = cap_proxy_outliers(
    proxies_train, proxies_test, cap_level=0.05, verbose=True
)

# Save validation report
validation_report = generate_proxy_report(
    proxies_train, proxies_test, cap_level=0.05,
    output_file='results/proxy_validation_report.csv'
)
print("\n✓ Proxy validation report saved: results/proxy_validation_report.csv")
print()

# Plot proxies comparison
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(proxies_train.index, proxies_train['parkinson'], label='Parkinson', alpha=0.7, linewidth=1)
ax.plot(proxies_train.index, proxies_train['garman_klass'], label='Garman-Klass', alpha=0.7, linewidth=1)
ax.plot(proxies_train.index, proxies_train['yang_zhang'], label='Yang-Zhang', alpha=0.7, linewidth=1)
ax.set_title('Volatility Proxy Comparison (Training Set - After Capping)', fontweight='bold', fontsize=12)
ax.set_ylabel('Volatility (%)')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('report/figures/fig_proxies_comparison.png', dpi=120, bbox_inches='tight')
plt.close()
print("Saved: fig_proxies_comparison.png")
print()

# ──────────────────────────────────────────────────────────────────────────
# SECTION 3: FIT HAR-RV WITH DIFFERENT PROXIES
# ──────────────────────────────────────────────────────────────────────────

print("SECTION 3: HAR-RV Model Fitting with Multiple Proxies")
print("-" * 80)

proxy_types = ['abs', 'parkinson', 'garman_klass', 'yang_zhang']
har_models = {}
har_forecasts = {}

print("\nFitting HAR models...")
for proxy_type in proxy_types:
    print(f"\n  {proxy_type.upper()}:")
    
    # Create model
    har = HARModel(proxy_type=proxy_type)
    
    # Compute RV for this proxy
    if proxy_type == 'abs':
        rv_train = train_data['Log_Return'].abs()
        rv_test = test_data['Log_Return'].abs()
    else:
        # Use exact column names from proxies DataFrame
        rv_train = proxies_train[proxy_type]
        rv_test = proxies_test[proxy_type]
    
    # Fit model
    try:
        har.fit(rv_train)
        r2 = har.get_r_squared()
        print(f"    R-squared: {r2:.4f}")
        
        har_models[proxy_type] = (har, rv_train, rv_test)
        
        # Generate rolling forecast
        print(f"    Generating rolling forecasts...")
        forecasts = []
        for i in range(len(test_data)):
            rv_window = pd.concat([rv_train, rv_test.iloc[:i+1]])
            try:
                fc = har.forecast(rv_window)
                forecasts.append(max(fc, 1e-8))
            except:
                forecasts.append(np.nan)
        
        har_forecasts[proxy_type] = pd.Series(forecasts, index=test_data.index)
        print(f"    Mean forecast: {np.nanmean(forecasts):.4f}%")
        
    except Exception as e:
        print(f"    ERROR: {str(e)[:60]}")
        continue

print()

# ──────────────────────────────────────────────────────────────────────────
# SECTION 4: FORECAST ACCURACY COMPARISON
# ──────────────────────────────────────────────────────────────────────────

print("SECTION 4: Forecast Accuracy Comparison")
print("-" * 80)

actual_rv = test_data['Log_Return'].abs()

# Compute loss metrics
loss_results = []
for proxy_type, forecast in har_forecasts.items():
    forecast_aligned = forecast.reindex(actual_rv.index)
    actual_aligned = actual_rv.reindex(forecast_aligned.index)
    
    errors = actual_aligned - forecast_aligned
    mse = np.nanmean(errors ** 2)
    mae = np.nanmean(np.abs(errors))
    qlike = np.nanmean(np.log(forecast_aligned ** 2) + actual_aligned / np.maximum(forecast_aligned ** 2, 1e-8))
    
    loss_results.append({
        'Proxy': proxy_type.upper(),
        'MSE': mse,
        'MAE': mae,
        'QLIKE': qlike,
    })

accuracy_df = pd.DataFrame(loss_results).sort_values('MSE')
print("\nAccuracy Metrics (Lower = Better):")
print(accuracy_df.to_string(index=False))
print()

# ──────────────────────────────────────────────────────────────────────────
# SECTION 5: DIEBOLD-MARIANO TEST WITH HOLM CORRECTION
# ──────────────────────────────────────────────────────────────────────────

print("SECTION 5: Diebold-Mariano Test with Holm Correction")
print("-" * 80)

# Use best model (ABS) as benchmark
benchmark_proxy = 'abs'
benchmark_forecast = har_forecasts[benchmark_proxy].values

pvalues = []
models_dm = []

print(f"\nComparing against benchmark: {benchmark_proxy.upper()}")
print(f"\n{'Proxy':<20} {'DM_MAE':<10} {'p-value':<10} {'Sig':<5}")
print("-" * 50)

for proxy_type, forecast in har_forecasts.items():
    if proxy_type == benchmark_proxy:
        continue
    
    forecast_values = forecast.values
    dm_result = diebold_mariano_test(
        actual_rv.values - benchmark_forecast,
        actual_rv.values - forecast_values,
        h=1,
        loss_type='mae'
    )
    
    pvalues.append(dm_result['p_value'])
    models_dm.append(proxy_type)
    
    sig = '***' if dm_result['p_value'] < 0.01 else ('**' if dm_result['p_value'] < 0.05 else '')
    print(f"{proxy_type:<20} {dm_result['dm_stat']:<10.4f} {dm_result['p_value']:<10.4f} {sig:<5}")

# Holm correction
print(f"\nHolm Correction (α = 0.05):")
print(f"{'Proxy':<20} {'Raw p-value':<15} {'Holm Threshold':<15} {'Reject H0':<10}")
print("-" * 60)

holm_results = holm_correction(np.array(pvalues), alpha=0.05)

for i, proxy_type in enumerate(models_dm):
    m = len(pvalues)
    threshold = 0.05 / (m - i)
    reject = holm_results[i]
    print(f"{proxy_type:<20} {pvalues[i]:<15.4f} {threshold:<15.4f} {str(reject):<10}")

print()

# ──────────────────────────────────────────────────────────────────────────
# SECTION 6: MODEL CONFIDENCE SET (MCS)
# ──────────────────────────────────────────────────────────────────────────

print("SECTION 6: Model Confidence Set Analysis")
print("-" * 80)

# Create loss matrix (MSE for all models)
loss_matrix = pd.DataFrame(index=test_data.index)
for proxy_type, forecast in har_forecasts.items():
    errors = (actual_rv - forecast) ** 2
    loss_matrix[proxy_type.upper()] = errors

# Run MCS
mcs_result = model_confidence_set(loss_matrix, alpha=0.10, B=1000)

print(f"\nMCS (90% confidence level):")
print(f"Models in MCS: {', '.join(mcs_result['mcs_models'])}")
print(f"\nElimination order:")
for model, p_val in mcs_result['elimination_order']:
    print(f"  {model}: p-value = {p_val:.4f}")

print(f"\nMean Loss (MSE):")
for model, loss in sorted(mcs_result['mean_losses'].items(), key=lambda x: x[1]):
    marker = " ✓" if model in mcs_result['mcs_models'] else ""
    print(f"  {model:<20}: {loss:.6f}{marker}")

print()

# ──────────────────────────────────────────────────────────────────────────
# SECTION 7: PROXY SENSITIVITY ROBUSTNESS CHECK
# ──────────────────────────────────────────────────────────────────────────

print("SECTION 7: Proxy Sensitivity & Robustness Analysis")
print("-" * 80)

# Correlation between proxies
print("\nProxy Correlations (Training Set):")
corr_matrix = proxies_train.corr()
print(corr_matrix.round(4).to_string())

# Ranking stability across subsamples
print("\nRanking Stability Across Market Regimes:")
print("-" * 60)

subsamples = {
    'Full Test': test_data,
    'First Half': test_data.iloc[:len(test_data)//2],
    'Second Half': test_data.iloc[len(test_data)//2:],
    'High Vol': test_data[test_data['Log_Return'].abs() > test_data['Log_Return'].abs().quantile(0.75)],
    'Low Vol': test_data[test_data['Log_Return'].abs() < test_data['Log_Return'].abs().quantile(0.25)],
}

ranking_stability = []

for subsample_name, subsample in subsamples.items():
    print(f"\n  {subsample_name} (n={len(subsample)}):")
    
    # Compute MSE for each proxy
    mse_scores = {}
    for proxy_type, forecast in har_forecasts.items():
        fc_sub = forecast.reindex(subsample.index)
        actual_sub = subsample['Log_Return'].abs()
        mse = np.nanmean((actual_sub - fc_sub) ** 2)
        mse_scores[proxy_type] = mse
    
    # Rank
    ranked = sorted(mse_scores.items(), key=lambda x: x[1])
    for rank, (proxy, mse) in enumerate(ranked, 1):
        print(f"    {rank}. {proxy.upper():<20} MSE={mse:.6f}")
    
    ranking_stability.append({
        'Subsample': subsample_name,
        **{f"{p.upper()}_Rank": r+1 for r, (p, _) in enumerate(ranked)}
    })

ranking_df = pd.DataFrame(ranking_stability)
print("\n" + ranking_df.to_string(index=False))
print()

# Create robustness heatmap
fig, ax = plt.subplots(figsize=(10, 6))

ranking_matrix = ranking_df.set_index('Subsample').iloc[:, :].values
im = ax.imshow(ranking_matrix, cmap='RdYlGn_r', vmin=1, vmax=4)

ax.set_xticks(range(len(ranking_df.columns)-1))
ax.set_xticklabels([col.replace('_Rank', '').upper() for col in ranking_df.columns[1:]])
ax.set_yticks(range(len(subsamples)))
ax.set_yticklabels(list(subsamples.keys()))

# Add text annotations
for i in range(len(subsamples)):
    for j in range(len(ranking_df.columns)-1):
        text = ax.text(j, i, int(ranking_matrix[i, j]),
                      ha="center", va="center", color="black", fontweight='bold')

ax.set_title('Model Rankings Across Market Regimes\n(1 = Best, 4 = Worst)', 
             fontweight='bold', fontsize=12)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Rank', rotation=270, labelpad=20)

plt.tight_layout()
plt.savefig('report/figures/fig_robustness_heatmap.png', dpi=120, bbox_inches='tight')
plt.close()
print("Saved: fig_robustness_heatmap.png")
print()

# ──────────────────────────────────────────────────────────────────────────
# SECTION 8: SUMMARY & RECOMMENDATIONS
# ──────────────────────────────────────────────────────────────────────────

print("=" * 80)
print("SUMMARY & KEY FINDINGS")
print("=" * 80)
print()

best_model = accuracy_df.iloc[0]
print(f"1. BEST PROXY: {best_model['Proxy']}")
print(f"   - MSE: {best_model['MSE']:.6f}")
print(f"   - MAE: {best_model['MAE']:.6f}")
print()

print(f"2. STATISTICAL SIGNIFICANCE:")
print(f"   - DM tests with Holm correction applied")
print(f"   - Models in 90% MCS: {', '.join(mcs_result['mcs_models'])}")
print()

print(f"3. ROBUSTNESS ASSESSMENT:")
print(f"   - Proxy correlations: {corr_matrix.iloc[0,1]:.4f} (Parkinson vs Garman-Klass)")
print(f"   - Ranking stability varies across market regimes")
print(f"   - Recommendation: Use ensemble of top proxies for robustness")
print()

print("=" * 80)
print("Analysis Complete!")
print("=" * 80)

# Save results
accuracy_df.to_csv('results/proxy_accuracy_comparison.csv', index=False)
ranking_df.to_csv('results/proxy_ranking_by_regime.csv', index=False)

print("\nResults saved:")
print("  results/proxy_accuracy_comparison.csv")
print("  results/proxy_ranking_by_regime.csv")
print("  report/figures/fig_proxies_comparison.png")
print("  report/figures/fig_robustness_heatmap.png")
