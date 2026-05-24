"""
Quick demo of volatility forecasting analysis.
Uses synthetic data to showcase the complete workflow.
"""

import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from arch import arch_model
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from scipy import stats
import os

os.makedirs('report/figures', exist_ok=True)
os.makedirs('results', exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 120,
    'figure.facecolor': 'white',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

print("="*70)
print(" VOLATILITY FORECASTING ANALYSIS — DEMO")
print("="*70)
print()

# Generate synthetic return data with GARCH(1,1) structure
print("Generating synthetic S&P 500-like return data...")
np.random.seed(42)
n = 3772
returns_true = np.zeros(n)
sigma2_true = np.ones(n) * 1.0

omega = 0.00001
alpha = 0.10
beta = 0.85

z = np.random.standard_t(5, size=n)

for t in range(1, n):
    sigma2_true[t] = omega + alpha * returns_true[t-1]**2 + beta * sigma2_true[t-1]
    returns_true[t] = np.sqrt(sigma2_true[t]) * z[t]

returns_true *= 100  # Scale to percentage

df = pd.DataFrame({
    'Date': pd.date_range('2010-01-05', periods=n, freq='B'),
    'Log_Return': returns_true,
})
df.set_index('Date', inplace=True)

print(f"Generated {len(df)} observations")
print(f"Mean return: {df['Log_Return'].mean():.4f}%")
print(f"Std dev: {df['Log_Return'].std():.4f}%")
print()

# Train/test split
split_idx = int(0.8 * len(df))
train = df.iloc[:split_idx]
test = df.iloc[split_idx:]

print(f"Train: {len(train)} obs ({train.index[0].date()} to {train.index[-1].date()})")
print(f"Test:  {len(test)} obs ({test.index[0].date()} to {test.index[-1].date()})")
print()

# Fit GARCH models
print("="*70)
print(" FITTING GARCH MODELS")
print("="*70)
print()

configs = [
    {'name': 'GARCH(1,1)-Normal', 'vol': 'Garch', 'p': 1, 'o': 0, 'q': 1, 'dist': 'normal'},
    {'name': 'GARCH(1,1)-Student-t', 'vol': 'Garch', 'p': 1, 'o': 0, 'q': 1, 'dist': 't'},
    {'name': 'GJR-GARCH(1,1)-Normal', 'vol': 'Garch', 'p': 1, 'o': 1, 'q': 1, 'dist': 'normal'},
    {'name': 'GJR-GARCH(1,1)-Student-t', 'vol': 'Garch', 'p': 1, 'o': 1, 'q': 1, 'dist': 't'},
    {'name': 'EGARCH(1,1)-Normal', 'vol': 'EGarch', 'p': 1, 'o': 1, 'q': 1, 'dist': 'normal'},
    {'name': 'EGARCH(1,1)-Student-t', 'vol': 'EGarch', 'p': 1, 'o': 1, 'q': 1, 'dist': 't'},
]

garch_results = {}
for cfg in configs:
    print(f"Fitting {cfg['name']}...", end='  ')
    try:
        am = arch_model(train['Log_Return'], vol=cfg['vol'], p=cfg['p'],
                       o=cfg['o'], q=cfg['q'], dist=cfg['dist'], mean='Constant')
        res = am.fit(disp='off', show_warning=False)
        garch_results[cfg['name']] = res
        print(f"AIC={res.aic:.1f}  |  BIC={res.bic:.1f}  |  LL={res.loglikelihood:.1f}")
    except Exception as e:
        print(f"FAILED: {str(e)[:40]}")

print()

# Model comparison
rows = []
for name, res in garch_results.items():
    rows.append({
        'Model': name,
        'Log-Likelihood': round(res.loglikelihood, 2),
        'AIC': round(res.aic, 2),
        'BIC': round(res.bic, 2),
    })

comparison_df = pd.DataFrame(rows).sort_values('AIC').reset_index(drop=True)
best_model_name = comparison_df.iloc[0]['Model']
best_model = garch_results[best_model_name]

print("Model Comparison (sorted by AIC):")
print(comparison_df.to_string(index=False))
print(f"\n-> Best model: {best_model_name}")
print()

# Rolling volatility forecast for best model
print("="*70)
print(" ROLLING VOLATILITY FORECAST")
print("="*70)
print()

cfg = [c for c in configs if c['name'] == best_model_name][0]

def rolling_forecast(full_returns, test_index, train_size, vol, p, o, q, dist):
    forecasts = []
    for i in range(len(test_index)):
        window = full_returns.iloc[:train_size + i]
        am = arch_model(window, vol=vol, p=p, o=o, q=q, dist=dist, mean='Constant')
        res = am.fit(disp='off', show_warning=False)
        fc = res.forecast(horizon=1, reindex=False)
        var_hat = fc.variance.values[-1, 0]
        forecasts.append(np.sqrt(max(var_hat, 1e-8)))
        
        if (i + 1) % 100 == 0:
            print(f'  {i+1}/{len(test_index)} steps completed...')
    
    return pd.Series(forecasts, index=test_index, name='Forecast')

print(f"Forecasting with {best_model_name}...")
garch_forecast = rolling_forecast(
    full_returns=df['Log_Return'],
    test_index=test.index,
    train_size=len(train),
    vol=cfg['vol'],
    p=cfg['p'],
    o=cfg['o'],
    q=cfg['q'],
    dist=cfg['dist']
)

print(f"Done. Avg forecasted volatility: {garch_forecast.mean():.4f}%")
print()

# HAR-RV Model
print("="*70)
print(" HAR-RV BENCHMARK MODEL")
print("="*70)
print()

def compute_realized_volatility_proxy(returns):
    return returns.abs()

def build_har_features(rv_series, lags_weekly=5, lags_monthly=22):
    df_har = pd.DataFrame(index=rv_series.index)
    df_har['RV_target'] = rv_series
    df_har['RV_d'] = rv_series.shift(1)
    df_har['RV_w'] = rv_series.shift(1).rolling(lags_weekly).mean()
    df_har['RV_m'] = rv_series.shift(1).rolling(lags_monthly).mean()
    df_har.dropna(inplace=True)
    return df_har

print("Fitting HAR-RV model on training set...")
rv = compute_realized_volatility_proxy(train['Log_Return'])
df_har = build_har_features(rv)

X = add_constant(df_har[['RV_d', 'RV_w', 'RV_m']])
y = df_har['RV_target']
har_model = OLS(y, X).fit()

print(f"HAR-RV R-squared: {har_model.rsquared:.4f}")
print(f"Coefficients:")
print(f"  β_d (daily):   {har_model.params['RV_d']:.4f}")
print(f"  β_w (weekly):  {har_model.params['RV_w']:.4f}")
print(f"  β_m (monthly): {har_model.params['RV_m']:.4f}")
print()

# HAR-RV rolling forecast
print("Rolling HAR-RV forecast...")
def rolling_har_forecast(full_returns, test_index, train_size):
    forecasts = []
    for i in range(len(test_index)):
        window = full_returns.iloc[:train_size + i]
        rv = window.abs()
        df_har = build_har_features(rv)
        
        X = add_constant(df_har[['RV_d', 'RV_w', 'RV_m']])
        y = df_har['RV_target']
        m = OLS(y, X).fit()
        
        rv_d = rv.iloc[-1]
        rv_w = rv.iloc[-5:].mean()
        rv_m = rv.iloc[-22:].mean()
        
        fc_val = (m.params['const'] + m.params['RV_d']*rv_d + 
                  m.params['RV_w']*rv_w + m.params['RV_m']*rv_m)
        forecasts.append(max(fc_val, 1e-8))
        
        if (i + 1) % 100 == 0:
            print(f'  {i+1}/{len(test_index)} steps completed...')
    
    return pd.Series(forecasts, index=test_index, name='HAR_RV')

har_forecast = rolling_har_forecast(df['Log_Return'], test.index, len(train))
print(f"Done. Avg HAR-RV volatility: {har_forecast.mean():.4f}%")
print()

# Value-at-Risk
print("="*70)
print(" VALUE-AT-RISK & EXPECTED SHORTFALL")
print("="*70)
print()

def compute_var_es(vol_forecast, mean_return=0.0, confidence_levels=None, dist='t', df_t=5.0):
    if confidence_levels is None:
        confidence_levels = [0.95, 0.99]
    
    output = {}
    for cl in confidence_levels:
        alpha = 1 - cl
        
        if dist == 'normal':
            z_var = stats.norm.ppf(alpha)
            z_es = stats.norm.pdf(stats.norm.ppf(alpha)) / alpha
            es = mean_return - z_es * vol_forecast
        else:
            z_var = stats.t.ppf(alpha, df=df_t)
            es_num = stats.t.pdf(z_var, df=df_t) * (df_t + z_var**2) / (df_t - 1)
            es = mean_return - (es_num / alpha) * vol_forecast
        
        var = mean_return + z_var * vol_forecast
        output[f'VaR_{int(cl*100)}'] = var
        output[f'ES_{int(cl*100)}'] = es
    
    return output

var_es_garch = compute_var_es(garch_forecast, dist='t')
var_es_har = compute_var_es(har_forecast, dist='normal')

print(f"GARCH VaR 95%: {var_es_garch['VaR_95'].mean():.4f}%  (expected: 5% violations)")
print(f"GARCH VaR 99%: {var_es_garch['VaR_99'].mean():.4f}%  (expected: 1% violations)")
print(f"HAR-RV VaR 95%: {var_es_har['VaR_95'].mean():.4f}%")
print(f"HAR-RV VaR 99%: {var_es_har['VaR_99'].mean():.4f}%")
print()

# Backtesting
print("="*70)
print(" BACKTESTING (Kupiec POF Test)")
print("="*70)
print()

def kupiec_test(actual, var_series, confidence):
    alpha = 1 - confidence
    violations = (actual < var_series).astype(int)
    T, N = len(actual), violations.sum()
    p_hat = N / T
    
    if N == 0 or N == T:
        return {'violations': int(N), 'percent': round(p_hat*100, 2), 'pass': 'N/A'}
    
    LR = -2 * (N * np.log(alpha / p_hat) + (T-N) * np.log((1-alpha) / (1-p_hat)))
    p_val = 1 - stats.chi2.cdf(LR, df=1)
    
    return {
        'violations': int(N),
        'percent': round(p_hat*100, 2),
        'expected': round(alpha*100, 2),
        'p_value': round(p_val, 4),
        'pass': 'YES' if p_val > 0.05 else 'NO'
    }

print("GARCH Model:")
k95 = kupiec_test(test['Log_Return'], var_es_garch['VaR_95'], 0.95)
k99 = kupiec_test(test['Log_Return'], var_es_garch['VaR_99'], 0.99)
print(f"  95% VaR: {k95['violations']} violations ({k95['percent']}%) — Pass: {k95['pass']}")
print(f"  99% VaR: {k99['violations']} violations ({k99['percent']}%) — Pass: {k99['pass']}")

print("\nHAR-RV Model:")
h95 = kupiec_test(test['Log_Return'], var_es_har['VaR_95'], 0.95)
h99 = kupiec_test(test['Log_Return'], var_es_har['VaR_99'], 0.99)
print(f"  95% VaR: {h95['violations']} violations ({h95['percent']}%) — Pass: {h95['pass']}")
print(f"  99% VaR: {h99['violations']} violations ({h99['percent']}%) — Pass: {h99['pass']}")
print()

# Forecast Accuracy
print("="*70)
print(" FORECAST ACCURACY COMPARISON")
print("="*70)
print()

actual_rv = test['Log_Return'].abs()

for name, fc in [('GARCH', garch_forecast), ('HAR-RV', har_forecast)]:
    fc_aligned = fc.reindex(actual_rv.index)
    rv_aligned = actual_rv.reindex(fc_aligned.index)
    
    mse = np.mean((rv_aligned - fc_aligned) ** 2)
    mae = np.mean(np.abs(rv_aligned - fc_aligned))
    
    print(f"{name}:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")

print()

# Save results
print("="*70)
print(" SAVING RESULTS")
print("="*70)
print()

comparison_df.to_csv('results/model_comparison.csv', index=False)

# VaR/ES results
var_results = pd.DataFrame()
var_results['GARCH_VaR95'] = var_es_garch['VaR_95']
var_results['GARCH_VaR99'] = var_es_garch['VaR_99']
var_results['HAR_RV_VaR95'] = var_es_har['VaR_95']
var_results['HAR_RV_VaR99'] = var_es_har['VaR_99']
var_results.to_csv('results/var_forecasts.csv')

# Volatility forecasts
vol_df = pd.DataFrame()
vol_df['GARCH'] = garch_forecast
vol_df['HAR_RV'] = har_forecast
vol_df['Actual'] = test['Log_Return'].abs()
vol_df.to_csv('results/volatility_forecasts.csv')

print("Saved:")
print("  results/model_comparison.csv")
print("  results/var_forecasts.csv")
print("  results/volatility_forecasts.csv")
print()

print("="*70)
print(" ANALYSIS COMPLETE")
print("="*70)
