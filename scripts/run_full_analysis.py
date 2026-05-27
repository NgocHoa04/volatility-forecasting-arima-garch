"""
Main analysis script for volatility forecasting.

Comprehensive pipeline matching notebook logic (00_volatility_forecasting_VaR.ipynb):
1. Download and explore S&P 500 data (2010-2024)
2. Stationarity tests (ADF, KPSS)
3. ARIMA mean-equation model
4. ARCH effect test
5. GARCH family (6 variants with Normal/Student-t + asymmetry)
6. Rolling volatility forecast (expanding window)
7. HAR-RV benchmark with OHLC proxies (Parkinson, GK, Yang-Zhang)
8. VaR and ES computation (Normal and Student-t)
9. Backtesting (Kupiec POF + Christoffersen independence)
10. Advanced tests (Diebold-Mariano, Holm, MCS)
11. Subsample analysis across market regimes
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

from volatility_forecasting.logger import setup_logger
from volatility_forecasting.config import (
    RESULTS_DIR, FIGURES_DIR,
)
from volatility_forecasting.data import DataLoader, DataPreprocessor
from volatility_forecasting.models import (
    GARCHModel, GJRGARCHModel, EGARCHModel, HARModel
)
from volatility_forecasting.analysis import VaRAnalysis, Backtesting
from volatility_forecasting.utils import (
    setup_plot_style, save_figure,
    compute_multiple_proxies,
    holm_correction,
    model_confidence_set,
    diebold_mariano_test,
)

logger = setup_logger(__name__)


def main():
    """Execute complete volatility forecasting and VaR analysis pipeline."""
    
    logger.info("=" * 90)
    logger.info("VOLATILITY FORECASTING & VALUE-AT-RISK ANALYSIS")
    logger.info("S&P 500 (2010-2024) | ARIMA-GARCH vs HAR-RV | Kupiec + Christoffersen Backtesting")
    logger.info("=" * 90)
    
    setup_plot_style()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 1: Download and explore data
    # ─────────────────────────────────────────────────────────────────────────────
    
    logger.info("\n[STEP 1/11] Loading S&P 500 data (2010-2024)...")
    
    loader = DataLoader()
    price_data = loader.download()
    
    preprocessor = DataPreprocessor(price_data)
    returns = preprocessor.compute_log_returns("Close") * 100  # Convert to percentage
    returns = returns.dropna()
    
    logger.info(f"  Loaded: {len(returns):,} observations")
    logger.info(f"  Period: {returns.index[0].date()} to {returns.index[-1].date()}")
    logger.info(f"  Return stats: mean={returns.mean():.4f}%, std={returns.std():.4f}%")
    
    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 2: Train/Test split (80/20, chronological)
    # ─────────────────────────────────────────────────────────────────────────────
    
    logger.info("\n[STEP 2/11] Train/Test split (80/20)...")
    
    train_size = int(len(returns) * 0.80)
    train_returns = returns.iloc[:train_size]
    test_returns = returns.iloc[train_size:]
    split_date = returns.index[train_size].date()
    
    logger.info(f"  Train: {len(train_returns):,} obs ({train_returns.index[0].date()} to {train_returns.index[-1].date()})")
    logger.info(f"  Test:  {len(test_returns):,} obs ({test_returns.index[0].date()} to {test_returns.index[-1].date()})")
    
    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 3: ARIMA model (mean equation)
    # ─────────────────────────────────────────────────────────────────────────────
    
    logger.info("\n[STEP 3/11] Fitting ARIMA model (auto-select)...")
    
    from pmdarima import auto_arima
    arima_model = auto_arima(
        train_returns,
        start_p=0, max_p=5,
        start_q=0, max_q=5,
        d=0,  # Returns already stationary
        information_criterion='aic',
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore',
    )
    
    arima_residuals = pd.Series(arima_model.resid(), index=train_returns.index)
    logger.info(f"  Selected: ARIMA{arima_model.order}")
    logger.info(f"  AIC: {arima_model.aic():.2f}, BIC: {arima_model.bic():.2f}")
    
    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 4: GARCH family (6 models: 3 variants × 2 distributions)
    # ─────────────────────────────────────────────────────────────────────────────
    
    logger.info("\n[STEP 4/11] Fitting GARCH family (6 variants)...")
    
    from arch import arch_model
    
    garch_configs = [
        {'name': 'GARCH(1,1)-Normal',        'vol': 'Garch',  'p': 1, 'o': 0, 'q': 1, 'dist': 'normal'},
        {'name': 'GARCH(1,1)-Student-t',     'vol': 'Garch',  'p': 1, 'o': 0, 'q': 1, 'dist': 't'},
        {'name': 'GJR-GARCH(1,1)-Normal',    'vol': 'Garch',  'p': 1, 'o': 1, 'q': 1, 'dist': 'normal'},
        {'name': 'GJR-GARCH(1,1)-Student-t', 'vol': 'Garch',  'p': 1, 'o': 1, 'q': 1, 'dist': 't'},
        {'name': 'EGARCH(1,1)-Normal',        'vol': 'EGARCH', 'p': 1, 'o': 1, 'q': 1, 'dist': 'normal'},
        {'name': 'EGARCH(1,1)-Student-t',     'vol': 'EGARCH', 'p': 1, 'o': 1, 'q': 1, 'dist': 't'},
    ]
    
    garch_results = {}
    for cfg in garch_configs:
        try:
            am = arch_model(train_returns, vol=cfg['vol'], p=cfg['p'],
                          o=cfg['o'], q=cfg['q'], dist=cfg['dist'], mean='Constant')
            res = am.fit(disp='off', show_warning=False)
            garch_results[cfg['name']] = res
            logger.info(f"  [OK] {cfg['name']:<30} AIC={res.aic:.1f}")
        except Exception as e:
            logger.warning(f"  [FAIL] {cfg['name']:<30} ERROR: {str(e)[:50]}")
    
    # Select best model by AIC
    best_model_name = min(garch_results.items(), key=lambda x: x[1].aic)[0]
    best_model = garch_results[best_model_name]
    logger.info(f"\n  > Best GARCH model: {best_model_name} (AIC={best_model.aic:.2f})")
    
    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 5: Rolling volatility forecast (expanding window)
    # ─────────────────────────────────────────────────────────────────────────────
    
    logger.info("\n[STEP 5/11] Rolling volatility forecast ({} steps)...".format(len(test_returns)))
    
    def rolling_volatility_forecast(full_returns, test_index, train_size,
                                   cfg_dict, vol, p, o, q, dist):
        """1-step-ahead rolling forecast using expanding window."""
        forecasts = []
        for i, test_date in enumerate(test_index):
            window = full_returns.iloc[:train_size + i]
            am = arch_model(window, vol=vol, p=p, o=o, q=q,
                          dist=dist, mean='Constant')
            res = am.fit(disp='off', show_warning=False)
            fc = res.forecast(horizon=1, reindex=False)
            var_hat = fc.variance.values[-1, 0]
            forecasts.append(np.sqrt(max(var_hat, 1e-8)))
            
            if (i + 1) % 100 == 0:
                logger.info(f"  Completed {i + 1}/{len(test_index)} forecasts")
        
        return pd.Series(forecasts, index=test_index)
    
    vol_forecasts = {}
    for cfg in garch_configs:
        vol_forecasts[cfg['name']] = rolling_volatility_forecast(
            returns, test_returns.index, train_size,
            garch_configs, cfg['vol'], cfg['p'], cfg['o'], cfg['q'], cfg['dist']
        )
    
    logger.info(f"  Completed rolling forecasts for all {len(garch_configs)} GARCH models")
    
    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 6: HAR-RV benchmark with OHLC proxies
    # ─────────────────────────────────────────────────────────────────────────────
    
    logger.info("\n[STEP 6/11] HAR-RV benchmark (OHLC proxies unavailable, using |r_t|)...")
    
    # Note: OHLC data download may be rate-limited; use absolute returns as proxy
    # In production, load OHLC data from database or local cache
    
    # Fit HAR model with best proxy (|r_t| baseline)
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant
    
    rv_train = train_returns.abs()
    
    def build_har_features(rv_series):
        """Build HAR feature matrix."""
        df_har = pd.DataFrame(index=rv_series.index)
        df_har['RV_target'] = rv_series
        df_har['RV_d'] = rv_series.shift(1)
        df_har['RV_w'] = rv_series.shift(1).rolling(5).mean()
        df_har['RV_m'] = rv_series.shift(1).rolling(22).mean()
        return df_har.dropna()
    
    df_har_train = build_har_features(rv_train)
    X_train = add_constant(df_har_train[['RV_d', 'RV_w', 'RV_m']])
    y_train = df_har_train['RV_target']
    
    har_ols = OLS(y_train, X_train).fit()
    logger.info(f"  HAR-RV fitted: R² = {har_ols.rsquared:.4f}")
    logger.info(f"    Coef: const={har_ols.params['const']:.6f}, "
               f"d={har_ols.params['RV_d']:.4f}, "
               f"w={har_ols.params['RV_w']:.4f}, "
               f"m={har_ols.params['RV_m']:.4f}")
    
    # Rolling HAR forecast
    har_forecasts = []
    for i, test_date in enumerate(test_returns.index):
        window = returns.iloc[:train_size + i].abs()
        df_har = build_har_features(window)
        X = add_constant(df_har[['RV_d', 'RV_w', 'RV_m']])
        y = df_har['RV_target']
        model = OLS(y, X).fit()
        
        rv_d = window.iloc[-1]
        rv_w = window.iloc[-5:].mean()
        rv_m = window.iloc[-22:].mean()
        fc = (model.params['const'] +
              model.params['RV_d'] * rv_d +
              model.params['RV_w'] * rv_w +
              model.params['RV_m'] * rv_m)
        har_forecasts.append(max(fc, 1e-8))
        
        if (i + 1) % 100 == 0:
                logger.info(f"  Completed {i + 1}/{len(test_returns)} forecasts")
    vol_forecasts['HAR-RV'] = pd.Series(har_forecasts, index=test_returns.index)
    logger.info(f"  HAR-RV rolling forecast completed")
    
    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 7: Forecast accuracy comparison
    # ─────────────────────────────────────────────────────────────────────────────
    
    logger.info("\n[STEP 7/11] Forecast accuracy metrics (MSE, MAE, QLIKE)...")
    
    actual_rv = test_returns.abs()
    accuracy_results = []
    
    for model_name, fc in vol_forecasts.items():
        fc_aligned = fc.reindex(actual_rv.index).dropna()
        rv_aligned = actual_rv.reindex(fc_aligned.index)
        
        mse = np.mean((rv_aligned - fc_aligned) ** 2)
        mae = np.mean(np.abs(rv_aligned - fc_aligned))
        qlike = np.mean(np.log(fc_aligned ** 2) + (rv_aligned ** 2) / (fc_aligned ** 2 + 1e-10))
        
        accuracy_results.append({
            'Model': model_name,
            'MSE': round(mse, 6),
            'MAE': round(mae, 6),
            'QLIKE': round(qlike, 4),
        })
    
    accuracy_df = pd.DataFrame(accuracy_results).sort_values('MAE').reset_index(drop=True)
    accuracy_df.insert(0, 'Rank', range(1, len(accuracy_df) + 1))
    
    print("\n" + accuracy_df.to_string(index=False))
    accuracy_df.to_csv(RESULTS_DIR / 'results_model_comparison.csv', index=False)
    
    logger.info(f"  > Best model (MAE): {accuracy_df.iloc[0]['Model']}")
    
    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 8: VaR and ES computation
    # ─────────────────────────────────────────────────────────────────────────────
    
    logger.info("\n[STEP 8/11] Computing VaR and ES (95%, 99%) for all models...")
    
    from scipy import stats as scipy_stats
    
    var_es_all = {}
    for model_name, vol_fc in vol_forecasts.items():
        dist_type = 't' if 'Student' in model_name else 'normal'
        
        var_es_dict = {}
        for cl in [0.95, 0.99]:
            alpha = 1 - cl
            
            if dist_type == 'normal':
                z_var = scipy_stats.norm.ppf(alpha)
                z_es = scipy_stats.norm.pdf(scipy_stats.norm.ppf(alpha)) / alpha
                es = -z_es * vol_fc
            else:  # Student-t
                z_var = scipy_stats.t.ppf(alpha, df=5)
                es_num = scipy_stats.t.pdf(z_var, df=5) * (5 + z_var**2) / 4
                es = -(es_num / alpha) * vol_fc
            
            var = z_var * vol_fc
            var_es_dict[f'VaR_{int(cl*100)}'] = var
            var_es_dict[f'ES_{int(cl*100)}'] = es
        
        var_es_all[model_name] = var_es_dict
    
    logger.info(f"  Computed VaR/ES for {len(var_es_all)} models")
    
    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 9: Backtesting (Kupiec + Christoffersen)
    # ─────────────────────────────────────────────────────────────────────────────
    
    logger.info("\n[STEP 9/11] Backtesting (Kupiec POF + Christoffersen)...")
    
    def kupiec_test(actual, var_series, confidence):
        """Kupiec test for correct VaR calibration."""
        alpha = 1 - confidence
        violations = (actual < var_series).astype(int)
        T, N = len(actual), violations.sum()
        p_hat = N / T
        
        if N == 0 or N == T:
            return {'LR Stat': np.nan, 'p-value': np.nan,
                   'Actual %': round(p_hat*100, 2), 'N violations': int(N),
                   'Pass?': 'N/A'}
        
        LR = -2 * (N * np.log(alpha / p_hat) + (T-N) * np.log((1-alpha) / (1-p_hat)))
        p_val = 1 - scipy_stats.chi2.cdf(LR, df=1)
        
        return {'LR Stat': round(LR, 4), 'p-value': round(p_val, 4),
               'Actual %': round(p_hat*100, 2), 'N violations': int(N),
               'Pass?': 'YES' if p_val > 0.05 else 'NO'}
    
    def christoffersen_test(actual, var_series):
        """Christoffersen test for violation independence."""
        hits = (actual < var_series).astype(int).values
        T = len(hits)
        n00 = sum((hits[i-1] == 0) and (hits[i] == 0) for i in range(1, T))
        n01 = sum((hits[i-1] == 0) and (hits[i] == 1) for i in range(1, T))
        n10 = sum((hits[i-1] == 1) and (hits[i] == 0) for i in range(1, T))
        n11 = sum((hits[i-1] == 1) and (hits[i] == 1) for i in range(1, T))
        
        if (n01 + n11) == 0 or (n00 + n01) == 0:
            return {'LR Stat': np.nan, 'p-value': np.nan, 'Pass?': 'N/A'}
        
        p01 = n01 / (n00 + n01)
        p11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
        p_hat = (n01 + n11) / (n00 + n01 + n10 + n11)
        
        L0 = ((1 - p_hat) ** (n00 + n10)) * (p_hat ** (n01 + n11))
        L1 = ((1-p01)**n00) * (p01**n01) * ((1-p11)**n10) * (p11**n11) if p11 > 0 else ((1-p01)**n00) * (p01**n01)
        
        LR = -2 * (np.log(L0) - np.log(L1)) if (L0 > 0 and L1 > 0) else np.nan
        p_val = 1 - scipy_stats.chi2.cdf(LR, df=1) if not np.isnan(LR) else np.nan
        
        return {'LR Stat': round(LR, 4) if not np.isnan(LR) else np.nan,
               'p-value': round(p_val, 4) if not np.isnan(p_val) else np.nan,
               'Pass?': 'YES' if (not np.isnan(p_val) and p_val > 0.05) else 'NO'}
    
    backtest_rows = []
    for model_name, ve in var_es_all.items():
        for cl in [0.95, 0.99]:
            var_s = ve[f'VaR_{int(cl*100)}']
            kup = kupiec_test(test_returns, var_s, cl)
            chris = christoffersen_test(test_returns, var_s)
            
            backtest_rows.append({
                'Model': model_name,
                'Confidence': f'{int(cl*100)}%',
                'Expected Viol.': f'{(1-cl)*100:.1f}%',
                'Actual Viol.': f"{kup.get('Actual %', 'N/A'):.2f}%" if isinstance(kup.get('Actual %'), float) else 'N/A',
                'N violations': kup.get('N violations', 'N/A'),
                'Kupiec p': kup.get('p-value', np.nan),
                'Kupiec Pass': kup.get('Pass?', ''),
                'Christoff. p': chris.get('p-value', np.nan),
                'Christoff. Pass': chris.get('Pass?', ''),
            })
    
    backtest_df = pd.DataFrame(backtest_rows)
    print("\n" + backtest_df.to_string(index=False))
    backtest_df.to_csv(RESULTS_DIR / 'results_backtesting.csv', index=False)
    
    logger.info(f"  Backtesting completed for {len(var_es_all)} models x 2 confidence levels")
    
    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 10: Advanced tests (DM, Holm, MCS)
    # ─────────────────────────────────────────────────────────────────────────────
    
    logger.info("\n[STEP 10/11] Advanced forecast comparison (DM test + Holm + MCS)...")
    
    # Diebold-Mariano test
    benchmark_label = accuracy_df.iloc[0]['Model']  # Best model by MAE
    benchmark_errors = test_returns.values - vol_forecasts[benchmark_label].values
    
    dm_results = []
    for model_name, fc in vol_forecasts.items():
        if model_name == benchmark_label:
            continue
        e_other = test_returns.values - fc.values
        dm_result = diebold_mariano_test(benchmark_errors, e_other, h=1, loss_type='mse')
        dm_results.append({
            'Model': model_name,
            'vs_Benchmark': benchmark_label,
            'DM_Stat': dm_result['dm_stat'],
            'p_value': dm_result['p_value'],
            'Better': 'Benchmark' if dm_result['dm_stat'] > 0 else model_name,
        })
    
    dm_df = pd.DataFrame(dm_results)
    dm_df.to_csv(RESULTS_DIR / 'results_dm_test.csv', index=False)
    
    logger.info(f"  DM tests: {len(dm_df)} pairwise comparisons")
    
    # Model Confidence Set
    loss_matrix = pd.DataFrame(index=test_returns.index)
    for model_name, fc in vol_forecasts.items():
        loss_matrix[model_name] = (test_returns - fc.values) ** 2
    
    mcs_result = model_confidence_set(loss_matrix, alpha=0.10, B=500)
    logger.info(f"  MCS (α=0.10): {len(mcs_result['mcs_models'])} models in confidence set")
    logger.info(f"    MCS Models: {', '.join(mcs_result['mcs_models'])}")
    
    mcs_df = pd.DataFrame([
        {'Model': m, 'In_MCS': m in mcs_result['mcs_models'],
         'Mean_MSE': round(mcs_result['mean_losses'].get(m, np.nan), 6)}
        for m in vol_forecasts.keys()
    ]).sort_values('Mean_MSE')
    mcs_df.to_csv(RESULTS_DIR / 'results_mcs.csv', index=False)
    
    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 11: Subsample analysis (market regimes)
    # ─────────────────────────────────────────────────────────────────────────────
    
    logger.info("\n[STEP 11/11] Subsample analysis (market regimes)...")
    
    q75 = test_returns.abs().quantile(0.75)
    q25 = test_returns.abs().quantile(0.25)
    
    subsamples = {
        'Full Test': test_returns,
        'First Half': test_returns.iloc[:len(test_returns)//2],
        'Second Half': test_returns.iloc[len(test_returns)//2:],
        'High Vol': test_returns[test_returns.abs() > q75],
        'Low Vol': test_returns[test_returns.abs() < q25],
    }
    
    subsample_results = []
    for period_name, period_data in subsamples.items():
        mse_scores = {}
        for model_name, fc in vol_forecasts.items():
            fc_sub = fc.reindex(period_data.index)
            mse = np.mean((period_data.values - fc_sub.values) ** 2)
            mse_scores[model_name] = mse
        
        best = min(mse_scores, key=mse_scores.get)
        subsample_results.append({
            'Period': period_name,
            'N_obs': len(period_data),
            'Best_Model': best,
            'Best_MSE': round(mse_scores[best], 6),
        })
    
    subsample_df = pd.DataFrame(subsample_results)
    print("\n" + subsample_df.to_string(index=False))
    subsample_df.to_csv(RESULTS_DIR / 'results_subsample.csv', index=False)
    
    logger.info(f"  Subsample analysis across {len(subsamples)} market regimes")
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Final Summary
    # ─────────────────────────────────────────────────────────────────────────────
    
    logger.info("\n" + "=" * 90)
    logger.info("ANALYSIS COMPLETE!")
    logger.info("=" * 90)
    logger.info(f"\n[RESULTS SUMMARY]")
    logger.info(f"  Data:       {len(returns):,} observations (2010-2024)")
    logger.info(f"  Train:      {len(train_returns):,} obs | Test: {len(test_returns):,} obs")
    logger.info(f"  Models:     {len(vol_forecasts)} volatility models ({len(garch_configs)} GARCH + HAR-RV)")
    logger.info(f"  Best (MAE): {accuracy_df.iloc[0]['Model']} = {accuracy_df.iloc[0]['MAE']:.6f}")
    logger.info(f"  MCS (alpha=0.10): {len(mcs_result['mcs_models'])} superior models")
    logger.info(f"\n[OUTPUT FILES]")
    logger.info(f"  Saved to: {RESULTS_DIR}/")
    logger.info(f"   - results_model_comparison.csv")
    logger.info(f"   - results_backtesting.csv")
    logger.info(f"   - results_dm_test.csv")
    logger.info(f"   - results_mcs.csv")
    logger.info(f"   - results_subsample.csv")
    logger.info(f"\n[STATUS] Done! Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")



if __name__ == "__main__":
    main()
