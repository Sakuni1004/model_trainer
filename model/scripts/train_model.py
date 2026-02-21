import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def print_performance_summary(name, y_train, train_pred, y_test, test_pred):
    """Calculates and prints performance metrics in percentage format"""
    
    r2_tr = r2_score(y_train, train_pred)
    corr_tr = np.corrcoef(y_train, train_pred)[0, 1]

    r2_te = r2_score(y_test, test_pred)
    corr_te = np.corrcoef(y_test, test_pred)[0, 1]

    mae_te = mean_absolute_error(y_test, test_pred)
    mse_te = mean_squared_error(y_test, test_pred)
    rmse_te = np.sqrt(mse_te)

    # 🔹 Percentage Metrics
    mean_price = np.mean(y_test)
    mape = np.mean(np.abs((y_test - test_pred) / y_test)) * 100
    mae_percent = (mae_te / mean_price) * 100
    rmse_percent = (rmse_te / mean_price) * 100
    mse_percent = (mse_te / (mean_price ** 2)) * 100   # ✅ MSE percentage

    summary = (
        f"\n        MODEL PERFORMANCE SUMMARY: {name.upper()}\n"
        f"========================================\n"
        f"TRAINING SET:\n"
        f"  R-squared (R²):  {r2_tr:.4f} ({r2_tr*100:.2f}%)\n"
        f"  Correlation (R): {corr_tr:.4f} ({corr_tr*100:.2f}%)\n"
        f"----------------------------------------\n"
        f"TEST SET:\n"
        f"  R-squared (R²):  {r2_te:.4f} ({r2_te*100:.2f}%)\n"
        f"  Correlation (R): {corr_te:.4f} ({corr_te*100:.2f}%)\n"
        f"  MAE:             Rs. {mae_te:,.2f} ({mae_percent:.2f}%)\n"
        f"  MSE:             ({mse_percent:.4f}%)\n"
        f"  RMSE:            Rs. {rmse_te:,.2f} ({rmse_percent:.2f}%)\n"
        f"  MAPE:            {mape:.2f}%\n"
        f"========================================\n"
    )

    print(summary)
    return summary

def save_model_diagnostics(model, X_test, y_test, features, output_path, sample_size=100):
    """Generates Detailed Accuracy and Feature Importance plots"""
    test_pred = model.predict(X_test)
    
    if len(y_test) > sample_size:
        indices = np.random.choice(len(y_test), sample_size, replace=False)
        y_plot = y_test.iloc[indices]
        pred_plot = test_pred[indices]
    else:
        y_plot = y_test
        pred_plot = test_pred

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # 1. ACCURACY PLOT
    ax1.scatter(y_plot, pred_plot, alpha=0.7, color='#8e44ad', edgecolors='k', label='Sampled Data')
    max_val = max(y_plot.max(), pred_plot.max())
    min_val = min(y_plot.min(), pred_plot.min())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    ax1.set_xlabel('Actual Price (Rs.)')
    ax1.set_ylabel('Predicted Price (Rs.)')
    ax1.set_title(f'Mangosteen Accuracy (Random {sample_size} Samples)')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)

    # 2. FEATURE IMPORTANCE
    importances = model.feature_importances_
    idx = np.argsort(importances)
    ax2.barh(range(len(idx)), importances[idx], color='#2ecc71', align='center', edgecolor='k')
    ax2.set_yticks(range(len(idx)))
    ax2.set_yticklabels([features[i] for i in idx])
    ax2.set_xlabel('Importance Score')
    ax2.set_title('Mangosteen Price Drivers')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)

def analyze_mangosteen_dynamics():
    # 1. SETUP & DATA LOADING
    base_dir = '/Users/sakunika/Desktop/model'
    data_path = os.path.join(base_dir, 'data/processed/mangosteen_prices_2023_2025.csv')
    output_dir = os.path.join(base_dir, 'output/mangosteen_analysis')
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        return

    df_fruit = pd.read_csv(data_path)

    # 2. FEATURE SELECTION
    features = ['region_encoded', 'temp_c', 'rain_mm', 'humid', 'year', 'month', 'day_of_week']
    X = df_fruit[features]
    y = df_fruit['price_f']

    # 3. TRAINING
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # 4. DIAGNOSTICS
    save_model_diagnostics(model, X_test, y_test, features, 
                           os.path.join(output_dir, 'mangosteen_accuracy_importance.png'), 
                           sample_size=100)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    perf_text = print_performance_summary("Mangosteen Specialized Model", y_train, train_pred, y_test, test_pred)
    
    with open(os.path.join(output_dir, 'model_summary.txt'), 'w') as f:
        f.write(perf_text)

    # 5. SENSITIVITY SIMULATION
    # Remove fruit_type_encoded from the baseline dictionary
    baseline = {
        'region_encoded': 4,
        'temp_c': df_fruit['temp_c'].mean(),
        'rain_mm': df_fruit['rain_mm'].mean(),
        'humid': df_fruit['humid'].mean(),
        'year': 2025,
        'month': 6,
        'day_of_week': 0
    }

    def simulate_change(feature_name, values):
        test_df = pd.DataFrame([baseline] * len(values))
        test_df[feature_name] = values
        return model.predict(test_df)

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    plt.subplots_adjust(hspace=0.4)

    cities = sorted(df_fruit['region_encoded'].unique())
    axs[0, 0].bar([f"City {c}" for c in cities], simulate_change('region_encoded', cities), color='#8e44ad')
    axs[0, 0].set_title("Price Change by City")

    h_range = np.linspace(df_fruit['humid'].min(), df_fruit['humid'].max(), 50)
    axs[0, 1].plot(h_range, simulate_change('humid', h_range), color='#2ecc71', lw=3)
    axs[0, 1].set_title("Impact of Humidity")

    r_range = np.linspace(df_fruit['rain_mm'].min(), df_fruit['rain_mm'].max(), 50)
    axs[1, 0].plot(r_range, simulate_change('rain_mm', r_range), color='#34495e', lw=3)
    axs[1, 0].set_title("Impact of Rain (mm)")

    t_range = np.linspace(df_fruit['temp_c'].min(), df_fruit['temp_c'].max(), 50)
    axs[1, 1].plot(t_range, simulate_change('temp_c', t_range), color='#e67e22', lw=3)
    axs[1, 1].set_title("Impact of Temperature")

    plt.suptitle(f"Mangosteen Price Dynamics Analysis\n(Overall R²={r2_score(y_test, test_pred):.2f})", fontsize=16)
    plt.savefig(os.path.join(output_dir, 'mangosteen_sensitivity.png'))
    
    # 6. SAVE ARTIFACTS
    joblib.dump(model, os.path.join(output_dir, 'mangosteen_specialized_model.pkl'))
    print(f"\nMangosteen Analysis Complete. Files saved in: {output_dir}")

if __name__ == "__main__":
    analyze_mangosteen_dynamics()