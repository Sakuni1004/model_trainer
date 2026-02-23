import pandas as pd
import numpy as np
import os
import joblib
import shap
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(_file_))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'mangosteen_prices_2023_2025.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'mangosteen_analysis')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)

# Features & Target
features = ['region_encoded', 'temp_c', 'rain_mm', 'humid', 'year', 'month', 'day_of_week']
X = df[features]
y = df['price_f']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# XGBoost Model (NEW algorithm)
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluation
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

r2 = r2_score(y_test, test_pred)
mae = mean_absolute_error(y_test, test_pred)
rmse = np.sqrt(mean_squared_error(y_test, test_pred))

print("Model Performance")
print("R2:", r2)
print("MAE:", mae)
print("RMSE:", rmse)

# Save model
model_path = os.path.join(OUTPUT_DIR, 'mangosteen_xgb_model.pkl')
joblib.dump(model, model_path)

# =====================
# SHAP Explainability
# =====================
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# Global importance
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig(os.path.join(OUTPUT_DIR, 'shap_summary.png'), bbox_inches='tight')

# Feature importance bar
plt.figure()
shap.plots.bar(shap_values, show=False)
plt.savefig(os.path.join(OUTPUT_DIR, 'shap_bar.png'), bbox_inches='tight')

print("Model and SHAP outputs saved.")