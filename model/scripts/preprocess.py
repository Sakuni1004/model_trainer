import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

def clean_numeric_value(val):
    if pd.isna(val) or str(val).lower() == 'nan':
        return np.nan
    clean_val = str(val).replace('C', '').replace('mm', '').strip()
    try:
        return float(clean_val)
    except ValueError:
        return np.nan

def get_preprocessed_mangosteen_2023_2025(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at: {file_path}")
        
    df = pd.read_csv(file_path, low_memory=False)

    # 1. Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

    # 2. Categorical Cleaning
    cat_cols = ['region', 'fruit_type', 'veg_type']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
            df.loc[df[col] == 'Nan', col] = np.nan
            mode_val = df[col].mode()
            df[col] = df[col].fillna(mode_val[0] if not mode_val.empty else 'Unknown')

    # 3. Numeric Cleaning
    potential_nums = ['temp_c', 'rain_mm', 'humid', 'score', 'price_f', 'price_v']
    found_nums = [c for c in potential_nums if c in df.columns]

    for col in found_nums:
        if col in ['temp_c', 'rain_mm']:
            df[col] = df[col].apply(clean_numeric_value)
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

    # 4. Outlier Handling for Price
    if 'price_f' in df.columns:
        med = df['price_f'].median()
        df['price_f'] = np.where(df['price_f'] > 1000, med, df['price_f'])

    # 5. Date Features
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce').ffill().bfill()
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
    else:
        raise KeyError("Column 'date' missing!")

    # 6. Encoding Region
    le = LabelEncoder()
    if 'region' in df.columns:
        df['region_encoded'] = le.fit_transform(df['region'].astype(str))
        # Optional: Print mapping to console
        mapping = dict(zip(le.transform(le.classes_), le.classes_))
        print("Region Mapping Code:", mapping)

    # 7. Scale Numeric Columns
    scale_targets = ['temp_c', 'rain_mm', 'humid']
    cols_to_scale = [c for c in scale_targets if c in df.columns]
    if cols_to_scale:
        scaler = StandardScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    # 8. Filter for Mangosteen only
    mangosteen_df = df[df['fruit_type'] == 'Mangosteen'].copy()
    
    # 9. Combine Real 2023-24 with Projected 2025
    df_real = mangosteen_df[mangosteen_df['year'].between(2023, 2024)].copy()
    df_2025 = mangosteen_df[mangosteen_df['year'] == 2024].copy()
    if not df_2025.empty:
        df_2025['year'] = 2025
        final_df = pd.concat([df_real, df_2025], ignore_index=True).drop_duplicates()
    else:
        final_df = df_real

    # 10. Final Column Selection
    # Rename 'region' to 'city' as requested and keep it in the final file
    final_df = final_df.rename(columns={'region': 'city'})
    
    # List of columns we definitely DON'T want
    unwanted = ['date', 'score', 'veg_type', 'price_v', 'fruit_type', 'veg_type_encoded']
    cols_to_drop = [c for c in unwanted if c in final_df.columns]
    final_df = final_df.drop(columns=cols_to_drop)
    
    return final_df

# Paths
input_path = '/Users/sakunika/Desktop/model/data/raw/vegetable_fruit_prices.csv'
output_path = '/Users/sakunika/Desktop/model/data/processed/mangosteen_prices_2023_2025.csv'

try:
    final_mangosteen_df = get_preprocessed_mangosteen_2023_2025(input_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_mangosteen_df.to_csv(output_path, index=False)
    print(f"Success! Saved with city names to {output_path}")
    print("Columns available:", final_mangosteen_df.columns.tolist())
except Exception as e:
    print(f"Error: {e}")