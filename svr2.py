import pandas as pd
import numpy as np
import matplotlib as plt
from datetime import datetime
import random

from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

from train import get_train_test_data
from joblib import dump
from joblib import load

print("NEWWWWWWWWWWWWWWWWWWWWWWWWW")
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Load Data
df = pd.read_csv("city_pollution_data2.csv")
train_set_dict, test_set_dict = get_train_test_data(df)

train_frames = []
test_frames = []
for city in train_set_dict.keys():
    train_frames.append(train_set_dict[city])
    test_frames.append(test_set_dict[city])

df_train_all = pd.concat(train_frames, axis=0).reset_index(drop=True)
df_test_all  = pd.concat(test_frames, axis=0).reset_index(drop=True)

print(f"First 5 rows:")
print(df_train_all["Population Staying at Home"].head())

# Pollutants & parameters
pollutants = [
    "pm25_median", 
    "pm10_median", 
    "o3_median", 
    "no2_median", 
    "so2_median", 
    "co_median"
]

HORIZON   = 7    # Next 7 days
LAG_DAYS  = 14   # Past 14 days input

# Create 14-lag + 7-future-day columns
def create_14lag_7future(df, pollutant):
    df = df.sort_values(["City", "Date"]).copy()
    # 14-lag features
    for i in range(1, LAG_DAYS + 1):
        df[f"{pollutant}_lag_{i}"] = df.groupby("City")[pollutant].shift(i)
    # 7 future days as targets
    for d in range(1, HORIZON + 1):
        df[f"{pollutant}_target_{d}"] = df.groupby("City")[pollutant].shift(-d)
    return df


exogenous_cols = [
    "Population Staying at Home",
    "Population Not Staying at Home",
    "humidity_median",
    "temperature_median",
    "dew_median",
    "wind-speed_median",
    "mil_miles",
    "wind-gust_median",
    "pressure_median",
    "pp_feat"
]


# Missing Data Approaches
def drop_incomplete_rows(df, needed_cols):
    return df.dropna(subset=needed_cols+pollutants)
'''
def impute_mean(df, needed_cols):
    for c in needed_cols:
        #print(f"First 5 rows of column {c}:")
        #print(df[c].head())
        df[c] = df[c].fillna(df[c].mean())
    return df

def impute_median(df, needed_cols):
    for c in needed_cols:
        df[c] = df[c].fillna(df[c].median())
    return df
'''
def impute_mean(df, needed_cols):
    for c in needed_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        mean = np.nanmean(df[c])
        df[c] = df[c].fillna(mean)
    return df

def impute_median(df, needed_cols):
    for c in needed_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        mean = np.nanmedian(df[c])
        df[c] = df[c].fillna(mean)
    return df

# Train function with single average RMSE output
def train_for_pollutant(p, df_train, df_test, approach_label, fill_func):
    print(f"\n=== Pollutant: {p}, Approach: {approach_label} ===")
    
    # Build 14-lag & 7-target columns
    df_train_p = create_14lag_7future(df_train, p)
    df_test_p  = create_14lag_7future(df_test, p)

    lag_cols    = [f"{p}_lag_{i}" for i in range(1, LAG_DAYS+1)]
    target_cols = [f"{p}_target_{d}" for d in range(1, HORIZON+1)]
    needed_cols = exogenous_cols + lag_cols + target_cols

    # Apply missing-data strategy
    df_train_p = fill_func(df_train_p, needed_cols)
    df_train_p = df_train_p.dropna(subset=needed_cols)

    df_test_p = fill_func(df_test_p, needed_cols)
    df_test_p = df_test_p.dropna(subset=needed_cols)

    # X, Y for train & test
    X_train = df_train_p[exogenous_cols + lag_cols]
    Y_train = df_train_p[target_cols]

    X_test  = df_test_p[exogenous_cols + lag_cols]
    Y_test  = df_test_p[target_cols]

    print("Train shape:", X_train.shape, Y_train.shape, 
          "Test shape:", X_test.shape,  Y_test.shape)

    # Scale X
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled  = scaler_X.transform(X_test)

    # Scale Y
    scaler_Y = StandardScaler()
    Y_train_scaled = scaler_Y.fit_transform(Y_train)

    # MultiOutput SVR: 7-day forecast
    svr = SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale')
    model = MultiOutputRegressor(svr)

    #Fit & predict
    model.fit(X_train_scaled, Y_train_scaled)
    Y_pred_scaled = model.predict(X_test_scaled)
    Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)

    # Single average RMSE across 7 days
    rmse_list = []
    for day_idx in range(HORIZON):
        true_day = Y_test.iloc[:, day_idx].values  # e.g. p_target_1
        pred_day = Y_pred[:, day_idx]
        day_rmse = root_mean_squared_error(true_day, pred_day)
        rmse_list.append(day_rmse)

    avg_rmse = np.mean(rmse_list)
    #dump(model, f"SVR_{p}.joblib")
    print(f"Average 7-day RMSE = {avg_rmse:.3f}")

# Main: loop over 3 strategies × 6 pollutants
Methods = [
    ("Deleting Rows",   drop_incomplete_rows),
    ("Mean Imputation", impute_mean),
    ("Median Imputation", impute_median)
]
Methods2 = [
    ("Mean Imputation", impute_mean)
]
for approach_label, fill_func in Methods2:
    print(f"\n## Approach: {approach_label} ##")
    for p in pollutants:
        train_for_pollutant(p, df_train_all, df_test_all, 
                            approach_label, fill_func)
