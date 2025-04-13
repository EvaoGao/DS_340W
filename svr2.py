import pandas as pd
import numpy as np
import matplotlib as plt
from datetime import datetime
import random

from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error,mean_squared_error, mean_absolute_percentage_error,r2_score

from train import get_train_test_data

df = pd.read_csv("city_pollution_data2.csv")
train_set_dict, test_set_dict = get_train_test_data(df)

train_frames = []
test_frames = []

for city in train_set_dict.keys():
    train_frames.append(train_set_dict[city])
    test_frames.append(test_set_dict[city])

df_train_all = pd.concat(train_frames, axis=0).reset_index(drop=True)
df_test_all = pd.concat(test_frames, axis=0).reset_index(drop=True)

pollutants = [
    "pm25_median", 
    "pm10_median", 
    "o3_median", 
    "no2_median", 
    "so2_median", 
    "co_median"
]

#choose forecasting length
forecast_horizon = 7  # ex: 1 = next-day forecast

def create_shifted_targets_naive(df, pollutants, horizon=1):
    # Sort by city and date so shift is correct
    df = df.sort_values(["City","Date"]).copy()
    
    for p in pollutants:
        df[p + "_target"] = df.groupby("City")[p].shift(-horizon)
    df = data_filters(df, [p + "_target" for p in pollutants])
    
    return df

def create_shifted_targets_imputations(df, pollutants, horizon=1):
    # Sort by city and date so that shift is correct
    df = df.sort_values(["City", "Date"]).copy()
    
    for p in pollutants:
        df[p + "_target"] = df.groupby("City")[p].shift(-horizon)
    
    target_cols = [p + "_target" for p in pollutants]
    
    # Replace missing values with the median of each column per city
    df[target_cols] = df.groupby("City")[target_cols].transform(lambda x: x.fillna(x.mean()))
    
    return df

def data_imputations(df, columns):
    df[columns] = df[columns].fillna(df[columns].mean())
    return df

def data_filters(df,columns): #Deleting rows with missing values
    df = df.dropna(subset=columns)
    return df



print("Without Imputations:")
print(f"Training size:{df_train_all.shape[0]}, Testing size:{df_test_all.shape[0]}\n")
df_train_all_naive = create_shifted_targets_naive(df_train_all, pollutants, forecast_horizon)
df_test_all_naive  = create_shifted_targets_naive(df_test_all, pollutants, forecast_horizon)
print(f"Training size:{df_train_all_naive.shape[0]}, Testing size:{df_test_all_naive.shape[0]}\n")

print("With Imputations:")
print(f"Training size:{df_train_all.shape[0]}, Testing size:{df_test_all.shape[0]}\n")
df_train_all_imp = create_shifted_targets_imputations(df_train_all, pollutants, forecast_horizon)
df_test_all_imp  = create_shifted_targets_imputations(df_test_all, pollutants, forecast_horizon)
print(f"Training size:{df_train_all_imp.shape[0]}, Testing size:{df_test_all_imp.shape[0]}\n")

feature_cols = [
    "Population Staying at Home", 
    "Population Not Staying at Home",
    "humidity_median",
    "temperature_median",
    "dew_median",
    "wind-speed_median",
    "mil_miles", #Traffic,
    "wind-gust_median",
    "pressure_median"
]

target_cols = [p + "_target" for p in pollutants]

print(f"Before filtering: {df_train_all_naive.shape[0]}")
df_train_all_naive = data_filters(df_train_all_naive, feature_cols + target_cols)
df_test_all_naive  = data_filters(df_test_all_naive, feature_cols + target_cols)
print(f"After filtering: {df_train_all_naive.shape[0]}")

X_train_naive = df_train_all_naive[feature_cols]
Y_train_naive = df_train_all_naive[target_cols]

X_test_naive = df_test_all_naive[feature_cols]
Y_test_naive = df_test_all_naive[target_cols]

svr_rbf = SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale')

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", MultiOutputRegressor(svr_rbf))
])

pipeline.fit(X_train_naive, Y_train_naive)
Y_pred = pipeline.predict(X_test_naive)

Y_pred_df = pd.DataFrame(
    data=Y_pred, 
    columns=target_cols, 
    index=X_test_naive.index
)

for i, p in enumerate(pollutants):
    true_vals = Y_test_naive[p + "_target"]
    pred_vals = Y_pred_df[p + "_target"]
    
    rmse = root_mean_squared_error(true_vals, pred_vals)
    mape = mean_absolute_percentage_error(true_vals, pred_vals)
    r2  = r2_score(true_vals, pred_vals)
    
    print("Results without imputations:")
    print(f"Pollutant: {p}")
    print(f"  RMSE = {rmse:.3f}")
    print(f"  MAPE = {mape:.3f}")
    print(f"  R²  = {r2:.3f}\n")



#SVR with data imputations

df_train_all_imp = data_imputations(df_train_all_imp, feature_cols + target_cols)
df_test_all_imp  = data_imputations(df_test_all_imp, feature_cols + target_cols)

X_train_imp = df_train_all_imp[feature_cols]
Y_train_imp = df_train_all_imp[target_cols]

X_test_imp = df_test_all_imp[feature_cols]
Y_test_imp = df_test_all_imp[target_cols]

pipeline.fit(X_train_imp, Y_train_imp)
Y_pred = pipeline.predict(X_test_imp)

Y_pred_df = pd.DataFrame(
    data=Y_pred, 
    columns=target_cols, 
    index=X_test_imp.index
)

for i, p in enumerate(pollutants):
    true_vals = Y_test_imp[p + "_target"]
    pred_vals = Y_pred_df[p + "_target"]
    
    rmse = root_mean_squared_error(true_vals, pred_vals)
    mape = mean_absolute_percentage_error(true_vals, pred_vals)
    r2  = r2_score(true_vals, pred_vals)
    
    print("Results with imputations")
    print(f"Pollutant: {p}")
    print(f"  RMSE = {rmse:.3f}")
    print(f"  MAPE = {mape:.3f}")
    print(f"  R²  = {r2:.3f}\n")
