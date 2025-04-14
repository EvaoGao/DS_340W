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


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

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


forecast_horizon = 7  # e.g., next 7 days


def data_filters(df, columns):
    """Drop rows with missing values in any of these columns."""
    df = df.dropna(subset=columns)
    return df

def create_shifted_targets_naive(df, pollutants, horizon=1):
    """Shift pollutants by `horizon` days to create naive next-day (or next-week) targets."""
    df = df.sort_values(["City", "Date"]).copy()
    for p in pollutants:
        df[p + "_target"] = df.groupby("City")[p].shift(-horizon)
    df = data_filters(df, [p + "_target" for p in pollutants])
    return df

def create_shifted_targets_imputations(df, pollutants, horizon=1):
    """Shift pollutants by `horizon` and impute missing values by city-wise mean."""
    df = df.sort_values(["City", "Date"]).copy()
    for p in pollutants:
        df[p + "_target"] = df.groupby("City")[p].shift(-horizon)
    target_cols = [p + "_target" for p in pollutants]
    # Impute by city-wise mean
    df[target_cols] = df.groupby("City")[target_cols].transform(lambda x: x.fillna(x.median()))
    return df

def data_imputations(df, columns):
    """Simple mean imputation."""
    df[columns] = df[columns].fillna(df[columns].median())
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

# features
exogenous_cols = [
    "Population Staying at Home", 
    "Population Not Staying at Home",
    "humidity_median",
    "temperature_median",
    "dew_median",
    "wind-speed_median",
    "mil_miles", #Traffic
    "wind-gust_median",
    "pressure_median",
    "pp_feat"
]


target_cols = [p + "_target" for p in pollutants]

df_train_all_naive = data_filters(df_train_all_naive, exogenous_cols + target_cols)
df_test_all_naive  = data_filters(df_test_all_naive, exogenous_cols + target_cols)

X_train_naive = df_train_all_naive[exogenous_cols]
Y_train_naive = df_train_all_naive[target_cols]

X_test_naive = df_test_all_naive[exogenous_cols]
Y_test_naive = df_test_all_naive[target_cols]

print("Shapes naive approach:")
print("  X_train:", X_train_naive.shape)
print("  Y_train:", Y_train_naive.shape)
print("  X_test :", X_test_naive.shape)
print("  Y_test :", Y_test_naive.shape)


# Train SVR
svr_rbf = SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale')
model_naive = MultiOutputRegressor(svr_rbf)

# Create separate scalers for X and Y
scaler_X_naive = StandardScaler()
scaler_Y_naive = StandardScaler()

# Fit on training data
X_train_naive_scaled = scaler_X_naive.fit_transform(X_train_naive)
Y_train_naive_scaled = scaler_Y_naive.fit_transform(Y_train_naive)

# Train model on scaled data
model_naive.fit(X_train_naive_scaled, Y_train_naive_scaled)

# Predict on test set
X_test_naive_scaled = scaler_X_naive.transform(X_test_naive)
Y_pred_naive_scaled = model_naive.predict(X_test_naive_scaled)

# Denormalize (inverse-transform) predictions back to original scale
Y_pred_naive = scaler_Y_naive.inverse_transform(Y_pred_naive_scaled)

# Convert to DataFrame for convenience
Y_pred_naive_df = pd.DataFrame(
    data=Y_pred_naive, 
    columns=target_cols, 
    index=X_test_naive.index
)


print("\nResults WITHOUT imputations (denormalized):")
for i, p in enumerate(pollutants):
    # The corresponding target column name
    col_name = p + "_target"
    true_vals = Y_test_naive[col_name].values
    pred_vals = Y_pred_naive_df[col_name].values

    rmse = root_mean_squared_error(true_vals, pred_vals)
    mape = mean_absolute_percentage_error(true_vals, pred_vals)
    r2   = r2_score(true_vals, pred_vals)

    print(f"Pollutant: {p}")
    print(f"  RMSE = {rmse:.3f}")
    print(f"  MAPE = {mape:.3f}")
    print(f"  R²   = {r2:.3f}\n")

# Repeat the same procedure for the data-with-imputations
df_train_all_imp = data_imputations(df_train_all_imp, exogenous_cols + target_cols)
df_test_all_imp  = data_imputations(df_test_all_imp, exogenous_cols + target_cols)

X_train_imp = df_train_all_imp[exogenous_cols]
Y_train_imp = df_train_all_imp[target_cols]

X_test_imp = df_test_all_imp[exogenous_cols]
Y_test_imp = df_test_all_imp[target_cols]

# Create new scalers
scaler_X_imp = StandardScaler()
scaler_Y_imp = StandardScaler()

X_train_imp_scaled = scaler_X_imp.fit_transform(X_train_imp)
Y_train_imp_scaled = scaler_Y_imp.fit_transform(Y_train_imp)

model_imp = MultiOutputRegressor(svr_rbf)
model_imp.fit(X_train_imp_scaled, Y_train_imp_scaled)

X_test_imp_scaled = scaler_X_imp.transform(X_test_imp)
Y_pred_imp_scaled = model_imp.predict(X_test_imp_scaled)

# Inverse-transform predictions
Y_pred_imp = scaler_Y_imp.inverse_transform(Y_pred_imp_scaled)
Y_pred_imp_df = pd.DataFrame(Y_pred_imp, columns=target_cols, index=X_test_imp.index)

print("\nResults WITH imputations (denormalized):")
for i, p in enumerate(pollutants):
    col_name = p + "_target"
    true_vals = Y_test_imp[col_name].values
    pred_vals = Y_pred_imp_df[col_name].values

    rmse = root_mean_squared_error(true_vals, pred_vals)
    mape = mean_absolute_percentage_error(true_vals, pred_vals)
    r2   = r2_score(true_vals, pred_vals)

    print(f"Pollutant: {p}")
    print(f"  RMSE = {rmse:.3f}")
    print(f"  MAPE = {mape:.3f}")
    print(f"  R²   = {r2:.3f}\n")
