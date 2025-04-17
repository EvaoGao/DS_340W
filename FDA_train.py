# fda_multifeature.py

import numpy as np
np.float_ = np.float64
import pandas as pd
import random
from datetime import datetime
from sklearn.metrics import mean_squared_error
from skfda.representation.grid import FDataGrid
from skfda.ml.regression import LinearRegression
from skfda.representation.basis import BSpline
from skfda.representation.basis import BSplineBasis, VectorValuedBasis
# ---------------- Constants ----------------
exogenous_cols = [
    "Population Staying at Home", "Population Not Staying at Home", "humidity_median",
    "temperature_median", "dew_median", "wind-speed_median", "mil_miles",
    "wind-gust_median", "pressure_median", "pp_feat"
]
pollutants = ["pm25_median", "pm10_median", "o3_median", "so2_median", "no2_median", "co_median"]

# ---------------- Helper Function ----------------
def get_train_test_data(df):
    df["Population Staying at Home"] = df["Population Staying at Home"].str.replace(",", "")
    df["Population Not Staying at Home"] = df["Population Not Staying at Home"].str.replace(",", "")
    df["weekday"] = df["Date"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d").weekday())
    df["month"] = df["Date"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d").month - 1)
    df = df.join(pd.get_dummies(df.pop('weekday'), prefix='day'))
    df = df.join(pd.get_dummies(df.pop('month'), prefix='month'))

    for col in df.columns:
        for x in ["min", "max", "count", "County", "past_week", "latitude", "longitude", "State", "variance"]:
            if x in col:
                df.drop([col], axis=1, inplace=True)

    cities_list = list(set(df['City']))
    city_df, test_set, train_set = {}, {}, {}
    TEST_SET_SIZE = 60

    for city in cities_list:
        city_df[city] = df[df['City'] == city].sort_values('Date').reset_index()
        for col in city_df[city].columns:
            if col in pollutants:
                continue
            try:
                _mean = np.nanmean(city_df[city][col].astype(float))
                #city_df[city][col].fillna(_mean if not np.isnan(_mean) else 0, inplace=True)
                city_df[city][col] = city_df[city][col].fillna(_mean if not np.isnan(_mean) else 0)

            except:
                pass

        random.seed(0)
        if city_df[city].shape[0] < TEST_SET_SIZE + 21:
            continue
        start = random.randint(0, city_df[city].shape[0] - TEST_SET_SIZE)
        test_set[city] = city_df[city].iloc[start:start + TEST_SET_SIZE]
        train_set[city] = city_df[city].drop(index=list(range(start, start + TEST_SET_SIZE)))

    return train_set, test_set

# ---------------- Load Data ----------------
df = pd.read_csv("city_pollution_data2.csv")
train_set, test_set = get_train_test_data(df)
cities_list = list(train_set.keys())
all_train = pd.concat([train_set[city] for city in cities_list])
all_test = pd.concat([test_set[city] for city in cities_list])
concat_df = pd.concat([all_train, all_test])

col_mean2 = {col: np.nanmean(concat_df[col].astype("float")) for col in pollutants}
col_std = {col: np.nanstd(concat_df[col].astype("float")) for col in pollutants}

# ---------------- FDA Model ----------------
print("\n--- FDA RMSE Results ---")
for pollutant in pollutants:
    feature_cols = exogenous_cols + [pollutant]
    X_all, Y_all = [], []

    for city in cities_list:
        df_train = train_set[city].copy()
        df_train[pollutant] = df_train[pollutant].astype(float)
        #df_train[pollutant].fillna(df_train[pollutant].mean(), inplace=True)
        df_train[pollutant] = df_train[pollutant].fillna(df_train[pollutant].mean())

        data = df_train[feature_cols].values
        series = df_train[pollutant].values

        for i in range(len(data) - 14 - 7):
            #x_seq = data[i:i+14].as        # shape: (14, 11)
            #y_seq = series[i+14:i+21]
            x_seq = data[i:i+14].astype(float)
            y_seq = series[i+14:i+21].astype(float)
            if not (np.any(np.isnan(x_seq)) or np.any(np.isnan(y_seq))):
                y_seq = (y_seq - col_mean2[pollutant]) / (col_std[pollutant] + 1e-3)
                X_all.append(x_seq)  # (14, 11)
                Y_all.append(y_seq)

    if len(X_all) == 0:
        continue

    X_all = np.array(X_all)  # shape: (N, 14, 11)
    Y_all = np.array(Y_all)

    #basis = BSpline(domain_range=(0, 14), n_basis=7)
    basis = BSplineBasis(domain_range=(0, 14), n_basis=7)
    vector_basis = VectorValuedBasis([basis] * 11)
    fd_X = FDataGrid(data_matrix=X_all, grid_points=np.arange(14))

    fd_basis_X = fd_X.to_basis(vector_basis)

    models = []
    for i in range(7):
        model = LinearRegression()
        model.fit(fd_basis_X, Y_all[:, i])
        models.append(model)

    # ---------------- Evaluation ----------------
    X_test, Y_test = [], []

    for city in cities_list:
        df_test = test_set[city].copy()
        df_test[pollutant] = df_test[pollutant].astype(float)
        #df_test[pollutant].fillna(df_test[pollutant].mean(), inplace=True)
        df_test[pollutant] = df_test[pollutant].fillna(df_test[pollutant].mean())

        data = df_test[feature_cols].values
        series = df_test[pollutant].values

        for i in range(len(data) - 14 - 7):
            x_seq = data[i:i+14].astype(np.float64)
            y_seq = series[i+14:i+21].astype(np.float64)
            if not (np.any(np.isnan(x_seq)) or np.any(np.isnan(y_seq))):
                y_seq = (y_seq - col_mean2[pollutant]) / (col_std[pollutant] + 1e-3)
                X_test.append(x_seq)  # shape: (14, 11)
                Y_test.append(y_seq)

    if len(X_test) == 0:
        continue

    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    fd_X_test = FDataGrid(data_matrix=X_test, grid_points=np.arange(14))
    fd_basis_X_test = fd_X_test.to_basis(vector_basis)

    Y_pred = [model.predict(fd_basis_X_test).reshape(-1, 1) for model in models]
    Y_pred = np.hstack(Y_pred)

    # denormalize
    mean, std = col_mean2[pollutant], col_std[pollutant]
    Y_test_denorm = Y_test * std + mean
    Y_pred_denorm = Y_pred * std + mean

    rmse = np.sqrt(mean_squared_error(Y_test_denorm, Y_pred_denorm))
    print(f"{pollutant}: RMSE = {rmse:.4f}")
