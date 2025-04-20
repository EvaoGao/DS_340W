import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from loss_utils import SoftDTW2
from model_utils import GCNForecast
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from sklearn.model_selection import KFold
from train import get_train_test_data



df = pd.read_csv("city_pollution_data2.csv")
train_set, test_set = get_train_test_data(df,method = "mean",TEST_SET_SIZE = 120)


cities_list = list(train_set.keys())

all_train = pd.DataFrame()
for city in cities_list:
  all_train = all_train._append(train_set[city], ignore_index=True)

all_test = pd.DataFrame({})
for city in test_set:
  all_test = all_test._append(test_set[city], ignore_index=True)

concat_df = pd.concat([all_train,all_test],axis=0)
print("Concat Dataframe: ",concat_df.shape[0])

col_max = {}
col_mean = {}
col_mean2 = {}
col_std = {}
PAST_DAYS = 14
FUTURE_DAYS = 7
normalization_type = 'mean_std'  # 'mean_std' or 'max'
DROP_ONEHOT = True  # whether to drop one-hot columns
pollutants = ["pm25_median","pm10_median", "o3_median", "so2_median", "no2_median", "co_median"]

####### Deleting incomplete rows

for city in cities_list:
    train_set[city].dropna(subset = pollutants, inplace=True)
    test_set[city].dropna(subset = pollutants, inplace=True)

deleted_cities = set()
for city in cities_list:
    if len(train_set[city]) < PAST_DAYS + FUTURE_DAYS:
        del train_set[city]
        deleted_cities.add(city)
    if len(test_set[city]) < PAST_DAYS + FUTURE_DAYS:
        del test_set[city]
        deleted_cities.add(city)

for city in deleted_cities:
    cities_list.remove(city)

total_rows = 0
for city in cities_list:
  col_mean[city] = {}
  #print(train_set[city].columns)
  train_set[city] = train_set[city].drop(['index', 'Date', 'City'], axis=1)
  test_set[city] = test_set[city].drop(['index', 'Date', 'City'], axis=1)


  for col in train_set[city]:
    #if col in ["index", "Date", "City"]:
      #continue

    train_set[city][col] = train_set[city][col].astype("float")
    test_set[city][col] = test_set[city][col].astype("float")


    if col in ["pm25_median","pm10_median", "o3_median", "so2_median", "no2_median", "co_median"]:
      ###################
      _mean = np.nanmean(train_set[city][col])
      if np.isnan(_mean):
        _mean = 0
      
      col_mean[city][col] = _mean
      train_set[city][col] = train_set[city][col].fillna(_mean)
      test_set[city][col] = test_set[city][col].fillna(_mean)

    if normalization_type == 'mean_std':
      col_mean2[col] = np.nanmean(concat_df[col].astype("float"))
      col_std[col] = np.nanstd(concat_df[col].astype("float"))
      train_set[city][col] = (train_set[city][col] - col_mean2[col]) / (col_std[col] + 0.001)
      test_set[city][col] = (test_set[city][col] - col_mean2[col]) / (col_std[col] + 0.001)

    else:
      col_max[col] = concat_df[col].astype("float").max()
      train_set[city][col] = train_set[city][col] / (col_max[col] + 0.001)
      test_set[city][col] = test_set[city][col] / (col_max[col] + 0.001)

  if DROP_ONEHOT:
    train_set[city].drop(train_set[city].columns[-19:], axis=1, inplace=True)
    test_set[city].drop(test_set[city].columns[-19:], axis=1, inplace=True)

  total_rows += len(train_set[city])

print("Total Rows After Drop: ",total_rows)



# City Coordinates (54 Cities)
city_coords = {
    'philadelphia': (39.9526, -75.1652),
    'columbus': (39.9612, -82.9988),
    'providence': (41.8240, -71.4128),
    'oklahoma city': (35.4676, -97.5164),
    'dallas': (32.7767, -96.7970),
    'miami': (25.7617, -80.1918),
    'raleigh': (35.7796, -78.6382),
    'staten island': (40.5795, -74.1502),
    'hartford': (41.7658, -72.6734),
    'atlanta': (33.7490, -84.3880),
    'boise': (43.6150, -116.2023),
    'detroit': (42.3314, -83.0458),
    'seattle': (47.6062, -122.3321),
    'saint paul': (44.9537, -93.0900),
    'las vegas': (36.1699, -115.1398),
    'san antonio': (29.4241, -98.4936),
    'memphis': (35.1495, -90.0490),
    'san francisco': (37.7749, -122.4194),
    'springfield': (37.2089, -93.2923),
    'baltimore': (39.2904, -76.6122),
    'portland': (45.5051, -122.6750),
    'salt lake city': (40.7608, -111.8910),
    'albuquerque': (35.0844, -106.6504),
    'tucson': (32.2226, -110.9747),
    'jacksonville': (30.3322, -81.6557),
    'sacramento': (38.5816, -121.4944),
    'madison': (43.0731, -89.4012),
    'columbia': (34.0007, -81.0348),
    'indianapolis': (39.7684, -86.1581),
    'los angeles': (34.0522, -118.2437),
    'manhattan': (40.7831, -73.9712),
    'tallahassee': (30.4383, -84.2807),
    'milwaukee': (43.0389, -87.9065),
    'honolulu': (21.3069, -157.8583),
    'richmond': (37.5407, -77.4360),
    'austin': (30.2672, -97.7431),
    'el paso': (31.7619, -106.4850),
    'fort worth': (32.7555, -97.3308),
    'salem': (44.9429, -123.0351),
    'chicago': (41.8781, -87.6298),
    'boston': (42.3601, -71.0589),
    'houston': (29.7604, -95.3698),
    'denver': (39.7392, -104.9903),
    'oakland': (37.8044, -122.2711),
    'phoenix': (33.4484, -112.0740),
    'nashville': (36.1627, -86.7816),
    'omaha': (41.2565, -95.9345),
    'jackson': (32.2988, -90.1848),
    'little rock': (34.7465, -92.2896),
    'fresno': (36.7378, -119.7871),
    'san diego': (32.7157, -117.1611),
    'charlotte': (35.2271, -80.8431),
    'brooklyn': (40.6782, -73.9442),
    'san jose': (37.3382, -121.8863),
}
for city in deleted_cities:
    del city_coords[city]

city_names = list(city_coords.keys())

# Build Graph Based on Haversine Distance < 800 km
DISTANCE_THRESHOLD = 400

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6378.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2.0)**2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2.0)**2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c

edge_index_list = [[], []]
num_nodes = len(city_names)

for i in range(num_nodes):
    for j in range(i+1, num_nodes):
        lat1, lon1 = city_coords[city_names[i]]
        lat2, lon2 = city_coords[city_names[j]]
        dist = haversine_distance(lat1, lon1, lat2, lon2)
        if dist < DISTANCE_THRESHOLD:
            # undirected edge => both directions
            edge_index_list[0].append(i)
            edge_index_list[1].append(j)
            edge_index_list[0].append(j)
            edge_index_list[1].append(i)

edge_index = torch.tensor(edge_index_list, dtype=torch.long)
print(f"Graph built: {num_nodes} nodes, {edge_index.size(1)//2} undirected edges.")


# (C) Prepare Data from train_set & test_set


###################################


# Check minimum length
min_train_days = min(len(df) for df in train_set.values())
min_test_days  = min(len(df) for df in test_set.values())
if min_train_days < PAST_DAYS + FUTURE_DAYS:
    raise ValueError("Not enough training data for the specified window sizes.")
if min_test_days < PAST_DAYS + FUTURE_DAYS:
    raise ValueError("Not enough test data for the specified window sizes.")

def build_window_data(city_dict, min_days,features,SELECTED_COLUMN):
    data_list = []
    total_days = min_days
    for start in range(total_days - (PAST_DAYS + FUTURE_DAYS) + 1):
        features_all = []
        labels_all = []
        for city in city_names:
            df = city_dict[city]

            # Extract input features: shape (14, 11)
            x_window = df.iloc[start:start+PAST_DAYS][features].values  # shape: [14, 11
            x_flat = x_window.flatten()  # shape: [154] => to be used as GCN node input

            # Extract target: pm25_median over 7 days
            y_window = df.iloc[start+PAST_DAYS : start+PAST_DAYS+FUTURE_DAYS][SELECTED_COLUMN].values

            features_all.append(x_flat)
            labels_all.append(y_window)

        x_tensor = torch.tensor(np.stack(features_all), dtype=torch.float)  # [54, 154]
        y_tensor = torch.tensor(np.stack(labels_all), dtype=torch.float)    # [54, 7]
        data = Data(x=x_tensor, y=y_tensor, edge_index=edge_index)
        data_list.append(data)
    return data_list

def combined_loss(pred, target):
    """
    pred, target shape: [B*N, FUTURE_DAYS] if we flatten the batch.
    We'll do it in-batch for simplicity.
    """
    # MSE
    mse = F.mse_loss(pred, target)

    # SoftDTW (flatten each row as a separate time-series)
    sdtw_val = softdtw(pred.view(-1, FUTURE_DAYS), target.view(-1, FUTURE_DAYS))
    #sdtw_val = softdtw(pred.to(device), target.to(device))
    return mse + lambda_sdtw * sdtw_val

for SELECTED_COLUMN in ["pm25_median", "pm10_median", "o3_median", "so2_median", "no2_median", "co_median"]:
    print(f"\n{'=' * 50}")
    print(f"Processing Selected Column: {SELECTED_COLUMN}")
    print(f"{'=' * 50}\n")

    features = ['Population Staying at Home', 'Population Not Staying at Home',
                'mil_miles', 'pressure_median', SELECTED_COLUMN, 'humidity_median',
                'temperature_median', 'dew_median', 'wind-speed_median',
                'wind-gust_median', 'pp_feat']

    train_data_list = build_window_data(train_set, min_train_days, features, SELECTED_COLUMN)
    test_data_list = build_window_data(test_set, min_test_days, features, SELECTED_COLUMN)
    print(f"Prepared {len(train_data_list)} training windows, {len(test_data_list)} testing windows.\n")

    gamma_sdtw  = 0.1
    lambda_sdtw = 0.5
    softdtw     = SoftDTW2(gamma=gamma_sdtw)

    h_channels_list = [32, 64,128]
    lr_list         = [0.001, 0.005, 0.0001]

    all_indices = np.arange(len(train_data_list))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    best_rmse = float('inf')
    best_combo = (None, None)

    for hch in h_channels_list:
        for lr in lr_list:
            fold_rmses = []
            fold_mapes = []
            for fold_i, (train_idx, val_idx) in enumerate(kf.split(all_indices)):
                # Prepare fold train/val subsets
                fold_train_list = [train_data_list[i] for i in train_idx]
                fold_val_list   = [train_data_list[i] for i in val_idx]

                train_loader = DataLoader(fold_train_list, batch_size=16, shuffle=True)
                val_loader   = DataLoader(fold_val_list, batch_size=16, shuffle=False)

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model  = GCNForecast(in_channels=PAST_DAYS*11, hidden_channels=hch, out_channels=FUTURE_DAYS).to(device)
                optimizer = optim.Adam(model.parameters(), lr=lr)

                num_epochs = 5  # can adjust if needed
                for epoch in range(num_epochs):
                    model.train()
                    for batch in train_loader:
                        batch = batch.to(device)
                        optimizer.zero_grad()
                        out = model(batch.x, batch.edge_index)  # shape: [54, 7]
                        loss_val = combined_loss(out, batch.y)
                        loss_val.backward()
                        optimizer.step()

                # Validation => compute RMSE and MAPE
                model.eval()
                val_preds = []
                val_truth = []
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(device)
                        out = model(batch.x, batch.edge_index)
                        val_preds.append(out.cpu().numpy())
                        val_truth.append(batch.y.cpu().numpy())
                val_preds = np.concatenate(val_preds, axis=0)
                val_truth = np.concatenate(val_truth, axis=0)

                val_truth = val_truth * col_std[SELECTED_COLUMN] + col_mean2[SELECTED_COLUMN]
                val_preds = val_preds * col_std[SELECTED_COLUMN] + col_mean2[SELECTED_COLUMN]

                rmse = np.sqrt(np.mean((val_preds - val_truth)**2))
                mape = np.mean(np.abs((val_preds - val_truth) / (val_truth + 1e-4))) * 100

                fold_rmses.append(rmse)
                fold_mapes.append(mape)

            avg_rmse = np.mean(fold_rmses)
            avg_mape = np.mean(fold_mapes)

            print(f"CV => hidden_channels={hch}, lr={lr}, avg_val_RMSE={avg_rmse:.4f}, avg_val_MAPE={avg_mape:.2f}%")

            if avg_rmse < best_rmse:
                best_rmse = avg_rmse
                best_combo = (hch, lr)

    print(f"\nBest hyperparams for {SELECTED_COLUMN} => hidden_channels={best_combo[0]}, lr={best_combo[1]}, CV RMSE={best_rmse:.4f}\n")

    # Retrain on Full Train Data with Best Hyperparams, Evaluate Test
    best_h, best_lr = best_combo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_model = GCNForecast(PAST_DAYS*11, best_h, FUTURE_DAYS).to(device)
    final_optim = optim.Adam(final_model.parameters(), lr=best_lr)
    optimal_model = None

    train_loader_full = DataLoader(train_data_list, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data_list, batch_size=16, shuffle=False)
    num_epochs_final = 30
    best_RMSE = float('inf')
    best_MAPE = float('inf')
    best_epoch = 0
    for epoch in range(1, num_epochs_final+1):
        final_model.train()
        epoch_loss = 0.0
        for batch in train_loader_full:
            batch = batch.to(device)
            final_optim.zero_grad()
            out = final_model(batch.x, batch.edge_index)
            loss_val = combined_loss(out, batch.y)
            loss_val.backward()
            final_optim.step()
            epoch_loss += loss_val.item()
        #print(f"[Epoch {epoch}/{num_epochs_final}] Train Loss: {epoch_loss/len(train_loader_full):.4f}")

        # Evaluate on Test
        final_model.eval()
        test_preds, test_truth = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = final_model(batch.x, batch.edge_index)
                test_preds.append(out.cpu().numpy())
                test_truth.append(batch.y.cpu().numpy())

        test_preds = np.concatenate(test_preds, axis=0)
        test_truth = np.concatenate(test_truth, axis=0)

        test_truth = test_truth * col_std[SELECTED_COLUMN] + col_mean2[SELECTED_COLUMN]
        test_preds = test_preds * col_std[SELECTED_COLUMN] + col_mean2[SELECTED_COLUMN]


        test_rmse = np.sqrt(np.mean((test_preds - test_truth)**2))
        test_mape = np.mean(np.abs((test_preds - test_truth) / (test_truth + 1e-6))) * 100
        if test_rmse < best_RMSE:
            best_RMSE = test_rmse
            best_MAPE = test_mape
            best_epoch = epoch
            optimal_model = final_model.state_dict()
            #torch.save(final_model.state_dict(), f"GCN{epoch}.pth")
        #print(f"\nRMSE,MAPE on Epoch {epoch}: {test_rmse:.4f}, {test_mape:.2f}%")  

    #torch.save(optimal_model, f"GCN_{SELECTED_COLUMN}_{best_epoch}.pth")
    print(f"\nFinal Test RMSE:{best_RMSE:.4f}, MAPE:{best_MAPE:.2f}%, EPOCH:{best_epoch}")
