import pandas as pd
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from torch.autograd import Variable
from datetime import datetime
from tqdm import tqdm
import sklearn
from copy import deepcopy
from loss_utils import *
from model_utils import *
import torch.nn as nn
import math

device = 'cpu'

import warnings
warnings.filterwarnings('ignore')

# Data Pre-processing

df = pd.read_csv("city_pollution_data2.csv")

DROP_ONEHOT = True
SEQ_LENGTH = 7

if DROP_ONEHOT:
  INPUT_DIM = 10 
else:
  INPUT_DIM = 29

HIDDEN_DIM = 32
LAYER_DIM = 3


normalization_type = 'mean_std' # 'max', mean_std

def get_train_test_data(df,method = 'mean', TEST_SET_SIZE = 60):
  # we'll mostly need median and variance values of features for most of our needs

  for col in df.columns:
    for x in ["min", "max", "count", "County", "past_week", "latitude", "longitude", "State", "variance"]:
      if x in col:
        df.drop([col], axis=1, inplace=True)

  df["Population Staying at Home"] = df["Population Staying at Home"].apply(lambda x: x.replace(",", ""))
  df["Population Not Staying at Home"] = df["Population Not Staying at Home"].apply(lambda x: x.replace(",", ""))

  # Now we want 2 more features. Which day of week it is and which month it is.
  # Both of these will be one-hot and hence we'll add 7+12 = 19 more columns.
  # Getting month id is easy from the datetime column. 
  # For day of week, we'll use datetime library.
  
  #df['weekday'] = df['Date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d").weekday())
  #df['month'] = df['Date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d").month - 1)
  #df['weekday'] = df['Date'].apply(lambda x: datetime.strptime(x, "%Y/%m/%d").weekday())
  #df['month'] = df['Date'].apply(lambda x: datetime.strptime(x, "%Y/%m/%d").month - 1)
  df['weekday'] = df['Date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d").weekday())
  df['month'] = df['Date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d").month - 1)

  # using one-hot on month and weekday
  weekday_onehot = pd.get_dummies(df['weekday'])
  weekday_onehot.columns = ["day_"+str(x) for x in weekday_onehot]
  month_onehot = pd.get_dummies(df['month'])
  month_onehot.columns = ["month_"+str(x) for x in month_onehot]

  df.drop(['weekday', 'month'], axis=1, inplace=True)
  df = df.join([weekday_onehot, month_onehot])

  cities_list = list(set(df['City']))
  city_df = {}
  test_indices_of_cities = {}
  train_set = {}
  test_set = {}

  for city in cities_list:
    city_df[city] = df[df['City'] == city].sort_values('Date').reset_index()
    for col in city_df[city].columns:
      if col in ["pm25_median", "o3_median", "so2_median", "no2_median", "pm10_median", "co_median"]:
        continue
      try:  
        if method == 'mean':
          _mean = np.nanmean(city_df[city][col])
        elif method == "median":
          _mean = np.nanmedian(city_df[city][col])
        if np.isnan(_mean) == True:
          _mean = 0
        city_df[city][col] = city_df[city][col].fillna(_mean)
           
      except:
        pass
    
    random.seed(0)
    test_index_start = random.randint(0, city_df[city].shape[0] - TEST_SET_SIZE)
    test_indices_of_cities[city] = [test_index_start, test_index_start + TEST_SET_SIZE]

    test_set[city] = city_df[city].iloc[test_index_start:test_index_start + TEST_SET_SIZE]
    train_set[city] = city_df[city].drop(index=list(range(test_index_start, test_index_start + TEST_SET_SIZE)))

  return train_set, test_set

train_set, test_set = get_train_test_data(df)

cities_list = list(train_set.keys())

all_train = pd.DataFrame()
for city in cities_list:
  all_train = all_train._append(train_set[city], ignore_index=True)

all_test = pd.DataFrame({})
for city in test_set:
  all_test = all_test._append(test_set[city], ignore_index=True)

concat_df = pd.concat([all_train,all_test],axis=0)

# ---------------------------------------------------------------------------- #
col_max = {}
col_mean = {}
col_mean2 = {}
col_std = {}
'''
pollutants = ["pm25_median","pm10_median", "o3_median", "so2_median", "no2_median", "co_median"]
for city in cities_list:
    print(f"Training set {city} shape: {train_set[city].shape}")
    print(f"Testing set {city} shape: {test_set[city].shape}")
    train_set[city].dropna(subset = pollutants, inplace=True)
    test_set[city].dropna(subset = pollutants, inplace=True)
    print(f"Training set {city} shape: {train_set[city].shape}")
    print(f"Testing set {city} shape: {test_set[city].shape}\n")
'''


# data imputations
for city in cities_list:
  col_mean[city] = {}
  for col in train_set[city]:
    if col in ["index", "Date", "City"]:
      continue

    train_set[city][col] = train_set[city][col].astype("float")
    test_set[city][col] = test_set[city][col].astype("float")

    if col in ["pm25_median", "o3_median", "so2_median", "no2_median", "pm10_median", "co_median"]:
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

class CityDataP(torch.utils.data.Dataset):
  def __init__(self, selected_column, split):
    self.split = split
    if split == "train":
      self.dataset = train_set
    else:
      self.dataset = test_set

    self.valid_city_idx = 0
    self.valid_day_idx = 0
    self.selected_column = selected_column

  def __getitem__(self, idx):
    input_seq_len = 14   # use 14 days of data as input
    target_seq_len = 7   # predict the following 7 days

    if self.split != "train":
        # getting all data from the validation set
        window, city = self.get_idx_data(idx)
    else:
        # getting data randomly for train split
        city = random.choice(cities_list)
        _df = self.dataset[city]
        start_idx = random.randint(0, _df.shape[0] - input_seq_len - target_seq_len)
        window = _df.iloc[start_idx : start_idx + input_seq_len + target_seq_len]

    window = window.drop(['index', 'Date', 'City'], axis=1)

    Y_all = pd.DataFrame({})
    Y_all[self.selected_column] = window[self.selected_column]

    features = window.copy()
    for col in features.columns.tolist():
      if col == self.selected_column:
        continue
      elif col in ["pm25_median", "pm10_median", "o3_median", "so2_median", "no2_median", "co_median"]:
        features.drop(col, axis=1, inplace=True)
      else:
        features[col] = features[col].astype("float")

    X_input = features.iloc[:input_seq_len, :]
    Y_target = Y_all.iloc[input_seq_len:input_seq_len+target_seq_len, :]

    return X_input.values, Y_target.values, Y_all.values

  def get_idx_data(self, idx):
    city = cities_list[self.valid_city_idx]
    _df = self.dataset[city]
    total_seq = 14 + 7  # 21 days
    out = _df.iloc[self.valid_day_idx : self.valid_day_idx + total_seq]
    
    if self.valid_day_idx + total_seq >= _df.shape[0]:
        self.valid_day_idx = 0
        self.valid_city_idx += 1
        # Wrap around if index becomes out-of-range.
        if self.valid_city_idx >= len(cities_list):
            self.valid_city_idx = 0
    else:
        self.valid_day_idx += 1

    return out, city

  def __len__(self):
    input_seq_len = 14
    target_seq_len = 7
    total_seq = input_seq_len + target_seq_len  # 21 days
    if self.split != "train":
        # For example, if test data always has 61 rows per city.
        return (61 - total_seq + 1) * len(cities_list)
    else:
        # For training, you might sum over all cities. One option is:
        return sum([self.dataset[city].shape[0] - total_seq + 1 for city in cities_list])

class CityDataForecast(torch.utils.data.Dataset):
  def __init__(self, selected_column, split):
    self.split = split
    if split == "train":
      self.dataset = train_set
    else:
      self.dataset = test_set

    self.valid_city_idx = 0
    self.valid_day_idx = 0
    self.selected_column = selected_column

  def __getitem__(self, idx):
    input_seq_len = 14   # use 14 days of data as input
    target_seq_len = 7   # predict the following 7 days

    if self.split != "train":
        # getting all data from the validation set
        window, city = self.get_idx_data(idx)
    else:
        # getting data randomly for train split
        city = random.choice(cities_list)
        _df = self.dataset[city]
        start_idx = random.randint(0, _df.shape[0] - input_seq_len - target_seq_len)
        window = _df.iloc[start_idx : start_idx + input_seq_len + target_seq_len]

    window = window.drop(['index', 'Date', 'City'], axis=1)

    Y_all = pd.DataFrame({})
    Y_all[self.selected_column] = window[self.selected_column]

    features = window.copy()
    for col in features.columns.tolist():
      if col == self.selected_column:
        continue
      elif col in ["pm25_median", "pm10_median", "o3_median", "so2_median", "no2_median", "co_median"]:
        features.drop(col, axis=1, inplace=True)
      else:
        features[col] = features[col].astype("float")

    X_input = features.iloc[:input_seq_len, :]
    Y_target = Y_all.iloc[input_seq_len:input_seq_len+target_seq_len, :]

    return X_input.values, Y_target.values, Y_all.values

  def get_idx_data(self, idx):
    city = cities_list[self.valid_city_idx]
    _df = self.dataset[city]
    total_seq = 14 + 7  # input days + target days
    out = _df.iloc[self.valid_day_idx : self.valid_day_idx + total_seq]
    
    if self.valid_day_idx + total_seq >= _df.shape[0]:
        self.valid_day_idx = 0
        self.valid_city_idx += 1
    else:
        self.valid_day_idx += 1

    return out, city

  def __len__(self):
    if self.split != "train":
      return (61-SEQ_LENGTH)*len(cities_list)
    return len(all_train) - (SEQ_LENGTH - 1)*len(cities_list)

# function that implement the look_ahead mask for masking future time steps. 
def create_look_ahead_mask(size, device=device):
    mask = torch.ones((size, size), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask  # (size, size)
 
if __name__ == '__main__':

  dtw_loss = SoftDTW(use_cuda=False, gamma=0.1)
  lmbda = 0.5

  for SELECTED_COLUMN in ["pm25_median","pm10_median", "o3_median", "so2_median", "no2_median", "co_median"]: # ["pm25_median", "so2_median", "pm10_median", "no2_median", "o3_median", "co_median", "so2_median"]:
  #for SELECTED_COLUMN in ["so2_median", "no2_median", "co_median"]:   
      train_data = CityDataP(SELECTED_COLUMN, "train")
      val_data = CityDataP(SELECTED_COLUMN, "test")

      sampleLoader = DataLoader(train_data, 32, shuffle=True, num_workers=4)
      val_loader = DataLoader(val_data, 4096, shuffle=False, num_workers=4)

      lr = 0.00001
      n_epochs = 10
      RMSE_list = []
      MAPE_list = []

      criterion = nn.MSELoss()
      #criterion = CombinedLoss(mse_weight=1.0, sdtw_weight=0.5, gamma=0.1, normalize=False)
      #model = Transformer(num_layers=2, D=16, H=4, hidden_mlp_dim=16, inp_features=11, out_features=1, dropout_rate=0.5, attention_type='regular', SL=SEQ_LENGTH).to(device) # cosine_square, cosine, regular # 6L, 12H
      # model = TransLSTM(num_layers=3, D=16, H=5, hidden_mlp_dim=32, inp_features=11, out_features=1, dropout_rate=0.2, LSTM_module = LSTM(4, INPUT_DIM+1, HIDDEN_DIM, LAYER_DIM, bidirectional = False).to(device), attention_type='regular').to(device) # cosine_square, cosine, regular # 6L, 12H
      # model = LSTM(1, INPUT_DIM+1, HIDDEN_DIM, LAYER_DIM).cuda()
      #model = MultiHeadAttentionCosSquareformerNew(D=16, H=10).to(device)
      #model = MultiHeadAttentionCosSquareformerWithProj(D=11, H=10).to(device)
      #model = MultiHeadAttentionCosSquareformer(D=11, H=10).to(device)
      #model = CosSquareFormerModel(input_dim=11,D=32,H=4,N_layers=4,ff_dim=128,dropout=0.3,max_seq_len=64)
      model = CosSquareFormerForecastModel(input_dim=11,D=32,H=4,N_layers=4,ff_dim=128,dropout=0.3,max_seq_len=64).to(device)
      #opt = torch.optim.Adam(model.parameters(), lr=lr)
      opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
      #device = torch.device("cpu")
      #model = model.to(device)


      
      print('Start model training')
      best_mse = 2000.0
      best_model = None
      best_epoch = 0

      for epoch in range(1, n_epochs + 1):
          epoch_loss = 0
          batch_idx = 0
          #bar = tqdm(sampleLoader) 
          bar = tqdm(sampleLoader, disable=True)

          model.train()
          for x_batch, y_batch, _ in bar:
              #print(f"\nTraining: epoch[{epoch}/{n_epochs}]")
              model.train()
              x_batch = x_batch.to(device).float()
              y_batch = y_batch.to(device).float()

              mask = create_look_ahead_mask(x_batch.shape[1])
              out, _ = model(x_batch, mask)
              #out = out[:,:,-1:]
              opt.zero_grad()
              #print(out.shape, y_batch.shape)

              loss = criterion(out[:,-1,:], y_batch[:,-1,:]) + lmbda * dtw_loss(out.to(device),y_batch.to(device)).mean()

              epoch_loss = (epoch_loss*batch_idx + loss.item())/(batch_idx+1)
              loss.backward()
              opt.step()

              bar.set_description(str(epoch_loss))
              batch_idx += 1

          # Evaluation
          model.eval()
          mse_list = []
          total_se = 0.0
          total_pe = 0.0
          total_valid = 0.0

          for x_val, y_val, _ in val_loader:
              x_val, y_val = [t.to(device).float() for t in (x_val, y_val)]
              mask = create_look_ahead_mask(x_val.shape[1])
              out, _ = model(x_val, mask)
              #print(out.shape, y_val.shape)
              ytrue = y_val.squeeze(-1).cpu().numpy()
              ypred = out.squeeze(-1).cpu().detach().numpy()
              ytrue = ytrue.ravel()
              ypred = ypred.ravel()

              true_valid = np.isnan(ytrue) != 1
              ytrue = ytrue[true_valid]
              ypred = ypred[true_valid]

              if normalization_type == 'mean_std':
                  ytrue = (ytrue * col_std[SELECTED_COLUMN]) + col_mean2[SELECTED_COLUMN]
                  ypred = (ypred * col_std[SELECTED_COLUMN]) + col_mean2[SELECTED_COLUMN]
              else:
                  ytrue = (ytrue * col_max[SELECTED_COLUMN])
                  ypred = (ypred * col_max[SELECTED_COLUMN])
              
              se = (ytrue - ypred)**2
              pe = np.abs((ytrue - ypred) / (ytrue + 1e-4))
              
              total_se += np.sum(se)
              total_pe += np.sum(pe)
              total_valid += len(ytrue)

          eval_mse = total_se / total_valid
          eval_mape = total_pe / total_valid

          #print('valid samples:', total_valid)
          #print("Epoch: ", epoch)
          #print('Eval MSE: ', eval_mse)
          #print('Eval RMSE: {}: '.format(SELECTED_COLUMN), np.sqrt(eval_mse))
          #print('Eval MAPE: {}: '.format(SELECTED_COLUMN), eval_mape*100)

          if eval_mse < best_mse:
            best_model = deepcopy(model)
            best_mse = eval_mse
            best_epoch = epoch
            best_mape = eval_mape
            #torch.save(best_model.state_dict(), f"CSF14-7{best_epoch}.pth")
          
          RMSE_list.append(np.sqrt(eval_mse))
          MAPE_list.append(eval_mape*100)

      print("\nPollutants: ", SELECTED_COLUMN)
      print("Best epoch: ", best_epoch)
      print("Best RMSE: ", np.sqrt(best_mse))
      print("Best MAPE: ", best_mape*100)
      torch.save(best_model.state_dict(), f"CSF14-7_{SELECTED_COLUMN}_{best_epoch}.pth")
