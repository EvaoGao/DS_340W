import matplotlib.pyplot as plt
import numpy as np

pollutants = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
# SVR results are obtained in svr.py, and cosSquareFormer results are reported in parent paper
svr_rmse = {'pm25': 15.941, 'pm10': 9.321, 'o3': 7.311, 'no2': 3.776, 'so2': 0.856, 'co': 1.169}
svr_mape = {'pm25': 41.8, 'pm10': 43.4, 'o3': 30.9, 'no2': 53.1, 'so2': 102.1, 'co': 83.2}
cos_rmse = {'pm25': 11.68, 'pm10': 8.06, 'o3': 8.14, 'no2': 3.49, 'so2': 1.75, 'co': 5.42}
cos_mape = {'pm25': 34.7,  'pm10': 45.9, 'o3': 146.6, 'no2': 43.5, 'so2': 69.1, 'co': 125.4}
svr_imp_rmse = {'pm25': 14.200, 'pm10': 8.800, 'o3': 7.000, 'no2': 3.500, 'so2': 0.750, 'co': 1.100}
svr_imp_mape = {'pm25': 38.5,  'pm10': 42.0, 'o3': 29.5, 'no2': 50.0, 'so2': 95.0,  'co': 80.5}

svr_rmse_vals  = [svr_rmse[p]  for p in pollutants]
svr_mape_vals  = [svr_mape[p]  for p in pollutants]
cos_rmse_vals  = [cos_rmse[p]  for p in pollutants]
cos_mape_vals  = [cos_mape[p]  for p in pollutants]
svr_imp_rmse_vals = [svr_imp_rmse[p] for p in pollutants]
svr_imp_mape_vals = [svr_imp_mape[p] for p in pollutants]

x = np.arange(len(pollutants))
width = 0.35 

# SVR mape Comparison
fig, ax = plt.subplots()
ax.bar(x - width/2, svr_imp_mape_vals, width, label='SVR-Imputation')
ax.bar(x + width/2, svr_mape_vals, width, label='SVR-Deletion')
ax.set_ylabel('MAPE')
ax.set_title('SVR with Imputation vs SVR with Deletion (MAPE)')
ax.set_xticks(x)
ax.set_xticklabels(pollutants)
ax.legend()
plt.tight_layout()
plt.show()

# SVR RMSE Comparison
fig, ax = plt.subplots()
ax.bar(x - width/2, svr_imp_rmse_vals, width, label='SVR-Imputation')
ax.bar(x + width/2, svr_rmse_vals, width, label='SVR-Deletion')
ax.set_ylabel('RMSE')
ax.set_title('SVR with Imputation vs SVR with Deletion (RMSE)')
ax.set_xticks(x)
ax.set_xticklabels(pollutants)
ax.legend()
plt.tight_layout()
plt.show()

#  Naive model RMSE comparison
fig, ax = plt.subplots()
ax.bar(x - width/2, svr_rmse_vals, width, label='SVR-RBF')
ax.bar(x + width/2, cos_rmse_vals, width, label='cosSquareFormer')
ax.set_ylabel('RMSE')
ax.set_title('RMSE Comparison by Pollutant (Without Imputation)')
ax.set_xticks(x)
ax.set_xticklabels(pollutants)
ax.legend()
plt.tight_layout()
plt.show()

# Naive model MAPE comparison
fig, ax = plt.subplots()
ax.bar(x - width/2, svr_mape_vals, width, label='SVR-RBF')
ax.bar(x + width/2, cos_mape_vals, width, label='cosSquareFormer')
ax.set_ylabel('MAPE')
ax.set_title('MAPE Comparison by Pollutant (Without Imputation)')
ax.set_xticks(x)
ax.set_xticklabels(pollutants)
ax.legend()
plt.tight_layout()
plt.show()

#  Imputation model RMSE comparison
fig, ax = plt.subplots()
ax.bar(x - width/2, svr_imp_rmse_vals, width, label='SVR-RBF')
ax.bar(x + width/2, cos_rmse_vals, width, label='cosSquareFormer')
ax.set_ylabel('RMSE')
ax.set_title('RMSE Comparison by Pollutant (With Imputation)')
ax.set_xticks(x)
ax.set_xticklabels(pollutants)
ax.legend()
plt.tight_layout()
plt.show()

# Imputation model MAPE comparison
fig, ax = plt.subplots()
ax.bar(x - width/2, svr_imp_mape_vals, width, label='SVR-RBF')
ax.bar(x + width/2, cos_mape_vals, width, label='cosSquareFormer')
ax.set_ylabel('MAPE')
ax.set_title('MAPE Comparison by Pollutant (With Imputation)')
ax.set_xticks(x)
ax.set_xticklabels(pollutants)
ax.legend()
plt.tight_layout()
plt.show()
