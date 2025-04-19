# Introduction
This repository is for Penn State DS 340W on SP 2025 

The source code (loss_utils.py, model_utils.py, train.py) and data is retrieved from https://github.com/mayukh18/deap

The codes have been modified and added with new functionalities.

Other files are all implemented by my group.

This project aims to analyze different machine learning models to do air foreacasting on United States. The city_pollution_data.csv contain all necessary data to reproduce our results. FDA_train.py, GCN_train_new.py, svr2.py, and train.py are the training of four models, which are Functional Linear Regression, Graph Convolutional Network, Support Vector Regression, and CosSquareFormer. GCN_Deletion.py is the training of GCN using deletion method (Deleting all incomplete rows). loss_utils.py and model_utils.py contain all loss functions and DL models architecture. Extreme_Eval.ipynb conduct extreme pollution levels evaluation on all models. visuals.ipynb generates all the graphs used in our paper. 

# Direction
Before running the code, make sure you read through the import modules and download all necessary modules using pip install. Also, ensure that you have city_pollution_data.csv in a same directory and you have changed pd.read_csv to the correct path.

train.py will take roughly 6-8 hours to run. To reduce the running time, modify the main training loop on line 323 to iterate through less pollutants.

GCN_train_new.py will take roughly 1-2 hours, and svr2.py and FDA_train.py should take less than an hour.

To get the results of Deletion vs. Mean Imputations vs. Median Imputation, you need to make some small changes to the codes. In svr2.py, you need to replace 'Methods2' with 'Methods' on line 173. GCN_Deletion.py gives you the results using deletion method. GCN_train_new.py use mean imputations. To change it to median imputations you need to do the following: line 20, change method = "mean" to method = "median"; in line 71, change np.nanmean to np.nanmedian. 

