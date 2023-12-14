# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
#pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org jupyter-dash -q

# +
#pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org dash-cytoscape -q

# +
# pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org streamlit

# +
#pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org scikit-learn

# +
#pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org xgboost

# +
#pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org shap
# -

"""
Created on Fri Dec 1 2023

@author: Naveen 
"""
import dash
from dash import dcc
from dash import html
import plotly.express as px
import pandas as pd
import numpy as np
import pickle
#import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import os

# Load dataset
#df3 = pd.read_csv("C:/data/us_chronic_disease_indicators.csv")
df3 = pd.read_csv("https://storage.googleapis.com/mbcc/datasets/us_chronic_disease_indicators.csv")

# +
#df.info()
#df.head(5)
# -

# Create a copy of the original data to use in Streamlit app
df3_original = df3.copy()

# Exploratory Data Analysis - Dropping Redundant Columns, since the same information is already available
# Drop the columns that are not needed for prediction
df3 = df3.drop(columns=["yearstart", "locationabbr", "datasource", "lowconfidencelimit", "highconfidencelimit", "geolocation", "topicid", "questionid", "datavaluetypeid", "stratificationid1", "stratificationcategory1"])
#df3 = df3.drop(columns=["locationabbr", "datasource", "lowconfidencelimit", "highconfidencelimit", "geolocation", "topicid", "questionid", "datavaluetypeid", "stratificationid1", "stratificationcategory1"])
#df3 = df3.drop(columns=["stratificationcategory1"])

#Apply filters create the model specific dataset
# Step 2: Filter data for the last decade (2010-2019)
#df3['yearend'] = pd.to_datetime(df3['yearend'], format='%Y')
#df3 = df3[(df3['yearend'] >= '2011-01-01') & (df3['yearend'] <= '2020-12-31')]
df3 = df3[(df3.topic.isin(['Alcohol','Cardiovascular Disease', 'Chronic Kidney Disease', 'Chronic Obstructive Pulmonary Disease', 'Diabetes']))]

# Create Final Dataset for Mortality Prediction 
#result_case_insensitive = df3[df3['Column1'].str.contains('AN', case=False)]
#print(result_case_insensitive)
df3_mortality = df3[df3['question'].str.contains('Mortality', case=False) 
#                  & df3.stratificationcategoryid1.isin(['GENDER', 'OVERALL'])
                   & df3.stratificationcategoryid1.isin([ 'OVERALL'])
#                  & df3.datavaluetype.isin(['Average Annual Number', 'Number'])
#                   & df3.datavaluetype.isin(['cases per 100,000', 'cases per 1,000,000', 'per 100,000'])
                   & df3.datavaluetype.isin(['Crude Rate'])
#                   & df3[~df3['locationdesc'].isin(['United States'])]
#                  & df3.locationdesc.isin(['Michigan', 'Texas', 'California', 'Tennessee', 'Florida', 'New York', 'Ohio', 'Illinois', 'Pennsylvania'])
#                  & df3.locationdesc.isin(['West Virginia', 'New Hampshire', 'Maine', 'Montana', 'Rhode Island', 'Delaware', 'South Dakota', 'North Dakota', 'Alaska', 'Vermont', 'Wyoming', 'Nebraska'])
                 ]
#df3_mortality = df3_mortality.drop(columns=['datavalueunit', 'datavaluetype','stratification1', 'stratificationcategoryid1'])
df3_mortality = df3_mortality[~df3_mortality['locationdesc'].isin(['United States'])]

df3_mortality['datavalue'] = np.where(df3_mortality['datavalueunit'] == 'cases per 100,000', 10 * df3_mortality['datavalue'], df3_mortality['datavalue'])
df3_mortality['datavalue'] = np.where(df3_mortality['datavalueunit'] == 'per 100,000', 10 * df3_mortality['datavalue'], df3_mortality['datavalue'])

# +
#df3_mortality.head(5)
# -

df3_mortality = df3_mortality.drop(columns=['question'])

df3_mortality['locationdesc'] = df3['locationdesc'].astype(str)
df3_mortality['topic'] = df3['topic'].astype(str)

df3_mortality = df3_mortality.groupby(['yearend', 'locationdesc', 'topic'] )['datavalue'].sum().reset_index()

# +
#df_mortality.head(20)
# -

# 'datavalue' is the Target Variable
# Create a copy of the original data to use in Streamlit app
df3_mortality_original = df3_mortality.copy()
y = df3_mortality['datavalue']
X = df3_mortality.drop(columns=['datavalue'])
y = y.astype(int)

# Convert categorical features into numeric by encoding the categories
label_encoders = {}
for column in X.columns:
    if X[column].dtype == type(object):
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column].astype(str))
        label_encoders[column] = le

# +
#X_train

# +
#X_test

# +
#y_train

# +
#y_test

# +
#class_mapping = {class_label: idx for idx, class_label in enumerate(sorted(y_train.unique()))}
#y_train = y_train.map(class_mapping)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a linear regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_pred = linear_model.predict(X_test)

# Fit a Lasso regression model
ridge_model = Ridge(alpha=0.1)
ridge_model.fit(X_train, y_train)
ridge_pred = ridge_model.predict(X_test)


# Fit a random forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Fit an XGBoost model
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

# Calculate mean squared error for each model
linear_mse = mean_squared_error(y_test, linear_pred)
ridge_mse = mean_squared_error(y_test, ridge_pred)
rf_mse = mean_squared_error(y_test, rf_pred)
xgb_mse = mean_squared_error(y_test, xgb_pred)


# Calculate Root mean squared error for each model
linear_rmse = np.sqrt(linear_mse)
ridge_rmse = np.sqrt(ridge_mse)
rf_rmse = np.sqrt(rf_mse)
xgb_rmse = np.sqrt(xgb_mse)


# Calculate R-squared (R2) for each model
linear_r2 = r2_score(y_test, linear_pred)
ridge_r2 = r2_score(y_test, ridge_pred)
rf_r2 = r2_score(y_test, rf_pred)
xgb_r2 = r2_score(y_test, xgb_pred)

linear_r2 = linear_r2
linear_rid = ridge_r2
linear_rf = rf_r2
linear_xgb = xgb_r2

# Convert the numerical value to a string before setting it as an environment variable
os.environ['linear_r2'] = str(linear_r2)
os.environ['linear_rid'] = str(linear_rid)
os.environ['linear_rf'] = str(linear_rf)
os.environ['linear_xgb'] = str(linear_xgb)

# Print the metric results to asses the model's ability to predict continous values.
print(f"Linear regression MSE, RMSE & R2: {linear_mse} , {linear_rmse} & {linear_r2} ")
print(f"Lasso regression MSE, RMSE & R2: {ridge_mse} , {ridge_rmse} & {ridge_r2} ")
print(f"Random forest MSE, RMSE & R2: {rf_mse} , {rf_rmse} & {rf_r2} ")
print(f"XGBoost MSE, RMSE & R2: {xgb_mse} , {xgb_rmse} & {xgb_r2} ")


# +
# Visualize predicted vs actual values (Linear Regression)
#plt.scatter(y_test, linear_pred)
#plt.xlabel('Actual Values')
#plt.ylabel('Predicted Values')
#plt.title('Actual vs Predicted Values')
#plt.show()

# +
# Visualize predicted vs actual values (Random Forest model)
#plt.scatter(y_test, rf_pred)
#plt.xlabel('Actual Values')
#plt.ylabel('Predicted Values')
#plt.title('Actual vs Predicted Values')
#plt.show()

# +
# Visualize predicted vs actual values (XGBoost model)
#plt.scatter(y_test, xgb_pred)
#plt.xlabel('Actual Values')
#plt.ylabel('Predicted Values')
#plt.title('Actual vs Predicted Values')
#plt.show()

# +
#print("Number of samples in X_train:", len(X_train))
#print("Number of samples in y_train:", len(y_train))

# +
#linear_model.intercept_

# +
#y_pred = linear_model.predict(X_train)
#residuals = y_train - y_pred

# +
#sum(residuals)/len(residuals) # mean residual

# +
#max(residuals), min(residuals)

# +
#Residuals plot for Linear Regression
#import matplotlib.pyplot as plt
#plt.scatter(y_train, residuals)
#plt.xlabel('Actual Values')
#plt.ylabel('Residuals')
#plt.show()

# +
#y_pred2 = rf_model.predict(X_train)
#residuals = y_train - y_pred2

# +
#Residuals plot for Random Forest Regression
#import matplotlib.pyplot as plt
#plt.scatter(y_train, residuals)
#plt.xlabel('Actual Values')
#plt.ylabel('Residuals')
#plt.show()

# +
#y_pred1 = xgb_model.predict(X_train)
#residuals = y_train - y_pred1

# +
#Residuals plot for Xgb Regression
#import matplotlib.pyplot as plt
#plt.scatter(y_train, residuals)
#plt.xlabel('Actual Values')
#plt.ylabel('Residuals')
#plt.show()

# +
# Save the trained model and the label encoders to disk for later use in the Streamlit app
with open('model.pkl', 'wb') as f:
    pickle.dump(linear_model, f)
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
with open('columns_order.pkl', 'wb') as f:
    pickle.dump(X_train.columns.tolist(), f)

# Load the saved model and label encoders
with open('model.pkl', 'rb') as f:
    linear_model = pickle.load(f)
with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)
with open('columns_order.pkl', 'rb') as f:
    columns_order = pickle.load(f) 
# -

# Function to make predictions 
def predict_mortality_data(mortality_data, columns_order):
    # Convert mortality data into the same format as the training data
    for col, value in mortality_data.items():
        if col in label_encoders:  # for categorical data
            le = label_encoders[col]
            if value in le.classes_:
                mortality_data[col] = le.transform([value])[0]
            else:
                mortality_data[col] = le.transform([le.classes_[0]])[0]  # if value is not in classes, use the first class
        else:  # for numerical data
            mortality_data[col] = float(value)

    # Convert patient data into dataframe
    df3 = pd.DataFrame([mortality_data])
    df3 = df3[columns_order]
    
    # Predict the Mortality Avg Number for selected disease
    mor = linear_model.predict(df3)
    return mor


# Function to make predictions 
def predict_mortality_data_rid(mortality_data, columns_order):
    # Convert mortality data into the same format as the training data
    for col, value in mortality_data.items():
        if col in label_encoders:  # for categorical data
            le = label_encoders[col]
            if value in le.classes_:
                mortality_data[col] = le.transform([value])[0]
            else:
                mortality_data[col] = le.transform([le.classes_[0]])[0]  # if value is not in classes, use the first class
        else:  # for numerical data
            mortality_data[col] = float(value)

    # Convert patient data into dataframe
    df3 = pd.DataFrame([mortality_data])
    df3 = df3[columns_order]
    
    # Predict the Mortality Avg Number for selected disease
    mor_rid = ridge_model.predict(df3)
    return mor_rid


# Function to make predictions 
def predict_mortality_data_rf(mortality_data, columns_order):
    # Convert mortality data into the same format as the training data
    for col, value in mortality_data.items():
        if col in label_encoders:  # for categorical data
            le = label_encoders[col]
            if value in le.classes_:
                mortality_data[col] = le.transform([value])[0]
            else:
                mortality_data[col] = le.transform([le.classes_[0]])[0]  # if value is not in classes, use the first class
        else:  # for numerical data
            mortality_data[col] = float(value)

    # Convert patient data into dataframe
    df3 = pd.DataFrame([mortality_data])
    df3 = df3[columns_order]
    
    # Predict the Mortality Avg Number for selected disease
    mor_rf = rf_model.predict(df3)
    return mor_rf


# Function to make predictions 
def predict_mortality_data_xgb(mortality_data, columns_order):
    # Convert mortality data into the same format as the training data
    for col, value in mortality_data.items():
        if col in label_encoders:  # for categorical data
            le = label_encoders[col]
            if value in le.classes_:
                mortality_data[col] = le.transform([value])[0]
            else:
                mortality_data[col] = le.transform([le.classes_[0]])[0]  # if value is not in classes, use the first class
        else:  # for numerical data
            mortality_data[col] = float(value)

    # Convert patient data into dataframe
    df3 = pd.DataFrame([mortality_data])
    df3 = df3[columns_order]
    
    # Predict the Mortality Avg Number for selected disease
    mor_xgb = xgb_model.predict(df3)
    return mor_xgb
