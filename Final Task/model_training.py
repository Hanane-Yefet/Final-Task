from madlan_data_prep import prepare_data
import numpy as np
import pandas as pd
import re
from datetime import datetime , timedelta
from sklearn.model_selection import train_test_split , KFold , cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import MinMaxScaler
import pickle

data = pd.read_csv('output_all_students_Train_v10.csv')
df = prepare_data(data)

# Split the data into features (X) and target variable (y)
X = df.drop(['price', 'description ', 'Street', 'entranceDate ','publishedDays ', 'furniture ','number_in_street', 'num_of_images'], axis=1)
y = df['price']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=747)

# Define the numerical and categorical columns
num_cols = ['room_number', 'Area', 'hasElevator ', 'hasParking ', 'hasBalcony ', 'hasMamad ', 'hasStorage ','floor','total_floors']
cat_cols = ['City', 'type', 'city_area', 'condition ']

# Define the preprocessing pipelines for numerical and categorical data
numerical_pipeline = Pipeline([
    ('numerical_imputation', SimpleImputer(strategy='most_frequent')),
    ('scaling', MinMaxScaler())
])

categorical_pipeline = Pipeline([
    ('categorical_imputation', SimpleImputer(strategy='constant', add_indicator=False, fill_value='missing')),
    ('one_hot_encoding', OneHotEncoder(sparse=False, handle_unknown='ignore'))
])

# Create the column transformer to preprocess the data
column_transformer = ColumnTransformer([
    ('numerical_preprocessing', numerical_pipeline, num_cols),
    ('categorical_preprocessing', categorical_pipeline, cat_cols),
], remainder='drop')

# Set the alpha and l1_ratio values for ElasticNet
alpha = 0.1
l1_ratio = 0.9

# Create the ElasticNet model
elastic_net = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)

# Create the preprocessing model pipeline
pipe_preprocessing_model = Pipeline([
    ('preprocessing_step', column_transformer),
    ('model', elastic_net)
])

# Train the model on the training set
pipe_preprocessing_model.fit(X_train, y_train)

# Perform grid search with cross-validation
k = 10
mse_scores = -cross_val_score(pipe_preprocessing_model, X_train, y_train, scoring='neg_mean_squared_error', cv=k)

# Get the best model and its RMSE score
mse = mse_scores.mean()
rmse = np.sqrt
pipe_preprocessing_model.fit(X_train, y_train)

with open('trained_model.pkl', 'wb') as file:
    pickle.dump(pipe_preprocessing_model, file)













