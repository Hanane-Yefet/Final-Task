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

def prepare_data(data):
  
    data.shape
    data = data.drop(columns = 'Unnamed: 23')
    data['price'] = data['price'].apply(lambda x: re.sub(r'\D', '', str(x))).replace('', np.nan)
    data.dropna(subset=['price'], inplace=True)
    data.price = data.price.astype(float)
    data['Area'] = data['Area'].apply(lambda x: re.sub(r'\D', '', str(x))).replace('', np.nan)
    data.Area = data.Area.astype(float)
    data['city_area'] = data['city_area'].str.replace(r'[^\u0590-\u05FF]+', '  ', regex=True)
    data['description '] = data['description '].str.replace(r'[^\u0590-\u05FF]+','  ', regex=True)
    data['Street'] = data['Street'].str.replace(r'[^א-ת]+','  ', regex=True)
    data = data.reset_index(drop=True)
    data['floor'] = data['floor_out_of'].str.extract(r'(\d+)').astype(float)
    data['total_floors'] = data['floor_out_of'].str.extract(r'\d+\s*מתוך\s*(\d+)').astype(float)
    data['entranceDate '] = data['entranceDate '].replace('מיידי', 'less_than_6 months')
    data['entranceDate '] = data['entranceDate '].replace('גמיש','flexible')
    data['entranceDate '] = data['entranceDate '].replace('לא צויין','not_defined')

    def check_date_range(value):
        today = datetime.today().date()
        try:
            value=datetime.strptime(value, "%d/%m/%Y").date()
            difference = value - today
            if difference < timedelta(days=183):  # Less than 0.5 years (0.5 * 365 = 183)
                return "less_than_6 months"
            elif timedelta(days=183) <= difference < timedelta(days=365):  # Between 0.5 and 1 year
                return "between 0.5 and 1 year"
            else:  # Above 1 year
                return "above 1 year"
        except ValueError:
            return value
    data['entranceDate '] = data['entranceDate '].apply(lambda x:check_date_range(x))
    columns=['hasElevator ' , 'hasParking ' , 'hasBars ' , 'hasStorage ' , 'hasAirCondition ' , 'hasBalcony ' , 'hasMamad ' , 'handicapFriendly '] 

    def replace_boolean_and_hebrew(row):
        if isinstance(row, float):
            return row
        elif 'אין' in row or 'FALSE' in row or 'לא' in row or 'no' in row:
            row = 0  
        else:
            row = 1
        return row

    for col in columns:
        data[col] = data[col].apply(lambda x: replace_boolean_and_hebrew(x))
    data['room_number'] = data['room_number'].apply(lambda x: re.sub(r'\D', '', str(x))).replace('', np.nan)
    data['number_in_street'] = data['number_in_street'].apply(lambda x: re.sub(r'\D', '', str(x))).replace('', np.nan)
    data['publishedDays '] = data['publishedDays '].apply(lambda x: re.sub(r'\D', '', str(x))).replace('', np.nan)
    data.room_number = data.room_number.astype(float)
    data.number_in_street = data.number_in_street.astype(float)
    data = data.drop('floor_out_of' , axis = 1)
    data['publishedDays ']=data['publishedDays '].astype(float)
    df = data.copy()
    return df
