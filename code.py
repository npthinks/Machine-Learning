# Importing packages, setting random seed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
np.random.seed(123)

# Reading in the data
train = pd.read_csv("data/train_data_ML.csv")
test = pd.read_csv("data/test_data.csv_ML")
train.rename(columns={'ResponseValue': 'y'}, inplace=True)
test.rename(columns={'ResponseValue': 'y'}, inplace=True)

# Imputing CurrentGameMode
na_indices_test = train[train["CurrentGameMode"].isna()].index.tolist()
for value in na_indices_test:
    train.at[value, "CurrentGameMode"] = "Career"

na_indices_test = test[test["CurrentGameMode"].isna()].index.tolist()
for value in na_indices_test:
    test.at[value, "CurrentGameMode"] = "Career"

# Imputing CurrentTask, dropping NAs for train data
train.dropna(subset=['CurrentTask'], inplace=True)

na_indices_test = test[test["CurrentTask"].isna()].index.tolist()
for value in na_indices_test:
    test.at[value, "CurrentTask"] = "SEATEMPLE"

# Imputing LastTaskCompleted, dropping NAs for train data

train.dropna(subset=['LastTaskCompleted'], inplace=True)

na_indices_test = test[test["LastTaskCompleted"].isna()].index.tolist()
for value in na_indices_test:
    test.at[value, "LastTaskCompleted"] = "WASH_PWVan"

# Imputing LevelProgression, dropping NAs for train data

train.dropna(subset=['LevelProgressionAmount'], inplace=True)

test['LevelProgressionAmount'] = test['LevelProgressionAmount'].transform(lambda x: x.fillna(x.mean()))

# Split Date to Year, Month, Day and convert to integer

train['DateUtc'] = train['TimeUtc'].str.split(' ').str[0]
train['hourUtc'] = train['TimeUtc'].str.split(' ').str[1]
train['YearUtc'] = train['DateUtc'].str.split('-').str[0]
train['MonthUtc'] = train['DateUtc'].str.split('-').str[1]
train['DayUtc'] = train['DateUtc'].str.split('-').str[2]
train['YearUtc'] = train['YearUtc'].astype(int)
train['MonthUtc'] = train['MonthUtc'].astype(int)
train['DayUtc'] = train['DayUtc'].astype(int)
train.drop('TimeUtc', axis = 1, inplace = True)
train.drop('DateUtc', axis = 1, inplace = True)
train['HourUtc'] = train['hourUtc'].str.split(':').str[0]
train['MinUtc'] = train['hourUtc'].str.split(':').str[1]
train['HourUtc'] = train['HourUtc'].astype(int)
train['MinUtc'] = train['MinUtc'].astype(int)
train.drop('hourUtc', axis = 1, inplace = True)

test['DateUtc'] = test['TimeUtc'].str.split(' ').str[0]
test['hourUtc'] = test['TimeUtc'].str.split(' ').str[1]
test['YearUtc'] = test['DateUtc'].str.split('-').str[0]
test['MonthUtc'] = test['DateUtc'].str.split('-').str[1]
test['DayUtc'] = test['DateUtc'].str.split('-').str[2]
test['YearUtc'] = test['YearUtc'].astype(int)
test['MonthUtc'] = test['MonthUtc'].astype(int)
test['DayUtc'] = test['DayUtc'].astype(int)
test.drop('TimeUtc', axis = 1, inplace = True)
test.drop('DateUtc', axis = 1, inplace = True)
test['HourUtc'] = test['hourUtc'].str.split(':').str[0]
test['MinUtc'] = test['hourUtc'].str.split(':').str[1]
test['HourUtc'] = test['HourUtc'].astype(int)
test['MinUtc'] = test['MinUtc'].astype(int)
test.drop('hourUtc', axis = 1, inplace = True)

# Creating dummy variables

train = pd.get_dummies(train, columns=['LastTaskCompleted', 'CurrentTask', 'CurrentGameMode'], dummy_na = False)

test = pd.get_dummies(test, columns=['LastTaskCompleted', 'CurrentTask', 'CurrentGameMode'], dummy_na = False)

# Factorizing 'QuestionTiming'

train['QuestionTiming'] = train['QuestionTiming'].map({'User Initiated':1, 'System Initiated':0})

test['QuestionTiming'] = test['QuestionTiming'].map({'User Initiated':1, 'System Initiated':0})

# Taking outliers out of CurrentSessionLength
threshold_cur_s = 2.5


def outliers_to_nan_cur_s (data):

    mean_cur_s = np.mean(data["CurrentSessionLength"])
    std_cur_s = np.std(data["CurrentSessionLength"])


    for index, row in data.iterrows():
        value = row["CurrentSessionLength"]
        
        if value < 0:
             data.at[index, "CurrentSessionLength"] = np.nan

        else:
            z_score = (value - mean_cur_s)/std_cur_s
            if np.abs(z_score) > threshold_cur_s:
                data.at[index, "CurrentSessionLength"] = np.nan


outliers_to_nan_cur_s(train)
outliers_to_nan_cur_s(test)

train["CurrentSessionLength"] = train["CurrentSessionLength"].transform(lambda x: x.fillna(x.mean()))
test["CurrentSessionLength"] = test["CurrentSessionLength"].transform(lambda x: x.fillna(x.mean()))

# Getting UserID dummies

train = pd.get_dummies(train, columns = ["UserID"])

test = pd.get_dummies(test, columns = ["UserID"])

# Removing unneeded columns

train = train.drop(columns = ['QuestionType'])

test = test.drop(columns = ['QuestionType'])

# Removing non-common columns, to comply with the model
common_columns = train.columns.intersection(test.columns)

train_columns_to_keep = common_columns.tolist() + ['y'] if 'y' in train.columns else common_columns

train = train[train_columns_to_keep]
test = test[common_columns]

# Splitting the data

X = train.drop(columns=["y"])
y = train["y"]

# Instantiating and fitting the model

ridge_model = Ridge(alpha = 0.2)
ridge_model.fit(X, y)

# Predicting and saving predictions
predictions_on_test = ridge_model.predict(test)
predictions_on_test = np.where(predictions_on_test > 920, 1000, predictions_on_test)
predictions_on_test = np.where(predictions_on_test < 90, 0, predictions_on_test)
np.savetxt("predicted.csv", predictions_on_test)
