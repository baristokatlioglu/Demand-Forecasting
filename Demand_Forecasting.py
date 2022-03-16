import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from helpers.eda import *
import seaborn as sns
import lightgbm as lgb
import warnings

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
warnings.filterwarnings('ignore')

# 3-month item-level sales forecast for different stores.
# There are 10 different stores and 50 different items in a 5-year dataset.
# Accordingly, we need to give forecasts for 3 months after the store-item breakdown.

###
# Reading Data
###
train = pd.read_csv('datasets/train.csv', parse_dates=['date'])
test = pd.read_csv('datasets/test.csv', parse_dates=['date'])
sample_sub = pd.read_csv('datasets/sample_submission.csv')
df = pd.concat([train, test], sort=False)

#####
# EDA
#####
check_df(df)

# there are 45000 blank observations in sales because we were asked to estimate the sales on the test set
# There are no ids in the train set, as we will train the algorithm over the sales variable.

# How is the sales distribution?
df["sales"].describe([0.10, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99])

# How many stores are there?
df[["store"]].nunique()

# How many items are there?
df[["item"]].nunique()

# Are there an equal number of unique items in each store?
df.groupby(["store"])["item"].nunique()

# So, is there an equal number of sales in each store?
df.groupby(["store", "item"]).agg({"sales": ["sum"]})

# sales statistics in store-item breakdown
df.groupby(["store", "item"]).agg({"sales": ["sum", "mean", "median", "std"]})

def create_date_features(df):
    df['month'] = df.date.dt.month
    df['day_of_month'] = df.date.dt.day
    df['day_of_year'] = df.date.dt.dayofyear
    df['week_of_year'] = df.date.dt.weekofyear
    df['day_of_week'] = df.date.dt.dayofweek
    df['year'] = df.date.dt.year
    df["is_wknd"] = df.date.dt.weekday // 4
    df['is_month_start'] = df.date.dt.is_month_start.astype(int)
    df['is_month_end'] = df.date.dt.is_month_end.astype(int)
    return df

df = create_date_features(df)

########################
# Random Noise
########################

def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))

# At this point, we add random noise to the variables that we will produce in the sales focus in order to prevent over-learning
# or to prevent bias. This will add noise prevents over-learning.

########################
# Lag/Shifted Features
########################
df.sort_values(by=['store', 'item', 'date'], axis=0, inplace=True)

df.groupby(["store", "item"])['sales'].head()

df.groupby(["store", "item"])['sales'].transform(lambda x: x.shift(1))

# Lag/Shifted means Latency. The time period yt was most affected, from yt-1 to yt-2 in order. Therefore, we assumed
# that the next day's sale of a store would be affected by the previous day's sale. We will need to derive features for this.

def lag_features(dataframe, lags):
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["store", "item"])['sales'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe


df = lag_features(df, [91, 98, 105, 112, 119, 126, 180, 364, 546, 728, 910, 1092])

########################
# Rolling Mean Features
########################
# Rolling Mean means moving average. These moving averages carry historical information. We will create them to put Level
# and Trend components. window represents the number of delay. The entered value takes the delay with itself and calculates the average.
#
# At this point, the own value of the observation unit should not be taken into account when calculating the moving average,
# but in this case, a feature independent of the current value can be produced that can express the trend from the past.

def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby(["store", "item"])['sales']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe

df = roll_mean_features(df, [180, 365, 546, 728, 910])
df.head()

########################
# Exponentially Weighted Mean Features
########################

# We know that the weighted average gives more weight to values close to the past. When the alpha is .99 it will
# give more weight to the nearest values, when it is 0.1 it will give more weight to the far values. In this example,
# we observe that as the weight of the past values increases, it approaches the normal mean value.

def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["store", "item"])['sales'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe

alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [91, 98, 105, 112, 180, 270, 365, 546, 728, 910]

df = ewm_features(df, alphas, lags)

########################
# One-Hot Encoding
########################

df = pd.get_dummies(df, columns=['store', 'item', 'day_of_week', 'month'])

# Since we will use lightgbm, one of the tree methods, we may prefer to log the dependent variable in order to make the
# gradient descent algorithm work faster.

########################
# Converting sales to log(1+sales)
########################

df['sales'] = np.log1p(df["sales"].values)

#####################################################
# Model
#####################################################

########################
# Custom Cost Function
########################

# MAE: mean absolute error
# MAPE: mean absolute percentage error
# SMAPE: Symmetric mean absolute percentage error (adjusted MAPE)

# SMAPE (Symmetric Mean Absolute Error) or MAPE allows us to evaluate the errors as a percentage. Returns a metric from 0-100.
# The expm1() function reverses the log transformation.
def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val

def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False

########################
# Time-Based Validation Sets
########################

train = df.loc[(df["date"] < "2017-10-01"), :]

val = df.loc[(df["date"] >= "2017-10-01") & (df["date"] < "2017-12-31"), :]

cols = [col for col in train.columns if col not in ['date', 'id', "sales", "year"]]

Y_train = train['sales']
X_train = train[cols]

Y_val = val['sales']
X_val = val[cols]

Y_train.shape, X_train.shape, Y_val.shape, X_val.shape

# At this point, we cannot separate the data set with train test split, because this function pulls random values
# from the data, thus distorting the context and structure of the time series. For this, we separate the dataset ourselves.

lgb_params = {'metric': {'mae'},
              'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'num_boost_round': 1000,
              'early_stopping_rounds': 200,
              'nthread': -1}

lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)
lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)

model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping_rounds=lgb_params['early_stopping_rounds'],
                  feval=lgbm_smape,
                  verbose_eval=100)

y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

smape(np.expm1(y_pred_val), np.expm1(Y_val))

# Normally we used lightgbm from within sklearn. Here we used Microsoft's own lightgbm, but for this the model requires
# its own data type. With lgb.Dataseet, we can convert this data to the desired data type.

########################
# Final Model
########################
train = df.loc[~df.sales.isna()]
Y_train = train['sales']
X_train = train[cols]

test = df.loc[df.sales.isna()]
X_test = test[cols]

lgb_params = {'metric': {'mae'},
              'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'nthread': -1,
              "num_boost_round": model.best_iteration}


# LightGBM dataset
lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)

model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=model.best_iteration)

test_preds = model.predict(X_test, num_iteration=model.best_iteration)
# 12.657650852363272

# Creat submission
submission_df = test.loc[:, ['id', 'sales']]
submission_df['sales'] = np.expm1(test_preds)
submission_df['id'] = submission_df.id.astype(int)

submission_df.to_csv('submission_demand.csv', index=False)
submission_df.head(20)

