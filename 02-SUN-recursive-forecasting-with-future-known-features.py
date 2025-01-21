#import libraries
import pandas as pd
import numpy as np
from docutils.nodes import label
from pyexpat import features

from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from skforecast.recursive import ForecasterRecursive

from feature_engine.datetime import DatetimeFeatures

import matplotlib.pyplot as plt

#data import and manipulation
url="https://raw.githubusercontent.com/tidyverts/tsibbledata/master/data-raw/vic_elec/VIC2015/demand.csv"
df=pd.read_csv(url)

df.dtypes
df.head()

#convert the integer date to an actual date with datetime type
df['date']=df['Date'].apply(
    lambda x:pd.Timestamp('1899-12-30') + pd.Timedelta(x,unit='days')
)

#create a timestamp from the integer period representing 30 minute intervals
df['date_time']=df['date'] +\
     pd.to_timedelta((df['Period']-1)*30,unit='m')

df.dropna(inplace=True)
#selecting the required columns and renaming it
df=df[['date_time','OperationalLessIndustrial']]
df.columns=['date_time','demand']

#resample to hourly
df=(df.set_index('date_time')
    .resample('H')
    .agg({'demand':'sum'})
)
#split the data into train and test
end_train=('2014-12-30 23:59:59')
x_train=df.loc[:end_train]
x_test=df.loc[end_train:]

#plotting the time series
fig,ax=plt.subplots(figsize=(7,3))
x_train.tail(500).plot(ax=ax)
x_test.head(500).plot(ax=ax)
ax.set_title('Hourly energy consumption')
ax.legend(['train','test'])
plt.savefig('Hourly enrgy cossumption 500.pdf',format='pdf')
plt.close()

#ML model
# we are going to use pipeline from sklearn
#using MinMax scaler to scale the variables

model=Pipeline([
    ('scaler',MinMaxScaler()),
    ('lasso',Lasso(random_state=9,alpha=10))
])

#adding datetime features
#features can be extracted from one ormore columns
#or from the index

datetime_f=DatetimeFeatures(
    features_to_extract=['month','day_of_week','hour'],
    drop_original=True,
)

#the input to the datetime features
datetime_df=pd.DataFrame(
    x_train.index,
    index=x_train.index
)

datetime_f.fit_transform(datetime_df)

#lets define the forecasterrecursive function
forecaster= ForecasterRecursive(
    regressor=model,   # the machine learning model
    lags=[1,24,6*24],  # the lag features to create
    transformer_exog=datetime_f, # to get the datetime features
    forecaster_id='recursive'
)
#next step is to fit the model
forecaster.fit(
    y=x_train['demand'], # the series for the lags
    exog=datetime_df, # the datetime for the datetime features
)

# we are checking the predictor features created by skforecsat
#to know what we are using for the forecast
X,y=forecaster.create_train_X_y(y=x_train['demand'],
                                exog=datetime_df,
                                )

X.head()
y

# since we have lag of 144 , intial 144 points is dropped,
#we can see that the data is starting from 07-01-2002

#forecast for next 24 hours
#for forecast generation we need to
# create similar exog variables

datetime_df_test =pd.DataFrame(
    x_test.head(24).index,
    index=x_test.head(24).index,
)

#predict for next 24 hours
predictions =forecaster.predict(
    steps=24,
    exog=datetime_df_test,
)
predictions

#plotting the forecast vs actuals
fig,ax=plt.subplots(figsize=(7,3))
x_train.tail(100)['demand'].plot(ax=ax,label='train')
x_test.head(24)['demand'].plot(ax=ax,label='test')
predictions.plot(ax=ax,label='predictions')
plt.title('Lasso Forecasting')
plt.ylabel('Energy demand hourly')
ax.legend()
plt.savefig('Lasso Forecasting Lag Datetime.pdf',format='pdf')
plt.close()

#predcition error mse and rmse
#mse
error_mse=mean_squared_error(
    y_true=x_test['demand'].head(24),
    y_pred=predictions
)
print(f' Test error (mse):{error_mse}')

#rmse
error_rmse=mean_squared_error(
    y_true=x_test['demand'].head(24),
    y_pred=predictions,
    squared=False,
)
print(f' Test error (rmse):{error_rmse}')

#predict at any point in the future
