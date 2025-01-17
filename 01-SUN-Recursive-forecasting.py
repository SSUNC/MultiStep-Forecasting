from cProfile import label

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from skforecast.recursive import ForecasterRecursive

#electricity demand
url = "https://raw.githubusercontent.com/tidyverts/tsibbledata/master/data-raw/vic_elec/VIC2015/demand.csv"
df=pd.read_csv(url)
#drop the columns
df.drop(columns=['Industrial'],inplace=True)
df.dtypes

#since the Date is in integer type
#lets covnert it into datetime format
df['date']=df['Date'].apply(
    lambda x:pd.Timestamp('1899-12-30') + pd.Timedelta(x,unit='days')
)

#create a timestampfrom the integer period respresenting 30 minute intervals
df['date_time']=df['date'] + \
                pd.to_timedelta((df['Period']-1)*30,unit='m')
df.dropna(inplace=True)

#selection of columns
#rename of columns
df=df[['date_time','OperationalLessIndustrial']]
df.columns=['date_time','demand']

#set the date_time as index
#resample to hourly
df=(
    df.set_index('date_time')
    .resample('H')
    .agg({'demand':'sum'})
)

#split the data into train and test
end_train='2014-12-31 23:59:59'
x_train=df.loc[:end_train]
x_test=df.loc[end_train:]

#plotting the time series in a line chart
fig,ax=plt.subplots(figsize=(7,3))
x_train.plot(ax=ax,label='train')
x_test.plot(ax=ax,label='test')
ax.set_title('Hourly energy consumption')
ax.legend(['train','test'])
plt.savefig("Hourly enegry consumption.pdf", format='pdf')
plt.close()

#reduce the number of data point in X_test
#to see the graph in a clear way
fig,ax=plt.subplots(figsize=(7,3))
x_train.tail(1000).plot(ax=ax)
x_test.plot(ax=ax)
ax.set_title('Hourly energy demand II')
ax.legend(['train','test'])
plt.savefig('Hourly energy demand II.pdf',format='pdf')
plt.close()

#even after reducing x_train to 1000 data points
#the graph is not still clear
fig,ax=plt.subplots(figsize=(7,3))
x_train.tail(500).plot(ax=ax)
x_test.head(500).plot(ax=ax)
ax.set_title('Hourly energy demand III')
plt.savefig('Hourly energy demand III.pdf',format='pdf')
plt.close()

#create and train forecaster
#define the model
lasso=Lasso(random_state=9)
#define forecasterautoreg from thee skforecast
forecaster=ForecasterRecursive(
    regressor=lasso, #the machine learning model
    lags=[1,24,6*24], #the lags created based on the past data
    forecaster_id='recursive'
)

#fit the forecaster
forecaster.fit(y=x_train['demand'])

forecaster

# predictions
predictions=forecaster.predict(steps= 24)
predictions.head()

#lets plot the predictions and the demand
fig,ax=plt.subplots(figsize=(7,3))
x_train.tail(100).plot(ax=ax,label='train')
x_test.head(24).plot(ax=ax,label='test')
predictions.plot(ax=ax,label='predictions')
plt.title('Lasso forecasting')
plt.ylabel('Energy demand per hour')
ax.legend()
plt.savefig('Lasso forecasting.pdf',format='pdf')
plt.close()

#prediction error-Mean sqared error
error_mse=mean_squared_error(y_true=x_test['demand'].head(24),y_pred=predictions)

print(f'Test error (mse):{error_mse}')
#error root mean square error
error_rmse=mean_squared_error(y_true=x_test['demand'].head(24),y_pred=predictions,squared=False)
print(f'Test Error(RMSE):{error_rmse}')

#predict at aby time point in the future
# we should know the minimum amount of data in the past
#need to create the features of lasso
forecaster.window_size

#lets predict it for the month of february
forecast_start='2015-02-01 00:00:00'
#we need the energy demand up to 144 hours forecast_start

past_data_available=x_test[:'2015-01-31 23:59:59'].tail(144)
past_data_available.head()

predictions=forecaster.predict(
    steps=24,
    last_window=past_data_available['demand']
)
predictions.head()

#plot the forecast vs the actual
fig,ax=plt.subplots(figsize=(7,3))
x_test['2015-01-31 23:59:59':].head(24).plot(ax=ax,label='test')
predictions.plot(ax=ax,label='predictions')
plt.title('Lasso forecasting')
plt.ylabel('Energy demand per hour')
ax.legend()
plt.savefig('Lasso forecasting using last 144 points.pdf',format='pdf')
plt.close()

#prediction error mse
error_mse=mean_squared_error(
    y_true=x_test['2015-01-31 23:59:59':].head(24),y_pred=predictions
)
print(f'Test error(mse):{error_mse}')

#RMSE
error_rmse=mean_squared_error(
    y_true=x_test['2015-01-31 23:59:59':].head(24),y_pred=predictions
,squared=False)
print(f'Test error(mse):{error_rmse}')
