# Dependencies
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import kpss, adfuller

class Time_series_preprocessor():
    
    def __init__(self, _id):
        self.id = _id
        self.d = 0

    def fit(self, _time_series):
        self.X = np.arange(0,len(_time_series))
        self.time_series = _time_series
        self.trend = self.calculate_trend(self.X, _time_series)
        return self

    def transform(self):
        """
            Description: Given function takes a time series and passes it through sequence of operation.
                # 1. logarithm_compression (Making time-series homoscedastic)
                # 2. - trend
                # 3. If trend is present or seasonal is present:
                #          perform differencing on whole time series.
            Note: For differencing will be used numpy.diff()
        """
        
        # 1. logarithm_compression (Making time-series homoscedastic)
        val = np.log(2)
        ts_modified = np.log(self.time_series)/val

        # 2. - trend
        ts_modified -= self.trend(self.X)
        
         
        self.last_component = []
        # 3. Testing to stationarity and seasonality and difference ts if required
        while not self.is_stationary(ts_modified):
            self.last_component.append(ts_modified[-1])
            ts_modified = np.diff(ts_modified, n = 1)
            self.d += 1
        
        # 4. Linking with dates
        dates = pd.date_range('1900-1-1', periods=len(ts_modified), freq='D')
        ts_pd = pd.DataFrame({'dates': dates, 'item_cnt_day': ts_modified})
        ts_pd = ts_pd.set_index('dates') 
        
        return ts_pd

    def inverse_transform(self, predicted, offset):
        """
            Description: Given function takes a time series and passes it through sequence of operation.
                # 1. If time_series was differenced then perform addition
                # 2. + Trend
                # 3. logarithm_decompression 
            Note: For differencing will be used numpy.diff()
        """   
       
        # 1. If we differenced ts at forward function then we need to recover it
        if self.d != 0:
            for i in reversed(self.last_component):
                predicted += i
        
        # 2. + trend
        predicted += self.trend(len(self.X) + offset)

        # 3. Logairthmic decompression
        predicted = 2**predicted

        return predicted
    
    def is_stationary(self, time_series):
        """
            Description: Given function performs KPSS & ADF tests for verifying in dataset stationarity.
        """
        self.kpss_p_value = kpss(time_series)[1]
        self.adf_p_value = adfuller(time_series)[1]
        return (self.kpss_p_value > 0.05) and (self.adf_p_value < 0.01)

    def calculate_trend(self, _X, _time_series):
        """
            Description: Given function calculates trend component in time series.
        """
        return np.poly1d(np.polyfit(_X, _time_series, deg = 1))
    
    def summary(self):
      print(f"===| Preprocessor |=======================================================\n")
      print(f"|  Preprocessor ID   |: {self.id}\n")
      print(f"| Time series length |: {len(self.X)}\n")
      print(f"|     Log base       |: 2\n")
      if self.d is not None and self.adf_p_value is not None and self.kpss_p_value is not None:
        print(f"|    d parameter     |: {self.d}\n")
        print(f"| Stationarity tests |: ADF & KPSS\n")
        print(f"|    ADF p-value     |: {self.adf_p_value}\n")
        print(f"|   KPSS p-value     |: {self.kpss_p_value}\n")
      print(f"==========================================================================\n")

    """
    Function for appending time series with new observations
    """
    def update(self, time_series):
        new_time_series = np.append(self.time_series, time_series)
        self.fit(new_time_series)
        ts_pd = self.transform()
        return ts_pd