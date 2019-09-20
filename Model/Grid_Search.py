
import pandas as pd
import numpy as np
from pmdarima.arima import auto_arima

class Grid_Search:

    def __init__(self, start_p = 1, start_q = 0, max_p = 5, max_q = 2):
        self.max_p = max_p
        self.max_q = max_q
        self.start_p = start_p
        self.start_q = start_q

    def fit(self, time_series):
        return auto_arima(time_series, stationary=True, seasonal=False, start_p=self.start_p, start_q = self.start_q, max_p=self.max_p, max_q=self.max_q, error_action='ignore', stepwise=False, njobs = 2)