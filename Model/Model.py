import pandas as pd
import numpy as np
from .Grid_Search import Grid_Search
from .Preprocessor import Time_series_preprocessor
import pickle
import os

class Pipeline:
    
    def __init__(self, id):
        if (os.path.exists(f"{id}.pickle")):
            with open(f"{id}.pickle", 'rb') as file:
                tmp_dict = pickle.load(file)
            self.__dict__.update(tmp_dict)
        else:
            self.__id = id
            self.__preprocessor = Time_series_preprocessor(id)
            self.__model = None
            self.__predicted = None

    def fit(self, time_series):
       
        self.__preprocessor.fit(time_series)
        self.__model = Grid_Search().fit( self.__preprocessor.transform() )
        
        # Predicting...
        predicted = self.__model.predict(2)

        transformed = []
        transformed.append(self.__preprocessor.inverse_transform(predicted[0], 0))
        transformed.append(self.__preprocessor.inverse_transform(predicted[1], 1))

        self.__predicted = np.array(transformed)
        self.__save()

    def predict(self):
        return self.__predicted


    """
    Function for updating model with new observations and making new prediction based on them
    """
    def update(self, time_series):
        self.__model = Grid_Search().fit(self.__preprocessor.update(time_series))
        
        predicted = self.__model.predict(2)

        transformed = []
        transformed.append(self.__preprocessor.inverse_transform(predicted[0], 0))
        transformed.append(self.__preprocessor.inverse_transform(predicted[1], 1))

        self.__predicted = np.array(transformed)
        self.__save()

    """
    Function for serializing class object for future use
    """
    def __save(self):
        with open(f"{self.__id}.pickle", 'wb') as file:
            pickle.dump(self.__dict__, file)
    
       
    def summary(self):
        print(f"====| Model |=============================================================\n")
        print(f"|  Model id   |: {self.__id}\n")
        if self.__predicted is not None:
            print(f"|  predicted  |: {self.__predicted}\n")
        print(f"==========================================================================\n")
        self.__preprocessor.summary()