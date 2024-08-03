import numpy as np
import pandas as pd
import os

def load_data(data_path):
    data = pd.read_csv(data_path)
    return data

def load_data_xx(data_path):
    data = pd.read_csv(data_path)
    return data