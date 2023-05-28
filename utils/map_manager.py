import pickle
import pandas as pd

def load_pickle_map(file_path):
    with open(file_path, "rb") as fp:
        return pickle.load(fp)
    return

def load_csv_path(file_path):
    return pd.read_csv(file_path)