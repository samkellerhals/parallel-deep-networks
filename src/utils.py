import os
import glob
import json
import pandas as pd

def find_latest_log():
    log_folder = glob.glob('src/log/*')
    latest_file = max(log_folder, key=os.path.getctime)
    return latest_file

def make_df():

    df_list = []

    for i in os.listdir('../data/exp2/'):
        with open('../data/exp2/' + i) as f:
            obj = json.load(f)
            df = pd.json_normalize(obj)
            df_list.append(df)

    metrics = pd.DataFrame()

    for i in range(len(df_list)):
        metrics = metrics.append(df_list[i], ignore_index=True)

    return metrics

if __name__ == "__main__":

    metrics_df = make_df()

    metrics_df.to_csv('log/training_metrics2.csv')