import os
import glob

def find_latest_log():
    log_folder = glob.glob('log/*')
    latest_file = max(log_folder, key=os.path.getctime)
    return latest_file