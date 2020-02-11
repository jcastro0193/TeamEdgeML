import datetime
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
import os
import pandas as pd
from localization import Localization

MAX_USERS = 1
# Each file corresponds to an entire day, do we consider this one long path? What about different locations throughout the day?
MAX_DAYS_PER_USER = 1

def loadData():
    # Import all data into pandas
    header_list = ['long', 'lat', 'time']
    all_data = pd.DataFrame(columns=header_list)
    cur_user = 0
    cur_day=0
    for (root, dirs, files) in os.walk('GeolifeTrajectories1.3/Data', topdown=True):
        if cur_user == MAX_USERS:
            break
        if root.rsplit('/', 1)[-1] == "Trajectory":
            # We have enterd a specific user's Trajectory folder
            cur_user += 1 
        user_id = root.rsplit('/', 2)[-2]
        for fname in files:
            if fname[-3:] == "plt":
                cur_day += 1
                ## Get latitude, longitude, and days since 12/30/1899.
                data = pd.read_csv(os.path.join(root, fname), names=header_list, skiprows=6, usecols=[0,1,6])
                # Convert last column to datetime, then extract H:M:S, then convert to timedelta, then seconds since 12 AM
                data['time'] = pd.to_timedelta(pd.to_datetime(data['time']).dt.strftime("%H:%M:%S"))
                data['time'] = data['time'].dt.total_seconds()
                # TODO: Associate times to certain day of week or week, then cluster
                data['user_id'] = int(user_id)
                all_data = all_data.append(data)
                if cur_day == MAX_DAYS_PER_USER:
                    break
    return all_data

def HMM(X):
    # Split 80/20
    train_set = X.sample(frac=0.8, random_state=0)
    test_set = X.drop(train_set.index)

    print('\nTraining HMM...')
    # TODO: Look into what number this should be
    num_components = 1
    model = hmm.GaussianHMM(n_components=num_components, covariance_type='diag', n_iter=1000)
    model.fit(train_set)

    hidden_states = model.predict(test_set)

    for i, state in enumerate(hidden_states):
        print(i, state)
 

if __name__ == "__main__":
    data = loadData()
    HMM(data)
