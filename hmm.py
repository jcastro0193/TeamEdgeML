import datetime
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
import os
import pandas as pd
import math
from sklearn.cluster import DBSCAN
import numpy as np

MAX_USERS = 1
# Each file corresponds to an entire day, do we consider this one long path? What about different locations throughout the day?
MAX_DAYS_PER_USER = 20

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
                data['user_id'] = int(user_id)
                all_data = all_data.append(data, ignore_index=True)
                if cur_day == MAX_DAYS_PER_USER:
                    break
    
    db = DBSCAN(eps=0.00005,  min_samples=10).fit(all_data[['long', 'lat']].values)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    all_data['cluster_label'] = labels
    all_data = all_data.iloc[core_samples_mask]
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    unique_labels = set(labels)
    locations = {}
    for label in unique_labels:
        locations[label] = []

    all_data = all_data.reset_index()
    return all_data

def myround(x, base):
    return base * round(x/base)

def cluster_points(df, max_distance, points_to_use=50, show=False):
    df = df[:(points_to_use if points_to_use < len(df) else len(df))]
    data = np.float32((np.concatenate([df['long'].tolist()]), np.concatenate([df['lat'].tolist()]))).transpose()
    # Define max_distance (eps parameter in DBSCAN())
    db = DBSCAN(eps=max_distance, min_samples=10).fit(data)
    # Extract a mask of core cluster members
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    # Extract labels (-1 is used for outliers)
    labels = db.labels_
    df['cluster_label'] = pd.Series(labels)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    unique_labels = set(labels)
    if show:
        # Plot up the results!
        min_x = np.min(data[:, 0])
        max_x = np.max(data[:, 0])
        min_y = np.min(data[:, 1])
        max_y = np.max(data[:, 1])

        fig = plt.figure(figsize=(12,6))
        plt.subplot(121)
        plt.plot(data[:,0], data[:,1], 'ko')
        plt.xlim(min_x, max_x)
        plt.ylim(min_y, max_y)
        plt.title('Original Data', fontsize = 20)

        plt.subplot(122)
        # The following is just a fancy way of plotting core, edge and outliers
        # Credit to: http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = (labels == k)
            xy = data[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=7)

            xy = data[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=3)
    
        plt.xlim(min_x, max_x)
        plt.ylim(min_y, max_y)
        plt.title('DBSCAN: %d clusters found' % n_clusters, fontsize = 20)
        fig.tight_layout()
        plt.subplots_adjust(left=0.03, right=0.98, top=0.9, bottom=0.05)
        plt.show()
    
    return df

def get_centroids(df, show=False):
    # Set up empty container
    labels = {}
    clusters = len(set(df['cluster_label'].tolist()))
    for label in set(df['cluster_label']):
        labels[label] = {}
        labels[label]['long'] = []
        labels[label]['lat'] = []

    count = 0
    for key1 in labels:
        for key2 in labels[key1]:
            count+=1
    print("count1:", count)


    # Insert elements into container
    for point in df.itertuples():
        label = point.cluster_label
        labels[label]['long'].append(point.long)
        labels[label]['lat'].append(point.lat)

    count = 0
    for key1 in labels:
        for key2 in labels[key1]:
            count+=1
    print("count2:", count)

    # Get average of each cluster
    for label in labels:
        for axis in labels[label]:
            data = labels[label][axis]
            labels[label][axis] = sum(data)/len(data)
    
    count = 0
    for key1 in labels:
        for key2 in labels[key1]:
            count+=1
    print("count3:", count)

    # Replace old datapoints with new averages
    points = {}
    for point in df.itertuples():
        label = point.cluster_label
        if label == -1:
            continue
        df.loc[point.Index, 'long'] = labels[label]['long']
        df.loc[point.Index, 'lat'] = labels[label]['lat']

    print("points:", len(set(df['long'])))
    print("points:", len(set(df['lat'])))
    
    if show:
        show_centroids(df)

    return df

def show_centroids(df):
    for x, y in zip(df['long'].values.tolist(), df['lat'].values.tolist()):
        plt.scatter(x, y)

    plt.ylabel('locations')
    plt.show()

""" TODO(joel): Make this plot work, looks nicer
    if True:
        # Plot up the results!
        min_x = np.min(data['long'])
        max_x = np.max(data['lat'])
        min_y = np.min(data['long'])
        max_y = np.max(data['lat'])
        print(min_x, max_x, min_y, max_y)

        fig = plt.figure(figsize=(12,6))
        plt.subplot(121)
        plt.plot(data['long'], data['lat'], 'ko')
        plt.xlim(min_x, max_x)
        plt.ylim(min_y, max_y)
        plt.title('Original Data', fontsize = 20)

        plt.subplot(122)
    
        plt.xlim(min_x, max_x)
        plt.ylim(min_y, max_y)
        plt.title('DBSCAN: %d clusters found' % (len(set(data['cluster_label']))-1), fontsize = 20)
        fig.tight_layout()
        plt.subplots_adjust(left=0.03, right=0.98, top=0.9, bottom=0.05)
        plt.show()
"""

def find_destinations(df, length_of_stay, show=False):
    length_of_stay *= 60 #convert to minutes
    last_time = df['time'][0]
    last_lat = df['lat'][0]
    last_long = df['long'][0]
    last_user = df['user_id'][0]
    start_destination_index = 0
    times = []
    lats = []
    longs = []
    users = []
    indices = []
    destinations = {}
    sequence_length = 0
    for i in range(len(df['lat'])):
        if df['lat'][i] != last_lat or df['long'][i] != last_long:
            if df['time'][i-1] - last_time >= length_of_stay:
                if (last_lat,last_long) in destinations.keys():
                    destinations[(last_lat,last_long)].append(last_time)
                else:
                     destinations[(last_lat,last_long)] = [last_time]
            last_time = df['time'][i]
            last_lat = df['lat'][i]
            last_long = df['long'][i]
            last_user = df['user_id'][i]
            start_destination_index = i

    df['time'][i] = last_time

    destinations_df = pd.DataFrame()
    destinations_list = []
    times_list = []
    for i, d in enumerate(destinations):
        for time in destinations[d]:
            destinations_list.append(i)
            times_list.append(time)
            lats.append(d[0])
            longs.append(d[1])

    destinations_df['destination'] = destinations_list
    destinations_df['time'] = times_list
#    destinations_df['user_id'] = users
    destinations_df['long'] = longs
    destinations_df['lat'] = lats
    
    if show:
        show_destinations(destinations_df.concat)

    return destinations_df[['destination', 'time']]
    
def show_destinations(df):
    plt.scatter(df['lat'], df['long'])
    plt.ylabel('locations')
    plt.show()

def HMM(X):
    # Cluster similar coordinates
    X = cluster_points(X, .00003, points_to_use=500, show=True)
    X = get_centroids(X, show=True)
    a
    
    # Find destinations based on 30 min stays
    destinations_df = find_destinations(X, .5)
    # Split 80/20
    train_set = X[:round(len(X)*.8)]
    test_set = X[round(len(X)*.8):]
    print('\nTraining HMM...')
    # TODO: Look into what number this should be
    model = hmm.GaussianHMM(n_components=8, covariance_type='diag', n_iter=1000)
    X = train_set.values.tolist()
    model.fit(X)
    
    hidden_states = model.predict(test_set[['time']])
    print(test_set['time'])
    print("test_set")
    print(test_set)
    print("hidden_states")
    print(hidden_states)

if __name__ == "__main__":
    data = loadData()
    HMM(data)
