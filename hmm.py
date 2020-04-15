import datetime
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
import os
import pandas as pd
import math
from sklearn.cluster import DBSCAN
import numpy as np
import pickle

MAX_USERS = 1
# Each file corresponds to an entire day, do we consider this one long path? What about different locations throughout the day?
MAX_DAYS_PER_USER = 10

def load_data():
    # Import all data into pandas
    header_list = ['long', 'lat', 'time']
    all_data = pd.DataFrame(columns=header_list)
    cur_user = 0
    for (root, dirs, files) in os.walk('GeolifeTrajectories1.3/Data', topdown=True):
        if cur_user == MAX_USERS:
            break
        if root.rsplit('/', 1)[-1] == "Trajectory":
            # We have enterd a specific user's Trajectory folder
            cur_user += 1 
        user_id = root.rsplit('/', 2)[-2]
        cur_day=0
        for i, fname in enumerate(files):
            if fname[-3:] == "plt":
                cur_day += 1
                ## Get latitude, longitude, and days since 12/30/1899.
                data = pd.read_csv(os.path.join(root, fname), names=header_list, skiprows=6, usecols=[0,1,6])
                # Convert last column to datetime, then extract H:M:S, then convert to timedelta, then seconds since 12 AM
                data['time'] = pd.to_datetime(data['time'])
                data['time'] = data['time'].dt.hour
                data['user_id'] = int(user_id)
                data['path_id'] = i
                all_data = all_data.append(data, ignore_index=True)
                if cur_day == MAX_DAYS_PER_USER:
                    break
    
    all_data = all_data.reset_index()
    return all_data

def cluster_points(df, max_distance, min_samples=5, show=True):
    data = np.float32((np.concatenate([df['long'].tolist()]), np.concatenate([df['lat'].tolist()]))).transpose()
    # Define max_distance (eps parameter in DBSCAN())
    db = DBSCAN(eps=max_distance, min_samples=min_samples).fit(data)
    # Extract a mask of core cluster members
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    # Extract labels (-1 is used for outliers)
    labels = db.labels_
    df = df.assign(cluster_label=labels)
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

def get_centroids(df, show=True):
    # Set up empty container
    labels = {}
    clusters = len(set(df['cluster_label'].tolist()))
    for label in set(df['cluster_label']):
        labels[label] = {}
        labels[label]['long'] = []
        labels[label]['lat'] = []

    # Insert elements into container
    for point in df.itertuples():
        label = point.cluster_label
        labels[label]['long'].append(point.long)
        labels[label]['lat'].append(point.lat)

    # Get average of each cluster
    for label in labels:
        for axis in labels[label]:
            data = labels[label][axis]
            labels[label][axis] = sum(data)/len(data)

    """ Don't need to do this
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
    """

    # Make new df of centroids
    centroids_df = pd.DataFrame()
    longs = []
    lats = []
    del labels[-1]
    for label in labels:
        longs.append(labels[label]['long'])
        lats.append(labels[label]['lat'])
    centroids_df['long'] = longs 
    centroids_df['lat'] = lats 
    if show:
        show_centroids(df, centroids_df)

    return centroids_df

def show_centroids(df, centroids_df):

    # Remove coordinates that do not belong to cluster
    df = df[df['cluster_label'] != -1]
    for x, y in zip(centroids_df['long'].values.tolist(), centroids_df['lat'].values.tolist()):
        plt.scatter(x, y)
    
    plt.title('Centroids')
    plt.show()

""" TODO(Joel): Get this plot to work.
    # Plot up the results!
    fig = plt.figure(figsize=(12,6))
    data = np.float32((np.concatenate([df['long'].tolist()]), np.concatenate([df['lat'].tolist()]))).transpose()

    min_x = np.min(data[:, 0])
    max_x = np.max(data[:, 0])
    min_y = np.min(data[:, 1])
    max_y = np.max(data[:, 1])

    core_samples_mask = np.zeros_like(df['cluster_label'], dtype=bool)
    core_samples_mask[df.index[df['cluster_label'] != -1]] = True
    # The following is just a fancy way of plotting core, edge and outliers
    # Credit to: http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(centroids_df['lat']))]
    for k, col in zip(centroids_df['lat'], colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (df['cluster_label'].tolist() == k)
        xy = data[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=7)

        xy = data[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=3)

    plt.plot(centroids_df['long'], centroids_df['lat'], 'o', markerfacecolor=(0,1,0,1),
            markeredgecolor='k', markersize=20)

    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    plt.title('DBSCAN: %d clusters found' % len(centroids_df), fontsize = 20)
    fig.tight_layout()
    plt.subplots_adjust(left=0.03, right=0.98, top=0.9, bottom=0.05)
    plt.show()
"""

def find_destinations(df, length_of_stay, show=True):
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

def map_points(df, centroids_df, max_distance, show=True):

    df = df.assign(centroid_index=pd.Series([None]*len(df), dtype=pd.Int64Dtype()))
    count = 0
    for i, point in enumerate(df.itertuples()):
        min_centroid = None
        min_distance = None
        for j, centroid in enumerate(centroids_df.itertuples()):
            distance = ((point.long - centroid.long)**2 + (point.long - centroid.long)**2)**.5

            # If centroid too far, skip
            if distance > max_distance:
                continue

            # Check if already centroid assigned
            if not min_centroid:
                min_centroid = centroid
                min_distance = distance
            
            # Check if centroid is closer
            if distance < min_distance:
                min_distance = distance
                min_centroid = centroid
    
        # If centroid found, assign centroid to point
        if min_centroid:
            count += 1
            df.at[point.Index, 'centroid_index'] = min_centroid.Index
    
    new_df = df[df['centroid_index'].notna()]

    if show:
        data = np.float32((np.concatenate([df['long'].tolist()]), np.concatenate([df['lat'].tolist()]))).transpose()
        centroids_data = np.float32((np.concatenate([centroids_df['long'].tolist()]), np.concatenate([centroids_df['lat'].tolist()]))).transpose()

        # Get bounds
        min_x = np.min(data[:, 0])
        max_x = np.max(data[:, 0])
        min_y = np.min(data[:, 1])
        max_y = np.max(data[:, 1])
        
        # Plot all centroids and points
        fig = plt.figure(figsize=(12,6))
        plt.subplot(121)
        plt.plot(data[:,0], data[:,1], 'ko')
        plt.plot(centroids_data[:,0], centroids_data[:,1], 'bo')
        plt.xlim(min_x, max_x)
        plt.ylim(min_y, max_y)
        plt.title('Original Data', fontsize = 20)

        # Plot all centroids and mapped points
        data = np.float32((np.concatenate([new_df['long'].tolist()]), np.concatenate([new_df['lat'].tolist()]))).transpose()
        plt.subplot(122)
        plt.plot(data[:,0], data[:,1], 'ko')
        plt.plot(centroids_data[:,0], centroids_data[:,1], 'bo')
        plt.xlim(min_x, max_x)
        plt.ylim(min_y, max_y)
        plt.title('Mapped {}% of points'.format(float(len(new_df))/float(len(df))*100), fontsize = 20)
        fig.tight_layout()
        plt.subplots_adjust(left=0.03, right=0.98, top=0.9, bottom=0.05)
        plt.show()
    
    return new_df
    
def get_paths(df):

    dfs = []
    for user_id in set(df['user_id'].tolist()):
        for path_id in set(df['path_id'].tolist()):
            path_df = df[df['user_id'] == user_id]
            path_df = path_df[path_df['path_id'] == path_id]
            if path_df.shape[0] > 0:
                dfs.append(path_df)
    
    for i, df in enumerate(dfs):
        indices = []
        old_centroid = -1
        for data in df.itertuples():
            if data.centroid_index == old_centroid:
                indices.append(False)
                continue

            indices.append(True)
            old_centroid = data.centroid_index

        dfs[i] = df[indices].reset_index()

    return dfs

def prepare_data(df):
    # Cluster similar coordinates
    eps = 0.000008
    min_samples = 3
    df = cluster_points(df, eps, min_samples)

    centroids_df = get_centroids(df)
    
    """ TODO(joel): Check to see if we'll ever need this
    # Find destinations based on 30 min stays
    # destinations_df = find_destinations(df, .5)
    """
    # Map points to centroids
    max_distance = .00001
    df = map_points(df, centroids_df, max_distance)
    
    # Get path (remove unbroken chains of GPS coordinates at same location)
    dfs = get_paths(df)
    
    # Split 80/20
    train_df = dfs[:round(len(dfs)*.8)-1]
    test_df = dfs[round(len(dfs)*.8)-1:]
    
    return train_df, test_df
    
class HMM:

    def train(self, dfs):
        print('\nTraining HMM...')

        # Get transition probabilities
        self.transitions = self.get_transition_probabilities(dfs)

        # Get transition probabilities
        self.emissions = self.get_emission_probabilities(dfs)

    def predict(self, dfs):

        predictions = []
        for df in dfs:
            last_location = df['centroid_index'][df.shape[0]-2] # -2 in order to get location before last one
            if last_location >= len(self.transitions):
                predictions.append(-1)
            decision = self.transitions[last_location].index(max(self.transitions[last_location]))
            predictions.append([decision, df['centroid_index'][df.shape[0]-1]])
        
        return predictions

    def get_transition_probabilities(self, dfs):

        n_centroids = max(max(df['centroid_index'].tolist()) for df in dfs)+1
        transitions = [[0 for _ in range(n_centroids)] for _ in range(n_centroids)]
        n_transitions = 0
        for df in dfs:
            old_index = df['centroid_index'].tolist()[0]
            for new_index in df['centroid_index'].tolist():
                if old_index != new_index:
                    old_index = new_index
                    n_transitions += 1
                    transitions[old_index][new_index] += 1

        # Normalize transition probabilities... all of them sum to 1
        transitions = [[num/n_transitions for num in _list] for _list in transitions]

        return transitions

    def get_emission_probabilities(self, dfs):
        n_hours = 24
        emissions = [[0 for _ in range(n_hours)] for _ in range(max(max(df['centroid_index'].tolist()) for df in dfs)+1)]
        n_emissions = 0
        for df in dfs:
            for data in df.itertuples():
                emissions[data.centroid_index][data.time] += 1
                n_emissions += 1

        # Normalize transition probabilities... all of them sum to 1
        emissions = [[num/n_emissions for num in _list] for _list in emissions]

        return emissions


if __name__ == "__main__":
    data = load_data()
    
    train_dfs, test_dfs = prepare_data(data)
    
    with open('train_dfs.data', 'wb') as f:
        pickle.dump(train_dfs, f)

    with open('test_dfs.data', 'wb') as f:
        pickle.dump(test_dfs, f)
    
    with open('train_dfs.data', 'rb') as f:
        train_dfs = pickle.load(f)

    with open('test_dfs.data', 'rb') as f:
        test_dfs = pickle.load(f)

    hmm = HMM()
    hmm.train(train_dfs)
    predictions = hmm.predict(test_dfs)
    print(predictions)
