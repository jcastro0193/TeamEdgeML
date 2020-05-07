Function: load_data()

Description: This function serves to import data from the GeoLifeDataset. If a user is specified via the keyword argument 'user', then only the data from that specific user will be loaded. A certin amount of weeks' worth of data can be loaded via the 'n_weeks' keyword argument. For each user, the function automatically finds the start date (which is a certain file) that leads to the most amount of available data to work with. The maximum amount of users and maximum amount of days per user can also be specified via the 'max_users' and 'max_days_per_user' keyword arguments. If 'max_days_per_user' and 'n_weeks' is specified, the function will stop loading data once the one of the two thresholds (# of days or # of weeks) has been reached.

Function: cluster_points()

Description: This function serves to find clusters within the points using the DB_SCAN function. DB_SCAN proved to not be the best solution to clustering points, thus, this needs to be updated. 'max_distance' is the maximum amount of distance (distance is defined by the cartesian distance between two GPS coordinates) that is allowed between two points before they are not considered to be in the same cluster. min_samples is the amount of points that should surround a point that is considered a core point of the cluster.

Function: get_centroids()

Description: This function serves to calculate the center of each cluster by taking the average of all the points in the cluster.

Function: map_points()

Description: This function maps GPS coordinates to one of the identified clusters. 'max_distance' can be set in order to specify how far a point should be from a centroid before it is too far to be mapped to it.

Function: get_paths()

Descriptioin: This function splits the large dataframe that contains all GPS coordinates into seperate dataframes where each frame corresponds to a path (defined as a user's route throughout the day).

Function: prepare_data()

Description: This function splits the data into test and training sets.

Function: plot_final_accuracies()

Description: This function makes a grouped bar graph of the results of running the tests.
