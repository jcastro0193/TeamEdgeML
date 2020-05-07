import os

users = {}

def getUserDataAvailability():
    for (root,dirs,files) in os.walk('GeolifeTrajectories1.3/Data/', topdown=True):
        if root.rsplit('/', 1)[-1] == "Trajectory":
            user_id = root.rsplit('/', 2)[-2]
            lines_count = 0
            for _file in files:
                _file = open(os.path.join(root,_file), "r")
                for line in _file:
                    lines_count += 1
                _file.close()

            users[user_id] = {}
            users[user_id]['days'] = len(files)
            users[user_id]['lines'] = lines_count

def getExtremes():
    users_sorted = list(sorted(users, key=lambda x: users[x]['days']))
    for user in users_sorted:
        print('User: {}: {}'.format(user, users[user]['days']))

if __name__ == "__main__":
    getUserDataAvailability()
    getExtremes()

