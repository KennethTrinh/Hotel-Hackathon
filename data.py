import os

os.chdir(os.path.join(os.getcwd(), 'Desktop', 'Hackathon'))

"""
months = ['jan', 'feb', 'mar', 'apr', 'may', 'june', 'july', 'aug', 'sept']


df = pd.read_csv('jan2019.csv', encoding='UTF-16 LE', sep='\t', header=0)
for file in [f'{month}2019.csv' for month in months][1:]:
    df = pd.concat([df, pd.read_csv(file, encoding='UTF-16 LE', sep='\t', header=0)])

df.to_csv('2019.csv', index=False)

df = pd.read_csv('jan2022.csv', encoding='UTF-16 LE', sep='\t', header=0)
for file in [f'{month}2022.csv' for month in months][1:]:
    df = pd.concat([df, pd.read_csv(file, encoding='UTF-16 LE', sep='\t', header=0)])

df.to_csv('2022.csv', index=False)



# check if 2019 in the Arrival Date column
df['Arrival Date'].str.contains('2019').value_counts()
# check where 2019 is not in the Arrival Date column
df[~df['Arrival Date'].str.contains('2019')]
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes

def generateClusters():
    CLUSTERS = 3
    YEAR = 2022
    for CLUSTERS in (3,4,5):
        df = pd.read_csv(f'{YEAR}.csv')
        # mapping = {'1/': 1, '2/': 2, '3/': 3, '4/': 4, '5/': 5, '6/': 6, '7/': 7, '8/': 8, '9/': 9}
        # df['Arrival Date'] = df['Arrival Date'].apply(lambda x: mapping[x[:2]])
        # rename Arrival Date to Month
        # df.rename(columns={'Arrival Date': 'Arrival Month'}, inplace=True)
        # save Arrival Month
        arrivalMonth = df['Arrival Date']
        # drop Room Nights
        df.drop(columns=['Room Nights', 'Arrival Date'], inplace=True)


        # find the columns with string values
        string_cols = [col for col in df.columns if df[col].dtype == 'object']
        categorical_features_idx = [df.columns.get_loc(col) for col in string_cols]
        mark_array=df.values

        #verbose =2
        kproto = KPrototypes(n_clusters=CLUSTERS, verbose=False, max_iter=20).fit(mark_array, categorical=categorical_features_idx)
        clusters = kproto.predict(mark_array, categorical=categorical_features_idx)
        df['cluster'] = list(clusters)

        # add Arrival Month back to the front
        df.insert(0, 'Arrival Month', arrivalMonth)

        #reorder the dataframe based on the cluster
        df = df.sort_values(by=['cluster'])

        # save the clusters
        df.to_csv(f'clusters{CLUSTERS}_{YEAR}.csv', index=False)
        print('Clusters: ', CLUSTERS)
        print(df['cluster'].value_counts())
        print()


files = ['clusters3_2019.csv', 'clusters4_2019.csv', 'clusters5_2019.csv', 
'clusters3_2022.csv', 'clusters4_2022.csv', 'clusters5_2022.csv']

df = pd.read_csv(files[0])

# plot the cluster value counts as a bar chart
# plot the value counts of each cluster on a bar chart as an axis

def nameAxis(num):
    if num == 0:
        return ['First', 'Second', 'Third']
    elif num == 1:
        return ['First', 'Second', 'Third', 'Fourth']
    elif num == 2:
        return ['First', 'Second', 'Third', 'Fourth', 'Fifth']
fig, ax = plt.subplots(2, 3, figsize=(15, 10))
for i, file in enumerate(files): 
    df = pd.read_csv(file)
    # bar graph with using nameAxis function with i%3 to get the correct axis
    ax[i//3, i%3].bar(nameAxis(i%3), df['cluster'].value_counts())
    # set the title
    ax[i//3, i%3].set_title(f'Clusters: {file[-8:-4]}, {file[8]} Clusters')
    # df['cluster'].value_counts().plot(kind='bar', ax=ax[i//3][i%3], title=f'{file[-8:-4]}Clusters: {file[8]}')
    ax[i//3][i%3].set_xlabel('Cluster')
    ax[i//3][i%3].set_ylabel('Count')


# translate the above code to a percentage
fig, ax = plt.subplots(2, 3, figsize=(15, 10))
for i, file in enumerate(files):
    df = pd.read_csv(file)
    # bar graph with using nameAxis function with i%3 to get the correct axis
    ax[i//3, i%3].bar(nameAxis(i%3), df['cluster'].value_counts(normalize=True))
    # set the title
    ax[i//3, i%3].set_title(f'Clusters: {file[-8:-4]}, {file[8]} Clusters')
    # df['cluster'].value_counts().plot(kind='bar', ax=ax[i//3][i%3], title=f'{file[-8:-4]}Clusters: {file[8]}')
    ax[i//3][i%3].set_xlabel('Cluster')
    ax[i//3][i%3].set_ylabel('Percentage')


import seaborn as sns
import matplotlib.pyplot as plt
import datetime
def getDayOfWeek(date):
    """
    date is in the format of 'Month/Day/Year'
    """
    return datetime.datetime.strptime(date, '%m/%d/%Y').strftime('%A')


# files = ['clusters4_2019_10000.csv', 'clusters4_2022_10000.csv']
# df = pd.read_csv(files[0])
# df['Arrival Month'] = df.apply(lambda x: getDayOfWeek(x['Arrival Month']), axis=1)
# df.rename(columns={'Arrival Month': 'Arrival Day of Week'}, inplace=True)

cluster2019 = pd.read_csv('clusters4_2019_10000.csv')
cluster2022 = pd.read_csv('clusters4_2022_10000.csv')

for df in (cluster2019, cluster2022):
    df['Arrival Month'] = df.apply(lambda x: getDayOfWeek(x['Arrival Month']), axis=1)
    df.rename(columns={'Arrival Month': 'Arrival Day of Week'}, inplace=True)

cluster2019Index = cluster2019['cluster'].value_counts().index[0]
cluster2022Index = cluster2022['cluster'].value_counts().index[0]

cluster2019[cluster2019['cluster'] == cluster2019Index]['Arrival Day of Week']
cluster2022[cluster2022['cluster'] == cluster2022Index]['Arrival Day of Week']

# for each cluster, make a boxplot for Booking Channel
fig, ax = plt.subplots(2, 1, figsize=(15, 10))

daysOfWeek = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
colors = ['#00A8E0','#114D97','#EF7722','#83235E', '#CE3262', '#828282', '#1E1E1E']
# pie chart for arrival day of week of cluster 0
ax[0].pie(cluster2019[cluster2019['cluster'] == cluster2019Index]['Arrival Day of Week'].value_counts()[daysOfWeek], labels=daysOfWeek, colors=colors, autopct = '%1.1f%%')
ax[0].set_title('Arrival Day of Week for Top Cluster 2019')
# pie chart for arrival day of week of cluster 1 and use same color scheme as cluster 0
ax[1].pie(cluster2022[cluster2022['cluster'] == cluster2022Index]['Arrival Day of Week'].value_counts()[daysOfWeek], labels=daysOfWeek, colors=colors, autopct = '%1.1f%%')
ax[1].set_title('Arrival Day of Week for Cluster 2022')


fig, ax = plt.subplots(2, 4, figsize=(15, 10))
#plot a pie chart for each cluster in 2019 in the first row
# plot a pie chart for each cluster in 2022 in the second row
for i in range(4):
    ax[0, i].pie(cluster2019[cluster2019['cluster'] == i]['Arrival Day of Week'].value_counts()[daysOfWeek], labels=daysOfWeek, colors=colors, autopct = '%1.1f%%')
    ax[0, i].set_title(f'Arrival Day of Week for Cluster {[3,2,0,1][i]} 2019')
    ax[1, i].pie(cluster2022[cluster2022['cluster'] == i]['Arrival Day of Week'].value_counts()[daysOfWeek], labels=daysOfWeek, colors=colors, autopct = '%1.1f%%')
    ax[1, i].set_title(f'Arrival Day of Week for Cluster {[3,0,2,1][i]} 2022')


# plot a pie chart for Location Type for each cluster in 2019 in the first row and combine 'Resort' and 'Urban' into 'Resort & Urban'
# plot a pie chart for Location Type for each cluster in 2022 in the second row and combine 'Resort' and 'Urban' into 'Resort & Urban'
fig, ax = plt.subplots(2, 4, figsize=(15, 10))

# combine 'Resort' and 'Urban' into 'Resort & Urban'
cluster2019['Location Type'] = cluster2019['Location Type'].apply(lambda x: 'Resort & Urban' if x == 'Resort' or x == 'Urban' else x)
cluster2022['Location Type'] = cluster2022['Location Type'].apply(lambda x: 'Resort & Urban' if x == 'Resort' or x == 'Urban' else x)

for i in range(4):
    ax[0, i].pie(cluster2019[cluster2019['cluster'] == i]['Location Type'].value_counts(), labels=cluster2019[cluster2019['cluster'] == i]['Location Type'].value_counts().index, autopct = '%1.1f%%')
    ax[0, i].set_title(f'Location Type for Cluster {[3,2,0,1][i]} 2019')
    ax[1, i].pie(cluster2022[cluster2022['cluster'] == i]['Location Type'].value_counts(), labels=cluster2022[cluster2022['cluster'] == i]['Location Type'].value_counts().index, autopct = '%1.1f%%')
    ax[1, i].set_title(f'Location Type for Cluster {[3,0,2,1][i]} 2022')




# df = pd.read_csv('2019.csv')

# arrivalMonth = df['Arrival Date']
# # drop Room Nights
# df.drop(columns=['Room Nights', 'Arrival Date'], inplace=True)

# # find the columns with string values
# string_cols = [col for col in df.columns if df[col].dtype == 'object']
# categorical_features_idx = [df.columns.get_loc(col) for col in string_cols]
# mark_array=df.values

# #verbose =2
# kproto = KPrototypes(n_clusters=CLUSTERS, verbose=True, max_iter=50).fit(mark_array, categorical=categorical_features_idx)
# clusters = kproto.predict(mark_array, categorical=categorical_features_idx)
# df['cluster'] = list(clusters)




