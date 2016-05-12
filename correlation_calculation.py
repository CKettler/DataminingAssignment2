# -*- coding: utf-8 -*-
"""
correlation features and mood
"""

import data_processing as dr
import numpy as np
import csv
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

data_path_figs = 'C:/Users/Celeste/Documents/GitHub/DataminingAssignment2'

# the dataframe
#filepath = 'data/test_file1.csv'
filepath = 'C:/Users/Celeste/Documents/Masters/Datamining/training_set_VU_DM_2014.csv'
data_aggregator = dr.DataAggregator(filepath)
data, variables, targets = data_aggregator.read()

#data, target, participants, variables, datatime = data_aggregator.read(method='separate')

new_features = []
for variable_name in variables:
    if 'date_time' == variable_name:
        pass
    else:
        a = np.array(data[variable_name])
        new_features.append(np.transpose(a))





features = np.array(new_features)
rows, cols = features.shape
print "rows"
print rows
print "cols"
print cols

corr_and_pvalue = []
name_array1 = []
name_array2 = []
target_names = ['click_bool', 'booking_bool']
k = 0
for i in range(0,rows):
    #for j in range(k,rows):
    #corr_and_pvalue.append(stats.spearmanr(features[i,:],features[j,:]))
    for j in range(2):
        corr_and_pvalue.append(stats.spearmanr(features[i, :], targets[j]))
        name_array1.append(variables[i+1])
        name_array2.append(target_names[j])
    k = k+1

# extract only the correlation
correlations = tuple(x[0] for x in corr_and_pvalue)

# combine correlations and name_array in one 2-dimensional matrix, in which you can see
# which correlation belongs to which name
corr_names_pre = np.vstack((correlations, name_array1))
corr_names2 = np.vstack((corr_names_pre, name_array2)).T
corr_names2 = corr_names2[corr_names2[:,0].argsort(axis=0)]


neg_array = []
pos_array = []
for i in range(0,len(corr_names2)):
    if float(corr_names2[i,0]) < 0:
        neg_array.append(corr_names2[i,:])
    else:
        pos_array.append(corr_names2[i,:])

neg_array_rev = neg_array[::-1]

corr_names2 = np.vstack((neg_array_rev,pos_array))

# plot the correlation against the number of the column of the feature
fig = plt.figure()
errax = fig.add_subplot(1, 1, 1)
errax.set_xlabel('features sorted from lowest correlation to highest correlation')
errax.set_ylabel('correlation coefficient')
errax.plot(corr_names2[:,0])
fig.savefig(data_path_figs + '/correlation_sorted_fig.png')
fig.show()

names = open('correlation_names.txt', 'w')

for values in corr_names2:
    names.write(str(values) + '\n')

names.close()

'''
count = 0
for var in variables:
    tf = df.loc[(df['variable'] == var)]
    tf = tf.ix[pd.to_datetime(tf.index).sort_values()]
    values = tf['value']
    if count == 0:
        data = values
    else:
        print len(df)
        print len(data)
        print len(values)
        data = [data,values]
    count += 1

print data

data = np.array(data)

rows,cols = data.shape

corr_and_pvalue.append()
for i in range(0,cols):
    for j in range(k,cols):
        corr_and_pvalue2.append(stats.spearmanr([:,i],features[:,j]))
        name_array1.append(feature_names[i])
        name_array2.append(feature_names[j])
    k = k+1
'''
#correlation = df['variable'].corr(method = 'pearson')
#print correlation

'''
# read all features
(features1,names1) = f.read_feature_directory(data_path, return_pathes = True)
(features2,names2) = f.read_feature_directory(data_path1, return_pathes = True)
(features3,names3) = f.read_feature_directory(data_path2, return_pathes = True)

# read the prizes
price = f.read_feature_directory(data_path3)

# UNCOMMENT IF YOU WANT TO GET RID OF SOME PRICES
#discard_indices = np.array(np.where(np.any(price>270000, axis=1))[0]).reshape(-1)
#discard_indices = np.array(np.where(np.any(price>1000000, axis=1))[0]).reshape(-1)
#features1 = np.delete(features1, discard_indices, axis=0)
#features2 = np.delete(features2, discard_indices, axis=0)
#features3 = np.delete(features3, discard_indices, axis=0)
#price = np.delete(price, discard_indices, axis=0)

corr_and_pvalue = []
corr_and_pvalue2 = []
correlations = []
neg_array = []
neg_array2 = []
pos_array = []
pos_array2 = []
name_array1 = []
name_array2 = []


# combine all features in one matrix
features_pre = np.hstack((features1, features2))
features = np.hstack((features_pre,features3))

# combina all names in one matrix
feature_names_pre = np.hstack((names1,names2))
feature_names = np.hstack((feature_names_pre, names3))

rows,cols = features.shape

# calculate all correlations and p values: corr_and_pvalue contains tuples of correlation and p value
for i in range(0,cols):
    corr_and_pvalue.append(stats.spearmanr(features[:,i],price))

# extract only the correlation
correlations = tuple(x[0] for x in corr_and_pvalue)

# plot the correlation against the number of the column of the feature
fig = plt.figure()
errax = fig.add_subplot(1, 1, 1)
errax.set_xlabel('features')
errax.set_ylabel('correlation coefficient')
errax.plot(correlations)
fig.savefig(data_path4 + '/correlation_fig.png')
fig.show()


# combine correlations and featurenames in one 2-dimensional matrix, in which you can see
# which correlation belongs to which name
corr_names = np.vstack((correlations, feature_names)).T
corr_names = corr_names[corr_names[:,0].argsort(axis=0)]

for i in range(0,len(corr_names)):
    if float(corr_names[i,0]) < 0:
        neg_array.append(corr_names[i,:])
    else:
        pos_array.append(corr_names[i,:])

neg_array_rev = neg_array[::-1]

corr_names = np.vstack((neg_array_rev,pos_array))


# plot the correlation against the number of the column of the feature
fig = plt.figure()
errax = fig.add_subplot(1, 1, 1)
errax.set_xlabel('features sorted from lowest correlation to highest correlation')
errax.set_ylabel('correlation coefficient')
errax.plot(corr_names[:,0])
fig.savefig(data_path4 + '/correlation_sorted_fig.png')
fig.show()

k=1

rows,cols = features.shape

# calculate all correlations and p values of all features agains all features:
# corr_and_pvalue contains tuples of correlation and p value
for i in range(0,cols):
    for j in range(k,cols):
        corr_and_pvalue2.append(stats.spearmanr(features[:,i],features[:,j]))
        name_array1.append(feature_names[i])
        name_array2.append(feature_names[j])
    k = k+1

# extract only the correlation
correlations2 = tuple(x[0] for x in corr_and_pvalue2)

# plot the correlation against the number of the column of the feature
fig = plt.figure()
errax = fig.add_subplot(1, 1, 1)
errax.set_xlabel('features')
errax.set_ylabel('correlation coefficient')
errax.plot(correlations2)
fig.savefig(data_path4 + '/correlation_fig2.png')
fig.show()


# combine correlations and name_array in one 2-dimensional matrix, in which you can see
# which correlation belongs to which name
corr_names_pre = np.vstack((correlations2, name_array1))
corr_names2 = np.vstack((corr_names_pre, name_array2)).T
corr_names2 = corr_names2[corr_names2[:,0].argsort(axis=0)]

for i in range(0,len(corr_names2)):
    if float(corr_names2[i,0]) < 0:
        neg_array2.append(corr_names2[i,:])
    else:
        pos_array2.append(corr_names2[i,:])

neg_array_rev2 = neg_array2[::-1]

corr_names2 = np.vstack((neg_array_rev2,pos_array2))

# plot the correlation against the number of the column of the feature
fig = plt.figure()
errax = fig.add_subplot(1, 1, 1)
errax.set_xlabel('features sorted from lowest correlation to highest correlation')
errax.set_ylabel('correlation coefficient')
errax.plot(corr_names2[:,0])
fig.savefig(data_path4 + '/correlation_sorted_fig2.png')
fig.show()
'''




