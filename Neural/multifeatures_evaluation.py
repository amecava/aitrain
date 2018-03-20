#!/usr/bin/env python3
'''
No name project
'''
# -*- encoding: utf-8 -*-

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import warnings
import itertools

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
'''
from sklearn.ensemble import IsolationForest
'''

#User input data

PLAYER = 'team'

'''
WL_MAX = 5
MATCH_DAY = 3
WORKLOAD_PREC = [3, 3, 3, 3, 3, 3, 3]
WELLNESS_VALUES = [4, 4, 4, 4, 4]
'''

# Initial parameters setup

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.logging.set_verbosity(tf.logging.INFO)
sess = tf.InteractiveSession()

# Initial variable setup

EPOCHS = 1000

FOLDER_PATH1 = os.path.dirname(os.path.abspath(__file__))
FOLDER_PATH2 = 'datasets'
INPUT_TRAINING = 'input_training.pickle'
INPUT_WELLNESS = 'input_wellness.pickle'
'''
INPUT_DATABASE = 'input_database.pickle'
'''

LABEL = 'GO Score'
OUTPUT = 'GO Score Prediction'
WORKLOAD = 'workload'
WELLNESS = ['wellness_sleep', 'wellness_stress', 'wellness_fatigue', 'wellness_soreness', 'wellness_mood']

SLEEP_PAR = [0.1, 0.1, 0.5, 0.8, 1]
STRESS_PAR = [0.1, 0.1, 1, 1, 1]
FATIGUE_PAR = [0.1, 0.1, 0.5, 0.8, 1]
SORENESS_PAR = [0.1, 0.1, 0.5, 0.8, 1]
MOOD_PAR = [0.1, 0.1, 1, 1, 1]

# Import and separate players training

def player_filter(p):

    return dataset1[dataset1['PlayerId'] == p]

dataset1 = pd.read_pickle(os.path.join(FOLDER_PATH1, FOLDER_PATH2, INPUT_TRAINING))

for i in range(1, 22):
    locals()['player' + str(i)] = player_filter('Player' + str(i))

for i in range(1, 22):
    locals()['player' + str(i)] = locals()['player' + str(i)].sort_values('Date').drop('PlayerId', axis=1)

# Import and separate players wellness

def player_filter2(p):

    return dataset2[dataset2['player'] == p]

dataset2 = pd.read_pickle(os.path.join(FOLDER_PATH1, FOLDER_PATH2, INPUT_WELLNESS))

for i in range(1, 22):
    locals()['player' + str(i) + '_w'] = player_filter2('Player' + str(i)).sort_values('date').drop('player', axis=1)

# Join training and wellness

def join_function(player, player_w):

    min_date = list(player_w['date'])[0]
    max_date = list(player_w['date'])[-1]
    idx = pd.date_range(min_date, max_date)
    player_w = player_w.set_index('date')

    player_w = player_w[~player_w.index.duplicated(keep='first')] # Needed for player5 dataset

    player_w = player_w.reindex(idx, fill_value=np.nan)
    player_w = player_w.fillna(player_w.mean())
    database = player_w.join(player.set_index('Date'))
    database = database.fillna(0)
    return database

for i in range(1, 22):
    locals()['dataset' + str(i)] = join_function(locals()['player' + str(i)], locals()['player' + str(i) + '_w'])
    '''
    locals()['dataset' + str(i)].to_pickle(FOLDER_PATH + 'dataset' + str(i) + '.pickle')
    '''

# Create team dataset

frames = []

for i in range(1, 22):
    frames.append(locals()['dataset' + str(i)])

datasetteam = pd.concat(frames)

'''
datasetteam.to_pickle(FOLDER_PATH + 'dataset' + 'team' + '.pickle')
'''

# Import selected dataset

train = locals()['dataset' + PLAYER]
print('List of features contained in dataset:', list(train.columns))

# Training categorization

'''
def training_categorization(Prod, MD, WL):

    if MD > 0:
        MD = MD - 7

    if MD == -5:
        if Prod % 60 == 0:
            if WL > (WL_MAX * 45 / 100) and (WL < (WL_MAX * 55 / 100)):
                return 1

    if MD == -4:
        if Prod % 24696 == 0:
            if WL > (WL_MAX * 60 / 100) and (WL < (WL_MAX * 80 / 100)):
                return 1

    if MD == -3:
        if Prod % 2352 == 0:
            if WL > (WL_MAX * 40 / 100) and (WL < (WL_MAX * 50 / 100)):
                return 1

    if MD == -2:
        if Prod % 10 == 0:
            if WL > (WL_MAX * 10 / 100) and (WL < (WL_MAX * 20 / 100)):
                return 1

    if MD == -1:
        if Prod % 30 == 0:
            if WL > (WL_MAX * 15 / 100) and (WL < (WL_MAX * 25 / 100)):
                return 1

    else:
        return 0
'''

# Input database

'''
training_database = pd.read_pickle(os.path.join(FOLDER_PATH1, FOLDER_PATH2, INPUT_DATABASE))

workload_columns = []

for shift in reversed(range(1, 8)):
    workload_columns.append(WORKLOAD + '_-_' + str(shift))

filtered_database = pd.DataFrame(columns = workload_columns + WELLNESS + list(training_database.drop(['Workload', 'Prodotti'], axis=1).columns.values))

i = 0
j = 0

for i in range(len(list(training_database['Workload']))):
    if(training_categorization(training_database['Prodotti'][i], MATCH_DAY, training_database['Workload'][i])) == 1:
        filtered_database.loc[i] = WORKLOAD_PREC + WELLNESS_VALUES + list(training_database.drop(['Workload', 'Prodotti'], axis=1).loc[i])
        j += 1
'''

# Import selected input dataset

'''
input_dataset = filtered_database
input_dataset = input_dataset.select_dtypes(exclude=['object'])
'''

# Remove spaces from features

list_features = []

for feature in list(train.columns.values):
    list_features.append(feature.replace(' ', '_'))

train.columns = list_features

'''
list_features = []

for feature in list(input_dataset.columns.values):
    list_features.append(feature.replace(' ', '_'))

input_dataset.columns = list_features
'''

# Remove spaces from defined features

LABEL = LABEL.replace(' ', '_')
OUTPUT = OUTPUT.replace(' ', '_')
WORKLOAD = WORKLOAD.replace(' ', '_')

for feature in range(0, 5):
    WELLNESS[feature] = WELLNESS[feature].replace(' ', '_')

# Shift work load and drop current

for shift in reversed(range(1, 8)):
    train[WORKLOAD + '_-_' + str(shift)] = train[WORKLOAD].shift(periods=shift, freq=None, axis=0)

train = train.drop(WORKLOAD, axis=1)

# Go Score algorithm and shift for prediction

def find_go(sleep, stress, fatigue, soreness, mood):

    try:
        return SLEEP_PAR[int(sleep) - 1] + STRESS_PAR[int(stress) - 1] + FATIGUE_PAR[int(fatigue) - 1] + SORENESS_PAR[int(soreness) - 1] + MOOD_PAR[int(mood) - 1]
    except BaseException:
        return np.nan

train[LABEL] = list(map(find_go, train[WELLNESS[0]], train[WELLNESS[1]], train[WELLNESS[2]], train[WELLNESS[3]], train[WELLNESS[4]]))
'''
input_dataset[LABEL] = list(map(find_go, input_dataset[WELLNESS[0]], input_dataset[WELLNESS[1]], input_dataset[WELLNESS[2]], input_dataset[WELLNESS[3]], input_dataset[WELLNESS[4]]))
'''

train[OUTPUT] = train[LABEL].shift(periods=-1, freq=None, axis=0)

# Drop wellness and NaN values

train = train.drop([WELLNESS[0], WELLNESS[1], WELLNESS[2], WELLNESS[3], WELLNESS[4]], axis=1)
'''
input_dataset = input_dataset.drop([WELLNESS[0], WELLNESS[1], WELLNESS[2], WELLNESS[3], WELLNESS[4]], axis=1)
'''

train = train.dropna()
'''
input_dataset = input_dataset.dropna()
'''

# Input dataset parser

'''
def input_dataset_parser(train_v, input_dataset_v):

    new_input = pd.DataFrame()

    for feature2 in list(train_v.columns):
        found = 0
        for input_feature in list(input_dataset_v.columns):
            if input_feature == feature2:
                new_input[feature2] = input_dataset_v[feature2]
                found = 1
                break
        if found == 0 and feature2 != OUTPUT:
            new_input[feature2] = 0

    return new_input

input_dataset = input_dataset_parser(train, input_dataset)
'''

# Isolate outliers with an IsolationForest algorithm

'''
clf = IsolationForest(max_samples=100, random_state=42)
clf.fit(train)
y_noano = clf.predict(train)
y_noano = pd.DataFrame(y_noano, columns=['Top'])
y_noano[y_noano['Top'] == 1].index.values

train = train.iloc[y_noano[y_noano['Top'] == 1].index.values]
train.reset_index(drop=True, inplace=True)
print('')
print('Number of Outliers:', y_noano[y_noano['Top'] == -1].shape[0])
print('Number of rows without outliers:', train.shape[0])
'''

# Preprocessing dataset with MinMaxScale function

col_train = list(train.columns)
col_train_bis = list(train.columns)

col_train_bis.remove(OUTPUT)

mat_train = np.matrix(train)
'''
mat_input = np.matrix(input_dataset)
'''
mat_new = np.matrix(train.drop(OUTPUT, axis=1))
mat_y = np.array(train[OUTPUT].reshape((train.shape[0], 1)))

prepro_y = MinMaxScaler()
prepro_y.fit(mat_y)

prepro = MinMaxScaler()
prepro.fit(mat_train)

'''
prepro_input = MinMaxScaler()
prepro_input.fit(mat_new)
'''

train = pd.DataFrame(prepro.transform(mat_train), columns=col_train)
'''
input_dataset = pd.DataFrame(prepro_input.transform(mat_input), columns=col_train_bis)
'''

# TensorFlow deep neural network

def input_fn(data_set, pred=False):

    if pred is False:
        feature_cols_fn = {k: tf.constant(data_set[k].values, shape=[data_set[k].size, 1]) for k in FEATURES}
        labels = tf.constant(data_set[OUTPUT].values)

        return feature_cols_fn, labels

    if pred is True:
        feature_cols_fn = {k: tf.constant(data_set[k].values, shape=[data_set[k].size, 1]) for k in FEATURES}

        return feature_cols_fn

# List of features

COLUMNS = col_train
FEATURES = col_train_bis

# Columns for tensorflow

feature_cols = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]

# Training set and Prediction set with the features to predict

training_set = train[COLUMNS]
prediction_set = train[OUTPUT]

# Train and Test split

x_train, x_test, y_train, y_test = train_test_split(training_set[FEATURES], prediction_set, test_size=0.1, random_state=42)
y_train = pd.DataFrame(y_train, columns=[OUTPUT])
training_set = pd.DataFrame(x_train, columns=FEATURES).merge(y_train, left_index=True, right_index=True)

# Training for submission

training_sub = training_set[col_train]
y_test = pd.DataFrame(y_test, columns=[OUTPUT])
testing_set = pd.DataFrame(x_test, columns=FEATURES).merge(y_test, left_index=True, right_index=True)


# Model

tf.logging.set_verbosity(tf.logging.INFO)
regressor = tf.contrib.learn.DNNRegressor(hidden_units=[256, 128, 64], feature_columns=feature_cols, activation_fn=tf.nn.leaky_relu, optimizer=tf.train.AdagradOptimizer(learning_rate=1e-1, initial_accumulator_value=1e-2)) # model_dir = 'regressor'

# Reset the index of training

training_set.reset_index(drop=True, inplace=True)

# Deep Neural Network Regressor with the training set which contain the data split by train test split

regressor.fit(input_fn=lambda: input_fn(training_set), steps=EPOCHS)

# Evaluation on the test set created by train_test_split

ev = regressor.evaluate(input_fn=lambda: input_fn(testing_set), steps=1)

# Display the score on the testing set

loss_score1 = ev['loss']
print('')
print('Final Loss on the testing set: {0:f}'.format(loss_score1))
print('')

# Predictions on testing set

y = regressor.predict_scores(input_fn=lambda: input_fn(testing_set))
predictions = list(itertools.islice(y, testing_set.shape[0]))

# Plot predictions x reality on dataset graph

predictions = prepro_y.inverse_transform(np.array(predictions).reshape(len(x_test), 1))
reality = pd.DataFrame(prepro.inverse_transform(testing_set), columns=[COLUMNS])[OUTPUT].values

font = {'family' : 'normal',
        'size'   : 10}

matplotlib.rc('font', **font)

fig, ax = plt.subplots(figsize=(50, 40))

axes = plt.gca()
axes.set_xlim([1, 5])
axes.set_ylim([1, 5])

plt.style.use('default')
plt.plot(predictions, reality, 'ro', color='red')
plt.xlabel('Predictions', fontsize=20)
plt.ylabel('Reality', fontsize=20)
plt.title('Predictions x Reality on dataset', fontsize=30)
ax.plot([1.125, 4.875], [1.125, 4.875], 'k--', lw=4)
plt.show()

# Prediction on input dataset

'''
y_predict = regressor.predict_scores(input_fn=lambda: input_fn(input_dataset, pred=True))

def to_submit(pred_y):

    y_predict2 = list(itertools.islice(pred_y, input_dataset.shape[0]))
    y_predict2 = pd.DataFrame(prepro_y.inverse_transform(np.array(y_predict2).reshape(len(y_predict2), 1)), columns=[OUTPUT])

    return y_predict2

results = to_submit(y_predict)

print('')
print(input_dataset)
print('')
print(results)
'''
