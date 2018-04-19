#!/usr/bin/env python3
'''
AITrain backend
'''
# -*- encoding: utf-8 -*-

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import warnings
import itertools
import threading

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf

import data_parser

MERGE = True
MODEL_EVALUATION = True

EPOCHS = 500
HIDDEN_UNITS = [256, 128, 64]

LABEL = 'GOScore'
OUTPUT = 'Prediction'

def input_fn(data_set, pred=False):

    if pred is False:
        feature_cols_fn = {k: tf.constant(data_set[k].values, shape=[data_set[k].size, 1]) for k in FEATURES}
        labels = tf.constant(data_set[OUTPUT].values)

        return feature_cols_fn, labels

    if pred is True:
        feature_cols_fn = {k: tf.constant(data_set[k].values, shape=[data_set[k].size, 1]) for k in FEATURES}

        return feature_cols_fn

def minmax_scale(dataframe):
    col_dataframe = list(dataframe.columns)
    col_dataframe_bis = list(dataframe.columns)

    col_dataframe_bis.remove(OUTPUT)

    mat_dataframe = np.matrix(dataframe)

    mat_y = np.array(dataframe[OUTPUT].reshape((dataframe.shape[0], 1)))

    prepro_y = MinMaxScaler()
    prepro_y.fit(mat_y)

    prepro = MinMaxScaler()
    prepro.fit(mat_dataframe)

    dataframe = pd.DataFrame(prepro.transform(mat_dataframe), columns=col_dataframe)

    return dataframe, col_dataframe, col_dataframe_bis

def isolation_forest(dataframe):
    clf = IsolationForest(max_samples=100, random_state=42)
    clf.fit(dataframe)
    y_noano = clf.predict(dataframe)
    y_noano = pd.DataFrame(y_noano, columns=['Top'])

    dataframe = dataframe.iloc[y_noano[y_noano['Top'] == 1].index.values]
    dataframe.reset_index(drop=True, inplace=True)

    return dataframe

def create_threads(players_list, players_data):
    threads = [threading.Thread(target=save_to_json, args=(player_data, player,)) for player, player_data in zip(players_list, players_data)]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

def create_dnn(train, evaluation=False):
    train = isolation_forest(train)

    train, COLUMNS, FEATURES = minmax_scale(train)

    feature_cols = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]

    training_set = train[COLUMNS]
    prediction_set = train[OUTPUT]

    if evaluation is True:
        x_train, x_test, y_train, y_test = train_test_split(training_set[FEATURES], prediction_set, test_size=0.1, random_state=42)
        y_train = pd.DataFrame(y_train, columns=[OUTPUT])
        training_set = pd.DataFrame(x_train, columns=FEATURES).merge(y_train, left_index=True, right_index=True)

        training_sub = training_set[col_dataframe]
        y_test = pd.DataFrame(y_test, columns=[OUTPUT])
        testing_set = pd.DataFrame(x_test, columns=FEATURES).merge(y_test, left_index=True, right_index=True)

    tf.logging.set_verbosity(tf.logging.INFO)
    regressor = tf.estimator.DNNRegressor(hidden_units=HIDDEN_UNITS, feature_columns=feature_cols, activation_fn=tf.nn.leaky_relu)

    training_set.reset_index(drop=True, inplace=True)

    regressor.train(input_fn=lambda: input_fn(training_set), steps=EPOCHS)

    if evaluation is True:
        ev = regressor.evaluate(input_fn=lambda: input_fn(testing_set), steps=1)

        loss_score1 = ev['loss']
        print('')
        print('Final Loss on the testing set: {0:f}'.format(loss_score1))
        print('')

        y = regressor.predict(input_fn=lambda: input_fn(testing_set))
        predictions = list(itertools.islice(y, testing_set.shape[0]))

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

    return regressor

def save_to_json(dataframe, name=None, merge=False):
    if merge is True:
        #####SAVE TEAM DATAFRAME data_parser.raw_to_input create_dnn
        print()

    if merge is False:
        #####SAVE PLAYER DATAFRAME data_parser.raw_to_input create_dnn
        print()

def mode_selection(merge, evaluation):
    if evaluation is True:
        create_dnn(data_parser.merge_data(), evaluation)

    else:
        if merge is True:
            save_to_json(data_parser.merge_data(), merge)

        if merge is False:
            players_list, players_data = data_parser.import_data()
            create_threads(players_list, players_data)

def main(argv):
    warnings.filterwarnings('ignore')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    tf.logging.set_verbosity(tf.logging.INFO)
    sess = tf.InteractiveSession()

    mode_selection(MERGE, MODEL_EVALUATION)

if __name__ == '__main__':
    tf.app.run()
