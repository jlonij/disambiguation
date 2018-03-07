#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# DAC Entity Linker
#
# Copyright (C) 2017-2018 Koninklijke Bibliotheek, National Library of
# the Netherlands
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import argparse
import json
import math
import os

import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedShuffleSplit

from keras.constraints import maxnorm
from keras.layers import concatenate
from keras.layers import Dense
from keras.layers import dot
from keras.layers import Dropout
from keras.layers import Input
from keras.models import load_model
from keras.models import Model
from keras.models import Sequential

np.random.seed(1337)

abs_path = os.path.dirname(os.path.realpath(__file__))
training_file = os.path.join(abs_path, 'training', 'training.csv')
feature_file_template = os.path.join(abs_path, 'features', '{}')
model_file_template = os.path.join(abs_path, 'models', '{}')


class BaseModel:
    def __init__(self):
        self.features = self.load_features('features.json')

    def load_features(self, feature_file):
        path = feature_file_template.format(feature_file)
        return json.load(open(path))['features']


class LinearSVM(BaseModel):
    def __init__(self, train=False):
        self.features = self.load_features('svm.json')
        self.model_file = model_file_template.format('svm.pkl')

        if train:
            self.load_csv()
            self.model = svm.SVC(kernel='linear', C=1.5,
                                 decision_function_shape='ovr',
                                 class_weight={0: 0.25, 1: 0.75})
        else:
            self.model = joblib.load(self.model_file)

    def load_csv(self):
        '''
        Transform tabular data set into NumPy arrays.
        '''
        print('Loading training set: {}'.format(training_file))
        df = pd.read_csv(training_file, sep='\t')

        self.data = df[self.features].as_matrix()
        self.labels = df[['label']].as_matrix().reshape(-1)
        preprocessing.LabelBinarizer().fit(self.labels)

        print('Number of examples: {}'.format(self.data.shape[0]))
        print('Number of features: {}'.format(self.data.shape[1]))

    def train(self):
        '''
        Train and save model.
        '''
        print('Training new model: {}()'.format(self.__class__.__name__))
        self.model.fit(self.data, self.labels)

        print('Saving model: {}'.format(self.model_file))
        joblib.dump(self.model, self.model_file)

    def validate(self):
        '''
        Ten-fold cross-validation with stratified sampling.
        '''
        print('Validating new model: {}()'.format(self.__class__.__name__))

        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []

        sss = StratifiedShuffleSplit(n_splits=10)
        for train_index, test_index in sss.split(self.data, self.labels):
            x_train, x_test = self.data[train_index], self.data[test_index]
            y_train, y_test = self.labels[train_index], self.labels[test_index]
            self.model.fit(x_train, y_train)

            y_pred = self.model.predict(x_test)
            accuracy_scores.append(accuracy_score(y_test, y_pred))
            precision_scores.append(precision_score(y_test, y_pred))
            recall_scores.append(recall_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred))

        print('Accuracy: {}'.format(np.mean(accuracy_scores)))
        print('Precision: {}'.format(np.mean(precision_scores)))
        print('Recall: {}'.format(np.mean(recall_scores)))
        print('F1-measure: {}'.format(np.mean(f1_scores)))

    def weights(self):
        '''
        Print model feature weights.
        '''
        for i, f in enumerate(self.features):
            print(f, self.model.coef_[:, i][0])

    def predict(self, example):
        '''
        Classify a new example.
        '''
        dec = self.model.decision_function([example])[0]
        prob = 1 / (1 + math.exp(dec * -1))
        return prob


class NeuralNet(BaseModel):
    def __init__(self, train=False):
        self.features = self.load_features('nn.json')
        self.model_file = model_file_template.format('nn.h5')

        if train:
            self.load_csv()
            self.model = self.create_model()
        else:
            self.model = load_model(self.model_file)

    def load_csv(self):
        '''
        Transform tabular data set into NumPy arrays.
        '''
        print('Loading training set: {}'.format(training_file))
        df = pd.read_csv(training_file, sep='\t')

        self.data = df[self.features].as_matrix()
        self.labels = df[['label']].as_matrix()

        print('Number of examples: {}'.format(self.data.shape[0]))
        print('Number of features: {}'.format(self.data.shape[1]))

    def create_model(self):
        '''
        Create new keras model.
        '''
        self.class_weight = {0: 0.25, 1: 0.75}

        model = Sequential()
        model.add(Dense(self.data.shape[1], input_dim=self.data.shape[1],
                        activation='relu', kernel_constraint=maxnorm(3)))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='RMSprop', loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    def train(self):
        '''
        Train and save model.
        '''
        print('Training new model: {}()'.format(self.__class__.__name__))
        self.model.fit(self.data, self.labels, epochs=100, batch_size=128,
                       class_weight=self.class_weight)

        print('Saving model: {}'.format(self.model_file))
        self.model.save(self.model_file)

    def validate(self):
        '''
        Ten-fold cross-validation with stratified sampling.
        '''
        print('Validating new model: {}()'.format(self.__class__.__name__))

        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []

        sss = StratifiedShuffleSplit(n_splits=10)
        for train_index, test_index in sss.split(self.data, self.labels):
            x_train, x_test = self.data[train_index], self.data[test_index]
            y_train, y_test = self.labels[train_index], self.labels[test_index]

            model = self.create_model()
            model.fit(x_train, y_train, epochs=100, batch_size=128,
                      class_weight=self.class_weight)
            y_pred = model.predict_classes(x_test, batch_size=128)

            accuracy_scores.append(accuracy_score(y_test, y_pred))
            precision_scores.append(precision_score(y_test, y_pred))
            recall_scores.append(recall_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred))

        print('')
        print('Accuracy: {}'.format(np.mean(accuracy_scores)))
        print('Precision: {}'.format(np.mean(precision_scores)))
        print('Recall: {}'.format(np.mean(recall_scores)))
        print('F1-measure: {}'.format(np.mean(f1_scores)))

    def predict(self, example):
        '''
        Classify a new example.
        '''
        example = np.array([example], dtype=np.float32)
        prob = self.model.predict(example, batch_size=1)
        return float(prob[0][0])


class BranchingNeuralNet(BaseModel):
    def __init__(self, train=False):
        self.features = self.load_features('bnn.json')

        self.entity_features = [f for f in self.features if
                                f.startswith('entity')]
        self.candidate_features = [f for f in self.features if
                                   f.startswith('candidate')]
        self.match_features = [f for f in self.features if
                               f.startswith('match')]

        self.c_start = len(self.entity_features)
        self.m_start = (len(self.entity_features) +
                        len(self.candidate_features))

        self.model_file = model_file_template.format('bnn.h5')

        if train:
            self.load_csv()
            self.model = self.create_model()
        else:
            self.model = load_model(self.model_file)

    def load_csv(self):
        '''
        Transform tabular data set into NumPy arrays.
        '''
        print('Loading training set: {}'.format(training_file))
        df = pd.read_csv(training_file, sep='\t')

        self.data = [df[self.entity_features].as_matrix(),
                     df[self.candidate_features].as_matrix(),
                     df[self.match_features].as_matrix()]

        self.labels = df[['label']].as_matrix()

        print('Number of examples: {}'.format(self.data[0].shape[0]))
        print('Number of entity features: {}'.format(self.data[0].shape[1]))
        print('Number of candidate features: {}'.format(self.data[1].shape[1]))
        print('Number of match features: {}'.format(self.data[2].shape[1]))

    def create_model(self):
        '''
        Create new keras model.
        '''
        self.class_weight = {0: 0.25, 1: 0.75}

        # Entity branch
        entity_inputs = Input(shape=(self.data[0].shape[1],))
        entity_x = Dense(self.data[0].shape[1], activation='relu',
                         kernel_constraint=maxnorm(3))(entity_inputs)
        entity_x = Dropout(0.25)(entity_x)
        # entity_x = Dense(50, activation='relu',
        #                  self.kernel_constraint=maxnorm(3))(entity_x)
        # entity_x = Dropout(0.25)(entity_x)

        # Candidate branch
        candidate_inputs = Input(shape=(self.data[1].shape[1],))
        candidate_x = Dense(self.data[1].shape[1], activation='relu',
                            kernel_constraint=maxnorm(3))(candidate_inputs)
        candidate_x = Dropout(0.25)(candidate_x)
        # candidate_x = Dense(50, activation='relu',
        #                     kernel_constraint=maxnorm(3))(candidate_x)
        # candidate_x = Dropout(0.25)(candidate_x)

        # Cosine proximity
        # cos_x = dot([entity_x, candidate_x], axes=1, normalize=False)
        # cos_x = concatenate([entity_x, candidate_x])
        # cos_output = Dense(1, activation='sigmoid')(cos_x)

        # Match branch
        match_inputs = Input(shape=(self.data[2].shape[1],))
        match_x = Dense(self.data[1].shape[1], activation='relu',
                        kernel_constraint=maxnorm(3))(match_inputs)
        match_x = Dropout(0.25)(match_x)

        # Merge
        x = concatenate([entity_x, candidate_x, match_x])
        x = Dense(32, activation='relu', kernel_constraint=maxnorm(3))(x)
        x = Dropout(0.25)(x)
        x = Dense(16, activation='relu', kernel_constraint=maxnorm(3))(x)
        x = Dropout(0.25)(x)
        x = Dense(8, activation='relu', kernel_constraint=maxnorm(3))(x)
        x = Dropout(0.25)(x)

        predictions = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=[entity_inputs, candidate_inputs, match_inputs],
                      outputs=predictions)
        model.compile(optimizer='RMSprop', loss='binary_crossentropy',
                      metrics=['accuracy'])

        return model

    def train(self):
        '''
        Train and save model.
        '''
        print('Training new model: {}()'.format(self.__class__.__name__))
        self.model.fit(self.data, self.labels, epochs=100, batch_size=128,
                       class_weight=self.class_weight)

        print('Saving model: {}'.format(self.model_file))
        self.model.save(self.model_file)

    def validate(self):
        '''
        Ten-fold cross-validation with stratified sampling.
        '''
        print('Validating new model: {}()'.format(self.__class__.__name__))

        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []

        sss = StratifiedShuffleSplit(n_splits=10)

        for train_index, test_index in sss.split(self.data[0], self.labels):
            x_train_0, x_test_0 = (self.data[0][train_index],
                                   self.data[0][test_index])
            x_train_1, x_test_1 = (self.data[1][train_index],
                                   self.data[1][test_index])
            x_train_2, x_test_2 = (self.data[2][train_index],
                                   self.data[2][test_index])

            y_train, y_test = self.labels[train_index], self.labels[test_index]

            model = self.create_model()
            model.fit([x_train_0, x_train_1, x_train_2], y_train, epochs=10,
                      batch_size=128, class_weight=self.class_weight)

            y_pred = model.predict([x_test_0, x_test_1, x_test_2],
                                   batch_size=128)
            y_pred = [1 if y[0] > 0.5 else 0 for y in y_pred]

            accuracy_scores.append(accuracy_score(y_test, y_pred))
            precision_scores.append(precision_score(y_test, y_pred))
            recall_scores.append(recall_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred))

        print('')
        print('Accuracy: {}'.format(np.mean(accuracy_scores)))
        print('Precision: {}'.format(np.mean(precision_scores)))
        print('Recall: {}'.format(np.mean(recall_scores)))
        print('F1-measure: {}'.format(np.mean(f1_scores)))

    def predict(self, example):
        '''
        Classify a new example.
        '''
        example_list = []
        example_list.append(np.array([example[:self.c_start]],
                            dtype=np.float32))
        example_list.append(np.array([example[self.c_start:self.m_start]],
                            dtype=np.float32))
        example_list.append(np.array([example[self.m_start:]],
                            dtype=np.float32))

        prob = self.model.predict(example_list, batch_size=1)
        return float(prob[0][0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-w', '--weights', required=False, action='store_true',
                        help='feature weights of current model')
    parser.add_argument('-t', '--train', required=False, action='store_true',
                        help='train and save new model')
    parser.add_argument('-v', '--validate', required=False,
                        action='store_true', help='cross-validate new model')
    parser.add_argument('-m', '--model', required=False, type=str,
                        default='svm', help='model type')

    args = parser.parse_args()

    if vars(args)['weights']:
        LinearSVM().weights()

    else:
        if vars(args)['model'] == 'svm':
            model = LinearSVM(train=True)
        elif vars(args)['model'] == 'nn':
            model = NeuralNet(train=True)
        elif vars(args)['model'] == 'bnn':
            model = BranchingNeuralNet(train=True)

        if vars(args)['train']:
            model.train()
        else:
            model.validate()
