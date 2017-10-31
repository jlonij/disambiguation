#!/usr/bin/env python                                                                                |
# -*- coding: utf-8 -*-
#
# DAC Entity Linker
#
# Copyright (C) 2017 Koninklijke Bibliotheek, National Library of
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

import json
import math
import numpy as np
import os

from keras.models import load_model
from sklearn.externals import joblib

abs_path = os.path.dirname(os.path.realpath(__file__))

class Model:
    def __init__(self):
        self.features = self.load_features('features.json')

    def load_features(self, feature_file):
        path = os.path.join(abs_path, 'features', feature_file)
        return json.load(open(path))['features']

class LinearSVM(Model):
    def __init__(self):
        self.clf = self.load_model('svm.pkl')
        self.features = self.load_features('svm.json')

    def load_model(self, model_file):
        path = os.path.join(abs_path, 'models', model_file)
        return joblib.load(path)

    def predict(self, example):
        dec = self.clf.decision_function([example])[0]
        prob = 1 / (1 + math.exp(dec * -1))
        return prob

class NeuralNet(Model):
    def __init__(self):
        self.model = self.load_model('nn.h5')
        self.features = self.load_features('nn.json')

    def load_model(self, model_file):
        model_file = os.path.join(abs_path, 'models', model_file)
        return load_model(model_file)

    def predict(self, example):
        example = np.array([example], dtype=np.float32)
        prob = self.model.predict(example, batch_size=1)
        return float(prob[0][0])

class BranchingNeuralNet(Model):

    def __init__(self):
        self.model = self.load_model('bnn.h5')
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

    def load_model(self, model_file):
        model_file = os.path.join(abs_path, 'models', model_file)
        return load_model(model_file)

    def predict(self, example):
        example_list = []
        example_list.append(np.array([example[:self.c_start]],
            dtype=np.float32))
        example_list.append(np.array([example[self.c_start:self.m_start]],
            dtype=np.float32))
        example_list.append(np.array([example[self.m_start:]],
            dtype=np.float32))

        prob = self.model.predict(example_list, batch_size=1)
        return float(prob[0][0])

