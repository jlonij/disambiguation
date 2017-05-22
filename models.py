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

import os
import csv
import math

import numpy as np
import tensorflow as tf

from sklearn.externals import joblib

class LinearSVM:

    def __init__(self):

        self.clf = joblib.load('models/model.pkl')

        self.features = [
            'pref_label_exact_match', 'pref_label_end_match', 'pref_label_match',
            'alt_label_exact_match', 'alt_label_end_match', 'alt_label_match',
            'last_part_match', 'levenshtein_ratio', 'name_conflict', 'date_match',
            'solr_iteration', 'solr_query', 'solr_position', 'solr_score', 'inlinks',
            'lang', 'ambig', 'quotes', 'type_match', 'role_match', 'spec_match',
            'keyword_match', 'subject_match', 'max_vec_sim', 'mean_vec_sim',
            'vec_match', 'entity_match'
        ]

    def predict(self, example):
        return self.clf.predict_proba([example])[0][1]


class NeuralNet:

    def __init__(self):

        self.features = [
            'pref_label_exact_match', 'pref_label_end_match', 'pref_label_match',
            'alt_label_exact_match', 'alt_label_end_match', 'alt_label_match',
            'last_part_match', 'levenshtein_ratio', 'name_conflict', 'date_match',
            'solr_iteration', 'solr_query', 'solr_position', 'solr_score', 'inlinks',
            'lang', 'ambig', 'quotes', 'type_match', 'role_match', 'spec_match',
            'keyword_match', 'subject_match', 'max_vec_sim', 'mean_vec_sim',
            'vec_match', 'entity_match'
        ]

    def predict(self, example):

        curr_dir = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-1]) + os.sep
        model_file = curr_dir + "models" + os.sep + "model.ckpt"

        graph = tf.Graph()

        with tf.Session(graph=graph) as session:

            num_hidden_nodes1 = 28
            num_hidden_nodes2 = 14
            num_hidden_nodes3 = 7

            num_features = len(self.features)
            num_labels = 2
            #global_step = tf.Variable(0)

            # Weights and biases for each network layer
            weights1 = tf.Variable(tf.truncated_normal([num_features,
                    num_hidden_nodes1], stddev=np.sqrt(2.0 / num_features)), name='weights1')
            biases1 = tf.Variable(tf.zeros([num_hidden_nodes1]), name='biases1')
            weights2 = tf.Variable(tf.truncated_normal([num_hidden_nodes1,
                    num_hidden_nodes2], stddev=np.sqrt(2.0 / num_hidden_nodes1)), name='weights2')
            biases2 = tf.Variable(tf.zeros([num_hidden_nodes2]), name='biases2')
            weights3 = tf.Variable(tf.truncated_normal([num_hidden_nodes2,
                    num_hidden_nodes3], stddev=np.sqrt(2.0 / num_hidden_nodes2)), name='weights3')
            biases3 = tf.Variable(tf.zeros([num_hidden_nodes3]), name='biases3')
            weights4 = tf.Variable(tf.truncated_normal([num_hidden_nodes3,
                    num_labels], stddev=np.sqrt(2.0 / num_hidden_nodes3)), name='weights4')
            biases4 = tf.Variable(tf.zeros([num_labels]), name='biases4')

            # Load saved values for weights and biases
            self.saver = tf.train.Saver()
            self.saver.restore(session, model_file)

            # Prediction for new examples
            x = tf.placeholder(tf.float32, shape=(1, num_features))
            lay1_y = tf.nn.relu(tf.matmul(x, weights1) + biases1)
            lay2_y = tf.nn.relu(tf.matmul(lay1_y, weights2) + biases2)
            lay3_y = tf.nn.relu(tf.matmul(lay2_y, weights3) + biases3)
            y = tf.nn.softmax(tf.matmul(lay3_y, weights4) + biases4)

            ex = np.ndarray(shape=(1, len(example)), dtype=np.float32)
            ex[0] = example

            return float(session.run(y, feed_dict={x: ex})[0,1])
