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

import math
import os

import numpy as np

from keras.models import load_model
from sklearn.externals import joblib

class LinearSVM:

    def __init__(self):

        abs_path = os.path.dirname(os.path.realpath(__file__))
        model_file = os.path.join(abs_path, 'models', 'model.pkl')
        self.clf = joblib.load(model_file)

        self.features = [
            'pref_label_exact_match', 'pref_label_end_match', 'pref_label_match',
            'alt_label_exact_match', 'alt_label_end_match', 'alt_label_match',
            'last_part_match', 'non_matching_labels', 'name_conflict', 'pref_lsr',
            'mean_lsr', 'wd_lsr', 'date_match', 'query_id_0', 'query_id_1',
            'query_id_2', 'query_id_3', 'substitution', 'solr_position',
            'solr_score', 'inlinks', 'inlinks_rel', 'inlinks_newspapers',
            'inlinks_newspapers_rel', 'outlinks', 'outlinks_rel', 'lang', 'ambig',
            'quotes', 'type_match', 'role_match', 'spec_match', 'keyword_match',
            'subject_match', 'max_vec_sim', 'mean_vec_sim', 'top_vec_sim',
            'entity_match', 'max_entity_vec_sim', 'mean_entity_vec_sim',
            'top_entity_vec_sim'
        ]

    def predict(self, example):
        dec = self.clf.decision_function([example])[0]
        prob = 1 / (1 + math.exp(dec * -1))
        return prob


class NeuralNet:

    def __init__(self):

        abs_path = os.path.dirname(os.path.realpath(__file__))
        model_file = os.path.join(abs_path, 'models', 'model.h5')
        self.model = load_model(model_file)

        self.features = [
            'pref_label_exact_match', 'pref_label_end_match', 'pref_label_match',
            'alt_label_exact_match', 'alt_label_end_match', 'alt_label_match',
            'last_part_match', 'non_matching_labels', 'name_conflict', 'pref_lsr',
            'mean_lsr', 'wd_lsr', 'date_match', 'query_id_0', 'query_id_1',
            'query_id_2', 'query_id_3', 'substitution', 'solr_position',
            'solr_score', 'inlinks', 'inlinks_rel', 'inlinks_newspapers',
            'inlinks_newspapers_rel', 'outlinks', 'outlinks_rel', 'lang', 'ambig',
            'quotes', 'type_match', 'role_match', 'spec_match', 'keyword_match',
            'subject_match', 'max_vec_sim', 'mean_vec_sim', 'top_vec_sim',
            'entity_match', 'max_entity_vec_sim', 'mean_entity_vec_sim',
            'top_entity_vec_sim'
        ]

    def predict(self, example):
        example = np.array([example], dtype=np.float32)
        prob = self.model.predict(example, batch_size=1)
        return float(prob[0][0])

