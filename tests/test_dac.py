#!/usr/bin/env python
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
import os
import sys

import requests

sys.path.insert(0, '../dac')
import dac

CONFIG_FILE = '../dac/config.json'
NN_MODEL_FILE = '../dac/models/nn.h5'
NN_FEATURE_FILE = '../dac/features/nn.json'

TEST_DOC = 'http://resolver.kb.nl/resolve?urn=ddd:010734861:mpeg21:a0002:ocr'


def dac_config_file_exists():
    '''
    >>> os.path.isfile(CONFIG_FILE)
    True
    '''


def tpta_url_is_defined():
    '''
    >>> data = json.load(open(CONFIG_FILE))
    >>> 'TPTA_URL' in data
    True
    >>> len(data['TPTA_URL']) > 0
    True
    '''


def tpta_is_working():
    '''
    >>> data = json.load(open(CONFIG_FILE))
    >>> payload = {'url': TEST_DOC}
    >>> response = requests.get(data['TPTA_URL'], params=payload, timeout=30)
    >>> response = response.json()
    >>> 'text' in response
    True
    >>> 'entities' in response
    True
    >>> len(response['entities'])
    20
    '''


def jsru_url_is_defined():
    '''
    >>> data = json.load(open(CONFIG_FILE))
    >>> 'JSRU_URL' in data
    True
    >>> len(data['JSRU_URL']) > 0
    True
    '''


def jsru_is_working():
    '''
    >>> data = json.load(open(CONFIG_FILE))
    >>> payload = {}
    >>> payload['operation'] = 'searchRetrieve'
    >>> payload['x-collection'] = 'DDD_artikel'
    >>> payload['query'] = 'uniqueKey=' + TEST_DOC.split('urn=')[-1][:-4]
    >>> response = requests.get(data['JSRU_URL'], params=payload, timeout=10)
    >>> response.text.find('Churchill') > -1
    True
    '''


def topics_url_is_defined():
    '''
    >>> data = json.load(open(CONFIG_FILE))
    >>> 'TOPICS_URL' in data
    True
    >>> len(data['TOPICS_URL']) > 0
    True
    '''


def topics_is_working():
    '''
    >>> data = json.load(open(CONFIG_FILE))
    >>> payload = {'url': TEST_DOC}
    >>> response = requests.get(data['TOPICS_URL'], params=payload, timeout=30)
    >>> response = response.json()
    >>> 'topics' in response
    True
    '''


def solr_url_is_defined():
    '''
    >>> data = json.load(open(CONFIG_FILE))
    >>> 'SOLR_URL' in data
    True
    >>> len(data['SOLR_URL']) > 0
    True
    '''


def solr_is_working():
    '''
    >>> data = json.load(open(CONFIG_FILE))
    >>> url = data['SOLR_URL'] + 'select/?'
    >>> payload = {'q': 'pref_label:Scheveningen', 'wt': 'json'}
    >>> response = requests.get(url, params=payload, timeout=30)
    >>> response = response.json()['response']
    >>> response['numFound']
    26
    '''


def w2v_url_is_defined():
    '''
    >>> data = json.load(open(CONFIG_FILE))
    >>> 'W2V_URL' in data
    True
    >>> len(data['W2V_URL']) > 0
    True
    '''


def w2v_is_working():
    '''
    >>> data = json.load(open(CONFIG_FILE))
    >>> payload = {'source': 'churchill'}
    >>> response = requests.get(data['W2V_URL'], params=payload, timeout=30)
    >>> response = response.json()
    >>> 'vectors' in response
    True
    >>> len(response['vectors'][0])
    100
    '''


def dac_nn_model_file_exists():
    '''
    >>> os.path.isfile(NN_MODEL_FILE)
    True
    '''


def dac_nn_feature_file_exists():
    '''
    >>> os.path.isfile(NN_FEATURE_FILE)
    True
    '''


def dac_nn_local_test():
    '''
    >>> linker = dac.EntityLinker(model='nn')
    >>> result = linker.link(TEST_DOC)
    >>> len(result['linkedNEs'])
    7
    '''


if __name__ == '__main__':
    import doctest
    doctest.testmod()
