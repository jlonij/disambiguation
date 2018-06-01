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

import json
import pprint
import requests

PATH_TO_TEST_SET = '../../../dac-web/users/test-spotlight/art.json'
SPOTLIGHT_URL = 'http://kbresearch.nl/spotlight_enrich/?'
DAC_URL = 'http://localhost:5002/?'

data = json.load(open(PATH_TO_TEST_SET))

documents = {}

for instance in data['instances']:
    if instance['links'][0] != 'none':
        if instance['url'] in documents:
            if instance['links'][0] not in documents[instance['url']]:
                documents[instance['url']].append(instance['links'][0])
        else:
            documents[instance['url']] = [instance['links'][0]]

true_links = 0
correct_link_predictions = 0
missing_link_predictions = 0
added_link_predictions = 0

for doc in documents:
    true_links += len(documents[doc])

    response = requests.get(DAC_URL, params={'url': doc}).json()
    links = [entity['link'] for entity in response['linkedNEs']]

    correct_link_predictions += len(set(documents[doc]) & set(links))
    missing_link_predictions += len(set(documents[doc]) - set(links))
    added_link_predictions += len(set(links) - set(documents[doc]))

print('true_links: {}'.format(true_links))
print('correct_link_predictions: {}'.format(correct_link_predictions))
print('missing_link_predictions: {}'.format(missing_link_predictions))
print('added_link_predictions: {}'.format(added_link_predictions))
