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

# Standard library imports
import argparse
import json
import logging
import sys
import time

# External library imports
import unicodecsv as csv

# DAC imports
sys.path.insert(0, '..')
import dac

def generate(input_file, output_file):
    '''
    Generate a training set consisting of entity - DBpedia description
    pairs (links and non-links) and associated feature values, based on the
    set of artices with manually linked entities created with the DAC training
    interface.
    '''

    logging.basicConfig(level=logging.INFO)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')

    handler = logging.FileHandler('generate.log', mode='w')
    handler.setFormatter(formatter)
    handler.setLevel(logging.ERROR)

    logger = logging.getLogger(__name__)
    logger.addHandler(handler)

    data = json.load(open(input_file))

    with open(output_file, 'w') as fh:

        linker = dac.EntityLinker(model='train', debug=True, candidates=True)

        header = ['entity_id', 'cand_id', 'url', 'ne', 'cand_uri']
        header += linker.model.features
        header += ['label']

        csv_writer = csv.writer(fh, delimiter='\t', encoding='utf-8')
        csv_writer.writerow(header)

        url = None
        url_result = None
        candidate_count = 1

        for i, inst in enumerate(data['instances']):
            logger.info('Reviewing instance ' + str(i) + ': ' +
                inst['ne_string'].encode('utf-8'))

            # Check if instance has been labeled
            if inst['links']:

                # Get linker results for entire article (once once per article)
                if inst['url'] != url:
                    logger.info('Getting linker result for url: ' + inst['url'])

                    url = inst['url']
                    try:
                        url_result = linker.link(inst['url'])['linkedNEs']
                    except:
                        logger.error('No linker result, skipping url: '
                            + inst['url'])
                        time.sleep(3)

                # Select result for current instance
                result = [r for r in url_result if
                        r['text'] == inst['ne_string']]

                if len(result) != 1:
                    logger.info('No result for: ' + inst['ne_string'])
                    continue
                else:
                    result = result[0]

                # Loop through result candidates, if any
                if 'candidates' in result:
                    for cand in result['candidates']:

                        # Metadata
                        row = []
                        row.append(str(inst['id']))
                        row.append(str(candidate_count))
                        row.append(inst['url'].encode('utf-8'))
                        row.append(inst['ne_string'].encode('utf-8'))
                        row.append(cand['id'].encode('utf-8'))

                        # Features
                        for f in linker.model.features:
                            value = cand['features'][f]
                            row.append("{0:.5f}".format(float(value)))

                        # Label
                        if cand['id'] in inst['links']:
                            row.append(str(1))
                        else:
                            row.append(str(0))

                        # Exclude candidates with name or date conflict
                        if cand['features']['match_str_conflict'] == 1:
                            continue
                        elif cand['features']['match_txt_date'] == -1:
                            continue
                        else:
                            candidate_count += 1
                            csv_writer.writerow(row)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', required=False, type=str,
        default='../../dac-web/users/tve/art.json', help='path to input file')
    parser.add_argument('-o', '--output', required=False, type=str,
        default='training.csv', help='path to output file')

    args = parser.parse_args()

    generate(vars(args)['input'], vars(args)['output'])
