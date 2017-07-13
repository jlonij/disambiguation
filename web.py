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

import os
import sys

from bottle import abort
from bottle import default_app
from bottle import request
from bottle import response
from bottle import route
from bottle import run
from bottle import template

abs_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, abs_path)

import dac

# Path to doctests output (from cron)
DOCTEST_OUTPUT = '/tmp/dac_test_results'
DOCTEST_OUTPUT_ERROR_PREFIX = 'dac_error_'

global stats
stats = {}
stats['urls_total'] = 0
stats['links_total'] = 0

def mk_md5sums():
    find = "find . -type f -not -path \*.pyc -not -path \*.swp "
    find += "-exec md5sum '{}' ';'"
    with os.popen(find) as fh:
        sums = fh.read()
    sums = [[f for f in l.split(' ') if f.strip()] for l in sums.split('\n')]
    sums1 = {}
    for i in sums:
        if i:
            sums1[i[0]] = i[1][2:]
    return sums1

def test_errors():
    nr = 0
    for fname in os.listdir('/tmp/'):
        if fname.startswith(DOCTEST_OUTPUT_ERROR_PREFIX):
            nr += 1
    return nr

route('/info')
def info():
    '''
    Return service information.
    '''
    try:
        test_report_stamp = time.ctime(os.path.getctime(DOCTEST_OUTPUT))
    except:
        test_report_stamp = ''

    try:
        with open('/tmp/dac_test_results') as fh:
            test_report = fh.read()
    except:
        test_report = ''

    resp = {'sums': mk_md5sums(),
            'solr': SOLR_URL,
            'tpta': TPTA_URL,
            'test_errors': test_errors(),
            'test_report': test_report,
            'test_report_stamp': test_report_stamp,
            'stats': stats}

    response.set_header('Content-Type', 'application/json')
    return resp

def array_to_utf(a):
    autf = []
    for v in a:
        if isinstance(v, unicode):
            autf.append(v.encode('utf-8'))
        elif isinstance(v, dict):
            autf.append(dict_to_utf(v))
        elif isinstance(v, list):
            autf.append(array_to_utf(v))
        else:
            autf.append(v)
    return autf

def dict_to_utf(d):
    dutf = {}
    for k,v in d.iteritems():
        if isinstance(v, unicode):
            dutf[k] = v.encode('utf-8')
        elif isinstance(v, list):
            dutf[k] = array_to_utf(v)
        elif isinstance(v, dict):
            dutf[k] = dict_to_utf(v)
        else:
            dutf[k] = v
    return dutf

@route('/')
def index():
    '''
    Return the entity linker result.
    '''
    url = request.params.get('url')
    ne = request.params.get('ne')
    model = request.params.get('model')
    debug = request.params.get('debug')
    features = request.params.get('features')
    candidates = request.params.get('candidates')
    callback = request.params.get('callback')

    if not url:
        abort(400, "No fitting argument (\"url=...\") given.")

    try:
        linker = dac.EntityLinker(model=model, debug=debug, features=features,
            candidates=candidates)
        result = linker.link(url, ne)
    except Exception as e:
        result = {'status': 'error', 'message': str(e)}

    if result['status'] == 'ok':
        result['linkedNEs'] = array_to_utf(result['linkedNEs'])

    if callback:
        result = callback + '(' + str(result) + ');'

    response.set_header('Content-Type', 'application/json')
    return result

if __name__ == '__main__':
    run(host='localhost', port=5002)
else:
    application = default_app()
