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

# Fix all path related stuff in one go ;)
# Don't do this at home kids!
ABS_PATH = '/var/www/dac/'
os.environ['PATH_INFO'] = ABS_PATH
os.chdir(os.path.dirname(ABS_PATH))
sys.path.insert(0, os.path.dirname(ABS_PATH))

import dac

# md5sums for all file, keeps things in sync
FIND = "find . -type f -not -path \*.pyc -not -path \*.swp -exec md5sum '{}' ';'"

# Path to doctests output (from cron)
DOCTEST_OUTPUT = '/tmp/dac_test_results'
DOCTEST_OUTPUT_ERROR_PREFIX = 'dac_error_'

global stats
stats = {}
stats['urls_total'] = 0
stats['links_total'] = 0


def mk_md5sums():
    with os.popen(FIND) as fh:
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


@route('/info')
def info():

    try:
        test_report_stamp = time.ctime(
                                os.path.getctime(DOCTEST_OUTPUT))
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


@route('/')
def index():
    os.chdir(os.path.dirname(__file__))

    url = request.params.get('url')
    ne = request.params.get('ne')
    callback = request.params.get('callback')
    debug = request.params.get('debug')
    features = request.params.get('features')
    model = request.params.get('model')

    if not url:
        abort(400, "No fitting argument (\"url=...\") given.")

    linker = dac.EntityLinker(model=model)
    results = linker.link(url, ne)

    resp = []
    for result in results:
        if result['link'] or debug:
            r = {}
            r['text'] = result['text'].encode('utf-8')
            if debug:
                r['reason'] = result['reason'].encode('utf-8')
                if result['prob'] > 0:
                    r['prob'] = result['prob']
            if result['link']:
                r['link'] = result['link'].encode('utf-8')
                r['label'] = result['label'].encode('utf-8')
                r['prob'] = result['prob']
                if features:
                    r['features'] = result['features']
                    #r['features'] = {key.encode('utf-8'):value for key,value in result['features'].items()}
            resp.append(r)

    resp = {'linkedNEs': resp}
    if callback:
        resp = callback + '(' + str(resp) + ');'

    response.set_header('Content-Type', 'application/json')
    return resp

application = default_app()
