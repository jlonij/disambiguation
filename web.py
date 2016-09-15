#!/usr/bin/env python

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
#

#
# Link service
#
# Expects ocr-text from supplied url parameter,
# reads data from url and calls named entity recognizer with the data,
# subsequently calls the disambiguation service. A particular named
# entity in the text can be specified with the ne parameter.
#

from bottle import abort, route, run, template, request, response, default_app

import os, sys
os.chdir(os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(__file__))

import disambiguation


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

    linker = disambiguation.EntityLinker(model=model)
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
