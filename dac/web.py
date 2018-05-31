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

import os
import socket
import sys
import json

from bottle import abort
from bottle import default_app
from bottle import request
from bottle import response
from bottle import route
from bottle import run

sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.realpath(__file__)), '..'))
from dac import dac
from dac import models

hostname = socket.gethostname()

model_dict = {}
model_dict['train'] = models.BaseModel()
model_dict['svm'] = models.LinearSVM()
model_dict['nn'] = models.NeuralNet()
model_dict['bnn'] = models.BranchingNeuralNet()

print type(model_dict['train'])

def array_to_utf(a):
    '''
    Encode array values with utf-8 encoding.
    '''
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
    '''
    Encode dictionary values with utf-8 encoding.
    '''
    dutf = {}
    for k, v in d.iteritems():
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
    global model_dict
    url = request.params.get('url')
    ne = request.params.get('ne')
    model = request.params.get('model')
    debug = request.params.get('debug')
    features = request.params.get('features')
    candidates = request.params.get('candidates')
    callback = request.params.get('callback')

    if not url:
        abort(400, 'Missing argument ("url=...").')

    if not model:
        model = 'nn'

    try:
        linker = dac.EntityLinker(model_dict.get(model), debug=debug, features=features,
                                  candidates=candidates)
        result = linker.link(url, ne)
    except Exception as e:
        result = {}
        result['status'] = 'error'
        result['message'] = str(e)
        result['hostname'] = hostname

    if result['status'] == 'ok':
        result['linkedNEs'] = array_to_utf(result['linkedNEs'])
        result['hostname'] = hostname
        result = json.dumps(result, sort_keys=True)

    if callback:
        result = unicode(callback) + u'(' + str(result) + u');'

    response.set_header('Content-Type', 'application/json')
    return result


if __name__ == '__main__':
    run(host='localhost', port=5002)
else:
    application = default_app()

