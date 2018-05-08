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

"""
  dac_test.py

  This program assumes DAC - web.py to be started as following:

  /etc/rc.local <- Should contain

  ==
    #!/bin/bash

    cd /var/www/dac/
    tmux new-session -d  -s "dac" "\
    while true
    do
    su www-data -s /bin/sh -c 'uwsgi_python -b 99999 --socket 127.0.0.1:8001 -p 500 --fs-brutal-reload /var/www/dac/dac/web.py --need-app --wsgi-file /var/www/dac/dac/web.py'
    echo $(date +%Y-%d_%h_%H:%M) >> /tmp/dac_error_log
    sleep 5
    done"

    exit 0
  ==
"""

import os
import sys
import json

__author__ = 'WillemJan Faber <willemjan.faber@kb.nl>'
__date__ = '2016-12-05'
__licence__ = 'GPLv3'

reload(sys)
sys.setdefaultencoding('utf8')

DAC_PORT = 8001
NODES = ['kbresearch.nl']
PATH_TO_CONFIG_FILE = '/var/www/dac/dac/config.json'
PATH_TO_DISAMBIGUATION_FILE = '/var/www/dac/dac/dac.py'
PATH_TO_UWSGI_FILE = '/var/www/dac/dac/web.py'
IP = ''

with os.popen('/sbin/ifconfig eth0 | /bin/grep "inet addr"') as fh:
    IP = fh.read()

if not IP:
    sys.stout.write("Failed to determine IP-number for this machine.")
    sys.exit(-1)
else:
    IP = IP.split(':')[1].split(' ')[0]

TEST_DOC_REMOTE = 'http://resolver.kb.nl/resolve?urn=ddd:010860175:mpeg21:a0206:ocr'

try:
    import requests
except:
    msg = """\
%s: Error: Could not import requests.
Try:
   apt-get install -y python-requests
"""
    sys.stdout.write(msg % __file__)
    sys.exit(-1)


def test_url(url):
    try:
        response = requests.get(url, timeout=5)
    except requests.exceptions.ConnectionError as error:
        return error.__repr__()
    return response


def tpta_url(path_to_file=PATH_TO_CONFIG_FILE):
    with open(path_to_file) as fh:
        data = json.load(fh)
        if 'TPTA_URL' in data:
            return data['TPTA_URL'].split('?')[0]
    return False


def tpta_is_alive_and_kicking():
    response = requests.get(tpta_url())
    print(response.text)


def tpta(text_url):
    url = tpta_url()

    url += '?url=%s' % text_url
    try:
        response = requests.get(url, timeout=10)
    except requests.exceptions.ConnectionError as error:
        print error
        return []

    data = response.json()
    loc = [e['ne'] for e in data['entities'] if e['type'] == 'location']
    return sorted(list(set(loc)))[:10]


def solr_url(path_to_file=PATH_TO_CONFIG_FILE):
    with open(path_to_file) as fh:
        data = json.load(fh)
        if 'SOLR_URL' in data:
            return data['SOLR_URL']
    return False


def solr_is_alive_and_kicking():
    url = solr_url() + 'select/?wt=json'

    try:
        response = requests.get(url)
    except requests.exceptions.ConnectionError as error:
        print error
        return []

    try:
        data = response.json()
    except Exception as error:
        print(error)
        return []

    return data.keys()


def solr(query):
    url = solr_url()
    url += 'select/?wt=json&q=%s' % query

    try:
        response = requests.get(url)
    except requests.exceptions.ConnectionError as error:
        print error
        return {}

    try:
        data = response.json()
    except Exception as error:
        print error
        return {}

    return data.get('response')


def dac_slave_node():
    res = []

    for NODE in NODES:
        url = 'http://%s/dac?url=%s' % (NODE, TEST_DOC_REMOTE)
        try:
            response = requests.get(url, timeout=100)
        except requests.exceptions.ConnectionError as error:
            print error
            return
        try:
            result = response.json()
        except Exception as error:
            print error
            return
        if 'linkedNEs' in result:
            res.append(len(result.get('linkedNEs')) - 65)

    if res == [42] * len(NODES):
        return 42

    return

#
# Below are the actual doctests.
#

def test_docs_avail():
    """
    >>> test_url(TEST_DOC_REMOTE).ok
    True
    """


def dac_webpy_file_exists_test():
    """
    >>> os.path.isfile(PATH_TO_UWSGI_FILE)
    True
    """


def dac_is_available_via_localhost_test_defaulterror_by_ip():
    """
    >>> res = test_url('http://127.0.0.1:%s/' % DAC_PORT)
    >>> len(res.text.split(':'))
    15
    """


def dac_is_available_via_localhost_test_defaulterror_by_name():
    """
    >>> res = test_url('http://localhost:%s/' % DAC_PORT)
    >>> len(res.text.split(':'))
    15
    """


def tpta_url_is_defined_test():
    """
    >>> len(tpta_url()) > 0
    True
    """


def disambiguation_standalone_test():
    """
    >>> sys.path.insert(0, '../dac')
    >>> import dac
    >>> linker = dac.EntityLinker()
    >>> result = linker.link(TEST_DOC_REMOTE)
    >>> len(result['linkedNEs'])
    107
    """


def tpta_is_alive_and_kicking_test():
    """
    >>> tpta_is_alive_and_kicking()
    {"error": "Missing argument ?url=http://resolver.kb.nl/resolve?urn=ddd:010381561:mpeg21:a0049:ocr"}
    """


def tpta_is_able_to_do_useful_things_test():
    """
    >>> tpta(text_url='%s' % TEST_DOC_REMOTE)
    [u"'s Hage", u'Agoeda', u'Agoedakringen', u'Amerika', u'Amsterdam', u'Amsterdam-Oost', u'Amsterdamschen Kerkeraad', u'Antwerpen', u'Arnhem', u'Beieren']
    """


def solr_url_is_defined_test():
    """
    >>> len(solr_url()) > 0
    True
    """


def solr_is_alive_and_kicking_test():
    """
    >>> solr_is_alive_and_kicking()
    [u'responseHeader', u'response']
    """


def solr_is_able_to_do_useful_things_test():
    """
    >>> int(solr("pref_label:Scheveningen").get('numFound')) + 16
    42
    """


def dac_front_to_back_surfnetonly_nginx_local_test():
    """
    >>> url = 'http://localhost/dac?url=%s' % TEST_DOC_REMOTE
    >>> response = requests.get(url)
    >>> response.ok
    True
    >>> result = json.loads(response.text)
    >>> len(result.get('linkedNEs')) - 65
    42
    """


def dac_front_to_back_surfnetonly_nginx_slave_nodes_test():
    """
    >>> dac_slave_node()
    42
    """


def dac_front_to_back_office_network_nginx_local_test():
    """
    >>> url = 'http://localhost/dac?url=%s' % TEST_DOC_REMOTE
    >>> response = requests.get(url, timeout=100)
    >>> response.ok
    True
    >>> result = json.loads(response.text)
    >>> len(result.get('linkedNEs')) - 65
    42
    """


def dac_front_to_back_office_network_nginx_loadbalancer_test():
    """
    >>> url = 'http://%s/dac?url=%s' % (IP, TEST_DOC_REMOTE)
    >>> response = requests.get(url, timeout=100)
    >>> result = json.loads(response.text)
    >>> len(result.get('linkedNEs')) - 65
    42
    >>> # FIN!
    """


if __name__ == '__main__':
    import doctest
    doctest.testmod()

