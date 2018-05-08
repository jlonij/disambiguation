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

"""
  dac_test.py

  This program assumes DAC - web.py to be started as following:

  /etc/rc.local <- Should contain

  ==
    #!/bin/bash

    cd /var/www/dac/dac
    tmux new-session -d  -s "dac" "\
    while true
    do
    su www-data -s /bin/sh -c 'uwsgi_python -b 99999 --socket 127.0.0.1:8001 -p 500 --fs-brutal-reload web.py --need-app --wsgi-file web.py'
    echo $(date +%Y-%d_%h_%H:%M) >> /tmp/dac_error_log
    sleep 5
    done"

    exit 0
  ==
"""

import json
import os
import sys

__author__ = 'WillemJan Faber <willemjan.faber@kb.nl>'
__date__ = '2016-12-05'
__licence__ = 'GPLv3'

reload(sys)
sys.setdefaultencoding('utf8')

DAC_PORT = 8001
PATH_TO_UWSGI_FILE = '/var/www/dac/dac/web.py'
IP = ''

with os.popen('/sbin/ifconfig eth0 | /bin/grep "inet addr"') as fh:
    IP = fh.read()

if not IP:
    sys.stdout.write("Failed to determine IP-number for this machine.")
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


def test_docs_avail():
    """
    >>> test_url(TEST_DOC_REMOTE).ok
    True
    """


def dac_uwsgi_file_exists_test():
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
