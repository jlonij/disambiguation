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
    su www-data -s /bin/sh -c 'uwsgi_python -b 99999 --socket 127.0.0.1:8001 -p 500 --fs-brutal-reload /var/www/dac/web.py --need-app --wsgi-file /var/www/dac/web.py'
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
NODES = ['145.100.57.52:8001']
PATH_TO_UWSGI_FILE = '/var/www/dac/web.py'
PATH_TO_DISAMBIGUATION_FILE = '/var/www/dac/disambiguation.py'
IP = ''

with os.popen('/sbin/ifconfig eth0 | /bin/grep "inet addr"') as fh:
    IP = fh.read()

if not IP:
    sys.stout.write("Failed to determine IP-number for this machine.")
    sys.exit(-1)
else:
    IP = IP.split(':')[1].split(' ')[0]

TEST_DOC_LOCAL0 = 'http://localhost:82/test1'
TEST_DOC_LOCAL1 = 'http://%s:82/test1' % IP
TEST_DOC_REMOTE = 'http://resolver.kb.nl/resolve?urn=ddd:010860175:mpeg21:a0206:ocr'
CMD_LINE_EXEC = '''uwsgi_python -b 99999 --socket 127.0.0.1:8001\
 -p 500 --fs-brutal-reload /var/www/dac/web.py\
 --need-app --wsgi-file /var/www/dac/web.py'''

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

try:
    import psutil
except:
    msg = """\
%s: Error: Could not import psutil.
Try:
   apt-get install -y python-psutil
"""
    sys.stdout.write(msg % __file__)
    sys.exit(-1)


def test_url(url):
    try:
        response = requests.get(url,
                                timeout=5)
    except requests.exceptions.ConnectionError as error:
        return error.__repr__()
    return response


def test_process(cmd_line_exec):
    all_ps = []
    for proc in psutil.process_iter():
        try:
            pinfo = proc.as_dict(attrs=['cmdline'])
        except psutil.NoSuchProcess as error:
            return error
        else:
            for v in pinfo.values():
                if " ".join(v) == CMD_LINE_EXEC:
                    return True
                else:
                    all_ps.append(" ".join(v))
    return all_ps


def tpta_url(path_to_file=PATH_TO_UWSGI_FILE):
    with open(path_to_file) as fh:
        line = fh.readline()
        while line:
            if line.startswith('TPTA_URL'):
                if "'" in line:
                    line = line.split("'")[1]
                    line = line.split('?')[0]
                    return line.strip()
            line = fh.readline()
    return False


def tpta_is_alive_and_kicking():
    response = requests.get(tpta_url())
    print(response.text)


def tpta(text=False, text_url=False):
    loc = []
    url = tpta_url()

    if text:
        url += '?lang=nl&text="%s"' % text
        try:
            response = requests.get(url, timeout=2)
        except requests.exceptions.ConnectionError as error:
            print error
            return []

    elif text_url:
        url += '?lang=nl&url=%s' % text_url
        try:
            response = requests.get(url, timeout=10)
        except requests.exceptions.ConnectionError as error:
            print error
            return []

    for line in response.text.encode('utf-8').split('\n'):
        if line.startswith('<location>'):
            line = line.split('>')[1]
            line = line.split('<')[0]
            if line not in loc:
                loc.append(line)

    return(sorted(loc))


def solr_url():
    with open(PATH_TO_UWSGI_FILE) as fh:
        line = fh.readline()
        while line:
            if line.startswith('SOLR_URL'):
                if '=' in line:
                    line = line.split('=')[1]
                    line = line.replace("'", "")
                    return line.strip()
            line = fh.readline()
    return False


def solr_is_alive_and_kicking():
    url = solr_url() + '/select/?wt=json'

    try:
        response = requests.get(url)
    except requests.exceptions.ConnectionError as error:
        print error
        return []

    try:
        response = json.loads(response.text.strip())
    except Exception as error:
        print(error)
        return []

    return response.keys()


def solr(query):
    url = solr_url()
    url += '/select/?wt=json&q=%s' % query

    try:
        response = requests.get(url)
    except requests.exceptions.ConnectionError as error:
        print error
        return {}

    try:
        response = json.loads(response.text.strip())
    except Exception as error:
        print error
        return {}
    return response.get('response')


def dac_slave_node():
    res = []

    for NODE in NODES:
        url = 'http://%s/dac?url=%s' % (NODE, TEST_DOC_LOCAL1)
        try:
            response = requests.get(url, timeout=20)
        except requests.exceptions.ConnectionError as error:
            print error
            return
        try:
            result = json.loads(response.text)
        except Exception as error:
            print error
            return
        if 'linkedNEs' in result:
            res.append(len(result.get('linkedNEs')) - 34)

    if res == [42] * len(NODES):
        return 42

    return

#
# Below are the actual doctests.
#


def test_docs_avail():
    """
    >>> test_url(TEST_DOC_LOCAL0).ok
    True
    >>> test_url(TEST_DOC_LOCAL1).ok
    True
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
    >>> len(res.split(':'))
    4
    """


def dac_is_available_via_localhost_test_defaulterror_by_name():
    """
    >>> res = test_url('http://localhost:%s/' % DAC_PORT)
    >>> len(res.split(':'))
    4
    """


def tpta_url_is_defined_test():
    """
    >>> len(tpta_url()) > 0
    True
    """


def disambiguation_standalone_test():
    """
    >>> cmd = 'PYTHONIOENCODING="UTF-8" /usr/bin/python %s http://%s:82/test1' % (PATH_TO_DISAMBIGUATION_FILE, IP)
    >>> fh = os.popen(cmd)
    >>> len(fh.read())
    209619
    >>> fh.close()
    """


def tpta_is_alive_and_kicking_test():
    """
    >>> tpta_is_alive_and_kicking()
    <TPTAResponse version="1.1.0-SNAPSHOT">
    <error>Either the "text" parameter or the "url" parameter should be supplied</error>
    </TPTAResponse>
    """


def tpta_is_able_to_do_basic_usefull_things_test():
    """
    >>> tpta(text="Den Haag, Rotterdam en Utrecht zijn steden in Nederland, en Amsterdam is de hoofdstad.")
    ['Amsterdam', 'Den Haag', 'Nederland', 'Rotterdam', 'Utrecht']
    """


def tpta_is_able_to_do_really_usefull_things_test():
    """
    >>> result = ",".join(tpta(text_url='%s' % TEST_DOC_LOCAL1)).decode('utf-8')
    >>> exp = 'Agoedakringen,Amerika,Amsterdam,Amsterdam-Oost,Andr,Antwerpen,Beieren,Berlijn,Bonnist-Ichenhauser,Breslau,Brussel,Centr,Centraal Isra\xc3\xabl,Duitschland,Engeland,Engelsch,Europa,Frankrijk,Gouda,Groningen,Haagsche Synagogekoor,Hanau,Hilversum,Holten,Joden,Jodendom,Jodenheid,Keren Erets,Keren Hajischoew,Leeuwarden,Londen,Matzesprijzen,Memmingen,Meyer,Muiderpoortkwartier,Napoleon,Ned,Nederland,New-York,Nijmegen,Noodfonds,Palestina,Polen,Poolsche,Rotterdam,Rusland,Spanje,S\xc3\xa8vres,Tsecho-Slowakije,Volkenbond,Watergraafsmeer,West-Europa,Winschoten,Zuid,Zwolle'
    >>> str(result) == str(exp)
    True
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


def solr_is_able_to_do_basic_usefull_things_test():
    """
    >>> int(solr("title:Scheveningen").get('numFound')) - 1
    42
    """


def dac_process_is_running_test():
    """
    >>> test_process(CMD_LINE_EXEC)
    True
    """


def dac_front_to_back_surfnetonly_nginx_local_test():
    """
    >>> url = 'http://localhost:81/dac?url=%s' % TEST_DOC_LOCAL1
    >>> response = requests.get(url)
    >>> response.ok
    True
    >>> result = json.loads(response.text)
    >>> len(result.get('linkedNEs')) - 34
    42
    """


def dac_front_to_back_surfnetonly_nginx_slave_nodes_test():
    """
    >>> dac_slave_node()
    42
    """


def dac_front_to_back_office_network_nginx_local_test():
    """
    >>> url = 'http://localhost:81/dac?url=%s' % TEST_DOC_REMOTE
    >>> response = requests.get(url, timeout=20)
    >>> response.ok
    True
    >>> result = json.loads(response.text)
    >>> len(result.get('linkedNEs')) - 34
    42
    """


def dac_front_to_back_office_network_nginx_loadbalancer_test():
    """
    >>> url = 'http://%s/dac?url=%s' % (IP, TEST_DOC_REMOTE)
    >>> response = requests.get(url, timeout=20)
    >>> result = json.loads(response.text)
    >>> len(result.get('linkedNEs')) - 34
    42
    >>> # FIN!
    """

if __name__ == '__main__':
    import doctest
    doctest.testmod()
