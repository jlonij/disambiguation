#!/usr/bin/env python

import json
import requests

#CONFIG_URL = 'http://145.100.58.199:82/config.json'
CONFIG_URL = 'http://kbresearch.nl/dac/config.json'

def parse_config(config_url=CONFIG_URL, local=False):
    if local:
        return json.load(open('config.json'))
    else:
        response = requests.get(config_url)
        if response.status_code == 200:
            return json.loads(response.content)
        return False
