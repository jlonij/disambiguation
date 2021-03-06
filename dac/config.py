#!/usr/bin/env python

import json
import os
import requests

# CONFIG_URL = 'http://145.100.58.195:82/config.json'
# CONFIG_URL = 'http://kbresearch.nl/dac/dac/config.json'
CONFIG_URL = None


def parse_config(config_url=CONFIG_URL):
    if config_url:
        response = requests.get(config_url)
        if response.status_code == 200:
            return json.loads(response.content)
        return False
    else:
        abs_path = os.path.dirname(os.path.realpath(__file__))
        return json.load(open(os.path.join(abs_path, 'config.json')))
