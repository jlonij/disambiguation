#!/usr/bin/env python

import json
import requests

CONFIG_URL = 'http://145.100.58.199:82/config.json'

def parse_config(config_url=CONFIG_URL):
    response = requests.get(config_url)
    if response.status_code == 200:
        return json.loads(response.content)
    return False

conf = parse_config()
print(conf.get('W2V_URL'))

