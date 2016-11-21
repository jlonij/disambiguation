#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import urllib


def query_sparql(dbp_id):

    #dbp_id = '<http://nl.dbpedia.org/resource/Albert_Einstein>'

    endpoint_url_nl = 'http://nl.dbpedia.org/sparql?'

    par = {}
    par['default-graph-uri'] = 'http://nl.dbpedia.org'
    par['query'] = 'select ?p ?o where {' + dbp_id.encode('utf-8') + ' ?p ?o}'
    par['format'] = 'json'
    par['timeout'] = '30000'
    par['debug'] = 'on'

    url = endpoint_url_nl + urllib.urlencode(par)
    result = urllib.urlopen(url).read()
    result = json.loads(result)
    result = result['results']['bindings']

    types = [r['o']['value'] for r in result if r['p']['value'].endswith('#type')]
    types = [t.split('/')[-1] for t in types if t.startswith('http://schema.org') or t.startswith('http://dbpedia.org/ontology')]
    types = list(set(types))

    #print types

    dbp_id_str = dbp_id.split('/')[-1][:-1]
    dbp_id_tokens = [t.lower() for t in dbp_id_str.split('_') if not t.startswith('(')]

    subjects = [r['o']['value'] for r in result if r['p']['value'].endswith('dc/terms/subject')]
    subjects = [s.split(':')[-1].lower() for s in subjects]
    subject_tokens = []
    for s in subjects:
        subject_tokens += s.split('_')
    subject_tokens = list(set(subject_tokens) - set(dbp_id_tokens))
    subject_tokens = [s for s in subject_tokens if len(s) >= 5]

    #print subject_tokens

    yob = None
    birth_date = [r['o']['value'] for r in result if r['p']['value'] == 'http://dbpedia.org/ontology/birthDate']
    if len(birth_date) == 1:
        yob = int(birth_date[0].split('-')[0])

    #print yob

    return types, subject_tokens, yob


