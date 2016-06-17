#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re


def clean(ne):
    remove_char = ["+", "&&", "||", "!", "(", ")", "{", u'„',
                   "}", "[", "]", "^", "\"", "~", "*", "?", ":"]

    for char in remove_char:
        if ne.find(char) > -1:
            ne = ne.replace(char, u'')

    ne = ne.strip()
    return ne


def normalize(ne):
    if ne.find('.') > -1:
        ne = ne.replace('.', ' ')
    if ne.find('-') > -1:
        ne = ne.replace('-', ' ')
    if ne.find(u'\u2013') > -1:
        ne = ne.replace(u'\u2013', ' ')
    while ne.find('  ') > -1:
        ne = ne.replace('  ', ' ')
    ne = ne.strip()
    ne = ne.lower()
    return ne


def tokenize(document):
    document = re.split('\W+', document, flags=re.UNICODE)
    return [t for t in document if t]

