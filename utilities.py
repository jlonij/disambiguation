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

