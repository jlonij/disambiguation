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

from segtok.segmenter import split_multi
from segtok.tokenizer import word_tokenizer
from unidecode import unidecode

def clean(s):
    '''
    Clean string by removing unwanted characters.
    '''
    chars = ['(', ')', '[', ']', '{', '}', '<', '>', '"', '`', u'„']
    chars = ['||', '&&', u'§', '+', '=', '^', '*', '~', '#', '_', '\\']
    for c in chars:
        s = s.replace(c, u'')
    s = u' '.join(s.split())
    return s

def normalize(s):
    '''
    Normalize string by removing punctuation, capitalization, diacritics.
    '''
    chars = ['/', '.', ',', ':', '?', '!', ';', '-', '\u2013']
    for c in chars:
        s = s.replace(c, u' ')
    s = s.replace("'", u'')
    s = unidecode(s)
    s = s.lower()
    s = u' '.join(s.split())
    return s

def get_last_name(s, exclude_first_part=False):
    '''
    Extract probable last name from a string, excluding numbers, Roman numerals
    and some well-known suffixes.
    '''
    last_name = None

    # Some suffixes that shouldn't qualify as last names
    suffixes = ['jr', 'sr', 'z', 'zn', 'fils']

    # Regex to match Roman numerals
    pattern = '^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$'

    parts = s.split()

    for part in reversed(parts):
        if exclude_first_part:
            if parts.index(part) == 0:
                continue
        if part.isdigit():
            continue
        if part in suffixes:
            continue
        if re.match(pattern, part, flags=re.IGNORECASE):
            continue
        last_name = part
        break

    return last_name

def tokenize(text):
    '''
    Tokenize text using SegTok segmenter and tokenizer.
    '''
    tokens = []
    for sentence in split_multi(text):
        tokens += [t for t in word_tokenizer(sentence) if len(t) > 1]
    return tokens
