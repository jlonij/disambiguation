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

import re

from segtok.segmenter import split_multi
from segtok.tokenizer import word_tokenizer
from unidecode import unidecode


def clean(s):
    '''
    Clean string by removing unwanted characters.
    '''
    chars = ['+', '=', '^', '*', '~', '#', '_', '\\']
    chars += ['(', ')', '[', ']', '{', '}', '<', '>']
    chars += ['\'', '"', '`', '%']
    for c in chars:
        s = s.replace(c, u'')
    s = u' '.join(s.split())
    return s


def normalize(s):
    '''
    Normalize string by removing punctuation, capitalization, diacritics.
    '''
    # Replace diactritics
    s = unidecode(s)
    # Remove unwanted characters
    s = clean(s)
    # Remove capitalization
    s = s.lower()
    # Replace regular punctuation by spaces
    chars = ['.', ',', ':', '?', '!', ';', '-', '/', '|', '&']
    for c in chars:
        s = s.replace(c, u' ')
    s = u' '.join(s.split())
    return s


def normalize_ocr(s):
    '''
    Generate a common OCR error tolerant search string from an already
    normalized string.
    '''
    if len(s) > 1:
        # Equate e, c (not as first character)
        s = ''.join([s[0], s[1:].replace('c', 'e')])

        # Equate i, l (not as first character)
        s = ''.join([s[0], s[1:].replace('l', 'i')])

        # Equate G, C, O (only as first character)
        if s[0] == 'c' or s[0] == 'o':
            s = ''.join(['g', s[1:]])

        # Equate B en E (only as first character)
        if s[0] == 'b':
            s = ''.join(['e', s[1:]])

    return s


def get_last_part(s, exclude_first_part=False):
    '''
    Extract probable last name from a string, excluding numbers, Roman
    numerals and some well-known suffixes.
    '''
    last_part = None

    # Some suffixes that shouldn't qualify as last names
    suffixes = ['jr', 'sr', 'z', 'zn', 'fils']

    # Regex to match Roman numerals
    pattern = '^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$'

    parts = s.split()

    for part in reversed(parts):
        if exclude_first_part:
            if parts.index(part) == 0:
                break
        if part.isdigit():
            continue
        if part in suffixes:
            continue
        if re.match(pattern, part, flags=re.IGNORECASE):
            continue
        last_part = part
        break

    prefixes = ['van', 'de', 'der', 'het', 'von']

    if last_part:
        for part in reversed(parts[:parts.index(last_part)]):
            if exclude_first_part:
                if parts.index(part) == 0:
                    break
            if part in prefixes:
                last_part = ' '.join([part, last_part])

    return last_part


def segment(text):
    '''
    Split text into sentences using SegTok segmenter.
    '''
    return split_multi(text)


def tokenize(text, segment=True, norm=True, unique=False, min_len=2):
    '''
    Tokenize text using SegTok segmenter and tokenizer.
    '''
    sentences = split_multi(text) if segment else [text]

    tokens = []

    for s in sentences:
        if norm:
            tokens += [w for t in word_tokenizer(s) for w in
                       normalize(t).split()]
        else:
            tokens += word_tokenizer(s)

    if unique:
        tokens = list(set(tokens))

    if min_len:
        tokens = [t for t in tokens if len(t) >= min_len]

    return tokens
