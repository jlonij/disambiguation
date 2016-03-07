#!/usr/bin/env python
# -*- coding: utf-8 -*-

import Levenshtein
import math
import models
import numpy as np
import re
import solr
import sys
import urllib
import warnings

from lxml import etree
from scipy import spatial


class Linker():

    url = None
    ne = None
    debug = False

    context = None
    to_return = []
    to_link = []


    def link(self, url, ne=None, debug=False):

        self.url = url
        self.debug = debug

        # Create global context for all ne's
        self.context = Context(url)

        # A prediction result should be returned for all names or a single ne
        # if specified
        if ne:
            self.ne = ne.decode('utf-8')
            for name in self.context.names:
                if name.text == self.ne:
                    self.to_return = [name]
                    for entity in self.context.entities:
                        if self.to_return[0] in entity.names:
                            self.to_link = [entity]
                            break
                    break
            if len(self.to_return) == 0:
                result = {}
                result['text'] = self.ne
                result['reason'] = "Unknown entity"
                result['link'] = None
                result['label'] = None
                result['prob'] = 0
                return [result]
        else:
            self.to_return = self.context.names
            self.to_link = self.context.entities

        # Link the relevant entities
        self.model = models.RadialSVM()
        for entity in self.to_link:
            entity.resolve(self.model)

            if self.debug:
                for d in entity.descriptions:
                    print d.document.get('id')
                    for j in range(len(self.model.features)):
                        print self.model.features[j], float(getattr(d,self.model.features[j]))
                    print 'prob', d.prob

        # Return the results
        results = []
        for name in self.to_return:
            result = {}
            result['text'] = name.text
            for entity in self.context.entities:
                if name in entity.names:
                    result['link'] = entity.link
                    result['label'] = entity.label
                    result['prob'] = entity.prob
                    result['reason'] = entity.reason
                    break
            results.append(result)
        return results


class Context():

    TPTA_URL = 'http://145.100.59.224:8080/tpta/analyse?lang=nl&url='

    url = None
    ocr = None
    publ_date = None
    publ_place = None
    names = []
    entities = []


    def __init__(self, url):
        self.url = url
        self.names = self.get_names(self.url)
        self.entities = self.get_entities(self.names)
        if len(self.entities) > 0:
            self.ocr = self.get_ocr(self.url)
            self.publ_date, self.publ_place = self.get_metadata(self.url)


    def get_names(self, url):
        names = []
        done = []
        try:
            data = urllib.urlopen(self.TPTA_URL + url).read()
            xml = etree.fromstring(data)
        except:
            return []
        for node in xml.iter():
            if node.text and len(node.text) > 1:
                unicode_text = node.text if isinstance(node.text, unicode) else node.text.decode('utf-8')
                if unicode_text not in done:
                    done.append(unicode_text)
                    names.append(Name(unicode_text, node.tag, self))
        return names


    def get_entities(self, names):
        entities = []
        # Empty names all become their own entity
        empty_names = [n for n in names if len(n.norm) == 0]
        for name in empty_names:
            entities.append(Entity([name], self))
        # Combine shorter names with extended versions
        non_empty_names = [n for n in names if len(n.norm) > 0]
        for source in non_empty_names:
            targets = []
            for target in non_empty_names:
                if (target.norm.split()[-1] == source.norm.split()[-1] and
                        len(target.norm.split()) > len(source.norm.split())):
                    targets.append(target)
            if len(targets) == 1:
                source.is_repr = False
                existing_entity = False
                for entity in entities:
                    for name in entity.names:
                        if name.norm == targets[0].norm:
                            entity.add_name(source)
                            existing_entity = True
                            break
                if not existing_entity:
                    entities.append(Entity([targets[0], source], self))
            else:
                entities.append(Entity([source], self))
        return entities


    def get_ocr(self, url):
        data = urllib.urlopen(url).read()
        xml = etree.fromstring(data)
        return etree.tostring(xml, encoding='utf8', method='text')


    def get_metadata(self, url):
        publ_date, publ_place = None, None
        url = url[:url.find('mpeg21') + 6]
        try:
            data = urllib.urlopen(url).read()
            xml = etree.fromstring(data)
        except:
            return None, None
        md_id = url[url.find('ddd:'):] + ':metadata'
        for node in xml.iter('{urn:mpeg:mpeg21:2002:02-DIDL-NS}Component'):
            if node.attrib['{http://purl.org/dc/elements/1.1/}identifier'] == md_id:
                dcx = node.find('{urn:mpeg:mpeg21:2002:02-DIDL-NS}Resource/{info:srw/schema/1/dc-v1.1}dcx')
                publ_date = dcx.findtext('{http://purl.org/dc/elements/1.1/}date')
                for sp in dcx.iter('{http://purl.org/dc/terms/}spatial'):
                    if '{http://www.w3.org/2001/XMLSchema-instance}type' in sp.attrib:
                        publ_place = sp.text
        return publ_date, publ_place


class Name():

    is_repr = True

    text = None
    tpta_type = None
    context = None

    cleaned = None
    norm = None
    last_part = None

    titles = []

    quotes = 0


    def __init__(self, text, tpta_type, context):
        self.text = text
        self.tpta_type = tpta_type
        self.context = context

        # Clean and normalize input text
        self.cleaned = self.clean(self.text)
        self.norm = self.normalize(self.cleaned)
        self.norm, self.titles = self.strip_titles(self.norm)
        self.last_part = self.strip_digits(self.norm)


    def clean(self, ne):
        remove_char = ["+", "&&", "||", "!", "(", ")", "{", u'„',
                       "}", "[", "]", "^", "\"", "~", "*", "?", ":"]

        for char in remove_char:
            if ne.find(char) > -1:
                ne = ne.replace(char, u'')

        ne = ne.strip()
        return ne


    def normalize(self, ne):
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


    def strip_titles(self, ne):
        GENERAL_TITLES_MALE = ['de heer', 'dhr']
        GENERAL_TITLES_FEMALE = ['mevrouw', 'mevr', 'mw', 'mejuffrouw', 'juffrouw',
            'mej']
        ACADEMIC_TITLES = ['professor', 'prof', 'drs', 'mr', 'ing', 'ir', 'dr',
            'doctor', 'meester', 'doctorandus', 'ingenieur']
        POLITICAL_TITLES = ['minister', 'minister-president', 'staatssecretaris',
            'ambassadeur', 'kamerlid', 'burgemeester', 'wethouder',
            'gemeenteraadslid', 'consul']
        MILITARY_TITLES = ['generaal', 'gen', 'majoor', 'maj', 'luitenant',
            'kolonel', 'kol', 'kapitein']
        RELIGIOUS_TITLES = ['dominee', 'ds', 'paus', 'kardinaal', 'aartsbisschop',
            'bisschop', 'monseigneur', 'mgr', 'kapelaan', 'deken', 'abt',
            'prior', 'pastoor', 'pater', 'predikant', 'opperrabbijn', 'rabbijn',
            'imam']
        TITLES = (GENERAL_TITLES_MALE + GENERAL_TITLES_FEMALE + ACADEMIC_TITLES +
            POLITICAL_TITLES + MILITARY_TITLES + RELIGIOUS_TITLES)

        titles = []

        for t in TITLES:
            regex = '(^|\W)('+t+')(\W|$)'
            if re.search(regex, ne) is not None:
                titles.append(t)
                ne = re.sub(regex, ' ', ne)
        while ne.find('  ') > -1:
            ne = ne.replace('  ', ' ')
        ne = ne.strip()
        return ne, titles


    def strip_digits(self, ne):
        ne_parts = ne.split()
        last_part = None
        for part in reversed(ne_parts):
            if not part.isdigit():
                last_part = part
                break
        return last_part


    def count_quotes(self):
        quotes = 0
        quote_chars = ['"', "'", '„', '”', '‚', '’']
        pos = [m.start() - 1 for m in re.finditer(re.escape(self.text), self.context.ocr)]
        pos.extend([m.end() for m in re.finditer(re.escape(self.text), self.context.ocr)])
        for p in pos:
            if self.context.ocr[p] in quote_chars:
                quotes += 1
        self.quotes = quotes
        return quotes


class Entity():

    SOLR_SERVER = 'http://linksolr.kbresearch.nl/dbpedia/'
    SOLR_ROWS = 20

    names = []
    context = None
    repr_name = None

    link = None
    label = None
    prob = 0
    reason = None

    solr_response = None
    solr_result_count = None
    inlinks_total = 0
    score_total = 0

    descriptions = []

    quotes = 0

    model = None


    def __init__(self, names, context):
        self.names = names
        self.context = context


    def add_name(self, name):
        if name not in self.names:
            if name.is_repr == True:
                for n in self.names:
                    n.is_repr = False
            self.names.append(name)


    def resolve(self, model):

        # Pre-process entity information
        self.repr_name = self.get_repr_name()
        if self.reason:
            return

        # If a valid entity is found, query Solr for DBpedia candidates
        self.solr_response, self.solr_result_count = self.query_solr()
        if self.reason:
            return

        # Initialize list of candidate entity descriptions
        descriptions = []
        for i in range(self.solr_result_count):
            description = Description(self.solr_response.results[i], self)
            descriptions.append(description)
        self.descriptions = descriptions

        # Prerequisites for feature calculation
        self.inlinks_total = self.get_total_inlinks()
        self.score_total = self.get_total_score()
        self.quotes = self.count_quotes()

        # Calculate probability for all candidates
        self.model = model
        for description in self.descriptions:
            example = []
            description.match()
            for j in range(len(self.model.features)):
                example.append(float(getattr(description, self.model.features[j])))
            description.prob = self.model.predict(example)

        # Select best candidate, if any, and return the result
        best_prob = 0
        best_match = 0
        for description in self.descriptions:
            if description.prob > best_prob:
                best_prob = description.prob
                best_match = self.descriptions.index(description)

        if best_prob >= 0.5:
            self.reason = "SVM classifier best probability"
            self.link = self.descriptions[best_match].document.get('id')[1:-1]
            self.label = self.descriptions[best_match].document.get('title')[0]
        else:
            self.reason = "Probability too low for: " + self.descriptions[best_match].document.get('title')[0]
        self.prob = self.descriptions[best_match].prob


    def get_repr_name(self):
        for name in self.names:
            if name.is_repr:
                if len(name.norm) < 2 or not name.last_part:
                    self.reason = "Entity too short"
                return name


    def query_solr(self):

        # Temporary until normalization in index
        ne_parts = self.repr_name.cleaned.split()
        last_part = None
        for part in reversed(ne_parts):
            if not part.isdigit():
                last_part = part
                break

        query = "title:\""
        query += self.repr_name.norm + "\" OR "
        query += "title_str:\""
        query += self.repr_name.cleaned + "\""
        query += " OR lastpart_str:\""
        query += last_part + "\""

        self.SOLR_CONNECTION = solr.SolrConnection(self.SOLR_SERVER)

        try:
            solr_response = self.SOLR_CONNECTION.query(
                q=query, rows=self.SOLR_ROWS, indent="on",
                sort="lang,inlinks", sort_order="desc")
            numfound = solr_response.numFound
            solr_result_count = numfound if numfound <= self.SOLR_ROWS else self.SOLR_ROWS
        except Exception as error_msg:
            self.reason = "Failed to query solr: " + str(error_msg)
            self.prob = -1.0

        if solr_response is not None and solr_response.numFound == 0:
            self.reason = "Nothing found"

        return solr_response, solr_result_count


    def get_total_inlinks(self):
        inlinks_total = 0
        for i in range(self.solr_result_count):
            document = self.solr_response.results[i]
            inlinks_total += document.get('inlinks')
        return inlinks_total


    def get_total_score(self):
        score_total = 0
        for i in range(self.solr_result_count):
            document = self.solr_response.results[i]
            score_total += document.get('score')
        return score_total


    def count_quotes(self):
        quotes = 0
        for name in self.names:
            quotes += name.count_quotes()
        return quotes


class Description():

    document = None
    entity = None

    norm_title_str = []

    solr_pos = 0
    solr_score = 0
    inlinks = 0
    lang = 0
    disambig = 0

    main_title_match = 0
    main_title_start_match = 0
    main_title_end_match = 0
    main_title_exact_match = 0
    title_match = 0
    title_start_match = 0
    title_end_match = 0
    title_exact_match = 0
    last_part_match = 0
    name_conflict = 0

    mean_levenshtein_ratio = 0

    date_match = 0
    type_match = 0
    cos_sim = 0
    quotes = 0

    non_matching_labels = []


    def __init__(self, document, entity):

        self.document = document
        self.entity = entity

        # Normalize titles here until they become available from the index
        norm_title_str = []
        for t in document.get('title_str'):
            norm = self.normalize(t)
            if len(norm) > 0:
                norm_title_str.append(norm)
        self.norm_title_str = norm_title_str


    def normalize(self, ne):
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


    def match(self):

        # Entity features
        self.quotes = self.entity.quotes

        # Description features
        self.solr_pos = self.entity.descriptions.index(self) / float(self.entity.SOLR_ROWS)
        if self.entity.score_total > 0:
            self.solr_score = self.document.get('score') / float(self.entity.score_total)
        if self.entity.inlinks_total > 0:
            self.inlinks = self.document.get('inlinks') / float(self.entity.inlinks_total)
        self.lang = 1 if self.document.get('lang') == 'nl' else 0
        self.disambig = self.document.get('disambig')

        # String matching
        self.match_id()
        self.match_titles()
        self.match_titles_last_part()
        self.match_titles_levenshtein()
        self.check_name_conflict()

        # Context matching
        self.match_date()
        self.match_type()
        self.match_abstract()


    def match_id(self):
        # Use normalized title string list until they are available from the index
        # match_label = self.description.document.get('title_str')
        match_label = self.norm_title_str[0]
        ne = self.entity.repr_name.norm

        if match_label == ne:
            self.main_title_exact_match = 1
        elif match_label.endswith(ne):
            self.main_title_end_match = 1
        elif match_label.startswith(ne):
            self.main_title_start_match = 1
        elif match_label.find(ne) > -1:
            self.main_title_match = 1


    def match_titles(self):
        title_match = 0
        title_start_match = 0
        title_end_match = 0
        title_exact_match = 0

        # Use normalized title string list until they are available from the index
        # match_label = self.description.document.get('title_str')
        match_label = self.norm_title_str
        ne = self.entity.repr_name.norm

        non_matching_labels = []

        for label in match_label:

            # Skip empty labels
            if not label.strip():
                continue

            fraction = 1 / float(len(match_label))

            if label == ne:
                title_exact_match += fraction
            elif label.endswith(ne):
                title_end_match += fraction
            elif label.startswith(ne):
                title_start_match += fraction
            elif label.find(ne) > -1:
                title_match += fraction
            else:
                non_matching_labels.append(label)

        self.title_match = title_match
        self.title_start_match = title_start_match
        self.title_end_match = title_end_match
        self.title_exact_match = title_exact_match

        self.non_matching_labels = non_matching_labels


    def match_titles_last_part(self):

        last_part_match = 0
        matching_labels = []
        ne = self.entity.repr_name.norm

        # Skip single word entities
        if not len(ne.split()) > 1:
            return

        # Preliminary check for ne's that are longer than main title:
        # There has to be at least one alternative title that matches the
        # longer version
        main_label = self.norm_title_str[0]
        alt_label = self.norm_title_str[1:]
        if len(ne.split()) > len(main_label.split()):
            skip = True
            for l in alt_label:
                if len(ne.split()) == len(l.split()) and ne.split()[-1] == l.split()[-1]:
                    match = True
                    for part in ne.split()[:-1]:
                        if len(ne.split()[0]) > 2 and part != l.split()[ne.split().index(part)]:
                            match = False
                            break
                        elif len(ne.split()[0]) <= 2 and part[0] != l.split()[ne.split().index(part)][0]:
                            match = False
                            break
                    if match:
                        skip = False
                        break
            if skip:
                return

        # Last part match for titles that haven't been matched yet
        match_label = self.non_matching_labels

        for l in match_label:

            # If the last words of the title and the ne match
            if Levenshtein.ratio(ne.split()[-1], l.split()[-1]) > 0.75:

                # Check for any conflicting preceding parts
                skip = False
                source = l.split() if len(ne.split()) > len(l.split()) else ne.split()
                target = ne.split() if len(ne.split()) > len(l.split()) else l.split()

                target_pos = 0
                for part in source[:-1]:
                    if target_pos < len(target[:-1]):
                        if len(ne.split()[0]) > 2 and part in target[target_pos:-1]:
                            target_pos = target.index(part) + 1
                        elif len(ne.split()[0]) <= 2 and part[0] in [p[0] for p in target[target_pos:-1]]:
                            target_pos = [p[0] for p in target[target_pos:-1]].index(part[0]) + 1
                        else:
                            skip = True
                            break
                    else:
                        break
                if skip:
                    continue

                last_part_match += 1
                matching_labels.append(l)

        self.last_part_match = last_part_match / float(len(self.norm_title_str))

        for l in matching_labels:
            self.non_matching_labels.remove(l)


    def match_titles_levenshtein(self):
        match_label = self.norm_title_str
        ne = self.entity.repr_name.norm
        sum = 0
        for l in match_label:
            sum += Levenshtein.ratio(ne, l)
        self.mean_levenshtein_ratio = sum / float(len(self.norm_title_str))


    def check_name_conflict(self):
        if not self.title_exact_match > 0:
            if not (self.title_start_match > 0 or self.title_end_match > 0):
                if not self.last_part_match > 0:
                    self.name_conflict = 1


    def match_date(self):
        if self.entity.context.publ_date:
            year_of_publ = int(self.entity.context.publ_date[:4])
            year_of_birth = self.document.get('yob')
            if year_of_birth is not None:
                if year_of_publ < year_of_birth:
                    self.date_match = -1
                else:
                    self.date_match = 1


    def match_type(self):

        TPTA_SCHEMA_MAPPING = {'person': 'Person', 'location': 'Place', 'organisation': 'Organization'}

        tpta_type = self.entity.repr_name.tpta_type

        if tpta_type in TPTA_SCHEMA_MAPPING:
            schema_types = self.document.get('schemaorgtype')
            if schema_types:
                for t in schema_types:
                    if t == TPTA_SCHEMA_MAPPING[tpta_type]:
                        self.type_match = 1
                        break


    def match_abstract(self):
        warnings.filterwarnings('ignore', message='.*Unicode equal comparison.*')
        abstract = self.document.get('abstract')
        ocr = self.entity.context.ocr

        if ocr and abstract:
            corpus = [ocr, abstract]

            # Tokenize both documents into bow's
            punctuation = [',', '.', '(', ')', '"', "'"]
            bow = []
            for d in corpus:
                for p in punctuation:
                    d = d.replace(p, '')
                d = d.lower()
                d = [t for t in d.split() if len(t) >= 5]
                if not len(d):
                    return
                bow.append(d)

            # Build vocabulary
            voc = []
            for b in bow:
                for t in b:
                    if not t in voc:
                        voc.append(t)

            # Create normalized word count vectors for both documents
            vec = []
            for b in bow:
                v = np.zeros(len(voc))
                for t in voc:
                    v[voc.index(t)] = b.count(t)
                v_norm = v / np.linalg.norm(v)
                vec.append(v_norm)

            # Calculate the distance between the resulting vectors
            self.cos_sim = 1 - spatial.distance.cosine(vec[0], vec[1])


if __name__ == '__main__':

    if not len(sys.argv) > 1:
        print("Usage: ./disambiguation.py [url (string)]")
    else:
        linker = Linker()

    if len(sys.argv) > 2:
        print(linker.link(sys.argv[1], sys.argv[2], debug=True))
    else:
        print(linker.link(sys.argv[1], debug=True))

