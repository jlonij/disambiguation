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
from operator import attrgetter
from scipy import spatial


class EntityLinker():

    debug = None
    model = None
    url = None
    ne = None
    document = None

    to_link = None


    def __init__(self, model=None, debug=None):
        self.debug = debug
        self.model = models.RadialSVM()


    def link(self, url, ne=None):
        self.url = url
        self.document = Document(self.url)

        if ne:
            self.ne = ne.decode('utf-8')
            for entity in self.document.entities:
                if self.ne in [mention.text for mention in entity.mentions]:
                    self.to_link = entity
                    break
            if not self.to_link:
                self.to_link = self.document.get_entity(Mention(self.ne, None, self.document), self.document.entities)
            result = self.to_link.get_result(self.model)
            result['text'] = self.ne
            return [result]

        results = []
        for mention in self.document.mentions:
            if mention.text not in [result['text'] for result in results]:
                for entity in self.document.entities:
                    if mention in entity.mentions:
                        result = entity.get_result(self.model)
                        result['text'] = mention.text
                        results.append(result)
        return results


class Document():

    TPTA_URL = 'http://145.100.59.224:8080/tpta/analyse?lang=nl&url='

    url = None
    ocr = None
    mentions = []
    entities = []

    publ_date = None
    publ_place = None


    def __init__(self, url):
        self.url = url
        self.ocr = self.get_ocr(self.url)
        self.mentions = self.get_mentions(self.url)
        self.entities = self.get_entities(self.mentions)

        # Later pas:
        self.publ_date, self.publ_place = self.get_metadata(self.url)


    def get_ocr(self, url):
        data = urllib.urlopen(url).read()
        xml = etree.fromstring(data)
        return etree.tostring(xml, encoding='utf8',
                method='text').decode('utf-8')


    def get_mentions(self, url):
        mentions = []
        try:
            data = urllib.urlopen(self.TPTA_URL + url).read()
            xml = etree.fromstring(data)
        except:
            return []
        doc_pos = 0
        for node in xml.iter():
            if node.text and len(node.text) > 1:
                unicode_text = node.text if isinstance(node.text, unicode) else node.text.decode('utf-8')
                mention = Mention(unicode_text, node.tag, self, doc_pos)
                doc_pos = mention.end_pos
                mentions.append(mention)
        return mentions


    def get_entities(self, mentions):
        sorted_mentions = sorted(mentions, key=attrgetter('norm'), reverse=True)
        sorted_mentions = sorted(sorted_mentions, key=attrgetter('length'), reverse=True)
        entities = []
        for mention in sorted_mentions:
            entity = self.get_entity(mention, entities)
            if mention not in entity.mentions:
                entity.mentions.append(mention)
            if entity not in entities:
                entities.append(entity)
        return entities


    def get_entity(self, mention, entities):
        if not mention.norm:
            return Entity([mention], self)
        for entity in entities:
            if mention.norm in [m.norm for m in entity.mentions]:
                return entity
        candidates = []
        for entity in entities:
            for m in entity.mentions:
                if m.norm and mention.norm.split()[-1] == m.norm.split()[-1]:
                    if len(m.norm.split()) > len(mention.norm.split()):
                        if m.tpta_type == 'person' and mention.tpta_type == 'person':
                            candidates.append(entity)
                            break
        if len(candidates) == 1:
            return candidates[0]
        else:
            return Entity([mention], self)


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


class Mention():
    text = None
    tpta_type = None
    document = None

    doc_pos = 0
    start_pos = 0
    end_pos = 0

    cleaned = None
    norm = None
    last_part = None

    titles = []

    quotes = 0


    def __init__(self, text, tpta_type, document, doc_pos=0):
        self.text = text
        self.tpta_type = tpta_type
        self.document = document
        self.doc_pos = doc_pos

        # Get position in text, window
        self.start_pos, self.end_pos = self.get_position(self.document.ocr,
                self.text, self.doc_pos)
        self.window = self.get_window(self.document.ocr,
                start_pos=self.start_pos, end_pos=self.end_pos, size=10)

        # Clean and normalize input text
        self.cleaned = self.clean(self.text)
        self.norm = self.normalize(self.cleaned)
        self.norm, self.titles = self.strip_titles(self.norm)
        self.last_part = self.strip_digits(self.norm)

        # Needed for sorting
        self.length = len(self.norm.split())


    def get_position(self, document, phrase, doc_pos=None):
        start_pos = document.find(phrase, doc_pos)
        end_pos = start_pos + len(phrase)
        if start_pos > 0 and end_pos <= len(document):
            return start_pos, end_pos
        else:
            return -1, -1


    def get_window(self, document, phrase=None, start_pos=None, end_pos=None, size=None, direction=None):

        left_bow = []
        right_bow = []

        if not start_pos or not end_pos:
            start_pos = document.find(phrase)
            end_pos = start_pos + len(phrase)

        if start_pos > 0 and end_pos <= len(document):
            left_space_pos = document.rfind(' ', 0, start_pos)
            if left_space_pos > 0:
                left_bow = tokenize(document[:left_space_pos])
            right_space_pos = document.find(' ', end_pos)
            if right_space_pos > 0:
                right_bow = tokenize(document[right_space_pos:])

        if size:
            left_bow = left_bow[-size:]
            right_bow = right_bow[:size]

        if direction == 'left':
            return left_bow
        elif direction == 'right':
            return right_bow
        return left_bow + right_bow


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


    def is_date(self):
        # Check for dates to exclude
        return False


    def count_quotes(self):
        quotes = 0
        quote_chars = [u'"', u"'", u'„', u'”', u'‚', u'’']
        pos = [m.start() - 1 for m in re.finditer(re.escape(self.text), self.document.ocr)]
        pos.extend([m.end() for m in re.finditer(re.escape(self.text), self.document.ocr)])
        for p in pos:
            if self.document.ocr[p] in quote_chars:
                quotes += 1
        self.quotes = quotes
        return quotes


class Entity():

    SOLR_SERVER = 'http://linksolr.kbresearch.nl/dbpedia/'
    SOLR_ROWS = 20

    mentions = []
    document = None
    valid = None
    model = None

    link = None
    label = None
    prob = 0
    reason = None

    solr_response = None
    solr_result_count = None
    inlinks_total = 0
    max_score = 0

    descriptions = []
    candidates = []

    quotes = 0


    def __init__(self, mentions, document):
        self.mentions = mentions
        self.document = document
        self.valid = self.is_valid()


    def get_result(self, model):
        self.model = model
        if not self.reason:
            self.resolve()
        result = {}
        result['link'] = self.link
        result['label'] = self.label
        result['prob'] = self.prob
        result['reason'] = self.reason
        return result


    def resolve(self):

        # If entity is valid, query Solr for DBpedia candidates
        self.solr_response, self.solr_result_count = self.query_solr()
        if self.reason:
            return

        # If any andidates were found, initialize list of candidate descriptions
        descriptions = []
        for i in range(self.solr_result_count):
            description = Description(self.solr_response.results[i], i, self)
            descriptions.append(description)
        self.descriptions = descriptions

        # Harde criteria: name conflict
        candidates = []
        for description in self.descriptions:
            if not description.has_name_conflict():
                candidates.append(description)
        self.candidates = candidates

        if len(self.candidates) == 0:
            self.reason = "Name conflict"
            return

        # If remaining candidates: apply probabilistic model
        # Prerequisites for feature calculation
        self.inlinks_total = self.get_total_inlinks()
        self.max_score = self.get_max_score()
        self.quotes = self.count_quotes()

        # Calculate probability for remaining candidates and select best
        best_prob = 0
        best_match = 0
        for description in self.candidates:
            description.get_probability()
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


    def is_valid(self):
        if len(self.mentions[0].norm) <= 2:
            self.reason = "Entity too short"
        elif not self.mentions[0].last_part:
            self.reason = "Entity is numeric"
        elif self.mentions[0].is_date():
            self.reason = "Entity is date"
        else:
            return True
        return False


    def query_solr(self):

        # Temporary until normalization in index
        ne_parts = self.mentions[0].cleaned.split()
        last_part = None
        for part in reversed(ne_parts):
            if not part.isdigit():
                last_part = part
                break

        query = "title:\""
        query += self.mentions[0].norm + "\" OR "
        query += "title_str:\""
        query += self.mentions[0].cleaned + "\""
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


    def get_max_score(self):
        max_score = 0
        for i in range(self.solr_result_count):
            document = self.solr_response.results[i]
            if document.get('score') > max_score:
                max_score = document.get('score')
        return max_score


    def count_quotes(self):
        quotes = 0
        for mention in self.mentions:
            quotes += mention.count_quotes()
        return quotes


class Description():

    document = None
    position = None
    entity = None

    labels = []
    non_matching_labels = []

    quotes = 0

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
    entity_match = 0
    cos_sim = 0


    def __init__(self, document, position, entity):
        self.document = document
        self.position = position
        self.entity = entity

        self.labels = self.get_labels()


    def get_labels(self):
        # Normalize titles here until they become available from the index
        labels = []
        for t in self.document.get('title_str'):
            norm = self.normalize(t)
            if len(norm) > 0:
                labels.append(norm)
        return labels


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


    def has_name_conflict(self):
        self.match_id()
        self.match_titles()
        self.match_titles_last_part()

        if not self.title_exact_match:
            if not (self.title_start_match or self.title_end_match):
                if not self.last_part_match:
                    self.name_conflict = 1
                    return True
        return False


    def get_probability(self):
        # Entity features
        self.quotes = self.entity.quotes

        # Description features
        self.solr_pos = self.position / float(self.entity.SOLR_ROWS)
        if self.entity.max_score > 0:
            self.solr_score = self.document.get('score') / float(self.entity.max_score)
        if self.entity.inlinks_total > 0:
            self.inlinks = self.document.get('inlinks') / float(self.entity.inlinks_total)
        self.lang = 1 if self.document.get('lang') == 'nl' else 0
        self.disambig = self.document.get('disambig')

        # Combination features
        # String matching
        self.match_titles_levenshtein()

        # Context matching
        self.match_date()
        self.match_type()
        self.match_entities()
        self.match_abstract()

        example = []
        #print self.document.get('id')
        for j in range(len(self.entity.model.features)):
            example.append(float(getattr(self, self.entity.model.features[j])))
            #print self.entity.model.features[j], float(getattr(self,self.entity.model.features[j]))
        self.prob = self.entity.model.predict(example)
        #print self.prob
        return self.prob


    def match_id(self):
        # Use normalized title string list until they are available from the index
        match_label = self.labels[0]
        ne = self.entity.mentions[0].norm

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
        match_label = self.labels
        ne = self.entity.mentions[0].norm

        non_matching_labels = []
        for label in match_label:
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

        ne = self.entity.mentions[0].norm

        # Preliminary check for ne's that are longer than main title:
        # There has to be at least one alternative title that matches the
        # longer version
        main_label = self.labels[0]
        alt_label = self.labels[1:]
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
        last_part_match = 0
        match_label = self.non_matching_labels

        for l in match_label:

            # If the last words of the title and the ne match
            if Levenshtein.distance(ne.split()[-1], l.split()[-1]) <= 1:

                # Single word entities
                if len(ne.split()) == 1:
                    last_part_match += 1
                    continue

                # Check for any conflicting preceding parts
                skip = False
                source = l.split() if len(ne.split()) > len(l.split()) else ne.split()
                target = ne.split() if len(ne.split()) > len(l.split()) else l.split()

                target_pos = 0
                for part in source[:-1]:
                    if target_pos < len(target[:-1]):
                        if len(part) > 2 and part in target[target_pos:-1]:
                            target_pos = target.index(part) + 1
                        elif len(part) > 2 and len([p for p in
                            target[target_pos:-1] if Levenshtein.distance(p, part) <= 1]) > 0:
                            for p in target[target_pos:-1]:
                                if Levenshtein.distance(p, part) <= 1:
                                    target_pos = target.index(p) + 1
                                    break
                        elif len(part) <= 2 and part[0] in [p[0] for p in target[target_pos:-1]]:
                            target_pos = [p[0] for p in target[target_pos:-1]].index(part[0]) + 1
                        else:
                            skip = True
                            break
                    else:
                        break
                if skip:
                    continue

                last_part_match += 1

        self.last_part_match = last_part_match / float(len(self.labels))


    def match_titles_levenshtein(self):
        ne = self.entity.mentions[0].norm
        sum = 0
        for l in self.labels:
            sum += Levenshtein.ratio(ne, l)
        self.mean_levenshtein_ratio = sum / float(len(self.labels))


    def match_date(self):
        if self.entity.document.publ_date:
            year_of_publ = int(self.entity.document.publ_date[:4])
            year_of_birth = self.document.get('yob')
            if year_of_birth is not None:
                if year_of_publ < year_of_birth:
                    self.date_match = -1
                else:
                    self.date_match = 1


    def match_type(self):

        TPTA_SCHEMA_MAPPING = {'person': 'Person', 'location': 'Place', 'organisation': 'Organization'}

        tpta_type = self.entity.mentions[0].tpta_type

        if tpta_type in TPTA_SCHEMA_MAPPING:
            schema_types = self.document.get('schemaorgtype')
            if schema_types:
                for t in schema_types:
                    if t == TPTA_SCHEMA_MAPPING[tpta_type]:
                        self.type_match = 1
                        break


    def match_entities(self):
        entity_match = 0
        entity_list = [e.mentions[0].cleaned for e in
                self.entity.document.entities if e.valid and e != self.entity
                and e.mentions[0].cleaned not in ['Nederland', 'Nederlandse', 'Amsterdam', 'Amsterdamse']]
        abstract = self.document.get('abstract')
        if abstract:
            for entity in entity_list:
                if abstract.find(entity) > -1:
                    entity_match += 1
        self.entity_match = entity_match


    def match_abstract(self):
        # warnings.filterwarnings('ignore', message='.*Unicode equal comparison.*')
        abstract = self.document.get('abstract')
        window = []
        for mention in self.entity.mentions:
            window += mention.window

        if window and abstract:

            # Tokenize
            abstract = tokenize(abstract)

            # Build vocabulary
            voc = []
            for b in [window, abstract]:
                for t in b:
                    if not t in voc and len(t) > 3:
                        voc.append(t)

            # Create normalized word count vectors for both documents
            vec = []
            for b in [window, abstract]:
                v = np.zeros(len(voc))
                for t in voc:
                    v[voc.index(t)] = b.count(t)
                v_norm = v / np.linalg.norm(v)
                vec.append(v_norm)

            # Calculate the distance between the resulting vectors
            self.cos_sim = 1 - spatial.distance.cosine(vec[0], vec[1])


def tokenize(document):
    document = re.split('\W+', document)
    return [t for t in document if t]


if __name__ == '__main__':
    if not len(sys.argv) > 1:
        print("Usage: ./disambiguation.py [url (string)]")
    else:
        linker = EntityLinker(debug=True)
        if len(sys.argv) > 2:
            print(linker.link(sys.argv[1], sys.argv[2]))
        else:
            print(linker.link(sys.argv[1]))

