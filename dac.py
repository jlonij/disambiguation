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

import dictionary
import Levenshtein
import math
import models
import re
import solr
import sys
import urllib
import utilities

from lxml import etree
from operator import attrgetter

TPTA_URL = 'http://tpta.kbresearch.nl/analyse?lang=nl&url='
SOLR_URL = 'http://linksolr1.kbresearch.nl/dbpedia'
SOLR_ROWS = 20
MIN_PROB = 0.5

class EntityLinker():
    '''
    Link named entity mention(s) in an article to a DBpedia description.
    '''

    def __init__(self, debug=None, model=None, tpta_url=None, solr_url=None):
        '''
        Initialize the disambiguation model and Solr connection.
        '''
        self.debug = debug

        if model == 'svm':
            self.model = models.LinearSVM()
        elif model == 'nn':
            self.model = models.NeuralNet()
        else:
            self.model = models.LinearSVM()

        self.tpta_url = tpta_url if tpta_url else TPTA_URL
        self.solr_url = solr_url if solr_url else SOLR_URL

        self.solr_connection = solr.SolrConnection(self.solr_url)

    def link(self, url, ne=None):
        '''
        Link named entity mention(s) in an article to a DBpedia description.
        '''
        # Get context information (article ocr and recognized entities)
        self.context = Context(url, self.tpta_url)

        # Group related entities into clusters
        clusters_to_link = self.get_clusters(self.context.entities)

        # If a specific ne was requested, search for a corresponding entity in
        # the list of recognized entities
        if ne:
            ne = ne.decode('utf-8')
            entity_to_link = None
            for entity in self.context.entities:
                if ne == entity.text:
                    entity_to_link = entity
            # If not found, create new one
            if not entity_to_link:
                entity_to_link = Entity(ne, None, self.context)
                self.context.entities.append(entity_to_link)
            # Link only the cluster to which the entity belongs
            clusters_to_link = [c for c in clusters_to_link if entity_to_link
                in c.entities]

        # Process all clusters to be linked
        linked = []

        while clusters_to_link:
            cluster = clusters_to_link.pop()
            result = cluster.link(self.solr_connection, SOLR_ROWS,
                self.model, MIN_PROB)

            # If a cluster consists of multiple entities and could not be linked
            # or was not linked to a person, split it up and return the parts to
            # the queue. If not, add the cluster to the linked list.
            dependencies = [e for e in cluster.entities if e.norm !=
                cluster.entities[0].norm]

            if dependencies:
                types = []
                if result.description:
                    if result.description.document.get('schema_type'):
                        types += result.description.document.get('schema_type')
                    if result.description.document.get('dbo_type'):
                        types += result.description.document.get('dbo_type')
                if not result.description or 'Person' not in types:
                    new_clusters = [Cluster([e for e in cluster.entities if e
                        not in dependencies])]
                    new_clusters.extend(self.get_clusters(dependencies))

                    # If linking a specific ne, only return the new cluster
                    # containing that ne to the queue
                    if ne:
                        clusters_to_link.extend([c for c in new_clusters if
                            entity_to_link in c.entities])
                    else:
                        clusters_to_link.extend(new_clusters)
            else:
                linked.append(cluster)

        # Return the result for each (unique) entity
        results = []
        to_return = [entity_to_link] if ne else self.context.entities
        for entity in to_return:
            if entity.text not in [result['text'] for result in results]:
                for cluster in linked:
                    if entity in cluster.entities:
                        result = cluster.result.get_dict()
                        result['text'] = entity.text
                        results.append(result)
        return results

    def get_clusters(self, entities):
        '''
        Group related entities into clusters.
        '''
        clusters = []
        sorted_entities = sorted(entities, key=attrgetter('norm'), reverse=True)
        sorted_entities = sorted(sorted_entities, key=lambda entity:
            len(entity.norm.split()), reverse=True)
        for entity in sorted_entities:
            clusters = self.cluster(entity, clusters)
        return clusters

    def cluster(self, entity, clusters):
        '''
        Either add entity to an existing cluster or create a new one.
        '''
        for cluster in clusters:
            for e in cluster.entities:
                if entity.text == e.text:
                    cluster.entities.append(entity)
                    return clusters
                if len(entity.norm) > 0 and len(e.norm) > 0:
                    if entity.norm == e.norm:
                        cluster.entities.append(entity)
                        return clusters
        candidates = []
        for cluster in clusters:
            for e in cluster.entities:
                if len(entity.norm) > 0 and len(e.norm) > 0:
                    if entity.norm.split()[-1] == e.norm.split()[-1]:
                        if e.norm.endswith(entity.norm):
                            if len(e.norm.split()) > len(entity.norm.split()):
                                candidates.append(cluster)
                                break
        if len(candidates) == 1:
            candidates[0].entities.append(entity)
        else:
            clusters.append(Cluster([entity]))
        return clusters


class Context():
    '''
    The context information for an entity.
    '''

    def __init__(self, url, tpta_url):
        '''
        Retrieve ocr, metadata, subjects and entities.
        '''
        self.ocr = self.get_ocr(url)
        self.publ_date = self.get_metadata(url)
        self.subjects = self.get_subjects(self.ocr)
        self.entities = self.get_entities(url, tpta_url)

    def get_ocr(self, url):
        '''
        Retrieve ocr from resolver url.
        '''
        ocr = None
        try:
            data = urllib.urlopen(url).read()
            xml = etree.fromstring(data)
            ocr = etree.tostring(xml, encoding='utf8',
                method='text').decode('utf-8')
        except:
            return ocr
        return ocr

    def get_metadata(self, url):
        '''
        Retrieve metadata (currently just publication date) with sru.
        '''
        publ_date = None

        jsru_url = 'http://jsru.kb.nl/sru/sru?'
        jsru_url += 'operation=searchRetrieve&x-collection=DDD_artikel'
        jsru_url += '&query=uniqueKey=' + url[url.find('ddd:'):-4]

        try:
            data = urllib.urlopen(jsru_url).read()
            xml = etree.fromstring(data)
        except:
            return publ_date

        path = '{http://www.loc.gov/zing/srw/}records/'
        path += '{http://www.loc.gov/zing/srw/}record/'
        path += '{http://www.loc.gov/zing/srw/}recordData/'
        path += '{http://purl.org/dc/elements/1.1/}date'

        date_element = xml.find(path)
        if date_element is not None:
            publ_date = date_element.text

        return publ_date

    def get_subjects(self, ocr):
        '''
        Extract subjects from ocr (based on dictionary for now).
        '''
        subjects = []
        for subject in dictionary.subjects:
            words = dictionary.subjects[subject]
            for role in dictionary.roles:
                if subject in dictionary.roles[role]['subjects']:
                    words += dictionary.roles[role]['words']
            window = [utilities.normalize(w) for w in utilities.tokenize(ocr)]
            if len(set(words) & set(window)) > 0:
                subjects.append(subject)
        return subjects

    def get_entities(self, url, tpta_url):
        '''
        Retrieve the list of entities from the NER service and instantiate an
        entity object for each one.
        '''
        entities = []
        try:
            data = urllib.urlopen(tpta_url + url).read()
            xml = etree.fromstring(data)
        except:
            return entities

        # Keep track of the position of each entity in the document so that
        # entity mentions with identical surface forms can be kept apart
        doc_pos = 0
        for node in xml.iter():
            if node.text and len(node.text) > 1:
                entity = Entity(node.text.decode('utf-8'), node.tag,
                    self, doc_pos)
                doc_pos = entity.end_pos if entity.end_pos > -1 else doc_pos
                entities.append(entity)
        return entities


class Entity():
    '''
    An entity mention occuring in an article.
    '''

    def __init__(self, text, tpta_type, context, doc_pos=0):
        '''
        Gather information about the entity and its immediate surroundings.
        '''
        self.text = text
        self.tpta_type = tpta_type
        self.context = context
        self.doc_pos = doc_pos

        self.norm = utilities.normalize(self.text)

        self.start_pos, self.end_pos = self.get_position(self.text,
            self.context.ocr, self.doc_pos)
        self.window_left, self.window_right = self.get_window(self.context.ocr,
            start_pos=self.start_pos, end_pos=self.end_pos, size=30)

        self.quotes = self.get_quotes()
        self.title, self.title_form = self.get_title()
        self.role, self.role_form = self.get_role()

        self.stripped = self.strip_titles()
        self.last_part = utilities.get_last_name(self.stripped)
        self.valid = self.is_valid()

        self.alt_type = self.get_alt_type()

    def get_position(self, phrase, document, doc_pos=None):
        '''
        Find the start and end position of the mention in the article.
        '''
        start_pos = document.find(phrase, doc_pos)
        end_pos = start_pos + len(phrase)
        if start_pos >= 0 and end_pos <= len(document):
            return start_pos, end_pos
        else:
            return -1, -1

    def get_window(self, document, start_pos=None, end_pos=None, size=None):
        '''
        Get the words appearing to the left and right of the entity.
        '''
        left_bow = []
        right_bow = []

        if start_pos >= 0 and end_pos <= len(document):
            left_space_pos = document.rfind(' ', 0, start_pos)
            left_new_line_pos = document.rfind('\n', 0, start_pos)
            left_pos = max([left_space_pos, left_new_line_pos])
            if left_pos > 0:
                left_bow = utilities.tokenize(document[:left_pos])
            right_space_pos = document.find(' ', end_pos)
            right_new_line_pos = document.find('\n', end_pos)
            right_pos = min([right_space_pos, right_new_line_pos])
            if right_pos > 0:
                right_bow = utilities.tokenize(document[right_space_pos:])

        if size:
            left_bow = left_bow[-size:]
            right_bow = right_bow[:size]

        return left_bow, right_bow

    def get_quotes(self):
        '''
        Count quote characters surrounding the entity.
        '''
        quotes = 0
        quote_chars = [u'"', u"'", u'„', u'”', u'‚', u'’']
        for pos in [self.start_pos - 1, self.start_pos, self.end_pos - 1,
                self.end_pos]:
            if self.context.ocr[pos] in quote_chars:
                quotes += 1
        return quotes

    def get_title(self):
        '''
        Check for titles near the beginning of the entity.
        '''
        words = [self.norm.split()[0]]
        if self.window_left:
            words.append(utilities.normalize(self.window_left[-1]))
        for word in words:
            if word in dictionary.titles:
                return True, word
        return None, None

    def get_role(self):
        '''
        Check for roles near the beginning and end of the entity.
        '''
        words = [self.norm.split()[0]]
        if self.window_left:
            words.append(utilities.normalize(self.window_left[-1]))
        if self.window_right and self.context.ocr[self.end_pos] == ',':
            words.append(utilities.normalize(self.window_right[0]))
        for word in words:
            for role in dictionary.roles:
                if word in dictionary.roles[role]['words']:
                    return role, word
        return None, None

    def strip_titles(self):
        '''
        Remove titles and roles appearing inside the entity.
        '''
        if self.title and self.norm.split()[0] == self.title_form:
            return ' '.join(self.norm.split()[1:])
        if self.role and self.norm.split()[0] == self.role_form:
            return ' '.join(self.norm.split()[1:])
        return self.norm

    def is_valid(self):
        '''
        Check entity validity.
        '''
        if len(self.stripped) >= 2 and self.last_part and not self.is_date():
            return True
        return False

    def is_date(self):
        '''
        Check if the entity is some sort of date.
        '''
        if [w for w in self.norm.split() if w in dictionary.months]:
            if [w for w in self.norm.split() if w.isdigit()]:
                return True
        return False

    def get_alt_type(self):
        '''
        Infer addtional information about entity type from context.
        '''
        if self.title:
            return 'person'

        if self.role:
            if len(dictionary.roles[self.role]['types']) == 1:
                return dictionary.roles[self.role]['types'][0]

        if self.window_left:
            prev_word = utilities.normalize(self.window_left[-1])
            if prev_word in ['in', 'te', 'uit']:
                return 'location'

        return None


class Cluster():
    '''
    Group of related entity mentions, presumed to refer to the same entity.
    '''

    def __init__(self, entities):
        '''
        Initialize cluster.
        '''
        self.entities = entities
        self.quotes_total = self.get_total_quotes()

    def link(self, solr_connection, solr_rows, model, min_prob):
        '''
        Get the link result for the cluster.
        '''
        # Check validity of the main entity
        if not self.entities[0].valid:
            self.result = Result("Invalid entity")
            return self.result

        # If entity is valid, try to query Solr for candidate descriptions
        try:
            cand_list = CandidateList(solr_connection, solr_rows,
                self.entities[0].norm, self.entities[0].last_part)
        except Exception as msg:
            self.result = Result("Failed to query solr: " + str(msg), -1.0)
            return self.result

        # Check the number of descriptions found
        cand_list.generate(self)
        if len(cand_list.candidates) == 0:
            self.result = Result("Nothing found")
            return self.result

        # Filter descriptions according to hard criteria, e.g. name conlfict
        cand_list.filter()
        if len(cand_list.filtered_candidates) == 0:
            self.result = Result("Name or date conflict")
            return self.result

        # If any candidates remain, calculate their feature values and
        # probability and select the best candidate
        cand_list.rank(model)
        best_match = cand_list.ranked_candidates[0]
        if best_match.prob >= min_prob:
            self.result = Result("Predicted link", best_match.prob, best_match)
        else:
            self.result= Result("Probability too low for: " +
                best_match.document.get('label'), best_match.prob)
        return self.result

    def get_total_quotes(self):
        '''
        Get the total number of quotes for all entities in the cluster.
        '''
        total_quotes = 0
        for e in self.entities:
            total_quotes += e.quotes
        return total_quotes


class CandidateList():
    '''
    List of candidate links for an entity cluster.
    '''

    def __init__(self, solr_connection, solr_rows, norm, last_part):
        '''
        Query the Solr index.
        '''
        self.solr_rows = solr_rows

        queries = []

        query = 'pref_label:"' + norm + '"'
        query += ' OR pref_label_str:"' + norm + '"'
        query += ' OR alt_label_str:"' + norm + '"'
        query += ' OR last_part_str:"' + last_part + '"'

        queries.append(query)

        solr_iteration = None
        solr_response = None
        for i, query in enumerate(queries):
            solr_iteration = i
            solr_response = solr_connection.query(q=query, rows=solr_rows,
                indent='on', sort='lang,inlinks', sort_order='desc')
            if solr_response.numFound > 0:
                break

        self.solr_iteration = solr_iteration
        self.solr_response = solr_response
        if solr_response.numFound < solr_rows:
            self.solr_result_count = solr_response.numFound
        else:
            self.solr_result_count = solr_rows

    def generate(self, cluster):
        '''
        Generate initial list of candidates.
        '''
        self.candidates = []
        for r in self.solr_response.results:
            self.candidates.append(Description(r, self, cluster))

    def filter(self):
        '''
        Filter descriptions according to hard criteria, e.g. name conlfict
        '''
        self.filtered_candidates = []
        for c in self.candidates:
            c.calculate_rule_features()
            if c.name_conflict == 0 and c.date_match > -1:
                self.filtered_candidates.append(c)

    def rank(self, model):
        '''
        Rank candidates according to trained model.
        '''
        self.model = model

        self.solr_max_score, self.cand_max_score = self.get_max_score()
        self.solr_inlinks_total, self.cand_inlinks_total = self.get_total_inlinks()
        self.titles_total = self.get_total_titles()

        for c in self.filtered_candidates:
            c.calculate_prob_features()
            example = []
            for j in range(len(model.features)):
                example.append(float(getattr(c, model.features[j])))
            c.prob = model.predict(example)

        self.ranked_candidates = sorted(self.filtered_candidates,
            key=attrgetter('prob'), reverse=True)

    def get_max_score(self):
        '''
        Get the maximum score for the result set.
        '''
        solr_max_score = 0
        for c in self.candidates:
            if c.document.get('score') > solr_max_score:
                solr_max_score = c.document.get('score')
        cand_max_score = 0
        for c in self.filtered_candidates:
            if c.document.get('score') > cand_max_score:
                cand_max_score = c.document.get('score')
        return solr_max_score, cand_max_score

    def get_total_inlinks(self):
        '''
        Get the total number of inlinks for the result set.
        '''
        solr_inlinks_total = 0
        for c in self.candidates:
            solr_inlinks_total += c.document.get('inlinks')
        cand_inlinks_total = 0
        for c in self.filtered_candidates:
            cand_inlinks_total += c.document.get('inlinks')
        return solr_inlinks_total, cand_inlinks_total

    def get_total_titles(self):
        '''
        Get the total number of titles for the result set.
        '''
        titles_total = 0
        for c in self.filtered_candidates:
            titles_total += len(c.labels)
        return titles_total


class Description():

    document = None
    position = None
    cluster = None

    labels = []
    non_matching_labels = []

    main_title_match = 0
    main_title_start_match = 0
    main_title_end_match = 0
    main_title_exact_match = 0

    title_match = 0
    title_start_match = 0
    title_end_match = 0
    title_exact_match = 0

    title_match_fraction = 0
    title_start_match_fraction = 0
    title_end_match_fraction = 0
    title_exact_match_fraction = 0

    last_part_match = 0
    last_part_match_fraction = 0

    name_conflict = 0

    solr_iteration = 0
    solr_pos = 0
    cand_pos = 0
    solr_score = 0
    cand_score = 0
    solr_inlinks = 0
    cand_inlinks = 0

    lang = 0
    quotes = 0
    disambig = 0

    mean_levenshtein_ratio = 0

    date_match = 0
    type_match = 0
    role_match = 0
    subject_match = 0
    entity_match = 0
    spec_match = 0
    cat_match = 0

    prob = 0


    def __init__(self, document, cand_list, cluster):
        self.document = document
        self.cand_list = cand_list
        self.cluster = cluster

        self.labels = self.get_labels()


    def get_labels(self):
        labels = [self.document.get('pref_label')]
        if self.document.get('alt_label'):
            labels += self.document.get('alt_label')
        return labels


    def calculate_rule_features(self):
        self.match_id()
        self.match_titles()
        self.match_titles_last_part()

        features = ['main_title_exact_match', 'main_title_end_match',
                'title_exact_match', 'title_end_match', 'last_part_match']
        name_conflict = 1
        for f in features:
            if getattr(self, f) > 0:
                name_conflict = 0
                break
        self.name_conflict = name_conflict

        self.match_date()


    def calculate_prob_features(self):
        self.solr_iteration = self.cand_list.solr_iteration

        # Position
        self.position = self.cand_list.solr_response.results.index(self.document)
        self.solr_pos = (self.position + 1) / float(self.cand_list.solr_rows)
        self.cand_pos = (self.cand_list.filtered_candidates.index(self) + 1) / float(self.cand_list.solr_rows)

        # Score
        if self.cand_list.solr_max_score > 0:
            self.solr_score = self.document.get('score') / float(self.cand_list.solr_max_score)
        if self.cand_list.cand_max_score > 0:
            self.cand_score = self.document.get('score') / float(self.cand_list.cand_max_score)

        # Inlinks
        if self.cand_list.solr_inlinks_total > 0:
            self.solr_inlinks = self.document.get('inlinks') / float(self.cand_list.solr_inlinks_total)
        if self.cand_list.cand_inlinks_total > 0:
            self.cand_inlinks = self.document.get('inlinks') / float(self.cand_list.cand_inlinks_total)

        self.quotes = self.cluster.quotes_total
        self.lang = 1 if self.document.get('lang') == 'nl' else 0
        self.disambig = 1 if self.document.get('ambig') == 0 else 1

        self.match_titles_levenshtein()
        self.match_type()
        self.match_role()
        self.match_subjects()
        self.match_entities()
        self.match_spec()
        self.match_cat()


    def match_id(self):
        match_label = self.labels[0]
        ne = self.cluster.entities[0].norm

        non_matching_labels = []
        if match_label == ne:
            self.main_title_exact_match = 1
        elif match_label.endswith(ne):
            self.main_title_end_match = 1
        elif match_label.startswith(ne):
            self.main_title_start_match = 1
        elif match_label.find(ne) > -1:
            self.main_title_match = 1
        else:
            non_matching_labels.append(match_label)
        self.non_matching_labels = non_matching_labels


    def match_titles(self):
        title_match = 0
        title_start_match = 0
        title_end_match = 0
        title_exact_match = 0

        match_label = self.labels[1:]
        ne = self.cluster.entities[0].norm

        non_matching_labels = []
        for label in match_label:
            if label == ne:
                title_exact_match += 1
            elif label.endswith(ne):
                title_end_match += 1
            elif label.startswith(ne):
                title_start_match += 1
            elif label.find(ne) > -1:
                title_match += 1
            else:
                non_matching_labels.append(label)

        self.title_match = title_match
        self.title_start_match = title_start_match
        self.title_end_match = title_end_match
        self.title_exact_match = title_exact_match

        if len(match_label) > 0:
            self.title_match_fraction = title_match / float(len(match_label))
            self.title_start_match_fraction = title_start_match / float(len(match_label))
            self.title_end_match_fraction = title_end_match / float(len(match_label))
            self.title_exact_match_fraction = title_exact_match / float(len(match_label))

        self.non_matching_labels += non_matching_labels


    def match_titles_last_part(self):
        ne = self.cluster.entities[0].stripped

        # Preliminary check for ne's that are longer than the main label:
        # There has to be at least one alternative label that matches the
        # longer version
        main_label = self.labels[0]
        alt_label = self.labels[1:]
        if len(ne.split()) > len(main_label.split()):
            skip = True
            for l in alt_label:
                if len(ne.split()) == len(l.split()) and ne.split()[-1] == l.split()[-1]:
                    match = True
                    for part in ne.split()[:-1]:
                        if len(ne.split()[0]) > 1 and part != l.split()[ne.split().index(part)]:
                            match = False
                            break
                        elif len(ne.split()[0]) <= 1 and part[0] != l.split()[ne.split().index(part)][0]:
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
                        if len(part) > 1 and part in target[target_pos:-1]:
                            target_pos = target.index(part) + 1
                        elif len(part) > 1 and len([p for p in
                            target[target_pos:-1] if Levenshtein.distance(p, part) <= 1]) > 0:
                            for p in target[target_pos:-1]:
                                if Levenshtein.distance(p, part) <= 1:
                                    target_pos = target.index(p) + 1
                                    break
                        elif len(part) <= 1 and part[0] in [p[0] for p in target[target_pos:-1]]:
                            target_pos = [p[0] for p in target[target_pos:-1]].index(part[0]) + 1
                        else:
                            skip = True
                            break
                    else:
                        break
                if skip:
                    continue

                last_part_match += 1

        self.last_part_match = last_part_match
        if len(self.labels) > 0:
            self.last_part_match_fraction = last_part_match / float(len(self.labels))


    def match_titles_levenshtein(self):
        ne = self.cluster.entities[0].norm
        sum = 0
        for l in self.labels:
            sum += Levenshtein.ratio(ne, l)
        self.mean_levenshtein_ratio = sum / float(len(self.labels))


    def match_date(self):
        publ_date = self.cluster.entities[0].context.publ_date
        if publ_date:
            year_of_publ = int(publ_date[:4])
            year_of_birth = self.document.get('birth_year')
            if year_of_birth:
                age = year_of_publ - year_of_birth
                #print age
                if age <= 0:
                    self.date_match = -1
                elif age < 20:
                    self.date_match = 0
                elif age < 100:
                    self.date_match = 2
                else:
                    self.date_match = 1


    def match_type(self):
        # Alt_type preferred over original tpta_type
        tpta_type = None
        if self.cluster.entities[0].alt_type:
            tpta_type = self.cluster.entities[0].alt_type
        elif self.cluster.entities[0].tpta_type:
            tpta_type = self.cluster.entities[0].tpta_type
        if not tpta_type or tpta_type not in dictionary.types:
            return

        schema_types = []
        if self.document.get('schema_type'):
            schema_types += self.document.get('schema_type')
        if self.document.get('dbo_type'):
            schema_types += self.document.get('dbo_type')
        # If no types available, try to deduce type from first sentence
        # of the abstract
        if not schema_types:
            abstract = self.document.get('abstract')
            if not abstract:
                return
            sentence = abstract[:abstract.find('. ')]
            #print sentence
            bow = utilities.tokenize(utilities.normalize(sentence))
            cand_types = []
            for role in [r for r in dictionary.roles if len(dictionary.roles[r]['types']) == 1]:
                if len(set(bow) & set(dictionary.roles[role]['words'])) > 0:
                    cand_types.append(dictionary.roles[role]['types'][0])
                    #print role
            for t in dictionary.types:
                if len(set(bow) & set(dictionary.types[t]['words'])) > 0:
                    cand_types.append(t)
                    #print t
            if len(set(cand_types)) == 1:
                schema_types = dictionary.types[cand_types[0]]['schema_types']
            else:
                return

        #print schema_types

        # Matching type
        for t in dictionary.types[tpta_type]['schema_types']:
            if t in schema_types:
                self.type_match = 1
                return

        # Non-matching: persons can't be locations or organizations
        if tpta_type == 'person':
            for confl in [d for d in dictionary.types if d != tpta_type]:
                for t in dictionary.types[confl]['schema_types']:
                    if t in schema_types:
                        self.type_match = -1
                        return

        # Non-matching: locations and organizations can't be persons
        elif 'Person' in schema_types:
            self.type_match = -1


    def match_role(self):
        roles = {e.role for e in self.cluster.entities if e.role}
        #print roles
        if not roles:
            return

        role_match = 0

        # Match schema.org types
        schema_types = []
        if self.document.get('schema_type'):
            schema_types += self.document.get('schema_type')
        if self.document.get('dbo_type'):
            schema_types += self.document.get('dbo_type')

        if schema_types:
            for role in roles:
                for t in dictionary.roles[role]['schema_types']:
                    if t in schema_types:
                        self.role_match = 1
                        return

        # Match first sentence abstract
        abstract = self.document.get('abstract')
        if abstract:
            sentence = abstract[:abstract.find('. ')]
            bow = utilities.tokenize(utilities.normalize(sentence))
            for role in roles:
                if len(set(bow) & set(dictionary.roles[role]['words'])) > 0:
                    self.role_match = 1
                    return

        # Check for conflict
        if schema_types:
            for role in [r for r in dictionary.roles if r not in roles]:
                for t in dictionary.roles[role]['schema_types']:
                    if t in schema_types:
                        self.role_match = -1
                        return
        if abstract:
            for role in [r for r in dictionary.roles if r not in roles]:
                if len(set(bow) & set(dictionary.roles[role]['words'])) > 0:
                    self.role_match = -1
                    return


    def match_subjects(self):
        subjects = self.cluster.entities[0].context.subjects
        if not subjects:
            return

        subject_match = 0
        abstract = self.document.get('abstract')
        if abstract:
            bow = utilities.tokenize(utilities.normalize(abstract))
            for subject in subjects:
                words = dictionary.subjects[subject]
                for role in dictionary.roles:
                    if subject in dictionary.roles[role]['subjects']:
                        words += dictionary.roles[role]['words']
                if len(set(words) & set(bow)) > 0:
                    subject_match += 1

            # Check for conflict
            if subject_match == 0:
                for subject in [s for s in dictionary.subjects if s not in
                        subjects]:
                    words = dictionary.subjects[subject]
                    for role in dictionary.roles:
                        if subject in dictionary.roles[role]['subjects']:
                            if len(set(dictionary.roles[role]['subjects']) &
                                    set(subjects)) == 0:
                                words += dictionary.roles[role]['words']
                    if len(set(words) & set(bow)) > 0:
                        subject_match = -1
                        break

        self.subject_match = subject_match


    def match_entities(self):
        entity_match = 0
        excluded_entities = ['Nederland', 'Nederlandse', 'Amsterdam', 'Amsterdamse']
        found_entities = []
        abstract = self.document.get('abstract')
        if abstract:
            entity_list = [e.text for e in self.cluster.entities[0].context.entities
                    if e.valid and self.cluster.entities[0].text.find(e.text)
                    == -1 and e.text not in excluded_entities]
            for entity in entity_list:
                if entity not in found_entities and abstract.find(entity) > -1:
                    found_entities.append(entity)
                    entity_match += 1
        entity_match = entity_match if entity_match < 3 else 3
        self.entity_match = entity_match


    def match_spec(self):
        if self.document.get('spec'):
            spec = self.document.get('spec')
            spec_words = [w[:int(math.ceil(len(w) * 0.75))] for w in
                    filter(None, re.split("[_\- ]+", spec)) if len(w) > 3]
            if not spec_words:
                return
            window = []
            for e in self.cluster.entities:
                window += e.window_left + e.window_right
            if not window:
                return
            spec_match = 0
            for word in spec_words:
                for w in window:
                    if w.startswith(word):
                        spec_match += 1
                        break
            self.spec_match = spec_match


    def match_cat(self):
        categories = self.document.get('keyword')
        if not categories:
            return
        ocr = self.cluster.entities[0].context.ocr
        tokens = utilities.tokenize(utilities.normalize(ocr))
        self.cat_match = len(set(categories) & set(tokens))
        #print self.cat_match


class Result():

    link = None
    label = None
    prob = None
    reason = None

    features = None

    description = None


    def __init__(self, reason, prob=0, description=None):
        self.reason = reason
        self.prob = prob
        if description:
            self.description = description
            self.link = description.document.get('id')
            self.label = description.document.get('label')

            features = {}
            for j in range(len(description.cand_list.model.features)):
                features[description.cand_list.model.features[j]] = float(getattr(description,
                        description.cand_list.model.features[j]))
            self.features = features


    def get_dict(self):
        result = {}
        result['link'] = self.link
        result['label'] = self.label
        result['prob'] = self.prob
        result['reason'] = self.reason
        result['features'] = self.features
        return result


if __name__ == '__main__':
    import pprint
    if not len(sys.argv) > 1:
        print("Usage: ./dac.py [url (string)]")
    else:
        linker = EntityLinker(debug=True)
        if len(sys.argv) > 2:
            pprint.pprint(linker.link(sys.argv[1], sys.argv[2]))
        else:
            pprint.pprint(linker.link(sys.argv[1]))

