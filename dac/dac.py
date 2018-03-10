#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# DAC Entity Linker
#
# Copyright (C) 2017-2018 Juliette Lonij - Koninklijke Bibliotheek,
# National Library of the Netherlands
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

# Standard library imports
import argparse
import math
import re

from operator import attrgetter
from operator import itemgetter
from pprint import pprint

# Third party imports
import Levenshtein
import numpy as np
import requests
from lxml import etree
from sklearn.metrics.pairwise import cosine_similarity

# DAC imports
import config
import dictionary
import models
import solr
import utilities

# Service locations
conf = config.parse_config()

TPTA_URL = conf.get('TPTA_URL')
JSRU_URL = conf.get('JSRU_URL')
SOLR_URL = conf.get('SOLR_URL')
W2V_URL = conf.get('W2V_URL')
TOPICS_URL = conf.get('TOPICS_URL')

# Constant values
WINDOW = 20
SOLR_ROWS = 25
MIN_PROB = 0.5


class EntityLinker(object):
    '''
    Link named entity mention(s) in an article to a DBpedia description.
    '''

    def __init__(self, model=None, debug=False, features=False,
                 candidates=False, error_handling=True):
        '''
        Initialize the disambiguation model and Solr connection.
        '''
        if model == 'train':
            self.model = models.BaseModel()
        elif model == 'svm':
            self.model = models.LinearSVM()
        elif model == 'nn':
            self.model = models.NeuralNet()
        elif model == 'bnn':
            self.model = models.BranchingNeuralNet()
        else:
            self.model = models.NeuralNet()

        self.debug = debug
        self.features = features
        self.candidates = candidates
        self.error_handling = error_handling

        self.solr_connection = solr.SolrConnection(SOLR_URL)

    def link(self, url, ne=None):
        '''
        Link named entity mention(s) in an article to a DBpedia description.
        '''
        # Get context information (article metadata, ocr, entities)
        ne = ne.decode('utf-8') if ne else None

        try:
            self.context = Context(url, ne)
        except Exception as e:
            if self.error_handling:
                return {'status': 'error', 'message':
                        'Error retrieving context: ' + str(e)}
            else:
                raise

        # If a specific ne was requested, select it from the list of entities
        if ne:
            entity_to_link = None
            for entity in self.context.entities:
                if ne == entity.text:
                    entity_to_link = entity

        # Group related entities into clusters
        clusters_to_link = self.get_clusters(self.context.entities)
        if ne:
            # Link only the cluster to which the entity belongs
            clusters_to_link = [c for c in clusters_to_link if entity_to_link
                                in c.entities]

        # Process all clusters to be linked
        clusters_linked = []

        while clusters_to_link:
            cluster = clusters_to_link.pop()
            try:
                result = cluster.link(self.solr_connection, self.model)
            except Exception as e:
                if self.error_handling:
                    return {'status': 'error', 'message':
                            'Error linking entity: ' + str(e)}
                else:
                    raise

            # If a cluster consists of multiple, significantly differing
            # entities and could not be linked, split it up and return the
            # new clustres to the queue.
            sub_entities = [e for e in cluster.entities if
                            Levenshtein.distance(e.norm,
                                                 cluster.entities[0].norm) > 1]

            if sub_entities:
                if not result.description:
                    new_clusters = [Cluster([e for e in cluster.entities if e
                                    not in sub_entities])]
                    new_clusters.extend(self.get_clusters(sub_entities))

                    # If linking a specific ne, only return the new cluster
                    # containing that ne to the queue
                    if ne:
                        clusters_to_link.extend([c for c in new_clusters if
                                                 entity_to_link in c.entities])
                    else:
                        clusters_to_link.extend(new_clusters)
                else:
                    clusters_linked.append(cluster)
            else:
                clusters_linked.append(cluster)

        # Return the result for each (unique) entity
        results = []
        to_return = [entity_to_link] if ne else self.context.entities
        for entity in to_return:
            if entity.text not in [result['text'] for result in results]:
                for cluster in clusters_linked:
                    if entity in cluster.entities:
                        result = cluster.result.get_dict(
                            features=self.features, candidates=self.candidates)
                        result['text'] = entity.text
                        if self.debug or 'link' in result:
                            results.append(result)

        return {'status': 'ok', 'linkedNEs': results}

    def get_clusters(self, entities):
        '''
        Group related entities into clusters.
        '''
        clusters = []
        # Arrange the entities in reversed alphabetical order
        sorted_entities = sorted(entities, key=attrgetter('norm'),
                                 reverse=True)
        # Arrange the entities by word length, longest first
        sorted_entities = sorted(sorted_entities, key=lambda entity:
                                 len(entity.norm.split()), reverse=True)
        # Assign each entity to a cluster
        for entity in sorted_entities:
            clusters = self.cluster(entity, clusters)
        # Merge possessives
        clusters = self.merge_possessives(clusters)

        return clusters

    def cluster(self, entity, clusters):
        '''
        Either add entity to an existing cluster or create a new one.
        '''
        # If the entity text or norm exactly matches an existing cluster,
        # add it to the cluster
        for cluster in clusters:
            for e in cluster.entities:
                if entity.text == e.text:
                    cluster.entities.append(entity)
                    return clusters
                if entity.norm and e.norm:
                    if entity.norm == e.norm:
                        cluster.entities.append(entity)
                        return clusters

        # Find candidate clusters that partially match an entity
        candidates = []
        for cluster in clusters:
            for e in cluster.entities:
                if entity.valid and e.valid:

                    # Last parts are the same
                    if entity.norm.split()[-1] == e.norm.split()[-1]:
                        # Any preceding parts are the same
                        if e.norm.endswith(entity.norm):
                            # The candidate norm is longer than the entity norm
                            if len(e.norm.split()) > len(entity.norm.split()):
                                candidates.append(cluster)
                                break

                    # First parts are the same
                    elif entity.norm.split()[0] == e.norm.split()[0]:
                        # Entity norm consists of exactly one word (first name)
                        if len(entity.norm.split()) == 1:
                            # The candidate norm is longer than the entity norm
                            if len(e.norm.split()) > len(entity.norm.split()):
                                # Both entities are probably persons
                                if e.tpta_type == 'person':
                                    if entity.tpta_type == 'person':
                                        candidates.append(cluster)
                                        break

        if len(candidates) == 1:
            candidates[0].entities.append(entity)
            return clusters

        clusters.append(Cluster([entity]))
        return clusters

    def merge_possessives(self, clusters):
        '''
        Try to add possessive forms to existing clusters.
        '''
        new_clusters = [c for c in clusters if not c.entities[0].valid or
                        (c.entities[0].norm[-1] != 's' or
                         len(c.entities[0].norm.split()) > 1)]
        poss_clusters = [c for c in clusters if c.entities[0].valid and
                         c.entities[0].norm[-1] == 's' and
                         len(c.entities[0].norm.split()) == 1]

        for p in poss_clusters:
            merge = False
            for n in new_clusters:
                if (p.entities[0].valid and n.entities[0].valid and
                        n.entities[0].norm.split()[-1] + 's' ==
                        p.entities[0].norm.split()[-1]):
                    n.entities.extend(p.entities)
                    merge = True
                    break
            if not merge:
                new_clusters.append(p)

        return new_clusters


class Context(object):
    '''
    The context information for an entity.
    '''

    def __init__(self, url, ne=None):
        '''
        Retrieve ocr, metadata, topics and entities.
        '''
        self.url = url
        self.ne = ne

        # Article enitities, ocr and metadata are retrieved immediately;
        # other context information (topics) are added later if needed.
        self.get_metadata()
        self.get_entities()

    def get_metadata(self):
        '''
        Retrieve article metadata from SRU API. Only news articles and
        illustrations with captions will be processed.
        '''
        payload = {}
        payload['operation'] = 'searchRetrieve'
        payload['x-collection'] = 'DDD_artikel'
        payload['query'] = 'uniqueKey=' + self.url.split('urn=')[-1][:-4]

        response = requests.get(JSRU_URL, params=payload, timeout=30)
        assert response.status_code == 200, 'Error retrieving metadata'

        xml = etree.fromstring(response.content)

        type_element = xml.find('.//{http://purl.org/dc/elements/1.1/}type')
        assert type_element is not None, 'Unknown article type'
        assert type_element.text in ['illustratie met onderschrift',
                                     'artikel'], 'Invalid article type'

        date_element = xml.find('.//{http://purl.org/dc/elements/1.1/}date')
        self.publ_year = (int(date_element.text[:4]) if date_element is not
                          None else None)

    def get_entities(self):
        '''
        Retrieve article ocr and recognized entities from NER service.
        '''
        payload = {}
        payload['url'] = self.url
        payload['ne'] = self.ne
        payload['context'] = WINDOW

        response = requests.get(TPTA_URL, params=payload, timeout=300)
        assert response.status_code == 200, 'TPTA error'

        response.encoding = 'utf-8'
        data = response.json()
        assert 'entities' in data, 'TPTA error retrieving entities'
        assert 'text' in data, 'TPTA error retrieving OCR'

        # Article ocr
        ocr = ''
        if 'title' in data['text']:
            ocr += data['text']['title']
        if 'p' in data['text']:
            ocr += u' ' + data['text']['p']

        self.ocr = ocr
        self.tokenize_ocr()

        # Article entities
        entities = []

        # Regular entities first
        for e in [e for e in data['entities'] if e['type'] != 'manual' and
                  len(e['ner_src']) >= 1 and e['ne'][0].isupper()]:
            if len(e['ne']) > 1:
                entity = Entity(e['ne'], e['count'], e['type'],
                                e['type_certainty'], e['pos'], e['ne_context'],
                                e['left_context'], e['right_context'], self)
                entities.append(entity)

        # User requested entity
        if self.ne:
            if self.ne not in [e.text for e in entities]:
                occurences = [e for e in data['entities'] if e['type'] ==
                              'manual' and e['pos'] > -1]
                if occurences:
                    e = sorted(occurences, key=itemgetter('source'),
                               reverse=True)[0]
                    entity = Entity(e['ne'], 0, None, 0, e['pos'],
                                    e['ne_context'], e['left_context'],
                                    e['right_context'], self)
                else:
                    entity = Entity(self.ne, 0, self.ne, 0, -1, self.ne, '',
                                    '', self)
                entities.append(entity)

        # Check entity - text ratio
        else:
            if len(entities) > 5:
                message = 'Invalid entity proportion'
                assert len(entities) / float(len(self.ocr_bow)) < 0.35, message

        self.entities = entities

    def get_topics(self):
        '''
        Retrieve topic probabilities from topics service.
        '''
        payload = {'url': self.url}
        response = requests.get(TOPICS_URL, params=payload, timeout=300)
        assert response.status_code == 200, 'Error retrieving topics'

        self.topics = response.json()['topics']

    def normalize_ocr(self):
        self.ocr_norm = utilities.normalize(self.ocr)

    def tokenize_ocr(self):
        self.ocr_bow = utilities.tokenize(self.ocr, unique=True)


class Entity(object):
    '''
    An entity mention occuring in an article.
    '''

    def __init__(self, text, count=0, tpta_type=None, type_certainty=0,
                 pos=-1, ne_context='', left_context='', right_context='',
                 context=None):

        '''
        Get information about the entity and its immediate surroundings.
        '''
        self.text = text
        self.count = count
        self.tpta_type = tpta_type
        self.type_certainty = type_certainty
        self.pos = pos
        self.ne_context = ne_context
        self.left_context = left_context
        self.right_context = right_context
        self.context = context

        # Tokenize context
        self.window_left = utilities.tokenize(left_context)
        self.window_right = utilities.tokenize(right_context)

        # Clean, analyze entity string
        self.norm = utilities.normalize(self.text)
        self.title, self.title_form = self.get_title()
        self.role, self.role_form = self.get_role()
        self.stripped = self.strip_titles()
        self.last_part = utilities.get_last_part(self.stripped)

        # Check result validity
        if self.is_valid():
            # Get some additional info
            self.quotes = self.get_quotes()
            self.alt_type = self.get_alt_type()

    def get_title(self):
        '''
        Check for titles near the beginning of the entity.
        '''
        words = []
        if len(self.norm.split()) > 1:
            words.append(self.norm.split()[0])
        if self.window_left:
            words.append(self.window_left[-1])
        for word in words:
            if word in dictionary.titles:
                return True, word
        return None, None

    def get_role(self):
        '''
        Check for roles near the beginning and end of the entity.
        '''
        words = []
        if len(self.norm.split()) > 1:
            words.append(self.norm.split()[0])
        if self.window_left:
            words.append(self.window_left[-1])
        if self.window_right and self.ne_context[-1] == ',':
            words.append(self.window_right[0])

        for word in words:
            for role in dictionary.roles_vocab:
                for r in dictionary.roles_vocab[role]:
                    if (word == r or word == r + 's' or word == r + 'en' or
                            len(r) >= 5 and word.endswith(r)):
                        return role, word

        return None, None

    def strip_titles(self):
        '''
        Remove titles and roles appearing inside the entity string.
        '''
        if self.norm:
            if self.title and self.norm.split()[0] == self.title_form:
                return ' '.join(self.norm.split()[1:])
            if self.role and self.norm.split()[0] == self.role_form:
                return ' '.join(self.norm.split()[1:])

        return self.norm

    def is_valid(self):
        '''
        Check entity validity.
        '''
        if [w for w in self.stripped.split() if len(w) >= 2]:
            if self.last_part and not self.is_date():
                self.valid = True
                return True
        self.valid = False
        return False

    def is_date(self):
        '''
        Check if the entity is some sort of date.
        '''
        if [w for w in self.norm.split() if w in dictionary.months]:
            if [w for w in self.norm.split() if w.isdigit()]:
                return True
        return False

    def get_quotes(self):
        '''
        Count quote characters surrounding the entity.
        '''
        quotes = 0
        quote_chars = [u'"', u"'", u'„', u'”', u'‚', u'’']
        for q in quote_chars:
            quotes += self.ne_context.count(q)
        return quotes

    def get_alt_type(self):
        '''
        Infer addtional information about entity type from context.
        '''
        # A title implies a person
        if self.title:
            return 'person'

        # A role may imply a type
        if self.role:
            if self.role.split('_')[1] in dictionary.types_dbo:
                return self.role.split('_')[1]

        # Preceding word may imply a type
        if self.window_left:
            if self.window_left[-1] in ['in', 'te', 'uit']:
                return 'location'

            for t in dictionary.types_vocab:
                for v in dictionary.types_vocab[t]:
                    if v in self.window_left[-1]:
                        return t

        return None

    def substitute(self):
        '''
        Try to substitute norm with basic spelling variant.
        '''
        subs = []

        # Replace oo with o
        if self.stripped.find('oo') > -1:
            subs.append(self.stripped.replace('oo', 'o'))

        # Replace y with ij
        if self.stripped.find('y') > -1:
            subs.append(self.stripped.replace('y', 'ij'))

        # Replace ae with aa
        if self.stripped.find('ae') > -1:
            subs.append(self.stripped.replace('ae', 'aa'))

        # Remove trailing s
        if self.stripped.endswith('s'):
            subs.append(self.stripped[:-1])

        # Replace sch(e) with s(e)
        pattern = r'(^|\s)([a-zA-Z]{2,})sch(e?)'
        if re.search(pattern, self.stripped):
            subs.append(re.sub(pattern, r'\1\2s\3', self.stripped))

        # Replace trailing v with w
        pattern = r'(^|\s)([a-zA-Z]{2,})v($|\s)'
        if re.search(pattern, self.stripped):
            subs.append(re.sub(pattern, r'\1\2w\3', self.stripped))

        # Replace trailing w with v
        pattern = r'(^|\s)([a-zA-Z]{2,})w($|\s)'
        if re.search(pattern, self.stripped):
            subs.append(re.sub(pattern, r'\1\2v\3', self.stripped))

        # If there is exactly one possible substitution, replace norm, stripped
        # and last_part
        if len(subs) == 1:
            self.norm = self.norm.replace(self.stripped, subs[0])
            self.stripped = subs[0]
            self.last_part = utilities.get_last_part(self.stripped)
            return True

        return False


class Cluster(object):
    '''
    Group of related entity mentions, presumed to refer to the same entity.
    '''

    def __init__(self, entities):
        '''
        Initialize cluster.
        '''
        self.entities = entities
        self.context = self.entities[0].context

    def link(self, solr_connection, model):
        '''
        Get the link result for the cluster.
        '''
        # Check validity of the main entity
        if not self.entities[0].valid:
            self.result = Result("Invalid entity")
            return self.result

        # If entity is valid, try to query Solr for candidate descriptions
        cand_list = CandidateList(self, solr_connection, model)

        # Check the number of descriptions found
        if not cand_list.candidates:
            self.result = Result("Nothing found")
            return self.result

        # Filter descriptions according to hard criteria, e.g. name conlfict
        cand_list.filter()
        if not cand_list.filtered_candidates:
            self.result = Result("Name or date conflict", cand_list=cand_list)
            return self.result

        # If any candidates remain, calculate probabilities and select the best
        cand_list.rank()
        best_match = cand_list.ranked_candidates[0]
        if best_match.prob >= MIN_PROB:
            self.result = Result("Predicted link", best_match.prob, best_match,
                                 cand_list=cand_list)
        else:
            self.result = Result("Probability too low for: " +
                                 best_match.document.get('label'),
                                 best_match.prob, best_match,
                                 cand_list=cand_list)
        return self.result

    def get_type_ratios(self):
        '''
        Get the type ratios for the cluster.
        '''
        type_ratios = {t: 0.0 for t in dictionary.types_dbo}

        types = [t for e in self.entities for t in [e.tpta_type] *
                 e.type_certainty if e.tpta_type]
        types += [t for e in self.entities for t in [e.alt_type] * 2
                  if e.alt_type]

        if types:
            for t in dictionary.types_dbo:
                type_ratios[t] = types.count(t) / float(len(types))

        self.type_ratios = type_ratios

    def get_window(self):
        '''
        Get combined window of all cluster entities, excluding entity parts.
        '''
        if not hasattr(self, 'entity_parts'):
            self.get_entity_parts()

        entity_parts = self.entity_parts

        window = []
        for e in self.entities:
            window += e.window_left
            window += e.window_right

            if e.title:
                window.append(e.title_form)
            if e.role:
                window.append(e.role_form)

        window = [w for w in window if len(w) > 4 and w not in entity_parts
                  and w not in dictionary.unwanted]
        window = list(set(window))

        self.window = window

    def get_entity_parts(self):
        self.entity_parts = list(set([p for e in self.entities for p in
                                 e.stripped.split()]))

    def get_context_entity_parts(self):
        if not hasattr(self, 'entity_parts'):
            self.get_entity_parts()

        context_entity_parts = [p for e in self.context.entities for p in
                                e.norm.split() if p not in self.entity_parts
                                and p not in dictionary.unwanted and
                                len(p) > 4 and e.valid]

        self.context_entity_parts = list(set(context_entity_parts))


class CandidateList(object):
    '''
    List of candidate links for an entity cluster.
    '''

    def __init__(self, cluster, solr_connection, model):
        '''
        Query the Solr index and generate initial list of candidates.
        '''
        self.cluster = cluster
        self.solr_connection = solr_connection
        self.model = model

        candidates = []

        for i in range(2):
            if candidates:
                break
            if i == 1:
                if not self.cluster.entities[0].substitute():
                    break

            norm = self.cluster.entities[0].norm
            stripped = self.cluster.entities[0].stripped
            last_part = self.cluster.entities[0].last_part

            queries = []
            queries.append('pref_label_str:"' + norm + '" OR pref_label_str:"'
                           + stripped + '"')
            queries.append('(alt_label_str:"' + norm + '" OR alt_label_str:"' +
                           stripped + '") AND pref_label:[* TO *]')
            queries.append('pref_label:"' + norm + '" OR pref_label:"' +
                           stripped + '"')
            queries.append('last_part_str:"' + last_part + '"')
            self.queries = queries

            for query_id, query in enumerate(queries):
                if not len(candidates) < SOLR_ROWS:
                    break
                else:
                    rows = SOLR_ROWS - len(candidates)

                solr_response = solr_connection.query(q=query, rows=rows,
                                                      indent='on',
                                                      sort='lang,inlinks',
                                                      sort_order='desc')

                for r in solr_response.results:
                    if r.get('id') not in [c.document.get('id') for c in
                                           candidates]:
                        candidates.append(Description(r, i, query_id, self,
                                          cluster))

        self.candidates = candidates

    def filter(self):
        '''
        Filter descriptions according to hard criteria, e.g. name conflict.
        '''
        self.filtered_candidates = []
        for c in self.candidates:
            c.set_rule_features()
            if c.match_str_conflict == 0 and c.match_txt_date > -1:
                self.filtered_candidates.append(c)

    def rank(self):
        '''
        Rank candidates according to trained model.
        '''
        for c in self.filtered_candidates:
            c.set_prob_features()

            # Only calculate prob if not in training mode
            if self.model.__class__.__name__ != 'BaseModel':
                example = []
                for j in range(len(self.model.features)):
                    feature = getattr(c, self.model.features[j])
                    example.append(float(feature))
                c.prob = self.model.predict(example)

        self.ranked_candidates = sorted(self.filtered_candidates,
                                        key=attrgetter('prob'), reverse=True)

    def set_max_score(self):
        '''
        Set the maximum Solr score of the filtered candidates.
        '''
        self.max_score = max([c.document.get('score') for c in
                              self.filtered_candidates])

    def set_sum_inlinks(self):
        '''
        Set the sum of inlinks and inlinks_newspapers for the filtered
        candidates.
        '''
        for link_type in ['inlinks', 'inlinks_newspapers']:
            link_sum = sum([c.document.get(link_type) for c in
                            self.filtered_candidates if
                            c.document.get(link_type)])
            setattr(self, 'sum_' + link_type, link_sum)


class Description(object):
    '''
    Description of a link candidate.
    '''
    def __init__(self, document, query_iteration, query_id, cand_list,
                 cluster):
        '''
        Initialize description.
        '''
        self.document = document
        self.query_iteration = query_iteration
        self.query_id = query_id
        self.cand_list = cand_list
        self.cluster = cluster
        self.prob = 0.0

        self.features = self.cand_list.model.features
        for f in self.features:
            setattr(self, f, 0)

    def set_rule_features(self):
        '''
        Set the feature values needed for rule-based candidate filtering.
        '''
        # Mention - description context match: date conflict
        if self.set_date_match() > -1:

            # Mention - description string match: name conflict
            self.set_pref_label_match()
            self.set_alt_label_match()
            self.set_last_part_match()
            self.set_non_matching()

            # Mention - description context match: name conflict
            self.set_txt_last_part_match()

            self.set_name_conflict()

    def set_date_match(self):
        '''
        Compare publication year of the article with birth and death years in
        the entity description.
        '''
        publ_year = self.cluster.context.publ_year
        if not publ_year:
            return 0

        birth_year = self.document.get('birth_year')
        death_year = self.document.get('death_year')
        if not birth_year:
            return 0
        if not death_year:
            death_year = birth_year + 80

        if publ_year < birth_year:
            self.match_txt_date = -1
        elif publ_year < birth_year + 20:
            self.match_txt_date = 0.5
        elif publ_year < death_year + 20:
            self.match_txt_date = 1
        else:
            self.match_txt_date = 0.75

        return self.match_txt_date

    def set_pref_label_match(self):
        '''
        Match the main description label with the normalized entity.
        '''
        self.non_matching = []

        label = self.document.get('pref_label')
        ne = self.cluster.entities[0].norm

        if not set(ne.split()) - set(label.split()):
            if label == ne:
                self.match_str_pref_label_exact = 1
            elif label.endswith(ne):
                self.match_str_pref_label_end = 1
            elif label.find(ne) > -1:
                self.match_str_pref_label = 1
            else:
                self.non_matching.append(label)
        else:
            self.non_matching.append(label)

    def set_alt_label_match(self):
        '''
        Match alternative labels with the normalized entity.
        '''
        labels = self.document.get('alt_label')
        if not labels:
            return

        ne = self.cluster.entities[0].norm

        alt_label_exact_match = 0
        alt_label_end_match = 0
        alt_label_match = 0
        for l in labels:
            if not set(ne.split()) - set(l.split()):
                if l == ne:
                    alt_label_exact_match += 1
                elif l.endswith(ne):
                    alt_label_end_match += 1
                elif l.find(ne) > -1:
                    alt_label_match += 1
                else:
                    self.non_matching.append(l)
            else:
                self.non_matching.append(l)

        self.match_str_alt_label_exact = math.tanh(alt_label_exact_match * 0.25)
        self.match_str_alt_label_end = math.tanh(alt_label_end_match * 0.25)
        self.match_str_alt_label = math.tanh(alt_label_match * 0.25)

    def set_last_part_match(self):
        '''
        Match the last part of the stripped entity with all labels,
        making sure preceding parts (e.g. initials) don't conflict.
        '''
        labels = self.non_matching
        if not labels:
            return

        ne = self.cluster.entities[0].stripped
        # if len(ne.split()) == 1:
        #    return

        last_part_match = 0
        for l in labels[:]:

            if len(ne.split()) > len(l.split()):
                continue

            # If the last words of the title and the ne match approximately,
            # i.e. edit distance does not exceed 1
            if Levenshtein.distance(ne.split()[-1], l.split()[-1]) <= 1:

                # Multi-word entities: check for conflicts among preceding parts
                conflict = False
                source = ne.split()
                target = l.split()

                target_pos = 0
                for part in source[:-1]:
                    if target_pos < len(target[:-1]):

                        # Full words
                        if len(part) > 1 and part in target[target_pos:-1]:
                            target_pos = target.index(part) + 1

                        # First names may differ with one character, but not
                        # the first character
                        elif len(part) > 1:
                            if [p for p in target[target_pos:-1]
                                    if p[0] == part[0] and
                                    Levenshtein.distance(p, part) == 1]:
                                for p in target[target_pos:-1]:
                                    if p[0] == part[0]:
                                        if Levenshtein.distance(p, part) == 1:
                                            target_pos = target.index(p) + 1
                                            break

                        # Initials
                        elif len(part) <= 1:
                            if part[0] in [p[0] for p in
                                           target[target_pos:-1]]:
                                target_pos = [p[0] for p in
                                              target[target_pos:-1]].index(
                                                  part[0]) + 1

                        else:
                            conflict = True
                            break
                    else:
                        conflict = True
                        break

                if not conflict:
                    last_part_match += 1
                    self.non_matching.remove(l)

        self.match_str_last_part = math.tanh(last_part_match * 0.25)

    def set_txt_last_part_match(self):
        '''
        Find last name labels in the article text, in case the entity
        mention consists of only a first name.
        '''
        ne = self.cluster.entities[0].norm
        if len(ne.split()) > 1:
            return

        last_part = self.document.get('last_part')
        if not last_part:
            return

        labels = [self.document.get('pref_label')]
        if self.document.get('wd_alt_label'):
            labels.extend(self.document.get('wd_alt_label'))
        labels = [l for l in labels if len(l.split()) > 1 and
                  l.split()[0] == ne]
        if not labels:
            return

        if not hasattr(self.cluster.context, 'ocr_norm'):
            self.cluster.context.normalize_ocr()
        ocr = self.cluster.context.ocr_norm

        self.match_txt_last_part = -1
        for l in labels:
            if ocr.find(' '.join(l.split()[1:])) > -1:
                self.match_txt_last_part = 1
                break

    def set_non_matching(self):
        '''
        Count total number of non matching labels.
        '''
        self.match_str_non_matching = math.tanh(len(self.non_matching) * 0.25)

    def set_name_conflict(self):
        '''
        Determine if the description has a name conflict, i.e. not a single
        sufficiently matching label was found.
        '''
        features = ['match_str_pref_label_exact', 'match_str_pref_label_end',
                    'match_str_alt_label_exact', 'match_str_alt_label_end',
                    'match_str_last_part', 'match_txt_last_part']

        if sum([getattr(self, f) for f in features]) == 0:
            self.match_str_conflict = 1
        else:
            self.match_str_conflict = 0

    def set_prob_features(self):
        '''
        Set the additional feature values needed for probability-based
        candidate ranking.
        '''
        # Mention representation
        self.set_entity_quotes()
        self.set_entity_confidence()

        # Description representation
        self.set_candidate_lang()
        self.set_candidate_ambig()
        self.set_candidate_inlinks()

        # Mention - description string match
        self.set_solr_properties()
        self.set_levenshtein()
        self.set_abbr_match()

        # Mention - description context match
        self.set_txt_labels_match()
        self.set_spec_match()
        self.set_keyword_match()
        self.set_role_match()
        self.set_type_match()
        self.set_topic_match()
        self.set_vector_match()
        self.set_entity_match()
        # self.set_entity_match_newspapers()
        self.set_entity_vector_match()

    def set_entity_quotes(self):
        '''
        Count number of quotes surrounding entity mentions.
        '''
        if 'entity_quotes' not in self.features:
            return

        if not hasattr(self.cluster, 'sum_quotes'):
            self.cluster.sum_quotes = sum([e.quotes for e in
                                           self.cluster.entities])

        self.entity_quotes = math.tanh(self.cluster.sum_quotes * 0.25)

    def set_entity_confidence(self):
        '''
        Calculate the mean NER confidence for the cluster.
        '''
        if 'entity_ner_confidence' not in self.features:
            return

        if not hasattr(self.cluster, 'mean_ner_confidence'):
            self.cluster.mean_ner_confidence = (sum([e.count for e in
                                                     self.cluster.entities]) /
                                                float(len(
                                                    self.cluster.entities)))

        self.entity_ner_confidence = self.cluster.mean_ner_confidence / 3

    def set_candidate_lang(self):
        '''
        Determine if description is available in Dutch.
        '''
        if 'candidate_lang' not in self.features:
            return

        self.candidate_lang = 1 if self.document.get('lang') == 'nl' else -1

    def set_candidate_ambig(self):
        '''
        Determine if the description label is ambiguous.
        '''
        if 'candidate_ambig' not in self.features:
            return

        self.candidate_ambig = 1 if self.document.get('ambig') == 1 else -1

    def set_candidate_inlinks(self):
        '''
        Determine inlinks feature values.
        '''
        if not [f for f in self.features if f.startswith('candidate_inlinks')]:
            return

        for link_type in ['inlinks', 'inlinks_newspapers']:
            link_count = self.document.get(link_type)
            if link_count:
                setattr(self, 'candidate_' + link_type,
                        math.tanh(link_count * 0.001))
                if not hasattr(self.cand_list, 'sum_' + link_type):
                    self.cand_list.set_sum_inlinks()
                link_sum = getattr(self.cand_list, 'sum_' + link_type)
                if link_sum:
                    link_count_rel = link_count / float(link_sum)
                    setattr(self, 'candidate_' + link_type + '_rel',
                            link_count_rel)

    def set_solr_properties(self):
        '''
        Determine Solr iteration, position and score.
        '''
        if not [f for f in self.features if f.startswith('match_str_solr')]:
            return

        # Solr iteration
        self.match_str_solr_query_0 = 1 if self.query_id == 0 else 0
        self.match_str_solr_query_1 = 1 if self.query_id == 1 else 0
        self.match_str_solr_query_2 = 1 if self.query_id == 2 else 0
        self.match_str_solr_query_3 = 1 if self.query_id == 3 else 0
        self.match_str_solr_substitution = 1 if self.query_iteration == 1 else 0

        # Solr position (relative to other remaining candidates)
        pos = self.cand_list.filtered_candidates.index(self)
        self.match_str_solr_position = 1.0 - math.tanh(pos * 0.25)

        # Solr score (relative to other remaining candidates)
        if not hasattr(self.cand_list, 'max_score'):
            self.cand_list.set_max_score()
        if self.cand_list.max_score:
            self.match_str_solr_score = (self.document.get('score') /
                                         float(self.cand_list.max_score))

    def set_levenshtein(self):
        '''
        Mean and max Levenshtein ratio for all labels.
        '''
        if not [f for f in self.features if f.startswith('match_str_lsr')]:
            return

        ne = self.cluster.entities[0].norm

        # Pref label
        l = self.document.get('pref_label')
        self.match_str_lsr_pref = Levenshtein.ratio(ne, l)

        # Wikidata alt labels
        if self.document.get('wd_alt_label'):
            wd_labels = self.document.get('wd_alt_label')
            ratios = [Levenshtein.ratio(ne, l) for l in wd_labels]
            self.match_str_lsr_wd_max = max(ratios) - 0.5
            self.match_str_lsr_wd_mean = (sum(ratios) /
                                          float(len(wd_labels))) - 0.375
        else:
            wd_labels = []

        # Any other alt labels
        if self.document.get('alt_label'):
            labels = self.document.get('alt_label')
            labels = [l for l in labels if l not in wd_labels]
            if labels:
                ratios = [Levenshtein.ratio(ne, l) for l in labels]
                self.match_str_lsr_alt_max = max(ratios) - 0.5
                self.match_str_lsr_alt_mean = (sum(ratios) /
                                               float(len(labels))) - 0.375

    def set_abbr_match(self):
        '''
        Match abbreviations with labels, initials of labels, abstract.
        '''
        if not [f for f in self.features if f.startswith('match_str_abbr') or
                f.startswith('entity_abbr')]:
            return

        ne = self.cluster.entities[0]
        if (ne.text.isupper() and len(ne.norm) <= 5 and
                len(ne.norm.split()) == 1):

            if 'entity_abbr' in self.features:
                self.entity_abbr = 1

            # Exactly match pref or alt label
            if (self.match_str_pref_label_exact or
                    self.match_str_alt_label_exact):
                self.match_str_abbr_labels = 1

            # Matches initials of pref or alt label
            labels = [self.document.get('pref_label')]
            if self.document.get('alt_label'):
                labels += self.document.get('alt_label')
            for l in labels:
                initials = ''.join([t[0] for t in l.split()])
                if ne.norm == initials:
                    self.match_str_abbr_initials = 1
                    break

            # Appears in abstract
            if not hasattr(self, 'abstract_bow'):
                self.tokenize_abstract()
            if ne.norm in self.abstract_bow[:50]:
                self.match_str_abbr_abstract = 1

    def set_txt_labels_match(self):
        '''
        Find labels longer than the entity in the article text.
        '''
        if 'match_txt_labels' not in self.features:
            return

        ne = self.cluster.entities[0].norm

        labels = []
        if len(ne) < len(self.document.get('pref_label')):
            labels.append(self.document.get('pref_label'))
        if self.document.get('wd_alt_label'):
            labels.extend([l for l in self.document.get('wd_alt_label') if
                          len(ne) < len(l)])

        if not labels:
            return

        if not hasattr(self.cluster.context, 'ocr_norm'):
            self.cluster.context.normalize_ocr()
        ocr = self.cluster.context.ocr_norm

        for l in labels:
            if ocr.find(l) > -1:
                self.match_txt_labels = 1
                break

    def set_spec_match(self):
        '''
        Find the specification (between brackets) in the article text.
        '''
        if 'match_txt_spec' not in self.features:
            return

        spec = self.document.get('spec')
        if spec:
            spec_stem = spec[:int(math.ceil(len(spec) * 0.8))]
        else:
            return

        if not hasattr(self.cluster.context, 'ocr_norm'):
            self.cluster.context.normalize_ocr()

        ocr = self.cluster.context.ocr_norm
        if ocr.find(spec_stem) > -1:
            self.match_txt_spec = 1

    def set_keyword_match(self):
        '''
        Find DBpedia category keywords in the article text.
        '''
        if 'match_txt_keyword' not in self.features:
            return

        if not self.document.get('keyword'):
            return

        key_stems = [w[:int(math.ceil(len(w) * 0.8))] for w in
                     self.document.get('keyword') if w not in
                     dictionary.unwanted]
        if not key_stems:
            return

        bow = self.cluster.context.ocr_bow
        key_match = len([w for w in bow for s in key_stems if w.startswith(s)])
        self.match_txt_keyword = math.tanh(key_match * 0.25)

    def set_role_match(self):
        '''
        Match entity and description role (politician, athlete, etc.).
        '''
        if 'match_txt_role' not in self.features:
            return

        roles = {e.role for e in self.cluster.entities if e.role}
        if not roles:
            return

        # Match DBpedia ontology types
        dbo_types = []
        if self.document.get('dbo_type'):
            dbo_types += self.document.get('dbo_type')

        if dbo_types:
            for role in roles:
                for t in dictionary.roles_dbo[role]:
                    if t in dbo_types:
                        self.match_txt_role = 1
                        return

            # Check for conflict
            for role in [r for r in dictionary.roles_dbo if r not in roles]:
                for t in dictionary.roles_dbo[role]:
                    if t in dbo_types:
                        self.match_txt_role = -1
                        return

        else:
            if not hasattr(self, 'topic_probs'):
                self.get_topics()

            description_role = ''

            if max(self.topic_probs.values()) >= 0.4:
                description_role += max(self.topic_probs,
                                        key=self.topic_probs.get)
                if max(self.type_probs.values()) >= 0.4:
                    description_role += '_' + max(self.type_probs,
                                                  key=self.type_probs.get)
                    if description_role.endswith('other'):
                        description_role = description_role.replace('other',
                                                                    'concept')

            if description_role in roles:
                self.match_txt_role = 1
                return

    def set_type_match(self):
        '''
        Match entity and description type (person, location or organization).
        '''
        etf = [f for f in self.features if f.startswith('entity_type')]
        ctf = [f for f in self.features if f.startswith('candidate_type')]
        mtf = [f for f in self.features if f.startswith('match_txt_type')]
        if not (etf or ctf or mtf):
            return

        # Set entity type features
        if not hasattr(self.cluster, 'type_ratios'):
            self.cluster.get_type_ratios()
        type_ratios = self.cluster.type_ratios

        if etf:
            for tr in type_ratios:
                setattr(self, 'entity_type_' + tr, type_ratios[tr])

        if ctf or mtf:
            # Set candidate type features
            description_types = {t: 0.0 for t in dictionary.types_dbo}

            dbo_types = []
            if self.document.get('dbo_type'):
                dbo_types += self.document.get('dbo_type')

            if dbo_types:
                for t in dbo_types:
                    for r in dictionary.types_dbo:
                        if t in dictionary.types_dbo[r]:
                            description_types[r] = 1.0

                if not sum(description_types.values()):
                    description_types['other'] = 1.0

            else:
                if not hasattr(self, 'topic_probs'):
                    self.get_topics()
                description_types = self.type_probs

            if ctf:
                for t in description_types:
                    setattr(self, 'candidate_type_' + t, description_types[t])

            if mtf:
                # Matching type
                for r in type_ratios:
                    if type_ratios[r] >= 0.25 and description_types[r] >= 0.5:
                        self.match_txt_type += (type_ratios[r] *
                                                description_types[r])

                if self.match_txt_type:
                    return

                # Non-matching: persons can't be locations or organisations
                if type_ratios['person'] >= 0.25:
                    if description_types['location'] >= 0.5:
                        self.match_txt_type -= (type_ratios['person'] *
                                                description_types['location'])
                    if description_types['organisation'] >= 0.5:
                        self.match_txt_type -= (type_ratios['person'] *
                                                description_types[
                                                    'organisation'])

                # Non-matching: locations and organisations can't be persons
                if type_ratios['location'] >= 0.25:
                    if description_types['person'] >= 0.5:
                        self.match_txt_type -= (description_types['person'] *
                                                type_ratios['location'])

                if type_ratios['organisation'] >= 0.25:
                    if description_types['person'] >= 0.5:
                        self.match_txt_type -= (description_types['person'] *
                                                type_ratios['organisation'])

    def set_topic_match(self):
        '''
        Match the topics identified for the article with the DBpedia
        abstract.
        '''
        etf = [f for f in self.features if f.startswith('entity_topic')]
        ctf = [f for f in self.features if f.startswith('candidate_topic')]
        mtf = [f for f in self.features if f.startswith('match_txt_topic')]
        if not (etf or ctf or mtf):
            return

        if not hasattr(self.cluster.context, 'topics'):
            self.cluster.context.get_topics()
        topics = self.cluster.context.topics

        if topics and etf:
            for t in topics:
                setattr(self, 'entity_topic_' + t, topics[t])

        if ctf or mtf:

            description_topics = {t: 0.0 for t in dictionary.topics}

            # Deduce topic(s) from role(s)
            dbo_types = []
            if self.document.get('dbo_type'):
                dbo_types += self.document.get('dbo_type')

            if dbo_types:
                for t in dbo_types:
                    for r in dictionary.roles_dbo:
                        if (t in dictionary.roles_dbo[r] and r.split('_')[0] in
                                description_topics):
                            description_topics[r.split('_')[0]] = 1.0

            # Predict topic(s) from abstract
            if not sum(description_topics.values()):
                if not hasattr(self, 'topic_probs'):
                    self.get_topics()
                description_topics = self.topic_probs

            if ctf:
                for t in description_topics:
                    setattr(self, 'candidate_topic_' + t,
                            description_topics[t])

            if mtf:
                self.match_txt_topic = (cosine_similarity(
                    [topics.values()], [description_topics.values()])[0][0]
                                        - 0.25)

    def set_vector_match(self):
        '''
        Match context word vectors with abstract word vectors.
        '''
        if not self.document.get('lang') == 'nl':
            return

        evf = [f for f in self.features if f.startswith('entity_vec')]
        mvf = [f for f in self.features if f.startswith('match_txt_vec')]
        if not (evf or mvf):
            return

        if not hasattr(self.cluster, 'window'):
            self.cluster.get_window()
        if not self.cluster.window:
            return

        if not hasattr(self.cluster, 'window_vectors'):
            self.cluster.window_vectors = self.get_vectors(self.cluster.window)
        if not self.cluster.window_vectors:
            return

        if evf:
            # Take mean of window vectors for now, need to find better
            # representation
            window_vectors_array = np.array(self.cluster.window_vectors)
            entity_vector = np.mean(window_vectors_array, axis=0).tolist()
            for i, v in enumerate(entity_vector):
                setattr(self, 'entity_vec_' + str(i), v)

        if not mvf:
            return

        if not hasattr(self, 'abstract_bow'):
            self.tokenize_abstract()
        bow = [w for w in self.abstract_bow[:25] if len(w) > 4 and w not in
               self.cluster.entity_parts and w not in dictionary.unwanted]
        if self.document.get('keyword'):
            bow += [w for w in self.document.get('keyword') if len(w) > 4 and w
                    not in self.cluster.entity_parts and w not in
                    dictionary.unwanted]
        if not bow:
            return

        cand_vectors = self.get_vectors(bow)
        if not cand_vectors:
            return

        sims = cosine_similarity(np.array(self.cluster.window_vectors),
                                 np.array(cand_vectors))

        self.match_txt_vec_max = sims.max() - 0.375
        self.match_txt_vec_mean = sims.mean() - 0.0625

    def set_entity_match(self):
        '''
        Match other entities appearing in the article with DBpedia abstract.
        '''
        if 'match_txt_entities' not in self.features:
            return

        if not hasattr(self.cluster, 'entity_parts'):
            self.cluster.get_entity_parts()
        if not hasattr(self.cluster, 'context_entity_parts'):
            self.cluster.get_context_entity_parts()
        if not self.cluster.context_entity_parts:
            return

        if not hasattr(self, 'abstract_bow'):
            self.tokenize_abstract()
        bow = [t for t in self.abstract_bow if len(t) > 4]

        entity_match = len(set(self.cluster.context_entity_parts) & set(bow))
        self.match_txt_entities = math.tanh(entity_match * 0.25)

    def set_entity_match_newspapers(self):
        '''
        Get number of newspaper articles where candidate pref label appears
        together with other entity mentions in the article.
        '''
        if 'match_txt_entities_newspapers' not in self.features:
            return

        # Candidate has to be person
        if not self.document.get('last_part'):
            return

        # Candidate pref label can't be ambiguous
        if self.document.get('ambig') == 1:
            return

        # Candidate pref label has to appear in newspapers on its own
        if not self.document.get('inlinks_newspapers'):
            return

        # Normalized entity has to differ from pref label
        pref_label = self.document.get('pref_label')
        if self.match_str_pref_label_exact:
            return
        # But partly match or last part match
        if not (self.match_str_pref_label_end or self.match_str_pref_label or
                self.match_str_last_part):
            return

        # Other entity mentions have to be available from context
        context_entities = [e.norm for e in self.cluster.context.entities
                            if e.norm.find(self.cluster.entities[0].norm) == -1
                            and self.cluster.entities[0].norm.find(e.norm) ==
                            -1 and e.norm.find(pref_label) == -1 and
                            pref_label.find(e.norm) == -1 and len(e.norm) >= 5]
        context_entities = list(set(context_entities))

        if not context_entities:
            return

        # Query for co-occurence
        query = '"' + pref_label + '" AND ('
        for i, e in enumerate(context_entities):
            if i > 0:
                query += ' OR '
            query += '"' + e + '"'
        query += ')'

        payload = {}
        payload['operation'] = 'searchRetrieve'
        payload['x-collection'] = 'DDD_artikel'
        payload['maximumRecords'] = 0
        payload['query'] = query

        try:
            response = requests.get(JSRU_URL, params=payload, timeout=60)
            xml = etree.fromstring(response.content)
            tag = '{http://www.loc.gov/zing/srw/}numberOfRecords'
            num_records = int(xml.find(tag).text)
            self.match_txt_entities_newspapers = (
                num_records / float(self.document.get('inlinks_newspapers')))
        except Exception:
            return

    def set_entity_vector_match(self):
        '''
        Match word vectors for other entities in the article with
        entity vector.
        '''
        cvf = [f for f in self.features if f.startswith('candidate_vec')]
        mvf = [f for f in self.features if
               f.startswith('match_txt_entity_vec')]
        if not (cvf or mvf):
            return

        if not self.document.get('uri_wd'):
            return

        wd_id = self.document.get('uri_wd').split('/')[-1]

        cand_vectors = self.get_vectors([wd_id])
        if not cand_vectors:
            return

        if cvf:
            for i, v in enumerate(cand_vectors[0]):
                setattr(self, 'candidate_vec_' + str(i), v)

        if not mvf:
            return

        if not hasattr(self.cluster, 'context_entity_parts'):
            self.cluster.get_context_entity_parts()
        if not self.cluster.context_entity_parts:
            return

        if not hasattr(self.cluster, 'context_entity_vectors'):
            self.cluster.context_entity_vectors = self.get_vectors(
                self.cluster.context_entity_parts)
        if not self.cluster.context_entity_vectors:
            return

        sims = cosine_similarity(np.array(self.cluster.context_entity_vectors),
                                 np.array(cand_vectors))
        self.match_txt_entity_vec_max = sims.max() - 0.375
        self.match_txt_entity_vec_mean = sims.mean() - 0.125

    def tokenize_abstract(self):
        '''
        Tokenize and normalize DBpedia abstract.
        '''
        self.abstract_bow = utilities.tokenize(self.document.get('abstract'),
                                               unique=True)

    def get_vectors(self, wordlist):
        '''
        Get word vectors for given word list.
        '''
        payload = {'source': ' '.join(wordlist)}
        response = requests.get(W2V_URL, params=payload, timeout=300)
        assert response.status_code == 200, 'Error retrieving word vectors'
        data = response.json()
        return data['vectors']

    def get_topics(self):
        '''
        Get topics and type from classifier service.
        '''
        payload = {'url': self.document.get('id')}
        response = requests.get(TOPICS_URL, params=payload,
                                timeout=300)
        assert response.status_code == 200, 'Error retrieving topics'
        self.topic_probs = response.json()['topics']
        self.type_probs = response.json()['types']


class Result(object):
    '''
    The link result for an entity cluster.
    '''

    def __init__(self, reason, prob=0.0, description=None, cand_list=None):
        '''
        Set the result attributes.
        '''
        self.reason = reason
        self.prob = prob
        self.description = description

        self.link = None
        self.label = None
        self.features = None
        self.candidates = None

        if description:
            self.features = {}
            for f in description.features:
                self.features[f] = float(getattr(description, f))
            if self.reason == 'Predicted link':
                self.link = description.document.get('id')
                self.label = description.document.get('label')

        if cand_list:
            self.candidates = []
            for description in cand_list.candidates:
                d = {}
                d['id'] = description.document.get('id')
                d['prob'] = description.prob
                d['features'] = {}
                for f in description.features:
                    d['features'][f] = float(getattr(description, f))
                d['document'] = description.document
                self.candidates.append(d)

    def get_dict(self, features=False, candidates=False):
        '''
        Return the result dictionary.
        '''
        result = {}
        result['reason'] = self.reason
        if self.prob:
            result['prob'] = '{0:.10f}'.format(self.prob)
        if self.link:
            result['link'] = self.link
        if self.label:
            result['label'] = self.label
        if features and self.features:
            result['features'] = self.features
        if candidates and self.candidates:
            result['candidates'] = self.candidates
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    SAMPLE_URL = 'http://resolver.kb.nl/resolve'
    SAMPLE_URL += '?urn=ddd:010734861:mpeg21:a0002:ocr'

    parser.add_argument('--url', required=False, type=str,
                        default=SAMPLE_URL,
                        help='resolver link of the article to be processed')
    parser.add_argument('--ne', required=False, type=str, default='',
                        help='specific named entity to be linked')
    parser.add_argument('-m', '--model', required=False, type=str,
                        default='nn', help='model used for link prediction')
    parser.add_argument('-d', '--debug', required=False, action='store_true',
                        help='include unlinked entities in response')
    parser.add_argument('-f', '--features', required=False,
                        action='store_true', help='return feature values')
    parser.add_argument('-c', '--candidates', required=False,
                        action='store_true', help='return candidate list')
    parser.add_argument('-e', '--errh', required=False, action='store_true',
                        help='turn on error handling')

    args = parser.parse_args()

    linker = EntityLinker(model=vars(args)['model'],
                          debug=vars(args)['debug'],
                          features=vars(args)['features'],
                          candidates=vars(args)['candidates'],
                          error_handling=vars(args)['errh'])

    pprint(linker.link(vars(args)['url'], vars(args)['ne']))
