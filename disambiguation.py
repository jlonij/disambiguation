#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dictionary
import Levenshtein
import math
import models
import re
import solr
import sparql
import sys
import urllib
import utilities

from lxml import etree
from operator import attrgetter


class EntityLinker():

    TPTA_URL = 'http://tpta.kbresearch.nl/analyse?lang=nl&url='
    #TPTA_URL = 'http://192.87.165.3:8080/tpta2/analyse?lang=nl&url='
    #TPTA_URL = 'http://tomcat.kbresearch.nl/tpta/analyse?lang=nl&url='
    #TPTA_URL = 'http://145.100.59.224:8080/tpta/analyse?lang=nl&url='
    #TPTA_URL = 'http://192.87.165.5:8080/tpta/analyse?lang=nl&url='

    SOLR_URL = 'http://linksolr.kbresearch.nl/dbpedia'
    #SOLR_URL = 'http://linksolr1.kbresearch.nl/dbpedia'
    #SOLR_URL = 'http://145.100.59.224:8081/solr/dbpedia'
    #SOLR_URL = 'http://192.87.165.5:8081/solr/dbpedia'

    SOLR_ROWS = 20
    MIN_PROB = 0.5

    debug = None
    model = None
    solr_connection = None

    url = None
    ne = None
    context = None

    linked = []


    def __init__(self, debug=None, model=None):

        self.debug = debug

        # Init Solr connection
        self.solr_connection = solr.SolrConnection(self.SOLR_URL)

        # Init prediction model
        if model == 'svm':
            self.model = models.LinearSVM()
        elif model == 'nn':
            self.model = models.NeuralNet()
        else:
            self.model = models.LinearSVM()


    def link(self, url, ne=None):
        self.url = url
        self.context = Context(self.url, self.TPTA_URL)
        self.ne = ne.decode('utf-8') if ne else None

        # If a specific ne was requested, search for a corresponding known entity
        if self.ne:
            entity_to_link = None
            for entity in self.context.entities:
                if self.ne == entity.text:
                    entity_to_link = entity
            # If not found, create new one
            if not entity_to_link:
                entity_to_link = Entity(self.ne, None, self.context)
                self.context.entities.append(entity_to_link)

        # Group related entities into clusters and try to link relevant clusters
        clusters = self.get_clusters(self.context.entities)
        clusters_to_link = [c for c in clusters if entity_to_link in c.entities] if self.ne else clusters

        linked = []
        while clusters_to_link:
            cluster = clusters_to_link.pop()
            result = cluster.resolve(self.solr_connection, self.SOLR_ROWS, self.model, self.MIN_PROB)
            dependencies = [e for e in cluster.entities if e.norm != cluster.entities[0].norm]
            # If cluster consists of multiple entities and could not be linked
            # or is not linked to a person split it up and return the parts to
            # the queue
            if dependencies and (not result.description or (result.description.document.get('schemaorgtype')
                    and 'Person' not in result.description.document.get('schemaorgtype'))):
                new_clusters = [Cluster([e for e in cluster.entities if e not in dependencies])]
                new_clusters.extend(self.get_clusters(dependencies))
                # If linking a specific ne, only return the new cluster containing
                # that ne to the queue
                if self.ne:
                    clusters_to_link.extend([c for c in new_clusters if entity_to_link in c.entities])
                else:
                    clusters_to_link.extend(new_clusters)
            else:
                linked.append(cluster)
        self.linked = linked

        if self.debug:
            for cluster in linked:
                if cluster.descriptions:
                    for description in cluster.descriptions:
                        print description.document.get('id').encode('utf-8')
                        print description.prob
                        if self.ne:
                            for j in range(len(self.model.features)):
                                print self.model.features[j], getattr(description, self.model.features[j])

        # Return the result for each enitity
        results = []
        to_return = [entity_to_link] if self.ne else self.context.entities
        for entity in to_return:
            if entity.text not in [result['text'] for result in results]:
                for cluster in linked:
                    if entity in cluster.entities:
                        result = cluster.result.get_dict()
                        result['text'] = entity.text
                        results.append(result)
        return results


    def get_clusters(self, entities):
        clusters = []
        sorted_entities = sorted(entities, key=attrgetter('norm'), reverse=True)
        sorted_entities = sorted(sorted_entities, key=attrgetter('word_length'), reverse=True)
        for entity in sorted_entities:
            clusters = self.cluster(entity, clusters)
        return clusters


    def cluster(self, entity, clusters):
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

    url = None
    tpta_url = None
    document = None
    entities = None


    def __init__(self, url, tpta_url):
        self.url = url
        self.tpta_url = tpta_url
        self.document = Document(self.url)
        self.entities = self.get_entities(self.url, self.tpta_url)


    def get_entities(self, url, tpta_url):
        entities = []
        try:
            data = urllib.urlopen(tpta_url + url).read()
            xml = etree.fromstring(data)
        except:
            return entities
        doc_pos = 0
        for node in xml.iter():
            if node.text and len(node.text) > 1:
                unicode_text = node.text if isinstance(node.text, unicode) else node.text.decode('utf-8')
                entity = Entity(unicode_text, node.tag, self, doc_pos)
                doc_pos = entity.end_pos if entity.end_pos > -1 else doc_pos
                entities.append(entity)
        return entities


class Document():

    url = None
    ocr = None
    publ_date = None
    publ_place = None

    subjects = []


    def __init__(self, url):
        self.url = url
        self.ocr = self.get_ocr(self.url)
        self.publ_date, self.publ_place = self.get_metadata(self.url)
        self.subjects = self.get_subjects()


    def get_ocr(self, url):
        data = urllib.urlopen(url).read()
        xml = etree.fromstring(data)
        return etree.tostring(xml, encoding='utf8',
                method='text').decode('utf-8')


    def get_metadata(self, url):
        publ_date, publ_place = None, None
        jsru_url = 'http://jsru.kb.nl/sru/sru?'
        jsru_url += 'operation=searchRetrieve&x-collection=DDD_artikel'
        jsru_url += '&query=uniqueKey=' + url[url.find('ddd:'):-4]
        try:
            data = urllib.urlopen(jsru_url).read()
            xml = etree.fromstring(data)
        except:
            return None, None

        path = '{http://www.loc.gov/zing/srw/}records/'
        path += '{http://www.loc.gov/zing/srw/}record/'
        path += '{http://www.loc.gov/zing/srw/}recordData/'
        path += '{http://purl.org/dc/elements/1.1/}date'

        date = xml.find(path)
        if date is not None:
            publ_date = date.text
        return publ_date, publ_place


    def get_subjects(self):
        subjects = []
        for subject in dictionary.subjects:
            words = dictionary.subjects[subject]
            for role in dictionary.roles:
                if subject in dictionary.roles[role]['subjects']:
                    words += dictionary.roles[role]['words']
            window = [utilities.normalize(w) for w in utilities.tokenize(self.ocr)]
            if len(set(words) & set(window)) > 0:
                subjects.append(subject)
        return subjects


class Entity():

    text = None
    tpta_type = None
    context = None
    doc_pos = None

    start_pos = None
    end_pos = None
    window_left = None
    window_right = None
    quotes = None

    clean = None
    norm = None
    word_length = None
    last_part = None

    valid = None

    title = None
    title_form = None
    gender = None
    gender_form = None
    role = None
    role_form = None

    alt_type = None


    def __init__(self, text, tpta_type, context, doc_pos=0):
        self.text = text
        self.tpta_type = tpta_type
        self.context = context
        self.doc_pos = doc_pos

        # Get position in text, window, quotes
        self.start_pos, self.end_pos = self.get_position(self.context.document.ocr,
                self.text, self.doc_pos)
        self.window_left, self.window_right = self.get_window(self.context.document.ocr,
                start_pos=self.start_pos, end_pos=self.end_pos, size=30)
        self.quotes = self.get_quotes(self.start_pos, self.end_pos)

        # Clean and normalize input text
        self.clean = utilities.clean(self.text)
        self.norm = utilities.normalize(self.clean)
        self.word_length = len(self.norm.split())
        self.last_part = self.get_last_part(self.norm)

        # Check and set validity
        self.valid = self.is_valid()

        # Get title, gender, role, check tpta_type
        if self.valid:
            self.title, self.title_form = self.get_title()
            self.gender, self.gender_form = self.get_gender()
            self.role, self.role_form = self.get_role()
            self.alt_type = self.get_alt_type()


    def get_position(self, document, phrase, doc_pos=None):
        start_pos = document.find(phrase, doc_pos)
        end_pos = start_pos + len(phrase)
        if start_pos > 0 and end_pos <= len(document):
            return start_pos, end_pos
        else:
            return -1, -1


    def get_window(self, document, phrase=None, start_pos=None, end_pos=None, size=None):
        left_bow = []
        right_bow = []

        if not start_pos or not end_pos:
            start_pos = document.find(phrase)
            end_pos = start_pos + len(phrase)

        if start_pos > 0 and end_pos <= len(document):
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


    def get_quotes(self, start_pos, end_pos):
        quotes = 0
        quote_chars = [u'"', u"'", u'„', u'”', u'‚', u'’']
        for pos in [start_pos - 1, end_pos]:
            if self.context.document.ocr[pos] in quote_chars:
                quotes += 1
        return quotes


    def get_last_part(self, ne):
        ne_parts = ne.split()
        last_part = None
        for part in reversed(ne_parts):
            if not part.isdigit():
                last_part = part
                break
        return last_part


    def is_valid(self):
        if self.valid is not None:
            return self.valid
        elif len(self.norm) > 2 and self.last_part and not self.is_date():
            return True
        return False


    def is_date(self):
        if len(self.norm.split()) > 1:
            if len([w for w in self.norm.split() if w in dictionary.months]):
                if len([w for w in self.norm.split() if w.isdigit()]):
                    return True
        return False


    def get_title(self):
        words = [self.norm.split()[0]]
        if self.window_left:
            words.append(utilities.normalize(self.window_left[-1]))
        for word in words:
            if word in dictionary.titles:
                return True, word
        return None, None


    def get_gender(self):
        words = [self.norm.split()[0]]
        if self.window_left:
            words.append(utilities.normalize(self.window_left[-1]))
        for word in words:
            for gender in dictionary.genders:
                if word in dictionary.genders[gender]:
                    return gender, word
        return None, None


    def get_role(self):
        words = [self.norm.split()[0]]
        if self.window_left:
            words.append(utilities.normalize(self.window_left[-1]))
        if self.window_right and self.context.document.ocr[self.end_pos] == ',':
            words.append(utilities.normalize(self.window_right[0]))
        for word in words:
            for role in dictionary.roles:
                if word in dictionary.roles[role]['words']:
                    return role, word
        return None, None


    def get_alt_type(self):

        if self.gender or self.title:
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

    entities = None
    result = None

    solr_rows = None
    model = None

    solr_iteration = None
    solr_result_count = None
    solr_response = None

    descriptions = None
    candidates = None

    solr_max_score = None
    cand_max_score = None
    solr_inlinks_total = None
    cand_inlinks_total = None
    titles_total = None
    quotes_total = None


    def __init__(self, entities):
        self.entities = entities


    def resolve(self, solr_connection, solr_rows, model, min_prob):
        self.solr_rows = solr_rows
        self.model = model

        # Check validity of the representative entity
        if not self.entities[0].is_valid():
            self.result = Result("Invalid entity")
            return self.result

        # If entity is valid, query Solr for candidate DBpedia descriptions
        try:
            self.solr_iteration, self.solr_response, self.solr_result_count = self.query_solr(solr_connection, solr_rows)
        except Exception as error_msg:
            self.result = Result("Failed to query solr: " + str(error_msg), -1.0)
            return self.result

        # If nothing found, return
        if self.solr_result_count == 0:
            self.result = Result("Nothing found")
            return self.result

        # If any descriptions were found, initialize list of candidate descriptions
        descriptions = []
        for i in range(self.solr_result_count):
            description = Description(self.solr_response.results[i], i, self)
            descriptions.append(description)
        self.descriptions = descriptions

        # Filter candidates according to hard criteria, e.g. name conlfict
        candidates = []
        for description in self.descriptions:
            description.calculate_rule_features()
            if description.name_conflict == 0 and description.date_match > -1:
                candidates.append(description)
        self.candidates = candidates

        if len(self.candidates) == 0:
            self.result = Result("Name or date conflict")
            return self.result

        # If any candidates remain, calculate their feature values and probability
        self.solr_max_score, self.cand_max_score = self.get_max_score()
        self.solr_inlinks_total, self.cand_inlinks_total = self.get_total_inlinks()
        self.titles_total = self.get_total_titles()
        self.quotes_total = self.get_total_quotes()

        best_match = candidates[0]
        for description in candidates:
            description.calculate_prob_features()
            example = []
            for j in range(len(model.features)):
                example.append(float(getattr(description, model.features[j])))
            description.prob = model.predict(example)
            if description.prob > best_match.prob:
                best_match = description

        if best_match.prob >= min_prob:
            self.result = Result("Predicted link", best_match.prob, best_match)
        else:
            self.result= Result("Probability too low for: " + best_match.document.get('title')[0], best_match.prob)
        return self.result


    def query_solr(self, solr_connection, solr_rows):

        # Temporary until normalization in index
        ne_parts = self.entities[0].clean.split()
        last_part = None
        for part in reversed(ne_parts):
            if not part.isdigit():
                last_part = part
                break

        queries = []

        query = 'title:"' + self.entities[0].norm + '"'
        query += ' OR title_str:"' + self.entities[0].clean + '"'
        query += ' OR lastpart_str:"' + self.entities[0].last_part.capitalize() + '"'

        queries.append(query)

        '''
        if self.entities[0].last_part == self.entities[0].norm.split()[0]:
            # Query #1: exact match
            query = 'title_str:"' + self.entities[0].clean + '"'
            query += ' OR lastpart_str:"' + last_part + '"'
            query += ' OR lastpart_str:"' + self.entities[0].last_part.capitalize() + '"'
            queries.append(query)

            # Query #2: normalized match
            query = 'title:"' + self.entities[0].norm + '"'
            queries.append(query)
        else:
            # Query #1: exact match
            query = 'title:"' + self.entities[0].norm + '"'
            query += ' OR title_str:"' + self.entities[0].clean + '"'
            queries.append(query)

            # Query #2: last part match
            query = 'lastpart_str:"' + last_part + '"'
            query += ' OR lastpart_str:"' + self.entities[0].last_part.capitalize() + '"'
            queries.append(query)
        '''

        # Query #3: stem match
        # Query #4: fuzzy match

        solr_iteration = None
        solr_response = None
        solr_result_count = None

        for i, query in enumerate(queries):
            solr_iteration = i
            solr_response = solr_connection.query(
	            q=query, rows=solr_rows, indent='on',
	            sort='lang,inlinks', sort_order='desc')
            solr_result_count = (solr_response.numFound if
                    solr_response.numFound <= solr_rows else solr_rows)
            if solr_result_count > 0:
                break

        return solr_iteration, solr_response, solr_result_count


    def get_max_score(self):
        solr_max_score = 0
        for d in self.descriptions:
            if d.document.get('score') > solr_max_score:
                solr_max_score = d.document.get('score')
        cand_max_score = 0
        for d in self.candidates:
            if d.document.get('score') > cand_max_score:
                cand_max_score = d.document.get('score')
        return solr_max_score, cand_max_score


    def get_total_inlinks(self):
        solr_inlinks_total = 0
        for d in self.descriptions:
            solr_inlinks_total += d.document.get('inlinks')
        cand_inlinks_total = 0
        for d in self.candidates:
            cand_inlinks_total += d.document.get('inlinks')
        return solr_inlinks_total, cand_inlinks_total


    def get_total_titles(self):
        titles_total = 0
        for d in self.candidates:
            titles_total += len(d.labels)
        return titles_total


    def get_total_quotes(self):
        total_quotes = 0
        for e in self.entities:
            total_quotes += e.quotes
        return total_quotes


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
            self.link = description.document.get('id')[1:-1]
            self.label = description.document.get('title')[0]

            features = {}
            for j in range(len(description.cluster.model.features)):
                features[description.cluster.model.features[j]] = float(getattr(description,
                        description.cluster.model.features[j]))
            self.features = features


    def get_dict(self):
        result = {}
        result['link'] = self.link
        result['label'] = self.label
        result['prob'] = self.prob
        result['reason'] = self.reason
        result['features'] = self.features
        return result


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


    def __init__(self, document, position, cluster):
        self.document = document
        self.position = position
        self.cluster = cluster
        self.labels = self.get_labels()
        self.add_sparql_results()


    def get_labels(self):
        labels = []
        for t in self.document.get('title_str'):
            if t.find(',') >= 0 and self.document.get('title_str')[0].find(',') < 0:
                continue
            # Normalize titles here until they become available from the index
            norm = utilities.normalize(utilities.clean(t))
            # Remove emtpy labels
            if len(norm) > 0 and norm not in labels:
                labels.append(norm)
        return labels


    def add_sparql_results(self):
        types, categories, yob = sparql.query_sparql(self.document.get('id'))
        if types:
            if 'schemaorgtype' in self.document:
                self.document['schemaorgtype'] = list(set(types + self.document['schemaorgtype']))
            else:
                self.document['schemaorgtype'] = types
        if categories:
            self.document['categories'] = categories
        if yob:
            self.document['yob'] = yob


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
        self.solr_iteration = self.cluster.solr_iteration

        # Position
        self.solr_pos = (self.position + 1) / float(self.cluster.solr_rows)
        self.cand_pos = (self.cluster.candidates.index(self) + 1) / float(self.cluster.solr_rows)
        # Score
        if self.cluster.solr_max_score > 0:
            self.solr_score = self.document.get('score') / float(self.cluster.solr_max_score)
        if self.cluster.cand_max_score > 0:
            self.cand_score = self.document.get('score') / float(self.cluster.cand_max_score)
        # Inlinks
        if self.cluster.solr_inlinks_total > 0:
            self.solr_inlinks = self.document.get('inlinks') / float(self.cluster.solr_inlinks_total)
        if self.cluster.cand_inlinks_total > 0:
            self.cand_inlinks = self.document.get('inlinks') / float(self.cluster.cand_inlinks_total)

        self.quotes = self.cluster.quotes_total
        self.lang = 1 if self.document.get('lang') == 'nl' else 0
        self.disambig = 1 if self.document.get('disambig') == 1 else 0

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
        ne = self.cluster.entities[0].norm
        if ((self.cluster.entities[0].gender and ne.split()[0] ==
            self.cluster.entities[0].gender_form) or (self.cluster.entities[0].role
            and ne.split()[0] == self.cluster.entities[0].role_form) or
            (self.cluster.entities[0].title and ne.split()[0] ==
                self.cluster.entities[0].title_form)):
            ne = ' '.join(ne.split()[1:])
            if not ne:
                return

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
        publ_date = self.cluster.entities[0].context.document.publ_date
        if publ_date:
            year_of_publ = int(publ_date[:4])
            year_of_birth = self.document.get('yob')
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

        schema_types = self.document.get('schemaorgtype')
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
        schema_types = self.document.get('schemaorgtype')
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
        subjects = self.cluster.entities[0].context.document.subjects
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
            entity_list = [e.clean for e in self.cluster.entities[0].context.entities
                    if e.valid and self.cluster.entities[0].clean.find(e.clean)
                    == -1 and e.clean not in excluded_entities]
            for entity in entity_list:
                if entity not in found_entities and abstract.find(entity) > -1:
                    found_entities.append(entity)
                    entity_match += 1
        entity_match = entity_match if entity_match < 3 else 3
        self.entity_match = entity_match


    def match_spec(self):
        if self.document.get('disambig') == 0 and self.document.get('lang') == 'nl':
            spec = self.document.get('id').split('(')[-1][:-2]
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
        categories = self.document.get('categories')
        if not categories:
            return
        ocr = self.cluster.entities[0].context.document.ocr
        tokens = utilities.tokenize(utilities.normalize(ocr))
        self.cat_match = len(set(categories) & set(tokens))
        #print self.cat_match


if __name__ == '__main__':
    if not len(sys.argv) > 1:
        print("Usage: ./disambiguation.py [url (string)]")
    else:
        linker = EntityLinker(debug=True)
        if len(sys.argv) > 2:
            print(linker.link(sys.argv[1], sys.argv[2]))
        else:
            print(linker.link(sys.argv[1]))

