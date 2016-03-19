#!/usr/bin/env python
# -*- coding: utf-8 -*-

import Levenshtein
import models
import solr
import sys
import urllib
import utilities

from lxml import etree
from operator import attrgetter


class EntityLinker():

    TPTA_URL = 'http://145.100.59.224:8080/tpta/analyse?lang=nl&url='
    SOLR_URL = 'http://linksolr.kbresearch.nl/dbpedia/'

    SOLR_ROWS = 20
    MIN_PROB = 0.5

    debug = None
    model = None
    solr_connection = None

    url = None
    ne = None
    context = None

    linked = []


    def __init__(self, debug=None):
        self.debug = debug
        self.model = models.RadialSVM()
        self.solr_connection = solr.SolrConnection(self.SOLR_URL)


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
            if dependencies and (not result.description or (result.description.document.get('schemaorgtype')
                    and 'Person' not in result.description.document.get('schemaorgtype'))):
                new_clusters = [Cluster([e for e in cluster.entities if e not in dependencies])]
                new_clusters.extend(self.get_clusters(dependencies))
                if self.ne:
                    clusters_to_link.extend([c for c in new_clusters if entity_to_link in c.entities])
                else:
                    clusters_to_link.extend(new_clusters)
            else:
                linked.append(cluster)
        self.linked = linked

        if self.debug:
            for cluster in linked:
                print '\n'
                print [e.text for e in cluster.entities]
                if cluster.descriptions:
                    for description in cluster.descriptions:
                        print description.document.get('id')
                        print description.prob
                        if entity_to_link:
                            for j in range(len(self.model.features)):
                                print self.model.features[j], getattr(description, self.model.features[j])
                            print '\n'

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


    def __init__(self, url):
        self.url = url
        self.ocr = self.get_ocr(self.url)
        self.publ_date, self.publ_place = self.get_metadata(self.url)


    def get_ocr(self, url):
        data = urllib.urlopen(url).read()
        xml = etree.fromstring(data)
        return etree.tostring(xml, encoding='utf8',
                method='text').decode('utf-8')


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


class Entity():

    text = None
    tpta_type = None
    context = None
    doc_pos = None

    start_pos = None
    end_pos = None
    window = None
    quotes = None

    clean = None
    norm = None
    word_length = None
    last_part = None

    valid = None


    def __init__(self, text, tpta_type, context, doc_pos=0):
        self.text = text
        self.tpta_type = tpta_type
        self.context = context
        self.doc_pos = doc_pos

        # Get position in text, window, quotes
        self.start_pos, self.end_pos = self.get_position(self.context.document.ocr,
                self.text, self.doc_pos)
        self.window = self.get_window(self.context.document.ocr,
                start_pos=self.start_pos, end_pos=self.end_pos, size=10)
        self.quotes = self.get_quotes(self.start_pos, self.end_pos)

        # Clean and normalize input text
        self.clean = utilities.clean(self.text)
        self.norm = utilities.normalize(self.clean)
        self.word_length = len(self.norm.split())
        self.last_part = self.get_last_part(self.norm)

        # Check and set validity
        self.valid = self.is_valid()


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
                left_bow = utilities.tokenize(document[:left_space_pos])
            right_space_pos = document.find(' ', end_pos)
            if right_space_pos > 0:
                right_bow = utilities.tokenize(document[right_space_pos:])

        if size:
            left_bow = left_bow[-size:]
            right_bow = right_bow[:size]

        if direction == 'left':
            return left_bow
        elif direction == 'right':
            return right_bow
        return left_bow + right_bow


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
        elif len(self.norm) > 2 and self.last_part:
            return True
        else:
            return False


class Cluster():

    entities = None
    result = None

    solr_rows = None
    solr_response = None
    solr_result_count = None
    descriptions = None

    quotes_total = None
    inlinks_total = None
    max_score = None


    def __init__(self, entities):
        self.entities = entities


    def resolve(self, solr_connection, solr_rows, model, min_prob):

        # Check validity of the representative entity
        if not self.entities[0].is_valid():
            self.result = Result("Invalid entity")
            return self.result

        # If entity is valid, query Solr for candidate DBpedia descriptions
        try:
            self.solr_response, self.solr_result_count = self.query_solr(solr_connection, solr_rows)
        except Exception as error_msg:
            self.result = Result("Failed to query solr: " + str(error_msg), -1.0)
            return self.result
        if self.solr_response is not None and self.solr_response.numFound == 0:
            self.result = Result("Nothing found")
            return self.result

        # If any descriptions were found, initialize list of candidate descriptions
        descriptions = []
        for i in range(self.solr_result_count):
            description = Description(self.solr_response.results[i], i + 1, self)
            descriptions.append(description)
        self.descriptions = descriptions
        
        # Filter candidates according to hard criteria, e.g. name conlfict
        candidates = []
        for description in self.descriptions:
            description.calculate_rule_features()
            if description.name_conflict == 0:
                candidates.append(description)
        if len(candidates) == 0:
            self.result = Result("Name conflict")
            return self.result

        # If any candidates remain, calculate their feature values and probability
        self.quotes_total = self.get_total_quotes()
        self.inlinks_total = self.get_total_inlinks()
        self.max_score = self.get_max_score()
        self.solr_rows = solr_rows

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
            self.result = Result("SVM classifier best probability", best_match.prob, best_match)
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

        query = 'title:"' + self.entities[0].norm + '"'
        query += ' OR title_str:"' + self.entities[0].clean + '"'
        query += ' OR lastpart_str:"' + last_part + '"'

        solr_response = solr_connection.query(
	        q=query, rows=solr_rows, indent='on',
	        sort='lang,inlinks', sort_order='desc')
        numfound = solr_response.numFound
        solr_result_count = numfound if numfound <= solr_rows else solr_rows

        return solr_response, solr_result_count


    def get_total_inlinks(self):
        inlinks_total = 0
        for d in self.descriptions:
            inlinks_total += d.document.get('inlinks')
        return inlinks_total


    def get_max_score(self):
        max_score = 0
        for d in self.descriptions:
            if d.document.get('score') > max_score:
                max_score = d.document.get('score')
        return max_score


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

    description = None


    def __init__(self, reason, prob=0, description=None):
        self.reason = reason
        self.prob = prob
        if description:
            self.description = description
            self.link = description.document.get('id')[1:-1]
            self.label = description.document.get('title')[0]


    def get_dict(self):
        result = {}
        result['link'] = self.link
        result['label'] = self.label
        result['prob'] = self.prob
        result['reason'] = self.reason
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
    last_part_match = 0
    name_conflict = 0

    quotes = 0

    solr_pos = 0
    solr_score = 0
    inlinks = 0
    lang = 0
    disambig = 0

    mean_levenshtein_ratio = 0

    date_match = 0
    type_match = 0
    entity_match = 0

    # Remove
    cos_sim = 0

    prob = 0


    def __init__(self, document, position, cluster):
        self.document = document
        self.position = position
        self.cluster = cluster
        self.labels = self.get_labels()


    def get_labels(self):
        # Normalize titles here until they become available from the index
        labels = []
        for t in self.document.get('title_str'):
            norm = utilities.normalize(t)
            if len(norm) > 0:
                labels.append(norm)
        return labels


    def calculate_rule_features(self):
        self.match_id()
        self.match_titles()
        self.match_titles_last_part()

        features = ['main_title_exact_match', 'main_title_start_match',
                'main_title_end_match', 'title_exact_match', 'title_start_match',
                'title_end_match', 'last_part_match']

        name_conflict = 1
        for f in features:
            if getattr(self, f) > 0:
                name_conflict = 0
                break
        self.name_conflict = name_conflict


    def calculate_prob_features(self):
        self.quotes = self.cluster.quotes_total
        self.solr_pos = self.position / float(self.cluster.solr_rows)
        if self.cluster.max_score > 0:
            self.solr_score = self.document.get('score') / float(self.cluster.max_score)
        if self.cluster.inlinks_total > 0:
            self.inlinks = self.document.get('inlinks') / float(self.cluster.inlinks_total)
        self.lang = 1 if self.document.get('lang') == 'nl' else 0
        self.disambig = 1 if self.document.get('disambig') == 1 else 0

        self.match_titles_levenshtein()
        self.match_date()
        self.match_type()
        self.match_entities()


    def match_id(self):
        # Use normalized title string list until they are available from the index
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

        # Use normalized title string list until they are available from the index
        match_label = self.labels[1:]
        ne = self.cluster.entities[0].norm

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
        self.non_matching_labels += non_matching_labels


    def match_titles_last_part(self):
        ne = self.cluster.entities[0].norm

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
            if year_of_birth is not None:
                if year_of_publ < year_of_birth:
                    self.date_match = -1
                else:
                    self.date_match = 1


    def match_type(self):
        mapping = {'person': 'Person', 'location': 'Place', 'organisation': 'Organization'}
        schema_types = self.document.get('schemaorgtype')
        type_match = 0

        if schema_types:
            entity_types = [e.tpta_type for e in self.cluster.entities if e.tpta_type and e.tpta_type in mapping]
            for entity_type in entity_types:
                for t in schema_types:
                    if t == mapping[entity_type]:
                        type_match += 1
                        break
        if type_match:    
            self.type_match = type_match / len(entity_types)


    def match_entities(self):
        entity_match = 0
        excluded_entities = ['Nederland', 'Nederlandse', 'Amsterdam', 'Amsterdamse']
        abstract = self.document.get('abstract')
        if abstract:
            entity_list = [e.clean for e in self.cluster.entities[0].context.entities
                    if e.valid and self.cluster.entities[0].clean.find(e.clean) == -1 and e.clean not in excluded_entities]
            for entity in entity_list:
                if abstract.find(entity) > -1:
                    entity_match += 1
        self.entity_match = entity_match


if __name__ == '__main__':
    if not len(sys.argv) > 1:
        print("Usage: ./disambiguation.py [url (string)]")
    else:
        linker = EntityLinker(debug=True)
        if len(sys.argv) > 2:
            print(linker.link(sys.argv[1], sys.argv[2]))
        else:
            print(linker.link(sys.argv[1]))

