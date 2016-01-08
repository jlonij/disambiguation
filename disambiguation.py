#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This code is work in progress,
# if you change anything, please re-run ./test_disambiguation.py
# And check the output.

    
import inspect
import Levenshtein
from lxml import etree
import numpy as np
import re
from scipy import spatial
import solr
import sys
import urllib
import warnings


class linkEntity():

    DEBUG = False

    # Todo: remove this option
    FAST = False # Quick decision or full evaluation

    FULL = False # All features of all candidates

    SOLR_SERVER = 'http://linksolr.kbresearch.nl/dbpedia/'
    SOLR_ROWS = 20
 
    query = ""
    solr_response = None
    solr_result_count = 0

    entity = None
    matches = []
    active_match_ids = []

    result = None
    flow = []


    def __init__(self, ne, ne_type='', url='', debug=False, full=True):

        if debug:
            self.DEBUG = debug
        if full:
            self.FULL = full
            
        # Normalize entity and get initial candidate set
        self.entity = Entity(ne, ne_type, url)
        ne = self.entity.ne
        if len(ne) < 2:
            reason = "Entity too short"
            self.result = None, -1.0, None, reason
        else:
            self.SOLR_CONNECTION = solr.SolrConnection(self.SOLR_SERVER)
            self.solr_response, self.solr_result_count = self.query_solr(ne)

        # If any candidate descriptions were found
        if not self.result:
            matches = []
            active_match_ids = []
            for i in range(self.solr_result_count):
                description = Description(self.solr_response.results[i])
                match = Match(self.entity, description, i)
                matches.append(match)
                active_match_ids.append(i)
            self.matches = matches
            self.active_match_ids = active_match_ids
            self.inlinks_nl, self.inlinks_en = self.get_total_inlinks()

            # Evaluate string similarity
            self.match_id()
            self.match_titles()
            self.match_last_part()

            remove_ids = []
            for i in self.active_match_ids:                  
                if self.matches[i].has_name_conflict():
                    remove_ids.append(i)
 
            # Eliminate name conflicts
            if not self.FULL:
                for r in remove_ids:
                    self.active_match_ids.remove(r)
            
            # If at least one candidate remains
            if len(self.active_match_ids) > 0:
                # Evaluate available date info
                remove_ids = []
                if self.entity.url:
                    self.entity.get_metadata()
                    for i in self.active_match_ids:
                        self.matches[i].match_date()
                        if self.matches[i].date_match < 0:
                            remove_ids.append(i)
                
                # Eliminate date conflicts
                if not self.FULL:
                    for r in remove_ids:
                        self.active_match_ids.remove(r)

            # If multiple candidates remain, or in full mode, evaluate other contextual features
            if len(self.active_match_ids) > 1 or self.FULL:
                # Match ne type
                if self.entity.ne_type:
                    for i in self.active_match_ids:
                        self.matches[i].match_type()
                # Match abstract
                if self.entity.url:
                    self.entity.get_ocr()
                    for i in self.active_match_ids:
                        self.matches[i].match_abstract()

            # Make prediction
            best_match_id = -1
            best_pred = -100

            for i in self.active_match_ids:
                match = self.matches[i]
                pred = (-1.407
                        + 0.149 * match.main_title_exact_match
                        + 0.137 * match.main_title_start_match
                        + 0.116 * match.main_title_end_match
                        - 0.966 * match.main_title_match
                        + 0.814 * match.title_exact_match
                        - 0.716 * match.title_start_match
                        + 0.887 * match.title_end_match
                        + 0.361 * match.title_match
                        + 0.315 * match.last_part_match
                        + 0.269 * match.name_conflict
                        + 0.757 * match.date_match
                        + 0.714 * match.type_match
                        - 0.194 * match.cos_sim)
                match.pred = pred
                if pred > best_pred:
                    best_pred = pred
                    best_match_id = i

            reason = "Linear SVM classifier"
            match = self.matches[best_match_id].description.document.get('id')
            label = self.matches[best_match_id].description.label
            prob = self.matches[best_match_id].pred
            self.result = match, prob, label, reason
            
            if self.DEBUG:
                for m in self.matches:
                    print m.description.document.get('id')
                    print m.pred

                    '''
                    print ('main_title_exact_match', m.main_title_exact_match)
                    print ('title_start_match', m.title_start_match)
                    print ('title_end_match', m.title_end_match)
                    print ('last_part_match', m.last_part_match)
                    print ('name_conflict', m.name_conflict)
                    print ('date_match', m.date_match)
                    print ('type_match', m.type_match)
                    print ('cos_sim', m.cos_sim)
                    '''
                print best_match_id
                print best_pred
                print self.active_match_ids
                print len(self.active_match_ids)
            
            if self.FAST: 
                if not self.result:
                    reason = 'Name conflict'
                    self.result = False, 0, False, reason


    def query_solr(self, ne):
        if self.DEBUG:
            self.flow.append(inspect.stack()[0][3])
            
        query = "title:\""
        query += ne + "\" OR "
        query += "title_str:\""
        query += self.entity.clean_ne + "\""
        query += " OR lastpart_str:\""
        query += self.entity.clean_ne.split(' ')[-1] + "\""

        if self.DEBUG:
            self.query = query + "&sort=lang+desc,inlinks+desc"

        try:
            solr_response = self.SOLR_CONNECTION.query(
                q=query, rows=self.SOLR_ROWS, indent="on",
                sort="lang,inlinks", sort_order="desc")
            numfound = solr_response.numFound
            solr_result_count = numfound if numfound <= self.SOLR_ROWS else self.SOLR_ROWS
        except Exception as error_msg:
            reason = "Failed to query solr: " + str(error_msg)
            self.result = None, -1.0, None, reason
        if solr_response is not None and solr_response.numFound == 0:
            self.result = False, 0, False, 'Nothing found'
        
        return solr_response, solr_result_count
        

    def get_total_inlinks(self):
        if self.DEBUG:
            self.flow.append(inspect.stack()[0][3])
        inlinks_nl = 0
        inlinks_en = 0
        for i in range(self.solr_result_count):
            document = self.matches[i].description.document
            lang = document.get('lang')
            if lang == 'nl':
                inlinks_nl += document.get('inlinks')
            else:
                inlinks_en += document.get('inlinks')
        return inlinks_nl, inlinks_en


    def match_id(self):
        if self.DEBUG:
            self.flow.append(inspect.stack()[0][3])

        for i in self.active_match_ids:
            id_match = self.matches[i].match_id()

            if self.FAST:
                # Settle for the first exact id match
                if id_match:
                    reason = "Main title match"
                    match = self.matches[i].description.document.get('id')
                    label = self.matches[i].description.label
                    prob = 1
                    self.result = match, prob, label, reason
                    return


    def match_titles(self):
        if self.DEBUG:
            self.flow.append(inspect.stack()[0][3])

        for i in self.active_match_ids:
            self.matches[i].match_titles()

        if self.FAST:
            # Count the number of times the asked ne appears in the title field
            # and settle for the first hit if it has the best count
            first_result_count = self.matches[0].title_start_match
            first_result_best = True

            for i in self.active_match_ids:
                if self.matches[i].title_start_match > first_result_count:
                    first_result_best = False

            if first_result_best and first_result_count > 2:
                if self.matches[0].title_end_match > 0:
                    reason = "First Solr hit best"
                    match = self.matches[0].description.document.get('id')
                    label = self.matches[i].description.label
                    prob = 1
                    self.result = match, prob, label, reason
                    return

            # If the first match wasn't the best and the ne has multiple parts
            # choose the first result with at least one title_start_match
            if len(self.entity.ne.split()) > 1:
                for i in self.active_match_ids:
                    if self.matches[i].title_start_match > 0:
                        reason = "Abbreviation test" # Why this reason?
                        match = self.matches[i].description.document.get('id')
                        label = self.matches[i].description.label
                        prob = 0.7
                        self.result = match, prob, label, reason
                        return


    def match_last_part(self):
        if self.DEBUG:
            self.flow.append(inspect.stack()[0][3])

        for i in self.active_match_ids:
            last_part_match = self.matches[i].match_last_part()
            
            if self.FAST:
                if last_part_match:
                    reason = "Lastpart match"
                    match = self.matches[i].description.document.get('id')
                    label = self.matches[i].description.label
                    # Calculate probability
                    match_label = self.matches[i].description.norm_title_str[0]
                    inlinks = self.matches[i].description.document.get('inlinks')
                    ne = self.entity.ne
                    if lang == 'nl':
                        p = self.calculate_propability(self.inlinks_nl, inlinks, match_label, ne)
                    else:
                        p = self.calculate_propability(self.inlinks_en, inlinks, match_label, ne)
                    self.result = match, p, label, reason
                    return 


    def calculate_propability(self, total, inlinks, match_label, ne):
        if total > 0:
            p = (float(inlinks) / total)
        else:
            p = 0.1
        if len(match_label[0].split()) - 1 > 2:
            c = float(len(ne.split())) / (len(match_label[0].split()) - 1)
        if p > 1:
            p = 1
        return p


    def __repr__(self):
        response = str(self.result)
        if self.DEBUG:
            response = str(self.result) + ", flow: " + ":".join(self.flow)
            response += ", query: " + self.SOLR_SERVER + "select?q=" + self.query
        return response


    def __getitem__(self, count):
        if self.result:
            return [i for i in self.result][count]
        return False


class Match():

    entity = None
    description = None
    
    solr_ranking = 0

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

    date_match = 0
    type_match = 0
    cos_sim = 0


    def __init__(self, entity, description, solr_ranking):
        self.entity = entity
        self.description = description


    def match_id(self):
        # Use normalized title string list until they are available from the index
        # match_label = self.description.document.get('title_str')
        ne = self.entity.ne
        match_label = self.description.norm_title_str[0]
        
        self.main_title_match = 1 if match_label.find(ne) > -1 else 0
        self.main_title_start_match = 1 if match_label.startswith(ne) else 0
        self.main_title_end_match = 1 if match_label.endswith(ne) else 0
        self.main_title_exact_match = 1 if match_label == ne else 0

        return self.main_title_exact_match


    def match_titles(self):
        title_match = 0
        title_start_match = 0
        title_end_match = 0
        title_exact_match = 0

        # Use normalized title string list until they are available from the index
        # match_label = self.description.document.get('title_str')
        ne = self.entity.ne
        match_label = self.description.norm_title_str

        for l in match_label:
            if l.find(ne) > -1:
                title_match += 1
            if l.startswith(ne):
                title_start_match +=1
            if l.endswith(ne):
                title_end_match += 1
            if l == ne:
                title_exact_match += 1
        
        self.title_match = title_match
        self.title_start_match = title_start_match
        self.title_end_match = title_end_match
        self.title_exact_match = title_exact_match


    def match_last_part(self):
        ne = self.entity.ne
        match_label = self.description.norm_title_str
        
        # If the entity consists of a single word
        if not len(ne.split()) > 1:                 
            # And the main label is longer than ne string and contains the ne string
            # and there is at least one label in which the ne string does and a bracket does not appear
            if len(match_label[0]) >= len(ne) and (self.main_title_start_match > 0 or self.main_title_end_match > 1) and True in [j.find(ne) > -1 and not j.find('(') > -1 for j in match_label]:
                    # And the main label has only one part, or the last part is the same as the ne and it does not contain 'et'
                    if len(match_label[0].split()) == 1 or (match_label[0].split()[-1] == ne and not match_label[0].find(' et ') > -1):
                        self.last_part_match = 1
                        return self.last_part_match

        # If the entity consists of multiple words
        else:
            # How many parts in the entity and the main label
            count_label_parts = len(match_label[0].split(' ')) - 1
            count_ne_parts = len(ne.split(' ')) - 1
            
            # If more parts in the ne than the main label, or the first ne part has more than two letters and is not the same as the main label first part
            # No match can be made, so return zero
            if count_label_parts < count_ne_parts or len(ne.split()[0]) > 2 and not ne.split()[0] == match_label[0].split()[0]:
                if not [i[0] for i in match_label[0].split()] == [i[0] for i in ne.split()]:
                    return self.last_part_match
            
            # Else
            for label in match_label:
                if not label.strip():
                    return self.last_part_match
                if label.endswith(ne.split()[-1]) and label.strip():
                    if len(ne.split()) == len(label.split()) and not [l[0] for l in ne.split()] == [l[0] for l in label.split()]:
                        return self.last_part_match
                    skip = False
                    for l in ne.split():
                        if not l[0] in [l[0] for l in label.split()]:
                            skip = True
                            break
                    if skip:
                        return self.last_part_match

                    if ne.find('.') > -1 or True in [len(l) <= 2 for l in ne.split()]:
                        self.last_part_match = 1
                        return self.last_part_match


    def has_name_conflict(self):
        if not self.main_title_exact_match and not self.last_part_match:
            if not (self.title_start_match > 0 and self.title_end_match > 0) and not (len(self.entity.ne.split()) > 1 and self.title_start_match > 0):
                self.name_conflict = 1
        return self.name_conflict


    def match_date(self):
        if self.entity.publ_date:
            year_of_publ = int(self.entity.publ_date[:4])
            year_of_birth = self.description.document.get('yob') 
            if year_of_birth is not None:
                if year_of_publ < year_of_birth:
                    self.date_match = -1
                else:
                    self.date_match = 1


    def match_type(self):

        TPTA_SCHEMA_MAPPING = {'person': 'Person', 'location': 'Place', 'organisation': 'Organization'}

        if self.entity.ne_type in TPTA_SCHEMA_MAPPING:
            schema_types = self.description.document.get('schemaorgtype')
            if schema_types:
                for t in schema_types:
                    if t == TPTA_SCHEMA_MAPPING[self.entity.ne_type]:
                        self.type_match = 1
                        break
    
    
    def match_abstract(self):
        abstract = self.description.document.get('abstract')
        if abstract:
            self.cos_sim(self.ocr, abstract)


    def match_abstract(self):
        '''
        Calculate cosine similarity between entity ocr and dbpedia abstract.
        '''
        warnings.filterwarnings('ignore', message='.*Unicode equal comparison.*')
        abstract = self.description.document.get('abstract')
        ocr = self.entity.ocr
        
        if ocr and abstract:
            corpus = [ocr, abstract]

            # Tokenize
            punctuation = [',', '.']
            bow = []
            for d in corpus:
                for p in punctuation:
                    d = d.replace(p, '')
                d = d.lower()
                d = d.split()
                bow.append(d)

            # Build vocabulary
            voc = []
            for b in bow:
                for t in b:
                    if not t in voc:
                        voc.append(t)

            # Vectorize
            vec = [] 
            for b in bow:
                v = np.zeros(len(voc))
                for t in voc:
                    v[voc.index(t)] = b.count(t)
                v_norm = v / np.linalg.norm(v)
                vec.append(v_norm)

            self.cos_sim = 1 - spatial.distance.cosine(vec[0], vec[1])


class Description():

    document = None
    norm_title_str = []
    label = ''


    def __init__(self, doc):
        self.document = doc
        self.label = doc.get('title')[0]
        
        # Normalize titles here until they become available from the index
        norm_title_str = []
        for t in doc.get('title_str'):
            norm_title_str.append(self.normalize(t))
        self.norm_title_str = norm_title_str


    def normalize(self, ne):
        '''
        Remove periods, hyphens, capitalization.
        '''
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


class Entity():

    orig_ne = ''
    clean_ne = ''
    norm_ne = ''
    ne = ''

    titles = []

    no_parts = 0

    ne_type = ''

    url = ''
    ocr = ''
    publ_date = ''
    publ_place = ''


    def __init__(self, ne, ne_type='', url=''):
        self.orig_ne = ne
        self.ne_type = ne_type
        self.url = url

        self.clean_ne = self.clean(ne)
        self.norm_ne = self.normalize(self.clean_ne)
        self.ne, self.titles = self.strip_titles(self.norm_ne)
        
        no_parts = len(self.ne.split())


    def clean(self, ne):
        ''' 
        Remove unwanted characters from the named entity.
        '''
        remove_char = ["+", "&&", "||", "!", "(", ")", "{", u'„',
                       "}", "[", "]", "^", "\"", "~", "*", "?", ":"]

        for char in remove_char:
            if ne.find(char) > -1:
                ne = ne.replace(char, u'')

        ne = ne.strip()

        return ne


    def normalize(self, ne):
        '''
        Remove periods, hyphens, capitalization.
        '''
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


    def get_ocr(self):
        '''
        Get plain text OCR for the article in which the ne occurs.
        '''
        url = self.url
        f = urllib.urlopen(url)
        ocr_string = f.read()
        f.close()
        ocr_tree = etree.fromstring(ocr_string)
        self.ocr = etree.tostring(ocr_tree, encoding='utf8', method='text')
        return self.ocr


    def get_metadata(self):
        '''
        Get the date and place of publication of the article in which the ne occurs.
        '''
        url = self.url
        pos = url.find('mpeg21') + 6
        url = url[:pos]
        f = urllib.urlopen(url) 
        md_string = f.read()
        f.close()
        md_tree = etree.fromstring(md_string)
        md_id = url[url.find('ddd:'):] + ':metadata'
        for node in md_tree.iter('{urn:mpeg:mpeg21:2002:02-DIDL-NS}Component'):
            if node.attrib['{http://purl.org/dc/elements/1.1/}identifier'] == md_id:
                dcx = node.find('{urn:mpeg:mpeg21:2002:02-DIDL-NS}Resource/{info:srw/schema/1/dc-v1.1}dcx')
                self.publ_date = dcx.findtext('{http://purl.org/dc/elements/1.1/}date')
                for sp in dcx.iter('{http://purl.org/dc/terms/}spatial'):
                    if '{http://www.w3.org/2001/XMLSchema-instance}type' in sp.attrib: 
                        self.publ_place = sp.text


if __name__ == '__main__':

    if not len(sys.argv) > 1:
        print("Usage: ./disambiguation.py [Named Entity (string)]")
    elif len(sys.argv) > 3:
        print(linkEntity(sys.argv[1], sys.argv[2], sys.argv[3], debug=True, full=True))
    else:
        print(linkEntity(sys.argv[1], debug=True))


