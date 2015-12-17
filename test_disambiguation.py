#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

sys.path.append('..' + os.sep)
sys.path.append('.' + os.sep)


def Nomatch():
    # Nomatch check
    """
    >>> from disambiguation import linkEntity
    >>> linkEntity("Heten")
    (False, 0, False, u'Name conflict (Angst essen Seele auf, Heten)')
    >>> linkEntity("Ton van Engelen")
    (False, 0, False, u'Name conflict (Engelen, Ton van Engelen)')
    >>> linkEntity("G. Smits")
    (False, 0, False, u'Name conflict (Reitze Smits, G Smits)')
    >>> linkEntity("Boy Lawson")
    (False, 0, False, u'Name conflict (Denis Lawson, Boy Lawson)')
    >>> linkEntity("Joop")
    (False, 0, False, u'Name conflict (Joop Alberda, Joop)')
    """

def Short_words():
    # Check for short words
    """
    >>> from disambiguation import linkEntity
    >>> linkEntity("Hij")
    (u'<http://nl.dbpedia.org/resource/Hij_(single)>', 0.40816326530612246, 'Hij', 'm == ne')
    >>> linkEntity("Hall")
    (u'<http://nl.dbpedia.org/resource/Hallenkerk>', 0.02755870624723084, u'Hallenkerk', ' p: inlinks : 311 total : 11285')
    >>> linkEntity("Juliette")
    (False, 0, False, u'Name conflict (Sainte-Juliette-sur-Viaur, Juliette)')
    >>> linkEntity("Christiaan")
    (False, 0, False, u'Name conflict (Christiaan V van Denemarken, Christiaan)')
    """

def Hyphen_test():
    # Test for hyphens (-)
    """
    >>> from disambiguation import linkEntity
    >>> linkEntity("Ina Boudier-Bakker")
    (u'<http://nl.dbpedia.org/resource/Ina_Boudier-Bakker>', 1, 'Ina Boudier-Bakker', 'ID match')
    >>> linkEntity("Ina Boudier Bakker")
    (u'<http://nl.dbpedia.org/resource/Ina_Boudier-Bakker>', 1, 'Ina Boudier Bakker', 'm == ne, disambig')
    >>> linkEntity("Zuid-Amerika")
    (u'<http://nl.dbpedia.org/resource/Zuid-Amerika>', 1, 'Zuid-Amerika', 'ID match')
    >>> linkEntity("Zuid Amerika")
    (u'<http://nl.dbpedia.org/resource/Zuid-Amerika>', 1, 'Zuid Amerika', 'm == ne, disambig')
    """

def Abbreviation_test():
    # Test abbreviations
    """
    >>> from disambiguation import linkEntity
    >>> linkEntity("W Hermans")
    (u'<http://nl.dbpedia.org/resource/Willem_Frederik_Hermans>', 0.3012345679012346, u'Willem Frederik Hermans', ' (lastpart match)')
    >>> linkEntity("W.F. Hermans")
    (u'<http://nl.dbpedia.org/resource/Willem_Frederik_Hermans>', 0.3012345679012346, u'Willem Frederik Hermans', ' (lastpart match)')
    >>> linkEntity("Willem F. Hermans")
    (u'<http://nl.dbpedia.org/resource/Willem_Frederik_Hermans>', 0.3012345679012346, u'Willem Frederik Hermans', ' (lastpart match)')
    >>> linkEntity("W F Hermans")
    (u'<http://nl.dbpedia.org/resource/Willem_Frederik_Hermans>', 0.3012345679012346, u'Willem Frederik Hermans', ' (lastpart match)')
    >>> linkEntity("W Drees")
    (u'<http://nl.dbpedia.org/resource/Willem_Drees>', 0.9635416666666666, u'Willem Drees', ' (lastpart match)')
    >>> linkEntity("W. Drees")
    (u'<http://nl.dbpedia.org/resource/Willem_Drees>', 0.9635416666666666, u'Willem Drees', ' (lastpart match)')
    >>> linkEntity("St Petersburg")
    (u'<http://nl.dbpedia.org/resource/Sint-Petersburg>', 0.7, u'Sint-Petersburg', 'Abrriviation test')
    """

def Dutch_test():
    # Should result in Dutch dbpedia
    """
    >>> from disambiguation import linkEntity
    >>> linkEntity("Moskou")
    (u'<http://nl.dbpedia.org/resource/Moskou>', 1, 'Moskou', 'ID match')
    >>> linkEntity("Germany")
    (u'<http://nl.dbpedia.org/resource/Duitsland>', 1, 'Germany', 'm == ne, disambig')
    >>> linkEntity("De Vries")
    (u'<http://nl.dbpedia.org/resource/Floris_de_Vries>', 0.03155479059093517, u'Floris de Vries', ' (lastpart match)')
    """

def Popularity_test():
    # Popularity test
    """
    >>> from disambiguation import linkEntity
    >>> linkEntity("A Einstein")
    (u'<http://nl.dbpedia.org/resource/Albert_Einstein>', 0.8754098360655738, u'Albert Einstein', ' (lastpart match)')
    >>> linkEntity("A. Einstein")
    (u'<http://nl.dbpedia.org/resource/Albert_Einstein>', 0.8754098360655738, u'Albert Einstein', ' (lastpart match)')
    >>> linkEntity("Einstein")
    (u'<http://nl.dbpedia.org/resource/Albert_Einstein>', 0.35201054713249835, u'Albert Einstein', ' p: inlinks : 534 total : 1517')
    >>> linkEntity("Rembrandt")
    (u'<http://nl.dbpedia.org/resource/Rembrandt_van_Rijn>', 1, 'Rembrandt', 'First solr hit best')
    >>> linkEntity("Prof Einstein")
    (False, 0, False, u'Name conflict (Bob Einstein, Prof Einstein)')
    >>> linkEntity("Bach")
    (u'<http://nl.dbpedia.org/resource/Johann_Sebastian_Bach>', 0.2492283950617284, u'Johann Sebastian Bach', ' p: inlinks : 1292 total : 5184')
    >>> linkEntity("Friedrich Wilhelm")
    (u'<http://nl.dbpedia.org/resource/William_Herschel>', 0.7, u'William Herschel', 'Abrriviation test')
    >>> linkEntity("Napoleon")[0]
    u'<http://nl.dbpedia.org/resource/Napoleon_Bonaparte>'
    """

def Ending_brackets():
    # Test things like: Germany_(country)
    '''
    >>> from disambiguation import linkEntity
    >>> linkEntity("Brandenburg")
    (u'<http://nl.dbpedia.org/resource/Brandenburg_(provincie)>', 0.5180244110133408, 'Brandenburg', 'm == ne')
    '''

if __name__ == "__main__":
    import sys
    reload(sys)
    sys.setdefaultencoding("UTF-8")
    import doctest
    doctest.testmod()
