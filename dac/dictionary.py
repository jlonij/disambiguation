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

unwanted = ['nederland', 'nederlands', 'nederlandse', 'holland', 'hollands',
    'hollandse', 'amsterdam', 'amsterdams', 'amsterdamse']

months = ['januari', 'februari', 'maart', 'april', 'mei', 'juni', 'juli',
    'augustus', 'september', 'oktober', 'november', 'december']

titles = ['heer', 'hr', 'dhr', 'meneer', 'mevrouw', 'mevr', 'mw', 'mej',
    'mejuffrouw', 'drs', 'ing', 'ir', 'dr', 'mr', 'sir', 'dame', 'lady', 'miss']

topics = ['politics', 'business', 'culture', 'science', 'sports']

types_dbo = {
    'person': ['Person'],
    'organisation': ['Organisation'],
    'location': ['Location'],
    'other': []
}

types_vocab = {
    'person': ['familie', 'zoon', 'dochter', 'geboren', 'overleden', 'dood',
        'man', 'vrouw', 'echtgenoot', 'echtgenote', 'vader', 'moeder'],
    'organisation': ['organisatie'],
    'location': ['plaats', 'gemeente', 'provincie', 'stad', 'dorp', 'regio',
        'wijk', 'gebied', 'waterschap', 'straat', 'district', 'county',
        'kanton', 'republiek', 'prefectuur', 'graafschap', 'arrondissement',
        'streek', 'staten', 'plein', 'station', 'vliegveld', 'gebouw', 'brug',
        'monument', 'metro', 'luchthaven', 'rivier', 'gebergte', 'eiland',
        'vulkaan', 'archipel'],
    'other': []
}

roles_dbo = {
    'politics_person': ['OfficeHolder', 'Politician', 'Royalty', 'Monarch',
        'Noble'],
    'military_person': ['MilitaryPerson'],
    'business_person': ['BusinessPerson', 'Economist'],
    'religion_person': ['Cleric', 'Religious'],
    'culture_person': ['Architect', 'Artist', 'Writer', 'Philosopher',
        'FictionalCharacter', 'Journalist', 'Presenter'],
    'science_person': ['Astronaut', 'Engineer', 'Scientist'],
    'sports_person': ['Athlete', 'Coach', 'SportsManager'],
    'politics_organisation': ['PoliticalParty', 'GovernmentAgency',
        'Legislature'],
    'military_organisation': ['MilitaryUnit'],
    'business_organisation': ['Company', 'TradeUnion'],
    'religion_organisation': ['ReligiousOrganisation'],
    'culture_organisation': ['Group', 'Broadcaster'],
    'science_organisation': ['EducationalInstitution'],
    'sports_organisation': ['SportsTeam', 'SportsLeague'],
    'politics_location': [],
    'military_location': ['MilitaryStructure'],
    'business_location': [],
    'religion_location': ['ReligiousBuilding'],
    'culture_location': ['Museum', 'Theatre'],
    'science_location': ['CelestialBody'],
    'sports_location': ['SportFacility'],
    'politics_concept': [],
    'military_concept': ['MilitaryConflict'],
    'business_concept': [],
    'religion_concept': ['Deity'],
    'culture_concept': ['Work', 'Genre', 'FilmFestival', 'MusicFestival'],
    'science_concept': [],
    'sports_concept': ['SportsEvent']
}

roles_vocab = {
    'politics_person': ['minister', 'premier', 'kamerlid', 'kamerleden',
        'partijleider', 'burgemeester', 'staatssecretaris', 'president',
        'wethouder', 'consul', 'ambassadeur', 'raadslid', 'raadsleden',
        'diplomaat', 'fractieleider', 'politicus', 'politica', 'politici',
        'staatsman', 'kanselier', 'keizer', 'koning', 'prins', 'majesteit',
        'hertog', 'graaf', 'gravin'],
    'military_person': ['generaal', 'majoor', 'luitenant', 'kolonel',
        'kapitein', 'bevelhebber', 'fuhrer'],
    'business_person': ['manager', 'teamleider', 'zakenman', 'bedrijfsleider',
        'ondernemer', 'handelaar', 'econoom'],
    'religion_person': ['dominee', 'paus', 'kardinaal', 'bisschop',
        'monseigneur', 'mgr', 'kapelaan', 'deken', 'abt', 'prior', 'pastoor',
        'pastor', 'pater', 'predikant', 'st', 'sint', 'rabbijn', 'imam',
        'geestelijke', 'frater'],
    'culture_person': ['acteur', 'actrice', 'regisseur', 'musicus', 'componist',
        'zanger', 'trompettist', 'orkestleider', 'gitarist', 'pianist',
        'songwriter', 'dirigent', 'muzikant', 'drummer', 'bassist',
        'toetsenist', 'vocalist', 'sopraan', 'tenor', 'danser', 'ballerina',
        'kunstenaar', 'kunstenares', 'schilder', 'beeldhouwer', 'architect',
        'fotograaf', 'ontwerper', 'striptekenaar', 'illustrator', 'auteur',
        'schrijver', 'schrijfster', 'dichter', 'journalist', 'pseudoniem',
        'personage', 'stripfiguur'],
    'science_person': ['prof', 'professor', 'natuurkundige', 'scheikundige',
        'wiskundige', 'bioloog', 'historicus', 'onderzoeker', 'wetenschapper',
        'astronoom', 'statisticus', 'hoogleraar', 'decaan', 'onderwijzer',
        'leerkracht'],
    'sports_person': ['atleet', 'atlete', 'sportman', 'sportvrouw', 'sporter',
        'wielrenner', 'voetballer', 'tennisser', 'zwemmer', 'spits', 'keeper',
        'scheidsrechter', 'schaatser', 'schaker', 'coureur', 'middenvelder',
        'trainer', 'coach', 'doelman', 'hockeyer', 'hockeyster', 'ruiter'],
    'politics_organisation': ['partij', 'overheidsdienst', 'ministerie',
        'overheidsinstelling'],
    'military_organisation': ['leger', 'bataljon', 'regiment', 'brigade'],
    'business_organisation': ['bedrijf', 'bank', 'luchtvaartmaatschappij',
        'onderneming', 'hotel', 'uitvoerder', 'fabrikant', 'vakbond',
        'aannemer'],
    'religion_organisation': ['kerk', 'bisdom', 'parochie'],
    'culture_organisation': ['band', 'rockband', 'popgroep', 'zender', 'omroep',
        'uitgever', 'persbureau', 'operagezelschap', 'balletgezelschap',
        'toneelgroep', 'kunstenaarscollectief', 'krant', 'tijdschrift',
        'dagblad', 'weekblad', 'radiostation'],
    'science_organisation': ['school', 'universiteit', 'onderzoeksinstituut',
        'onderzoeksinstelling', 'faculteit'],
    'sports_organisation': ['club'],
    'politics_location': ['paleis', 'parlement'],
    'military_location': ['fort', 'kamp', 'kazerne'],
    'business_location': ['bedrijventerrein'],
    'religion_location': ['kathedraal', 'tempel', 'kapel', 'heiligdom'],
    'culture_location': ['museum', 'bioscoop', 'theater', 'concertzaal',
        'poppodium', 'operahuis'],
    'science_location': ['planeet', 'ster', 'asteroide', 'maan'],
    'sports_location': ['stadion', 'arena'],
    'politics_concept': ['verkiezingen', 'wet'],
    'military_concept': ['oorlog', 'slag'],
    'business_concept': ['staking'],
    'religion_concept': ['god'],
    'culture_concept': ['film', 'album', 'plaat', 'nummer', 'single', 'boek',
        'roman', 'novelle', 'bundel', 'script', 'serie', 'toneelstuk',
        'gedicht', 'schilderij', 'beeld', 'strip', 'verhalen', 'genre'],
    'science_concept': [],
    'sports_concept': ['wedstrijd', 'toernooi', 'competitie']
}

