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

uninformative = ['nederland', 'holland', 'amsterdam']

days = ['maandag', 'dinsdag', 'woensdag', 'donderdag', 'vrijdag', 'zaterdag',
    'zondag']

months = ['januari', 'februari', 'maart', 'april', 'mei', 'juni', 'juli',
    'augustus', 'september', 'oktober', 'november', 'december']

titles = ['heer', 'hr', 'dhr', 'meneer', 'mevrouw', 'mevr', 'mw', 'mej',
    'mejuffrouw', 'drs', 'ing', 'ir', 'dr', 'sir', 'mr']

types = {
    'person': {
        'schema_types': ['Person'],
        'words': ['familie', 'zoon', 'dochter', 'persoon', 'geboren',
            'overleden', 'dood', 'man', 'vrouw', 'leven', 'echtgenoot',
            'echtgenote', 'vader', 'moeder']
        },
    'location': {
        'schema_types': ['Place', 'Location'],
        'words': ['inwoners', 'oppervlakte', 'gelegen', 'noorden', 'zuiden',
            'oosten', 'westen', 'noordoosten', 'noordwesten', 'zuidoosten',
            'zuidwesten', 'grens', 'zuidelijke', 'noordelijke', 'westelijke',
            'oostelijke', 'omgeving']
        },
    'organisation': {
        'schema_types': ['Organization', 'Organisation'],
        'words': ['organisatie']
        }
    }

roles = {
    # Persons
    'politician': {
        'words': ['minister', 'ministers', 'premier', 'kamerlid', 'kamerleden',
            'partijleider', 'burgemeester', 'staatssecretaris', 'president',
            'wethouder', 'consul', 'ambassadeur', 'gemeenteraadslid',
            'diplomaat', 'fractieleider', 'politicus', 'politica', 'politici',
            'staatsman'],
        'schema_types': ['Politician', 'OfficeHolder', 'Judge',
            'MemberOfParliament', 'President', 'PrimeMinister',
            'Governor', 'Congressman', 'Mayor'],
        'subjects': ['politics'],
        'types': ['person']
        },
    'royalty': {
        'words': ['keizer', 'koning', 'koningin', 'vorst', 'prins',
            'prinses', 'kroonprins', 'kroonprinses', 'majesteit', 'hertog',
            'hertogin', 'graaf', 'gravin'],
        'schema_types': ['Royalty', 'Monarch', 'Noble'],
        'subjects': ['politics'],
        'types': ['person']
        },
    'military_person': {
        'words': ['generaal', 'majoor', 'luitenant', 'kolonel',
            'kapitein', 'bevelhebber'],
        'schema_types': ['MilitaryPerson'],
        'subjects': ['politics'],
        'types': ['person']
        },
    'sports_person': {
        'words': ['atleet', 'atlete', 'sportman', 'sportvrouw', 'sporter',
            'wielrenner', 'voetballer', 'tennisser', 'zwemmer', 'spits',
            'keeper', 'scheidsrechter', 'profvoetballer', 'schaatser', 'spits',
            'langebaanschaatser', 'schaker', 'voetballers', 'autocoureur',
            'middenvelder', 'coureur', 'voetbaltrainer', 'trainer', 'coach',
            'voetbalcoach', 'doelman'],
        'schema_types': ['Athlete', 'SoccerPlayer', 'Cyclist', 'SoccerManager',
            'TennisPlayer', 'Swimmer', 'Boxer', 'Wrestler', 'Speedskater',
            'Skier', 'WinterSportPlayer', 'GolfPlayer', 'RacingDriver',
            'MotorsportRacer', 'Canoist', 'Cricketer', 'RugbyPlayer',
            'HorseRider', 'AmericanFootballPlayer', 'Rower', 'MotorcycleRider',
            'Skater', 'BaseballPlayer', 'BasketballPlayer', 'Gymnast',
            'SportsManager', 'IceHockeyPlayer', 'FigureSkater',
            'HandballPlayer'],
        'subjects': ['sports'],
        'types': ['person']
        },
    'performing_artist': {
        'words': ['acteur', 'toneelspeler', 'filmregisseur', 'regisseur',
            'actrice'],
        'schema_types': ['Actor', 'VoiceActor', 'Presenter', 'Comedian'],
        'subjects': ['culture'],
        'types': ['person']
        },
    'musical_artist': {
         'words': ['musicus', 'componist', 'zanger', 'zangeres',
             'trompetspeler', 'orkestleider', 'gitarist', 'pianist',
             'songwriter', 'dirigent', 'muzikant', 'drummer', 'bassist'],
        'schema_types': ['MusicalArtist', 'ClassicalMusicArtist'],
        'subjects': ['culture'],
        'types': ['person']
        },
    'visual_artist': {
         'words': ['kunstenaar', 'schilder', 'beeldhouwer', 'architect',
            'fotograaf', 'ontwerper', 'kunstschilder', 'striptekenaar',
            'illustrator', 'kunstenaars'],
        'schema_types': ['Painter', 'Architect', 'Photographer',
            'FashionDesigner'],
        'subjects': ['culture'],
        'types': ['person']
        },
    'writer': {
        'words': ['auteur', 'schrijver', 'schrijfster', 'schrijvers', 'dichter',
            'journalist', 'pseudoniem'],
        'schema_types': ['Writer', 'Journalist', 'Screenwriter',
            'Poet'],
        'subjects': ['culture'],
        'types': ['person']
        },
    'fictional_character': {
        'words': ['personage', 'stripfiguur'],
        'schema_types': ['FictionalCharacter', 'SoapCharacter'],
        'subjects': ['culture'],
        'types': ['person']
        },
    'business_person': {
        'words': ['manager', 'teamleider', 'zakenman', 'directeur',
            'bedrijfsleider', 'ondernemer'],
        'schema_types': ['BusinessPerson'],
        'subjects': ['business'],
        'types': ['person']
        },
    'scientist': {
        'words': ['prof', 'professor', 'natuurkundige', 'scheikundige',
            'wiskundige', 'bioloog', 'historicus', 'onderzoeker',
            'wetenschapper', 'filosoof', 'docent', 'astronoom'],
        'schema_types': ['Scientist', 'Historian', 'Biologist', 'Philosopher',
            'Professor'],
        'subjects': ['science'],
        'types': ['person']
        },
    'religious_person': {
        'words': ['dominee', 'paus', 'kardinaal', 'aartsbisschop',
            'bisschop', 'monseigneur', 'mgr', 'kapelaan', 'deken',
            'abt', 'prior', 'pastoor', 'pater', 'predikant', 'st',
            'opperrabbijn', 'rabbijn', 'imam', 'geestelijke', 'frater'],
        'schema_types': ['ChristianBishop', 'Cardinal', 'Cleric', 'Saint',
            'Pope'],
        'subjects': ['religion'],
        'types': ['person']
        },
    # Locations
    'settlement': {
        'words': ['plaats', 'gemeente', 'provincie', 'provincies', 'stad',
            'dorp', 'regio', 'wijk', 'gebied', 'stadsdeel', 'waterschap',
            'straat', 'staat', 'district', 'deelstaat', 'departement',
            'county', 'kanton', 'hoofdstad', 'republiek', 'prefectuur',
            'graafschap', 'land', 'arrondissement', 'streek', 'landgoed',
            'stadje', 'staten', 'gemeenten', 'districten', 'deelgemeente',
            'gebieden', 'havenstad', 'dorpje', 'plein'],
        'schema_types': ['Settlement', 'Village', 'Municipality', 'Town',
            'AdministrativeRegion', 'City', 'HistoricPlace', 'PopulatedPlace',
            'ProtectedArea', 'CityDistrict', 'Country', 'SubMunicipality',
            'Street', 'District'],
        'subjects': [],
        'types': ['location']
        },
    'infrastructure': {
        'words': ['station', 'metrostation', 'vliegveld', 'gebouw', 'brug',
            'monument', 'metro', 'luchthaven', 'metrolijn'],
        'schema_types': ['Building', 'Road', 'Station', 'RailwayStation',
            'Airport', 'HistoricBuilding', 'Bridge', 'Dam',
            'ArchitecturalStructure', 'Monument', 'Castle', 'MetroStation'],
        'subjects': [],
        'types': ['location']
        },
    'natural_location': {
        'words': ['rivier', 'gebergte', 'meer', 'planeet', 'eiland',
            'eilanden', 'eilandengroep', 'kust', 'kuststrook', 'kustgebied',
            'schiereiland', 'rivieren', 'vulkaan', 'archipel'],
        'schema_types': ['River', 'Mountain', 'Lake', 'CelestialBody',
            'Asteroid', 'Planet', 'Island', 'MountainRange', 'BodyOfWater',
            'MountainPass'],
        'subjects': [],
        'types': ['location']
        },
    'sports_location': {
        'words': ['stadion', 'arena'],
        'schema_types': ['Stadium', 'Arena'],
        'subjects': ['sports'],
        'types': ['location']
        },
    'religious_location': {
        'words': ['bisdom', 'kathedraal', 'tempel', 'kapel', 'heiligdom'],
        'schema_types': ['Church', 'ReligiousBuilding', 'Diocese'],
        'subjects': ['religion'],
        'types': ['location', 'organisation']
        },
    # Organizations
    'company': {
        'words': ['bedrijf', 'bank', 'luchtvaartmaatschappij', 'onderneming',
            'hotel', 'uitvoerder', 'fabrikant'],
        'schema_types': ['Company', 'Bank', 'Airline', 'Hotel'],
        'subjects': ['business'],
        'types': ['organisation']
        },
    'school': {
        'words': ['basisschool', 'school', 'hogeschool', 'universiteit',
            'onderzoeksinstituut', 'faculteit'],
        'schema_types': ['School', 'University'],
        'subjects': ['science'],
        'types': ['organisation', 'location']
        },
    'political_organisation': {
        'words': ['partij'],
        'schema_types': ['PoliticalParty', 'GovernmentAgency'],
        'subjects': ['politics'],
        'types': ['organisation']
        },
    'sports_organisation': {
        'words': ['club', 'voetbalclub'],
        'schema_types': ['SoccerClub', 'RugbyClub', 'SportsTeam',
            'HockeyTeam'],
        'subjects': ['sports'],
        'types': ['organisation']
        },
    'cultural_organisation': {
        'words': ['museum', 'band', 'rockband', 'popgroep', 'orkest',
            'metalband'],
        'schema_types': ['Band', 'MusicGroup', 'RecordLabel', 'Museum'],
        'subjects': ['culture'],
        'types': ['organisation']
        },
    'military_organisation': {
        'words': ['leger', 'bataljon', 'regiment', 'brigade'],
        'schema_types': ['MilitaryUnit'],
        'subjects': ['politics'],
        'types': ['organisation']
        },
    'media_organisation': {
        'words': ['krant', 'tijdschrift', 'zender', 'televisiezender',
            'radiozender', 'dagblad', 'weekblad', 'radiostation'],
        'schema_types': ['Newspaper', 'Magazine', 'RadioStation', 'Publisher',
            'TelevisionStation', 'AcademicJournal', 'PeriodicalLiterature'],
        'subjects': [],
        'types': []
        },
    # Other
    'creative_work': {
        'words': ['film', 'films', 'album', 'plaat', 'nummer', 'single', 'boek',
            'boeken', 'roman', 'romans', 'novelle', 'bundel', 'dichtbundel',
            'script', 'serie', 'televisieserie', 'opera', 'toneelstuk',
            'gedicht', 'schilderij', 'beeld', 'strip', 'strips', 'verhalen',
            'animatieserie', 'speelfilm', 'studioalbum', 'debuutalbum'],
        'schema_types': ['CreativeWork', 'Film', 'Album', 'Single', 'Book',
            'TelevisionShow', 'TelevisionEpisode', 'Song', 'MusicalWork',
            'ArtWork', 'WrittenWork', 'Play'],
        'subjects': ['culture'],
        'types': []
        },
    'product': {
        'words': [],
        'schema_types': ['Product'],
        'subjects': ['business'],
        'types': []
        },
    'ship' : {
        'words': ['ss', 'stoomschip', 'passagiersschip', 'cruiseschip',
            'schip', 'vlaggeschip', 'zeilschip', 'jacht'],
        'schema_types': ['Ship'],
        'subjects': ['business'],
        'types': []
    },
    'sports_event': {
        'words': ['wedstrijd', 'voetbalcompetitie', 'toernooi',
            'voetbaltoernooi', 'wielerwedstrijd'],
        'schema_types': ['SoccerLeague', 'OlympicEvent', 'SoccerTournament',
            'GrandPrix', 'TennisTournament', 'FootballMatch', 'CyclingRace',
            'SportsEvent'],
        'subjects': ['sports'],
        'types': []
        },
    'military_event': {
        'words': ['oorlog', 'veldslag'],
        'schema_types': ['MilitaryConflict'],
        'subjects': ['politics'],
        'types': []
        }
    }

subjects = {
    'politics': ['regering', 'kabinet', 'fractie', 'tweede kamer',
        'eerste kamer', 'politiek', 'politieke', 'vorstenhuis',
        'koningshuis', 'koninklijk huis', 'troon', 'rijk',
        'keizerrijk', 'monarchie', 'leger', 'oorlog', 'troepen',
        'strijdkrachten', 'militair', 'militaire', 'partij', 'politie',
        'overheid', 'ministerie'],
    'sports': ['sport', 'voetbal', 'wielersport', 'speler', 'spelers',
        'sporten', 'wedstrijden', 'goal', 'goals'],
    'culture': ['kunst', 'cultuur', 'muziek', 'toneel', 'theater', 'cinema',
        'fictief', 'fictieve', 'rock', 'jazz', 'geregisseerd', 'kunstveiling',
        'kunstveilingen', 'piano', 'gitaar', 'pianisten', 'recital',
        'televisie', 'sciencefiction', 'metal', 'hoofdrol', 'genre',
        'afleveringen', 'cultureel', 'culturele'],
    'business': ['economie', 'beurs', 'aandelen', 'bedrijfsleven',
        'management', 'werknemer', 'werknemers', 'salaris', 'staking',
        'personeel', 'beleggers', 'rente'],
    'science': ['wetenschap', 'studie', 'onderzoek', 'uitvinding',
        'ontdekking', 'gezondheid', 'wiskunde', 'natuurkunde', 'biologie',
        'scheikunde', 'astronomie', 'wetenschappelijk', 'wetenschappelijke'],
    'religion': ['geloof', 'religie', 'katholiek', 'katholieke', 'protestants',
        'protestantse', 'katholicisme', 'protestantisme', 'gereformeerd',
        'gereformeerde', 'kerk', 'christelijk', 'christelijke', 'christus',
        'religieus', 'religieuze', 'rooms', 'theoloog', 'theologie',
        'theologisch']
    }
