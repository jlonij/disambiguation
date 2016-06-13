nnnn = ['maandag', 'dinsdag', 'woensdag', 'donderdag', 'vrijdag', 'zaterdag',
'zondag']

months = ['januari', 'februari', 'maart', 'april', 'mei', 'juni', 'juli',
'augustus', 'september', 'oktober', 'november', 'december']

genders = {
    'male': ['heer', 'hr', 'dhr', 'meneer'],
    'female': ['mevrouw', 'mevr', 'mw', 'mej', 'mejuffrouw']
    }

types = {
    'person': {
        'schema_types': ['Person', 'Agent']
    },
    'location': {
        'schema_types': ['Place', 'Location']
    },
    'organisation':
        'schema_types': ['Organization', 'Organisation']
    }

roles = {
    # Persons
    'politician': {
        'types': ['person'],
        'schema_types': ['Politician', 'OfficeHolder', 'Judge',
            'MemberOfParliament', 'President', 'PrimeMinister',
            'Governor', 'Congresman'],
        'words': ['minister', 'premier', 'kamerlid', 'partijleider',
            'burgemeester', 'staatssecretaris', 'president',
            'wethouder', 'consul', 'ambassadeur', 'gemeenteraadslid',
            'fractieleider', 'politicus']
        },
    'royalty': {
        'types': ['person'],
        'schema_types': ['Royalty', 'Monarch', 'Noble'],
        'words': ['keizer', 'koning', 'koningin', 'vorst', 'prins',
            'prinses', 'kroonprins', 'kroonprinses']
        },
    'military_person': {
        'types': ['person'],
        'schema_types': ['MilitaryPerson'],
        'words': ['generaal', 'gen', 'majoor', 'maj', 'luitenant',
            'kolonel', 'kol', 'kapitein', 'bevelhebber']
        },
    'athlete': {
        'types': ['person'],
        'schema_types': ['Athlete', 'SoccerPlayer', 'Cyclist', 'SoccerManager',
            'TennisPlayer', 'Swimmer', 'Boxer', 'Wrestler', 'Speedskater',
            'Skier', 'WinterSportPlayer', 'GolfPlayer', 'RacingDriver',
            'MotorsportRacer', 'Canoist', 'Cricketer', 'RugbyPlayer',
            'HorseRider', 'AmericanFootballPlayer', 'Rower',
            'Skater', 'BaseballPlayer', 'BasketballPlayer',
            'SportsManager', 'IceHockeyPlayer'],
        'words': ['atleet', 'sportman', 'sportvrouw', 'sporter',
            'wielrenner', 'voetballer', 'tennisser', 'zwemmer'],
        },
    'artist': {
        'types': ['person'],
        'schema_types': ['MusicalArtist', 'Artist', 'Writer', 'Actor',
            'Painter', 'Journalist', 'Architect', 'Screenwriter',
            'VoiceActor', 'Presenter', 'Photographer',
            'ClassicalMusicArtist', 'Poet', 'FashionDesigner',
            'Comedian'],
        'words': ['schrijver', 'auteur', 'acteur', 'kunstenaar',
            'schilder', 'beeldhouwer', 'architect', 'musicus',
            'schrijver', 'componist', 'fotograaf', 'dichter',
            'ontwerper', 'toneelspeler', 'filmregisseur', 'regisseur',
            'zanger', 'zangeres', 'actrice', 'trompetspeler', 'orkestleider']
        },
    'scientist': {
        'types': ['person'],
        'schema_types': ['Scientist'],
        'words': ['prof', 'professor', 'dr', 'ingenieur', 'ir',
            'natuurkundige', 'scheikundige', 'wiskundige', 'bioloog',
            'historicus', 'onderzoeker', 'drs', 'ing']
        },
    'religious_person': {
        'types': ['person'],
        'schema_types': ['ChristianBishop', 'Cardinal', 'Cleric', 'Saint', 'Pope'],
        'words': ['dominee', 'paus', 'kardinaal', 'aartsbisschop',
            'bisschop', 'monseigneur', 'mgr', 'kapelaan', 'deken',
            'abt', 'prior', 'pastoor', 'pater', 'predikant',
            'opperrabbijn', 'rabbijn', 'imam', 'geestelijke', 'frater']
        },
    # Locations
    'settlement': {
        'types': ['location'],
        'schema_types': ['Settlement', 'Village', 'Municipality', 'Town',
            'AdministrativeRegion', 'City', 'HistoricPlace', 'PopulatedPlace',
            'ProtectedArea', 'CityDistrict', 'Country', 'MountainRange'],
        'words': ['gemeente', 'provincie', 'stad', 'dorp', 'regio', 'wijk',
            'gebied']
        },
    'infrastructure': {
        'types': ['location'],
        'schema_types': ['Building', 'Road', 'Station', 'RailwayStation',
            'Airport', 'HistoricBuilding', 'Bridge', 'Dam'],
        'words': ['station', 'metrostation', 'vliegveld', 'gebouw', 'brug']
        },
    'natural_location': {
        'types': ['location'],
        'schema_types': ['River', 'Mountain', 'Lake', 'CelestialBody',
            'Asteroid', 'Planet', 'Island'],
        'words': ['rivier', 'gebergte', 'meer', 'planeet', 'eiland']
        },
    # Organizations
    'company': {
        'types': [],
        'schema_types': ['Company'],
        'words': []
        },
    'school': {
        'types': [],
        'schema_types': ['School', 'University'],
        'words': []
        },
    'political_organisation': {
        'types': ['organisation'],
        'schema_types': ['PoliticalParty']
        },
    'sports_organisation': {
        'types': ['organisation'],
        'schema_types': ['SoccerClub']
        },
    'cultural_organisation': {
        'types': ['organisation'],
        'schema_types': ['Band', 'MusicGroup'],
        'words': ['band', 'gezelschap']
        },
    'military_organisation': {
        'types': [],
        'schema_types': ['MilitaryUnit'],
        'words': []
        },
    # Other
    'creative_work': {
        'types': [],
        'schema_types': ['CreativeWork', 'Film', 'Album', 'Single', 'Book',
            'TelevisionShow', 'TelevisionEpisode', 'Song', 'MusicalWork'],
        'words': []
        },
    'sports_event': {
        'types': [],
        'schema_types': ['OlympicEvent', 'SoccerTournament',
            'TennisTournament', 'FootballMatch'],
        'words': []
        },
    'military_event': {
        'types': [],
        'schema_types': ['MilitaryConflict'],
        'words': []
        }
    }

subjects = {
    'politics': {
        'roles': ['politician', 'royalty', 'military_person',
            'political_orginisation', 'military_organisation', 'military_event'],
        'words': ['regering', 'kabinet', 'fractie', 'partij',
            'tweede kamer', 'eerste kamer', 'politiek', 'politicus',
            'vorstenhuis', 'koningshuis', 'koninklijk huis', 'troon', 'rijk',
            'keizerrijk', 'monarchie', 'leger', 'oorlog', 'troepen', 'gevecht',
            'strijd', 'strijdkrachten']
    'sports': {
        'roles': ['athlete', 'sports_organisation', 'sports_event'],
        'words': ['sport', 'voetbal', 'wielersport', 'wedstrijd']
        },
    'culture': {
        'roles': ['artist', 'cultural_organisation', 'creative_work'],
        'words': ['kunst', 'cultuur', 'roman', 'boek', 'gedicht',
            'bundel', 'werk', 'schilderij', 'beeld', 'muziek',
            'toneel', 'theater', 'film']
        },
    'science': {
        'roles': ['scientist', 'school'],
        'words': ['wetenschap', 'wetenschapper', 'studie',
            'onderzoek', 'uitvinding', 'ontdekking']
        },
    'religion': {
        'roles': ['religious_person'],
        'words': ['kerk', 'parochie', 'geloof', 'religie']
        }
    }

