days = ['maandag', 'dinsdag', 'woensdag', 'donderdag', 'vrijdag', 'zaterdag',
'zondag']

months = ['januari', 'februari', 'maart', 'april', 'mei', 'juni', 'juli',
'augustus', 'september', 'oktober', 'november', 'december']

titles = ['drs', 'ing', 'ir', 'dr']

genders = {
    'male': ['heer', 'hr', 'dhr', 'meneer'],
    'female': ['mevrouw', 'mevr', 'mw', 'mej', 'mejuffrouw']
    }

types = {
    'person': {
        'schema_types': ['Person', 'Agent'],
        'words': ['familie', 'zoon', 'dochter', 'persoon', 'geboren',
            'overleden', 'dood', 'man', 'vrouw', 'leven']
        },
    'location': {
        'schema_types': ['Place', 'Location'],
        'words': ['inwoners', 'oppervlakte', 'gelegen', 'noorden', 'zuiden',
            'oosten', 'westen', 'noordoosten', 'noordwesten', 'zuidoosten',
            'zuidwesten', 'grens', 'eilanden', 'eilandengroep']
        },
    'organisation': {
        'schema_types': ['Organization', 'Organisation'],
        'words': ['organisatie']
        }
    }

roles = {
    # Persons
    'politician': {
        'words': ['minister', 'premier', 'kamerlid', 'partijleider',
            'burgemeester', 'staatssecretaris', 'president',
            'wethouder', 'consul', 'ambassadeur', 'gemeenteraadslid',
            'fractieleider', 'politicus', 'politica', 'staatsman'],
        'schema_types': ['Politician', 'OfficeHolder', 'Judge',
            'MemberOfParliament', 'President', 'PrimeMinister',
            'Governor', 'Congressman', 'Mayor'],
        'subjects': ['politics'],
        'types': ['person']
        },
    'royalty': {
        'words': ['keizer', 'koning', 'koningin', 'vorst', 'prins',
            'prinses', 'kroonprins', 'kroonprinses', 'majesteit'],
        'schema_types': ['Royalty', 'Monarch', 'Noble'],
        'subjects': ['politics'],
        'types': ['person']
        },
    'military_person': {
        'words': ['generaal', 'gen', 'majoor', 'maj', 'luitenant',
            'kolonel', 'kol', 'kapitein', 'bevelhebber'],
        'schema_types': ['MilitaryPerson'],
        'subjects': ['politics'],
        'types': ['person']
        },
    'sports_person': {
        'words': ['atleet', 'sportman', 'sportvrouw', 'sporter',
            'wielrenner', 'voetballer', 'tennisser', 'zwemmer', 'spits',
            'keeper', 'scheidsrechter', 'profvoetballer', 'schaatser', 'spits',
            'langebaanschaatser', 'schaker'],
        'schema_types': ['Athlete', 'SoccerPlayer', 'Cyclist', 'SoccerManager',
            'TennisPlayer', 'Swimmer', 'Boxer', 'Wrestler', 'Speedskater',
            'Skier', 'WinterSportPlayer', 'GolfPlayer', 'RacingDriver',
            'MotorsportRacer', 'Canoist', 'Cricketer', 'RugbyPlayer',
            'HorseRider', 'AmericanFootballPlayer', 'Rower', 'MotorcycleRider',
            'Skater', 'BaseballPlayer', 'BasketballPlayer', 'Gymnast',
            'SportsManager', 'IceHockeyPlayer', 'FigureSkater', 'HandballPlayer'],
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
             'trompetspeler', 'orkestleider', 'gitarist', 'pianist'],
        'schema_types': ['MusicalArtist', 'ClassicalMusicArtist'],
        'subjects': ['culture'],
        'types': ['person']
        },
    'visual_artist': {
         'words': ['kunstenaar', 'schilder', 'beeldhouwer', 'architect',
            'fotograaf', 'ontwerper', 'kunstschilder', 'striptekenaar',
            'illustrator'],
        'schema_types': ['Painter', 'Architect', 'Photographer',
            'FashionDesigner'],
        'subjects': ['culture'],
        'types': ['person']
        },
    'writer': {
        'words': ['auteur', 'schrijver', 'dichter', 'journalist'],
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
        'words': ['manager', 'teamleider', 'directeur', 'bedrijfsleider', 'ondernemer'],
        'schema_types': ['BusinessPerson'],
        'subjects': ['business'],
        'types': ['person']
        },
    'scientist': {
        'words': ['prof', 'professor', 'natuurkundige', 'scheikundige',
            'wiskundige', 'bioloog', 'historicus', 'onderzoeker', 'wetenschapper',
            'filosoof', 'docent'],
        'schema_types': ['Scientist', 'Historian', 'Biologist', 'Philosopher',
            'Professor'],
        'subjects': ['science'],
        'types': ['person']
        },
    'religious_person': {
        'words': ['dominee', 'paus', 'kardinaal', 'aartsbisschop',
            'bisschop', 'monseigneur', 'mgr', 'kapelaan', 'deken',
            'abt', 'prior', 'pastoor', 'pater', 'predikant',
            'opperrabbijn', 'rabbijn', 'imam', 'geestelijke', 'frater'],
        'schema_types': ['ChristianBishop', 'Cardinal', 'Cleric', 'Saint', 'Pope'],
        'subjects': ['religion'],
        'types': ['person']
        },
    # Locations
    'settlement': {
        'words': ['plaats', 'gemeente', 'provincie', 'stad', 'dorp', 'regio', 'wijk',
            'gebied', 'stadsdeel', 'waterschap', 'straat', 'staat', 'district',
            'deelstaat', 'departement', 'county', 'kanton', 'hoofdstad',
            'graafschap', 'land', 'arrondissement', 'streek', 'landgoed'],
        'schema_types': ['Settlement', 'Village', 'Municipality', 'Town',
            'AdministrativeRegion', 'City', 'HistoricPlace', 'PopulatedPlace',
            'ProtectedArea', 'CityDistrict', 'Country', 'SubMunicipality', 'Street'
            'District'],
        'subjects': [],
        'types': ['location']
        },
    'infrastructure': {
        'words': ['station', 'metrostation', 'vliegveld', 'gebouw', 'brug',
            'monument', 'metro', 'luchthaven'],
        'schema_types': ['Building', 'Road', 'Station', 'RailwayStation',
            'Airport', 'HistoricBuilding', 'Bridge', 'Dam', 'ArchitecturalStructure',
            'Monument', 'Castle', 'WorldHeritageSite', 'MetroStation'],
        'subjects': [],
        'types': ['location']
        },
    'natural_location': {
        'words': ['rivier', 'gebergte', 'meer', 'planeet', 'eiland'],
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
        'words': ['bisdom', 'kerk', 'kathedraal', 'tempel', 'kapel', 'heiligdom'],
        'schema_types': ['Church', 'ReligiousBuilding', 'Diocese'],
        'subjects': ['religion'],
        'types': ['location', 'organisation']
        },
    # Organizations
    'company': {
        'words': ['bedrijf', 'bank', 'luchtvaartmaatschappij', 'onderneming',
            'hotel', 'uitvoerder'],
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
        'words': ['museum', 'band', 'rockband', 'popgroep', 'orkest'],
        'schema_types': ['Band', 'MusicGroup', 'RecordLabel', 'Museum'],
        'subjects': ['culture'],
        'types': ['organisation']
        },
    'military_organisation': {
        'words': [],
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
        'words': ['film', 'album', 'plaat', 'nummer', 'single', 'boek', 'roman',
            'novelle', 'bundel', 'dichtbundel', 'script', 'serie', 'televisieserie',
            'opera', 'toneelstuk', 'gedicht', 'schilderij', 'beeld', 'strip'],
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
        'words': ['wedstrijd', 'voetbalcompetitie', 'toernooi'],
        'schema_types': ['SoccerLeague', 'OlympicEvent', 'SoccerTournament', 'GrandPrix'
            'TennisTournament', 'FootballMatch', 'CyclingRace', 'SportsEvent'],
        'subjects': ['sports'],
        'types': []
        },
    'military_event': {
        'words': ['oorlog'],
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
        'strijdkrachten', 'militair', 'partij', 'ministers', 'politie'],
    'sports': ['sport', 'voetbal', 'wielersport', 'speler', 'spelers',
        'sporten', 'wedstrijden'],
    'culture': ['kunst', 'cultuur', 'muziek', 'toneel', 'theater', 'cinema',
        'romans', 'verhalen', 'schrijvers', 'fictief', 'fictieve', 'rock',
        'jazz', 'geregisseerd', 'strips', 'kunstveiling', 'kunstveilingen',
        'piano', 'gitaar', 'pianisten', 'recital'],
    'business': ['economie', 'beurs', 'aandelen', 'bedrijfsleven',
        'management', 'werknemer', 'werknemers', 'salaris', 'staking',
        'personeel', 'beleggers'],
    'science': ['wetenschap', 'studie', 'onderzoek', 'uitvinding', 'ontdekking'],
    'religion': ['geloof', 'religie']
    }
