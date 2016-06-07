days = ['maandag', 'dinsdag', 'woensdag', 'donderdag', 'vrijdag', 'zaterdag',
'zondag']

months = ['januari', 'februari', 'maart', 'april', 'mei', 'juni', 'juli',
'augustus', 'september', 'oktober', 'november', 'december']

genders = {
    'male': ['heer', 'hr', 'dhr', 'meneer'],
    'female': ['mevrouw', 'mevr', 'mw', 'mej', 'mejuffrouw']
    }

types = {
    'person': {
        'schema_types': ['Person'],
        'subtypes': {
            'politician': {
                'schema_types': ['Politician', 'OfficeHolder'],
                'words': ['minister', 'premier', 'burgemeester']
                }
            },
            'soccer_player': {

            }
        },
    'location': 'Place',
    'organisation': 'Organization'
    }

types_subtypes = {
    'person': ['politicus', 'auteur'],
    'location': ['gemeente', 'provincie'],
    'organisation': ['bedrijf']
    }

subtypes = {
    'politicus': {
        'schema_types' : ['Politician', 'OfficeHolder'],
        'signals' : ['minister', 'premier', 'burgemeester']
        }
    }

subjects = {
    'politics': {
        'signal_words' : ['politiek', 'regering'],
        'subtypes' : ['politicus']
        }
    }

