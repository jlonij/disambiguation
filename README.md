# DAC Entity Linker

Entity linker for the Dutch historical newspaper collection of the National Library of the Netherlands. The linker links named entity mentions to DBpedia descriptions using either a binary SVM classifier or a neural net.

## Usage

Basic command line execution with the default values for all options:

```
$ ./dac.py
```

This will link all recognized entities in an example article using a neural network:

```
{'linkedNEs': [{'label': u'Winston Churchill',
                'link': u'http://nl.dbpedia.org/resource/Winston_Churchill',
                'prob': '0.9997673631',
                'reason': 'Predicted link',
                'text': u'Churchill'},
               {'label': u'Willem Drees',
                'link': u'http://nl.dbpedia.org/resource/Willem_Drees',
                'prob': '0.9968996048',
                'reason': 'Predicted link',
                'text': u'Drees'},
                ...
```
