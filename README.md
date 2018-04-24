# DAC Entity Linker

Entity linker for the [Dutch historical newspaper collection](https://www.delpher.nl/nl/kranten) of the [Koninklijke Bibliotheek](https://www.kb.nl), National Library of the Netherlands. The linker links named entity mentions in newspaper articles to relevant DBpedia descriptions using either a binary SVM classifier or a neural net. For background information, please see the [project description](https://www.kb.nl/en/organisation/research-expertise/enrichment-of-digital-content) on the Koninklijke Bibliotheek website.

## Usage

Basic command line execution with the default values for all options:

```
$ cd dac
$ ./dac.py
```

This will link all recognized entities in a sample article using a neural network:

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

## Command line interface

Additional options when using the command line interface:

```
usage: dac.py [-h] [--url URL] [--ne NE] [-m MODEL] [-d] [-f] [-c] [-e]

optional arguments:
  -h, --help                  show this help message and exit
  --url URL                   resolver link of the article to be processed
  --ne NE                     specific named entity to be linked
  -m MODEL, --model MODEL     model used for link prediction (svm, nn or bnn)
  -d, --debug                 include unlinked entities in response
  -f, --features              return feature values
  -c, --candidates            return candidate list
  -e, --errh                  turn on error handling
```

## Web interface

The DAC Entity Linker can be started as a web application by running:

```
$ ./web.py
```

This starts a Bottle web server listening on `http://localhost:5002`. The URL parameters are similar to the command line options:

```
mandatory arguments:
  - url          resolver link of the article to be processed

optional arguments:
  - ne           specific named entity to be linked
  - model        model used for link prediction (svm, nn or bnn)
  - debug        include unlinked entities in response
  - features     include feature values for predicted links
  - candidates   include the list of candidates for each entity
  - callback     name of a JavaScript callback function
```

## Training new models

Given the availability of training set in the format created by the [DAC Web Interface](https://github.com/jlonij/dac-web), new models can be trained in two simple steps. First, the web interface training set is extended with the features values for each training example:

```
$ cd training
$ ./generate.py
```

The default input file used here is `../../../dac-web/users/tve/art.json` and the output is written to a `training.csv` file. These locations can be adjusted, however, using the `--input` and `--output` options of the `generate.py` script.

The resulting `training.csv` file can now be used to train new models. Note that existing models in the `models` directory will be replaced, so these need to be backed up manually if they are to be preserved. To train, for example, a new Support Vector Machine, run:

```
$ ./models.py -t -m svm
```

This will create a `models/svm.pkl` file that can now be applied to new named entity examples. 

Full command line options for training and cross-validation:

```
usage: models.py [-h] [-w] [-t] [-v] [-m MODEL]

optional arguments:
  -h, --help                  show this help message and exit
  -w, --weights               show the feature weights of the current model
  -t, --train                 train and save new model
  -v, --validate              cross-validate new model
  -m MODEL, --model MODEL     model type (svm, nn or bnn)
```

## Evaluation

Once one or more models have been trained, the linker performance can be evaluated on a separate training set in the format created by the [DAC Web Interface](https://github.com/jlonij/dac-web). To test the performance of, e.g., a first version of a neural net, run:

```
$ cd training
$ ./test.py -m nn -v 1
```

This will evaluate the current neural network model on the `../../../dac-web/users/test-clean/art.json` file, but a different test set can be specified with the `--input` option.

A summary of the results will be printed out:

```
Number of instances: 500
Number of correct predictions: 467
Prediction accuracy: 0.934
---
Number of correct link predictions: 347
(Min) number of link instances: 362
(Max) number of link instances: 382
(Min) link recall: 0.908376963351
(Mean) link recall: 0.933470249631
(Max) link recall: 0.958563535912
---
Number of correct link predictions: 347
Number of link predictions: 358
Link precision: 0.969273743017
---
(Mean) link F1-measure: 0.951035143299
(Max) link F1-measure: 0.963888888889
```

The version number specified will be used to name a file containing the full results of the test run, e.g. `training/results-nn-1.csv`.

Further command line options for the test script:

```
usage: test.py [-h] -m MODEL -v VERSION [-i INPUT]

mandatory arguments:
  -m MODEL, --model MODEL     model name (svm, nn or bnn)
  -v VERSION                  version number
  
optional arguments:
  -h, --help                  show this help message and exit
  -i INPUT                    path to test set
```
