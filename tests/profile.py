#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import time

sys.path.insert(0, '../dac')
import dac

from pycallgraph import GlobbingFilter
from pycallgraph import PyCallGraph
from pycallgraph import Config
from pycallgraph.output import GraphvizOutput

config = Config(max_depth=10, verbose=True)
config.trace_filter = GlobbingFilter(exclude=['lxml.*', 'requests.*',
    'sklearn.*', 'pycallgraph.*'], include=['dac.EntityLinker.link',
    'dac.Context.get_metadata', 'dac.Context.get_entities',
    'dac.Context.get_topics', 'dac.Description.get_vectors',
    'dac.CandidateList.query_solr', 'dac.Entity.suggest'])
graphviz = GraphvizOutput(output_file='profile_' + str(int(time.time())) + '.png')

with PyCallGraph(output=graphviz, config=config):
    linker = dac.EntityLinker(model='svm', debug=True)
    linker.link('http://resolver.kb.nl/resolve?urn=ddd:010734861:mpeg21:a0002:ocr')
    linker.link('http://resolver.kb.nl/resolve?urn=ddd:010616555:mpeg21:a0126:ocr')
    linker.link('http://resolver.kb.nl/resolve?urn=ddd:110577489:mpeg21:a0193:ocr')
    linker.link('http://resolver.kb.nl/resolve?urn=ddd:010620323:mpeg21:a0248:ocr')
    linker.link('http://resolver.kb.nl/resolve?urn=ddd:010369397:mpeg21:a0040:ocr')

