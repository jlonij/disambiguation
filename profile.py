#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dac
import time

from pycallgraph import GlobbingFilter
from pycallgraph import PyCallGraph
from pycallgraph import Config
from pycallgraph.output import GraphvizOutput

config = Config(max_depth=5, verbose=True)
config.trace_filter = GlobbingFilter(exclude=['lxml.*', 'requests.*',
    'sklearn.*', 'pycallgraph.*'])
graphviz = GraphvizOutput(output_file='profile_' + str(int(time.time())) + '.png')

with PyCallGraph(output=graphviz, config=config):
    linker = dac.EntityLinker(model='svm', debug=True)
    linker.link('http://resolver.kb.nl/resolve?urn=ddd:010734861:mpeg21:a0002:ocr')

