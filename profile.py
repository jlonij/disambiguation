#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dac

from pycallgraph import PyCallGraph
from pycallgraph import Config
from pycallgraph.output import GraphvizOutput

config = Config(max_depth=5)
graphviz = GraphvizOutput(output_file='filter_max_depth_article.png')

with PyCallGraph(output=graphviz, config=config):
    linker = dac.EntityLinker(model='svm', debug=True, features=True,
                candidates=True)
    linker.link('http://resolver.kb.nl/resolve?urn=ddd:010734861:mpeg21:a0002:ocr')
