#!/usr/bin/env python                                                                                |
# -*- coding: utf-8 -*-

import os
import csv
import math
import xml.etree.ElementTree as etree


class LinearSVM:

    features = []
    weights = []
    means = []
    variances = []
    bias = 0


    def __init__(self):

        model_file = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-1]) + os.sep
        model_file += "linear_svm.mod"
        tree = etree.parse(model_file)
        root = tree.getroot()

        features = []
        nodes = root.findall(".//attributeConstructions/string")
        for n in nodes:
            features.append(n.text)
        self.features = features

        weights = []
        nodes = root.findall(".//weights/double")
        for n in nodes:
            weights.append(float(n.text))
        self.weights = weights

        means = []
        nodes = root.findall(".//meanVarianceMap//mean")
        for n in nodes:
            means.append(float(n.text))
        self.means = means

        variances = []
        nodes = root.findall(".//meanVarianceMap//variance")
        for n in nodes:
            variances.append(float(n.text))
        self.variances = variances

        nodes = root.findall(".//b")
        self.bias = float(nodes[0].text)


    def predict(self, example):

        # Start with bias
        pred = self.bias

        for i in range(len(example)):
            # Scale example feature values if means and variances are available
            if len(self.means) == len(self.weights) and len(self.variances) == len(self.weights):
                value = (example[i] - self.means[i]) / math.sqrt(self.variances[i])
            else:
                value = example[i]
            # Add scaled value multiplied by feature weigth
            pred += self.weights[i] * value

        # Convert prediction into probability
        prob = 1 / (1 + math.exp(pred * -1))
        return prob


class RadialSVM:

    features = []
    examples = []
    alphas = []
    gamma = 0
    bias = 0


    def __init__(self):

        model_file = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-1]) + os.sep
        model_file += "radial_svm.mod"
        tree = etree.parse(model_file)
        root = tree.getroot()

        nodes = root.findall(".//b")
        self.bias = float(nodes[0].text)

        nodes = root.findall(".//gamma")
        self.gamma = float(nodes[0].text)

        alphas = []
        nodes = root.findall(".//alphas/double")
        for n in nodes:
            alphas.append(float(n.text))
        self.alphas = alphas

        features = []
        nodes = root.findall(".//attributeConstructions/string")
        for n in nodes:
            features.append(n.text)
        self.features = features

        examples = []
        att_nodes = root.findall(".//the__examples/atts/double-array")
        pos_nodes = root.findall(".//the__examples/index/int-array")

        for i in range(len(pos_nodes)):
            example = [0] * len(self.features)
            for j in range(len(pos_nodes[i])):
                example[int(pos_nodes[i][j].text)] = float(att_nodes[i][j].text)
            examples.append(example)
        self.examples = examples


    def predict(self, example):

        # Start with bias
        pred = self.bias

        # Add kernel value for each training example with non-zero alpha
        for i in range(len(self.examples)):
            if self.alphas[i] != 0:
                pred += self.alphas[i] * self.kernel_value(example, self.examples[i])

        # Convert function value to confidence value for positive class (i.e. link)
        prob = 1 / (1 + math.exp(pred * -1))
        return prob


    def kernel_value(self, x, y):
        result = 0
        for i in range(len(x)):
            tmp = x[i] - y[i]
            result += tmp * tmp
        result = math.exp(self.gamma * result)
        return result


class DecisionTree:
    def __init__(self):
        stub = True


class NeuralNet:
    def __init__(self):
        stub = True


if __name__ == '__main__':
    model = RadialSVM()
    model.predict([0,0,0,0,0,0,0,0,0,1,0,1,0.180])


