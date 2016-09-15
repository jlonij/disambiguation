#!/usr/bin/env python                                                                                |
# -*- coding: utf-8 -*-

import os
import csv
import math

import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as etree


class LinearSVM:

    features = []
    weights = []
    means = []
    variances = []
    bias = 0


    def __init__(self):

        model_file = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-1]) + os.sep
        model_file += "models" + os.sep + "linear_svm.mod"
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
        model_file += "models" + os.sep + "radial_svm.mod"
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


class NeuralNet:

    def __init__(self):

        self.features = [
                'solr_pos',
                'cand_pos',
                'solr_score',
                'cand_score',
                'solr_inlinks',
                'cand_inlinks',
                'quotes',
                'lang',
                'disambig',
                'main_title_exact_match',
                'main_title_start_match',
                'main_title_end_match',
                'main_title_match',
                'title_exact_match_fraction',
                'title_start_match_fraction',
                'title_end_match_fraction',
                'title_match_fraction',
                'last_part_match_fraction',
                'mean_levenshtein_ratio',
                'name_conflict',
                'date_match',
                'type_match',
                'role_match',
                'subject_match',
                'entity_match',
                'spec_match'
                ]


    def predict(self, example):

        num_hidden_nodes1 = 26
        num_hidden_nodes2 = 26
        num_features = 26
        num_labels = 2

        curr_dir = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-1]) + os.sep
        model_file = curr_dir + "models" + os.sep + "model.ckpt"

        graph = tf.Graph()

        with tf.Session(graph=graph) as session:

            global_step = tf.Variable(0)

            # Weights and biases for each network layer
            weights1 = tf.Variable(tf.truncated_normal([num_features,
                    num_hidden_nodes1], stddev=np.sqrt(2.0 / num_features)))
            biases1 = tf.Variable(tf.zeros([num_hidden_nodes1]))
            weights2 = tf.Variable(tf.truncated_normal([num_hidden_nodes1,
                    num_hidden_nodes2], stddev=np.sqrt(2.0 / num_hidden_nodes1)))
            biases2 = tf.Variable(tf.zeros([num_hidden_nodes2]))
            weights3 = tf.Variable(tf.truncated_normal([num_hidden_nodes2,
                    num_labels], stddev=np.sqrt(2.0 / num_hidden_nodes2)))
            biases3 = tf.Variable(tf.zeros([num_labels]))

            # Saver
            self.saver = tf.train.Saver()
            self.saver.restore(session, model_file)

            # Prediction for new examples
            x = tf.placeholder(tf.float32, shape=(1, num_features))
            lay1_y = tf.nn.relu(tf.matmul(x, weights1) + biases1)
            lay2_y = tf.nn.relu(tf.matmul(lay1_y, weights2) + biases2)
            y = tf.nn.softmax(tf.matmul(lay2_y, weights3) + biases3)

            ex = np.ndarray(shape=(1, len(example)), dtype=np.float32)
            ex[0] = example

            return float(session.run(y, feed_dict={x: ex})[0,1])


if __name__ == '__main__':
    model = RadialSVM()
    model.predict([0,0,0,0,0,0,0,0,0,1,0,1,0.180])


