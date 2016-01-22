import math
import xml.etree.ElementTree as etree


class LinearSVM:

    weights = []
    means = []
    variances = []
    bias = 0


    def __init__(self):

        # Load model parameters: weights, means, variances and bias
        model_file = "models/linear_svm.xml"
        tree = etree.parse(model_file)
        root = tree.getroot()

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
            # Scale example feature values
            value = (example[i] - self.means[i]) / math.sqrt(self.variances[i])
            # Add scaled value multiplied by feature weigth
            pred += self.weights[i] * value

        # Convert prediction into probability
        prob = 1 / (1 + math.exp(pred * -1))
        return prob


class RadialSVM:
    def __init__(self):
        stub = True
    def predict(self, example):
        stub = True

class DecisionTree:
    def __init__(self):
        stub = True

class NeuralNet:
    def __init__(self):
        stub = True

