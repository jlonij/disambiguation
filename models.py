import math


class LinearSVM:

    weights = [
            0.1487593872800257,
            0.13678282646369272,
            0.11636411923890391,
            -0.966440513241986,
            0.8137087548911804,
            -0.7159666452623034,
            0.8869852160033139,
            0.3606624046577357,
            0.3148134944056568,
            0.26862793705631166,
            0.7568306670272671,
            0.7140937509970398,
            -0.1939577714186576
            ]

    means = [
            0.06723053567555845,
            0.1676425937974409,
            0.3142485361093038,
            0.500542181739319,
            0.15744957709824334,
            1.0707004988072002,
            1.1229668184775536,
            2.6389069616135328,
            0.08761656907395358,
            0.803079592279332,
            0.006289308176100629,
            0.23096942094990242,
            0.4273316096314469
            ]

    variances = [
            0.06271059074833592,
            0.13953855454230712,
            0.21549639366246337,
            0.24999970603896154,
            0.35690557515217425,
            6.315105538362362,
            4.719426981982917,
            14.068484079611046,
            0.0799399058976627,
            0.15814276074379385,
            0.017960878348057492,
            0.17762254753596918,
            0.03662715134103564
            ]

    bias = -1.4071898873590305

    def __init__(self):

        # Load model parameters: weights, means, variances and bias
        stub = True

    def predict(self, example):
        
        # Start with bias
        pred = self.bias

        for i in range(len(example)):
            # Scale example feature values
            value = (example[i] - float(self.means[i])) / math.sqrt(float(self.variances[i]))
            # Add scaled value multiplied by feature weigth
            pred += float(self.weights[i]) * value

        # Convert prediction into probability
        prob = 1 / (1 + math.exp(pred * -1))
        return prob


class RadialSVM:
    def __init__(self):
        stub = True

class DecisionTree:
    def __init__(self):
        stub = True

class NeuralNet:
    def __init__(self):
        stub = True

