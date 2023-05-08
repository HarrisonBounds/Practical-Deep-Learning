#I will be using the github repo "Micrograd" from Andrej Karpathy to illustrate and understand backpropogation

#We need a data structure to maintain the mathematical expressions that make up a Neural Network

class Value:

    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return f"Value(data={self.data}"