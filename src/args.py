import sys

MULTI_PERCEPTRON = '--multi-perceptron'
DECISION_TREE = '--decision-tree'
RANDOM_FOREST = '--random-forest'


class ArgsParser:
    def __init__(self):
        self.args = sys.argv[1:]

    def is_random_forest(self):
        return RANDOM_FOREST in self.args

    def is_decision_tree(self):
        return DECISION_TREE in self.args

    def is_perceptron(self):
        return MULTI_PERCEPTRON in self.args

    def validate(self):
        methods = sum([self.is_decision_tree(),
                      self.is_random_forest(), self.is_perceptron()])
        if methods != 1:
            raise ValueError(
                f'Você precisa especificar um dos três métodos: {DECISION_TREE}, {RANDOM_FOREST} ou {MULTI_PERCEPTRON}')
