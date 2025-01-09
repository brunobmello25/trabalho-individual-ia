from typing import Any, List, Tuple
from ucimlrepo import fetch_ucirepo
from itertools import product

from src.decision_tree import DecisionTree
from src.random_forest import RandomForest
from src.multilayer_perceptron import MultilayerPerceptron
from src.args import ArgsParser
import src.consts as Consts


def generate_layer_configurations(max_neurons: int, max_layers: int, step: int = 10) -> List[Tuple[int, ...]]:
    neuron_counts = list(range(step, max_neurons + 1, step))

    configurations = []
    for num_layers in range(1, max_layers + 1):
        for neurons in neuron_counts:
            configuration = (neurons,) * num_layers
            configurations.append(configuration)

    return configurations


# Exemplo de uso:
configurations = generate_layer_configurations(
    max_neurons=350, max_layers=5, step=50)


def main():
    args = ArgsParser()
    args.validate()

    dataset: Any = fetch_ucirepo(id=Consts.UCI_REPO_ID)
    random_state = Consts.RANDOM_STATE

    if args.is_decision_tree():
        DecisionTree(dataset=dataset, random_state=random_state).run()
    elif args.is_random_forest():
        RandomForest(dataset=dataset, random_state=random_state,
                     tree_count=Consts.TREE_COUNT).run()
    elif args.is_perceptron():
        MultilayerPerceptron(
            dataset=dataset, random_state=random_state, max_iter=Consts.MAX_PERCEPTRON_ITER, hidden_layer_configurations=configurations).run()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'Error: {e}')
