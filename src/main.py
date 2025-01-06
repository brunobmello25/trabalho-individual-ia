from typing import Any
from ucimlrepo import fetch_ucirepo

from src.decision_tree import DecisionTree
from src.random_forest import RandomForest
from src.multilayer_perceptron import MultilayerPerceptron
from src.args import ArgsParser
import src.consts as Consts


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
            dataset=dataset, random_state=random_state, max_iter=Consts.MAX_PERCEPTRON_ITER, hidden_layer_sizes=Consts.PERCEPTRON_HIDDEN_LAYER_SIZES).run()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'Error: {e}')
