from typing import Any
from ucimlrepo import fetch_ucirepo

from src.decision_tree import DecisionTree
from src.random_forest import RandomForest
from src.args import ArgsParser
import src.consts as Consts
import src.utils as Utils


def main():
    args = ArgsParser()
    args.validate()

    dataset: Any = fetch_ucirepo(id=Consts.UCI_REPO_ID)
    random_state = Consts.RANDOM_STATE

    # features = dataset.data.features
    # Utils.print_features_info(features)

    if args.is_decision_tree():
        DecisionTree(dataset=dataset, random_state=random_state).run()
    elif args.is_random_forest():
        RandomForest(dataset=dataset, random_state=random_state,
                     tree_count=Consts.TREE_COUNT).run()
    elif args.is_perceptron():
        raise NotImplementedError('Perceptron is not implemented yet')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'Error: {e}')
