from ucimlrepo import fetch_ucirepo

from src.decision_tree import DecisionTree
from src.args import ArgsParser
import src.consts as Consts


def main():
    args = ArgsParser()
    args.validate()

    dataset = fetch_ucirepo(id=Consts.UCI_REPO_ID)
    random_state = Consts.RANDOM_STATE

    if args.is_decision_tree():
        DecisionTree(dataset=dataset, random_state=random_state).run()
    elif args.is_random_forest():
        raise NotImplementedError('Random Forest is not implemented yet')
    elif args.is_perceptron():
        raise NotImplementedError('Perceptron is not implemented yet')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'Error: {e}')
