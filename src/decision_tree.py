import time
from typing import Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

import src.utils as Utils


class DecisionTree:
    def __init__(self, dataset, random_state):
        self.dataset: Any = dataset
        self.random_state: int = random_state

    def run(self):
        features = self.dataset.data.features
        targets = self.dataset.data.targets

        encoded_features, encoder = self._onehotencode(features)

        features_train, features_test, target_train, target_test = self._split(
            encoded_features, targets)

        classifier = self._make_classifier()

        # self._print_classifier_parameters(classifier)

        self._train_classifier(classifier, features_train, target_train)

        prediction = self._predict(classifier, features_test)

        accuracy = accuracy_score(target_test, prediction)
        print(f"Precisão da Decision Tree: {accuracy:.2f}")

        self._plot_tree(classifier, encoder)

    def _onehotencode(self, features):
        encoder = OneHotEncoder()
        before = time.time()
        encoded = encoder.fit_transform(features)
        after = time.time()
        duration = Utils.format_duration(before, after)
        print(f"tempo para encode: {duration}ms")
        return encoded, encoder

    def _split(self, encoded_features, targets):
        before = time.time()
        features_train, features_test, target_train, target_test = train_test_split(
            encoded_features, targets, test_size=0.3, random_state=self.random_state)
        after = time.time()
        duration = Utils.format_duration(before, after)
        print(f"tempo para split: {duration}ms")
        return features_train, features_test, target_train, target_test

    def _make_classifier(self):
        classifier = DecisionTreeClassifier(
            random_state=self.random_state, criterion='gini', max_depth=10)
        return classifier

    def _print_classifier_parameters(self, classifier):
        print('---------------------')
        print('parameters:')
        print('criterion: ', classifier.criterion)
        print('max_depth: ', classifier.max_depth)
        print('---------------------')

    def _train_classifier(self, classifier, features_train, target_train):
        before = time.time()
        classifier.fit(features_train, target_train)
        after = time.time()
        duration = Utils.format_duration(before, after)
        print(f"tempo para treinar: {duration}ms")

    def _predict(self, classifier, features_test):
        before = time.time()
        prediction = classifier.predict(features_test)
        after = time.time()
        duration = Utils.format_duration(before, after)
        print(f"tempo para predict: {duration}ms")
        return prediction

    def _plot_tree(self, classifier, encoder):
        plt.figure(figsize=(80, 40))
        plot_tree(classifier, filled=True,
                  feature_names=encoder.get_feature_names_out(), class_names=True)
        plt.title("Árvore de Decisão")
        plt.savefig("decision_tree.png")
        plt.close()
