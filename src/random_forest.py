import time
from typing import Any

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

import src.utils as Utils


class RandomForest:
    def __init__(self, dataset, random_state, tree_count):
        self.dataset: Any = dataset
        self.random_state: int = random_state
        self.tree_count = tree_count

    def run(self):
        features = self.dataset.data.features
        targets = self.dataset.data.targets
        # Necessário para converter o shape de (n, 1) para (n,) (matriz de uma coluna pra vetor)
        targets = targets.values.ravel()

        encoded_features = self._onehotencode(features)

        features_train, features_test, target_train, target_test = self._split(
            encoded_features, targets)

        classifier = self._make_classifier()
        self._print_classifier_parameters(classifier)

        self._train_classifier(classifier, features_train, target_train)

        prediction = self._predict(classifier, features_test)

        accuracy = accuracy_score(target_test, prediction)
        print(f"Precisão da Random Forest: {accuracy:.4f}")

    def _onehotencode(self, features):
        encoder = OneHotEncoder()
        before = time.time()
        encoded_features = encoder.fit_transform(features)
        after = time.time()
        duration = Utils.format_duration(before, after)
        print(f"tempo para encode: {duration}ms")
        return encoded_features

    def _split(self, encoded_features, targets):
        before = time.time()
        features_train, features_test, target_train, target_test = train_test_split(
            encoded_features, targets, test_size=0.3, random_state=self.random_state)
        after = time.time()
        duration = Utils.format_duration(before, after)
        print(f"tempo para split: {duration}ms")

        return features_train, features_test, target_train, target_test

    def _make_classifier(self):
        classifier = RandomForestClassifier(
            random_state=self.random_state, n_estimators=self.tree_count, criterion='gini')
        return classifier

    def _train_classifier(self, classifier, features_train, target_train):
        before = time.time()
        classifier.fit(features_train, target_train)
        after = time.time()
        duration = Utils.format_duration(before, after)
        print(
            f"tempo para treinar com {self.tree_count} árvores: {duration}ms")

    def _predict(self, classifier, features_test):
        before = time.time()
        prediction = classifier.predict(features_test)
        after = time.time()
        duration = Utils.format_duration(before, after)
        print(f"tempo para predict: {duration}ms")
        return prediction

    def _print_classifier_parameters(self, classifier):
        print('---------------------')
        print('parameters:')
        print('criterion: ', classifier.criterion)
        print('n_estimators: ', classifier.n_estimators)
        print('---------------------')
