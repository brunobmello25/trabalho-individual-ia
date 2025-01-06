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
        print(targets.shape)
        # Necessário para converter o shape de (n, 1) para (n,) (matriz de uma coluna pra vetor)
        targets = targets.values.ravel()
        print(targets.shape)

        encoder = OneHotEncoder()
        before = time.time()
        encoded_features = encoder.fit_transform(features)
        after = time.time()
        duration = Utils.format_duration(before, after)
        print(f"tempo para encode: {duration}ms")

        before = time.time()
        features_train, features_test, target_train, target_test = train_test_split(
            encoded_features, targets, test_size=0.3, random_state=self.random_state)
        after = time.time()
        duration = Utils.format_duration(before, after)
        print(f"tempo para split: {duration}ms")

        classifier = RandomForestClassifier(
            random_state=self.random_state, n_estimators=self.tree_count)
        before = time.time()
        classifier.fit(features_train, target_train)
        after = time.time()
        duration = Utils.format_duration(before, after)
        print(
            f"tempo para treinar com {self.tree_count} árvores: {duration}ms")

        before = time.time()
        prediction = classifier.predict(features_test)
        after = time.time()
        duration = Utils.format_duration(before, after)
        print(f"tempo para predict: {duration}ms")

        accuracy = accuracy_score(target_test, prediction)
        print(f"Precisão da Random Forest: {accuracy:.2f}")
