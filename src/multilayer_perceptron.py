import time
from typing import Any

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier

import src.utils as Utils


class MultilayerPerceptron:
    def __init__(self, dataset, random_state, max_iter, hidden_layer_sizes):
        self.dataset: Any = dataset
        self.random_state: int = random_state
        self.max_iter = max_iter
        self.hidden_layer_sizes = hidden_layer_sizes

    def run(self):
        features = self.dataset.data.features
        targets = self.dataset.data.targets
        print(targets.shape)
        targets = targets.values.ravel()
        print(targets.shape)

        Utils.print_features_info(features)

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

        scaler = StandardScaler(with_mean=False)
        before = time.time()
        features_train = scaler.fit_transform(features_train)
        features_test = scaler.transform(features_test)
        after = time.time()
        duration = Utils.format_duration(before, after)
        print(f"tempo para normalizar: {duration}ms")

        classifier = MLPClassifier(
            random_state=self.random_state,
            hidden_layer_sizes=self.hidden_layer_sizes,
            # Necessário pra garantir que o treinamento termine
            max_iter=self.max_iter
        )
        before = time.time()
        classifier.fit(features_train, target_train)
        after = time.time()
        duration = Utils.format_duration(before, after)
        print(
            f"tempo para treinar com layers {self.hidden_layer_sizes}: {duration}ms")

        before = time.time()
        prediction = classifier.predict(features_test)
        after = time.time()
        duration = Utils.format_duration(before, after)
        print(f"tempo para predict: {duration}ms")

        accuracy = accuracy_score(target_test, prediction)
        print(f"Precisão da Multilayer Perceptron: {accuracy:.2f}")
