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
        targets = targets.values.ravel()

        encoded_features = self._onehotencode(features)

        features_train, features_test, target_train, target_test = self._split(
            encoded_features, targets)

        features_train, features_test = self._normalize(
            features_train, features_test)

        classifier = self._train_classifier(features_train, target_train)

        prediction = self._predict(classifier, features_test)

        accuracy = accuracy_score(target_test, prediction)
        print(f"Precisão da Multilayer Perceptron: {accuracy:.2f}")

        overfitting_chance = self._measure_overfitting_chance(
            classifier, features_train, target_train, features_test, target_test)
        print(f"Chance de overfitting: {overfitting_chance:.2f}")

    def _measure_overfitting_chance(self, classifier, features_train, target_train, features_test, target_test):
        # TODO: confirmar com a professora se
        # isso faz sentido
        """ 
        Mede a acurácia dos dados de treinamento e dos
        dados de teste. se tiver uma acurácia alta no
        treinamento e baixa nos testes, então possívelmente
        houve overfitting
        """
        train_accuracy = accuracy_score(
            target_train, classifier.predict(features_train))
        target_accuracy = accuracy_score(
            target_test, classifier.predict(features_test))

        return abs(target_accuracy - train_accuracy)

    def _onehotencode(self, features):
        encoder = OneHotEncoder()
        before = time.time()
        encoded = encoder.fit_transform(features)
        after = time.time()
        duration = Utils.format_duration(before, after)
        print(f"tempo para encode: {duration}ms")
        return encoded

    def _split(self, encoded_features, targets):
        before = time.time()
        features_train, features_test, target_train, target_test = train_test_split(
            encoded_features, targets, test_size=0.3, random_state=self.random_state)
        after = time.time()
        duration = Utils.format_duration(before, after)
        print(f"tempo para split: {duration}ms")
        return features_train, features_test, target_train, target_test

    def _normalize(self, features_train, features_test):
        scaler = StandardScaler(with_mean=False)
        before = time.time()
        features_train = scaler.fit_transform(features_train)
        features_test = scaler.transform(features_test)
        after = time.time()
        duration = Utils.format_duration(before, after)
        print(f"tempo para normalizar: {duration}ms")

        return features_train, features_test

    def _train_classifier(self, features_train, target_train):
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

        return classifier

    def _predict(self, classifier, features_test):
        before = time.time()
        prediction = classifier.predict(features_test)
        after = time.time()
        duration = Utils.format_duration(before, after)
        print(f"tempo para predict: {duration}ms")
        return prediction
