import time
from typing import Any, List, Tuple
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
import src.utils as Utils


class MultilayerPerceptron:
    def __init__(self, dataset, random_state, max_iter):
        self.dataset: Any = dataset
        self.random_state: int = random_state
        self.max_iter = max_iter
        self.hidden_layer_configurations = self.generate_layer_configurations(
            max_neurons=350, max_layers=5, step=50)
        self.best_accuracy = -1
        self.best_f1_score = -1
        self.best_accuracy_config = None
        self.best_f1_score_config = None

    def generate_layer_configurations(self, max_neurons: int, max_layers: int, step: int = 10) -> List[Tuple[int, ...]]:
        neuron_counts = list(range(step, max_neurons + 1, step))

        configurations = []
        for num_layers in range(1, max_layers + 1):
            for neurons in neuron_counts:
                configuration = (neurons,) * num_layers
                configurations.append(configuration)

        return configurations

    def run(self):
        features = self.dataset.data.features
        targets = self.dataset.data.targets
        targets = targets.values.ravel()
        encoded_features = self._onehotencode(features)
        features_train, features_test, target_train, target_test = self._split(
            encoded_features, targets)

        # Testar diferentes configurações de camadas ocultas
        results = []
        i = 0
        for hidden_layers in self.hidden_layer_configurations:
            print(
                f"Testando configuração {i+1}/{len(self.hidden_layer_configurations)} de camadas ocultas: {hidden_layers}")
            i += 1
            classifier = self._make_classifier(hidden_layers)
            accuracy, f1 = self._train_and_evaluate(
                classifier, features_train, target_train, features_test, target_test)
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_accuracy_config = hidden_layers
            if f1 > self.best_f1_score:
                self.best_f1_score = f1
                self.best_f1_score_config = hidden_layers
            results.append((hidden_layers, accuracy, f1))
            print('--------------------')

        # Exibir resultados
        print("\nResultados:")
        for config, accuracy, f1 in results:
            print(
                f"Camadas ocultas: {config}, Precisão: {accuracy:.4f}, F1-score: {f1:.4f}")
        print('Melhor configuração pela acurácia: ', self.best_accuracy_config)
        print('Melhor acurácia: ', self.best_accuracy)
        print('Melhor configuração pela F1-score: ', self.best_f1_score_config)
        print('Melhor F1-score: ', self.best_f1_score)

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

    def _make_classifier(self, hidden_layer_sizes):
        classifier = MLPClassifier(
            random_state=self.random_state,
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=self.max_iter
        )
        return classifier

    def _train_and_evaluate(self, classifier, features_train, target_train, features_test, target_test):
        self._train_classifier(classifier, features_train, target_train)
        prediction = self._predict(classifier, features_test)
        accuracy = accuracy_score(target_test, prediction)
        f1 = f1_score(target_test, prediction, average='weighted')
        return accuracy, f1

    def _train_classifier(self, classifier, features_train, target_train):
        before = time.time()
        classifier.fit(features_train, target_train)
        after = time.time()
        duration = Utils.format_duration(before, after)

    def _predict(self, classifier, features_test):
        before = time.time()
        prediction = classifier.predict(features_test)
        after = time.time()
        duration = Utils.format_duration(before, after)
        return prediction
