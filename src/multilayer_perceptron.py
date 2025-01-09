import time
from typing import Any, List, Tuple
from sklearn.metrics import accuracy_score
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
            accuracy, overfitting_chance = self._train_and_evaluate(
                classifier, features_train, target_train, features_test, target_test)
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_configuration = hidden_layers
            results.append((hidden_layers, accuracy, overfitting_chance))
            print('--------------------')

        # Exibir resultados
        print("\nResultados:")
        for config, accuracy, overfitting in results:
            print(
                f"Camadas ocultas: {config}, Precisão: {accuracy:.4f}, Chance de Overfitting: {overfitting:.2f}")
        print('melhor configuração: ', self.best_configuration)
        print('melhor acurácia: ', self.best_accuracy)

    def _measure_overfitting_chance(self, classifier, features_train, target_train, features_test, target_test):
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
        # print(f"Precisão da Multilayer Perceptron: {accuracy:.4f}")
        overfitting_chance = self._measure_overfitting_chance(
            classifier, features_train, target_train, features_test, target_test)
        # print(f"Chance de overfitting: {overfitting_chance:.2f}")
        return accuracy, overfitting_chance

    def _train_classifier(self, classifier, features_train, target_train):
        before = time.time()
        classifier.fit(features_train, target_train)
        after = time.time()
        duration = Utils.format_duration(before, after)
        # print(
        #     f"tempo para treinar com layers {classifier.hidden_layer_sizes}: {duration}ms")

    def _predict(self, classifier, features_test):
        before = time.time()
        prediction = classifier.predict(features_test)
        after = time.time()
        duration = Utils.format_duration(before, after)
        # print(f"tempo para predict: {duration}ms")
        return prediction


# Exemplo de como inicializar e executar
hidden_layer_configurations = [
    (10,),
    (50,),
    (100,),
    (50, 50),
    (100, 50),
    (50, 100),
    (100, 100)
]

# Você deve fornecer o dataset apropriado ao MultilayerPerceptron
# mlp = MultilayerPerceptron(dataset, random_state=42, max_iter=200, hidden_layer_configurations=hidden_layer_configurations)
# mlp.run()
