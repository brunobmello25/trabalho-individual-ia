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

        classifier = DecisionTreeClassifier(
            random_state=self.random_state, criterion='gini')
        print('---------------------')
        print('parameters:')
        print('criterion: ', classifier.criterion)
        print('max_depth: ', classifier.max_depth)
        print('min_samples_split: ', classifier.min_samples_split)
        print('min_samples_leaf: ', classifier.min_samples_leaf)
        print('max_features: ', classifier.max_features)
        print('class_weight: ', classifier.class_weight)
        print('---------------------')
        before = time.time()
        classifier.fit(features_train, target_train)
        after = time.time()
        duration = Utils.format_duration(before, after)
        print(f"tempo para treinar: {duration}ms")

        before = time.time()
        prediction = classifier.predict(features_test)
        after = time.time()
        duration = Utils.format_duration(before, after)
        print(f"tempo para predict: {duration}ms")

        accuracy = accuracy_score(target_test, prediction)
        print(f"Precisão da Decision Tree: {accuracy:.2f}")

        plt.figure(figsize=(80, 40))
        plot_tree(classifier, filled=True,
                  feature_names=encoder.get_feature_names_out(), class_names=True)
        plt.title("Árvore de Decisão")
        plt.savefig("decision_tree.png")  # Salva a imagem no formato PNG
        plt.close()
