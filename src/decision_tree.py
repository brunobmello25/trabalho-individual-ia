import time
from typing import Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import src.utils as Utils


class DecisionTree:
    def __init__(self, dataset, random_state):
        self.dataset: Any = dataset
        self.random_state: int = random_state

    def run(self):
        features = self.dataset.data.features
        targets = self.dataset.data.targets

        """
        OneHot encoding é o procedimento responsável por converter
        uma matriz de variáveis textuais em variáveis binárias. Então,
        se temos uma Variável e três registros cada um com uma característica,
        ficamos com três colunas "Característica 1", "Característica 2" e "Característica 3",
        booleanas. Em seguida, o primeiro registro tem a característica 1, porém não tem a 2 nem 3,
        o segundo tem a 2, mas não tem a 1 nem 3 e o terceiro tem a 3, mas
        não tem a 1 nem 2.
        Esse processo gera uma coluna nova para cada variável de
        cada feature do dataset original. Como a matriz resultante é
        composta por uma grande maioria de zeros, o encoder gera uma
        matriz esparsa, que grava apenas os valores non zero e suas posições.
        O scikit já está preparado para lidar com matrizes esparsas, então
        não precisamos fazer nada a respeito disso.
        """
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

        classifier = DecisionTreeClassifier(random_state=self.random_state)
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
