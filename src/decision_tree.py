from typing import Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


class DecisionTree:
    def __init__(self, dataset, random_state):
        self.dataset: Any = dataset
        self.random_state: int = random_state

    def run(self):
        pass
        features = self.dataset.data.features
        targets = self.dataset.data.targets

        print(f"num registros: {features.shape[0]}")
        print(f"num atributos: {features.shape[1]}")
        print(f"atributos: {features.columns}")

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
        encoded_features = encoder.fit_transform(features)

        features_train, features_test, target_train, target_test = train_test_split(
            encoded_features, targets, test_size=0.25, random_state=self.random_state)

        classifier = DecisionTreeClassifier(random_state=self.random_state)
        classifier.fit(features_train, target_train)

        prediction = classifier.predict(features_test)

        accuracy = accuracy_score(target_test, prediction)
        print(f"Decision Tree Accuracy: {accuracy:.2f}")
