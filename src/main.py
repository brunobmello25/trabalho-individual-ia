from typing import Any
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def print_information(mushroom):
    X = mushroom.data.features

    print("Metadata:")
    print(mushroom.metadata)
    print()

    print("Variables:")
    print(mushroom.variables)
    print()

    print("Feature names from dataframe X:")
    print(X.columns)


def main():
    mushroom: Any = fetch_ucirepo(id=848)

    X = mushroom.data.features
    Y = mushroom.data.targets

    # TODO: verificar se realmente é necessário isso
    # TODO: documentar
    X = X.to_numpy()
    Y = Y.to_numpy()

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
    X_encoded = encoder.fit_transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_encoded, Y, test_size=0.4, random_state=42)

    classifier = DecisionTreeClassifier(random_state=42)
    classifier.fit(X_train, Y_train)

    Y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(Y_test, Y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")


if __name__ == '__main__':
    main()
