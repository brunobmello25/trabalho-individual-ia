# Dataset escolhido

https://github.com/uci-ml-repo/ucimlrepo
https://archive.ics.uci.edu/dataset/94/spambase

xadrez: https://archive.ics.uci.edu/dataset/22/chess+king+rook+vs+king+pawn
cogumelo: https://archive.ics.uci.edu/dataset/848/secondary+mushroom+dataset
adult income: https://archive.ics.uci.edu/dataset/2/adult

# estrutura do relatório

- como rodar o projeto
- estrutura básica do código
- dataset escolhido
- encoder utilizado e por que optei pelo onehotencoder ao invés do get_dummies
- valores testados para split (70-30, 60-40, 80-20)
- acurácia dos métodos

# timestamps

## Decision tree

tempo para encode: 21.37ms
tempo para split: 2.93ms
tempo para treinar: 9.25ms
tempo para predict: 0.27ms
Precisão da Decision Tree: 0.98

## Random forest com 10 árvores

tempo para encode: 21.60ms
tempo para split: 0.95ms
tempo para treinar com 10 árvores: 29.01ms
tempo para predict: 1.34ms
Random Forest Accuracy: 0.98

## Random forest com 30 árvores

tempo para encode: 21.80ms
tempo para split: 0.74ms
tempo para treinar com 30 árvores: 86.47ms
tempo para predict: 2.94ms
Random Forest Accuracy: 0.98

## Random forest com 50 árvores

tempo para encode: 21.02ms
tempo para split: 0.73ms
tempo para treinar com 50 árvores: 140.91ms
tempo para predict: 4.80ms
Random Forest Accuracy: 0.98

## Random forest com 75 árvores

tempo para encode: 20.52ms
tempo para split: 0.72ms
tempo para treinar com 75 árvores: 207.38ms
tempo para predict: 6.60ms
Random Forest Accuracy: 0.98

## Random forest com 100 árvores

tempo para encode: 20.50ms
tempo para split: 0.74ms
tempo para treinar com 100 árvores: 277.19ms
tempo para predict: 8.90ms
Random Forest Accuracy: 0.98

## Multi-layer Perceptron com (100,) camadas e 300 iterações

tempo para encode: 20.69ms
tempo para split: 0.76ms
tempo para normalizar: 1.24ms
tempo para treinar com layers (100,): 995.19ms
tempo para predict: 0.81ms
Precisão da Multilayer Perceptron: 0.98

## Multi-layer Perceptron com (100,100) camadas e 300 iterações

tempo para encode: 20.70ms
tempo para split: 0.71ms
tempo para normalizar: 1.21ms
tempo para treinar com layers (100, 100): 911.81ms
tempo para predict: 2.89ms
Precisão da Multilayer Perceptron: 0.98

# Dúvidas

- [ ] a acurácia de decision tree e multilayer perceptron com uma camada de 100 neuronios foram as mais bem sucedidas, ambas com menos de 1ms de de tempo de predict e 98% de precisão. Os outros modelos não ficaram mt diferentes, mas levaram mais tempo de execução. Isso significa que os dados são bem separáveis e um modelo simples já basta, né?
- [ ] Perguntar se o formato do código já está bom
