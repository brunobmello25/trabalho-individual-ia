# Relatório - Atividade Scikit Learning

## Como rodar o projeto

1. Criar um novo ambiente virtual python com `python -m venv venv`
2. Ativar o ambiente virtual com `source ./venv/bin/activate`
3. Instalar as dependências com `pip install -r requirements.txt`
4. Rodar o projeto com um dos três métodos abaixo:
    - `python -m src.main --decision-tree`
    - `python -m src.main --random-forest`
    - `python -m src.main --multi-perceptron`
5. Customizar os parâmetros do arquivo `src/consts.py`, se necessário:
    - Se desejar mudar o dataset utilizado, ajuste a variável `UCI_REPO_ID`
    - Se desejar mudar a quantidade de árvores do método random forest, ajuste a variável `TREE_COUNT`
    - Se desejar mudar os parâmetros do método de multilayer perceptron, ajuste as variáveis `MAX_PERCEPTRON_ITER` e `PERCEPTRON_HIDDEN_LAYER_SIZES`

## Estrutura do código

O código está dividido nos seguintes arquivos:

- **src/main.py**: Entrypoint da aplicação, com uma função main que chama a classe de parse dos argumentos 
- **src/consts.py**: Arquivo com alguns valores constantes que são reutilizados ao longo do projeto
- **src/args.py**: Classe responsável por parsear os argumentos de CLI. Usada para converter os parâmetros `--decision-tree`, `--random-forest` e `--multi-perceptron`
- **src/decision_tree.py**: Classe com toda implementação do método Decision Tree
- **src/random_forest.py**: Classe com toda implementação do método Random Forest
- **src/multilayer_perceptron.py**: Classe com toda implementação do método Multilayer Perceptron
- **src/utils.py**: Arquivo com funções utilitárias que foram utilizadas ao longo do desenvolvimento.

## Algoritmos de aprendizado de máquina e implementações extras utilizados

Para o desenvolvimento do projeto foram utilizadas as três classes principais do scikit learn: `DecisionTreeClassifier`, `RandomForestClassifier` e `MLPClassifier`. Além disso, para separar o dataset em uma parcela de treinamento e uma parcela de teste foi utilizada a função `train_test_split`, também do scikit learn.

Nos três métodos foi necessária uma etapa de pré processamento para transformar os dados categóricos em dados númericos. Para isso, foi utilizada a classe `OneHotEncoder`. Inicialmente a etapa de pré processamento estava utilizando o método `get_dummies` da biblioteca pandas, porém após realizar alguns testes de tempo de execução percebi que a codificação via onehotencoding resultava em um menor consumo de memória, por conta da matriz esparsa gerada pelo onehotencoding, então resolvi seguir com ela.

Por fim, a função `accuracy_score` do sklearn foi utilizada para medir a precisão da previsão do modelo.

## Variações de parâmetros utilizadas

### Decision Tree

- o parâmetro `criterion` foi testado com os valores `gini` e `entropy`, que influencia na forma em que as divisões são avaliadas. Isso pode influenciar na estrutura final da árvore. Não houve mudança significativa no tempo de execução ou na acurácia entre esses dois métodos.
- o parâmetro `max_depth` foi testado com os valores `5`, `10`, `15`, `20` e `25`


