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

- **src/main.py**: entrypoint da aplicação, com uma função main que chama a classe de parse dos argumentos 
- **src/args.py**: classe responsável por parsear os argumentos de CLI. Usada para converter os parâmetros `--decision-tree`, `--random-forest` e `--multi-perceptron`
- **src/decision_tree.py**: classe com o código necessário para
