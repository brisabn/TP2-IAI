Para a execução do programa, deve ser utilizado Python3.
**ଘ(੭◌ˊ ᵕ ˋ)੭** ★ Este trabalho prático foi desenvolvido para a disciplina de Algoritmos 2, da Universidade Federal de Minas Gerais (UFMG). O objetivo era implementar um algoritmo exato (Branch and Bound) e dois aproximativos (Christofides e Twice Around The Tree) para resolver o Problema do Caixeiro Viajante euclidiano. O foco da análise concentrou-se na comparação de desempenho, avaliando métricas de tempo, espaço e otimalidade.
Execução no Windows:
    python TP2.py <algorithm> [y | scatter] [scikit]

Cada parâmetro corresponde:
     <algorithm> Escolha o algoritmo de aprendizado de máquina. Deve ser uma das seguintes opções: 'knn', 'kmeans'
     [ y | scatter]: Parâmetros opcionais e exclusivos. y permite o plot de múltiplas execuções. scatter permite o plot 3d da dispersão dos dados.
     [scikit]: O parâmetro opcional "scikit" permite o a execução do algoritmo com a biblioteca scikit-learn, para fins de comparação

Exemplos de uso
    python TP2.py nba_treino.csv nba_teste.csv knn scatter
    : executa o knn no dataset fornecido e plota um gráfico de dispersão

    python TP2.py nba_treino.csv nba_teste.csv kmeans y
    : executa o kmeans no dataset fornecido múltiplas vezes (30) e plota um gráfico relacionando a acurácia de cada execução

    python TP2.py nba_treino.csv nba_teste.csv knn scatter scikit
    : executa o knn da biblioteca scikit-learn no dataset fornecido e plota um gráfico de dispersão

Dependências
    Este projeto depende das seguintes bibliotecas Python:
    - [matplotlib](https://matplotlib.org/)
    - [numpy](https://numpy.org/)
    - [scikit-learn (sklearn)](https://scikit-learn.org/)
