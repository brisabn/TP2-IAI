# 📃Aprendizado de Máquina: KNN e KMeans
**ଘ(੭◌ˊ ᵕ ˋ)੭** ★ Este trabalho prático foi desenvolvido para a disciplina de Algoritmos 2, da Universidade Federal de Minas Gerais (UFMG). O objetivo era classificar jogadores da NBA e determinar se suas carreiras duraram pelo menos 5 anos na liga. Para isso, foram implementados dois algoritmos de aprendizado de máquina: o KNN e o KMeans.

## ｡₊⊹⭒˚｡⋆Execução
Para a execução do programa, deve ser utilizado Python3.

Execução no Windows:
**```python TP2.py <algorithm> [y | scatter] [scikit]```**

Cada parâmetro corresponde:
* **```<algorithm>```** Escolha o algoritmo de aprendizado de máquina. Deve ser uma das seguintes opções: 'knn', 'kmeans'
* **```[ y | scatter]```**: Parâmetros opcionais e exclusivos. y permite o plot de múltiplas execuções. scatter permite o plot 3d da dispersão dos dados.
* **```[scikit]```**: O parâmetro opcional "scikit" permite o a execução do algoritmo com a biblioteca scikit-learn, para fins de comparação

## ｡₊⊹⭒˚｡⋆Exemplos de uso
𖤐 Executa o knn no dataset fornecido e plota um gráfico de dispersão
```
python main.py nba_treino.csv nba_teste.csv knn scatter
```

𖤐 Executa o kmeans no dataset fornecido múltiplas vezes (30) e plota um gráfico relacionando a acurácia de cada execução
```
python main.py nba_treino.csv nba_teste.csv kmeans y
```

𖤐 Executa o knn da biblioteca scikit-learn no dataset fornecido e plota um gráfico de dispersão
``` 
python main.py nba_treino.csv nba_teste.csv knn scatter scikit
```

## ｡₊⊹⭒˚｡⋆Dependências
Este projeto depende das seguintes bibliotecas Python:
* **matplotlib** (https://matplotlib.org/)
* **numpy** (https://numpy.org/)
* **scikit-learn (sklearn)**(https://scikit-learn.org/)
