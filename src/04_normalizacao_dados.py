"""
> Este arquivo é útil para:
    - Normalizar os dados para entrar no processo de treinamento da rede neural;
    - Cria o dataset limpo, finalmente.

> Para instruções gerais, olhar o arquivo "src/instrucoes.txt"
"""

import pandas as pd

# Carrega o arquivo filtrado anteriormente
dataset = pd.read_csv("./dados_jogadores/nba_playoffs_2016_2017.csv")

# Abaixo as colunas que serão mantidas para analisar
colunas_essenciais = [
    "League", "Season", "Stage", "Player", "Team",
    "GP", "MIN", "FGM", "FGA",
    "3PM", "3PA", "FTM", "FTA",
    "ORB", "DRB", "REB",
    "AST", "STL", "BLK",
    "PTS", "TOV", "PF"
]

dataset = dataset[colunas_essenciais]

# Remove qualquer linha totalmente vazia
dataset = dataset.dropna(how="all")

# Salva o dataset já limpo
dataset.to_csv("./dados_jogadores/nba_playoffs_2016_2017_limpo.csv", index=False)

# Print para conferência do cabeçalho e das informações de tipso e memória
print("Dataset limpo salvo como: nba_playoffs_2016_2017_limpo.csv")
print(dataset.head())
print(dataset.info())