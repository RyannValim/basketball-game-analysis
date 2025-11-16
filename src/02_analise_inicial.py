"""
> Este arquivo é útil para:
    - Ler o dataset da pasta dados_jogadores com a ajuda da biblioteca Pandas;
    - Printar algumas informações úteis.
    
> Para instruções gerais, olhar o arquivo "src/instrucoes.txt"
"""

import pandas as pd

# Leitura do arquivo .CSV utilizando o pandas
dataset_csv = pd.read_csv("./dados_jogadores/players_stats_by_season_full_details.csv")

# Prints iniciais da estrutura do arquivo .CSV, das colunas e das informações gerais
print(dataset_csv.head())
print(dataset_csv.columns)
print(dataset_csv.info())

# Será utilizada a liga NBA
print("\nValores únicos de League:")
print(dataset_csv["League"].unique())

# Será analisada a season de 2016 - 2017
print("\nValores únicos de Season:")
print(dataset_csv["Season"].unique()[:20])

# O estágio da temporada que analisaremos serão os Playoffs
print("\nValores únicos de Stage:")
print(dataset_csv["Stage"].unique())