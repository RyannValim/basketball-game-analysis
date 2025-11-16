"""
> Este arquivo é útil para:
    - Separar um novo arquivo CSV baseado no arquivo dataset maior;
    - Filtra apenas pelos campos:
        - League: NBA, analisaremos a liga americana NBA;
        - Season: 2016 - 2017, esta season foi escolhida pois os jogadores que atuaram nela são considerados os melhores do mundo;
        - Stage: Playoffs, que é quando os times se classificam para jogar o mata-mata (pois têm jogos decisivos - mais interessantes).
    - Printa as informações do dataset para conferência;
    - Cria o dataset menor filtrado com os campos acima para análise futura.
    
> Para instruções gerais, olhar o arquivo "src/instrucoes.txt"
"""

import pandas as pd

# Realiza a leitura do dataset maior extraído na etapa 01
dataset = pd.read_csv("./dados_jogadores/players_stats_by_season_full_details.csv")

# Filtra este dataset para um escopo em que apareça só os jogos dos playoffs da temporada 2016/2017 da liga americana NBA
dataset_filtrado = dataset[
    (dataset["League"] == "NBA") &
    (dataset["Season"] == "2016 - 2017") &
    (dataset["Stage"] == "Playoffs")
]

# Printa a quantidade de jogadores e as cinco primeiras linhas deste novo dataset em .CSV
print("Quantidade de jogadores encontrados:", len(dataset_filtrado))
print(dataset_filtrado.head())

# Salva o novo dataset na mesma pasta "src/dados_jogadores/" para análise futura
dataset_filtrado.to_csv("./dados_jogadores/nba_playoffs_2016_2017.csv", index=False)
print("Arquivo salvo como: nba_playoffs_2016_2017.csv")