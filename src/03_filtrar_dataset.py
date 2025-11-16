"""
> Este arquivo é útil para:
    - 
    
> Para instruções gerais, olhar o arquivo "src/instrucoes.txt"
"""

import pandas as pd

df = pd.read_csv("./dados_jogadores/players_stats_by_season_full_details.csv")

df_filtrado = df[
    (df["League"] == "NBA") &
    (df["Season"] == "2016 - 2017") &
    (df["Stage"] == "Playoffs")
]

print("Quantidade de jogadores encontrados:", len(df_filtrado))
print(df_filtrado.head())

df_filtrado.to_csv("./dados_jogadores/nba_playoffs_2016_2017.csv", index=False)
print("Arquivo salvo como: nba_playoffs_2016_2017.csv")