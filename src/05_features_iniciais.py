"""
> Este arquivo é útil para:
    - Criar novas features proporcionais (por minuto), permitindo que a rede neural receba dados mais significativos.

> Para instruções gerais, olhar o arquivo "src/instrucoes.txt"
"""

import pandas as pd

# Carrega o dataset limpo que você já gerou
df = pd.read_csv("./dados_jogadores/nba_playoffs_2016_2017_limpo.csv")

# ===== FEATURES PROPORCIONAIS (POR MINUTO) =====
# PTS/MIN
df["PTS_per_min"] = df["PTS"] / df["MIN"]

# REB/MIN
df["REB_per_min"] = df["REB"] / df["MIN"]

# AST/MIN
df["AST_per_min"] = df["AST"] / df["MIN"]

# STL/MIN
df["STL_per_min"] = df["STL"] / df["MIN"]

# BLK/MIN
df["BLK_per_min"] = df["BLK"] / df["MIN"]

# TOV/MIN
df["TOV_per_min"] = df["TOV"] / df["MIN"]

# TP_pct
df["3P_pct"] = df["3PM"] / df["3PA"]
df["3P_pct"] = df["3P_pct"].fillna(0)

# FT_pct
df["FT_pct"] = df["FTM"] / df["FTA"]
df["FT_pct"] = df["FT_pct"].fillna(0)

# FG%_pct
df["FG_pct"] = df["FGM"] / df["FGA"]
df["FG_pct"] = df["FG_pct"].fillna(0)

# Verificação
print("MÉTRICAS APLICADAS!!!\n\n", df.head())

# Salvando o arquivo com as features novas
df.to_csv("./dados_jogadores/nba_playoffs_2016_2017_features.csv", index=False)