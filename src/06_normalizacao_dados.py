"""
> Este arquivo é útil para:
    - Normalizar os dados para entrar no processo de treinamento da rede neural;
    - Aplicar MinMaxScaler (0–1) para padronização;
    - Criar a métrica EFF (eficiência total) já sobre os dados normalizados.

> Para instruções gerais, olhar o arquivo "src/instrucoes.txt"
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

# 1. Carregar dataset de features
df = pd.read_csv("./dados_jogadores/nba_playoffs_2016_2017_features.csv")

# 2. Selecionar apenas colunas numéricas
df_numerico = df.select_dtypes(include=['float64', 'int64'])

# 3. Aplicar MinMaxScaler
scaler = MinMaxScaler()
dados_normalizados = scaler.fit_transform(df_numerico)

# 4. Criar novo DataFrame normalizado
df_normalizado = pd.DataFrame(
    dados_normalizados,
    columns=df_numerico.columns
)

# 5. Criar coluna EFF (Eficiência Total) usando dados normalizados
df_normalizado["EFF"] = (
    (df_normalizado["PTS"] + df_normalizado["REB"] + df_normalizado["AST"]
     + df_normalizado["STL"] + df_normalizado["BLK"])
    -
    ((df_normalizado["FGA"] - df_normalizado["FGM"])
     + (df_normalizado["FTA"] - df_normalizado["FTM"])
     + df_normalizado["TOV_per_min"])
)

# 6. Salvar CSV final normalizado
df_normalizado.to_csv("./dados_jogadores/nba_playoffs_2016_2017_normalizado.csv", index=False)

# 7. Salvar scaler
joblib.dump(scaler, "./dados_jogadores/scaler_normalizacao.pkl")

print("Normalização concluída, EFF criada e arquivo salvo!")