"""
> Este arquivo é útil para:
    - Normalizar os dados para entrar no processo de treinamento da rede neural;
    - Aplicar MinMaxScaler (0–1) para padronização.

> Para instruções gerais, olhar o arquivo "src/instrucoes.txt"
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

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

# 5. Salvar CSV final
df_normalizado.to_csv("./dados_jogadores/nba_playoffs_2016_2017_normalizado.csv")
print("Normalização concluída e arquivo salvo!")