"""
> Este arquivo é útil para:
    - carregar o modelo treinado (ReLU);
    - carregar o scaler (pq as entradas precisam ser normalizadas igual no treinamento);
    - prever a eficiência (EFF) de um jogador;
    - prever EFF de vários jogadores.

> Para instruções gerais, olhar o arquivo "src/instrucoes.txt"
"""

import torch
import torch.nn as nn
import pandas as pd
import joblib

# =============================
# 1. Carregar scaler e modelo
# =============================

CAMINHO_MODELO = "./dados_jogadores/modelo_eff_relu.pth"
CAMINHO_SCALER = "./dados_jogadores/scaler_normalizacao.pkl"

scaler = joblib.load(CAMINHO_SCALER)


# =============================
# 2. Criar arquitetura da RedeMLP
# =============================

class RedeMLP(nn.Module):
    def __init__(self, funcao_ativacao):
        super().__init__()
        self.modelo = nn.Sequential(
            nn.Linear(26, 64),
            funcao_ativacao(),
            nn.Linear(64, 32),
            funcao_ativacao(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.modelo(x)


# Criar modelo e carregar pesos
modelo = RedeMLP(nn.ReLU)
modelo.load_state_dict(torch.load(CAMINHO_MODELO, weights_only=True))
modelo.eval()


# =============================
# 3. Função para prever um jogador
# =============================

def prever_jogador(dados_jogador):
    """
    dados_jogador: dicionário OU lista de 26 valores (features)
    Exemplo de dicionário:
    {
        "GP": 80,
        "MIN": 2000,
        "FGM": ...,
        ...
    }
    """

    # Se for dicionário, converte para DataFrame
    if isinstance(dados_jogador, dict):
        df = pd.DataFrame([dados_jogador])
    else:
        raise ValueError("Os dados precisam estar em um dicionário com as 26 features.")

    # Normalizar com o scaler salvo
    dados_norm = scaler.transform(df)

    # Converter para tensor
    tensor = torch.tensor(dados_norm, dtype=torch.float32)

    # Fazer previsão
    with torch.no_grad():
        pred = modelo(tensor).item()

    return pred


# =============================
# 4. Função para prever vários jogadores
# =============================

def prever_jogadores(df):
    """
    df: DataFrame contendo várias linhas (jogadores)
    """

    dados_norm = scaler.transform(df)
    tensor = torch.tensor(dados_norm, dtype=torch.float32)

    with torch.no_grad():
        preds = modelo(tensor).numpy().flatten()

    return preds


# =============================
# TESTE RÁPIDO (opcional)
# =============================
if __name__ == "__main__":
    print("Carregando modelo e scaler... OK!")

    # Exemplo fictício (NÃO REAL)
    exemplo = {
        "GP": 1,
        "MIN": 20,
        "FGM": 5,
        "FGA": 10,
        "3PM": 2,
        "3PA": 5,
        "FTM": 1,
        "FTA": 2,
        "ORB": 1,
        "DRB": 3,
        "REB": 4,
        "AST": 2,
        "STL": 1,
        "BLK": 0,
        "PTS": 13,
        "TOV": 2,
        "PF": 3,
        "PTS_per_min": 13 / 20,
        "REB_per_min": 4 / 20,
        "AST_per_min": 2 / 20,
        "STL_per_min": 1 / 20,
        "BLK_per_min": 0 / 20,
        "TOV_per_min": 2 / 20,
        "3P_pct": 2 / 5,
        "FT_pct": 1 / 2,
        "FG_pct": 5 / 10
    }

    resultado = prever_jogador(exemplo)
    print(f"EFF prevista para o jogador fictício: {resultado:.4f}")