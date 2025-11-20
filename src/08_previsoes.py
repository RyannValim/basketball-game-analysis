"""
> Este arquivo é útil para:
    - carregar o modelo treinado (ReLU);
    - carregar o scaler (para normalizar novos dados);
    - carregar o dataset real de jogadores dos playoffs 2016-2017;
    - prever a eficiência (EFF) de vários jogadores de um time;
    - simular um cenário de jogo e indicar o melhor jogador do banco.

> Para instruções gerais, olhar o arquivo "src/instrucoes.txt"
"""

import torch
import torch.nn as nn
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# 1. Caminhos de arquivos e carregamento de modelo + scaler + dataset
CAMINHO_MODELO = "./dados_jogadores/modelo_eff_relu.pth"
CAMINHO_SCALER = "./dados_jogadores/scaler_normalizacao.pkl"
CAMINHO_FEATURES = "./dados_jogadores/nba_playoffs_2016_2017_features.csv"

scaler = joblib.load(CAMINHO_SCALER)

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

modelo = RedeMLP(nn.ReLU)
modelo.load_state_dict(torch.load(CAMINHO_MODELO, weights_only=True))
modelo.eval()

df_features = pd.read_csv(CAMINHO_FEATURES)

print("Modelo, scaler e dataset de features carregados com sucesso!\n")

# 2. Prever EFF para vários jogadores (DataFrame)
def prever_jogadores(df: pd.DataFrame) -> pd.Series:
    """
    Recebe um DataFrame com pelo menos:
        - colunas numéricas usadas no treino (iguais ao scaler.feature_names_in_)
        - pode conter colunas extras como 'Player' e 'Team'

    Retorna uma Series com a EFF prevista para cada linha (jogador).
    """
    colunas_numericas = scaler.feature_names_in_
    df_numerico = df[colunas_numericas]

    dados_norm = scaler.transform(df_numerico)
    tensor = torch.tensor(dados_norm, dtype=torch.float32)

    with torch.no_grad():
        preds = modelo(tensor).numpy().flatten()

    return pd.Series(preds, index=df.index, name="EFF_prevista")


# 3. Escolher o melhor jogador de um conjunto (banco, por exemplo)
def escolher_melhor_jogador(df: pd.DataFrame):
    """
    Recebe um DataFrame contendo:
        - Player
        - Team
        - todas as features numéricas necessárias

    Retorna:
        - nome do jogador escolhido
        - valor da EFF prevista
        - texto explicativo
    """

    df = df.copy()
    df["EFF_prevista"] = prever_jogadores(df)

    melhor = df.sort_values("EFF_prevista", ascending=False).iloc[0]

    jogador = melhor.get("Player", "Jogador desconhecido")
    time = melhor.get("Team", "Time desconhecido")
    eff = melhor["EFF_prevista"]

    explicacao = (
        f"O jogador {jogador} ({time}) é o melhor candidato para entrar, "
        f"com EFF prevista de {eff:.3f}. Esse valor resume a contribuição "
        f"esperada do atleta em termos de pontuação, rebotes, assistências, "
        f"defesa e perdas de posse, segundo as estatísticas históricas e o "
        f"modelo treinado."
    )

    return jogador, eff, explicacao

# 4. Simulação de cenário de jogo para um time específico
def simular_cenario_jogo(
    nome_time: str,
    minutos_restantes: int = 6,
    diferenca_pontos: int = -10
):
    """
    Simula um cenário de jogo para um time específico:

        - Seleciona todos os jogadores do time nos Playoffs 2016-2017;
        - Sorteia 5 jogadores "em quadra";
        - Considera o restante como "banco";
        - Usa o modelo para prever a EFF dos jogadores do banco;
        - Recomenda o melhor jogador para entrar.

    nome_time: nome do time exatamente como aparece na coluna 'Team'
    minutos_restantes: tempo que falta para acabar o período/jogo
    diferenca_pontos: pontos do time alvo - pontos do adversário
                      (valor negativo significa que o time está perdendo)
    """

    df_time = df_features[df_features["Team"] == nome_time]

    if df_time.empty:
        raise ValueError(f"Time '{nome_time}' não encontrado no dataset.")

    if len(df_time) < 6:
        raise ValueError("O time precisa ter ao menos 6 jogadores para a simulação.")

    print(f"Time alvo: {nome_time}")
    print("Jogadores disponíveis no dataset:")
    print(df_time["Player"].unique(), "\n")

    # Sorteia quem está em quadra (5 jogadores) e quem está no banco
    df_em_quadra = df_time.sample(5)
    df_banco = df_time.drop(df_em_quadra.index)

    print("Jogadores em quadra (sorteados):")
    print(df_em_quadra["Player"].tolist())
    print("\nJogadores no banco:")
    print(df_banco["Player"].tolist(), "\n")

    # Contexto do jogo
    print(f"Situação do jogo: {diferenca_pontos} pontos de saldo "
          f"(negativo = perdendo), faltando {minutos_restantes} minutos.\n")

        # Escolher melhor jogador do banco segundo o modelo
    jogador, eff, texto = escolher_melhor_jogador(df_banco)

    print("===== RECOMENDAÇÃO DO MODELO =====")
    print(texto)
    print("==================================")

    # --------------------------------------------------
    # Gráfico: EFF prevista de TODOS os jogadores do time
    # --------------------------------------------------
    df_time_com_eff = df_time.copy()
    df_time_com_eff["EFF_prevista"] = prever_jogadores(df_time_com_eff)

    # Ordena do maior para o menor
    df_ordenado = df_time_com_eff.sort_values("EFF_prevista", ascending=False)

    # Define cores: vermelho para o jogador recomendado, azul para o resto
    cores = [
        "tab:red" if nome == jogador else "tab:blue"
        for nome in df_ordenado["Player"]
    ]

    plt.figure(figsize=(10, 6))
    plt.bar(df_ordenado["Player"], df_ordenado["EFF_prevista"], color=cores)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("EFF prevista")
    plt.title(f"EFF prevista - time {nome_time}")

    # Linha pontilhada na EFF do recomendado (só pra reforçar)
    plt.axhline(y=eff, color="red", linestyle="--", linewidth=1, alpha=0.7)

    plt.tight_layout()
    plt.show()

# 5. Execução direta (exemplo de uso)
if __name__ == "__main__":
    simular_cenario_jogo("GSW", minutos_restantes=4, diferenca_pontos=-11)
