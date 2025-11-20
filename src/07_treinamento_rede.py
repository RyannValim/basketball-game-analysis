"""
> Este arquivo é útil para:
    - carregar o dataset normalizado com as estatísticas dos jogadores;
    - separar as variáveis de entrada (X) e a saída (y = EFF);
    - dividir os dados em treino e teste;
    - converter os dados para tensores do PyTorch;
    - definir a arquitetura da rede neural (RedeMLP);
    - treinar dois modelos (ReLU e Tanh) usando PyTorch;
    - registrar o erro (perda) por época de treinamento;
    - avaliar o desempenho no conjunto de teste (MSE e R²);
    - salvar o modelo final treinado (ReLU) em disco para uso posterior.

> Para instruções gerais, olhar o arquivo "src/instrucoes.txt".
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# 1. Carregar dataset normalizado
df = pd.read_csv("./dados_jogadores/nba_playoffs_2016_2017_normalizado.csv")

# 2. Separar entradas (X) e saída (y)
X = df.drop(columns=["EFF"])
y = df["EFF"]

print("Colunas de entrada (X):", list(X.columns))
print("Total de jogadores:", X.shape[0])


# 3. Dividir dados em treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X.values,
    y.values.reshape(-1, 1),
    test_size=0.2,
    random_state=42
)

print("Formato X_treino:", X_treino.shape)
print("Formato y_treino:", y_treino.shape)
print("Formato X_teste:", X_teste.shape)
print("Formato y_teste:", y_teste.shape)

# 4. Converter para tensores PyTorch
X_treino_tensor = torch.tensor(X_treino, dtype=torch.float32)
y_treino_tensor = torch.tensor(y_treino, dtype=torch.float32)

X_teste_tensor = torch.tensor(X_teste, dtype=torch.float32)
y_teste_tensor = torch.tensor(y_teste, dtype=torch.float32)

print("\nTensores criados com sucesso:")
print("X_treino_tensor:", X_treino_tensor.shape)
print("y_treino_tensor:", y_treino_tensor.shape)
print("X_teste_tensor:", X_teste_tensor.shape)
print("y_teste_tensor:", y_teste_tensor.shape)

# 5. Criar Classe da Rede Neural (MLP)
class RedeMLP(nn.Module):
    """
    Rede Neural Multi-Layer Perceptron (MLP)
    - usa função de ativação passada no construtor
    - camada de saída é linear (regressão)
    """

    def __init__(self, funcao_ativacao):
        super().__init__()

        self.modelo = nn.Sequential(
            nn.Linear(26, 64),     # 1ª camada oculta
            funcao_ativacao(),     # ativação escolhida
            nn.Linear(64, 32),     # 2ª camada oculta
            funcao_ativacao(),     # ativação escolhida
            nn.Linear(32, 1)       # saída (linear)
        )

    def forward(self, x):
        return self.modelo(x)


"""
26  = número de features (entradas)
64  = tamanho da primeira camada oculta
32  = tamanho da segunda camada oculta
1   = saída (valor previsto da EFF)
"""

# 6. Preparar o loop de treinamento
def treinar_modelo(modelo, X_treino, y_treino,
                   eta=0.001,
                   epocas=200,
                   tamanho_lote=16):
    """
    Função que treina o modelo da rede neural.
        - modelo: instância da RedeMLP;
        - X_treino e y_treino: tensores com os dados de treino;
        - eta: taxa de aprendizado;
        - epocas: cada época representa um treinamento da RedeMLP;
        - tamanho_lote: quantidade de exemplos usada em cada atualização dos pesos.
    """
    
    criterio_perda = nn.MSELoss()                           # função de perda
    otimizador = optim.Adam(modelo.parameters(), lr=eta)    # otimizador Adam
    vetor_perdas = []                                       # armazena erro/época
    
    # Loop de treinamento
    for epoca in range(epocas):
        modelo.train()
        soma_perda = 0.0
        
        # Criar lotes
        for i in range(0, len(X_treino), tamanho_lote):
            X_lote = X_treino[i:i + tamanho_lote]
            y_lote = y_treino[i:i + tamanho_lote]
            
            # Forward
            y_pred = modelo(X_lote)
            
            # Cálculo de perda
            perda = criterio_perda(y_pred, y_lote)
            
            # Backpropagation
            otimizador.zero_grad()
            perda.backward()
            otimizador.step()
            
            soma_perda += perda.item()
            
        perda_medio = soma_perda / (len(X_treino) // tamanho_lote)
        vetor_perdas.append(perda_medio)
        
        # Mostrar progresso
        if (epoca + 1) % 20 == 0:
            print(f"Época {epoca + 1}/{epocas} - Perda média: {perda_medio:.5f}")
    
    return vetor_perdas

# 7. Treinamento com os modelos utilizando ReLU e Tanh
if __name__ == "__main__":
    # Para reprodutibilidade
    torch.manual_seed(42)
    
    # Modelo 1: ReLU
    print("\nTreinando modelo com ReLU...")
    modelo_relu = RedeMLP(nn.ReLU)
    perdas_relu = treinar_modelo(
        modelo_relu,
        X_treino_tensor,
        y_treino_tensor,
        eta=0.001,
        epocas=200,
        tamanho_lote=16
    )

    # Modelo 2: Tanh
    print("\nTreinando modelo com Tanh...")
    modelo_tanh = RedeMLP(nn.Tanh)
    perdas_tanh = treinar_modelo(
        modelo_tanh,
        X_treino_tensor,
        y_treino_tensor,
        eta=0.001,
        epocas=200,
        tamanho_lote=16
    )

    print("\nTreino concluído!")
    print(f"Última perda (ReLU): {perdas_relu[-1]:.5f}")
    print(f"Última perda (Tanh): {perdas_tanh[-1]:.5f}")
    
    # 8. Avaliação no conjunto de teste
    modelo_relu.eval()
    modelo_tanh.eval()

    with torch.no_grad():
        pred_relu_teste = modelo_relu(X_teste_tensor).detach().numpy().flatten()
        pred_tanh_teste = modelo_tanh(X_teste_tensor).detach().numpy().flatten()
        y_teste_np = y_teste_tensor.detach().numpy().flatten()

    mse_relu_teste = mean_squared_error(y_teste_np, pred_relu_teste)
    mse_tanh_teste = mean_squared_error(y_teste_np, pred_tanh_teste)

    r2_relu = r2_score(y_teste_np, pred_relu_teste)
    r2_tanh = r2_score(y_teste_np, pred_tanh_teste)

    print("\nDesempenho no CONJUNTO DE TESTE:")
    print(f"ReLU  -> MSE: {mse_relu_teste:.6f} | R²: {r2_relu:.4f}")
    print(f"Tanh  -> MSE: {mse_tanh_teste:.6f} | R²: {r2_tanh:.4f}")
    
    # 9. Plotar gráficos de perda
    plt.figure(figsize=(10, 5))
    plt.plot(perdas_relu, label="ReLU", linewidth=2)
    plt.plot(perdas_tanh, label="Tanh", linewidth=2)

    plt.title("Curva de Perda por Época")
    plt.xlabel("Épocas")
    plt.ylabel("Perda (MSE)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 10. Salvar modelo final (ReLU)
    caminho_modelo = "./dados_jogadores/modelo_eff_relu.pth"
    torch.save(modelo_relu.state_dict(), caminho_modelo)
    print(f"\nModelo final (ReLU) salvo em: {caminho_modelo}")