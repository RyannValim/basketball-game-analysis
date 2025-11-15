# Análise jogo de basquete

Projeto desenvolvido para a matéria de Sistemas Inteligentes, ministrada durante o 4° período da faculdade de Ciência da Computação na Universidade Positivo.

### 1° Passo - Escolha e Preparação do Dataset (0,50)

Achar um dataset sobre estatísticas de jogadores de basquete e:

* Descrever o dataset com fonte, período, quantidade de amostras e atributos;
* Justificar a escolha e o motivo de usar cada atributo;
* Padronizar dados;
* Definir qual será o target para treinamento da rede neural.

### 2° Passo - Implementação da Rede Neural (0,50)

Precisa ser no modelo MLP com no mínimo 3 camadas ocultas (quanto mais camadas, melhor acurácia, porém maior tempo de execução).

##### Requisitos

* Python - PyTorch: biblioteca para mexer com aprendizado de máquina;
* Usaremos as funções Tanh (-1, 1) para as camadas centrais e para a última camada a Sigmoid (que tem um grau de acerto melhor para a camada de saída);
* Exibir no terminal sempre que a porcentagem de erros for corrigida pela função de perda;
* Apresentar gráficos de desempenho.

### 3° Passo - Relatório Técnico (0,50)

Serão cobrados os seguintes itens:

* **Introdução e Contexto** – breve descrição do tema e da importância da análise;
* **Descrição do Dataset** – fonte, variáveis, período, quantidade de amostras;
* **Metodologia** – arquitetura da rede, funções de ativação, parâmetros ajustados;
* **Resultados Obtidos** – métricas, gráficos e comparação entre funções de ativação;
* **Discussão** – análise crítica dos resultados e limitações do modelo;
* **Conclusão** – lições aprendidas e potenciais melhorias;
* O relatório pode ser elaborado em Word ou PDF e deve estar nas normas da ABNT.


## Para Entrega:

1. Dataset utilizado no projeto;
2. Código do modelo MLP (seja arquivo .py ou .ipynb);
3. Relatório completo do projeto detalhado.
