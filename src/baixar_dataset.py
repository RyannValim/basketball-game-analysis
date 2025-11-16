# Para instruções gerais, olhar o arquivo "src/instrucoes.txt"

from kaggle import api

DATASET_REF = "jacobbaruch/basketball-players-stats-per-season-49-leagues"
DESTINO = "./dados_jogadores"

# O comando abaixo salva o arquivo .CSV no caminho da variável DESTINO (./dados_jogadores)
api.dataset_download_files(DATASET_REF, path=DESTINO, unzip=True)

print("Download concluído! Arquivos salvos em:", DESTINO)