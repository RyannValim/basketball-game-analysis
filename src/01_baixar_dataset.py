"""
> Este arquivo é útil para:
    - Salvar a referência da dataset que virá da API da Kaggle.com;
    - Salvar o destino do arquivo .CSV.
    
> Para instruções gerais, olhar o arquivo "src/instrucoes.txt"
"""

from kaggle import api

# Salva a referência que vem da Kaggle.com e o destino para onde o arquivo será salvo.
    # Para acessar: "kaggle.com/jacobbaruch/basketball-players-stats-per-season-49-leagues"
DATASET_REF = "jacobbaruch/basketball-players-stats-per-season-49-leagues"
DESTINO = "./dados_jogadores"

# Faz um request para a API da Kaggle para realizar o download da dataset.
api.dataset_download_files(DATASET_REF, path=DESTINO, unzip=True)
print("Download concluído! Arquivos salvos em:", DESTINO)