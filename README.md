Este aplicativo serve para baixar os dados do SINAN (meningite; anos 2007 a 2025), SIM (2007 a 2024) e CIHA (2011 a 2025) referentes ao município, ao estado do Rio de Janeiro e a todos os estados, e convertê-los para os respectivos formatos parquet e duckdb, para fins de análise epidemiológica.

- **SINAN**: notificações/casos
- **SIM**: óbitos
- **CIHA**: internações/atendimentos

# Baixando os bancos de dados
Ao extrair os arquivos "SINAN - scripts", "CIHA - scripts" e "SIM - scripts" que estão em formato RAR, haverão scripts separados para as diferentes etapas - baixar os arquivos do datasus, processar e compilar o que foi baixado para o formato parquet e para o formato duckdb, separado por ano.
  
Alternativamente, pode-se baixar os arquivos já compilados diretamente através dos "Banco de Dados" em formato .RAR. Os CIDs incluídos foram: "A170", "A390", "A87", "G00", "G01", "G02", "G03".

Observação: Como os dados disponibilizados pelo CIHA são separados por mês para cada respectivo ano, optou-se por mesclar os meses para um único ano, apenas.

# Em construção - Formulário Digital para Investigação de meningite 
Utilizando XLXsforms, criei um espelho da ficha de investigação de meningite elaborada pelo SINAN. O propósito foi me familiarizar com este formato de planilha e quais possibilidades ela proporciona.

Link: https://ee.kobotoolbox.org/x/ifAQUhNw.
  
# *Em construção e verificação* - Painel Streamlit para Parquets do SINAN, SIM e CIHA

Este app em Python foi feito para análise epidemiológica a partir de arquivos `.parquet` do DATASUS, com foco nos três bancos de dados supracitados.

Observação: Caso seja necesssário a escolha de um código de município, o código para o Rio de Janeiro é "330455 ou 3304557". O código do estado do Rio de Janeiro é "33".

## O que o app faz

- lê um ou mais Parquets por base
- aceita **upload** ou **caminho local/glob** -> Eventualmente: fazer com o que o aplicativo já leia automaticamente os bancos de dados em vez de exigir que o usuário faça o upload.
- detecta automaticamente colunas prováveis de:
  - data
  - sexo
  - idade
  - município
  - desfecho/classificação
  - diagnóstico/CID
- gera gráficos epidemiológicos interativos
- permite download em CSV das tabelas agregadas de cada gráfico
- compara séries temporais entre bases

## Gráficos incluídos

### Para SINAN
- série temporal por ano/mês/semana
- heatmap ano × mês
- distribuição etária em faixas de 5 anos
- pirâmide etária por sexo
- top diagnósticos/agravos
- top desfechos/classificações
- top municípios
- completude de campos-chave

### Para SIM
- série temporal de óbitos
- heatmap ano × mês
- distribuição etária
- pirâmide etária por sexo
- top causas básicas / diagnóstico
- top variáveis de desfecho/local do óbito
- completude

### Para CIHA
- série temporal de internações/atendimentos
- heatmap ano × mês
- distribuição etária
- pirâmide etária por sexo
- top diagnósticos/procedimentos
- top desfechos de saída
- completude

## Instalação

Crie e ative um ambiente virtual, se desejar, e depois instale as dependências:

```bash
pip install -r requirements.txt
```

## Execução

No diretório do projeto, rode:

```bash
streamlit run app_streamlit_epidemiologia.py
```

## Como usar

### Opção 1: upload
Envie um ou mais arquivos `.parquet` em cada aba da base desejada.

### Opção 2: pasta/glob local
Informe um padrão local, por exemplo:

```text
Bases_Datasus_Municipio_Rio_de_Janeiro/SINAN/data/parquet/*.parquet
Bases_Datasus_Municipio_Rio_de_Janeiro/SIM/data/parquet/*.parquet
Bases_Datasus_Municipio_Rio_de_Janeiro/CIHA/data/parquet/*.parquet
```

## Observações importantes

- O app foi desenhado para funcionar com layouts **variáveis** do DATASUS, mas pode ser necessário ajustar manualmente as colunas detectadas.
- Para idade codificada do DATASUS, há a opção **"DATASUS codificada"**. A conversão para anos é aproximada para registros em horas, dias e meses.
- Se os parquets já estiverem filtrados para um município específico, os gráficos respeitarão esse recorte.
- A comparação entre bases é **exploratória** e faz mais sentido quando o agravo, o território e a janela temporal são os mesmos.

## Sugestões de uso epidemiológico

- Use a **série temporal** como gráfico principal para monitorar tendência.
- Use o **heatmap ano × mês** para sazonalidade.
- Use a **pirâmide etária por sexo** para perfil demográfico.
- Use **top diagnósticos/desfechos** para caracterização clínica e gravidade.
- Use a **completude** para avaliar qualidade da informação antes de interpretar resultados.
