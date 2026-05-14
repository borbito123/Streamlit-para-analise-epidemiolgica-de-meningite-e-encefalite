Os bancos de dados do DATASUS que são trabalhados neste programa: 
- **SINAN**: notificações/casos sobre determinados agravos (no caso, meningite)
- **SIM**: óbitos registrados
- **CIHA**: internações/atendimentos

Este aplicativo cumpre duas funções:

1) Baixar os dados do SINAN (meningite; anos 2007 a 2025), SIM (2007 a 2024) e CIHA (2011 a 2025) referentes ao município, ao estado do Rio de Janeiro e a todos os estados, e convertê-los para os respectivos formatos parquet e duckdb, para fins de análise epidemiológica.

2) Fornecer uma plataforma dinâmica de análise de dados via streamlit.

# Baixando os bancos de dados
Ao extrair os arquivos "SINAN - scripts", "CIHA - scripts" e "SIM - scripts" que estão em formato RAR, haverão scripts separados para as diferentes etapas - baixar os arquivos do datasus, processar e compilar o que foi baixado para o formato parquet e para o formato duckdb, separado por ano.Alternativamente, pode-se baixar os arquivos já compilados diretamente através dos "Banco de Dados" em formato .RAR. 

Os  CID-10 incluídos são: "A170", "A390", "A87", "G00", "G01", "G02", "G03", "G04.2". Caso deseje saber o que eles significam: http://www2.datasus.gov.br/cid10/V2008/WebHelp/g00_g09.htm.

Observação: Como os dados disponibilizados pelo CIHA são separados por mês para cada respectivo ano, optou-se por mesclar os meses referentes a um dado ano.

# Em construção - Formulário Digital para Investigação de meningite 
Utilizando XLXsforms, criei um espelho da ficha de investigação de meningite elaborada pelo SINAN. O propósito foi me familiarizar com este formato de planilha e quais possibilidades ela proporciona.
No momento, o formuláro aparenta estar plenamente funcional, apenas faltando ajustes a sua apresentação estética.

Link: https://ee.kobotoolbox.org/x/ifAQUhNw.
  
# Em construção - Painel Streamlit para análise do banco de dados = SINAN, SIM e CIHA

Este app em Python foi feito para análise epidemiológica a partir de arquivos `.parquet ou .duckdb` do DATASUS, com foco nos três bancos de dados supracitados.

Link para a versão no streamlti: https://fgwybuegynhnli87zeyurr.streamlit.app/

Observação: Caso seja necesssário a escolha de um código de município, o código para o Rio de Janeiro é "330455 ou 3304557". O código do estado do Rio de Janeiro é "33". Se preferível, esse tipo de informação sobre os códigos dos municípios e dos estados pode ser obtido no site do IBGE, ou em sites como: https://atendimento.tecnospeed.com.br/hc/pt-br/articles/360021494734-Tabela-de-C%C3%B3digo-de-UF-do-IBGE e https://www.gov.br/receitafederal/dados/municipios.csv.

## _O que o app faz_

- Lê os parquets da release mais atual deste aplicativo (https://github.com/borbito123/Teste---Dados-Epidemiol-gicos-para-meningite-SINAN-CIHA-SIM---Rio-de-Janeiro/releases/tag/v1.0) e já os carrega automaticamente no programa. Cabe ao usuário escolher quais bancos de dados carregar -> Em construção; por hora, apenas estado do Rio.
- Também aceita **upload** ou **caminho local/glob** dos parquets / duckdbs que o usuário escolher.
- Fornece um dicionário operacional para guiar o usuário em relação aos campos mais relevantes para análise epidemiológico que o banco de dados escolhido possui.
- Gera gráficos epidemiológicos interativos.
- Permite download em CSV das tabelas agregadas de cada gráfico.

## _Gráficos incluídos_

### Para SINAN
- Indicadores -> Fornece: gráficos com porcentagem e valor absoluto da classificação dos casos, letalidade dos casos (podendo separar por grupo etiológico), se houve internação / hospitalização, 
- Análise temporal -> Fornece: análise da sazonalidade por meio de heatmap ano × mês, série temporal que pode ser estratificada conforme sexo e CID-10 para todos os bancos de dados, classificação da meningite + classificação final do caso (apenas para o SINAN). 
- Análise do CID-10 -> Fornece: distribuição dos casos por classificação final (confirmado, descartado...), distribuição dos casos por conclusão diagnóstica (especifica o grupo etiológico), conversão dos grupos etiológicos preenchidos no banco de dados para os devidos CID-10 (o streamlt mostra a regra usada para converter CON_DIAGES em CID-10), distribuição dos casos conforme evolução, distribuição dos casos por critério diagnóstico utilizado, distribuição conforme realiização de punção laboratorial, gráficos de distribuição dos principais parâmetros liquóricos analisados (glicose, leucócitos, proteínas, neutrófiilos).
- Demografia -> Fornece: Distribuição por faixa etária de 5 anos, pirâmide etária por sexo, distribuição por raça/cor, top municípios com maior prevalência de casos
- Campos importantes não preenchidos -> Fornece: quantos registros não foram preenchidos conforme certas variáveis de maior relevância
- Prévia -> Fornece: prévia do dados presentes no banco de dados, sendo possível exportar para o formato .CSV

### Para SIM
- Indicadores -> Fornece: total e percentual de óbitos nos quais a meningite está envolvida, distinguindo os casos onde houve menção de meningite ou onde a meningite foi a causa basica, gravidez e puerpério correlacionadas com os óbitos por meningite.
- Análise temporal -> Fornece: análise da sazonalidade por meio de heatmap ano × mês, série temporal que pode ser estratificada conforme sexo e CID-10 para todos os bancos de dados.
- Análise do CID-10 -> Fornece: distribuição dos óbitos conforme o CID-10, 
- Demografia -> Fornece: Distribuição por faixa etária de 5 anos, pirâmide etária por sexo, distribuição por raça/cor, top municípios com maior prevalência de casos
- Campos importantes não preenchidos -> Fornece: quantos registros não foram preenchidos conforme certas variáveis de maior relevância
- Prévia -> Fornece: prévia do dados presentes no banco de dados, sendo possível exportar para o formato .CSV

### Para CIHA
- Indicadores -> Fornece: o total de atendimentos e as mortes administrativas, dias de permanênca no ambiente hospitalar, 
- Análise temporal -> Fornece: análise da sazonalidade por meio de heatmap ano × mês, série temporal que pode ser estratificada conforme sexo e CID-10 para todos os bancos de dados, classificação da meningite + classificação final do caso (apenas para o SINAN). 
- Análise do CID-10 -> Fornece: tabela indicando o que cada CID-10 significa, distribuição dos casos por CID-10, 
- Demografia -> Fornece: Distribuição por faixa etária de 5 anos, pirâmide etária por sexo, distribuição por raça/cor, top municípios com maior prevalência de casos
- Campos importantes não preenchidos -> Fornece: quantos registros não foram preenchidos conforme certas variáveis de maior relevância
- Prévia -> Fornece: prévia do dados presentes no banco de dados, sendo possível exportar para o formato .CSV

### Comparação entre bancos de dados
- Comparação temporal (semanas, meses, anos)
- Possibilidade de estratiificar por CID-10.


## _Instalação_

Crie e ative um ambiente virtual, se desejar, e depois instale as dependências:

```bash
pip install -r requirements.txt
```

## _Execução_

No diretório do projeto, rode:

```bash
streamlit run app_streamlit_epidemiologia.py
```

## _Como usar_

  Em construção -> O programa irá automaticamente ler os parquets disponíveis na release mais atual. Quando houver a disponibilização dos parquets referentes a todos os estados, o usuário poderá escolher qual análise ele irá fazer (todos os estados ou algum estado específico).

### Opção 1: upload
Envie um ou mais arquivos `.parquet ou .duckdb` na respectiva aba do banco de dados desejado.

### Opção 2: pasta/glob local
Informe um padrão local, por exemplo:

```text
Bases_Datasus_Municipio_Rio_de_Janeiro/SINAN/data/parquet/*.parquet
Bases_Datasus_Municipio_Rio_de_Janeiro/SIM/data/parquet/*.parquet
Bases_Datasus_Municipio_Rio_de_Janeiro/CIHA/data/parquet/*.parquet
```

## _Observações importantes_

- Se os parquets já estiverem filtrados para um município específico, os gráficos respeitarão esse recorte.
- A comparação entre bases é **exploratória** e faz mais sentido quando o agravo, o território e a janela temporal são os mesmos.

## _Sugestões de uso epidemiológico_

- Use a **série temporal** como gráfico principal para monitorar tendência.
- Use o **heatmap ano × mês** para sazonalidade.
- Use a **pirâmide etária por sexo** para perfil demográfico.
- Use **top diagnósticos/desfechos** para caracterização clínica e gravidade.
- Use a **completude** para avaliar qualidade da informação antes de interpretar resultados.
