Os bancos de dados do DATASUS que são trabalhados neste programa: 
- **SINAN**: notificações/casos sobre determinados agravos (no caso, meningite)
- **SIM**: óbitos registrados
- **CIHA**: internações/atendimentos hospitalares e/ou ambulatorais.

Este aplicativo cumpre duas funções:

1) Baixar os dados do SINAN (meningite; anos 2007 a 2025), SIM (2007 a 2024) e CIHA (2011 a 2025) referentes ao município, ao estado do Rio de Janeiro e a todos os estados, e convertê-los para os respectivos formatos parquet e duckdb, para fins de análise epidemiológica.

2) Fornecer uma plataforma dinâmica de análise de dados via streamlit.

# Baixando os bancos de dados e convertendo-os
Ao extrair os arquivos "SINAN - scripts", "CIHA - scripts" e "SIM - scripts" que estão em formato RAR, haverão scripts separados para as diferentes etapas - baixar os arquivos do datasus, processar e compilar o que foi baixado para o formato parquet e para o formato duckdb, separado por ano. Bastar executar os scripts. Preferiu-se não unificar os arquivos para que o usuário tenha liberdade de escolher o que baixar. 
Alternativamente, pode-se baixar os arquivos já compilados diretamente através dos "Banco de Dados" em formato .RAR.

Quando os bancos de dados em .dbc são convertidos para .parquet, alguns filtros são aplicados para restringir quais casos são relevantes para a análise epidemiológica da meningite, da encefalite e da meningoencefalite. Além disso, como os dados disponibilizados pelo CIHA são separados por mês para cada respectivo ano, optou-se por mesclar os meses referentes a um dado ano, com a finalidade de analisar mais facilmente os casos referentes a um dado ano.

Em um primeiro momento, os CID-10 utilizados eram: "A170", "A390", "A87", "G00", "G01", "G02", "G03", "G04", G05". Contudo, para SIM e o CIHA, havia um problema significativo em relação ao que constava no banco de dados. Por exemplo, os CID B58.2 (meningoencefalite por toxoplasmose) e B01.1 (encefalite por varicela) deveriam estar inclusos dentro do CID G05, mas, ao analisar os bancos de dados crus do DATASUS, essa inclusão não era realizada, restando apenas os CID-10 avulsos. 

Para contornar esse impasse, foi feita uma análise dos CID-10 existentes em busca daqueles que incluem meningite, encefalite e/ou meningoencefalite sem incluir outras condições em um único CID. Desse modo, atualmente os CID-10 inclusos estão divididos em prefixados (G00, G01, G02, G03, G04, G05) e avulsos (A17, A22.8, A32.1, A39, A83, A84, A85, A86, A87, B00.3, B00.4, B01.0, B1.1, B2.0, B2.1, B06, B26.1, B26.2, B37.5, B38.4, B45.1, B58.2, B57.4 B60.2).

Referências utilizadas: http://www2.datasus.gov.br/cid10/V2008/WebHelp/g00_g09.htm e http://www2.datasus.gov.br/cid10/V2008/cid10.htm.

# Em construção - Formulário Digital para Investigação de meningite 

Utilizando XLXsforms, criei um espelho da ficha de investigação de meningite elaborada pelo SINAN. O propósito foi me familiarizar com este formato de planilha e quais possibilidades ela proporciona.
No momento, o formuláro está plenamente funcional, apenas faltando alguns ajustes para aprimorar sua apresentação estética. Caso queira acesso aos dados de preenchimento, favor entrar em contato.

Link: https://ee.kobotoolbox.org/x/ifAQUhNw.
  
# Em construção - Painel Streamlit para análise do banco de dados = SINAN, SIM e CIHA

Este app em Python foi feito para análise epidemiológica a partir de arquivos `.parquet ou .duckdb` do DATASUS, com foco nos três bancos de dados supracitados.
Link para a versão no streamlti: https://fgwybuegynhnli87zeyurr.streamlit.app/

## _O que o app faz_

- Lê os parquets da release mais atual deste aplicativo (https://github.com/borbito123/Teste---Dados-Epidemiol-gicos-para-meningite-SINAN-CIHA-SIM---Rio-de-Janeiro/releases/tag/v1.0) e já os carrega automaticamente no programa. Cabe ao usuário escolher quais bancos de dados carregar. Atualmente são disponibilizados os dados referente ao estado do RJ e logo mais os bancos de todas as UFs juntas serão disponibilizados.
- Também aceita **upload** ou **caminho local/glob** dos parquets / duckdbs que o usuário escolher.
- Fornece um dicionário operacional para guiar o usuário em relação aos campos mais relevantes para análise epidemiológico que o banco de dados escolhido possui.
- Gera gráficos epidemiológicos interativos.
- Permite download em CSV das tabelas agregadas de cada gráfico.

_Observação: Para contornar eventuais problemas de memória ou crashes do aplicativo, foram impostas algumas limitações que podem ser modificadas pelo usuário. No canto esquerdo da aba "Orientação" há a opção "desempenho e memória" que permite ajustar essas limitações._

## _Gráficos incluídos_

### Para SINAN
- Indicadores -> Fornece: gráficos com porcentagem e valor absoluto da classificação dos casos, letalidade dos casos (podendo separar por grupo etiológico), se houve internação / hospitalização, 
- Análise temporal -> Fornece: análise da sazonalidade por meio de heatmap ano × mês, série temporal que pode ser estratificada conforme sexo e CID-10 para todos os bancos de dados, classificação da meningite + classificação final do caso (apenas para o SINAN). 
- Análise do CID-10 -> Fornece: distribuição dos casos por classificação final (confirmado, descartado...), distribuição dos casos por conclusão diagnóstica (especifica o grupo etiológico), conversão dos grupos etiológicos preenchidos no banco de dados para os devidos CID-10 (o streamlt mostra a regra usada para converter CON_DIAGES em CID-10), distribuição dos casos conforme evolução, distribuição dos casos por critério diagnóstico utilizado, distribuição conforme realiização de punção laboratorial, gráficos de distribuição dos principais parâmetros liquóricos analisados (glicose, leucócitos, proteínas, neutrófiilos).
- Demografia -> Fornece: Distribuição por faixa etária de 5 anos, pirâmide etária por sexo, distribuição por raça/cor, top municípios com maior prevalência de casos
- Campos importantes não preenchidos -> Fornece: quantos registros não foram preenchidos conforme certas variáveis de maior relevância
- Prévia -> Fornece: prévia do dados presentes no banco de dados, sendo possível exportar para o formato .CSV

_Explicando o que foi feito na tabela de conversão encontrada no SINAN:_ Originalmente, o SINAN agrupa todos os seus casos sob o CID "G03.9". Caso haja diagnóstico e confirmação, então se especifica a meningite em algumas categorias (veja a seção "Classificação do Caso" em https://portalsinan.saude.gov.br/images/documentos/Agravos/Meningite/Meningite_v5.pdf). Na seção "CID-10 / classificação", haverá um gráfico de conversão que aloca todos os casos confirmados e os enquadra em algum dos seguintes CID: G00, G01, G02, G03, G04, G05, A39, A17, A87.

A se ponderar: A17 e A39 se enquadrariam no CID G01, mas atualmnte se encontram separadas. Em contrapartida, meningite por haemophilus e meningocóccica já foram incluídas no CID G00. Isso representaria uma certa inconsistência que precisaria ser corrigida.

A referência utilizada para alocação foi: https://portalsinan.saude.gov.br/images/documentos/Agravos/Meningite/Meningite_v5.pdf.

### Para SIM
- Indicadores -> Fornece: total e percentual de óbitos nos quais a meningite está envolvida, distinguindo os casos onde houve menção de meningite ou onde a meningite foi a causa basica, gravidez e puerpério correlacionadas com os óbitos por meningite.
- Análise temporal -> Fornece: análise da sazonalidade por meio de heatmap ano × mês, série temporal que pode ser estratificada conforme sexo e CID-10 para todos os bancos de dados.
- Análise do CID-10 -> Fornece: distribuição dos óbitos conforme o CID-10, gráfico que converte os CID-10 para o padrão utilizado no gráfico de conversão do SINAN. 
- Demografia -> Fornece: Distribuição por faixa etária de 5 anos, pirâmide etária por sexo, distribuição por raça/cor, top municípios com maior prevalência de casos
- Campos importantes não preenchidos -> Fornece: quantos registros não foram preenchidos conforme certas variáveis de maior relevância
- Prévia -> Fornece: prévia do dados presentes no banco de dados, sendo possível exportar para o formato .CSV

_Explicando o que foi feito na tabela de conversão encontrada no SIM:_ Por conta do jeito que o banco de dados é preenchido e disponiblizado, muitos CIDs que são incluídos em um dos CIDs prefixados (G00, G01, G02, G03, G04, G05) ficariam perdidos se o script de conversão não procurasse por eles explicitamente. Desse modo, os novos CIDs mencionados na seção "Baixando os bancos de dados e convertendo-os" deste readme.md foram inclusos para evitar que não fossem perdidos. Na seção "CID-10 / classificação", haverá um gráfico de conversão que aloca todos os casos confirmados e os enquadra em algum dos seguintes CID: G00, G01, G02, G03, G04, G05, A39, A17, A87.

A se ponderar: A17 e A39 se enquadrariam no CID G01, mas atualmnte se encontram separadas. Em contrapartida, meningite por haemophilus e meningocóccica já foram incluídas no CID G00. Isso representaria uma certa inconsistência que precisaria ser corrigida. 

### Para CIHA
- Indicadores -> Fornece: o total de atendimentos e as mortes administrativas, dias de permanênca no ambiente hospitalar, 
- Análise temporal -> Fornece: análise da sazonalidade por meio de heatmap ano × mês, série temporal que pode ser estratificada conforme sexo e CID-10 para todos os bancos de dados, classificação da meningite + classificação final do caso (apenas para o SINAN). 
- Análise do CID-10 -> Fornece: distribuição dos casos por CID-10, gráfico que converte os CID-10 para o padrão utilizado no gráfico de conversão do SINAN. 
- Demografia -> Fornece: Distribuição por faixa etária de 5 anos, pirâmide etária por sexo, distribuição por raça/cor, top municípios com maior prevalência de casos
- Campos importantes não preenchidos -> Fornece: quantos registros não foram preenchidos conforme certas variáveis de maior relevância
- Prévia -> Fornece: prévia do dados presentes no banco de dados, sendo possível exportar para o formato .CSV

_Explicando o que foi feito na tabela de conversão encontrada no CIHA:_ Por conta do jeito que o banco de dados é preenchido e disponiblizado, muitos CIDs que são incluídos em um dos CIDs prefixados (G00, G01, G02, G03, G04, G05) ficariam perdidos se o script de conversão não procurasse por eles explicitamente. Desse modo, os novos CIDs mencionados na seção "Baixando os bancos de dados e convertendo-os" deste readme.md foram inclusos para evitar que não fossem perdidos. Na seção "CID-10 / classificação", haverá um gráfico de conversão que aloca todos os casos confirmados e os enquadra em algum dos seguintes CID: G00, G01, G02, G03, G04, G05, A39, A17, A87.

A se ponderar: A17 e A39 se enquadrariam no CID G01, mas atualmnte se encontram separadas. Em contrapartida, meningite por haemophilus e meningocóccica já foram incluídas no CID G00. Isso representaria uma certa inconsistência que precisaria ser corrigida. 

### Comparação entre bancos de dados
- Comparação temporal (semanas, meses, anos)
- Possibilidade de estratiificar por CID-10.

Observação: A comparação entre bases é **exploratória** e faz mais sentido quando o agravo, o território e a janela temporal são os mesmos.

# _Instalação_

Crie e ative um ambiente virtual, se desejar, e depois instale as dependências:

```bash
pip install -r requirements.txt
```

## _Execução_

No diretório do projeto, rode:

```bash
streamlit run app_streamlit_app.py
```

## _Como usar_

Disclaimer: Atenção ao limite de parquets / duckdbs que estão sendo lidos ao mesmo tempo em uma única seção. Isso pode ser alterado, mas bastante cuidado em quantos arquivos são carregados simultaneamente.

### Opção 1: leitura automática dos parquets disponibilizados na release mais atual do github
Basta selecionar quais anos deseja-se analisar. 

### Opção 2: upload
Envie um ou mais arquivos `.parquet ou .duckdb` na respectiva aba do banco de dados desejado.

### Opção 3: pasta/glob local
Informe um padrão local, por exemplo:

```text
Bases_Datasus_Municipio_Rio_de_Janeiro/SINAN/data/parquet/*.parquet
Bases_Datasus_Municipio_Rio_de_Janeiro/SIM/data/parquet/*.parquet
Bases_Datasus_Municipio_Rio_de_Janeiro/CIHA/data/parquet/*.parquet
```
