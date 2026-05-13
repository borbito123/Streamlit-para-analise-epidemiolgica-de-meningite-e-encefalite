# -*- coding: utf-8 -*-
"""
Painel epidemiológico para meningite — SINAN, SIM e CIHA

O app aceita arquivos DuckDB ou Parquet, calcula indicadores descritivos e separa:
- CID-10 bruto do caso/óbito/atendimento;
- classificação epidemiológica específica do SINAN, especialmente CON_DIAGES;
- definições operacionais de série: notificações, confirmados, descartados, óbitos etc.

Executar:
    streamlit run app_meningite_epidemiologico.py

Dependências:
    pip install streamlit duckdb pandas numpy plotly
"""

from __future__ import annotations

import glob
import hashlib
import os
import tempfile
import textwrap
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# =============================================================================
# Configuração geral
# =============================================================================

st.set_page_config(
    page_title="Meningite — SINAN, SIM e CIHA",
    page_icon="🧫",
    layout="wide",
)

APP_VERSION = "2026-05-13-v5-cid-sinan"


CID_RULES = [
    {
        "grupo": "A17.0",
        "prefixo": "A170",
        "rotulo": "A17.0 — meningite tuberculosa",
        "padrao": "A170",
    },
    {
        "grupo": "A39.0",
        "prefixo": "A390",
        "rotulo": "A39.0 — meningite meningocócica",
        "padrao": "A390",
    },
    {
        "grupo": "A87",
        "prefixo": "A87",
        "rotulo": "A87 — meningite viral",
        "padrao": "A87*",
    },
    {
        "grupo": "G00",
        "prefixo": "G00",
        "rotulo": "G00 — meningite bacteriana",
        "padrao": "G00*",
    },
    {
        "grupo": "G01",
        "prefixo": "G01",
        "rotulo": "G01 — meningite bacteriana em doença classificada em outra parte",
        "padrao": "G01*",
    },
    {
        "grupo": "G02",
        "prefixo": "G02",
        "rotulo": "G02 — meningite em outras doenças infecciosas/parasitárias",
        "padrao": "G02*",
    },
    {
        "grupo": "G03",
        "prefixo": "G03",
        "rotulo": "G03 — meningite por outras causas / não especificada",
        "padrao": "G03*",
    },
]

# Aceita CIDs com ponto, sem ponto, precedidos de * e dentro de campos compostos.
CID_MENINGITE_REGEX = (
    r"(A17[\.]?0|A39[\.]?0|A87[\.]?[0-9A-Z]?|G00[\.]?[0-9A-Z]?|"
    r"G01[\.]?[0-9A-Z]?|G02[\.]?[0-9A-Z]?|G03[\.]?[0-9A-Z]?)"
)

SINAN_CON_DIAGES = {
    "01": "01 — meningococcemia",
    "02": "02 — meningite meningocócica",
    "03": "03 — meningite meningocócica com meningococcemia",
    "04": "04 — meningite tuberculosa",
    "05": "05 — meningite por outras bactérias",
    "06": "06 — meningite não especificada",
    "07": "07 — meningite asséptica",
    "08": "08 — meningite por outra etiologia",
    "09": "09 — meningite por Haemophilus influenzae",
    "10": "10 — meningite por Streptococcus pneumoniae / pneumocócica",
}

SINAN_CON_GROUP = {
    "01": "Meningocócica / meningococcemia",
    "02": "Meningocócica / meningococcemia",
    "03": "Meningocócica / meningococcemia",
    "04": "Tuberculosa",
    "05": "Outras bacterianas",
    "06": "Não especificada",
    "07": "Asséptica / viral provável",
    "08": "Outra etiologia",
    "09": "Haemophilus influenzae",
    "10": "Pneumocócica",
}

# Conversão operacional CON_DIAGES (SINAN) -> CID-10 para comparação com SIM/CIHA.
# A categoria 01 (meningococcemia isolada) é propositalmente não convertida,
# pois não representa meningite quando aparece sem a forma meningítica.
SINAN_CID10_FROM_CON_DIAGES = {
    "02": {
        "grupo": "A39.0",
        "rotulo": "A39.0 — meningite meningocócica",
        "origem": "02 — meningite meningocócica",
    },
    "03": {
        "grupo": "A39.0",
        "rotulo": "A39.0 — meningite meningocócica",
        "origem": "03 — meningite meningocócica com meningococcemia",
    },
    "04": {
        "grupo": "A17.0",
        "rotulo": "A17.0 — meningite tuberculosa",
        "origem": "04 — meningite tuberculosa",
    },
    "05": {
        "grupo": "G00",
        "rotulo": "G00 — meningite bacteriana não classificada em outra parte",
        "origem": "05 — meningite por outras bactérias",
    },
    "06": {
        "grupo": "G03",
        "rotulo": "G03 — meningite por outras causas / não especificada",
        "origem": "06 — meningite não especificada",
    },
    "07": {
        "grupo": "A87",
        "rotulo": "A87 — meningite viral",
        "origem": "07 — meningite asséptica (operacionalmente viral no SINAN)",
    },
    "08": {
        "grupo": "G02",
        "rotulo": "G02 — meningite em outras doenças infecciosas/parasitárias",
        "origem": "08 — meningite por outra etiologia",
    },
    "09": {
        "grupo": "G00",
        "rotulo": "G00 — meningite bacteriana não classificada em outra parte",
        "origem": "09 — meningite por Haemophilus influenzae",
    },
    "10": {
        "grupo": "G00",
        "rotulo": "G00 — meningite bacteriana não classificada em outra parte",
        "origem": "10 — meningite pneumocócica",
    },
}

SINAN_CID10_NOT_CONVERTED = {
    "01": "Não convertido — meningococcemia isolada",
}

SINAN_CID10_MAPPING_ROWS = [
    {
        "CON_DIAGES": "04",
        "Grupo SINAN": "Meningite tuberculosa",
        "CID-10 convertido": "A17.0",
        "Observação": "Mantida como A17.0/A170 para comparação com CID bruto.",
    },
    {
        "CON_DIAGES": "02, 03",
        "Grupo SINAN": "Meningite meningocócica; meningite meningocócica com meningococcemia",
        "CID-10 convertido": "A39.0",
        "Observação": "Meningococcemia isolada (01) não entra nesta conversão.",
    },
    {
        "CON_DIAGES": "07",
        "Grupo SINAN": "Meningite asséptica",
        "CID-10 convertido": "A87",
        "Observação": "No SINAN, a categoria asséptica é tratada operacionalmente como viral; usar G03 somente para asséptica sem evidência/definição viral em CID bruto externo.",
    },
    {
        "CON_DIAGES": "05, 09, 10",
        "Grupo SINAN": "Outras bacterianas; Haemophilus influenzae; pneumocócica",
        "CID-10 convertido": "G00",
        "Observação": "Agrega G00.0, G00.1 e G00.8/G00.9 para comparação por família CID.",
    },
    {
        "CON_DIAGES": "08",
        "Grupo SINAN": "Meningite por outra etiologia",
        "CID-10 convertido": "G02",
        "Observação": "Correção lógica: no dicionário SINAN, esta categoria cobre principalmente fungos/protozoários/parasitas; portanto é mais compatível com G02 do que com G03.",
    },
    {
        "CON_DIAGES": "06",
        "Grupo SINAN": "Meningite não especificada",
        "CID-10 convertido": "G03",
        "Observação": "Usada para causa não especificada/outras causas não melhor classificadas.",
    },
    {
        "CON_DIAGES": "01",
        "Grupo SINAN": "Meningococcemia",
        "CID-10 convertido": "Não convertido",
        "Observação": "Excluído para evitar incluir pacientes sem meningite na comparação.",
    },
]

SINAN_CLASSI_FIN = {
    "1": "1 — confirmado",
    "2": "2 — descartado",
    "8": "8 — inconclusivo",
}

SINAN_EVOLUCAO = {
    "1": "1 — alta",
    "2": "2 — óbito por meningite",
    "3": "3 — óbito por outra causa",
    "9": "9 — ignorado",
}

SINAN_CRITERIO = {
    "1": "1 — cultura",
    "2": "2 — CIE",
    "3": "3 — látex",
    "4": "4 — clínico",
    "5": "5 — bacterioscopia",
    "6": "6 — quimiocitológico",
    "7": "7 — clínico-epidemiológico",
    "8": "8 — isolamento viral",
    "9": "9 — PCR",
    "10": "10 — outro",
}

YES_NO_IGN = {
    "1": "Sim",
    "2": "Não",
    "9": "Ignorado",
}

# Campos do exame quimiocitológico do líquor no SINAN.
# Os nomes abaixo seguem o dicionário SINAN NET para meningite; os seletores do app
# também aceitam variações próximas caso o banco venha renomeado.
SINAN_QUIMIO_PARAMS = {
    "hema": {"label": "Hemácias", "default_col": "LAB_HEMA"},
    "neutro": {"label": "Neutrófilos", "default_col": "LAB_NEUTRO"},
    "glico": {"label": "Glicose", "default_col": "LAB_GLICO"},
    "leuco": {"label": "Leucócitos", "default_col": "LAB_LEUCO"},
    "eosi": {"label": "Eosinófilos", "default_col": "LAB_EOSI"},
    "prot": {"label": "Proteínas", "default_col": "LAB_PROT"},
    "mono": {"label": "Monócitos", "default_col": "LAB_MONO"},
    "linfo": {"label": "Linfócitos", "default_col": "LAB_LINFO"},
    "clor": {"label": "Cloreto", "default_col": "LAB_CLOR"},
}

RACA_COR = {
    "1": "Branca",
    "2": "Preta",
    "3": "Amarela",
    "4": "Parda",
    "5": "Indígena",
    "9": "Ignorada",
}

CIHA_MODALIDADE = {
    "01": "01 — hospitalar",
    "02": "02 — ambulatorial",
}

SIM_LOCOCOR = {
    "1": "1 — hospital",
    "2": "2 — outro estabelecimento de saúde",
    "3": "3 — domicílio",
    "4": "4 — via pública",
    "5": "5 — outros",
    "9": "9 — ignorado",
}


SIM_OBITOGRAV = {
    "1": "1 — durante a gravidez",
    "2": "2 — durante o parto",
    "3": "3 — durante o aborto",
    "4": "4 — até 42 dias após o término da gestação",
    "5": "5 — 43 dias a 1 ano após o término da gestação",
    "8": "8 — não ocorreu no ciclo gravídico-puerperal",
    "9": "9 — ignorado",
}

SIM_OBITOPUERP = {
    "1": "1 — até 42 dias após o parto",
    "2": "2 — de 43 dias a 1 ano após o parto",
    "3": "3 — não ocorreu no puerpério",
    "8": "8 — não se aplica",
    "9": "9 — ignorado",
}

# Dicionário auxiliar nacional de municípios brasileiros.
# Fonte: tabela pública GOV/Receita Federal com códigos de município IBGE.
# Muitos bancos DATASUS usam os seis primeiros dígitos do código IBGE municipal.
BR_MUNICIPIOS_IBGE = {
    "000000": "Exterior/EX — IBGE 0000000",
    "110001": "Alta Floresta D'Oeste/RO — IBGE 1100015",
    "110002": "Ariquemes/RO — IBGE 1100023",
    "110003": "Cabixi/RO — IBGE 1100031",
    "110004": "Cacoal/RO — IBGE 1100049",
    "110005": "Cerejeiras/RO — IBGE 1100056",
    "110006": "Colorado do Oeste/RO — IBGE 1100064",
    "110007": "Corumbiara/RO — IBGE 1100072",
    "110008": "Costa Marques/RO — IBGE 1100080",
    "110009": "Espigão D'Oeste/RO — IBGE 1100098",
    "110010": "Guajará-Mirim/RO — IBGE 1100106",
    "110011": "Jaru/RO — IBGE 1100114",
    "110012": "Ji-Paraná/RO — IBGE 1100122",
    "110013": "Machadinho D'Oeste/RO — IBGE 1100130",
    "110014": "Nova Brasilândia D'Oeste/RO — IBGE 1100148",
    "110015": "Ouro Preto do Oeste/RO — IBGE 1100155",
    "110018": "Pimenta Bueno/RO — IBGE 1100189",
    "110020": "Porto Velho/RO — IBGE 1100205",
    "110025": "Presidente Médici/RO — IBGE 1100254",
    "110026": "Rio Crespo/RO — IBGE 1100262",
    "110028": "Rolim de Moura/RO — IBGE 1100288",
    "110029": "Santa Luzia D'Oeste/RO — IBGE 1100296",
    "110030": "Vilhena/RO — IBGE 1100304",
    "110032": "São Miguel do Guaporé/RO — IBGE 1100320",
    "110033": "Nova Mamoré/RO — IBGE 1100338",
    "110034": "Alvorada D'Oeste/RO — IBGE 1100346",
    "110037": "Alto Alegre dos Parecis/RO — IBGE 1100379",
    "110040": "Alto Paraíso/RO — IBGE 1100403",
    "110045": "Buritis/RO — IBGE 1100452",
    "110050": "Novo Horizonte do Oeste/RO — IBGE 1100502",
    "110060": "Cacaulândia/RO — IBGE 1100601",
    "110070": "Campo Novo de Rondônia/RO — IBGE 1100700",
    "110080": "Candeias do Jamari/RO — IBGE 1100809",
    "110090": "Castanheiras/RO — IBGE 1100908",
    "110092": "Chupinguaia/RO — IBGE 1100924",
    "110094": "Cujubim/RO — IBGE 1100940",
    "110100": "Governador Jorge Teixeira/RO — IBGE 1101005",
    "110110": "Itapuã do Oeste/RO — IBGE 1101104",
    "110120": "Ministro Andreazza/RO — IBGE 1101203",
    "110130": "Mirante da Serra/RO — IBGE 1101302",
    "110140": "Monte Negro/RO — IBGE 1101401",
    "110143": "Nova União/RO — IBGE 1101435",
    "110145": "Parecis/RO — IBGE 1101450",
    "110146": "Pimenteiras do Oeste/RO — IBGE 1101468",
    "110147": "Primavera de Rondônia/RO — IBGE 1101476",
    "110148": "São Felipe D'Oeste/RO — IBGE 1101484",
    "110149": "São Francisco do Guaporé/RO — IBGE 1101492",
    "110150": "Seringueiras/RO — IBGE 1101500",
    "110155": "Teixeirópolis/RO — IBGE 1101559",
    "110160": "Theobroma/RO — IBGE 1101609",
    "110170": "Urupá/RO — IBGE 1101708",
    "110175": "Vale do Anari/RO — IBGE 1101757",
    "110180": "Vale do Paraíso/RO — IBGE 1101807",
    "120001": "Acrelândia/AC — IBGE 1200013",
    "120005": "Assis Brasil/AC — IBGE 1200054",
    "120010": "Brasiléia/AC — IBGE 1200104",
    "120013": "Bujari/AC — IBGE 1200138",
    "120017": "Capixaba/AC — IBGE 1200179",
    "120020": "Cruzeiro do Sul/AC — IBGE 1200203",
    "120025": "Epitaciolândia/AC — IBGE 1200252",
    "120030": "Feijó/AC — IBGE 1200302",
    "120032": "Jordão/AC — IBGE 1200328",
    "120033": "Mâncio Lima/AC — IBGE 1200336",
    "120034": "Manoel Urbano/AC — IBGE 1200344",
    "120035": "Marechal Thaumaturgo/AC — IBGE 1200351",
    "120038": "Plácido de Castro/AC — IBGE 1200385",
    "120039": "Porto Walter/AC — IBGE 1200393",
    "120040": "Rio Branco/AC — IBGE 1200401",
    "120042": "Rodrigues Alves/AC — IBGE 1200427",
    "120043": "Santa Rosa do Purus/AC — IBGE 1200435",
    "120045": "Senador Guiomard/AC — IBGE 1200450",
    "120050": "Sena Madureira/AC — IBGE 1200500",
    "120060": "Tarauacá/AC — IBGE 1200609",
    "120070": "Xapuri/AC — IBGE 1200708",
    "120080": "Porto Acre/AC — IBGE 1200807",
    "130002": "Alvarães/AM — IBGE 1300029",
    "130006": "Amaturá/AM — IBGE 1300060",
    "130008": "Anamã/AM — IBGE 1300086",
    "130010": "Anori/AM — IBGE 1300102",
    "130014": "Apuí/AM — IBGE 1300144",
    "130020": "Atalaia do Norte/AM — IBGE 1300201",
    "130030": "Autazes/AM — IBGE 1300300",
    "130040": "Barcelos/AM — IBGE 1300409",
    "130050": "Barreirinha/AM — IBGE 1300508",
    "130060": "Benjamin Constant/AM — IBGE 1300607",
    "130063": "Beruri/AM — IBGE 1300631",
    "130068": "Boa Vista do Ramos/AM — IBGE 1300680",
    "130070": "Boca do Acre/AM — IBGE 1300706",
    "130080": "Borba/AM — IBGE 1300805",
    "130083": "Caapiranga/AM — IBGE 1300839",
    "130090": "Canutama/AM — IBGE 1300904",
    "130100": "Carauari/AM — IBGE 1301001",
    "130110": "Careiro/AM — IBGE 1301100",
    "130115": "Careiro da Várzea/AM — IBGE 1301159",
    "130120": "Coari/AM — IBGE 1301209",
    "130130": "Codajás/AM — IBGE 1301308",
    "130140": "Eirunepé/AM — IBGE 1301407",
    "130150": "Envira/AM — IBGE 1301506",
    "130160": "Fonte Boa/AM — IBGE 1301605",
    "130165": "Guajará/AM — IBGE 1301654",
    "130170": "Humaitá/AM — IBGE 1301704",
    "130180": "Ipixuna/AM — IBGE 1301803",
    "130185": "Iranduba/AM — IBGE 1301852",
    "130190": "Itacoatiara/AM — IBGE 1301902",
    "130195": "Itamarati/AM — IBGE 1301951",
    "130200": "Itapiranga/AM — IBGE 1302009",
    "130210": "Japurá/AM — IBGE 1302108",
    "130220": "Juruá/AM — IBGE 1302207",
    "130230": "Jutaí/AM — IBGE 1302306",
    "130240": "Lábrea/AM — IBGE 1302405",
    "130250": "Manacapuru/AM — IBGE 1302504",
    "130255": "Manaquiri/AM — IBGE 1302553",
    "130260": "Manaus/AM — IBGE 1302603",
    "130270": "Manicoré/AM — IBGE 1302702",
    "130280": "Maraã/AM — IBGE 1302801",
    "130290": "Maués/AM — IBGE 1302900",
    "130300": "Nhamundá/AM — IBGE 1303007",
    "130310": "Nova Olinda do Norte/AM — IBGE 1303106",
    "130320": "Novo Airão/AM — IBGE 1303205",
    "130330": "Novo Aripuanã/AM — IBGE 1303304",
    "130340": "Parintins/AM — IBGE 1303403",
    "130350": "Pauini/AM — IBGE 1303502",
    "130353": "Presidente Figueiredo/AM — IBGE 1303536",
    "130356": "Rio Preto da Eva/AM — IBGE 1303569",
    "130360": "Santa Isabel do Rio Negro/AM — IBGE 1303601",
    "130370": "Santo Antônio do Içá/AM — IBGE 1303700",
    "130380": "São Gabriel da Cachoeira/AM — IBGE 1303809",
    "130390": "São Paulo de Olivença/AM — IBGE 1303908",
    "130395": "São Sebastião do Uatumã/AM — IBGE 1303957",
    "130400": "Silves/AM — IBGE 1304005",
    "130406": "Tabatinga/AM — IBGE 1304062",
    "130410": "Tapauá/AM — IBGE 1304104",
    "130420": "Tefé/AM — IBGE 1304203",
    "130423": "Tonantins/AM — IBGE 1304237",
    "130426": "Uarini/AM — IBGE 1304260",
    "130430": "Urucará/AM — IBGE 1304302",
    "130440": "Urucurituba/AM — IBGE 1304401",
    "140002": "Amajari/RR — IBGE 1400027",
    "140005": "Alto Alegre/RR — IBGE 1400050",
    "140010": "Boa Vista/RR — IBGE 1400100",
    "140015": "Bonfim/RR — IBGE 1400159",
    "140017": "Cantá/RR — IBGE 1400175",
    "140020": "Caracaraí/RR — IBGE 1400209",
    "140023": "Caroebe/RR — IBGE 1400233",
    "140028": "Iracema/RR — IBGE 1400282",
    "140030": "Mucajaí/RR — IBGE 1400308",
    "140040": "Normandia/RR — IBGE 1400407",
    "140045": "Pacaraima/RR — IBGE 1400456",
    "140047": "Rorainópolis/RR — IBGE 1400472",
    "140050": "São João da Baliza/RR — IBGE 1400506",
    "140060": "São Luiz/RR — IBGE 1400605",
    "140070": "Uiramutã/RR — IBGE 1400704",
    "150010": "Abaetetuba/PA — IBGE 1500107",
    "150013": "Abel Figueiredo/PA — IBGE 1500131",
    "150020": "Acará/PA — IBGE 1500206",
    "150030": "Afuá/PA — IBGE 1500305",
    "150034": "Água Azul do Norte/PA — IBGE 1500347",
    "150040": "Alenquer/PA — IBGE 1500404",
    "150050": "Almeirim/PA — IBGE 1500503",
    "150060": "Altamira/PA — IBGE 1500602",
    "150070": "Anajás/PA — IBGE 1500701",
    "150080": "Ananindeua/PA — IBGE 1500800",
    "150085": "Anapu/PA — IBGE 1500859",
    "150090": "Augusto Corrêa/PA — IBGE 1500909",
    "150095": "Aurora do Pará/PA — IBGE 1500958",
    "150100": "Aveiro/PA — IBGE 1501006",
    "150110": "Bagre/PA — IBGE 1501105",
    "150120": "Baião/PA — IBGE 1501204",
    "150125": "Bannach/PA — IBGE 1501253",
    "150130": "Barcarena/PA — IBGE 1501303",
    "150140": "Belém/PA — IBGE 1501402",
    "150145": "Belterra/PA — IBGE 1501451",
    "150150": "Benevides/PA — IBGE 1501501",
    "150157": "Bom Jesus do Tocantins/PA — IBGE 1501576",
    "150160": "Bonito/PA — IBGE 1501600",
    "150170": "Bragança/PA — IBGE 1501709",
    "150172": "Brasil Novo/PA — IBGE 1501725",
    "150175": "Brejo Grande do Araguaia/PA — IBGE 1501758",
    "150178": "Breu Branco/PA — IBGE 1501782",
    "150180": "Breves/PA — IBGE 1501808",
    "150190": "Bujaru/PA — IBGE 1501907",
    "150195": "Cachoeira do Piriá/PA — IBGE 1501956",
    "150200": "Cachoeira do Arari/PA — IBGE 1502004",
    "150210": "Cametá/PA — IBGE 1502103",
    "150215": "Canaã dos Carajás/PA — IBGE 1502152",
    "150220": "Capanema/PA — IBGE 1502202",
    "150230": "Capitão Poço/PA — IBGE 1502301",
    "150240": "Castanhal/PA — IBGE 1502400",
    "150250": "Chaves/PA — IBGE 1502509",
    "150260": "Colares/PA — IBGE 1502608",
    "150270": "Conceição do Araguaia/PA — IBGE 1502707",
    "150275": "Concórdia do Pará/PA — IBGE 1502756",
    "150276": "Cumaru do Norte/PA — IBGE 1502764",
    "150277": "Curionópolis/PA — IBGE 1502772",
    "150280": "Curralinho/PA — IBGE 1502806",
    "150285": "Curuá/PA — IBGE 1502855",
    "150290": "Curuçá/PA — IBGE 1502905",
    "150293": "Dom Eliseu/PA — IBGE 1502939",
    "150295": "Eldorado do Carajás/PA — IBGE 1502954",
    "150300": "Faro/PA — IBGE 1503002",
    "150304": "Floresta do Araguaia/PA — IBGE 1503044",
    "150307": "Garrafão do Norte/PA — IBGE 1503077",
    "150309": "Goianésia do Pará/PA — IBGE 1503093",
    "150310": "Gurupá/PA — IBGE 1503101",
    "150320": "Igarapé-Açu/PA — IBGE 1503200",
    "150330": "Igarapé-Miri/PA — IBGE 1503309",
    "150340": "Inhangapi/PA — IBGE 1503408",
    "150345": "Ipixuna do Pará/PA — IBGE 1503457",
    "150350": "Irituia/PA — IBGE 1503507",
    "150360": "Itaituba/PA — IBGE 1503606",
    "150370": "Itupiranga/PA — IBGE 1503705",
    "150375": "Jacareacanga/PA — IBGE 1503754",
    "150380": "Jacundá/PA — IBGE 1503804",
    "150390": "Juruti/PA — IBGE 1503903",
    "150400": "Limoeiro do Ajuru/PA — IBGE 1504000",
    "150405": "Mãe do Rio/PA — IBGE 1504059",
    "150410": "Magalhães Barata/PA — IBGE 1504109",
    "150420": "Marabá/PA — IBGE 1504208",
    "150430": "Maracanã/PA — IBGE 1504307",
    "150440": "Marapanim/PA — IBGE 1504406",
    "150442": "Marituba/PA — IBGE 1504422",
    "150445": "Medicilândia/PA — IBGE 1504455",
    "150450": "Melgaço/PA — IBGE 1504505",
    "150460": "Mocajuba/PA — IBGE 1504604",
    "150470": "Moju/PA — IBGE 1504703",
    "150475": "Mojuí dos Campos/PA — IBGE 1504752",
    "150480": "Monte Alegre/PA — IBGE 1504802",
    "150490": "Muaná/PA — IBGE 1504901",
    "150495": "Nova Esperança do Piriá/PA — IBGE 1504950",
    "150497": "Nova Ipixuna/PA — IBGE 1504976",
    "150500": "Nova Timboteua/PA — IBGE 1505007",
    "150503": "Novo Progresso/PA — IBGE 1505031",
    "150506": "Novo Repartimento/PA — IBGE 1505064",
    "150510": "Óbidos/PA — IBGE 1505106",
    "150520": "Oeiras do Pará/PA — IBGE 1505205",
    "150530": "Oriximiná/PA — IBGE 1505304",
    "150540": "Ourém/PA — IBGE 1505403",
    "150543": "Ourilândia do Norte/PA — IBGE 1505437",
    "150548": "Pacajá/PA — IBGE 1505486",
    "150549": "Palestina do Pará/PA — IBGE 1505494",
    "150550": "Paragominas/PA — IBGE 1505502",
    "150553": "Parauapebas/PA — IBGE 1505536",
    "150555": "Pau D'Arco/PA — IBGE 1505551",
    "150560": "Peixe-Boi/PA — IBGE 1505601",
    "150563": "Piçarra/PA — IBGE 1505635",
    "150565": "Placas/PA — IBGE 1505650",
    "150570": "Ponta de Pedras/PA — IBGE 1505700",
    "150580": "Portel/PA — IBGE 1505809",
    "150590": "Porto de Moz/PA — IBGE 1505908",
    "150600": "Prainha/PA — IBGE 1506005",
    "150610": "Primavera/PA — IBGE 1506104",
    "150611": "Quatipuru/PA — IBGE 1506112",
    "150613": "Redenção/PA — IBGE 1506138",
    "150616": "Rio Maria/PA — IBGE 1506161",
    "150618": "Rondon do Pará/PA — IBGE 1506187",
    "150619": "Rurópolis/PA — IBGE 1506195",
    "150620": "Salinópolis/PA — IBGE 1506203",
    "150630": "Salvaterra/PA — IBGE 1506302",
    "150635": "Santa Bárbara do Pará/PA — IBGE 1506351",
    "150640": "Santa Cruz do Arari/PA — IBGE 1506401",
    "150650": "Santa Izabel do Pará/PA — IBGE 1506500",
    "150655": "Santa Luzia do Pará/PA — IBGE 1506559",
    "150658": "Santa Maria das Barreiras/PA — IBGE 1506583",
    "150660": "Santa Maria do Pará/PA — IBGE 1506609",
    "150670": "Santana do Araguaia/PA — IBGE 1506708",
    "150680": "Santarém/PA — IBGE 1506807",
    "150690": "Santarém Novo/PA — IBGE 1506906",
    "150700": "Santo Antônio do Tauá/PA — IBGE 1507003",
    "150710": "São Caetano de Odivelas/PA — IBGE 1507102",
    "150715": "São Domingos do Araguaia/PA — IBGE 1507151",
    "150720": "São Domingos do Capim/PA — IBGE 1507201",
    "150730": "São Félix do Xingu/PA — IBGE 1507300",
    "150740": "São Francisco do Pará/PA — IBGE 1507409",
    "150745": "São Geraldo do Araguaia/PA — IBGE 1507458",
    "150746": "São João da Ponta/PA — IBGE 1507466",
    "150747": "São João de Pirabas/PA — IBGE 1507474",
    "150750": "São João do Araguaia/PA — IBGE 1507508",
    "150760": "São Miguel do Guamá/PA — IBGE 1507607",
    "150770": "São Sebastião da Boa Vista/PA — IBGE 1507706",
    "150775": "Sapucaia/PA — IBGE 1507755",
    "150780": "Senador José Porfírio/PA — IBGE 1507805",
    "150790": "Soure/PA — IBGE 1507904",
    "150795": "Tailândia/PA — IBGE 1507953",
    "150796": "Terra Alta/PA — IBGE 1507961",
    "150797": "Terra Santa/PA — IBGE 1507979",
    "150800": "Tomé-Açu/PA — IBGE 1508001",
    "150803": "Tracuateua/PA — IBGE 1508035",
    "150805": "Trairão/PA — IBGE 1508050",
    "150808": "Tucumã/PA — IBGE 1508084",
    "150810": "Tucuruí/PA — IBGE 1508100",
    "150812": "Ulianópolis/PA — IBGE 1508126",
    "150815": "Uruará/PA — IBGE 1508159",
    "150820": "Vigia/PA — IBGE 1508209",
    "150830": "Viseu/PA — IBGE 1508308",
    "150835": "Vitória do Xingu/PA — IBGE 1508357",
    "150840": "Xinguara/PA — IBGE 1508407",
    "160005": "Serra do Navio/AP — IBGE 1600055",
    "160010": "Amapá/AP — IBGE 1600105",
    "160015": "Pedra Branca do Amapari/AP — IBGE 1600154",
    "160020": "Calçoene/AP — IBGE 1600204",
    "160021": "Cutias/AP — IBGE 1600212",
    "160023": "Ferreira Gomes/AP — IBGE 1600238",
    "160025": "Itaubal/AP — IBGE 1600253",
    "160027": "Laranjal do Jari/AP — IBGE 1600279",
    "160030": "Macapá/AP — IBGE 1600303",
    "160040": "Mazagão/AP — IBGE 1600402",
    "160050": "Oiapoque/AP — IBGE 1600501",
    "160053": "Porto Grande/AP — IBGE 1600535",
    "160055": "Pracuúba/AP — IBGE 1600550",
    "160060": "Santana/AP — IBGE 1600600",
    "160070": "Tartarugalzinho/AP — IBGE 1600709",
    "160080": "Vitória do Jari/AP — IBGE 1600808",
    "170025": "Abreulândia/TO — IBGE 1700251",
    "170030": "Aguiarnópolis/TO — IBGE 1700301",
    "170035": "Aliança do Tocantins/TO — IBGE 1700350",
    "170040": "Almas/TO — IBGE 1700400",
    "170070": "Alvorada/TO — IBGE 1700707",
    "170100": "Ananás/TO — IBGE 1701002",
    "170105": "Angico/TO — IBGE 1701051",
    "170110": "Aparecida do Rio Negro/TO — IBGE 1701101",
    "170130": "Aragominas/TO — IBGE 1701309",
    "170190": "Araguacema/TO — IBGE 1701903",
    "170200": "Araguaçu/TO — IBGE 1702000",
    "170210": "Araguaína/TO — IBGE 1702109",
    "170215": "Araguanã/TO — IBGE 1702158",
    "170220": "Araguatins/TO — IBGE 1702208",
    "170230": "Arapoema/TO — IBGE 1702307",
    "170240": "Arraias/TO — IBGE 1702406",
    "170255": "Augustinópolis/TO — IBGE 1702554",
    "170270": "Aurora do Tocantins/TO — IBGE 1702703",
    "170290": "Axixá do Tocantins/TO — IBGE 1702901",
    "170300": "Babaçulândia/TO — IBGE 1703008",
    "170305": "Bandeirantes do Tocantins/TO — IBGE 1703057",
    "170307": "Barra do Ouro/TO — IBGE 1703073",
    "170310": "Barrolândia/TO — IBGE 1703107",
    "170320": "Bernardo Sayão/TO — IBGE 1703206",
    "170330": "Bom Jesus do Tocantins/TO — IBGE 1703305",
    "170360": "Brasilândia do Tocantins/TO — IBGE 1703602",
    "170370": "Brejinho de Nazaré/TO — IBGE 1703701",
    "170380": "Buriti do Tocantins/TO — IBGE 1703800",
    "170382": "Cachoeirinha/TO — IBGE 1703826",
    "170384": "Campos Lindos/TO — IBGE 1703842",
    "170386": "Cariri do Tocantins/TO — IBGE 1703867",
    "170388": "Carmolândia/TO — IBGE 1703883",
    "170389": "Carrasco Bonito/TO — IBGE 1703891",
    "170390": "Caseara/TO — IBGE 1703909",
    "170410": "Centenário/TO — IBGE 1704105",
    "170460": "Chapada de Areia/TO — IBGE 1704600",
    "170510": "Chapada da Natividade/TO — IBGE 1705102",
    "170550": "Colinas do Tocantins/TO — IBGE 1705508",
    "170555": "Combinado/TO — IBGE 1705557",
    "170560": "Conceição do Tocantins/TO — IBGE 1705607",
    "170600": "Couto Magalhães/TO — IBGE 1706001",
    "170610": "Cristalândia/TO — IBGE 1706100",
    "170625": "Crixás do Tocantins/TO — IBGE 1706258",
    "170650": "Darcinópolis/TO — IBGE 1706506",
    "170700": "Dianópolis/TO — IBGE 1707009",
    "170710": "Divinópolis do Tocantins/TO — IBGE 1707108",
    "170720": "Dois Irmãos do Tocantins/TO — IBGE 1707207",
    "170730": "Dueré/TO — IBGE 1707306",
    "170740": "Esperantina/TO — IBGE 1707405",
    "170755": "Fátima/TO — IBGE 1707553",
    "170765": "Figueirópolis/TO — IBGE 1707652",
    "170770": "Filadélfia/TO — IBGE 1707702",
    "170820": "Formoso do Araguaia/TO — IBGE 1708205",
    "170825": "Tabocão/TO — IBGE 1708254",
    "170830": "Goianorte/TO — IBGE 1708304",
    "170900": "Goiatins/TO — IBGE 1709005",
    "170930": "Guaraí/TO — IBGE 1709302",
    "170950": "Gurupi/TO — IBGE 1709500",
    "170980": "Ipueiras/TO — IBGE 1709807",
    "171050": "Itacajá/TO — IBGE 1710508",
    "171070": "Itaguatins/TO — IBGE 1710706",
    "171090": "Itapiratins/TO — IBGE 1710904",
    "171110": "Itaporã do Tocantins/TO — IBGE 1711100",
    "171150": "Jaú do Tocantins/TO — IBGE 1711506",
    "171180": "Juarina/TO — IBGE 1711803",
    "171190": "Lagoa da Confusão/TO — IBGE 1711902",
    "171195": "Lagoa do Tocantins/TO — IBGE 1711951",
    "171200": "Lajeado/TO — IBGE 1712009",
    "171215": "Lavandeira/TO — IBGE 1712157",
    "171240": "Lizarda/TO — IBGE 1712405",
    "171245": "Luzinópolis/TO — IBGE 1712454",
    "171250": "Marianópolis do Tocantins/TO — IBGE 1712504",
    "171270": "Mateiros/TO — IBGE 1712702",
    "171280": "Maurilândia do Tocantins/TO — IBGE 1712801",
    "171320": "Miracema do Tocantins/TO — IBGE 1713205",
    "171330": "Miranorte/TO — IBGE 1713304",
    "171360": "Monte do Carmo/TO — IBGE 1713601",
    "171370": "Monte Santo do Tocantins/TO — IBGE 1713700",
    "171380": "Palmeiras do Tocantins/TO — IBGE 1713809",
    "171395": "Muricilândia/TO — IBGE 1713957",
    "171420": "Natividade/TO — IBGE 1714203",
    "171430": "Nazaré/TO — IBGE 1714302",
    "171488": "Nova Olinda/TO — IBGE 1714880",
    "171500": "Nova Rosalândia/TO — IBGE 1715002",
    "171510": "Novo Acordo/TO — IBGE 1715101",
    "171515": "Novo Alegre/TO — IBGE 1715150",
    "171525": "Novo Jardim/TO — IBGE 1715259",
    "171550": "Oliveira de Fátima/TO — IBGE 1715507",
    "171570": "Palmeirante/TO — IBGE 1715705",
    "171575": "Palmeirópolis/TO — IBGE 1715754",
    "171610": "Paraíso do Tocantins/TO — IBGE 1716109",
    "171620": "Paranã/TO — IBGE 1716208",
    "171630": "Pau D'Arco/TO — IBGE 1716307",
    "171650": "Pedro Afonso/TO — IBGE 1716505",
    "171660": "Peixe/TO — IBGE 1716604",
    "171665": "Pequizeiro/TO — IBGE 1716653",
    "171670": "Colméia/TO — IBGE 1716703",
    "171700": "Pindorama do Tocantins/TO — IBGE 1717008",
    "171720": "Piraquê/TO — IBGE 1717206",
    "171750": "Pium/TO — IBGE 1717503",
    "171780": "Ponte Alta do Bom Jesus/TO — IBGE 1717800",
    "171790": "Ponte Alta do Tocantins/TO — IBGE 1717909",
    "171800": "Porto Alegre do Tocantins/TO — IBGE 1718006",
    "171820": "Porto Nacional/TO — IBGE 1718204",
    "171830": "Praia Norte/TO — IBGE 1718303",
    "171840": "Presidente Kennedy/TO — IBGE 1718402",
    "171845": "Pugmil/TO — IBGE 1718451",
    "171850": "Recursolândia/TO — IBGE 1718501",
    "171855": "Riachinho/TO — IBGE 1718550",
    "171865": "Rio da Conceição/TO — IBGE 1718659",
    "171870": "Rio dos Bois/TO — IBGE 1718709",
    "171875": "Rio Sono/TO — IBGE 1718758",
    "171880": "Sampaio/TO — IBGE 1718808",
    "171884": "Sandolândia/TO — IBGE 1718840",
    "171886": "Santa Fé do Araguaia/TO — IBGE 1718865",
    "171888": "Santa Maria do Tocantins/TO — IBGE 1718881",
    "171889": "Santa Rita do Tocantins/TO — IBGE 1718899",
    "171890": "Santa Rosa do Tocantins/TO — IBGE 1718907",
    "171900": "Santa Tereza do Tocantins/TO — IBGE 1719004",
    "172000": "Santa Terezinha do Tocantins/TO — IBGE 1720002",
    "172010": "São Bento do Tocantins/TO — IBGE 1720101",
    "172015": "São Félix do Tocantins/TO — IBGE 1720150",
    "172020": "São Miguel do Tocantins/TO — IBGE 1720200",
    "172025": "São Salvador do Tocantins/TO — IBGE 1720259",
    "172030": "São Sebastião do Tocantins/TO — IBGE 1720309",
    "172049": "São Valério/TO — IBGE 1720499",
    "172065": "Silvanópolis/TO — IBGE 1720655",
    "172080": "Sítio Novo do Tocantins/TO — IBGE 1720804",
    "172085": "Sucupira/TO — IBGE 1720853",
    "172090": "Taguatinga/TO — IBGE 1720903",
    "172093": "Taipas do Tocantins/TO — IBGE 1720937",
    "172097": "Talismã/TO — IBGE 1720978",
    "172100": "Palmas/TO — IBGE 1721000",
    "172110": "Tocantínia/TO — IBGE 1721109",
    "172120": "Tocantinópolis/TO — IBGE 1721208",
    "172125": "Tupirama/TO — IBGE 1721257",
    "172130": "Tupiratins/TO — IBGE 1721307",
    "172208": "Wanderlândia/TO — IBGE 1722081",
    "172210": "Xambioá/TO — IBGE 1722107",
    "210005": "Açailândia/MA — IBGE 2100055",
    "210010": "Afonso Cunha/MA — IBGE 2100105",
    "210015": "Água Doce do Maranhão/MA — IBGE 2100154",
    "210020": "Alcântara/MA — IBGE 2100204",
    "210030": "Aldeias Altas/MA — IBGE 2100303",
    "210040": "Altamira do Maranhão/MA — IBGE 2100402",
    "210043": "Alto Alegre do Maranhão/MA — IBGE 2100436",
    "210047": "Alto Alegre do Pindaré/MA — IBGE 2100477",
    "210050": "Alto Parnaíba/MA — IBGE 2100501",
    "210055": "Amapá do Maranhão/MA — IBGE 2100550",
    "210060": "Amarante do Maranhão/MA — IBGE 2100600",
    "210070": "Anajatuba/MA — IBGE 2100709",
    "210080": "Anapurus/MA — IBGE 2100808",
    "210083": "Apicum-Açu/MA — IBGE 2100832",
    "210087": "Araguanã/MA — IBGE 2100873",
    "210090": "Araioses/MA — IBGE 2100907",
    "210095": "Arame/MA — IBGE 2100956",
    "210100": "Arari/MA — IBGE 2101004",
    "210110": "Axixá/MA — IBGE 2101103",
    "210120": "Bacabal/MA — IBGE 2101202",
    "210125": "Bacabeira/MA — IBGE 2101251",
    "210130": "Bacuri/MA — IBGE 2101301",
    "210135": "Bacurituba/MA — IBGE 2101350",
    "210140": "Balsas/MA — IBGE 2101400",
    "210150": "Barão de Grajaú/MA — IBGE 2101509",
    "210160": "Barra do Corda/MA — IBGE 2101608",
    "210170": "Barreirinhas/MA — IBGE 2101707",
    "210173": "Belágua/MA — IBGE 2101731",
    "210177": "Bela Vista do Maranhão/MA — IBGE 2101772",
    "210180": "Benedito Leite/MA — IBGE 2101806",
    "210190": "Bequimão/MA — IBGE 2101905",
    "210193": "Bernardo do Mearim/MA — IBGE 2101939",
    "210197": "Boa Vista do Gurupi/MA — IBGE 2101970",
    "210200": "Bom Jardim/MA — IBGE 2102002",
    "210203": "Bom Jesus das Selvas/MA — IBGE 2102036",
    "210207": "Bom Lugar/MA — IBGE 2102077",
    "210210": "Brejo/MA — IBGE 2102101",
    "210215": "Brejo de Areia/MA — IBGE 2102150",
    "210220": "Buriti/MA — IBGE 2102200",
    "210230": "Buriti Bravo/MA — IBGE 2102309",
    "210232": "Buriticupu/MA — IBGE 2102325",
    "210235": "Buritirana/MA — IBGE 2102358",
    "210237": "Cachoeira Grande/MA — IBGE 2102374",
    "210240": "Cajapió/MA — IBGE 2102408",
    "210250": "Cajari/MA — IBGE 2102507",
    "210255": "Campestre do Maranhão/MA — IBGE 2102556",
    "210260": "Cândido Mendes/MA — IBGE 2102606",
    "210270": "Cantanhede/MA — IBGE 2102705",
    "210275": "Capinzal do Norte/MA — IBGE 2102754",
    "210280": "Carolina/MA — IBGE 2102804",
    "210290": "Carutapera/MA — IBGE 2102903",
    "210300": "Caxias/MA — IBGE 2103000",
    "210310": "Cedral/MA — IBGE 2103109",
    "210312": "Central do Maranhão/MA — IBGE 2103125",
    "210315": "Centro do Guilherme/MA — IBGE 2103158",
    "210317": "Centro Novo do Maranhão/MA — IBGE 2103174",
    "210320": "Chapadinha/MA — IBGE 2103208",
    "210325": "Cidelândia/MA — IBGE 2103257",
    "210330": "Codó/MA — IBGE 2103307",
    "210340": "Coelho Neto/MA — IBGE 2103406",
    "210350": "Colinas/MA — IBGE 2103505",
    "210355": "Conceição do Lago-Açu/MA — IBGE 2103554",
    "210360": "Coroatá/MA — IBGE 2103604",
    "210370": "Cururupu/MA — IBGE 2103703",
    "210375": "Davinópolis/MA — IBGE 2103752",
    "210380": "Dom Pedro/MA — IBGE 2103802",
    "210390": "Duque Bacelar/MA — IBGE 2103901",
    "210400": "Esperantinópolis/MA — IBGE 2104008",
    "210405": "Estreito/MA — IBGE 2104057",
    "210407": "Feira Nova do Maranhão/MA — IBGE 2104073",
    "210408": "Fernando Falcão/MA — IBGE 2104081",
    "210409": "Formosa da Serra Negra/MA — IBGE 2104099",
    "210410": "Fortaleza dos Nogueiras/MA — IBGE 2104107",
    "210420": "Fortuna/MA — IBGE 2104206",
    "210430": "Godofredo Viana/MA — IBGE 2104305",
    "210440": "Gonçalves Dias/MA — IBGE 2104404",
    "210450": "Governador Archer/MA — IBGE 2104503",
    "210455": "Governador Edison Lobão/MA — IBGE 2104552",
    "210460": "Governador Eugênio Barros/MA — IBGE 2104602",
    "210462": "Governador Luiz Rocha/MA — IBGE 2104628",
    "210465": "Governador Newton Bello/MA — IBGE 2104651",
    "210467": "Governador Nunes Freire/MA — IBGE 2104677",
    "210470": "Graça Aranha/MA — IBGE 2104701",
    "210480": "Grajaú/MA — IBGE 2104800",
    "210490": "Guimarães/MA — IBGE 2104909",
    "210500": "Humberto de Campos/MA — IBGE 2105005",
    "210510": "Icatu/MA — IBGE 2105104",
    "210515": "Igarapé do Meio/MA — IBGE 2105153",
    "210520": "Igarapé Grande/MA — IBGE 2105203",
    "210530": "Imperatriz/MA — IBGE 2105302",
    "210535": "Itaipava do Grajaú/MA — IBGE 2105351",
    "210540": "Itapecuru Mirim/MA — IBGE 2105401",
    "210542": "Itinga do Maranhão/MA — IBGE 2105427",
    "210545": "Jatobá/MA — IBGE 2105450",
    "210547": "Jenipapo dos Vieiras/MA — IBGE 2105476",
    "210550": "João Lisboa/MA — IBGE 2105500",
    "210560": "Joselândia/MA — IBGE 2105609",
    "210565": "Junco do Maranhão/MA — IBGE 2105658",
    "210570": "Lago da Pedra/MA — IBGE 2105708",
    "210580": "Lago do Junco/MA — IBGE 2105807",
    "210590": "Lago Verde/MA — IBGE 2105906",
    "210592": "Lagoa do Mato/MA — IBGE 2105922",
    "210594": "Lago dos Rodrigues/MA — IBGE 2105948",
    "210596": "Lagoa Grande do Maranhão/MA — IBGE 2105963",
    "210598": "Lajeado Novo/MA — IBGE 2105989",
    "210600": "Lima Campos/MA — IBGE 2106003",
    "210610": "Loreto/MA — IBGE 2106102",
    "210620": "Luís Domingues/MA — IBGE 2106201",
    "210630": "Magalhães de Almeida/MA — IBGE 2106300",
    "210632": "Maracaçumé/MA — IBGE 2106326",
    "210635": "Marajá do Sena/MA — IBGE 2106359",
    "210637": "Maranhãozinho/MA — IBGE 2106375",
    "210640": "Mata Roma/MA — IBGE 2106409",
    "210650": "Matinha/MA — IBGE 2106508",
    "210660": "Matões/MA — IBGE 2106607",
    "210663": "Matões do Norte/MA — IBGE 2106631",
    "210667": "Milagres do Maranhão/MA — IBGE 2106672",
    "210670": "Mirador/MA — IBGE 2106706",
    "210675": "Miranda do Norte/MA — IBGE 2106755",
    "210680": "Mirinzal/MA — IBGE 2106805",
    "210690": "Monção/MA — IBGE 2106904",
    "210700": "Montes Altos/MA — IBGE 2107001",
    "210710": "Morros/MA — IBGE 2107100",
    "210720": "Nina Rodrigues/MA — IBGE 2107209",
    "210725": "Nova Colinas/MA — IBGE 2107258",
    "210730": "Nova Iorque/MA — IBGE 2107308",
    "210735": "Nova Olinda do Maranhão/MA — IBGE 2107357",
    "210740": "Olho d'Água das Cunhãs/MA — IBGE 2107407",
    "210745": "Olinda Nova do Maranhão/MA — IBGE 2107456",
    "210750": "Paço do Lumiar/MA — IBGE 2107506",
    "210760": "Palmeirândia/MA — IBGE 2107605",
    "210770": "Paraibano/MA — IBGE 2107704",
    "210780": "Parnarama/MA — IBGE 2107803",
    "210790": "Passagem Franca/MA — IBGE 2107902",
    "210800": "Pastos Bons/MA — IBGE 2108009",
    "210805": "Paulino Neves/MA — IBGE 2108058",
    "210810": "Paulo Ramos/MA — IBGE 2108108",
    "210820": "Pedreiras/MA — IBGE 2108207",
    "210825": "Pedro do Rosário/MA — IBGE 2108256",
    "210830": "Penalva/MA — IBGE 2108306",
    "210840": "Peri Mirim/MA — IBGE 2108405",
    "210845": "Peritoró/MA — IBGE 2108454",
    "210850": "Pindaré-Mirim/MA — IBGE 2108504",
    "210860": "Pinheiro/MA — IBGE 2108603",
    "210870": "Pio XII/MA — IBGE 2108702",
    "210880": "Pirapemas/MA — IBGE 2108801",
    "210890": "Poção de Pedras/MA — IBGE 2108900",
    "210900": "Porto Franco/MA — IBGE 2109007",
    "210905": "Porto Rico do Maranhão/MA — IBGE 2109056",
    "210910": "Presidente Dutra/MA — IBGE 2109106",
    "210920": "Presidente Juscelino/MA — IBGE 2109205",
    "210923": "Presidente Médici/MA — IBGE 2109239",
    "210927": "Presidente Sarney/MA — IBGE 2109270",
    "210930": "Presidente Vargas/MA — IBGE 2109304",
    "210940": "Primeira Cruz/MA — IBGE 2109403",
    "210945": "Raposa/MA — IBGE 2109452",
    "210950": "Riachão/MA — IBGE 2109502",
    "210955": "Ribamar Fiquene/MA — IBGE 2109551",
    "210960": "Rosário/MA — IBGE 2109601",
    "210970": "Sambaíba/MA — IBGE 2109700",
    "210975": "Santa Filomena do Maranhão/MA — IBGE 2109759",
    "210980": "Santa Helena/MA — IBGE 2109809",
    "210990": "Santa Inês/MA — IBGE 2109908",
    "211000": "Santa Luzia/MA — IBGE 2110005",
    "211003": "Santa Luzia do Paruá/MA — IBGE 2110039",
    "211010": "Santa Quitéria do Maranhão/MA — IBGE 2110104",
    "211020": "Santa Rita/MA — IBGE 2110203",
    "211023": "Santana do Maranhão/MA — IBGE 2110237",
    "211027": "Santo Amaro do Maranhão/MA — IBGE 2110278",
    "211030": "Santo Antônio dos Lopes/MA — IBGE 2110302",
    "211040": "São Benedito do Rio Preto/MA — IBGE 2110401",
    "211050": "São Bento/MA — IBGE 2110500",
    "211060": "São Bernardo/MA — IBGE 2110609",
    "211065": "São Domingos do Azeitão/MA — IBGE 2110658",
    "211070": "São Domingos do Maranhão/MA — IBGE 2110708",
    "211080": "São Félix de Balsas/MA — IBGE 2110807",
    "211085": "São Francisco do Brejão/MA — IBGE 2110856",
    "211090": "São Francisco do Maranhão/MA — IBGE 2110906",
    "211100": "São João Batista/MA — IBGE 2111003",
    "211102": "São João do Carú/MA — IBGE 2111029",
    "211105": "São João do Paraíso/MA — IBGE 2111052",
    "211107": "São João do Soter/MA — IBGE 2111078",
    "211110": "São João dos Patos/MA — IBGE 2111102",
    "211120": "São José de Ribamar/MA — IBGE 2111201",
    "211125": "São José dos Basílios/MA — IBGE 2111250",
    "211130": "São Luís/MA — IBGE 2111300",
    "211140": "São Luís Gonzaga do Maranhão/MA — IBGE 2111409",
    "211150": "São Mateus do Maranhão/MA — IBGE 2111508",
    "211153": "São Pedro da Água Branca/MA — IBGE 2111532",
    "211157": "São Pedro dos Crentes/MA — IBGE 2111573",
    "211160": "São Raimundo das Mangabeiras/MA — IBGE 2111607",
    "211163": "São Raimundo do Doca Bezerra/MA — IBGE 2111631",
    "211167": "São Roberto/MA — IBGE 2111672",
    "211170": "São Vicente Ferrer/MA — IBGE 2111706",
    "211172": "Satubinha/MA — IBGE 2111722",
    "211174": "Senador Alexandre Costa/MA — IBGE 2111748",
    "211176": "Senador La Rocque/MA — IBGE 2111763",
    "211178": "Serrano do Maranhão/MA — IBGE 2111789",
    "211180": "Sítio Novo/MA — IBGE 2111805",
    "211190": "Sucupira do Norte/MA — IBGE 2111904",
    "211195": "Sucupira do Riachão/MA — IBGE 2111953",
    "211200": "Tasso Fragoso/MA — IBGE 2112001",
    "211210": "Timbiras/MA — IBGE 2112100",
    "211220": "Timon/MA — IBGE 2112209",
    "211223": "Trizidela do Vale/MA — IBGE 2112233",
    "211227": "Tufilândia/MA — IBGE 2112274",
    "211230": "Tuntum/MA — IBGE 2112308",
    "211240": "Turiaçu/MA — IBGE 2112407",
    "211245": "Turilândia/MA — IBGE 2112456",
    "211250": "Tutóia/MA — IBGE 2112506",
    "211260": "Urbano Santos/MA — IBGE 2112605",
    "211270": "Vargem Grande/MA — IBGE 2112704",
    "211280": "Viana/MA — IBGE 2112803",
    "211285": "Vila Nova dos Martírios/MA — IBGE 2112852",
    "211290": "Vitória do Mearim/MA — IBGE 2112902",
    "211300": "Vitorino Freire/MA — IBGE 2113009",
    "211400": "Zé Doca/MA — IBGE 2114007",
    "220005": "Acauã/PI — IBGE 2200053",
    "220010": "Agricolândia/PI — IBGE 2200103",
    "220020": "Água Branca/PI — IBGE 2200202",
    "220025": "Alagoinha do Piauí/PI — IBGE 2200251",
    "220027": "Alegrete do Piauí/PI — IBGE 2200277",
    "220030": "Alto Longá/PI — IBGE 2200301",
    "220040": "Altos/PI — IBGE 2200400",
    "220045": "Alvorada do Gurguéia/PI — IBGE 2200459",
    "220050": "Amarante/PI — IBGE 2200509",
    "220060": "Angical do Piauí/PI — IBGE 2200608",
    "220070": "Anísio de Abreu/PI — IBGE 2200707",
    "220080": "Antônio Almeida/PI — IBGE 2200806",
    "220090": "Aroazes/PI — IBGE 2200905",
    "220095": "Aroeiras do Itaim/PI — IBGE 2200954",
    "220100": "Arraial/PI — IBGE 2201002",
    "220105": "Assunção do Piauí/PI — IBGE 2201051",
    "220110": "Avelino Lopes/PI — IBGE 2201101",
    "220115": "Baixa Grande do Ribeiro/PI — IBGE 2201150",
    "220117": "Barra D'Alcântara/PI — IBGE 2201176",
    "220120": "Barras/PI — IBGE 2201200",
    "220130": "Barreiras do Piauí/PI — IBGE 2201309",
    "220140": "Barro Duro/PI — IBGE 2201408",
    "220150": "Batalha/PI — IBGE 2201507",
    "220155": "Bela Vista do Piauí/PI — IBGE 2201556",
    "220157": "Belém do Piauí/PI — IBGE 2201572",
    "220160": "Beneditinos/PI — IBGE 2201606",
    "220170": "Bertolínia/PI — IBGE 2201705",
    "220173": "Betânia do Piauí/PI — IBGE 2201739",
    "220177": "Boa Hora/PI — IBGE 2201770",
    "220180": "Bocaina/PI — IBGE 2201804",
    "220190": "Bom Jesus/PI — IBGE 2201903",
    "220191": "Bom Princípio do Piauí/PI — IBGE 2201919",
    "220192": "Bonfim do Piauí/PI — IBGE 2201929",
    "220194": "Boqueirão do Piauí/PI — IBGE 2201945",
    "220196": "Brasileira/PI — IBGE 2201960",
    "220198": "Brejo do Piauí/PI — IBGE 2201988",
    "220200": "Buriti dos Lopes/PI — IBGE 2202000",
    "220202": "Buriti dos Montes/PI — IBGE 2202026",
    "220205": "Cabeceiras do Piauí/PI — IBGE 2202059",
    "220207": "Cajazeiras do Piauí/PI — IBGE 2202075",
    "220208": "Cajueiro da Praia/PI — IBGE 2202083",
    "220209": "Caldeirão Grande do Piauí/PI — IBGE 2202091",
    "220210": "Campinas do Piauí/PI — IBGE 2202109",
    "220211": "Campo Alegre do Fidalgo/PI — IBGE 2202117",
    "220213": "Campo Grande do Piauí/PI — IBGE 2202133",
    "220217": "Campo Largo do Piauí/PI — IBGE 2202174",
    "220220": "Campo Maior/PI — IBGE 2202208",
    "220225": "Canavieira/PI — IBGE 2202251",
    "220230": "Canto do Buriti/PI — IBGE 2202307",
    "220240": "Capitão de Campos/PI — IBGE 2202406",
    "220245": "Capitão Gervásio Oliveira/PI — IBGE 2202455",
    "220250": "Caracol/PI — IBGE 2202505",
    "220253": "Caraúbas do Piauí/PI — IBGE 2202539",
    "220255": "Caridade do Piauí/PI — IBGE 2202554",
    "220260": "Castelo do Piauí/PI — IBGE 2202604",
    "220265": "Caxingó/PI — IBGE 2202653",
    "220270": "Cocal/PI — IBGE 2202703",
    "220271": "Cocal de Telha/PI — IBGE 2202711",
    "220272": "Cocal dos Alves/PI — IBGE 2202729",
    "220273": "Coivaras/PI — IBGE 2202737",
    "220275": "Colônia do Gurguéia/PI — IBGE 2202752",
    "220277": "Colônia do Piauí/PI — IBGE 2202778",
    "220280": "Conceição do Canindé/PI — IBGE 2202802",
    "220285": "Coronel José Dias/PI — IBGE 2202851",
    "220290": "Corrente/PI — IBGE 2202901",
    "220300": "Cristalândia do Piauí/PI — IBGE 2203008",
    "220310": "Cristino Castro/PI — IBGE 2203107",
    "220320": "Curimatá/PI — IBGE 2203206",
    "220323": "Currais/PI — IBGE 2203230",
    "220325": "Curralinhos/PI — IBGE 2203255",
    "220327": "Curral Novo do Piauí/PI — IBGE 2203271",
    "220330": "Demerval Lobão/PI — IBGE 2203305",
    "220335": "Dirceu Arcoverde/PI — IBGE 2203354",
    "220340": "Dom Expedito Lopes/PI — IBGE 2203404",
    "220342": "Domingos Mourão/PI — IBGE 2203420",
    "220345": "Dom Inocêncio/PI — IBGE 2203453",
    "220350": "Elesbão Veloso/PI — IBGE 2203503",
    "220360": "Eliseu Martins/PI — IBGE 2203602",
    "220370": "Esperantina/PI — IBGE 2203701",
    "220375": "Fartura do Piauí/PI — IBGE 2203750",
    "220380": "Flores do Piauí/PI — IBGE 2203800",
    "220385": "Floresta do Piauí/PI — IBGE 2203859",
    "220390": "Floriano/PI — IBGE 2203909",
    "220400": "Francinópolis/PI — IBGE 2204006",
    "220410": "Francisco Ayres/PI — IBGE 2204105",
    "220415": "Francisco Macedo/PI — IBGE 2204154",
    "220420": "Francisco Santos/PI — IBGE 2204204",
    "220430": "Fronteiras/PI — IBGE 2204303",
    "220435": "Geminiano/PI — IBGE 2204352",
    "220440": "Gilbués/PI — IBGE 2204402",
    "220450": "Guadalupe/PI — IBGE 2204501",
    "220455": "Guaribas/PI — IBGE 2204550",
    "220460": "Hugo Napoleão/PI — IBGE 2204600",
    "220465": "Ilha Grande/PI — IBGE 2204659",
    "220470": "Inhuma/PI — IBGE 2204709",
    "220480": "Ipiranga do Piauí/PI — IBGE 2204808",
    "220490": "Isaías Coelho/PI — IBGE 2204907",
    "220500": "Itainópolis/PI — IBGE 2205003",
    "220510": "Itaueira/PI — IBGE 2205102",
    "220515": "Jacobina do Piauí/PI — IBGE 2205151",
    "220520": "Jaicós/PI — IBGE 2205201",
    "220525": "Jardim do Mulato/PI — IBGE 2205250",
    "220527": "Jatobá do Piauí/PI — IBGE 2205276",
    "220530": "Jerumenha/PI — IBGE 2205300",
    "220535": "João Costa/PI — IBGE 2205359",
    "220540": "Joaquim Pires/PI — IBGE 2205409",
    "220545": "Joca Marques/PI — IBGE 2205458",
    "220550": "José de Freitas/PI — IBGE 2205508",
    "220551": "Juazeiro do Piauí/PI — IBGE 2205516",
    "220552": "Júlio Borges/PI — IBGE 2205524",
    "220553": "Jurema/PI — IBGE 2205532",
    "220554": "Lagoinha do Piauí/PI — IBGE 2205540",
    "220555": "Lagoa Alegre/PI — IBGE 2205557",
    "220556": "Lagoa do Barro do Piauí/PI — IBGE 2205565",
    "220557": "Lagoa de São Francisco/PI — IBGE 2205573",
    "220558": "Lagoa do Piauí/PI — IBGE 2205581",
    "220559": "Lagoa do Sítio/PI — IBGE 2205599",
    "220560": "Landri Sales/PI — IBGE 2205607",
    "220570": "Luís Correia/PI — IBGE 2205706",
    "220580": "Luzilândia/PI — IBGE 2205805",
    "220585": "Madeiro/PI — IBGE 2205854",
    "220590": "Manoel Emídio/PI — IBGE 2205904",
    "220595": "Marcolândia/PI — IBGE 2205953",
    "220600": "Marcos Parente/PI — IBGE 2206001",
    "220605": "Massapê do Piauí/PI — IBGE 2206050",
    "220610": "Matias Olímpio/PI — IBGE 2206100",
    "220620": "Miguel Alves/PI — IBGE 2206209",
    "220630": "Miguel Leão/PI — IBGE 2206308",
    "220635": "Milton Brandão/PI — IBGE 2206357",
    "220640": "Monsenhor Gil/PI — IBGE 2206407",
    "220650": "Monsenhor Hipólito/PI — IBGE 2206506",
    "220660": "Monte Alegre do Piauí/PI — IBGE 2206605",
    "220665": "Morro Cabeça no Tempo/PI — IBGE 2206654",
    "220667": "Morro do Chapéu do Piauí/PI — IBGE 2206670",
    "220669": "Murici dos Portelas/PI — IBGE 2206696",
    "220670": "Nazaré do Piauí/PI — IBGE 2206704",
    "220672": "Nazária/PI — IBGE 2206720",
    "220675": "Nossa Senhora de Nazaré/PI — IBGE 2206753",
    "220680": "Nossa Senhora dos Remédios/PI — IBGE 2206803",
    "220690": "Novo Oriente do Piauí/PI — IBGE 2206902",
    "220695": "Novo Santo Antônio/PI — IBGE 2206951",
    "220700": "Oeiras/PI — IBGE 2207009",
    "220710": "Olho D'Água do Piauí/PI — IBGE 2207108",
    "220720": "Padre Marcos/PI — IBGE 2207207",
    "220730": "Paes Landim/PI — IBGE 2207306",
    "220735": "Pajeú do Piauí/PI — IBGE 2207355",
    "220740": "Palmeira do Piauí/PI — IBGE 2207405",
    "220750": "Palmeirais/PI — IBGE 2207504",
    "220755": "Paquetá/PI — IBGE 2207553",
    "220760": "Parnaguá/PI — IBGE 2207603",
    "220770": "Parnaíba/PI — IBGE 2207702",
    "220775": "Passagem Franca do Piauí/PI — IBGE 2207751",
    "220777": "Patos do Piauí/PI — IBGE 2207777",
    "220779": "Pau D'Arco do Piauí/PI — IBGE 2207793",
    "220780": "Paulistana/PI — IBGE 2207801",
    "220785": "Pavussu/PI — IBGE 2207850",
    "220790": "Pedro II/PI — IBGE 2207900",
    "220793": "Pedro Laurentino/PI — IBGE 2207934",
    "220795": "Nova Santa Rita/PI — IBGE 2207959",
    "220800": "Picos/PI — IBGE 2208007",
    "220810": "Pimenteiras/PI — IBGE 2208106",
    "220820": "Pio IX/PI — IBGE 2208205",
    "220830": "Piracuruca/PI — IBGE 2208304",
    "220840": "Piripiri/PI — IBGE 2208403",
    "220850": "Porto/PI — IBGE 2208502",
    "220855": "Porto Alegre do Piauí/PI — IBGE 2208551",
    "220860": "Prata do Piauí/PI — IBGE 2208601",
    "220865": "Queimada Nova/PI — IBGE 2208650",
    "220870": "Redenção do Gurguéia/PI — IBGE 2208700",
    "220880": "Regeneração/PI — IBGE 2208809",
    "220885": "Riacho Frio/PI — IBGE 2208858",
    "220887": "Ribeira do Piauí/PI — IBGE 2208874",
    "220890": "Ribeiro Gonçalves/PI — IBGE 2208908",
    "220900": "Rio Grande do Piauí/PI — IBGE 2209005",
    "220910": "Santa Cruz do Piauí/PI — IBGE 2209104",
    "220915": "Santa Cruz dos Milagres/PI — IBGE 2209153",
    "220920": "Santa Filomena/PI — IBGE 2209203",
    "220930": "Santa Luz/PI — IBGE 2209302",
    "220935": "Santana do Piauí/PI — IBGE 2209351",
    "220937": "Santa Rosa do Piauí/PI — IBGE 2209377",
    "220940": "Santo Antônio de Lisboa/PI — IBGE 2209401",
    "220945": "Santo Antônio dos Milagres/PI — IBGE 2209450",
    "220950": "Santo Inácio do Piauí/PI — IBGE 2209500",
    "220955": "São Braz do Piauí/PI — IBGE 2209559",
    "220960": "São Félix do Piauí/PI — IBGE 2209609",
    "220965": "São Francisco de Assis do Piauí/PI — IBGE 2209658",
    "220970": "São Francisco do Piauí/PI — IBGE 2209708",
    "220975": "São Gonçalo do Gurguéia/PI — IBGE 2209757",
    "220980": "São Gonçalo do Piauí/PI — IBGE 2209807",
    "220985": "São João da Canabrava/PI — IBGE 2209856",
    "220987": "São João da Fronteira/PI — IBGE 2209872",
    "220990": "São João da Serra/PI — IBGE 2209906",
    "220995": "São João da Varjota/PI — IBGE 2209955",
    "220997": "São João do Arraial/PI — IBGE 2209971",
    "221000": "São João do Piauí/PI — IBGE 2210003",
    "221005": "São José do Divino/PI — IBGE 2210052",
    "221010": "São José do Peixe/PI — IBGE 2210102",
    "221020": "São José do Piauí/PI — IBGE 2210201",
    "221030": "São Julião/PI — IBGE 2210300",
    "221035": "São Lourenço do Piauí/PI — IBGE 2210359",
    "221037": "São Luis do Piauí/PI — IBGE 2210375",
    "221038": "São Miguel da Baixa Grande/PI — IBGE 2210383",
    "221039": "São Miguel do Fidalgo/PI — IBGE 2210391",
    "221040": "São Miguel do Tapuio/PI — IBGE 2210409",
    "221050": "São Pedro do Piauí/PI — IBGE 2210508",
    "221060": "São Raimundo Nonato/PI — IBGE 2210607",
    "221062": "Sebastião Barros/PI — IBGE 2210623",
    "221063": "Sebastião Leal/PI — IBGE 2210631",
    "221065": "Sigefredo Pacheco/PI — IBGE 2210656",
    "221070": "Simões/PI — IBGE 2210706",
    "221080": "Simplício Mendes/PI — IBGE 2210805",
    "221090": "Socorro do Piauí/PI — IBGE 2210904",
    "221093": "Sussuapara/PI — IBGE 2210938",
    "221095": "Tamboril do Piauí/PI — IBGE 2210953",
    "221097": "Tanque do Piauí/PI — IBGE 2210979",
    "221100": "Teresina/PI — IBGE 2211001",
    "221110": "União/PI — IBGE 2211100",
    "221120": "Uruçuí/PI — IBGE 2211209",
    "221130": "Valença do Piauí/PI — IBGE 2211308",
    "221135": "Várzea Branca/PI — IBGE 2211357",
    "221140": "Várzea Grande/PI — IBGE 2211407",
    "221150": "Vera Mendes/PI — IBGE 2211506",
    "221160": "Vila Nova do Piauí/PI — IBGE 2211605",
    "221170": "Wall Ferraz/PI — IBGE 2211704",
    "230010": "Abaiara/CE — IBGE 2300101",
    "230015": "Acarape/CE — IBGE 2300150",
    "230020": "Acaraú/CE — IBGE 2300200",
    "230030": "Acopiara/CE — IBGE 2300309",
    "230040": "Aiuaba/CE — IBGE 2300408",
    "230050": "Alcântaras/CE — IBGE 2300507",
    "230060": "Altaneira/CE — IBGE 2300606",
    "230070": "Alto Santo/CE — IBGE 2300705",
    "230075": "Amontada/CE — IBGE 2300754",
    "230080": "Antonina do Norte/CE — IBGE 2300804",
    "230090": "Apuiarés/CE — IBGE 2300903",
    "230100": "Aquiraz/CE — IBGE 2301000",
    "230110": "Aracati/CE — IBGE 2301109",
    "230120": "Aracoiaba/CE — IBGE 2301208",
    "230125": "Ararendá/CE — IBGE 2301257",
    "230130": "Araripe/CE — IBGE 2301307",
    "230140": "Aratuba/CE — IBGE 2301406",
    "230150": "Arneiroz/CE — IBGE 2301505",
    "230160": "Assaré/CE — IBGE 2301604",
    "230170": "Aurora/CE — IBGE 2301703",
    "230180": "Baixio/CE — IBGE 2301802",
    "230185": "Banabuiú/CE — IBGE 2301851",
    "230190": "Barbalha/CE — IBGE 2301901",
    "230195": "Barreira/CE — IBGE 2301950",
    "230200": "Barro/CE — IBGE 2302008",
    "230205": "Barroquinha/CE — IBGE 2302057",
    "230210": "Baturité/CE — IBGE 2302107",
    "230220": "Beberibe/CE — IBGE 2302206",
    "230230": "Bela Cruz/CE — IBGE 2302305",
    "230240": "Boa Viagem/CE — IBGE 2302404",
    "230250": "Brejo Santo/CE — IBGE 2302503",
    "230260": "Camocim/CE — IBGE 2302602",
    "230270": "Campos Sales/CE — IBGE 2302701",
    "230280": "Canindé/CE — IBGE 2302800",
    "230290": "Capistrano/CE — IBGE 2302909",
    "230300": "Caridade/CE — IBGE 2303006",
    "230310": "Cariré/CE — IBGE 2303105",
    "230320": "Caririaçu/CE — IBGE 2303204",
    "230330": "Cariús/CE — IBGE 2303303",
    "230340": "Carnaubal/CE — IBGE 2303402",
    "230350": "Cascavel/CE — IBGE 2303501",
    "230360": "Catarina/CE — IBGE 2303600",
    "230365": "Catunda/CE — IBGE 2303659",
    "230370": "Caucaia/CE — IBGE 2303709",
    "230380": "Cedro/CE — IBGE 2303808",
    "230390": "Chaval/CE — IBGE 2303907",
    "230393": "Choró/CE — IBGE 2303931",
    "230395": "Chorozinho/CE — IBGE 2303956",
    "230400": "Coreaú/CE — IBGE 2304004",
    "230410": "Crateús/CE — IBGE 2304103",
    "230420": "Crato/CE — IBGE 2304202",
    "230423": "Croatá/CE — IBGE 2304236",
    "230425": "Cruz/CE — IBGE 2304251",
    "230426": "Deputado Irapuan Pinheiro/CE — IBGE 2304269",
    "230427": "Ereré/CE — IBGE 2304277",
    "230428": "Eusébio/CE — IBGE 2304285",
    "230430": "Farias Brito/CE — IBGE 2304301",
    "230435": "Forquilha/CE — IBGE 2304350",
    "230440": "Fortaleza/CE — IBGE 2304400",
    "230445": "Fortim/CE — IBGE 2304459",
    "230450": "Frecheirinha/CE — IBGE 2304509",
    "230460": "General Sampaio/CE — IBGE 2304608",
    "230465": "Graça/CE — IBGE 2304657",
    "230470": "Granja/CE — IBGE 2304707",
    "230480": "Granjeiro/CE — IBGE 2304806",
    "230490": "Groaíras/CE — IBGE 2304905",
    "230495": "Guaiúba/CE — IBGE 2304954",
    "230500": "Guaraciaba do Norte/CE — IBGE 2305001",
    "230510": "Guaramiranga/CE — IBGE 2305100",
    "230520": "Hidrolândia/CE — IBGE 2305209",
    "230523": "Horizonte/CE — IBGE 2305233",
    "230526": "Ibaretama/CE — IBGE 2305266",
    "230530": "Ibiapina/CE — IBGE 2305308",
    "230533": "Ibicuitinga/CE — IBGE 2305332",
    "230535": "Icapuí/CE — IBGE 2305357",
    "230540": "Icó/CE — IBGE 2305407",
    "230550": "Iguatu/CE — IBGE 2305506",
    "230560": "Independência/CE — IBGE 2305605",
    "230565": "Ipaporanga/CE — IBGE 2305654",
    "230570": "Ipaumirim/CE — IBGE 2305704",
    "230580": "Ipu/CE — IBGE 2305803",
    "230590": "Ipueiras/CE — IBGE 2305902",
    "230600": "Iracema/CE — IBGE 2306009",
    "230610": "Irauçuba/CE — IBGE 2306108",
    "230620": "Itaiçaba/CE — IBGE 2306207",
    "230625": "Itaitinga/CE — IBGE 2306256",
    "230630": "Itapajé/CE — IBGE 2306306",
    "230640": "Itapipoca/CE — IBGE 2306405",
    "230650": "Itapiúna/CE — IBGE 2306504",
    "230655": "Itarema/CE — IBGE 2306553",
    "230660": "Itatira/CE — IBGE 2306603",
    "230670": "Jaguaretama/CE — IBGE 2306702",
    "230680": "Jaguaribara/CE — IBGE 2306801",
    "230690": "Jaguaribe/CE — IBGE 2306900",
    "230700": "Jaguaruana/CE — IBGE 2307007",
    "230710": "Jardim/CE — IBGE 2307106",
    "230720": "Jati/CE — IBGE 2307205",
    "230725": "Jijoca de Jericoacoara/CE — IBGE 2307254",
    "230730": "Juazeiro do Norte/CE — IBGE 2307304",
    "230740": "Jucás/CE — IBGE 2307403",
    "230750": "Lavras da Mangabeira/CE — IBGE 2307502",
    "230760": "Limoeiro do Norte/CE — IBGE 2307601",
    "230763": "Madalena/CE — IBGE 2307635",
    "230765": "Maracanaú/CE — IBGE 2307650",
    "230770": "Maranguape/CE — IBGE 2307700",
    "230780": "Marco/CE — IBGE 2307809",
    "230790": "Martinópole/CE — IBGE 2307908",
    "230800": "Massapê/CE — IBGE 2308005",
    "230810": "Mauriti/CE — IBGE 2308104",
    "230820": "Meruoca/CE — IBGE 2308203",
    "230830": "Milagres/CE — IBGE 2308302",
    "230835": "Milhã/CE — IBGE 2308351",
    "230837": "Miraíma/CE — IBGE 2308377",
    "230840": "Missão Velha/CE — IBGE 2308401",
    "230850": "Mombaça/CE — IBGE 2308500",
    "230860": "Monsenhor Tabosa/CE — IBGE 2308609",
    "230870": "Morada Nova/CE — IBGE 2308708",
    "230880": "Moraújo/CE — IBGE 2308807",
    "230890": "Morrinhos/CE — IBGE 2308906",
    "230900": "Mucambo/CE — IBGE 2309003",
    "230910": "Mulungu/CE — IBGE 2309102",
    "230920": "Nova Olinda/CE — IBGE 2309201",
    "230930": "Nova Russas/CE — IBGE 2309300",
    "230940": "Novo Oriente/CE — IBGE 2309409",
    "230945": "Ocara/CE — IBGE 2309458",
    "230950": "Orós/CE — IBGE 2309508",
    "230960": "Pacajus/CE — IBGE 2309607",
    "230970": "Pacatuba/CE — IBGE 2309706",
    "230980": "Pacoti/CE — IBGE 2309805",
    "230990": "Pacujá/CE — IBGE 2309904",
    "231000": "Palhano/CE — IBGE 2310001",
    "231010": "Palmácia/CE — IBGE 2310100",
    "231020": "Paracuru/CE — IBGE 2310209",
    "231025": "Paraipaba/CE — IBGE 2310258",
    "231030": "Parambu/CE — IBGE 2310308",
    "231040": "Paramoti/CE — IBGE 2310407",
    "231050": "Pedra Branca/CE — IBGE 2310506",
    "231060": "Penaforte/CE — IBGE 2310605",
    "231070": "Pentecoste/CE — IBGE 2310704",
    "231080": "Pereiro/CE — IBGE 2310803",
    "231085": "Pindoretama/CE — IBGE 2310852",
    "231090": "Piquet Carneiro/CE — IBGE 2310902",
    "231095": "Pires Ferreira/CE — IBGE 2310951",
    "231100": "Poranga/CE — IBGE 2311009",
    "231110": "Porteiras/CE — IBGE 2311108",
    "231120": "Potengi/CE — IBGE 2311207",
    "231123": "Potiretama/CE — IBGE 2311231",
    "231126": "Quiterianópolis/CE — IBGE 2311264",
    "231130": "Quixadá/CE — IBGE 2311306",
    "231135": "Quixelô/CE — IBGE 2311355",
    "231140": "Quixeramobim/CE — IBGE 2311405",
    "231150": "Quixeré/CE — IBGE 2311504",
    "231160": "Redenção/CE — IBGE 2311603",
    "231170": "Reriutaba/CE — IBGE 2311702",
    "231180": "Russas/CE — IBGE 2311801",
    "231190": "Saboeiro/CE — IBGE 2311900",
    "231195": "Salitre/CE — IBGE 2311959",
    "231200": "Santana do Acaraú/CE — IBGE 2312007",
    "231210": "Santana do Cariri/CE — IBGE 2312106",
    "231220": "Santa Quitéria/CE — IBGE 2312205",
    "231230": "São Benedito/CE — IBGE 2312304",
    "231240": "São Gonçalo do Amarante/CE — IBGE 2312403",
    "231250": "São João do Jaguaribe/CE — IBGE 2312502",
    "231260": "São Luís do Curu/CE — IBGE 2312601",
    "231270": "Senador Pompeu/CE — IBGE 2312700",
    "231280": "Senador Sá/CE — IBGE 2312809",
    "231290": "Sobral/CE — IBGE 2312908",
    "231300": "Solonópole/CE — IBGE 2313005",
    "231310": "Tabuleiro do Norte/CE — IBGE 2313104",
    "231320": "Tamboril/CE — IBGE 2313203",
    "231325": "Tarrafas/CE — IBGE 2313252",
    "231330": "Tauá/CE — IBGE 2313302",
    "231335": "Tejuçuoca/CE — IBGE 2313351",
    "231340": "Tianguá/CE — IBGE 2313401",
    "231350": "Trairi/CE — IBGE 2313500",
    "231355": "Tururu/CE — IBGE 2313559",
    "231360": "Ubajara/CE — IBGE 2313609",
    "231370": "Umari/CE — IBGE 2313708",
    "231375": "Umirim/CE — IBGE 2313757",
    "231380": "Uruburetama/CE — IBGE 2313807",
    "231390": "Uruoca/CE — IBGE 2313906",
    "231395": "Varjota/CE — IBGE 2313955",
    "231400": "Várzea Alegre/CE — IBGE 2314003",
    "231410": "Viçosa do Ceará/CE — IBGE 2314102",
    "240010": "Acari/RN — IBGE 2400109",
    "240020": "Açu/RN — IBGE 2400208",
    "240030": "Afonso Bezerra/RN — IBGE 2400307",
    "240040": "Água Nova/RN — IBGE 2400406",
    "240050": "Alexandria/RN — IBGE 2400505",
    "240060": "Almino Afonso/RN — IBGE 2400604",
    "240070": "Alto do Rodrigues/RN — IBGE 2400703",
    "240080": "Angicos/RN — IBGE 2400802",
    "240090": "Antônio Martins/RN — IBGE 2400901",
    "240100": "Apodi/RN — IBGE 2401008",
    "240110": "Areia Branca/RN — IBGE 2401107",
    "240120": "Arês/RN — IBGE 2401206",
    "240130": "Campo Grande/RN — IBGE 2401305",
    "240140": "Baía Formosa/RN — IBGE 2401404",
    "240145": "Baraúna/RN — IBGE 2401453",
    "240150": "Barcelona/RN — IBGE 2401503",
    "240160": "Bento Fernandes/RN — IBGE 2401602",
    "240165": "Bodó/RN — IBGE 2401651",
    "240170": "Bom Jesus/RN — IBGE 2401701",
    "240180": "Brejinho/RN — IBGE 2401800",
    "240185": "Caiçara do Norte/RN — IBGE 2401859",
    "240190": "Caiçara do Rio do Vento/RN — IBGE 2401909",
    "240200": "Caicó/RN — IBGE 2402006",
    "240210": "Campo Redondo/RN — IBGE 2402105",
    "240220": "Canguaretama/RN — IBGE 2402204",
    "240230": "Caraúbas/RN — IBGE 2402303",
    "240240": "Carnaúba dos Dantas/RN — IBGE 2402402",
    "240250": "Carnaubais/RN — IBGE 2402501",
    "240260": "Ceará-Mirim/RN — IBGE 2402600",
    "240270": "Cerro Corá/RN — IBGE 2402709",
    "240280": "Coronel Ezequiel/RN — IBGE 2402808",
    "240290": "Coronel João Pessoa/RN — IBGE 2402907",
    "240300": "Cruzeta/RN — IBGE 2403004",
    "240310": "Currais Novos/RN — IBGE 2403103",
    "240320": "Doutor Severiano/RN — IBGE 2403202",
    "240325": "Parnamirim/RN — IBGE 2403251",
    "240330": "Encanto/RN — IBGE 2403301",
    "240340": "Equador/RN — IBGE 2403400",
    "240350": "Espírito Santo/RN — IBGE 2403509",
    "240360": "Extremoz/RN — IBGE 2403608",
    "240370": "Felipe Guerra/RN — IBGE 2403707",
    "240375": "Fernando Pedroza/RN — IBGE 2403756",
    "240380": "Florânia/RN — IBGE 2403806",
    "240390": "Francisco Dantas/RN — IBGE 2403905",
    "240400": "Frutuoso Gomes/RN — IBGE 2404002",
    "240410": "Galinhos/RN — IBGE 2404101",
    "240420": "Goianinha/RN — IBGE 2404200",
    "240430": "Governador Dix-Sept Rosado/RN — IBGE 2404309",
    "240440": "Grossos/RN — IBGE 2404408",
    "240450": "Guamaré/RN — IBGE 2404507",
    "240460": "Ielmo Marinho/RN — IBGE 2404606",
    "240470": "Ipanguaçu/RN — IBGE 2404705",
    "240480": "Ipueira/RN — IBGE 2404804",
    "240485": "Itajá/RN — IBGE 2404853",
    "240490": "Itaú/RN — IBGE 2404903",
    "240500": "Jaçanã/RN — IBGE 2405009",
    "240510": "Jandaíra/RN — IBGE 2405108",
    "240520": "Janduís/RN — IBGE 2405207",
    "240530": "Januário Cicco/RN — IBGE 2405306",
    "240540": "Japi/RN — IBGE 2405405",
    "240550": "Jardim de Angicos/RN — IBGE 2405504",
    "240560": "Jardim de Piranhas/RN — IBGE 2405603",
    "240570": "Jardim do Seridó/RN — IBGE 2405702",
    "240580": "João Câmara/RN — IBGE 2405801",
    "240590": "João Dias/RN — IBGE 2405900",
    "240600": "José da Penha/RN — IBGE 2406007",
    "240610": "Jucurutu/RN — IBGE 2406106",
    "240615": "Jundiá/RN — IBGE 2406155",
    "240620": "Lagoa d'Anta/RN — IBGE 2406205",
    "240630": "Lagoa de Pedras/RN — IBGE 2406304",
    "240640": "Lagoa de Velhos/RN — IBGE 2406403",
    "240650": "Lagoa Nova/RN — IBGE 2406502",
    "240660": "Lagoa Salgada/RN — IBGE 2406601",
    "240670": "Lajes/RN — IBGE 2406700",
    "240680": "Lajes Pintadas/RN — IBGE 2406809",
    "240690": "Lucrécia/RN — IBGE 2406908",
    "240700": "Luís Gomes/RN — IBGE 2407005",
    "240710": "Macaíba/RN — IBGE 2407104",
    "240720": "Macau/RN — IBGE 2407203",
    "240725": "Major Sales/RN — IBGE 2407252",
    "240730": "Marcelino Vieira/RN — IBGE 2407302",
    "240740": "Martins/RN — IBGE 2407401",
    "240750": "Maxaranguape/RN — IBGE 2407500",
    "240760": "Messias Targino/RN — IBGE 2407609",
    "240770": "Montanhas/RN — IBGE 2407708",
    "240780": "Monte Alegre/RN — IBGE 2407807",
    "240790": "Monte das Gameleiras/RN — IBGE 2407906",
    "240800": "Mossoró/RN — IBGE 2408003",
    "240810": "Natal/RN — IBGE 2408102",
    "240820": "Nísia Floresta/RN — IBGE 2408201",
    "240830": "Nova Cruz/RN — IBGE 2408300",
    "240840": "Olho d'Água do Borges/RN — IBGE 2408409",
    "240850": "Ouro Branco/RN — IBGE 2408508",
    "240860": "Paraná/RN — IBGE 2408607",
    "240870": "Paraú/RN — IBGE 2408706",
    "240880": "Parazinho/RN — IBGE 2408805",
    "240890": "Parelhas/RN — IBGE 2408904",
    "240895": "Rio do Fogo/RN — IBGE 2408953",
    "240910": "Passa e Fica/RN — IBGE 2409100",
    "240920": "Passagem/RN — IBGE 2409209",
    "240930": "Patu/RN — IBGE 2409308",
    "240933": "Santa Maria/RN — IBGE 2409332",
    "240940": "Pau dos Ferros/RN — IBGE 2409407",
    "240950": "Pedra Grande/RN — IBGE 2409506",
    "240960": "Pedra Preta/RN — IBGE 2409605",
    "240970": "Pedro Avelino/RN — IBGE 2409704",
    "240980": "Pedro Velho/RN — IBGE 2409803",
    "240990": "Pendências/RN — IBGE 2409902",
    "241000": "Pilões/RN — IBGE 2410009",
    "241010": "Poço Branco/RN — IBGE 2410108",
    "241020": "Portalegre/RN — IBGE 2410207",
    "241025": "Porto do Mangue/RN — IBGE 2410256",
    "241030": "Serra Caiada/RN — IBGE 2410306",
    "241040": "Pureza/RN — IBGE 2410405",
    "241050": "Rafael Fernandes/RN — IBGE 2410504",
    "241060": "Rafael Godeiro/RN — IBGE 2410603",
    "241070": "Riacho da Cruz/RN — IBGE 2410702",
    "241080": "Riacho de Santana/RN — IBGE 2410801",
    "241090": "Riachuelo/RN — IBGE 2410900",
    "241100": "Rodolfo Fernandes/RN — IBGE 2411007",
    "241105": "Tibau/RN — IBGE 2411056",
    "241110": "Ruy Barbosa/RN — IBGE 2411106",
    "241120": "Santa Cruz/RN — IBGE 2411205",
    "241140": "Santana do Matos/RN — IBGE 2411403",
    "241142": "Santana do Seridó/RN — IBGE 2411429",
    "241150": "Santo Antônio/RN — IBGE 2411502",
    "241160": "São Bento do Norte/RN — IBGE 2411601",
    "241170": "São Bento do Trairí/RN — IBGE 2411700",
    "241180": "São Fernando/RN — IBGE 2411809",
    "241190": "São Francisco do Oeste/RN — IBGE 2411908",
    "241200": "São Gonçalo do Amarante/RN — IBGE 2412005",
    "241210": "São João do Sabugi/RN — IBGE 2412104",
    "241220": "São José de Mipibu/RN — IBGE 2412203",
    "241230": "São José do Campestre/RN — IBGE 2412302",
    "241240": "São José do Seridó/RN — IBGE 2412401",
    "241250": "São Miguel/RN — IBGE 2412500",
    "241255": "São Miguel do Gostoso/RN — IBGE 2412559",
    "241260": "São Paulo do Potengi/RN — IBGE 2412609",
    "241270": "São Pedro/RN — IBGE 2412708",
    "241280": "São Rafael/RN — IBGE 2412807",
    "241290": "São Tomé/RN — IBGE 2412906",
    "241300": "São Vicente/RN — IBGE 2413003",
    "241310": "Senador Elói de Souza/RN — IBGE 2413102",
    "241320": "Senador Georgino Avelino/RN — IBGE 2413201",
    "241330": "Serra de São Bento/RN — IBGE 2413300",
    "241335": "Serra do Mel/RN — IBGE 2413359",
    "241340": "Serra Negra do Norte/RN — IBGE 2413409",
    "241350": "Serrinha/RN — IBGE 2413508",
    "241355": "Serrinha dos Pintos/RN — IBGE 2413557",
    "241360": "Severiano Melo/RN — IBGE 2413607",
    "241370": "Sítio Novo/RN — IBGE 2413706",
    "241380": "Taboleiro Grande/RN — IBGE 2413805",
    "241390": "Taipu/RN — IBGE 2413904",
    "241400": "Tangará/RN — IBGE 2414001",
    "241410": "Tenente Ananias/RN — IBGE 2414100",
    "241415": "Tenente Laurentino Cruz/RN — IBGE 2414159",
    "241420": "Tibau do Sul/RN — IBGE 2414209",
    "241430": "Timbaúba dos Batistas/RN — IBGE 2414308",
    "241440": "Touros/RN — IBGE 2414407",
    "241445": "Triunfo Potiguar/RN — IBGE 2414456",
    "241450": "Umarizal/RN — IBGE 2414506",
    "241460": "Upanema/RN — IBGE 2414605",
    "241470": "Várzea/RN — IBGE 2414704",
    "241475": "Venha-Ver/RN — IBGE 2414753",
    "241480": "Vera Cruz/RN — IBGE 2414803",
    "241490": "Viçosa/RN — IBGE 2414902",
    "241500": "Vila Flor/RN — IBGE 2415008",
    "250010": "Água Branca/PB — IBGE 2500106",
    "250020": "Aguiar/PB — IBGE 2500205",
    "250030": "Alagoa Grande/PB — IBGE 2500304",
    "250040": "Alagoa Nova/PB — IBGE 2500403",
    "250050": "Alagoinha/PB — IBGE 2500502",
    "250053": "Alcantil/PB — IBGE 2500536",
    "250057": "Algodão de Jandaíra/PB — IBGE 2500577",
    "250060": "Alhandra/PB — IBGE 2500601",
    "250070": "São João do Rio do Peixe/PB — IBGE 2500700",
    "250073": "Amparo/PB — IBGE 2500734",
    "250077": "Aparecida/PB — IBGE 2500775",
    "250080": "Araçagi/PB — IBGE 2500809",
    "250090": "Arara/PB — IBGE 2500908",
    "250100": "Araruna/PB — IBGE 2501005",
    "250110": "Areia/PB — IBGE 2501104",
    "250115": "Areia de Baraúnas/PB — IBGE 2501153",
    "250120": "Areial/PB — IBGE 2501203",
    "250130": "Aroeiras/PB — IBGE 2501302",
    "250135": "Assunção/PB — IBGE 2501351",
    "250140": "Baía da Traição/PB — IBGE 2501401",
    "250150": "Bananeiras/PB — IBGE 2501500",
    "250153": "Baraúna/PB — IBGE 2501534",
    "250157": "Barra de Santana/PB — IBGE 2501575",
    "250160": "Barra de Santa Rosa/PB — IBGE 2501609",
    "250170": "Barra de São Miguel/PB — IBGE 2501708",
    "250180": "Bayeux/PB — IBGE 2501807",
    "250190": "Belém/PB — IBGE 2501906",
    "250200": "Belém do Brejo do Cruz/PB — IBGE 2502003",
    "250205": "Bernardino Batista/PB — IBGE 2502052",
    "250210": "Boa Ventura/PB — IBGE 2502102",
    "250215": "Boa Vista/PB — IBGE 2502151",
    "250220": "Bom Jesus/PB — IBGE 2502201",
    "250230": "Bom Sucesso/PB — IBGE 2502300",
    "250240": "Bonito de Santa Fé/PB — IBGE 2502409",
    "250250": "Boqueirão/PB — IBGE 2502508",
    "250260": "Igaracy/PB — IBGE 2502607",
    "250270": "Borborema/PB — IBGE 2502706",
    "250280": "Brejo do Cruz/PB — IBGE 2502805",
    "250290": "Brejo dos Santos/PB — IBGE 2502904",
    "250300": "Caaporã/PB — IBGE 2503001",
    "250310": "Cabaceiras/PB — IBGE 2503100",
    "250320": "Cabedelo/PB — IBGE 2503209",
    "250330": "Cachoeira dos Índios/PB — IBGE 2503308",
    "250340": "Cacimba de Areia/PB — IBGE 2503407",
    "250350": "Cacimba de Dentro/PB — IBGE 2503506",
    "250355": "Cacimbas/PB — IBGE 2503555",
    "250360": "Caiçara/PB — IBGE 2503605",
    "250370": "Cajazeiras/PB — IBGE 2503704",
    "250375": "Cajazeirinhas/PB — IBGE 2503753",
    "250380": "Caldas Brandão/PB — IBGE 2503803",
    "250390": "Camalaú/PB — IBGE 2503902",
    "250400": "Campina Grande/PB — IBGE 2504009",
    "250403": "Capim/PB — IBGE 2504033",
    "250407": "Caraúbas/PB — IBGE 2504074",
    "250410": "Carrapateira/PB — IBGE 2504108",
    "250415": "Casserengue/PB — IBGE 2504157",
    "250420": "Catingueira/PB — IBGE 2504207",
    "250430": "Catolé do Rocha/PB — IBGE 2504306",
    "250435": "Caturité/PB — IBGE 2504355",
    "250440": "Conceição/PB — IBGE 2504405",
    "250450": "Condado/PB — IBGE 2504504",
    "250460": "Conde/PB — IBGE 2504603",
    "250470": "Congo/PB — IBGE 2504702",
    "250480": "Coremas/PB — IBGE 2504801",
    "250485": "Coxixola/PB — IBGE 2504850",
    "250490": "Cruz do Espírito Santo/PB — IBGE 2504900",
    "250500": "Cubati/PB — IBGE 2505006",
    "250510": "Cuité/PB — IBGE 2505105",
    "250520": "Cuitegi/PB — IBGE 2505204",
    "250523": "Cuité de Mamanguape/PB — IBGE 2505238",
    "250527": "Curral de Cima/PB — IBGE 2505279",
    "250530": "Curral Velho/PB — IBGE 2505303",
    "250535": "Damião/PB — IBGE 2505352",
    "250540": "Desterro/PB — IBGE 2505402",
    "250550": "Vista Serrana/PB — IBGE 2505501",
    "250560": "Diamante/PB — IBGE 2505600",
    "250570": "Dona Inês/PB — IBGE 2505709",
    "250580": "Duas Estradas/PB — IBGE 2505808",
    "250590": "Emas/PB — IBGE 2505907",
    "250600": "Esperança/PB — IBGE 2506004",
    "250610": "Fagundes/PB — IBGE 2506103",
    "250620": "Frei Martinho/PB — IBGE 2506202",
    "250625": "Gado Bravo/PB — IBGE 2506251",
    "250630": "Guarabira/PB — IBGE 2506301",
    "250640": "Gurinhém/PB — IBGE 2506400",
    "250650": "Gurjão/PB — IBGE 2506509",
    "250660": "Ibiara/PB — IBGE 2506608",
    "250670": "Imaculada/PB — IBGE 2506707",
    "250680": "Ingá/PB — IBGE 2506806",
    "250690": "Itabaiana/PB — IBGE 2506905",
    "250700": "Itaporanga/PB — IBGE 2507002",
    "250710": "Itapororoca/PB — IBGE 2507101",
    "250720": "Itatuba/PB — IBGE 2507200",
    "250730": "Jacaraú/PB — IBGE 2507309",
    "250740": "Jericó/PB — IBGE 2507408",
    "250750": "João Pessoa/PB — IBGE 2507507",
    "250760": "Juarez Távora/PB — IBGE 2507606",
    "250770": "Juazeirinho/PB — IBGE 2507705",
    "250780": "Junco do Seridó/PB — IBGE 2507804",
    "250790": "Juripiranga/PB — IBGE 2507903",
    "250800": "Juru/PB — IBGE 2508000",
    "250810": "Lagoa/PB — IBGE 2508109",
    "250820": "Lagoa de Dentro/PB — IBGE 2508208",
    "250830": "Lagoa Seca/PB — IBGE 2508307",
    "250840": "Lastro/PB — IBGE 2508406",
    "250850": "Livramento/PB — IBGE 2508505",
    "250855": "Logradouro/PB — IBGE 2508554",
    "250860": "Lucena/PB — IBGE 2508604",
    "250870": "Mãe d'Água/PB — IBGE 2508703",
    "250880": "Malta/PB — IBGE 2508802",
    "250890": "Mamanguape/PB — IBGE 2508901",
    "250900": "Manaíra/PB — IBGE 2509008",
    "250905": "Marcação/PB — IBGE 2509057",
    "250910": "Mari/PB — IBGE 2509107",
    "250915": "Marizópolis/PB — IBGE 2509156",
    "250920": "Massaranduba/PB — IBGE 2509206",
    "250930": "Mataraca/PB — IBGE 2509305",
    "250933": "Matinhas/PB — IBGE 2509339",
    "250937": "Mato Grosso/PB — IBGE 2509370",
    "250939": "Maturéia/PB — IBGE 2509396",
    "250940": "Mogeiro/PB — IBGE 2509404",
    "250950": "Montadas/PB — IBGE 2509503",
    "250960": "Monte Horebe/PB — IBGE 2509602",
    "250970": "Monteiro/PB — IBGE 2509701",
    "250980": "Mulungu/PB — IBGE 2509800",
    "250990": "Natuba/PB — IBGE 2509909",
    "251000": "Nazarezinho/PB — IBGE 2510006",
    "251010": "Nova Floresta/PB — IBGE 2510105",
    "251020": "Nova Olinda/PB — IBGE 2510204",
    "251030": "Nova Palmeira/PB — IBGE 2510303",
    "251040": "Olho d'Água/PB — IBGE 2510402",
    "251050": "Olivedos/PB — IBGE 2510501",
    "251060": "Ouro Velho/PB — IBGE 2510600",
    "251065": "Parari/PB — IBGE 2510659",
    "251070": "Passagem/PB — IBGE 2510709",
    "251080": "Patos/PB — IBGE 2510808",
    "251090": "Paulista/PB — IBGE 2510907",
    "251100": "Pedra Branca/PB — IBGE 2511004",
    "251110": "Pedra Lavrada/PB — IBGE 2511103",
    "251120": "Pedras de Fogo/PB — IBGE 2511202",
    "251130": "Piancó/PB — IBGE 2511301",
    "251140": "Picuí/PB — IBGE 2511400",
    "251150": "Pilar/PB — IBGE 2511509",
    "251160": "Pilões/PB — IBGE 2511608",
    "251170": "Pilõezinhos/PB — IBGE 2511707",
    "251180": "Pirpirituba/PB — IBGE 2511806",
    "251190": "Pitimbu/PB — IBGE 2511905",
    "251200": "Pocinhos/PB — IBGE 2512002",
    "251203": "Poço Dantas/PB — IBGE 2512036",
    "251207": "Poço de José de Moura/PB — IBGE 2512077",
    "251210": "Pombal/PB — IBGE 2512101",
    "251220": "Prata/PB — IBGE 2512200",
    "251230": "Princesa Isabel/PB — IBGE 2512309",
    "251240": "Puxinanã/PB — IBGE 2512408",
    "251250": "Queimadas/PB — IBGE 2512507",
    "251260": "Quixaba/PB — IBGE 2512606",
    "251270": "Remígio/PB — IBGE 2512705",
    "251272": "Pedro Régis/PB — IBGE 2512721",
    "251274": "Riachão/PB — IBGE 2512747",
    "251275": "Riachão do Bacamarte/PB — IBGE 2512754",
    "251276": "Riachão do Poço/PB — IBGE 2512762",
    "251278": "Riacho de Santo Antônio/PB — IBGE 2512788",
    "251280": "Riacho dos Cavalos/PB — IBGE 2512804",
    "251290": "Rio Tinto/PB — IBGE 2512903",
    "251300": "Salgadinho/PB — IBGE 2513000",
    "251310": "Salgado de São Félix/PB — IBGE 2513109",
    "251315": "Santa Cecília/PB — IBGE 2513158",
    "251320": "Santa Cruz/PB — IBGE 2513208",
    "251330": "Santa Helena/PB — IBGE 2513307",
    "251335": "Santa Inês/PB — IBGE 2513356",
    "251340": "Santa Luzia/PB — IBGE 2513406",
    "251350": "Santana de Mangueira/PB — IBGE 2513505",
    "251360": "Santana dos Garrotes/PB — IBGE 2513604",
    "251365": "Joca Claudino/PB — IBGE 2513653",
    "251370": "Santa Rita/PB — IBGE 2513703",
    "251380": "Santa Teresinha/PB — IBGE 2513802",
    "251385": "Santo André/PB — IBGE 2513851",
    "251390": "São Bento/PB — IBGE 2513901",
    "251392": "São Bentinho/PB — IBGE 2513927",
    "251394": "São Domingos do Cariri/PB — IBGE 2513943",
    "251396": "São Domingos/PB — IBGE 2513968",
    "251398": "São Francisco/PB — IBGE 2513984",
    "251400": "São João do Cariri/PB — IBGE 2514008",
    "251410": "São João do Tigre/PB — IBGE 2514107",
    "251420": "São José da Lagoa Tapada/PB — IBGE 2514206",
    "251430": "São José de Caiana/PB — IBGE 2514305",
    "251440": "São José de Espinharas/PB — IBGE 2514404",
    "251445": "São José dos Ramos/PB — IBGE 2514453",
    "251450": "São José de Piranhas/PB — IBGE 2514503",
    "251455": "São José de Princesa/PB — IBGE 2514552",
    "251460": "São José do Bonfim/PB — IBGE 2514602",
    "251465": "São José do Brejo do Cruz/PB — IBGE 2514651",
    "251470": "São José do Sabugi/PB — IBGE 2514701",
    "251480": "São José dos Cordeiros/PB — IBGE 2514800",
    "251490": "São Mamede/PB — IBGE 2514909",
    "251500": "São Miguel de Taipu/PB — IBGE 2515005",
    "251510": "São Sebastião de Lagoa de Roça/PB — IBGE 2515104",
    "251520": "São Sebastião do Umbuzeiro/PB — IBGE 2515203",
    "251530": "Sapé/PB — IBGE 2515302",
    "251540": "São Vicente do Seridó/PB — IBGE 2515401",
    "251550": "Serra Branca/PB — IBGE 2515500",
    "251560": "Serra da Raiz/PB — IBGE 2515609",
    "251570": "Serra Grande/PB — IBGE 2515708",
    "251580": "Serra Redonda/PB — IBGE 2515807",
    "251590": "Serraria/PB — IBGE 2515906",
    "251593": "Sertãozinho/PB — IBGE 2515930",
    "251597": "Sobrado/PB — IBGE 2515971",
    "251600": "Solânea/PB — IBGE 2516003",
    "251610": "Soledade/PB — IBGE 2516102",
    "251615": "Sossêgo/PB — IBGE 2516151",
    "251620": "Sousa/PB — IBGE 2516201",
    "251630": "Sumé/PB — IBGE 2516300",
    "251640": "Tacima/PB — IBGE 2516409",
    "251650": "Taperoá/PB — IBGE 2516508",
    "251660": "Tavares/PB — IBGE 2516607",
    "251670": "Teixeira/PB — IBGE 2516706",
    "251675": "Tenório/PB — IBGE 2516755",
    "251680": "Triunfo/PB — IBGE 2516805",
    "251690": "Uiraúna/PB — IBGE 2516904",
    "251700": "Umbuzeiro/PB — IBGE 2517001",
    "251710": "Várzea/PB — IBGE 2517100",
    "251720": "Vieirópolis/PB — IBGE 2517209",
    "251740": "Zabelê/PB — IBGE 2517407",
    "260005": "Abreu e Lima/PE — IBGE 2600054",
    "260010": "Afogados da Ingazeira/PE — IBGE 2600104",
    "260020": "Afrânio/PE — IBGE 2600203",
    "260030": "Agrestina/PE — IBGE 2600302",
    "260040": "Água Preta/PE — IBGE 2600401",
    "260050": "Águas Belas/PE — IBGE 2600500",
    "260060": "Alagoinha/PE — IBGE 2600609",
    "260070": "Aliança/PE — IBGE 2600708",
    "260080": "Altinho/PE — IBGE 2600807",
    "260090": "Amaraji/PE — IBGE 2600906",
    "260100": "Angelim/PE — IBGE 2601003",
    "260105": "Araçoiaba/PE — IBGE 2601052",
    "260110": "Araripina/PE — IBGE 2601102",
    "260120": "Arcoverde/PE — IBGE 2601201",
    "260130": "Barra de Guabiraba/PE — IBGE 2601300",
    "260140": "Barreiros/PE — IBGE 2601409",
    "260150": "Belém de Maria/PE — IBGE 2601508",
    "260160": "Belém do São Francisco/PE — IBGE 2601607",
    "260170": "Belo Jardim/PE — IBGE 2601706",
    "260180": "Betânia/PE — IBGE 2601805",
    "260190": "Bezerros/PE — IBGE 2601904",
    "260200": "Bodocó/PE — IBGE 2602001",
    "260210": "Bom Conselho/PE — IBGE 2602100",
    "260220": "Bom Jardim/PE — IBGE 2602209",
    "260230": "Bonito/PE — IBGE 2602308",
    "260240": "Brejão/PE — IBGE 2602407",
    "260250": "Brejinho/PE — IBGE 2602506",
    "260260": "Brejo da Madre de Deus/PE — IBGE 2602605",
    "260270": "Buenos Aires/PE — IBGE 2602704",
    "260280": "Buíque/PE — IBGE 2602803",
    "260290": "Cabo de Santo Agostinho/PE — IBGE 2602902",
    "260300": "Cabrobó/PE — IBGE 2603009",
    "260310": "Cachoeirinha/PE — IBGE 2603108",
    "260320": "Caetés/PE — IBGE 2603207",
    "260330": "Calçado/PE — IBGE 2603306",
    "260340": "Calumbi/PE — IBGE 2603405",
    "260345": "Camaragibe/PE — IBGE 2603454",
    "260350": "Camocim de São Félix/PE — IBGE 2603504",
    "260360": "Camutanga/PE — IBGE 2603603",
    "260370": "Canhotinho/PE — IBGE 2603702",
    "260380": "Capoeiras/PE — IBGE 2603801",
    "260390": "Carnaíba/PE — IBGE 2603900",
    "260392": "Carnaubeira da Penha/PE — IBGE 2603926",
    "260400": "Carpina/PE — IBGE 2604007",
    "260410": "Caruaru/PE — IBGE 2604106",
    "260415": "Casinhas/PE — IBGE 2604155",
    "260420": "Catende/PE — IBGE 2604205",
    "260430": "Cedro/PE — IBGE 2604304",
    "260440": "Chã de Alegria/PE — IBGE 2604403",
    "260450": "Chã Grande/PE — IBGE 2604502",
    "260460": "Condado/PE — IBGE 2604601",
    "260470": "Correntes/PE — IBGE 2604700",
    "260480": "Cortês/PE — IBGE 2604809",
    "260490": "Cumaru/PE — IBGE 2604908",
    "260500": "Cupira/PE — IBGE 2605004",
    "260510": "Custódia/PE — IBGE 2605103",
    "260515": "Dormentes/PE — IBGE 2605152",
    "260520": "Escada/PE — IBGE 2605202",
    "260530": "Exu/PE — IBGE 2605301",
    "260540": "Feira Nova/PE — IBGE 2605400",
    "260545": "Fernando de Noronha/PE — IBGE 2605459",
    "260550": "Ferreiros/PE — IBGE 2605509",
    "260560": "Flores/PE — IBGE 2605608",
    "260570": "Floresta/PE — IBGE 2605707",
    "260580": "Frei Miguelinho/PE — IBGE 2605806",
    "260590": "Gameleira/PE — IBGE 2605905",
    "260600": "Garanhuns/PE — IBGE 2606002",
    "260610": "Glória do Goitá/PE — IBGE 2606101",
    "260620": "Goiana/PE — IBGE 2606200",
    "260630": "Granito/PE — IBGE 2606309",
    "260640": "Gravatá/PE — IBGE 2606408",
    "260650": "Iati/PE — IBGE 2606507",
    "260660": "Ibimirim/PE — IBGE 2606606",
    "260670": "Ibirajuba/PE — IBGE 2606705",
    "260680": "Igarassu/PE — IBGE 2606804",
    "260690": "Iguaracy/PE — IBGE 2606903",
    "260700": "Inajá/PE — IBGE 2607000",
    "260710": "Ingazeira/PE — IBGE 2607109",
    "260720": "Ipojuca/PE — IBGE 2607208",
    "260730": "Ipubi/PE — IBGE 2607307",
    "260740": "Itacuruba/PE — IBGE 2607406",
    "260750": "Itaíba/PE — IBGE 2607505",
    "260760": "Ilha de Itamaracá/PE — IBGE 2607604",
    "260765": "Itambé/PE — IBGE 2607653",
    "260770": "Itapetim/PE — IBGE 2607703",
    "260775": "Itapissuma/PE — IBGE 2607752",
    "260780": "Itaquitinga/PE — IBGE 2607802",
    "260790": "Jaboatão dos Guararapes/PE — IBGE 2607901",
    "260795": "Jaqueira/PE — IBGE 2607950",
    "260800": "Jataúba/PE — IBGE 2608008",
    "260805": "Jatobá/PE — IBGE 2608057",
    "260810": "João Alfredo/PE — IBGE 2608107",
    "260820": "Joaquim Nabuco/PE — IBGE 2608206",
    "260825": "Jucati/PE — IBGE 2608255",
    "260830": "Jupi/PE — IBGE 2608305",
    "260840": "Jurema/PE — IBGE 2608404",
    "260845": "Lagoa do Carro/PE — IBGE 2608453",
    "260850": "Lagoa de Itaenga/PE — IBGE 2608503",
    "260860": "Lagoa do Ouro/PE — IBGE 2608602",
    "260870": "Lagoa dos Gatos/PE — IBGE 2608701",
    "260875": "Lagoa Grande/PE — IBGE 2608750",
    "260880": "Lajedo/PE — IBGE 2608800",
    "260890": "Limoeiro/PE — IBGE 2608909",
    "260900": "Macaparana/PE — IBGE 2609006",
    "260910": "Machados/PE — IBGE 2609105",
    "260915": "Manari/PE — IBGE 2609154",
    "260920": "Maraial/PE — IBGE 2609204",
    "260930": "Mirandiba/PE — IBGE 2609303",
    "260940": "Moreno/PE — IBGE 2609402",
    "260950": "Nazaré da Mata/PE — IBGE 2609501",
    "260960": "Olinda/PE — IBGE 2609600",
    "260970": "Orobó/PE — IBGE 2609709",
    "260980": "Orocó/PE — IBGE 2609808",
    "260990": "Ouricuri/PE — IBGE 2609907",
    "261000": "Palmares/PE — IBGE 2610004",
    "261010": "Palmeirina/PE — IBGE 2610103",
    "261020": "Panelas/PE — IBGE 2610202",
    "261030": "Paranatama/PE — IBGE 2610301",
    "261040": "Parnamirim/PE — IBGE 2610400",
    "261050": "Passira/PE — IBGE 2610509",
    "261060": "Paudalho/PE — IBGE 2610608",
    "261070": "Paulista/PE — IBGE 2610707",
    "261080": "Pedra/PE — IBGE 2610806",
    "261090": "Pesqueira/PE — IBGE 2610905",
    "261100": "Petrolândia/PE — IBGE 2611002",
    "261110": "Petrolina/PE — IBGE 2611101",
    "261120": "Poção/PE — IBGE 2611200",
    "261130": "Pombos/PE — IBGE 2611309",
    "261140": "Primavera/PE — IBGE 2611408",
    "261150": "Quipapá/PE — IBGE 2611507",
    "261153": "Quixaba/PE — IBGE 2611533",
    "261160": "Recife/PE — IBGE 2611606",
    "261170": "Riacho das Almas/PE — IBGE 2611705",
    "261180": "Ribeirão/PE — IBGE 2611804",
    "261190": "Rio Formoso/PE — IBGE 2611903",
    "261200": "Sairé/PE — IBGE 2612000",
    "261210": "Salgadinho/PE — IBGE 2612109",
    "261220": "Salgueiro/PE — IBGE 2612208",
    "261230": "Saloá/PE — IBGE 2612307",
    "261240": "Sanharó/PE — IBGE 2612406",
    "261245": "Santa Cruz/PE — IBGE 2612455",
    "261247": "Santa Cruz da Baixa Verde/PE — IBGE 2612471",
    "261250": "Santa Cruz do Capibaribe/PE — IBGE 2612505",
    "261255": "Santa Filomena/PE — IBGE 2612554",
    "261260": "Santa Maria da Boa Vista/PE — IBGE 2612604",
    "261270": "Santa Maria do Cambucá/PE — IBGE 2612703",
    "261280": "Santa Terezinha/PE — IBGE 2612802",
    "261290": "São Benedito do Sul/PE — IBGE 2612901",
    "261300": "São Bento do Una/PE — IBGE 2613008",
    "261310": "São Caitano/PE — IBGE 2613107",
    "261320": "São João/PE — IBGE 2613206",
    "261330": "São Joaquim do Monte/PE — IBGE 2613305",
    "261340": "São José da Coroa Grande/PE — IBGE 2613404",
    "261350": "São José do Belmonte/PE — IBGE 2613503",
    "261360": "São José do Egito/PE — IBGE 2613602",
    "261370": "São Lourenço da Mata/PE — IBGE 2613701",
    "261380": "São Vicente Férrer/PE — IBGE 2613800",
    "261390": "Serra Talhada/PE — IBGE 2613909",
    "261400": "Serrita/PE — IBGE 2614006",
    "261410": "Sertânia/PE — IBGE 2614105",
    "261420": "Sirinhaém/PE — IBGE 2614204",
    "261430": "Moreilândia/PE — IBGE 2614303",
    "261440": "Solidão/PE — IBGE 2614402",
    "261450": "Surubim/PE — IBGE 2614501",
    "261460": "Tabira/PE — IBGE 2614600",
    "261470": "Tacaimbó/PE — IBGE 2614709",
    "261480": "Tacaratu/PE — IBGE 2614808",
    "261485": "Tamandaré/PE — IBGE 2614857",
    "261500": "Taquaritinga do Norte/PE — IBGE 2615003",
    "261510": "Terezinha/PE — IBGE 2615102",
    "261520": "Terra Nova/PE — IBGE 2615201",
    "261530": "Timbaúba/PE — IBGE 2615300",
    "261540": "Toritama/PE — IBGE 2615409",
    "261550": "Tracunhaém/PE — IBGE 2615508",
    "261560": "Trindade/PE — IBGE 2615607",
    "261570": "Triunfo/PE — IBGE 2615706",
    "261580": "Tupanatinga/PE — IBGE 2615805",
    "261590": "Tuparetama/PE — IBGE 2615904",
    "261600": "Venturosa/PE — IBGE 2616001",
    "261610": "Verdejante/PE — IBGE 2616100",
    "261618": "Vertente do Lério/PE — IBGE 2616183",
    "261620": "Vertentes/PE — IBGE 2616209",
    "261630": "Vicência/PE — IBGE 2616308",
    "261640": "Vitória de Santo Antão/PE — IBGE 2616407",
    "261650": "Xexéu/PE — IBGE 2616506",
    "270010": "Água Branca/AL — IBGE 2700102",
    "270020": "Anadia/AL — IBGE 2700201",
    "270030": "Arapiraca/AL — IBGE 2700300",
    "270040": "Atalaia/AL — IBGE 2700409",
    "270050": "Barra de Santo Antônio/AL — IBGE 2700508",
    "270060": "Barra de São Miguel/AL — IBGE 2700607",
    "270070": "Batalha/AL — IBGE 2700706",
    "270080": "Belém/AL — IBGE 2700805",
    "270090": "Belo Monte/AL — IBGE 2700904",
    "270100": "Boca da Mata/AL — IBGE 2701001",
    "270110": "Branquinha/AL — IBGE 2701100",
    "270120": "Cacimbinhas/AL — IBGE 2701209",
    "270130": "Cajueiro/AL — IBGE 2701308",
    "270135": "Campestre/AL — IBGE 2701357",
    "270140": "Campo Alegre/AL — IBGE 2701407",
    "270150": "Campo Grande/AL — IBGE 2701506",
    "270160": "Canapi/AL — IBGE 2701605",
    "270170": "Capela/AL — IBGE 2701704",
    "270180": "Carneiros/AL — IBGE 2701803",
    "270190": "Chã Preta/AL — IBGE 2701902",
    "270200": "Coité do Nóia/AL — IBGE 2702009",
    "270210": "Colônia Leopoldina/AL — IBGE 2702108",
    "270220": "Coqueiro Seco/AL — IBGE 2702207",
    "270230": "Coruripe/AL — IBGE 2702306",
    "270235": "Craíbas/AL — IBGE 2702355",
    "270240": "Delmiro Gouveia/AL — IBGE 2702405",
    "270250": "Dois Riachos/AL — IBGE 2702504",
    "270255": "Estrela de Alagoas/AL — IBGE 2702553",
    "270260": "Feira Grande/AL — IBGE 2702603",
    "270270": "Feliz Deserto/AL — IBGE 2702702",
    "270280": "Flexeiras/AL — IBGE 2702801",
    "270290": "Girau do Ponciano/AL — IBGE 2702900",
    "270300": "Ibateguara/AL — IBGE 2703007",
    "270310": "Igaci/AL — IBGE 2703106",
    "270320": "Igreja Nova/AL — IBGE 2703205",
    "270330": "Inhapi/AL — IBGE 2703304",
    "270340": "Jacaré dos Homens/AL — IBGE 2703403",
    "270350": "Jacuípe/AL — IBGE 2703502",
    "270360": "Japaratinga/AL — IBGE 2703601",
    "270370": "Jaramataia/AL — IBGE 2703700",
    "270375": "Jequiá da Praia/AL — IBGE 2703759",
    "270380": "Joaquim Gomes/AL — IBGE 2703809",
    "270390": "Jundiá/AL — IBGE 2703908",
    "270400": "Junqueiro/AL — IBGE 2704005",
    "270410": "Lagoa da Canoa/AL — IBGE 2704104",
    "270420": "Limoeiro de Anadia/AL — IBGE 2704203",
    "270430": "Maceió/AL — IBGE 2704302",
    "270440": "Major Isidoro/AL — IBGE 2704401",
    "270450": "Maragogi/AL — IBGE 2704500",
    "270460": "Maravilha/AL — IBGE 2704609",
    "270470": "Marechal Deodoro/AL — IBGE 2704708",
    "270480": "Maribondo/AL — IBGE 2704807",
    "270490": "Mar Vermelho/AL — IBGE 2704906",
    "270500": "Mata Grande/AL — IBGE 2705002",
    "270510": "Matriz de Camaragibe/AL — IBGE 2705101",
    "270520": "Messias/AL — IBGE 2705200",
    "270530": "Minador do Negrão/AL — IBGE 2705309",
    "270540": "Monteirópolis/AL — IBGE 2705408",
    "270550": "Murici/AL — IBGE 2705507",
    "270560": "Novo Lino/AL — IBGE 2705606",
    "270570": "Olho d'Água das Flores/AL — IBGE 2705705",
    "270580": "Olho d'Água do Casado/AL — IBGE 2705804",
    "270590": "Olho d'Água Grande/AL — IBGE 2705903",
    "270600": "Olivença/AL — IBGE 2706000",
    "270610": "Ouro Branco/AL — IBGE 2706109",
    "270620": "Palestina/AL — IBGE 2706208",
    "270630": "Palmeira dos Índios/AL — IBGE 2706307",
    "270640": "Pão de Açúcar/AL — IBGE 2706406",
    "270642": "Pariconha/AL — IBGE 2706422",
    "270644": "Paripueira/AL — IBGE 2706448",
    "270650": "Passo de Camaragibe/AL — IBGE 2706505",
    "270660": "Paulo Jacinto/AL — IBGE 2706604",
    "270670": "Penedo/AL — IBGE 2706703",
    "270680": "Piaçabuçu/AL — IBGE 2706802",
    "270690": "Pilar/AL — IBGE 2706901",
    "270700": "Pindoba/AL — IBGE 2707008",
    "270710": "Piranhas/AL — IBGE 2707107",
    "270720": "Poço das Trincheiras/AL — IBGE 2707206",
    "270730": "Porto Calvo/AL — IBGE 2707305",
    "270740": "Porto de Pedras/AL — IBGE 2707404",
    "270750": "Porto Real do Colégio/AL — IBGE 2707503",
    "270760": "Quebrangulo/AL — IBGE 2707602",
    "270770": "Rio Largo/AL — IBGE 2707701",
    "270780": "Roteiro/AL — IBGE 2707800",
    "270790": "Santa Luzia do Norte/AL — IBGE 2707909",
    "270800": "Santana do Ipanema/AL — IBGE 2708006",
    "270810": "Santana do Mundaú/AL — IBGE 2708105",
    "270820": "São Brás/AL — IBGE 2708204",
    "270830": "São José da Laje/AL — IBGE 2708303",
    "270840": "São José da Tapera/AL — IBGE 2708402",
    "270850": "São Luís do Quitunde/AL — IBGE 2708501",
    "270860": "São Miguel dos Campos/AL — IBGE 2708600",
    "270870": "São Miguel dos Milagres/AL — IBGE 2708709",
    "270880": "São Sebastião/AL — IBGE 2708808",
    "270890": "Satuba/AL — IBGE 2708907",
    "270895": "Senador Rui Palmeira/AL — IBGE 2708956",
    "270900": "Tanque d'Arca/AL — IBGE 2709004",
    "270910": "Taquarana/AL — IBGE 2709103",
    "270915": "Teotônio Vilela/AL — IBGE 2709152",
    "270920": "Traipu/AL — IBGE 2709202",
    "270930": "União dos Palmares/AL — IBGE 2709301",
    "270940": "Viçosa/AL — IBGE 2709400",
    "280010": "Amparo do São Francisco/SE — IBGE 2800100",
    "280020": "Aquidabã/SE — IBGE 2800209",
    "280030": "Aracaju/SE — IBGE 2800308",
    "280040": "Arauá/SE — IBGE 2800407",
    "280050": "Areia Branca/SE — IBGE 2800506",
    "280060": "Barra dos Coqueiros/SE — IBGE 2800605",
    "280067": "Boquim/SE — IBGE 2800670",
    "280070": "Brejo Grande/SE — IBGE 2800704",
    "280100": "Campo do Brito/SE — IBGE 2801009",
    "280110": "Canhoba/SE — IBGE 2801108",
    "280120": "Canindé de São Francisco/SE — IBGE 2801207",
    "280130": "Capela/SE — IBGE 2801306",
    "280140": "Carira/SE — IBGE 2801405",
    "280150": "Carmópolis/SE — IBGE 2801504",
    "280160": "Cedro de São João/SE — IBGE 2801603",
    "280170": "Cristinápolis/SE — IBGE 2801702",
    "280190": "Cumbe/SE — IBGE 2801900",
    "280200": "Divina Pastora/SE — IBGE 2802007",
    "280210": "Estância/SE — IBGE 2802106",
    "280220": "Feira Nova/SE — IBGE 2802205",
    "280230": "Frei Paulo/SE — IBGE 2802304",
    "280240": "Gararu/SE — IBGE 2802403",
    "280250": "General Maynard/SE — IBGE 2802502",
    "280260": "Gracho Cardoso/SE — IBGE 2802601",
    "280270": "Ilha das Flores/SE — IBGE 2802700",
    "280280": "Indiaroba/SE — IBGE 2802809",
    "280290": "Itabaiana/SE — IBGE 2802908",
    "280300": "Itabaianinha/SE — IBGE 2803005",
    "280310": "Itabi/SE — IBGE 2803104",
    "280320": "Itaporanga d'Ajuda/SE — IBGE 2803203",
    "280330": "Japaratuba/SE — IBGE 2803302",
    "280340": "Japoatã/SE — IBGE 2803401",
    "280350": "Lagarto/SE — IBGE 2803500",
    "280360": "Laranjeiras/SE — IBGE 2803609",
    "280370": "Macambira/SE — IBGE 2803708",
    "280380": "Malhada dos Bois/SE — IBGE 2803807",
    "280390": "Malhador/SE — IBGE 2803906",
    "280400": "Maruim/SE — IBGE 2804003",
    "280410": "Moita Bonita/SE — IBGE 2804102",
    "280420": "Monte Alegre de Sergipe/SE — IBGE 2804201",
    "280430": "Muribeca/SE — IBGE 2804300",
    "280440": "Neópolis/SE — IBGE 2804409",
    "280445": "Nossa Senhora Aparecida/SE — IBGE 2804458",
    "280450": "Nossa Senhora da Glória/SE — IBGE 2804508",
    "280460": "Nossa Senhora das Dores/SE — IBGE 2804607",
    "280470": "Nossa Senhora de Lourdes/SE — IBGE 2804706",
    "280480": "Nossa Senhora do Socorro/SE — IBGE 2804805",
    "280490": "Pacatuba/SE — IBGE 2804904",
    "280500": "Pedra Mole/SE — IBGE 2805000",
    "280510": "Pedrinhas/SE — IBGE 2805109",
    "280520": "Pinhão/SE — IBGE 2805208",
    "280530": "Pirambu/SE — IBGE 2805307",
    "280540": "Poço Redondo/SE — IBGE 2805406",
    "280550": "Poço Verde/SE — IBGE 2805505",
    "280560": "Porto da Folha/SE — IBGE 2805604",
    "280570": "Propriá/SE — IBGE 2805703",
    "280580": "Riachão do Dantas/SE — IBGE 2805802",
    "280590": "Riachuelo/SE — IBGE 2805901",
    "280600": "Ribeirópolis/SE — IBGE 2806008",
    "280610": "Rosário do Catete/SE — IBGE 2806107",
    "280620": "Salgado/SE — IBGE 2806206",
    "280630": "Santa Luzia do Itanhy/SE — IBGE 2806305",
    "280640": "Santana do São Francisco/SE — IBGE 2806404",
    "280650": "Santa Rosa de Lima/SE — IBGE 2806503",
    "280660": "Santo Amaro das Brotas/SE — IBGE 2806602",
    "280670": "São Cristóvão/SE — IBGE 2806701",
    "280680": "São Domingos/SE — IBGE 2806800",
    "280690": "São Francisco/SE — IBGE 2806909",
    "280700": "São Miguel do Aleixo/SE — IBGE 2807006",
    "280710": "Simão Dias/SE — IBGE 2807105",
    "280720": "Siriri/SE — IBGE 2807204",
    "280730": "Telha/SE — IBGE 2807303",
    "280740": "Tobias Barreto/SE — IBGE 2807402",
    "280750": "Tomar do Geru/SE — IBGE 2807501",
    "280760": "Umbaúba/SE — IBGE 2807600",
    "290010": "Abaíra/BA — IBGE 2900108",
    "290020": "Abaré/BA — IBGE 2900207",
    "290030": "Acajutiba/BA — IBGE 2900306",
    "290035": "Adustina/BA — IBGE 2900355",
    "290040": "Água Fria/BA — IBGE 2900405",
    "290050": "Érico Cardoso/BA — IBGE 2900504",
    "290060": "Aiquara/BA — IBGE 2900603",
    "290070": "Alagoinhas/BA — IBGE 2900702",
    "290080": "Alcobaça/BA — IBGE 2900801",
    "290090": "Almadina/BA — IBGE 2900900",
    "290100": "Amargosa/BA — IBGE 2901007",
    "290110": "Amélia Rodrigues/BA — IBGE 2901106",
    "290115": "América Dourada/BA — IBGE 2901155",
    "290120": "Anagé/BA — IBGE 2901205",
    "290130": "Andaraí/BA — IBGE 2901304",
    "290135": "Andorinha/BA — IBGE 2901353",
    "290140": "Angical/BA — IBGE 2901403",
    "290150": "Anguera/BA — IBGE 2901502",
    "290160": "Antas/BA — IBGE 2901601",
    "290170": "Antônio Cardoso/BA — IBGE 2901700",
    "290180": "Antônio Gonçalves/BA — IBGE 2901809",
    "290190": "Aporá/BA — IBGE 2901908",
    "290195": "Apuarema/BA — IBGE 2901957",
    "290200": "Aracatu/BA — IBGE 2902005",
    "290205": "Araçás/BA — IBGE 2902054",
    "290210": "Araci/BA — IBGE 2902104",
    "290220": "Aramari/BA — IBGE 2902203",
    "290225": "Arataca/BA — IBGE 2902252",
    "290230": "Aratuípe/BA — IBGE 2902302",
    "290240": "Aurelino Leal/BA — IBGE 2902401",
    "290250": "Baianópolis/BA — IBGE 2902500",
    "290260": "Baixa Grande/BA — IBGE 2902609",
    "290265": "Banzaê/BA — IBGE 2902658",
    "290270": "Barra/BA — IBGE 2902708",
    "290280": "Barra da Estiva/BA — IBGE 2902807",
    "290290": "Barra do Choça/BA — IBGE 2902906",
    "290300": "Barra do Mendes/BA — IBGE 2903003",
    "290310": "Barra do Rocha/BA — IBGE 2903102",
    "290320": "Barreiras/BA — IBGE 2903201",
    "290323": "Barro Alto/BA — IBGE 2903235",
    "290327": "Barrocas/BA — IBGE 2903276",
    "290330": "Barro Preto/BA — IBGE 2903300",
    "290340": "Belmonte/BA — IBGE 2903409",
    "290350": "Belo Campo/BA — IBGE 2903508",
    "290360": "Biritinga/BA — IBGE 2903607",
    "290370": "Boa Nova/BA — IBGE 2903706",
    "290380": "Boa Vista do Tupim/BA — IBGE 2903805",
    "290390": "Bom Jesus da Lapa/BA — IBGE 2903904",
    "290395": "Bom Jesus da Serra/BA — IBGE 2903953",
    "290400": "Boninal/BA — IBGE 2904001",
    "290405": "Bonito/BA — IBGE 2904050",
    "290410": "Boquira/BA — IBGE 2904100",
    "290420": "Botuporã/BA — IBGE 2904209",
    "290430": "Brejões/BA — IBGE 2904308",
    "290440": "Brejolândia/BA — IBGE 2904407",
    "290450": "Brotas de Macaúbas/BA — IBGE 2904506",
    "290460": "Brumado/BA — IBGE 2904605",
    "290470": "Buerarema/BA — IBGE 2904704",
    "290475": "Buritirama/BA — IBGE 2904753",
    "290480": "Caatiba/BA — IBGE 2904803",
    "290485": "Cabaceiras do Paraguaçu/BA — IBGE 2904852",
    "290490": "Cachoeira/BA — IBGE 2904902",
    "290500": "Caculé/BA — IBGE 2905008",
    "290510": "Caém/BA — IBGE 2905107",
    "290515": "Caetanos/BA — IBGE 2905156",
    "290520": "Caetité/BA — IBGE 2905206",
    "290530": "Cafarnaum/BA — IBGE 2905305",
    "290540": "Cairu/BA — IBGE 2905404",
    "290550": "Caldeirão Grande/BA — IBGE 2905503",
    "290560": "Camacan/BA — IBGE 2905602",
    "290570": "Camaçari/BA — IBGE 2905701",
    "290580": "Camamu/BA — IBGE 2905800",
    "290590": "Campo Alegre de Lourdes/BA — IBGE 2905909",
    "290600": "Campo Formoso/BA — IBGE 2906006",
    "290610": "Canápolis/BA — IBGE 2906105",
    "290620": "Canarana/BA — IBGE 2906204",
    "290630": "Canavieiras/BA — IBGE 2906303",
    "290640": "Candeal/BA — IBGE 2906402",
    "290650": "Candeias/BA — IBGE 2906501",
    "290660": "Candiba/BA — IBGE 2906600",
    "290670": "Cândido Sales/BA — IBGE 2906709",
    "290680": "Cansanção/BA — IBGE 2906808",
    "290682": "Canudos/BA — IBGE 2906824",
    "290685": "Capela do Alto Alegre/BA — IBGE 2906857",
    "290687": "Capim Grosso/BA — IBGE 2906873",
    "290689": "Caraíbas/BA — IBGE 2906899",
    "290690": "Caravelas/BA — IBGE 2906907",
    "290700": "Cardeal da Silva/BA — IBGE 2907004",
    "290710": "Carinhanha/BA — IBGE 2907103",
    "290720": "Casa Nova/BA — IBGE 2907202",
    "290730": "Castro Alves/BA — IBGE 2907301",
    "290740": "Catolândia/BA — IBGE 2907400",
    "290750": "Catu/BA — IBGE 2907509",
    "290755": "Caturama/BA — IBGE 2907558",
    "290760": "Central/BA — IBGE 2907608",
    "290770": "Chorrochó/BA — IBGE 2907707",
    "290780": "Cícero Dantas/BA — IBGE 2907806",
    "290790": "Cipó/BA — IBGE 2907905",
    "290800": "Coaraci/BA — IBGE 2908002",
    "290810": "Cocos/BA — IBGE 2908101",
    "290820": "Conceição da Feira/BA — IBGE 2908200",
    "290830": "Conceição do Almeida/BA — IBGE 2908309",
    "290840": "Conceição do Coité/BA — IBGE 2908408",
    "290850": "Conceição do Jacuípe/BA — IBGE 2908507",
    "290860": "Conde/BA — IBGE 2908606",
    "290870": "Condeúba/BA — IBGE 2908705",
    "290880": "Contendas do Sincorá/BA — IBGE 2908804",
    "290890": "Coração de Maria/BA — IBGE 2908903",
    "290900": "Cordeiros/BA — IBGE 2909000",
    "290910": "Coribe/BA — IBGE 2909109",
    "290920": "Coronel João Sá/BA — IBGE 2909208",
    "290930": "Correntina/BA — IBGE 2909307",
    "290940": "Cotegipe/BA — IBGE 2909406",
    "290950": "Cravolândia/BA — IBGE 2909505",
    "290960": "Crisópolis/BA — IBGE 2909604",
    "290970": "Cristópolis/BA — IBGE 2909703",
    "290980": "Cruz das Almas/BA — IBGE 2909802",
    "290990": "Curaçá/BA — IBGE 2909901",
    "291000": "Dário Meira/BA — IBGE 2910008",
    "291005": "Dias d'Ávila/BA — IBGE 2910057",
    "291010": "Dom Basílio/BA — IBGE 2910107",
    "291020": "Dom Macedo Costa/BA — IBGE 2910206",
    "291030": "Elísio Medrado/BA — IBGE 2910305",
    "291040": "Encruzilhada/BA — IBGE 2910404",
    "291050": "Entre Rios/BA — IBGE 2910503",
    "291060": "Esplanada/BA — IBGE 2910602",
    "291070": "Euclides da Cunha/BA — IBGE 2910701",
    "291072": "Eunápolis/BA — IBGE 2910727",
    "291075": "Fátima/BA — IBGE 2910750",
    "291077": "Feira da Mata/BA — IBGE 2910776",
    "291080": "Feira de Santana/BA — IBGE 2910800",
    "291085": "Filadélfia/BA — IBGE 2910859",
    "291090": "Firmino Alves/BA — IBGE 2910909",
    "291100": "Floresta Azul/BA — IBGE 2911006",
    "291110": "Formosa do Rio Preto/BA — IBGE 2911105",
    "291120": "Gandu/BA — IBGE 2911204",
    "291125": "Gavião/BA — IBGE 2911253",
    "291130": "Gentio do Ouro/BA — IBGE 2911303",
    "291140": "Glória/BA — IBGE 2911402",
    "291150": "Gongogi/BA — IBGE 2911501",
    "291160": "Governador Mangabeira/BA — IBGE 2911600",
    "291165": "Guajeru/BA — IBGE 2911659",
    "291170": "Guanambi/BA — IBGE 2911709",
    "291180": "Guaratinga/BA — IBGE 2911808",
    "291185": "Heliópolis/BA — IBGE 2911857",
    "291190": "Iaçu/BA — IBGE 2911907",
    "291200": "Ibiassucê/BA — IBGE 2912004",
    "291210": "Ibicaraí/BA — IBGE 2912103",
    "291220": "Ibicoara/BA — IBGE 2912202",
    "291230": "Ibicuí/BA — IBGE 2912301",
    "291240": "Ibipeba/BA — IBGE 2912400",
    "291250": "Ibipitanga/BA — IBGE 2912509",
    "291260": "Ibiquera/BA — IBGE 2912608",
    "291270": "Ibirapitanga/BA — IBGE 2912707",
    "291280": "Ibirapuã/BA — IBGE 2912806",
    "291290": "Ibirataia/BA — IBGE 2912905",
    "291300": "Ibitiara/BA — IBGE 2913002",
    "291310": "Ibititá/BA — IBGE 2913101",
    "291320": "Ibotirama/BA — IBGE 2913200",
    "291330": "Ichu/BA — IBGE 2913309",
    "291340": "Igaporã/BA — IBGE 2913408",
    "291345": "Igrapiúna/BA — IBGE 2913457",
    "291350": "Iguaí/BA — IBGE 2913507",
    "291360": "Ilhéus/BA — IBGE 2913606",
    "291370": "Inhambupe/BA — IBGE 2913705",
    "291380": "Ipecaetá/BA — IBGE 2913804",
    "291390": "Ipiaú/BA — IBGE 2913903",
    "291400": "Ipirá/BA — IBGE 2914000",
    "291410": "Ipupiara/BA — IBGE 2914109",
    "291420": "Irajuba/BA — IBGE 2914208",
    "291430": "Iramaia/BA — IBGE 2914307",
    "291440": "Iraquara/BA — IBGE 2914406",
    "291450": "Irará/BA — IBGE 2914505",
    "291460": "Irecê/BA — IBGE 2914604",
    "291465": "Itabela/BA — IBGE 2914653",
    "291470": "Itaberaba/BA — IBGE 2914703",
    "291480": "Itabuna/BA — IBGE 2914802",
    "291490": "Itacaré/BA — IBGE 2914901",
    "291500": "Itaeté/BA — IBGE 2915007",
    "291510": "Itagi/BA — IBGE 2915106",
    "291520": "Itagibá/BA — IBGE 2915205",
    "291530": "Itagimirim/BA — IBGE 2915304",
    "291535": "Itaguaçu da Bahia/BA — IBGE 2915353",
    "291540": "Itaju do Colônia/BA — IBGE 2915403",
    "291550": "Itajuípe/BA — IBGE 2915502",
    "291560": "Itamaraju/BA — IBGE 2915601",
    "291570": "Itamari/BA — IBGE 2915700",
    "291580": "Itambé/BA — IBGE 2915809",
    "291590": "Itanagra/BA — IBGE 2915908",
    "291600": "Itanhém/BA — IBGE 2916005",
    "291610": "Itaparica/BA — IBGE 2916104",
    "291620": "Itapé/BA — IBGE 2916203",
    "291630": "Itapebi/BA — IBGE 2916302",
    "291640": "Itapetinga/BA — IBGE 2916401",
    "291650": "Itapicuru/BA — IBGE 2916500",
    "291660": "Itapitanga/BA — IBGE 2916609",
    "291670": "Itaquara/BA — IBGE 2916708",
    "291680": "Itarantim/BA — IBGE 2916807",
    "291685": "Itatim/BA — IBGE 2916856",
    "291690": "Itiruçu/BA — IBGE 2916906",
    "291700": "Itiúba/BA — IBGE 2917003",
    "291710": "Itororó/BA — IBGE 2917102",
    "291720": "Ituaçu/BA — IBGE 2917201",
    "291730": "Ituberá/BA — IBGE 2917300",
    "291733": "Iuiu/BA — IBGE 2917334",
    "291735": "Jaborandi/BA — IBGE 2917359",
    "291740": "Jacaraci/BA — IBGE 2917409",
    "291750": "Jacobina/BA — IBGE 2917508",
    "291760": "Jaguaquara/BA — IBGE 2917607",
    "291770": "Jaguarari/BA — IBGE 2917706",
    "291780": "Jaguaripe/BA — IBGE 2917805",
    "291790": "Jandaíra/BA — IBGE 2917904",
    "291800": "Jequié/BA — IBGE 2918001",
    "291810": "Jeremoabo/BA — IBGE 2918100",
    "291820": "Jiquiriçá/BA — IBGE 2918209",
    "291830": "Jitaúna/BA — IBGE 2918308",
    "291835": "João Dourado/BA — IBGE 2918357",
    "291840": "Juazeiro/BA — IBGE 2918407",
    "291845": "Jucuruçu/BA — IBGE 2918456",
    "291850": "Jussara/BA — IBGE 2918506",
    "291855": "Jussari/BA — IBGE 2918555",
    "291860": "Jussiape/BA — IBGE 2918605",
    "291870": "Lafaiete Coutinho/BA — IBGE 2918704",
    "291875": "Lagoa Real/BA — IBGE 2918753",
    "291880": "Laje/BA — IBGE 2918803",
    "291890": "Lajedão/BA — IBGE 2918902",
    "291900": "Lajedinho/BA — IBGE 2919009",
    "291905": "Lajedo do Tabocal/BA — IBGE 2919058",
    "291910": "Lamarão/BA — IBGE 2919108",
    "291915": "Lapão/BA — IBGE 2919157",
    "291920": "Lauro de Freitas/BA — IBGE 2919207",
    "291930": "Lençóis/BA — IBGE 2919306",
    "291940": "Licínio de Almeida/BA — IBGE 2919405",
    "291950": "Livramento de Nossa Senhora/BA — IBGE 2919504",
    "291955": "Luís Eduardo Magalhães/BA — IBGE 2919553",
    "291960": "Macajuba/BA — IBGE 2919603",
    "291970": "Macarani/BA — IBGE 2919702",
    "291980": "Macaúbas/BA — IBGE 2919801",
    "291990": "Macururé/BA — IBGE 2919900",
    "291992": "Madre de Deus/BA — IBGE 2919926",
    "291995": "Maetinga/BA — IBGE 2919959",
    "292000": "Maiquinique/BA — IBGE 2920007",
    "292010": "Mairi/BA — IBGE 2920106",
    "292020": "Malhada/BA — IBGE 2920205",
    "292030": "Malhada de Pedras/BA — IBGE 2920304",
    "292040": "Manoel Vitorino/BA — IBGE 2920403",
    "292045": "Mansidão/BA — IBGE 2920452",
    "292050": "Maracás/BA — IBGE 2920502",
    "292060": "Maragogipe/BA — IBGE 2920601",
    "292070": "Maraú/BA — IBGE 2920700",
    "292080": "Marcionílio Souza/BA — IBGE 2920809",
    "292090": "Mascote/BA — IBGE 2920908",
    "292100": "Mata de São João/BA — IBGE 2921005",
    "292105": "Matina/BA — IBGE 2921054",
    "292110": "Medeiros Neto/BA — IBGE 2921104",
    "292120": "Miguel Calmon/BA — IBGE 2921203",
    "292130": "Milagres/BA — IBGE 2921302",
    "292140": "Mirangaba/BA — IBGE 2921401",
    "292145": "Mirante/BA — IBGE 2921450",
    "292150": "Monte Santo/BA — IBGE 2921500",
    "292160": "Morpará/BA — IBGE 2921609",
    "292170": "Morro do Chapéu/BA — IBGE 2921708",
    "292180": "Mortugaba/BA — IBGE 2921807",
    "292190": "Mucugê/BA — IBGE 2921906",
    "292200": "Mucuri/BA — IBGE 2922003",
    "292205": "Mulungu do Morro/BA — IBGE 2922052",
    "292210": "Mundo Novo/BA — IBGE 2922102",
    "292220": "Muniz Ferreira/BA — IBGE 2922201",
    "292225": "Muquém do São Francisco/BA — IBGE 2922250",
    "292230": "Muritiba/BA — IBGE 2922300",
    "292240": "Mutuípe/BA — IBGE 2922409",
    "292250": "Nazaré/BA — IBGE 2922508",
    "292260": "Nilo Peçanha/BA — IBGE 2922607",
    "292265": "Nordestina/BA — IBGE 2922656",
    "292270": "Nova Canaã/BA — IBGE 2922706",
    "292273": "Nova Fátima/BA — IBGE 2922730",
    "292275": "Nova Ibiá/BA — IBGE 2922755",
    "292280": "Nova Itarana/BA — IBGE 2922805",
    "292285": "Nova Redenção/BA — IBGE 2922854",
    "292290": "Nova Soure/BA — IBGE 2922904",
    "292300": "Nova Viçosa/BA — IBGE 2923001",
    "292303": "Novo Horizonte/BA — IBGE 2923035",
    "292305": "Novo Triunfo/BA — IBGE 2923050",
    "292310": "Olindina/BA — IBGE 2923100",
    "292320": "Oliveira dos Brejinhos/BA — IBGE 2923209",
    "292330": "Ouriçangas/BA — IBGE 2923308",
    "292335": "Ourolândia/BA — IBGE 2923357",
    "292340": "Palmas de Monte Alto/BA — IBGE 2923407",
    "292350": "Palmeiras/BA — IBGE 2923506",
    "292360": "Paramirim/BA — IBGE 2923605",
    "292370": "Paratinga/BA — IBGE 2923704",
    "292380": "Paripiranga/BA — IBGE 2923803",
    "292390": "Pau Brasil/BA — IBGE 2923902",
    "292400": "Paulo Afonso/BA — IBGE 2924009",
    "292405": "Pé de Serra/BA — IBGE 2924058",
    "292410": "Pedrão/BA — IBGE 2924108",
    "292420": "Pedro Alexandre/BA — IBGE 2924207",
    "292430": "Piatã/BA — IBGE 2924306",
    "292440": "Pilão Arcado/BA — IBGE 2924405",
    "292450": "Pindaí/BA — IBGE 2924504",
    "292460": "Pindobaçu/BA — IBGE 2924603",
    "292465": "Pintadas/BA — IBGE 2924652",
    "292467": "Piraí do Norte/BA — IBGE 2924678",
    "292470": "Piripá/BA — IBGE 2924702",
    "292480": "Piritiba/BA — IBGE 2924801",
    "292490": "Planaltino/BA — IBGE 2924900",
    "292500": "Planalto/BA — IBGE 2925006",
    "292510": "Poções/BA — IBGE 2925105",
    "292520": "Pojuca/BA — IBGE 2925204",
    "292525": "Ponto Novo/BA — IBGE 2925253",
    "292530": "Porto Seguro/BA — IBGE 2925303",
    "292540": "Potiraguá/BA — IBGE 2925402",
    "292550": "Prado/BA — IBGE 2925501",
    "292560": "Presidente Dutra/BA — IBGE 2925600",
    "292570": "Presidente Jânio Quadros/BA — IBGE 2925709",
    "292575": "Presidente Tancredo Neves/BA — IBGE 2925758",
    "292580": "Queimadas/BA — IBGE 2925808",
    "292590": "Quijingue/BA — IBGE 2925907",
    "292593": "Quixabeira/BA — IBGE 2925931",
    "292595": "Rafael Jambeiro/BA — IBGE 2925956",
    "292600": "Remanso/BA — IBGE 2926004",
    "292610": "Retirolândia/BA — IBGE 2926103",
    "292620": "Riachão das Neves/BA — IBGE 2926202",
    "292630": "Riachão do Jacuípe/BA — IBGE 2926301",
    "292640": "Riacho de Santana/BA — IBGE 2926400",
    "292650": "Ribeira do Amparo/BA — IBGE 2926509",
    "292660": "Ribeira do Pombal/BA — IBGE 2926608",
    "292665": "Ribeirão do Largo/BA — IBGE 2926657",
    "292670": "Rio de Contas/BA — IBGE 2926707",
    "292680": "Rio do Antônio/BA — IBGE 2926806",
    "292690": "Rio do Pires/BA — IBGE 2926905",
    "292700": "Rio Real/BA — IBGE 2927002",
    "292710": "Rodelas/BA — IBGE 2927101",
    "292720": "Ruy Barbosa/BA — IBGE 2927200",
    "292730": "Salinas da Margarida/BA — IBGE 2927309",
    "292740": "Salvador/BA — IBGE 2927408",
    "292750": "Santa Bárbara/BA — IBGE 2927507",
    "292760": "Santa Brígida/BA — IBGE 2927606",
    "292770": "Santa Cruz Cabrália/BA — IBGE 2927705",
    "292780": "Santa Cruz da Vitória/BA — IBGE 2927804",
    "292790": "Santa Inês/BA — IBGE 2927903",
    "292800": "Santaluz/BA — IBGE 2928000",
    "292805": "Santa Luzia/BA — IBGE 2928059",
    "292810": "Santa Maria da Vitória/BA — IBGE 2928109",
    "292820": "Santana/BA — IBGE 2928208",
    "292830": "Santanópolis/BA — IBGE 2928307",
    "292840": "Santa Rita de Cássia/BA — IBGE 2928406",
    "292850": "Santa Terezinha/BA — IBGE 2928505",
    "292860": "Santo Amaro/BA — IBGE 2928604",
    "292870": "Santo Antônio de Jesus/BA — IBGE 2928703",
    "292880": "Santo Estêvão/BA — IBGE 2928802",
    "292890": "São Desidério/BA — IBGE 2928901",
    "292895": "São Domingos/BA — IBGE 2928950",
    "292900": "São Félix/BA — IBGE 2929008",
    "292905": "São Félix do Coribe/BA — IBGE 2929057",
    "292910": "São Felipe/BA — IBGE 2929107",
    "292920": "São Francisco do Conde/BA — IBGE 2929206",
    "292925": "São Gabriel/BA — IBGE 2929255",
    "292930": "São Gonçalo dos Campos/BA — IBGE 2929305",
    "292935": "São José da Vitória/BA — IBGE 2929354",
    "292937": "São José do Jacuípe/BA — IBGE 2929370",
    "292940": "São Miguel das Matas/BA — IBGE 2929404",
    "292950": "São Sebastião do Passé/BA — IBGE 2929503",
    "292960": "Sapeaçu/BA — IBGE 2929602",
    "292970": "Sátiro Dias/BA — IBGE 2929701",
    "292975": "Saubara/BA — IBGE 2929750",
    "292980": "Saúde/BA — IBGE 2929800",
    "292990": "Seabra/BA — IBGE 2929909",
    "293000": "Sebastião Laranjeiras/BA — IBGE 2930006",
    "293010": "Senhor do Bonfim/BA — IBGE 2930105",
    "293015": "Serra do Ramalho/BA — IBGE 2930154",
    "293020": "Sento Sé/BA — IBGE 2930204",
    "293030": "Serra Dourada/BA — IBGE 2930303",
    "293040": "Serra Preta/BA — IBGE 2930402",
    "293050": "Serrinha/BA — IBGE 2930501",
    "293060": "Serrolândia/BA — IBGE 2930600",
    "293070": "Simões Filho/BA — IBGE 2930709",
    "293075": "Sítio do Mato/BA — IBGE 2930758",
    "293076": "Sítio do Quinto/BA — IBGE 2930766",
    "293077": "Sobradinho/BA — IBGE 2930774",
    "293080": "Souto Soares/BA — IBGE 2930808",
    "293090": "Tabocas do Brejo Velho/BA — IBGE 2930907",
    "293100": "Tanhaçu/BA — IBGE 2931004",
    "293105": "Tanque Novo/BA — IBGE 2931053",
    "293110": "Tanquinho/BA — IBGE 2931103",
    "293120": "Taperoá/BA — IBGE 2931202",
    "293130": "Tapiramutá/BA — IBGE 2931301",
    "293135": "Teixeira de Freitas/BA — IBGE 2931350",
    "293140": "Teodoro Sampaio/BA — IBGE 2931400",
    "293150": "Teofilândia/BA — IBGE 2931509",
    "293160": "Teolândia/BA — IBGE 2931608",
    "293170": "Terra Nova/BA — IBGE 2931707",
    "293180": "Tremedal/BA — IBGE 2931806",
    "293190": "Tucano/BA — IBGE 2931905",
    "293200": "Uauá/BA — IBGE 2932002",
    "293210": "Ubaíra/BA — IBGE 2932101",
    "293220": "Ubaitaba/BA — IBGE 2932200",
    "293230": "Ubatã/BA — IBGE 2932309",
    "293240": "Uibaí/BA — IBGE 2932408",
    "293245": "Umburanas/BA — IBGE 2932457",
    "293250": "Una/BA — IBGE 2932507",
    "293260": "Urandi/BA — IBGE 2932606",
    "293270": "Uruçuca/BA — IBGE 2932705",
    "293280": "Utinga/BA — IBGE 2932804",
    "293290": "Valença/BA — IBGE 2932903",
    "293300": "Valente/BA — IBGE 2933000",
    "293305": "Várzea da Roça/BA — IBGE 2933059",
    "293310": "Várzea do Poço/BA — IBGE 2933109",
    "293315": "Várzea Nova/BA — IBGE 2933158",
    "293317": "Varzedo/BA — IBGE 2933174",
    "293320": "Vera Cruz/BA — IBGE 2933208",
    "293325": "Vereda/BA — IBGE 2933257",
    "293330": "Vitória da Conquista/BA — IBGE 2933307",
    "293340": "Wagner/BA — IBGE 2933406",
    "293345": "Wanderley/BA — IBGE 2933455",
    "293350": "Wenceslau Guimarães/BA — IBGE 2933505",
    "293360": "Xique-Xique/BA — IBGE 2933604",
    "310010": "Abadia dos Dourados/MG — IBGE 3100104",
    "310020": "Abaeté/MG — IBGE 3100203",
    "310030": "Abre Campo/MG — IBGE 3100302",
    "310040": "Acaiaca/MG — IBGE 3100401",
    "310050": "Açucena/MG — IBGE 3100500",
    "310060": "Água Boa/MG — IBGE 3100609",
    "310070": "Água Comprida/MG — IBGE 3100708",
    "310080": "Aguanil/MG — IBGE 3100807",
    "310090": "Águas Formosas/MG — IBGE 3100906",
    "310100": "Águas Vermelhas/MG — IBGE 3101003",
    "310110": "Aimorés/MG — IBGE 3101102",
    "310120": "Aiuruoca/MG — IBGE 3101201",
    "310130": "Alagoa/MG — IBGE 3101300",
    "310140": "Albertina/MG — IBGE 3101409",
    "310150": "Além Paraíba/MG — IBGE 3101508",
    "310160": "Alfenas/MG — IBGE 3101607",
    "310163": "Alfredo Vasconcelos/MG — IBGE 3101631",
    "310170": "Almenara/MG — IBGE 3101706",
    "310180": "Alpercata/MG — IBGE 3101805",
    "310190": "Alpinópolis/MG — IBGE 3101904",
    "310200": "Alterosa/MG — IBGE 3102001",
    "310205": "Alto Caparaó/MG — IBGE 3102050",
    "310210": "Alto Rio Doce/MG — IBGE 3102100",
    "310220": "Alvarenga/MG — IBGE 3102209",
    "310230": "Alvinópolis/MG — IBGE 3102308",
    "310240": "Alvorada de Minas/MG — IBGE 3102407",
    "310250": "Amparo do Serra/MG — IBGE 3102506",
    "310260": "Andradas/MG — IBGE 3102605",
    "310270": "Cachoeira de Pajeú/MG — IBGE 3102704",
    "310280": "Andrelândia/MG — IBGE 3102803",
    "310285": "Angelândia/MG — IBGE 3102852",
    "310290": "Antônio Carlos/MG — IBGE 3102902",
    "310300": "Antônio Dias/MG — IBGE 3103009",
    "310310": "Antônio Prado de Minas/MG — IBGE 3103108",
    "310320": "Araçaí/MG — IBGE 3103207",
    "310330": "Aracitaba/MG — IBGE 3103306",
    "310340": "Araçuaí/MG — IBGE 3103405",
    "310350": "Araguari/MG — IBGE 3103504",
    "310360": "Arantina/MG — IBGE 3103603",
    "310370": "Araponga/MG — IBGE 3103702",
    "310375": "Araporã/MG — IBGE 3103751",
    "310380": "Arapuá/MG — IBGE 3103801",
    "310390": "Araújos/MG — IBGE 3103900",
    "310400": "Araxá/MG — IBGE 3104007",
    "310410": "Arceburgo/MG — IBGE 3104106",
    "310420": "Arcos/MG — IBGE 3104205",
    "310430": "Areado/MG — IBGE 3104304",
    "310440": "Argirita/MG — IBGE 3104403",
    "310445": "Aricanduva/MG — IBGE 3104452",
    "310450": "Arinos/MG — IBGE 3104502",
    "310460": "Astolfo Dutra/MG — IBGE 3104601",
    "310470": "Ataléia/MG — IBGE 3104700",
    "310480": "Augusto de Lima/MG — IBGE 3104809",
    "310490": "Baependi/MG — IBGE 3104908",
    "310500": "Baldim/MG — IBGE 3105004",
    "310510": "Bambuí/MG — IBGE 3105103",
    "310520": "Bandeira/MG — IBGE 3105202",
    "310530": "Bandeira do Sul/MG — IBGE 3105301",
    "310540": "Barão de Cocais/MG — IBGE 3105400",
    "310550": "Barão de Monte Alto/MG — IBGE 3105509",
    "310560": "Barbacena/MG — IBGE 3105608",
    "310570": "Barra Longa/MG — IBGE 3105707",
    "310590": "Barroso/MG — IBGE 3105905",
    "310600": "Bela Vista de Minas/MG — IBGE 3106002",
    "310610": "Belmiro Braga/MG — IBGE 3106101",
    "310620": "Belo Horizonte/MG — IBGE 3106200",
    "310630": "Belo Oriente/MG — IBGE 3106309",
    "310640": "Belo Vale/MG — IBGE 3106408",
    "310650": "Berilo/MG — IBGE 3106507",
    "310660": "Bertópolis/MG — IBGE 3106606",
    "310665": "Berizal/MG — IBGE 3106655",
    "310670": "Betim/MG — IBGE 3106705",
    "310680": "Bias Fortes/MG — IBGE 3106804",
    "310690": "Bicas/MG — IBGE 3106903",
    "310700": "Biquinhas/MG — IBGE 3107000",
    "310710": "Boa Esperança/MG — IBGE 3107109",
    "310720": "Bocaina de Minas/MG — IBGE 3107208",
    "310730": "Bocaiúva/MG — IBGE 3107307",
    "310740": "Bom Despacho/MG — IBGE 3107406",
    "310750": "Bom Jardim de Minas/MG — IBGE 3107505",
    "310760": "Bom Jesus da Penha/MG — IBGE 3107604",
    "310770": "Bom Jesus do Amparo/MG — IBGE 3107703",
    "310780": "Bom Jesus do Galho/MG — IBGE 3107802",
    "310790": "Bom Repouso/MG — IBGE 3107901",
    "310800": "Bom Sucesso/MG — IBGE 3108008",
    "310810": "Bonfim/MG — IBGE 3108107",
    "310820": "Bonfinópolis de Minas/MG — IBGE 3108206",
    "310825": "Bonito de Minas/MG — IBGE 3108255",
    "310830": "Borda da Mata/MG — IBGE 3108305",
    "310840": "Botelhos/MG — IBGE 3108404",
    "310850": "Botumirim/MG — IBGE 3108503",
    "310855": "Brasilândia de Minas/MG — IBGE 3108552",
    "310860": "Brasília de Minas/MG — IBGE 3108602",
    "310870": "Brás Pires/MG — IBGE 3108701",
    "310880": "Braúnas/MG — IBGE 3108800",
    "310890": "Brazópolis/MG — IBGE 3108909",
    "310900": "Brumadinho/MG — IBGE 3109006",
    "310910": "Bueno Brandão/MG — IBGE 3109105",
    "310920": "Buenópolis/MG — IBGE 3109204",
    "310925": "Bugre/MG — IBGE 3109253",
    "310930": "Buritis/MG — IBGE 3109303",
    "310940": "Buritizeiro/MG — IBGE 3109402",
    "310945": "Cabeceira Grande/MG — IBGE 3109451",
    "310950": "Cabo Verde/MG — IBGE 3109501",
    "310960": "Cachoeira da Prata/MG — IBGE 3109600",
    "310970": "Cachoeira de Minas/MG — IBGE 3109709",
    "310980": "Cachoeira Dourada/MG — IBGE 3109808",
    "310990": "Caetanópolis/MG — IBGE 3109907",
    "311000": "Caeté/MG — IBGE 3110004",
    "311010": "Caiana/MG — IBGE 3110103",
    "311020": "Cajuri/MG — IBGE 3110202",
    "311030": "Caldas/MG — IBGE 3110301",
    "311040": "Camacho/MG — IBGE 3110400",
    "311050": "Camanducaia/MG — IBGE 3110509",
    "311060": "Cambuí/MG — IBGE 3110608",
    "311070": "Cambuquira/MG — IBGE 3110707",
    "311080": "Campanário/MG — IBGE 3110806",
    "311090": "Campanha/MG — IBGE 3110905",
    "311100": "Campestre/MG — IBGE 3111002",
    "311110": "Campina Verde/MG — IBGE 3111101",
    "311115": "Campo Azul/MG — IBGE 3111150",
    "311120": "Campo Belo/MG — IBGE 3111200",
    "311130": "Campo do Meio/MG — IBGE 3111309",
    "311140": "Campo Florido/MG — IBGE 3111408",
    "311150": "Campos Altos/MG — IBGE 3111507",
    "311160": "Campos Gerais/MG — IBGE 3111606",
    "311170": "Canaã/MG — IBGE 3111705",
    "311180": "Canápolis/MG — IBGE 3111804",
    "311190": "Cana Verde/MG — IBGE 3111903",
    "311200": "Candeias/MG — IBGE 3112000",
    "311205": "Cantagalo/MG — IBGE 3112059",
    "311210": "Caparaó/MG — IBGE 3112109",
    "311220": "Capela Nova/MG — IBGE 3112208",
    "311230": "Capelinha/MG — IBGE 3112307",
    "311240": "Capetinga/MG — IBGE 3112406",
    "311250": "Capim Branco/MG — IBGE 3112505",
    "311260": "Capinópolis/MG — IBGE 3112604",
    "311265": "Capitão Andrade/MG — IBGE 3112653",
    "311270": "Capitão Enéas/MG — IBGE 3112703",
    "311280": "Capitólio/MG — IBGE 3112802",
    "311290": "Caputira/MG — IBGE 3112901",
    "311300": "Caraí/MG — IBGE 3113008",
    "311310": "Caranaíba/MG — IBGE 3113107",
    "311320": "Carandaí/MG — IBGE 3113206",
    "311330": "Carangola/MG — IBGE 3113305",
    "311340": "Caratinga/MG — IBGE 3113404",
    "311350": "Carbonita/MG — IBGE 3113503",
    "311360": "Careaçu/MG — IBGE 3113602",
    "311370": "Carlos Chagas/MG — IBGE 3113701",
    "311380": "Carmésia/MG — IBGE 3113800",
    "311390": "Carmo da Cachoeira/MG — IBGE 3113909",
    "311400": "Carmo da Mata/MG — IBGE 3114006",
    "311410": "Carmo de Minas/MG — IBGE 3114105",
    "311420": "Carmo do Cajuru/MG — IBGE 3114204",
    "311430": "Carmo do Paranaíba/MG — IBGE 3114303",
    "311440": "Carmo do Rio Claro/MG — IBGE 3114402",
    "311450": "Carmópolis de Minas/MG — IBGE 3114501",
    "311455": "Carneirinho/MG — IBGE 3114550",
    "311460": "Carrancas/MG — IBGE 3114600",
    "311470": "Carvalhópolis/MG — IBGE 3114709",
    "311480": "Carvalhos/MG — IBGE 3114808",
    "311490": "Casa Grande/MG — IBGE 3114907",
    "311500": "Cascalho Rico/MG — IBGE 3115003",
    "311510": "Cássia/MG — IBGE 3115102",
    "311520": "Conceição da Barra de Minas/MG — IBGE 3115201",
    "311530": "Cataguases/MG — IBGE 3115300",
    "311535": "Catas Altas/MG — IBGE 3115359",
    "311540": "Catas Altas da Noruega/MG — IBGE 3115409",
    "311545": "Catuji/MG — IBGE 3115458",
    "311547": "Catuti/MG — IBGE 3115474",
    "311550": "Caxambu/MG — IBGE 3115508",
    "311560": "Cedro do Abaeté/MG — IBGE 3115607",
    "311570": "Central de Minas/MG — IBGE 3115706",
    "311580": "Centralina/MG — IBGE 3115805",
    "311590": "Chácara/MG — IBGE 3115904",
    "311600": "Chalé/MG — IBGE 3116001",
    "311610": "Chapada do Norte/MG — IBGE 3116100",
    "311615": "Chapada Gaúcha/MG — IBGE 3116159",
    "311620": "Chiador/MG — IBGE 3116209",
    "311630": "Cipotânea/MG — IBGE 3116308",
    "311640": "Claraval/MG — IBGE 3116407",
    "311650": "Claro dos Poções/MG — IBGE 3116506",
    "311660": "Cláudio/MG — IBGE 3116605",
    "311670": "Coimbra/MG — IBGE 3116704",
    "311680": "Coluna/MG — IBGE 3116803",
    "311690": "Comendador Gomes/MG — IBGE 3116902",
    "311700": "Comercinho/MG — IBGE 3117009",
    "311710": "Conceição da Aparecida/MG — IBGE 3117108",
    "311720": "Conceição das Pedras/MG — IBGE 3117207",
    "311730": "Conceição das Alagoas/MG — IBGE 3117306",
    "311740": "Conceição de Ipanema/MG — IBGE 3117405",
    "311750": "Conceição do Mato Dentro/MG — IBGE 3117504",
    "311760": "Conceição do Pará/MG — IBGE 3117603",
    "311770": "Conceição do Rio Verde/MG — IBGE 3117702",
    "311780": "Conceição dos Ouros/MG — IBGE 3117801",
    "311783": "Cônego Marinho/MG — IBGE 3117836",
    "311787": "Confins/MG — IBGE 3117876",
    "311790": "Congonhal/MG — IBGE 3117900",
    "311800": "Congonhas/MG — IBGE 3118007",
    "311810": "Congonhas do Norte/MG — IBGE 3118106",
    "311820": "Conquista/MG — IBGE 3118205",
    "311830": "Conselheiro Lafaiete/MG — IBGE 3118304",
    "311840": "Conselheiro Pena/MG — IBGE 3118403",
    "311850": "Consolação/MG — IBGE 3118502",
    "311860": "Contagem/MG — IBGE 3118601",
    "311870": "Coqueiral/MG — IBGE 3118700",
    "311880": "Coração de Jesus/MG — IBGE 3118809",
    "311890": "Cordisburgo/MG — IBGE 3118908",
    "311900": "Cordislândia/MG — IBGE 3119005",
    "311910": "Corinto/MG — IBGE 3119104",
    "311920": "Coroaci/MG — IBGE 3119203",
    "311930": "Coromandel/MG — IBGE 3119302",
    "311940": "Coronel Fabriciano/MG — IBGE 3119401",
    "311950": "Coronel Murta/MG — IBGE 3119500",
    "311960": "Coronel Pacheco/MG — IBGE 3119609",
    "311970": "Coronel Xavier Chaves/MG — IBGE 3119708",
    "311980": "Córrego Danta/MG — IBGE 3119807",
    "311990": "Córrego do Bom Jesus/MG — IBGE 3119906",
    "311995": "Córrego Fundo/MG — IBGE 3119955",
    "312000": "Córrego Novo/MG — IBGE 3120003",
    "312010": "Couto de Magalhães de Minas/MG — IBGE 3120102",
    "312015": "Crisólita/MG — IBGE 3120151",
    "312020": "Cristais/MG — IBGE 3120201",
    "312030": "Cristália/MG — IBGE 3120300",
    "312040": "Cristiano Otoni/MG — IBGE 3120409",
    "312050": "Cristina/MG — IBGE 3120508",
    "312060": "Crucilândia/MG — IBGE 3120607",
    "312070": "Cruzeiro da Fortaleza/MG — IBGE 3120706",
    "312080": "Cruzília/MG — IBGE 3120805",
    "312083": "Cuparaque/MG — IBGE 3120839",
    "312087": "Curral de Dentro/MG — IBGE 3120870",
    "312090": "Curvelo/MG — IBGE 3120904",
    "312100": "Datas/MG — IBGE 3121001",
    "312110": "Delfim Moreira/MG — IBGE 3121100",
    "312120": "Delfinópolis/MG — IBGE 3121209",
    "312125": "Delta/MG — IBGE 3121258",
    "312130": "Descoberto/MG — IBGE 3121308",
    "312140": "Desterro de Entre Rios/MG — IBGE 3121407",
    "312150": "Desterro do Melo/MG — IBGE 3121506",
    "312160": "Diamantina/MG — IBGE 3121605",
    "312170": "Diogo de Vasconcelos/MG — IBGE 3121704",
    "312180": "Dionísio/MG — IBGE 3121803",
    "312190": "Divinésia/MG — IBGE 3121902",
    "312200": "Divino/MG — IBGE 3122009",
    "312210": "Divino das Laranjeiras/MG — IBGE 3122108",
    "312220": "Divinolândia de Minas/MG — IBGE 3122207",
    "312230": "Divinópolis/MG — IBGE 3122306",
    "312235": "Divisa Alegre/MG — IBGE 3122355",
    "312240": "Divisa Nova/MG — IBGE 3122405",
    "312245": "Divisópolis/MG — IBGE 3122454",
    "312247": "Dom Bosco/MG — IBGE 3122470",
    "312250": "Dom Cavati/MG — IBGE 3122504",
    "312260": "Dom Joaquim/MG — IBGE 3122603",
    "312270": "Dom Silvério/MG — IBGE 3122702",
    "312280": "Dom Viçoso/MG — IBGE 3122801",
    "312290": "Dona Euzébia/MG — IBGE 3122900",
    "312300": "Dores de Campos/MG — IBGE 3123007",
    "312310": "Dores de Guanhães/MG — IBGE 3123106",
    "312320": "Dores do Indaiá/MG — IBGE 3123205",
    "312330": "Dores do Turvo/MG — IBGE 3123304",
    "312340": "Doresópolis/MG — IBGE 3123403",
    "312350": "Douradoquara/MG — IBGE 3123502",
    "312352": "Durandé/MG — IBGE 3123528",
    "312360": "Elói Mendes/MG — IBGE 3123601",
    "312370": "Engenheiro Caldas/MG — IBGE 3123700",
    "312380": "Engenheiro Navarro/MG — IBGE 3123809",
    "312385": "Entre Folhas/MG — IBGE 3123858",
    "312390": "Entre Rios de Minas/MG — IBGE 3123908",
    "312400": "Ervália/MG — IBGE 3124005",
    "312410": "Esmeraldas/MG — IBGE 3124104",
    "312420": "Espera Feliz/MG — IBGE 3124203",
    "312430": "Espinosa/MG — IBGE 3124302",
    "312440": "Espírito Santo do Dourado/MG — IBGE 3124401",
    "312450": "Estiva/MG — IBGE 3124500",
    "312460": "Estrela Dalva/MG — IBGE 3124609",
    "312470": "Estrela do Indaiá/MG — IBGE 3124708",
    "312480": "Estrela do Sul/MG — IBGE 3124807",
    "312490": "Eugenópolis/MG — IBGE 3124906",
    "312500": "Ewbank da Câmara/MG — IBGE 3125002",
    "312510": "Extrema/MG — IBGE 3125101",
    "312520": "Fama/MG — IBGE 3125200",
    "312530": "Faria Lemos/MG — IBGE 3125309",
    "312540": "Felício dos Santos/MG — IBGE 3125408",
    "312550": "São Gonçalo do Rio Preto/MG — IBGE 3125507",
    "312560": "Felisburgo/MG — IBGE 3125606",
    "312570": "Felixlândia/MG — IBGE 3125705",
    "312580": "Fernandes Tourinho/MG — IBGE 3125804",
    "312590": "Ferros/MG — IBGE 3125903",
    "312595": "Fervedouro/MG — IBGE 3125952",
    "312600": "Florestal/MG — IBGE 3126000",
    "312610": "Formiga/MG — IBGE 3126109",
    "312620": "Formoso/MG — IBGE 3126208",
    "312630": "Fortaleza de Minas/MG — IBGE 3126307",
    "312640": "Fortuna de Minas/MG — IBGE 3126406",
    "312650": "Francisco Badaró/MG — IBGE 3126505",
    "312660": "Francisco Dumont/MG — IBGE 3126604",
    "312670": "Francisco Sá/MG — IBGE 3126703",
    "312675": "Franciscópolis/MG — IBGE 3126752",
    "312680": "Frei Gaspar/MG — IBGE 3126802",
    "312690": "Frei Inocêncio/MG — IBGE 3126901",
    "312695": "Frei Lagonegro/MG — IBGE 3126950",
    "312700": "Fronteira/MG — IBGE 3127008",
    "312705": "Fronteira dos Vales/MG — IBGE 3127057",
    "312707": "Fruta de Leite/MG — IBGE 3127073",
    "312710": "Frutal/MG — IBGE 3127107",
    "312720": "Funilândia/MG — IBGE 3127206",
    "312730": "Galiléia/MG — IBGE 3127305",
    "312733": "Gameleiras/MG — IBGE 3127339",
    "312735": "Glaucilândia/MG — IBGE 3127354",
    "312737": "Goiabeira/MG — IBGE 3127370",
    "312738": "Goianá/MG — IBGE 3127388",
    "312740": "Gonçalves/MG — IBGE 3127404",
    "312750": "Gonzaga/MG — IBGE 3127503",
    "312760": "Gouveia/MG — IBGE 3127602",
    "312770": "Governador Valadares/MG — IBGE 3127701",
    "312780": "Grão Mogol/MG — IBGE 3127800",
    "312790": "Grupiara/MG — IBGE 3127909",
    "312800": "Guanhães/MG — IBGE 3128006",
    "312810": "Guapé/MG — IBGE 3128105",
    "312820": "Guaraciaba/MG — IBGE 3128204",
    "312825": "Guaraciama/MG — IBGE 3128253",
    "312830": "Guaranésia/MG — IBGE 3128303",
    "312840": "Guarani/MG — IBGE 3128402",
    "312850": "Guarará/MG — IBGE 3128501",
    "312860": "Guarda-Mor/MG — IBGE 3128600",
    "312870": "Guaxupé/MG — IBGE 3128709",
    "312880": "Guidoval/MG — IBGE 3128808",
    "312890": "Guimarânia/MG — IBGE 3128907",
    "312900": "Guiricema/MG — IBGE 3129004",
    "312910": "Gurinhatã/MG — IBGE 3129103",
    "312920": "Heliodora/MG — IBGE 3129202",
    "312930": "Iapu/MG — IBGE 3129301",
    "312940": "Ibertioga/MG — IBGE 3129400",
    "312950": "Ibiá/MG — IBGE 3129509",
    "312960": "Ibiaí/MG — IBGE 3129608",
    "312965": "Ibiracatu/MG — IBGE 3129657",
    "312970": "Ibiraci/MG — IBGE 3129707",
    "312980": "Ibirité/MG — IBGE 3129806",
    "312990": "Ibitiúra de Minas/MG — IBGE 3129905",
    "313000": "Ibituruna/MG — IBGE 3130002",
    "313005": "Icaraí de Minas/MG — IBGE 3130051",
    "313010": "Igarapé/MG — IBGE 3130101",
    "313020": "Igaratinga/MG — IBGE 3130200",
    "313030": "Iguatama/MG — IBGE 3130309",
    "313040": "Ijaci/MG — IBGE 3130408",
    "313050": "Ilicínea/MG — IBGE 3130507",
    "313055": "Imbé de Minas/MG — IBGE 3130556",
    "313060": "Inconfidentes/MG — IBGE 3130606",
    "313065": "Indaiabira/MG — IBGE 3130655",
    "313070": "Indianópolis/MG — IBGE 3130705",
    "313080": "Ingaí/MG — IBGE 3130804",
    "313090": "Inhapim/MG — IBGE 3130903",
    "313100": "Inhaúma/MG — IBGE 3131000",
    "313110": "Inimutaba/MG — IBGE 3131109",
    "313115": "Ipaba/MG — IBGE 3131158",
    "313120": "Ipanema/MG — IBGE 3131208",
    "313130": "Ipatinga/MG — IBGE 3131307",
    "313140": "Ipiaçu/MG — IBGE 3131406",
    "313150": "Ipuiúna/MG — IBGE 3131505",
    "313160": "Iraí de Minas/MG — IBGE 3131604",
    "313170": "Itabira/MG — IBGE 3131703",
    "313180": "Itabirinha/MG — IBGE 3131802",
    "313190": "Itabirito/MG — IBGE 3131901",
    "313200": "Itacambira/MG — IBGE 3132008",
    "313210": "Itacarambi/MG — IBGE 3132107",
    "313220": "Itaguara/MG — IBGE 3132206",
    "313230": "Itaipé/MG — IBGE 3132305",
    "313240": "Itajubá/MG — IBGE 3132404",
    "313250": "Itamarandiba/MG — IBGE 3132503",
    "313260": "Itamarati de Minas/MG — IBGE 3132602",
    "313270": "Itambacuri/MG — IBGE 3132701",
    "313280": "Itambé do Mato Dentro/MG — IBGE 3132800",
    "313290": "Itamogi/MG — IBGE 3132909",
    "313300": "Itamonte/MG — IBGE 3133006",
    "313310": "Itanhandu/MG — IBGE 3133105",
    "313320": "Itanhomi/MG — IBGE 3133204",
    "313330": "Itaobim/MG — IBGE 3133303",
    "313340": "Itapagipe/MG — IBGE 3133402",
    "313350": "Itapecerica/MG — IBGE 3133501",
    "313360": "Itapeva/MG — IBGE 3133600",
    "313370": "Itatiaiuçu/MG — IBGE 3133709",
    "313375": "Itaú de Minas/MG — IBGE 3133758",
    "313380": "Itaúna/MG — IBGE 3133808",
    "313390": "Itaverava/MG — IBGE 3133907",
    "313400": "Itinga/MG — IBGE 3134004",
    "313410": "Itueta/MG — IBGE 3134103",
    "313420": "Ituiutaba/MG — IBGE 3134202",
    "313430": "Itumirim/MG — IBGE 3134301",
    "313440": "Iturama/MG — IBGE 3134400",
    "313450": "Itutinga/MG — IBGE 3134509",
    "313460": "Jaboticatubas/MG — IBGE 3134608",
    "313470": "Jacinto/MG — IBGE 3134707",
    "313480": "Jacuí/MG — IBGE 3134806",
    "313490": "Jacutinga/MG — IBGE 3134905",
    "313500": "Jaguaraçu/MG — IBGE 3135001",
    "313505": "Jaíba/MG — IBGE 3135050",
    "313507": "Jampruca/MG — IBGE 3135076",
    "313510": "Janaúba/MG — IBGE 3135100",
    "313520": "Januária/MG — IBGE 3135209",
    "313530": "Japaraíba/MG — IBGE 3135308",
    "313535": "Japonvar/MG — IBGE 3135357",
    "313540": "Jeceaba/MG — IBGE 3135407",
    "313545": "Jenipapo de Minas/MG — IBGE 3135456",
    "313550": "Jequeri/MG — IBGE 3135506",
    "313560": "Jequitaí/MG — IBGE 3135605",
    "313570": "Jequitibá/MG — IBGE 3135704",
    "313580": "Jequitinhonha/MG — IBGE 3135803",
    "313590": "Jesuânia/MG — IBGE 3135902",
    "313600": "Joaíma/MG — IBGE 3136009",
    "313610": "Joanésia/MG — IBGE 3136108",
    "313620": "João Monlevade/MG — IBGE 3136207",
    "313630": "João Pinheiro/MG — IBGE 3136306",
    "313640": "Joaquim Felício/MG — IBGE 3136405",
    "313650": "Jordânia/MG — IBGE 3136504",
    "313652": "José Gonçalves de Minas/MG — IBGE 3136520",
    "313655": "José Raydan/MG — IBGE 3136553",
    "313657": "Josenópolis/MG — IBGE 3136579",
    "313660": "Nova União/MG — IBGE 3136603",
    "313665": "Juatuba/MG — IBGE 3136652",
    "313670": "Juiz de Fora/MG — IBGE 3136702",
    "313680": "Juramento/MG — IBGE 3136801",
    "313690": "Juruaia/MG — IBGE 3136900",
    "313695": "Juvenília/MG — IBGE 3136959",
    "313700": "Ladainha/MG — IBGE 3137007",
    "313710": "Lagamar/MG — IBGE 3137106",
    "313720": "Lagoa da Prata/MG — IBGE 3137205",
    "313730": "Lagoa dos Patos/MG — IBGE 3137304",
    "313740": "Lagoa Dourada/MG — IBGE 3137403",
    "313750": "Lagoa Formosa/MG — IBGE 3137502",
    "313753": "Lagoa Grande/MG — IBGE 3137536",
    "313760": "Lagoa Santa/MG — IBGE 3137601",
    "313770": "Lajinha/MG — IBGE 3137700",
    "313780": "Lambari/MG — IBGE 3137809",
    "313790": "Lamim/MG — IBGE 3137908",
    "313800": "Laranjal/MG — IBGE 3138005",
    "313810": "Lassance/MG — IBGE 3138104",
    "313820": "Lavras/MG — IBGE 3138203",
    "313830": "Leandro Ferreira/MG — IBGE 3138302",
    "313835": "Leme do Prado/MG — IBGE 3138351",
    "313840": "Leopoldina/MG — IBGE 3138401",
    "313850": "Liberdade/MG — IBGE 3138500",
    "313860": "Lima Duarte/MG — IBGE 3138609",
    "313862": "Limeira do Oeste/MG — IBGE 3138625",
    "313865": "Lontra/MG — IBGE 3138658",
    "313867": "Luisburgo/MG — IBGE 3138674",
    "313868": "Luislândia/MG — IBGE 3138682",
    "313870": "Luminárias/MG — IBGE 3138708",
    "313880": "Luz/MG — IBGE 3138807",
    "313890": "Machacalis/MG — IBGE 3138906",
    "313900": "Machado/MG — IBGE 3139003",
    "313910": "Madre de Deus de Minas/MG — IBGE 3139102",
    "313920": "Malacacheta/MG — IBGE 3139201",
    "313925": "Mamonas/MG — IBGE 3139250",
    "313930": "Manga/MG — IBGE 3139300",
    "313940": "Manhuaçu/MG — IBGE 3139409",
    "313950": "Manhumirim/MG — IBGE 3139508",
    "313960": "Mantena/MG — IBGE 3139607",
    "313970": "Maravilhas/MG — IBGE 3139706",
    "313980": "Mar de Espanha/MG — IBGE 3139805",
    "313990": "Maria da Fé/MG — IBGE 3139904",
    "314000": "Mariana/MG — IBGE 3140001",
    "314010": "Marilac/MG — IBGE 3140100",
    "314015": "Mário Campos/MG — IBGE 3140159",
    "314020": "Maripá de Minas/MG — IBGE 3140209",
    "314030": "Marliéria/MG — IBGE 3140308",
    "314040": "Marmelópolis/MG — IBGE 3140407",
    "314050": "Martinho Campos/MG — IBGE 3140506",
    "314053": "Martins Soares/MG — IBGE 3140530",
    "314055": "Mata Verde/MG — IBGE 3140555",
    "314060": "Materlândia/MG — IBGE 3140605",
    "314070": "Mateus Leme/MG — IBGE 3140704",
    "314080": "Matias Barbosa/MG — IBGE 3140803",
    "314085": "Matias Cardoso/MG — IBGE 3140852",
    "314090": "Matipó/MG — IBGE 3140902",
    "314100": "Mato Verde/MG — IBGE 3141009",
    "314110": "Matozinhos/MG — IBGE 3141108",
    "314120": "Matutina/MG — IBGE 3141207",
    "314130": "Medeiros/MG — IBGE 3141306",
    "314140": "Medina/MG — IBGE 3141405",
    "314150": "Mendes Pimentel/MG — IBGE 3141504",
    "314160": "Mercês/MG — IBGE 3141603",
    "314170": "Mesquita/MG — IBGE 3141702",
    "314180": "Minas Novas/MG — IBGE 3141801",
    "314190": "Minduri/MG — IBGE 3141900",
    "314200": "Mirabela/MG — IBGE 3142007",
    "314210": "Miradouro/MG — IBGE 3142106",
    "314220": "Miraí/MG — IBGE 3142205",
    "314225": "Miravânia/MG — IBGE 3142254",
    "314230": "Moeda/MG — IBGE 3142304",
    "314240": "Moema/MG — IBGE 3142403",
    "314250": "Monjolos/MG — IBGE 3142502",
    "314260": "Monsenhor Paulo/MG — IBGE 3142601",
    "314270": "Montalvânia/MG — IBGE 3142700",
    "314280": "Monte Alegre de Minas/MG — IBGE 3142809",
    "314290": "Monte Azul/MG — IBGE 3142908",
    "314300": "Monte Belo/MG — IBGE 3143005",
    "314310": "Monte Carmelo/MG — IBGE 3143104",
    "314315": "Monte Formoso/MG — IBGE 3143153",
    "314320": "Monte Santo de Minas/MG — IBGE 3143203",
    "314330": "Montes Claros/MG — IBGE 3143302",
    "314340": "Monte Sião/MG — IBGE 3143401",
    "314345": "Montezuma/MG — IBGE 3143450",
    "314350": "Morada Nova de Minas/MG — IBGE 3143500",
    "314360": "Morro da Garça/MG — IBGE 3143609",
    "314370": "Morro do Pilar/MG — IBGE 3143708",
    "314380": "Munhoz/MG — IBGE 3143807",
    "314390": "Muriaé/MG — IBGE 3143906",
    "314400": "Mutum/MG — IBGE 3144003",
    "314410": "Muzambinho/MG — IBGE 3144102",
    "314420": "Nacip Raydan/MG — IBGE 3144201",
    "314430": "Nanuque/MG — IBGE 3144300",
    "314435": "Naque/MG — IBGE 3144359",
    "314437": "Natalândia/MG — IBGE 3144375",
    "314440": "Natércia/MG — IBGE 3144409",
    "314450": "Nazareno/MG — IBGE 3144508",
    "314460": "Nepomuceno/MG — IBGE 3144607",
    "314465": "Ninheira/MG — IBGE 3144656",
    "314467": "Nova Belém/MG — IBGE 3144672",
    "314470": "Nova Era/MG — IBGE 3144706",
    "314480": "Nova Lima/MG — IBGE 3144805",
    "314490": "Nova Módica/MG — IBGE 3144904",
    "314500": "Nova Ponte/MG — IBGE 3145000",
    "314505": "Nova Porteirinha/MG — IBGE 3145059",
    "314510": "Nova Resende/MG — IBGE 3145109",
    "314520": "Nova Serrana/MG — IBGE 3145208",
    "314530": "Novo Cruzeiro/MG — IBGE 3145307",
    "314535": "Novo Oriente de Minas/MG — IBGE 3145356",
    "314537": "Novorizonte/MG — IBGE 3145372",
    "314540": "Olaria/MG — IBGE 3145406",
    "314545": "Olhos-d'Água/MG — IBGE 3145455",
    "314550": "Olímpio Noronha/MG — IBGE 3145505",
    "314560": "Oliveira/MG — IBGE 3145604",
    "314570": "Oliveira Fortes/MG — IBGE 3145703",
    "314580": "Onça de Pitangui/MG — IBGE 3145802",
    "314585": "Oratórios/MG — IBGE 3145851",
    "314587": "Orizânia/MG — IBGE 3145877",
    "314590": "Ouro Branco/MG — IBGE 3145901",
    "314600": "Ouro Fino/MG — IBGE 3146008",
    "314610": "Ouro Preto/MG — IBGE 3146107",
    "314620": "Ouro Verde de Minas/MG — IBGE 3146206",
    "314625": "Padre Carvalho/MG — IBGE 3146255",
    "314630": "Padre Paraíso/MG — IBGE 3146305",
    "314640": "Paineiras/MG — IBGE 3146404",
    "314650": "Pains/MG — IBGE 3146503",
    "314655": "Pai Pedro/MG — IBGE 3146552",
    "314660": "Paiva/MG — IBGE 3146602",
    "314670": "Palma/MG — IBGE 3146701",
    "314675": "Palmópolis/MG — IBGE 3146750",
    "314690": "Papagaios/MG — IBGE 3146909",
    "314700": "Paracatu/MG — IBGE 3147006",
    "314710": "Pará de Minas/MG — IBGE 3147105",
    "314720": "Paraguaçu/MG — IBGE 3147204",
    "314730": "Paraisópolis/MG — IBGE 3147303",
    "314740": "Paraopeba/MG — IBGE 3147402",
    "314750": "Passabém/MG — IBGE 3147501",
    "314760": "Passa Quatro/MG — IBGE 3147600",
    "314770": "Passa Tempo/MG — IBGE 3147709",
    "314780": "Passa Vinte/MG — IBGE 3147808",
    "314790": "Passos/MG — IBGE 3147907",
    "314795": "Patis/MG — IBGE 3147956",
    "314800": "Patos de Minas/MG — IBGE 3148004",
    "314810": "Patrocínio/MG — IBGE 3148103",
    "314820": "Patrocínio do Muriaé/MG — IBGE 3148202",
    "314830": "Paula Cândido/MG — IBGE 3148301",
    "314840": "Paulistas/MG — IBGE 3148400",
    "314850": "Pavão/MG — IBGE 3148509",
    "314860": "Peçanha/MG — IBGE 3148608",
    "314870": "Pedra Azul/MG — IBGE 3148707",
    "314875": "Pedra Bonita/MG — IBGE 3148756",
    "314880": "Pedra do Anta/MG — IBGE 3148806",
    "314890": "Pedra do Indaiá/MG — IBGE 3148905",
    "314900": "Pedra Dourada/MG — IBGE 3149002",
    "314910": "Pedralva/MG — IBGE 3149101",
    "314915": "Pedras de Maria da Cruz/MG — IBGE 3149150",
    "314920": "Pedrinópolis/MG — IBGE 3149200",
    "314930": "Pedro Leopoldo/MG — IBGE 3149309",
    "314940": "Pedro Teixeira/MG — IBGE 3149408",
    "314950": "Pequeri/MG — IBGE 3149507",
    "314960": "Pequi/MG — IBGE 3149606",
    "314970": "Perdigão/MG — IBGE 3149705",
    "314980": "Perdizes/MG — IBGE 3149804",
    "314990": "Perdões/MG — IBGE 3149903",
    "314995": "Periquito/MG — IBGE 3149952",
    "315000": "Pescador/MG — IBGE 3150000",
    "315010": "Piau/MG — IBGE 3150109",
    "315015": "Piedade de Caratinga/MG — IBGE 3150158",
    "315020": "Piedade de Ponte Nova/MG — IBGE 3150208",
    "315030": "Piedade do Rio Grande/MG — IBGE 3150307",
    "315040": "Piedade dos Gerais/MG — IBGE 3150406",
    "315050": "Pimenta/MG — IBGE 3150505",
    "315053": "Pingo-d'Água/MG — IBGE 3150539",
    "315057": "Pintópolis/MG — IBGE 3150570",
    "315060": "Piracema/MG — IBGE 3150604",
    "315070": "Pirajuba/MG — IBGE 3150703",
    "315080": "Piranga/MG — IBGE 3150802",
    "315090": "Piranguçu/MG — IBGE 3150901",
    "315100": "Piranguinho/MG — IBGE 3151008",
    "315110": "Pirapetinga/MG — IBGE 3151107",
    "315120": "Pirapora/MG — IBGE 3151206",
    "315130": "Piraúba/MG — IBGE 3151305",
    "315140": "Pitangui/MG — IBGE 3151404",
    "315150": "Piumhi/MG — IBGE 3151503",
    "315160": "Planura/MG — IBGE 3151602",
    "315170": "Poço Fundo/MG — IBGE 3151701",
    "315180": "Poços de Caldas/MG — IBGE 3151800",
    "315190": "Pocrane/MG — IBGE 3151909",
    "315200": "Pompéu/MG — IBGE 3152006",
    "315210": "Ponte Nova/MG — IBGE 3152105",
    "315213": "Ponto Chique/MG — IBGE 3152131",
    "315217": "Ponto dos Volantes/MG — IBGE 3152170",
    "315220": "Porteirinha/MG — IBGE 3152204",
    "315230": "Porto Firme/MG — IBGE 3152303",
    "315240": "Poté/MG — IBGE 3152402",
    "315250": "Pouso Alegre/MG — IBGE 3152501",
    "315260": "Pouso Alto/MG — IBGE 3152600",
    "315270": "Prados/MG — IBGE 3152709",
    "315280": "Prata/MG — IBGE 3152808",
    "315290": "Pratápolis/MG — IBGE 3152907",
    "315300": "Pratinha/MG — IBGE 3153004",
    "315310": "Presidente Bernardes/MG — IBGE 3153103",
    "315320": "Presidente Juscelino/MG — IBGE 3153202",
    "315330": "Presidente Kubitschek/MG — IBGE 3153301",
    "315340": "Presidente Olegário/MG — IBGE 3153400",
    "315350": "Alto Jequitibá/MG — IBGE 3153509",
    "315360": "Prudente de Morais/MG — IBGE 3153608",
    "315370": "Quartel Geral/MG — IBGE 3153707",
    "315380": "Queluzito/MG — IBGE 3153806",
    "315390": "Raposos/MG — IBGE 3153905",
    "315400": "Raul Soares/MG — IBGE 3154002",
    "315410": "Recreio/MG — IBGE 3154101",
    "315415": "Reduto/MG — IBGE 3154150",
    "315420": "Resende Costa/MG — IBGE 3154200",
    "315430": "Resplendor/MG — IBGE 3154309",
    "315440": "Ressaquinha/MG — IBGE 3154408",
    "315445": "Riachinho/MG — IBGE 3154457",
    "315450": "Riacho dos Machados/MG — IBGE 3154507",
    "315460": "Ribeirão das Neves/MG — IBGE 3154606",
    "315470": "Ribeirão Vermelho/MG — IBGE 3154705",
    "315480": "Rio Acima/MG — IBGE 3154804",
    "315490": "Rio Casca/MG — IBGE 3154903",
    "315500": "Rio Doce/MG — IBGE 3155009",
    "315510": "Rio do Prado/MG — IBGE 3155108",
    "315520": "Rio Espera/MG — IBGE 3155207",
    "315530": "Rio Manso/MG — IBGE 3155306",
    "315540": "Rio Novo/MG — IBGE 3155405",
    "315550": "Rio Paranaíba/MG — IBGE 3155504",
    "315560": "Rio Pardo de Minas/MG — IBGE 3155603",
    "315570": "Rio Piracicaba/MG — IBGE 3155702",
    "315580": "Rio Pomba/MG — IBGE 3155801",
    "315590": "Rio Preto/MG — IBGE 3155900",
    "315600": "Rio Vermelho/MG — IBGE 3156007",
    "315610": "Ritápolis/MG — IBGE 3156106",
    "315620": "Rochedo de Minas/MG — IBGE 3156205",
    "315630": "Rodeiro/MG — IBGE 3156304",
    "315640": "Romaria/MG — IBGE 3156403",
    "315645": "Rosário da Limeira/MG — IBGE 3156452",
    "315650": "Rubelita/MG — IBGE 3156502",
    "315660": "Rubim/MG — IBGE 3156601",
    "315670": "Sabará/MG — IBGE 3156700",
    "315680": "Sabinópolis/MG — IBGE 3156809",
    "315690": "Sacramento/MG — IBGE 3156908",
    "315700": "Salinas/MG — IBGE 3157005",
    "315710": "Salto da Divisa/MG — IBGE 3157104",
    "315720": "Santa Bárbara/MG — IBGE 3157203",
    "315725": "Santa Bárbara do Leste/MG — IBGE 3157252",
    "315727": "Santa Bárbara do Monte Verde/MG — IBGE 3157278",
    "315730": "Santa Bárbara do Tugúrio/MG — IBGE 3157302",
    "315733": "Santa Cruz de Minas/MG — IBGE 3157336",
    "315737": "Santa Cruz de Salinas/MG — IBGE 3157377",
    "315740": "Santa Cruz do Escalvado/MG — IBGE 3157401",
    "315750": "Santa Efigênia de Minas/MG — IBGE 3157500",
    "315760": "Santa Fé de Minas/MG — IBGE 3157609",
    "315765": "Santa Helena de Minas/MG — IBGE 3157658",
    "315770": "Santa Juliana/MG — IBGE 3157708",
    "315780": "Santa Luzia/MG — IBGE 3157807",
    "315790": "Santa Margarida/MG — IBGE 3157906",
    "315800": "Santa Maria de Itabira/MG — IBGE 3158003",
    "315810": "Santa Maria do Salto/MG — IBGE 3158102",
    "315820": "Santa Maria do Suaçuí/MG — IBGE 3158201",
    "315830": "Santana da Vargem/MG — IBGE 3158300",
    "315840": "Santana de Cataguases/MG — IBGE 3158409",
    "315850": "Santana de Pirapama/MG — IBGE 3158508",
    "315860": "Santana do Deserto/MG — IBGE 3158607",
    "315870": "Santana do Garambéu/MG — IBGE 3158706",
    "315880": "Santana do Jacaré/MG — IBGE 3158805",
    "315890": "Santana do Manhuaçu/MG — IBGE 3158904",
    "315895": "Santana do Paraíso/MG — IBGE 3158953",
    "315900": "Santana do Riacho/MG — IBGE 3159001",
    "315910": "Santana dos Montes/MG — IBGE 3159100",
    "315920": "Santa Rita de Caldas/MG — IBGE 3159209",
    "315930": "Santa Rita de Jacutinga/MG — IBGE 3159308",
    "315935": "Santa Rita de Minas/MG — IBGE 3159357",
    "315940": "Santa Rita de Ibitipoca/MG — IBGE 3159407",
    "315950": "Santa Rita do Itueto/MG — IBGE 3159506",
    "315960": "Santa Rita do Sapucaí/MG — IBGE 3159605",
    "315970": "Santa Rosa da Serra/MG — IBGE 3159704",
    "315980": "Santa Vitória/MG — IBGE 3159803",
    "315990": "Santo Antônio do Amparo/MG — IBGE 3159902",
    "316000": "Santo Antônio do Aventureiro/MG — IBGE 3160009",
    "316010": "Santo Antônio do Grama/MG — IBGE 3160108",
    "316020": "Santo Antônio do Itambé/MG — IBGE 3160207",
    "316030": "Santo Antônio do Jacinto/MG — IBGE 3160306",
    "316040": "Santo Antônio do Monte/MG — IBGE 3160405",
    "316045": "Santo Antônio do Retiro/MG — IBGE 3160454",
    "316050": "Santo Antônio do Rio Abaixo/MG — IBGE 3160504",
    "316060": "Santo Hipólito/MG — IBGE 3160603",
    "316070": "Santos Dumont/MG — IBGE 3160702",
    "316080": "São Bento Abade/MG — IBGE 3160801",
    "316090": "São Brás do Suaçuí/MG — IBGE 3160900",
    "316095": "São Domingos das Dores/MG — IBGE 3160959",
    "316100": "São Domingos do Prata/MG — IBGE 3161007",
    "316105": "São Félix de Minas/MG — IBGE 3161056",
    "316110": "São Francisco/MG — IBGE 3161106",
    "316120": "São Francisco de Paula/MG — IBGE 3161205",
    "316130": "São Francisco de Sales/MG — IBGE 3161304",
    "316140": "São Francisco do Glória/MG — IBGE 3161403",
    "316150": "São Geraldo/MG — IBGE 3161502",
    "316160": "São Geraldo da Piedade/MG — IBGE 3161601",
    "316165": "São Geraldo do Baixio/MG — IBGE 3161650",
    "316170": "São Gonçalo do Abaeté/MG — IBGE 3161700",
    "316180": "São Gonçalo do Pará/MG — IBGE 3161809",
    "316190": "São Gonçalo do Rio Abaixo/MG — IBGE 3161908",
    "316200": "São Gonçalo do Sapucaí/MG — IBGE 3162005",
    "316210": "São Gotardo/MG — IBGE 3162104",
    "316220": "São João Batista do Glória/MG — IBGE 3162203",
    "316225": "São João da Lagoa/MG — IBGE 3162252",
    "316230": "São João da Mata/MG — IBGE 3162302",
    "316240": "São João da Ponte/MG — IBGE 3162401",
    "316245": "São João das Missões/MG — IBGE 3162450",
    "316250": "São João del Rei/MG — IBGE 3162500",
    "316255": "São João do Manhuaçu/MG — IBGE 3162559",
    "316257": "São João do Manteninha/MG — IBGE 3162575",
    "316260": "São João do Oriente/MG — IBGE 3162609",
    "316265": "São João do Pacuí/MG — IBGE 3162658",
    "316270": "São João do Paraíso/MG — IBGE 3162708",
    "316280": "São João Evangelista/MG — IBGE 3162807",
    "316290": "São João Nepomuceno/MG — IBGE 3162906",
    "316292": "São Joaquim de Bicas/MG — IBGE 3162922",
    "316294": "São José da Barra/MG — IBGE 3162948",
    "316295": "São José da Lapa/MG — IBGE 3162955",
    "316300": "São José da Safira/MG — IBGE 3163003",
    "316310": "São José da Varginha/MG — IBGE 3163102",
    "316320": "São José do Alegre/MG — IBGE 3163201",
    "316330": "São José do Divino/MG — IBGE 3163300",
    "316340": "São José do Goiabal/MG — IBGE 3163409",
    "316350": "São José do Jacuri/MG — IBGE 3163508",
    "316360": "São José do Mantimento/MG — IBGE 3163607",
    "316370": "São Lourenço/MG — IBGE 3163706",
    "316380": "São Miguel do Anta/MG — IBGE 3163805",
    "316390": "São Pedro da União/MG — IBGE 3163904",
    "316400": "São Pedro dos Ferros/MG — IBGE 3164001",
    "316410": "São Pedro do Suaçuí/MG — IBGE 3164100",
    "316420": "São Romão/MG — IBGE 3164209",
    "316430": "São Roque de Minas/MG — IBGE 3164308",
    "316440": "São Sebastião da Bela Vista/MG — IBGE 3164407",
    "316443": "São Sebastião da Vargem Alegre/MG — IBGE 3164431",
    "316447": "São Sebastião do Anta/MG — IBGE 3164472",
    "316450": "São Sebastião do Maranhão/MG — IBGE 3164506",
    "316460": "São Sebastião do Oeste/MG — IBGE 3164605",
    "316470": "São Sebastião do Paraíso/MG — IBGE 3164704",
    "316480": "São Sebastião do Rio Preto/MG — IBGE 3164803",
    "316490": "São Sebastião do Rio Verde/MG — IBGE 3164902",
    "316500": "São Tiago/MG — IBGE 3165008",
    "316510": "São Tomás de Aquino/MG — IBGE 3165107",
    "316520": "São Tomé das Letras/MG — IBGE 3165206",
    "316530": "São Vicente de Minas/MG — IBGE 3165305",
    "316540": "Sapucaí-Mirim/MG — IBGE 3165404",
    "316550": "Sardoá/MG — IBGE 3165503",
    "316553": "Sarzedo/MG — IBGE 3165537",
    "316555": "Setubinha/MG — IBGE 3165552",
    "316556": "Sem-Peixe/MG — IBGE 3165560",
    "316557": "Senador Amaral/MG — IBGE 3165578",
    "316560": "Senador Cortes/MG — IBGE 3165602",
    "316570": "Senador Firmino/MG — IBGE 3165701",
    "316580": "Senador José Bento/MG — IBGE 3165800",
    "316590": "Senador Modestino Gonçalves/MG — IBGE 3165909",
    "316600": "Senhora de Oliveira/MG — IBGE 3166006",
    "316610": "Senhora do Porto/MG — IBGE 3166105",
    "316620": "Senhora dos Remédios/MG — IBGE 3166204",
    "316630": "Sericita/MG — IBGE 3166303",
    "316640": "Seritinga/MG — IBGE 3166402",
    "316650": "Serra Azul de Minas/MG — IBGE 3166501",
    "316660": "Serra da Saudade/MG — IBGE 3166600",
    "316670": "Serra dos Aimorés/MG — IBGE 3166709",
    "316680": "Serra do Salitre/MG — IBGE 3166808",
    "316690": "Serrania/MG — IBGE 3166907",
    "316695": "Serranópolis de Minas/MG — IBGE 3166956",
    "316700": "Serranos/MG — IBGE 3167004",
    "316710": "Serro/MG — IBGE 3167103",
    "316720": "Sete Lagoas/MG — IBGE 3167202",
    "316730": "Silveirânia/MG — IBGE 3167301",
    "316740": "Silvianópolis/MG — IBGE 3167400",
    "316750": "Simão Pereira/MG — IBGE 3167509",
    "316760": "Simonésia/MG — IBGE 3167608",
    "316770": "Sobrália/MG — IBGE 3167707",
    "316780": "Soledade de Minas/MG — IBGE 3167806",
    "316790": "Tabuleiro/MG — IBGE 3167905",
    "316800": "Taiobeiras/MG — IBGE 3168002",
    "316805": "Taparuba/MG — IBGE 3168051",
    "316810": "Tapira/MG — IBGE 3168101",
    "316820": "Tapiraí/MG — IBGE 3168200",
    "316830": "Taquaraçu de Minas/MG — IBGE 3168309",
    "316840": "Tarumirim/MG — IBGE 3168408",
    "316850": "Teixeiras/MG — IBGE 3168507",
    "316860": "Teófilo Otoni/MG — IBGE 3168606",
    "316870": "Timóteo/MG — IBGE 3168705",
    "316880": "Tiradentes/MG — IBGE 3168804",
    "316890": "Tiros/MG — IBGE 3168903",
    "316900": "Tocantins/MG — IBGE 3169000",
    "316905": "Tocos do Moji/MG — IBGE 3169059",
    "316910": "Toledo/MG — IBGE 3169109",
    "316920": "Tombos/MG — IBGE 3169208",
    "316930": "Três Corações/MG — IBGE 3169307",
    "316935": "Três Marias/MG — IBGE 3169356",
    "316940": "Três Pontas/MG — IBGE 3169406",
    "316950": "Tumiritinga/MG — IBGE 3169505",
    "316960": "Tupaciguara/MG — IBGE 3169604",
    "316970": "Turmalina/MG — IBGE 3169703",
    "316980": "Turvolândia/MG — IBGE 3169802",
    "316990": "Ubá/MG — IBGE 3169901",
    "317000": "Ubaí/MG — IBGE 3170008",
    "317005": "Ubaporanga/MG — IBGE 3170057",
    "317010": "Uberaba/MG — IBGE 3170107",
    "317020": "Uberlândia/MG — IBGE 3170206",
    "317030": "Umburatiba/MG — IBGE 3170305",
    "317040": "Unaí/MG — IBGE 3170404",
    "317043": "União de Minas/MG — IBGE 3170438",
    "317047": "Uruana de Minas/MG — IBGE 3170479",
    "317050": "Urucânia/MG — IBGE 3170503",
    "317052": "Urucuia/MG — IBGE 3170529",
    "317057": "Vargem Alegre/MG — IBGE 3170578",
    "317060": "Vargem Bonita/MG — IBGE 3170602",
    "317065": "Vargem Grande do Rio Pardo/MG — IBGE 3170651",
    "317070": "Varginha/MG — IBGE 3170701",
    "317075": "Varjão de Minas/MG — IBGE 3170750",
    "317080": "Várzea da Palma/MG — IBGE 3170800",
    "317090": "Varzelândia/MG — IBGE 3170909",
    "317100": "Vazante/MG — IBGE 3171006",
    "317103": "Verdelândia/MG — IBGE 3171030",
    "317107": "Veredinha/MG — IBGE 3171071",
    "317110": "Veríssimo/MG — IBGE 3171105",
    "317115": "Vermelho Novo/MG — IBGE 3171154",
    "317120": "Vespasiano/MG — IBGE 3171204",
    "317130": "Viçosa/MG — IBGE 3171303",
    "317140": "Vieiras/MG — IBGE 3171402",
    "317150": "Mathias Lobato/MG — IBGE 3171501",
    "317160": "Virgem da Lapa/MG — IBGE 3171600",
    "317170": "Virgínia/MG — IBGE 3171709",
    "317180": "Virginópolis/MG — IBGE 3171808",
    "317190": "Virgolândia/MG — IBGE 3171907",
    "317200": "Visconde do Rio Branco/MG — IBGE 3172004",
    "317210": "Volta Grande/MG — IBGE 3172103",
    "317220": "Wenceslau Braz/MG — IBGE 3172202",
    "320010": "Afonso Cláudio/ES — IBGE 3200102",
    "320013": "Águia Branca/ES — IBGE 3200136",
    "320016": "Água Doce do Norte/ES — IBGE 3200169",
    "320020": "Alegre/ES — IBGE 3200201",
    "320030": "Alfredo Chaves/ES — IBGE 3200300",
    "320035": "Alto Rio Novo/ES — IBGE 3200359",
    "320040": "Anchieta/ES — IBGE 3200409",
    "320050": "Apiacá/ES — IBGE 3200508",
    "320060": "Aracruz/ES — IBGE 3200607",
    "320070": "Atílio Vivácqua/ES — IBGE 3200706",
    "320080": "Baixo Guandu/ES — IBGE 3200805",
    "320090": "Barra de São Francisco/ES — IBGE 3200904",
    "320100": "Boa Esperança/ES — IBGE 3201001",
    "320110": "Bom Jesus do Norte/ES — IBGE 3201100",
    "320115": "Brejetuba/ES — IBGE 3201159",
    "320120": "Cachoeiro de Itapemirim/ES — IBGE 3201209",
    "320130": "Cariacica/ES — IBGE 3201308",
    "320140": "Castelo/ES — IBGE 3201407",
    "320150": "Colatina/ES — IBGE 3201506",
    "320160": "Conceição da Barra/ES — IBGE 3201605",
    "320170": "Conceição do Castelo/ES — IBGE 3201704",
    "320180": "Divino de São Lourenço/ES — IBGE 3201803",
    "320190": "Domingos Martins/ES — IBGE 3201902",
    "320200": "Dores do Rio Preto/ES — IBGE 3202009",
    "320210": "Ecoporanga/ES — IBGE 3202108",
    "320220": "Fundão/ES — IBGE 3202207",
    "320225": "Governador Lindenberg/ES — IBGE 3202256",
    "320230": "Guaçuí/ES — IBGE 3202306",
    "320240": "Guarapari/ES — IBGE 3202405",
    "320245": "Ibatiba/ES — IBGE 3202454",
    "320250": "Ibiraçu/ES — IBGE 3202504",
    "320255": "Ibitirama/ES — IBGE 3202553",
    "320260": "Iconha/ES — IBGE 3202603",
    "320265": "Irupi/ES — IBGE 3202652",
    "320270": "Itaguaçu/ES — IBGE 3202702",
    "320280": "Itapemirim/ES — IBGE 3202801",
    "320290": "Itarana/ES — IBGE 3202900",
    "320300": "Iúna/ES — IBGE 3203007",
    "320305": "Jaguaré/ES — IBGE 3203056",
    "320310": "Jerônimo Monteiro/ES — IBGE 3203106",
    "320313": "João Neiva/ES — IBGE 3203130",
    "320316": "Laranja da Terra/ES — IBGE 3203163",
    "320320": "Linhares/ES — IBGE 3203205",
    "320330": "Mantenópolis/ES — IBGE 3203304",
    "320332": "Marataízes/ES — IBGE 3203320",
    "320334": "Marechal Floriano/ES — IBGE 3203346",
    "320335": "Marilândia/ES — IBGE 3203353",
    "320340": "Mimoso do Sul/ES — IBGE 3203403",
    "320350": "Montanha/ES — IBGE 3203502",
    "320360": "Mucurici/ES — IBGE 3203601",
    "320370": "Muniz Freire/ES — IBGE 3203700",
    "320380": "Muqui/ES — IBGE 3203809",
    "320390": "Nova Venécia/ES — IBGE 3203908",
    "320400": "Pancas/ES — IBGE 3204005",
    "320405": "Pedro Canário/ES — IBGE 3204054",
    "320410": "Pinheiros/ES — IBGE 3204104",
    "320420": "Piúma/ES — IBGE 3204203",
    "320425": "Ponto Belo/ES — IBGE 3204252",
    "320430": "Presidente Kennedy/ES — IBGE 3204302",
    "320435": "Rio Bananal/ES — IBGE 3204351",
    "320440": "Rio Novo do Sul/ES — IBGE 3204401",
    "320450": "Santa Leopoldina/ES — IBGE 3204500",
    "320455": "Santa Maria de Jetibá/ES — IBGE 3204559",
    "320460": "Santa Teresa/ES — IBGE 3204609",
    "320465": "São Domingos do Norte/ES — IBGE 3204658",
    "320470": "São Gabriel da Palha/ES — IBGE 3204708",
    "320480": "São José do Calçado/ES — IBGE 3204807",
    "320490": "São Mateus/ES — IBGE 3204906",
    "320495": "São Roque do Canaã/ES — IBGE 3204955",
    "320500": "Serra/ES — IBGE 3205002",
    "320501": "Sooretama/ES — IBGE 3205010",
    "320503": "Vargem Alta/ES — IBGE 3205036",
    "320506": "Venda Nova do Imigrante/ES — IBGE 3205069",
    "320510": "Viana/ES — IBGE 3205101",
    "320515": "Vila Pavão/ES — IBGE 3205150",
    "320517": "Vila Valério/ES — IBGE 3205176",
    "320520": "Vila Velha/ES — IBGE 3205200",
    "320530": "Vitória/ES — IBGE 3205309",
    "330010": "Angra dos Reis/RJ — IBGE 3300100",
    "330015": "Aperibé/RJ — IBGE 3300159",
    "330020": "Araruama/RJ — IBGE 3300209",
    "330022": "Areal/RJ — IBGE 3300225",
    "330023": "Armação dos Búzios/RJ — IBGE 3300233",
    "330025": "Arraial do Cabo/RJ — IBGE 3300258",
    "330030": "Barra do Piraí/RJ — IBGE 3300308",
    "330040": "Barra Mansa/RJ — IBGE 3300407",
    "330045": "Belford Roxo/RJ — IBGE 3300456",
    "330050": "Bom Jardim/RJ — IBGE 3300506",
    "330060": "Bom Jesus do Itabapoana/RJ — IBGE 3300605",
    "330070": "Cabo Frio/RJ — IBGE 3300704",
    "330080": "Cachoeiras de Macacu/RJ — IBGE 3300803",
    "330090": "Cambuci/RJ — IBGE 3300902",
    "330093": "Carapebus/RJ — IBGE 3300936",
    "330095": "Comendador Levy Gasparian/RJ — IBGE 3300951",
    "330100": "Campos dos Goytacazes/RJ — IBGE 3301009",
    "330110": "Cantagalo/RJ — IBGE 3301108",
    "330115": "Cardoso Moreira/RJ — IBGE 3301157",
    "330120": "Carmo/RJ — IBGE 3301207",
    "330130": "Casimiro de Abreu/RJ — IBGE 3301306",
    "330140": "Conceição de Macabu/RJ — IBGE 3301405",
    "330150": "Cordeiro/RJ — IBGE 3301504",
    "330160": "Duas Barras/RJ — IBGE 3301603",
    "330170": "Duque de Caxias/RJ — IBGE 3301702",
    "330180": "Engenheiro Paulo de Frontin/RJ — IBGE 3301801",
    "330185": "Guapimirim/RJ — IBGE 3301850",
    "330187": "Iguaba Grande/RJ — IBGE 3301876",
    "330190": "Itaboraí/RJ — IBGE 3301900",
    "330200": "Itaguaí/RJ — IBGE 3302007",
    "330205": "Italva/RJ — IBGE 3302056",
    "330210": "Itaocara/RJ — IBGE 3302106",
    "330220": "Itaperuna/RJ — IBGE 3302205",
    "330225": "Itatiaia/RJ — IBGE 3302254",
    "330227": "Japeri/RJ — IBGE 3302270",
    "330230": "Laje do Muriaé/RJ — IBGE 3302304",
    "330240": "Macaé/RJ — IBGE 3302403",
    "330245": "Macuco/RJ — IBGE 3302452",
    "330250": "Magé/RJ — IBGE 3302502",
    "330260": "Mangaratiba/RJ — IBGE 3302601",
    "330270": "Maricá/RJ — IBGE 3302700",
    "330280": "Mendes/RJ — IBGE 3302809",
    "330285": "Mesquita/RJ — IBGE 3302858",
    "330290": "Miguel Pereira/RJ — IBGE 3302908",
    "330300": "Miracema/RJ — IBGE 3303005",
    "330310": "Natividade/RJ — IBGE 3303104",
    "330320": "Nilópolis/RJ — IBGE 3303203",
    "330330": "Niterói/RJ — IBGE 3303302",
    "330340": "Nova Friburgo/RJ — IBGE 3303401",
    "330350": "Nova Iguaçu/RJ — IBGE 3303500",
    "330360": "Paracambi/RJ — IBGE 3303609",
    "330370": "Paraíba do Sul/RJ — IBGE 3303708",
    "330380": "Paraty/RJ — IBGE 3303807",
    "330385": "Paty do Alferes/RJ — IBGE 3303856",
    "330390": "Petrópolis/RJ — IBGE 3303906",
    "330395": "Pinheiral/RJ — IBGE 3303955",
    "330400": "Piraí/RJ — IBGE 3304003",
    "330410": "Porciúncula/RJ — IBGE 3304102",
    "330411": "Porto Real/RJ — IBGE 3304110",
    "330412": "Quatis/RJ — IBGE 3304128",
    "330414": "Queimados/RJ — IBGE 3304144",
    "330415": "Quissamã/RJ — IBGE 3304151",
    "330420": "Resende/RJ — IBGE 3304201",
    "330430": "Rio Bonito/RJ — IBGE 3304300",
    "330440": "Rio Claro/RJ — IBGE 3304409",
    "330450": "Rio das Flores/RJ — IBGE 3304508",
    "330452": "Rio das Ostras/RJ — IBGE 3304524",
    "330455": "Rio de Janeiro/RJ — IBGE 3304557",
    "330460": "Santa Maria Madalena/RJ — IBGE 3304607",
    "330470": "Santo Antônio de Pádua/RJ — IBGE 3304706",
    "330475": "São Francisco de Itabapoana/RJ — IBGE 3304755",
    "330480": "São Fidélis/RJ — IBGE 3304805",
    "330490": "São Gonçalo/RJ — IBGE 3304904",
    "330500": "São João da Barra/RJ — IBGE 3305000",
    "330510": "São João de Meriti/RJ — IBGE 3305109",
    "330513": "São José de Ubá/RJ — IBGE 3305133",
    "330515": "São José do Vale do Rio Preto/RJ — IBGE 3305158",
    "330520": "São Pedro da Aldeia/RJ — IBGE 3305208",
    "330530": "São Sebastião do Alto/RJ — IBGE 3305307",
    "330540": "Sapucaia/RJ — IBGE 3305406",
    "330550": "Saquarema/RJ — IBGE 3305505",
    "330555": "Seropédica/RJ — IBGE 3305554",
    "330560": "Silva Jardim/RJ — IBGE 3305604",
    "330570": "Sumidouro/RJ — IBGE 3305703",
    "330575": "Tanguá/RJ — IBGE 3305752",
    "330580": "Teresópolis/RJ — IBGE 3305802",
    "330590": "Trajano de Moraes/RJ — IBGE 3305901",
    "330600": "Três Rios/RJ — IBGE 3306008",
    "330610": "Valença/RJ — IBGE 3306107",
    "330615": "Varre-Sai/RJ — IBGE 3306156",
    "330620": "Vassouras/RJ — IBGE 3306206",
    "330630": "Volta Redonda/RJ — IBGE 3306305",
    "350010": "Adamantina/SP — IBGE 3500105",
    "350020": "Adolfo/SP — IBGE 3500204",
    "350030": "Aguaí/SP — IBGE 3500303",
    "350040": "Águas da Prata/SP — IBGE 3500402",
    "350050": "Águas de Lindóia/SP — IBGE 3500501",
    "350055": "Águas de Santa Bárbara/SP — IBGE 3500550",
    "350060": "Águas de São Pedro/SP — IBGE 3500600",
    "350070": "Agudos/SP — IBGE 3500709",
    "350075": "Alambari/SP — IBGE 3500758",
    "350080": "Alfredo Marcondes/SP — IBGE 3500808",
    "350090": "Altair/SP — IBGE 3500907",
    "350100": "Altinópolis/SP — IBGE 3501004",
    "350110": "Alto Alegre/SP — IBGE 3501103",
    "350115": "Alumínio/SP — IBGE 3501152",
    "350120": "Álvares Florence/SP — IBGE 3501202",
    "350130": "Álvares Machado/SP — IBGE 3501301",
    "350140": "Álvaro de Carvalho/SP — IBGE 3501400",
    "350150": "Alvinlândia/SP — IBGE 3501509",
    "350160": "Americana/SP — IBGE 3501608",
    "350170": "Américo Brasiliense/SP — IBGE 3501707",
    "350180": "Américo de Campos/SP — IBGE 3501806",
    "350190": "Amparo/SP — IBGE 3501905",
    "350200": "Analândia/SP — IBGE 3502002",
    "350210": "Andradina/SP — IBGE 3502101",
    "350220": "Angatuba/SP — IBGE 3502200",
    "350230": "Anhembi/SP — IBGE 3502309",
    "350240": "Anhumas/SP — IBGE 3502408",
    "350250": "Aparecida/SP — IBGE 3502507",
    "350260": "Aparecida d'Oeste/SP — IBGE 3502606",
    "350270": "Apiaí/SP — IBGE 3502705",
    "350275": "Araçariguama/SP — IBGE 3502754",
    "350280": "Araçatuba/SP — IBGE 3502804",
    "350290": "Araçoiaba da Serra/SP — IBGE 3502903",
    "350300": "Aramina/SP — IBGE 3503000",
    "350310": "Arandu/SP — IBGE 3503109",
    "350315": "Arapeí/SP — IBGE 3503158",
    "350320": "Araraquara/SP — IBGE 3503208",
    "350330": "Araras/SP — IBGE 3503307",
    "350335": "Arco-Íris/SP — IBGE 3503356",
    "350340": "Arealva/SP — IBGE 3503406",
    "350350": "Areias/SP — IBGE 3503505",
    "350360": "Areiópolis/SP — IBGE 3503604",
    "350370": "Ariranha/SP — IBGE 3503703",
    "350380": "Artur Nogueira/SP — IBGE 3503802",
    "350390": "Arujá/SP — IBGE 3503901",
    "350395": "Aspásia/SP — IBGE 3503950",
    "350400": "Assis/SP — IBGE 3504008",
    "350410": "Atibaia/SP — IBGE 3504107",
    "350420": "Auriflama/SP — IBGE 3504206",
    "350430": "Avaí/SP — IBGE 3504305",
    "350440": "Avanhandava/SP — IBGE 3504404",
    "350450": "Avaré/SP — IBGE 3504503",
    "350460": "Bady Bassitt/SP — IBGE 3504602",
    "350470": "Balbinos/SP — IBGE 3504701",
    "350480": "Bálsamo/SP — IBGE 3504800",
    "350490": "Bananal/SP — IBGE 3504909",
    "350500": "Barão de Antonina/SP — IBGE 3505005",
    "350510": "Barbosa/SP — IBGE 3505104",
    "350520": "Bariri/SP — IBGE 3505203",
    "350530": "Barra Bonita/SP — IBGE 3505302",
    "350535": "Barra do Chapéu/SP — IBGE 3505351",
    "350540": "Barra do Turvo/SP — IBGE 3505401",
    "350550": "Barretos/SP — IBGE 3505500",
    "350560": "Barrinha/SP — IBGE 3505609",
    "350570": "Barueri/SP — IBGE 3505708",
    "350580": "Bastos/SP — IBGE 3505807",
    "350590": "Batatais/SP — IBGE 3505906",
    "350600": "Bauru/SP — IBGE 3506003",
    "350610": "Bebedouro/SP — IBGE 3506102",
    "350620": "Bento de Abreu/SP — IBGE 3506201",
    "350630": "Bernardino de Campos/SP — IBGE 3506300",
    "350635": "Bertioga/SP — IBGE 3506359",
    "350640": "Bilac/SP — IBGE 3506409",
    "350650": "Birigui/SP — IBGE 3506508",
    "350660": "Biritiba Mirim/SP — IBGE 3506607",
    "350670": "Boa Esperança do Sul/SP — IBGE 3506706",
    "350680": "Bocaina/SP — IBGE 3506805",
    "350690": "Bofete/SP — IBGE 3506904",
    "350700": "Boituva/SP — IBGE 3507001",
    "350710": "Bom Jesus dos Perdões/SP — IBGE 3507100",
    "350715": "Bom Sucesso de Itararé/SP — IBGE 3507159",
    "350720": "Borá/SP — IBGE 3507209",
    "350730": "Boracéia/SP — IBGE 3507308",
    "350740": "Borborema/SP — IBGE 3507407",
    "350745": "Borebi/SP — IBGE 3507456",
    "350750": "Botucatu/SP — IBGE 3507506",
    "350760": "Bragança Paulista/SP — IBGE 3507605",
    "350770": "Braúna/SP — IBGE 3507704",
    "350775": "Brejo Alegre/SP — IBGE 3507753",
    "350780": "Brodowski/SP — IBGE 3507803",
    "350790": "Brotas/SP — IBGE 3507902",
    "350800": "Buri/SP — IBGE 3508009",
    "350810": "Buritama/SP — IBGE 3508108",
    "350820": "Buritizal/SP — IBGE 3508207",
    "350830": "Cabrália Paulista/SP — IBGE 3508306",
    "350840": "Cabreúva/SP — IBGE 3508405",
    "350850": "Caçapava/SP — IBGE 3508504",
    "350860": "Cachoeira Paulista/SP — IBGE 3508603",
    "350870": "Caconde/SP — IBGE 3508702",
    "350880": "Cafelândia/SP — IBGE 3508801",
    "350890": "Caiabu/SP — IBGE 3508900",
    "350900": "Caieiras/SP — IBGE 3509007",
    "350910": "Caiuá/SP — IBGE 3509106",
    "350920": "Cajamar/SP — IBGE 3509205",
    "350925": "Cajati/SP — IBGE 3509254",
    "350930": "Cajobi/SP — IBGE 3509304",
    "350940": "Cajuru/SP — IBGE 3509403",
    "350945": "Campina do Monte Alegre/SP — IBGE 3509452",
    "350950": "Campinas/SP — IBGE 3509502",
    "350960": "Campo Limpo Paulista/SP — IBGE 3509601",
    "350970": "Campos do Jordão/SP — IBGE 3509700",
    "350980": "Campos Novos Paulista/SP — IBGE 3509809",
    "350990": "Cananéia/SP — IBGE 3509908",
    "350995": "Canas/SP — IBGE 3509957",
    "351000": "Cândido Mota/SP — IBGE 3510005",
    "351010": "Cândido Rodrigues/SP — IBGE 3510104",
    "351015": "Canitar/SP — IBGE 3510153",
    "351020": "Capão Bonito/SP — IBGE 3510203",
    "351030": "Capela do Alto/SP — IBGE 3510302",
    "351040": "Capivari/SP — IBGE 3510401",
    "351050": "Caraguatatuba/SP — IBGE 3510500",
    "351060": "Carapicuíba/SP — IBGE 3510609",
    "351070": "Cardoso/SP — IBGE 3510708",
    "351080": "Casa Branca/SP — IBGE 3510807",
    "351090": "Cássia dos Coqueiros/SP — IBGE 3510906",
    "351100": "Castilho/SP — IBGE 3511003",
    "351110": "Catanduva/SP — IBGE 3511102",
    "351120": "Catiguá/SP — IBGE 3511201",
    "351130": "Cedral/SP — IBGE 3511300",
    "351140": "Cerqueira César/SP — IBGE 3511409",
    "351150": "Cerquilho/SP — IBGE 3511508",
    "351160": "Cesário Lange/SP — IBGE 3511607",
    "351170": "Charqueada/SP — IBGE 3511706",
    "351190": "Clementina/SP — IBGE 3511904",
    "351200": "Colina/SP — IBGE 3512001",
    "351210": "Colômbia/SP — IBGE 3512100",
    "351220": "Conchal/SP — IBGE 3512209",
    "351230": "Conchas/SP — IBGE 3512308",
    "351240": "Cordeirópolis/SP — IBGE 3512407",
    "351250": "Coroados/SP — IBGE 3512506",
    "351260": "Coronel Macedo/SP — IBGE 3512605",
    "351270": "Corumbataí/SP — IBGE 3512704",
    "351280": "Cosmópolis/SP — IBGE 3512803",
    "351290": "Cosmorama/SP — IBGE 3512902",
    "351300": "Cotia/SP — IBGE 3513009",
    "351310": "Cravinhos/SP — IBGE 3513108",
    "351320": "Cristais Paulista/SP — IBGE 3513207",
    "351330": "Cruzália/SP — IBGE 3513306",
    "351340": "Cruzeiro/SP — IBGE 3513405",
    "351350": "Cubatão/SP — IBGE 3513504",
    "351360": "Cunha/SP — IBGE 3513603",
    "351370": "Descalvado/SP — IBGE 3513702",
    "351380": "Diadema/SP — IBGE 3513801",
    "351385": "Dirce Reis/SP — IBGE 3513850",
    "351390": "Divinolândia/SP — IBGE 3513900",
    "351400": "Dobrada/SP — IBGE 3514007",
    "351410": "Dois Córregos/SP — IBGE 3514106",
    "351420": "Dolcinópolis/SP — IBGE 3514205",
    "351430": "Dourado/SP — IBGE 3514304",
    "351440": "Dracena/SP — IBGE 3514403",
    "351450": "Duartina/SP — IBGE 3514502",
    "351460": "Dumont/SP — IBGE 3514601",
    "351470": "Echaporã/SP — IBGE 3514700",
    "351480": "Eldorado/SP — IBGE 3514809",
    "351490": "Elias Fausto/SP — IBGE 3514908",
    "351492": "Elisiário/SP — IBGE 3514924",
    "351495": "Embaúba/SP — IBGE 3514957",
    "351500": "Embu das Artes/SP — IBGE 3515004",
    "351510": "Embu-Guaçu/SP — IBGE 3515103",
    "351512": "Emilianópolis/SP — IBGE 3515129",
    "351515": "Engenheiro Coelho/SP — IBGE 3515152",
    "351518": "Espírito Santo do Pinhal/SP — IBGE 3515186",
    "351519": "Espírito Santo do Turvo/SP — IBGE 3515194",
    "351520": "Estrela d'Oeste/SP — IBGE 3515202",
    "351530": "Estrela do Norte/SP — IBGE 3515301",
    "351535": "Euclides da Cunha Paulista/SP — IBGE 3515350",
    "351540": "Fartura/SP — IBGE 3515400",
    "351550": "Fernandópolis/SP — IBGE 3515509",
    "351560": "Fernando Prestes/SP — IBGE 3515608",
    "351565": "Fernão/SP — IBGE 3515657",
    "351570": "Ferraz de Vasconcelos/SP — IBGE 3515707",
    "351580": "Flora Rica/SP — IBGE 3515806",
    "351590": "Floreal/SP — IBGE 3515905",
    "351600": "Flórida Paulista/SP — IBGE 3516002",
    "351610": "Florínea/SP — IBGE 3516101",
    "351620": "Franca/SP — IBGE 3516200",
    "351630": "Francisco Morato/SP — IBGE 3516309",
    "351640": "Franco da Rocha/SP — IBGE 3516408",
    "351650": "Gabriel Monteiro/SP — IBGE 3516507",
    "351660": "Gália/SP — IBGE 3516606",
    "351670": "Garça/SP — IBGE 3516705",
    "351680": "Gastão Vidigal/SP — IBGE 3516804",
    "351685": "Gavião Peixoto/SP — IBGE 3516853",
    "351690": "General Salgado/SP — IBGE 3516903",
    "351700": "Getulina/SP — IBGE 3517000",
    "351710": "Glicério/SP — IBGE 3517109",
    "351720": "Guaiçara/SP — IBGE 3517208",
    "351730": "Guaimbê/SP — IBGE 3517307",
    "351740": "Guaíra/SP — IBGE 3517406",
    "351750": "Guapiaçu/SP — IBGE 3517505",
    "351760": "Guapiara/SP — IBGE 3517604",
    "351770": "Guará/SP — IBGE 3517703",
    "351780": "Guaraçaí/SP — IBGE 3517802",
    "351790": "Guaraci/SP — IBGE 3517901",
    "351800": "Guarani d'Oeste/SP — IBGE 3518008",
    "351810": "Guarantã/SP — IBGE 3518107",
    "351820": "Guararapes/SP — IBGE 3518206",
    "351830": "Guararema/SP — IBGE 3518305",
    "351840": "Guaratinguetá/SP — IBGE 3518404",
    "351850": "Guareí/SP — IBGE 3518503",
    "351860": "Guariba/SP — IBGE 3518602",
    "351870": "Guarujá/SP — IBGE 3518701",
    "351880": "Guarulhos/SP — IBGE 3518800",
    "351885": "Guatapará/SP — IBGE 3518859",
    "351890": "Guzolândia/SP — IBGE 3518909",
    "351900": "Herculândia/SP — IBGE 3519006",
    "351905": "Holambra/SP — IBGE 3519055",
    "351907": "Hortolândia/SP — IBGE 3519071",
    "351910": "Iacanga/SP — IBGE 3519105",
    "351920": "Iacri/SP — IBGE 3519204",
    "351925": "Iaras/SP — IBGE 3519253",
    "351930": "Ibaté/SP — IBGE 3519303",
    "351940": "Ibirá/SP — IBGE 3519402",
    "351950": "Ibirarema/SP — IBGE 3519501",
    "351960": "Ibitinga/SP — IBGE 3519600",
    "351970": "Ibiúna/SP — IBGE 3519709",
    "351980": "Icém/SP — IBGE 3519808",
    "351990": "Iepê/SP — IBGE 3519907",
    "352000": "Igaraçu do Tietê/SP — IBGE 3520004",
    "352010": "Igarapava/SP — IBGE 3520103",
    "352020": "Igaratá/SP — IBGE 3520202",
    "352030": "Iguape/SP — IBGE 3520301",
    "352040": "Ilhabela/SP — IBGE 3520400",
    "352042": "Ilha Comprida/SP — IBGE 3520426",
    "352044": "Ilha Solteira/SP — IBGE 3520442",
    "352050": "Indaiatuba/SP — IBGE 3520509",
    "352060": "Indiana/SP — IBGE 3520608",
    "352070": "Indiaporã/SP — IBGE 3520707",
    "352080": "Inúbia Paulista/SP — IBGE 3520806",
    "352090": "Ipaussu/SP — IBGE 3520905",
    "352100": "Iperó/SP — IBGE 3521002",
    "352110": "Ipeúna/SP — IBGE 3521101",
    "352115": "Ipiguá/SP — IBGE 3521150",
    "352120": "Iporanga/SP — IBGE 3521200",
    "352130": "Ipuã/SP — IBGE 3521309",
    "352140": "Iracemápolis/SP — IBGE 3521408",
    "352150": "Irapuã/SP — IBGE 3521507",
    "352160": "Irapuru/SP — IBGE 3521606",
    "352170": "Itaberá/SP — IBGE 3521705",
    "352180": "Itaí/SP — IBGE 3521804",
    "352190": "Itajobi/SP — IBGE 3521903",
    "352200": "Itaju/SP — IBGE 3522000",
    "352210": "Itanhaém/SP — IBGE 3522109",
    "352215": "Itaoca/SP — IBGE 3522158",
    "352220": "Itapecerica da Serra/SP — IBGE 3522208",
    "352230": "Itapetininga/SP — IBGE 3522307",
    "352240": "Itapeva/SP — IBGE 3522406",
    "352250": "Itapevi/SP — IBGE 3522505",
    "352260": "Itapira/SP — IBGE 3522604",
    "352265": "Itapirapuã Paulista/SP — IBGE 3522653",
    "352270": "Itápolis/SP — IBGE 3522703",
    "352280": "Itaporanga/SP — IBGE 3522802",
    "352290": "Itapuí/SP — IBGE 3522901",
    "352300": "Itapura/SP — IBGE 3523008",
    "352310": "Itaquaquecetuba/SP — IBGE 3523107",
    "352320": "Itararé/SP — IBGE 3523206",
    "352330": "Itariri/SP — IBGE 3523305",
    "352340": "Itatiba/SP — IBGE 3523404",
    "352350": "Itatinga/SP — IBGE 3523503",
    "352360": "Itirapina/SP — IBGE 3523602",
    "352370": "Itirapuã/SP — IBGE 3523701",
    "352380": "Itobi/SP — IBGE 3523800",
    "352390": "Itu/SP — IBGE 3523909",
    "352400": "Itupeva/SP — IBGE 3524006",
    "352410": "Ituverava/SP — IBGE 3524105",
    "352420": "Jaborandi/SP — IBGE 3524204",
    "352430": "Jaboticabal/SP — IBGE 3524303",
    "352440": "Jacareí/SP — IBGE 3524402",
    "352450": "Jaci/SP — IBGE 3524501",
    "352460": "Jacupiranga/SP — IBGE 3524600",
    "352470": "Jaguariúna/SP — IBGE 3524709",
    "352480": "Jales/SP — IBGE 3524808",
    "352490": "Jambeiro/SP — IBGE 3524907",
    "352500": "Jandira/SP — IBGE 3525003",
    "352510": "Jardinópolis/SP — IBGE 3525102",
    "352520": "Jarinu/SP — IBGE 3525201",
    "352530": "Jaú/SP — IBGE 3525300",
    "352540": "Jeriquara/SP — IBGE 3525409",
    "352550": "Joanópolis/SP — IBGE 3525508",
    "352560": "João Ramalho/SP — IBGE 3525607",
    "352570": "José Bonifácio/SP — IBGE 3525706",
    "352580": "Júlio Mesquita/SP — IBGE 3525805",
    "352585": "Jumirim/SP — IBGE 3525854",
    "352590": "Jundiaí/SP — IBGE 3525904",
    "352600": "Junqueirópolis/SP — IBGE 3526001",
    "352610": "Juquiá/SP — IBGE 3526100",
    "352620": "Juquitiba/SP — IBGE 3526209",
    "352630": "Lagoinha/SP — IBGE 3526308",
    "352640": "Laranjal Paulista/SP — IBGE 3526407",
    "352650": "Lavínia/SP — IBGE 3526506",
    "352660": "Lavrinhas/SP — IBGE 3526605",
    "352670": "Leme/SP — IBGE 3526704",
    "352680": "Lençóis Paulista/SP — IBGE 3526803",
    "352690": "Limeira/SP — IBGE 3526902",
    "352700": "Lindóia/SP — IBGE 3527009",
    "352710": "Lins/SP — IBGE 3527108",
    "352720": "Lorena/SP — IBGE 3527207",
    "352725": "Lourdes/SP — IBGE 3527256",
    "352730": "Louveira/SP — IBGE 3527306",
    "352740": "Lucélia/SP — IBGE 3527405",
    "352750": "Lucianópolis/SP — IBGE 3527504",
    "352760": "Luís Antônio/SP — IBGE 3527603",
    "352770": "Luiziânia/SP — IBGE 3527702",
    "352780": "Lupércio/SP — IBGE 3527801",
    "352790": "Lutécia/SP — IBGE 3527900",
    "352800": "Macatuba/SP — IBGE 3528007",
    "352810": "Macaubal/SP — IBGE 3528106",
    "352820": "Macedônia/SP — IBGE 3528205",
    "352830": "Magda/SP — IBGE 3528304",
    "352840": "Mairinque/SP — IBGE 3528403",
    "352850": "Mairiporã/SP — IBGE 3528502",
    "352860": "Manduri/SP — IBGE 3528601",
    "352870": "Marabá Paulista/SP — IBGE 3528700",
    "352880": "Maracaí/SP — IBGE 3528809",
    "352885": "Marapoama/SP — IBGE 3528858",
    "352890": "Mariápolis/SP — IBGE 3528908",
    "352900": "Marília/SP — IBGE 3529005",
    "352910": "Marinópolis/SP — IBGE 3529104",
    "352920": "Martinópolis/SP — IBGE 3529203",
    "352930": "Matão/SP — IBGE 3529302",
    "352940": "Mauá/SP — IBGE 3529401",
    "352950": "Mendonça/SP — IBGE 3529500",
    "352960": "Meridiano/SP — IBGE 3529609",
    "352965": "Mesópolis/SP — IBGE 3529658",
    "352970": "Miguelópolis/SP — IBGE 3529708",
    "352980": "Mineiros do Tietê/SP — IBGE 3529807",
    "352990": "Miracatu/SP — IBGE 3529906",
    "353000": "Mira Estrela/SP — IBGE 3530003",
    "353010": "Mirandópolis/SP — IBGE 3530102",
    "353020": "Mirante do Paranapanema/SP — IBGE 3530201",
    "353030": "Mirassol/SP — IBGE 3530300",
    "353040": "Mirassolândia/SP — IBGE 3530409",
    "353050": "Mococa/SP — IBGE 3530508",
    "353060": "Mogi das Cruzes/SP — IBGE 3530607",
    "353070": "Mogi Guaçu/SP — IBGE 3530706",
    "353080": "Mogi Mirim/SP — IBGE 3530805",
    "353090": "Mombuca/SP — IBGE 3530904",
    "353100": "Monções/SP — IBGE 3531001",
    "353110": "Mongaguá/SP — IBGE 3531100",
    "353120": "Monte Alegre do Sul/SP — IBGE 3531209",
    "353130": "Monte Alto/SP — IBGE 3531308",
    "353140": "Monte Aprazível/SP — IBGE 3531407",
    "353150": "Monte Azul Paulista/SP — IBGE 3531506",
    "353160": "Monte Castelo/SP — IBGE 3531605",
    "353170": "Monteiro Lobato/SP — IBGE 3531704",
    "353180": "Monte Mor/SP — IBGE 3531803",
    "353190": "Morro Agudo/SP — IBGE 3531902",
    "353200": "Morungaba/SP — IBGE 3532009",
    "353205": "Motuca/SP — IBGE 3532058",
    "353210": "Murutinga do Sul/SP — IBGE 3532108",
    "353215": "Nantes/SP — IBGE 3532157",
    "353220": "Narandiba/SP — IBGE 3532207",
    "353230": "Natividade da Serra/SP — IBGE 3532306",
    "353240": "Nazaré Paulista/SP — IBGE 3532405",
    "353250": "Neves Paulista/SP — IBGE 3532504",
    "353260": "Nhandeara/SP — IBGE 3532603",
    "353270": "Nipoã/SP — IBGE 3532702",
    "353280": "Nova Aliança/SP — IBGE 3532801",
    "353282": "Nova Campina/SP — IBGE 3532827",
    "353284": "Nova Canaã Paulista/SP — IBGE 3532843",
    "353286": "Nova Castilho/SP — IBGE 3532868",
    "353290": "Nova Europa/SP — IBGE 3532900",
    "353300": "Nova Granada/SP — IBGE 3533007",
    "353310": "Nova Guataporanga/SP — IBGE 3533106",
    "353320": "Nova Independência/SP — IBGE 3533205",
    "353325": "Novais/SP — IBGE 3533254",
    "353330": "Nova Luzitânia/SP — IBGE 3533304",
    "353340": "Nova Odessa/SP — IBGE 3533403",
    "353350": "Novo Horizonte/SP — IBGE 3533502",
    "353360": "Nuporanga/SP — IBGE 3533601",
    "353370": "Ocauçu/SP — IBGE 3533700",
    "353380": "Óleo/SP — IBGE 3533809",
    "353390": "Olímpia/SP — IBGE 3533908",
    "353400": "Onda Verde/SP — IBGE 3534005",
    "353410": "Oriente/SP — IBGE 3534104",
    "353420": "Orindiúva/SP — IBGE 3534203",
    "353430": "Orlândia/SP — IBGE 3534302",
    "353440": "Osasco/SP — IBGE 3534401",
    "353450": "Oscar Bressane/SP — IBGE 3534500",
    "353460": "Osvaldo Cruz/SP — IBGE 3534609",
    "353470": "Ourinhos/SP — IBGE 3534708",
    "353475": "Ouroeste/SP — IBGE 3534757",
    "353480": "Ouro Verde/SP — IBGE 3534807",
    "353490": "Pacaembu/SP — IBGE 3534906",
    "353500": "Palestina/SP — IBGE 3535002",
    "353510": "Palmares Paulista/SP — IBGE 3535101",
    "353520": "Palmeira d'Oeste/SP — IBGE 3535200",
    "353530": "Palmital/SP — IBGE 3535309",
    "353540": "Panorama/SP — IBGE 3535408",
    "353550": "Paraguaçu Paulista/SP — IBGE 3535507",
    "353560": "Paraibuna/SP — IBGE 3535606",
    "353570": "Paraíso/SP — IBGE 3535705",
    "353580": "Paranapanema/SP — IBGE 3535804",
    "353590": "Paranapuã/SP — IBGE 3535903",
    "353600": "Parapuã/SP — IBGE 3536000",
    "353610": "Pardinho/SP — IBGE 3536109",
    "353620": "Pariquera-Açu/SP — IBGE 3536208",
    "353625": "Parisi/SP — IBGE 3536257",
    "353630": "Patrocínio Paulista/SP — IBGE 3536307",
    "353640": "Paulicéia/SP — IBGE 3536406",
    "353650": "Paulínia/SP — IBGE 3536505",
    "353657": "Paulistânia/SP — IBGE 3536570",
    "353660": "Paulo de Faria/SP — IBGE 3536604",
    "353670": "Pederneiras/SP — IBGE 3536703",
    "353680": "Pedra Bela/SP — IBGE 3536802",
    "353690": "Pedranópolis/SP — IBGE 3536901",
    "353700": "Pedregulho/SP — IBGE 3537008",
    "353710": "Pedreira/SP — IBGE 3537107",
    "353715": "Pedrinhas Paulista/SP — IBGE 3537156",
    "353720": "Pedro de Toledo/SP — IBGE 3537206",
    "353730": "Penápolis/SP — IBGE 3537305",
    "353740": "Pereira Barreto/SP — IBGE 3537404",
    "353750": "Pereiras/SP — IBGE 3537503",
    "353760": "Peruíbe/SP — IBGE 3537602",
    "353770": "Piacatu/SP — IBGE 3537701",
    "353780": "Piedade/SP — IBGE 3537800",
    "353790": "Pilar do Sul/SP — IBGE 3537909",
    "353800": "Pindamonhangaba/SP — IBGE 3538006",
    "353810": "Pindorama/SP — IBGE 3538105",
    "353820": "Pinhalzinho/SP — IBGE 3538204",
    "353830": "Piquerobi/SP — IBGE 3538303",
    "353850": "Piquete/SP — IBGE 3538501",
    "353860": "Piracaia/SP — IBGE 3538600",
    "353870": "Piracicaba/SP — IBGE 3538709",
    "353880": "Piraju/SP — IBGE 3538808",
    "353890": "Pirajuí/SP — IBGE 3538907",
    "353900": "Pirangi/SP — IBGE 3539004",
    "353910": "Pirapora do Bom Jesus/SP — IBGE 3539103",
    "353920": "Pirapozinho/SP — IBGE 3539202",
    "353930": "Pirassununga/SP — IBGE 3539301",
    "353940": "Piratininga/SP — IBGE 3539400",
    "353950": "Pitangueiras/SP — IBGE 3539509",
    "353960": "Planalto/SP — IBGE 3539608",
    "353970": "Platina/SP — IBGE 3539707",
    "353980": "Poá/SP — IBGE 3539806",
    "353990": "Poloni/SP — IBGE 3539905",
    "354000": "Pompéia/SP — IBGE 3540002",
    "354010": "Pongaí/SP — IBGE 3540101",
    "354020": "Pontal/SP — IBGE 3540200",
    "354025": "Pontalinda/SP — IBGE 3540259",
    "354030": "Pontes Gestal/SP — IBGE 3540309",
    "354040": "Populina/SP — IBGE 3540408",
    "354050": "Porangaba/SP — IBGE 3540507",
    "354060": "Porto Feliz/SP — IBGE 3540606",
    "354070": "Porto Ferreira/SP — IBGE 3540705",
    "354075": "Potim/SP — IBGE 3540754",
    "354080": "Potirendaba/SP — IBGE 3540804",
    "354085": "Pracinha/SP — IBGE 3540853",
    "354090": "Pradópolis/SP — IBGE 3540903",
    "354100": "Praia Grande/SP — IBGE 3541000",
    "354105": "Pratânia/SP — IBGE 3541059",
    "354110": "Presidente Alves/SP — IBGE 3541109",
    "354120": "Presidente Bernardes/SP — IBGE 3541208",
    "354130": "Presidente Epitácio/SP — IBGE 3541307",
    "354140": "Presidente Prudente/SP — IBGE 3541406",
    "354150": "Presidente Venceslau/SP — IBGE 3541505",
    "354160": "Promissão/SP — IBGE 3541604",
    "354165": "Quadra/SP — IBGE 3541653",
    "354170": "Quatá/SP — IBGE 3541703",
    "354180": "Queiroz/SP — IBGE 3541802",
    "354190": "Queluz/SP — IBGE 3541901",
    "354200": "Quintana/SP — IBGE 3542008",
    "354210": "Rafard/SP — IBGE 3542107",
    "354220": "Rancharia/SP — IBGE 3542206",
    "354230": "Redenção da Serra/SP — IBGE 3542305",
    "354240": "Regente Feijó/SP — IBGE 3542404",
    "354250": "Reginópolis/SP — IBGE 3542503",
    "354260": "Registro/SP — IBGE 3542602",
    "354270": "Restinga/SP — IBGE 3542701",
    "354280": "Ribeira/SP — IBGE 3542800",
    "354290": "Ribeirão Bonito/SP — IBGE 3542909",
    "354300": "Ribeirão Branco/SP — IBGE 3543006",
    "354310": "Ribeirão Corrente/SP — IBGE 3543105",
    "354320": "Ribeirão do Sul/SP — IBGE 3543204",
    "354323": "Ribeirão dos Índios/SP — IBGE 3543238",
    "354325": "Ribeirão Grande/SP — IBGE 3543253",
    "354330": "Ribeirão Pires/SP — IBGE 3543303",
    "354340": "Ribeirão Preto/SP — IBGE 3543402",
    "354350": "Riversul/SP — IBGE 3543501",
    "354360": "Rifaina/SP — IBGE 3543600",
    "354370": "Rincão/SP — IBGE 3543709",
    "354380": "Rinópolis/SP — IBGE 3543808",
    "354390": "Rio Claro/SP — IBGE 3543907",
    "354400": "Rio das Pedras/SP — IBGE 3544004",
    "354410": "Rio Grande da Serra/SP — IBGE 3544103",
    "354420": "Riolândia/SP — IBGE 3544202",
    "354425": "Rosana/SP — IBGE 3544251",
    "354430": "Roseira/SP — IBGE 3544301",
    "354440": "Rubiácea/SP — IBGE 3544400",
    "354450": "Rubinéia/SP — IBGE 3544509",
    "354460": "Sabino/SP — IBGE 3544608",
    "354470": "Sagres/SP — IBGE 3544707",
    "354480": "Sales/SP — IBGE 3544806",
    "354490": "Sales Oliveira/SP — IBGE 3544905",
    "354500": "Salesópolis/SP — IBGE 3545001",
    "354510": "Salmourão/SP — IBGE 3545100",
    "354515": "Saltinho/SP — IBGE 3545159",
    "354520": "Salto/SP — IBGE 3545209",
    "354530": "Salto de Pirapora/SP — IBGE 3545308",
    "354540": "Salto Grande/SP — IBGE 3545407",
    "354550": "Sandovalina/SP — IBGE 3545506",
    "354560": "Santa Adélia/SP — IBGE 3545605",
    "354570": "Santa Albertina/SP — IBGE 3545704",
    "354580": "Santa Bárbara d'Oeste/SP — IBGE 3545803",
    "354600": "Santa Branca/SP — IBGE 3546009",
    "354610": "Santa Clara d'Oeste/SP — IBGE 3546108",
    "354620": "Santa Cruz da Conceição/SP — IBGE 3546207",
    "354625": "Santa Cruz da Esperança/SP — IBGE 3546256",
    "354630": "Santa Cruz das Palmeiras/SP — IBGE 3546306",
    "354640": "Santa Cruz do Rio Pardo/SP — IBGE 3546405",
    "354650": "Santa Ernestina/SP — IBGE 3546504",
    "354660": "Santa Fé do Sul/SP — IBGE 3546603",
    "354670": "Santa Gertrudes/SP — IBGE 3546702",
    "354680": "Santa Isabel/SP — IBGE 3546801",
    "354690": "Santa Lúcia/SP — IBGE 3546900",
    "354700": "Santa Maria da Serra/SP — IBGE 3547007",
    "354710": "Santa Mercedes/SP — IBGE 3547106",
    "354720": "Santana da Ponte Pensa/SP — IBGE 3547205",
    "354730": "Santana de Parnaíba/SP — IBGE 3547304",
    "354740": "Santa Rita d'Oeste/SP — IBGE 3547403",
    "354750": "Santa Rita do Passa Quatro/SP — IBGE 3547502",
    "354760": "Santa Rosa de Viterbo/SP — IBGE 3547601",
    "354765": "Santa Salete/SP — IBGE 3547650",
    "354770": "Santo Anastácio/SP — IBGE 3547700",
    "354780": "Santo André/SP — IBGE 3547809",
    "354790": "Santo Antônio da Alegria/SP — IBGE 3547908",
    "354800": "Santo Antônio de Posse/SP — IBGE 3548005",
    "354805": "Santo Antônio do Aracanguá/SP — IBGE 3548054",
    "354810": "Santo Antônio do Jardim/SP — IBGE 3548104",
    "354820": "Santo Antônio do Pinhal/SP — IBGE 3548203",
    "354830": "Santo Expedito/SP — IBGE 3548302",
    "354840": "Santópolis do Aguapeí/SP — IBGE 3548401",
    "354850": "Santos/SP — IBGE 3548500",
    "354860": "São Bento do Sapucaí/SP — IBGE 3548609",
    "354870": "São Bernardo do Campo/SP — IBGE 3548708",
    "354880": "São Caetano do Sul/SP — IBGE 3548807",
    "354890": "São Carlos/SP — IBGE 3548906",
    "354900": "São Francisco/SP — IBGE 3549003",
    "354910": "São João da Boa Vista/SP — IBGE 3549102",
    "354920": "São João das Duas Pontes/SP — IBGE 3549201",
    "354925": "São João de Iracema/SP — IBGE 3549250",
    "354930": "São João do Pau d'Alho/SP — IBGE 3549300",
    "354940": "São Joaquim da Barra/SP — IBGE 3549409",
    "354950": "São José da Bela Vista/SP — IBGE 3549508",
    "354960": "São José do Barreiro/SP — IBGE 3549607",
    "354970": "São José do Rio Pardo/SP — IBGE 3549706",
    "354980": "São José do Rio Preto/SP — IBGE 3549805",
    "354990": "São José dos Campos/SP — IBGE 3549904",
    "354995": "São Lourenço da Serra/SP — IBGE 3549953",
    "355000": "São Luiz do Paraitinga/SP — IBGE 3550001",
    "355010": "São Manuel/SP — IBGE 3550100",
    "355020": "São Miguel Arcanjo/SP — IBGE 3550209",
    "355030": "São Paulo/SP — IBGE 3550308",
    "355040": "São Pedro/SP — IBGE 3550407",
    "355050": "São Pedro do Turvo/SP — IBGE 3550506",
    "355060": "São Roque/SP — IBGE 3550605",
    "355070": "São Sebastião/SP — IBGE 3550704",
    "355080": "São Sebastião da Grama/SP — IBGE 3550803",
    "355090": "São Simão/SP — IBGE 3550902",
    "355100": "São Vicente/SP — IBGE 3551009",
    "355110": "Sarapuí/SP — IBGE 3551108",
    "355120": "Sarutaiá/SP — IBGE 3551207",
    "355130": "Sebastianópolis do Sul/SP — IBGE 3551306",
    "355140": "Serra Azul/SP — IBGE 3551405",
    "355150": "Serrana/SP — IBGE 3551504",
    "355160": "Serra Negra/SP — IBGE 3551603",
    "355170": "Sertãozinho/SP — IBGE 3551702",
    "355180": "Sete Barras/SP — IBGE 3551801",
    "355190": "Severínia/SP — IBGE 3551900",
    "355200": "Silveiras/SP — IBGE 3552007",
    "355210": "Socorro/SP — IBGE 3552106",
    "355220": "Sorocaba/SP — IBGE 3552205",
    "355230": "Sud Mennucci/SP — IBGE 3552304",
    "355240": "Sumaré/SP — IBGE 3552403",
    "355250": "Suzano/SP — IBGE 3552502",
    "355255": "Suzanápolis/SP — IBGE 3552551",
    "355260": "Tabapuã/SP — IBGE 3552601",
    "355270": "Tabatinga/SP — IBGE 3552700",
    "355280": "Taboão da Serra/SP — IBGE 3552809",
    "355290": "Taciba/SP — IBGE 3552908",
    "355300": "Taguaí/SP — IBGE 3553005",
    "355310": "Taiaçu/SP — IBGE 3553104",
    "355320": "Taiúva/SP — IBGE 3553203",
    "355330": "Tambaú/SP — IBGE 3553302",
    "355340": "Tanabi/SP — IBGE 3553401",
    "355350": "Tapiraí/SP — IBGE 3553500",
    "355360": "Tapiratiba/SP — IBGE 3553609",
    "355365": "Taquaral/SP — IBGE 3553658",
    "355370": "Taquaritinga/SP — IBGE 3553708",
    "355380": "Taquarituba/SP — IBGE 3553807",
    "355385": "Taquarivaí/SP — IBGE 3553856",
    "355390": "Tarabai/SP — IBGE 3553906",
    "355395": "Tarumã/SP — IBGE 3553955",
    "355400": "Tatuí/SP — IBGE 3554003",
    "355410": "Taubaté/SP — IBGE 3554102",
    "355420": "Tejupá/SP — IBGE 3554201",
    "355430": "Teodoro Sampaio/SP — IBGE 3554300",
    "355440": "Terra Roxa/SP — IBGE 3554409",
    "355450": "Tietê/SP — IBGE 3554508",
    "355460": "Timburi/SP — IBGE 3554607",
    "355465": "Torre de Pedra/SP — IBGE 3554656",
    "355470": "Torrinha/SP — IBGE 3554706",
    "355475": "Trabiju/SP — IBGE 3554755",
    "355480": "Tremembé/SP — IBGE 3554805",
    "355490": "Três Fronteiras/SP — IBGE 3554904",
    "355495": "Tuiuti/SP — IBGE 3554953",
    "355500": "Tupã/SP — IBGE 3555000",
    "355510": "Tupi Paulista/SP — IBGE 3555109",
    "355520": "Turiúba/SP — IBGE 3555208",
    "355530": "Turmalina/SP — IBGE 3555307",
    "355535": "Ubarana/SP — IBGE 3555356",
    "355540": "Ubatuba/SP — IBGE 3555406",
    "355550": "Ubirajara/SP — IBGE 3555505",
    "355560": "Uchoa/SP — IBGE 3555604",
    "355570": "União Paulista/SP — IBGE 3555703",
    "355580": "Urânia/SP — IBGE 3555802",
    "355590": "Uru/SP — IBGE 3555901",
    "355600": "Urupês/SP — IBGE 3556008",
    "355610": "Valentim Gentil/SP — IBGE 3556107",
    "355620": "Valinhos/SP — IBGE 3556206",
    "355630": "Valparaíso/SP — IBGE 3556305",
    "355635": "Vargem/SP — IBGE 3556354",
    "355640": "Vargem Grande do Sul/SP — IBGE 3556404",
    "355645": "Vargem Grande Paulista/SP — IBGE 3556453",
    "355650": "Várzea Paulista/SP — IBGE 3556503",
    "355660": "Vera Cruz/SP — IBGE 3556602",
    "355670": "Vinhedo/SP — IBGE 3556701",
    "355680": "Viradouro/SP — IBGE 3556800",
    "355690": "Vista Alegre do Alto/SP — IBGE 3556909",
    "355695": "Vitória Brasil/SP — IBGE 3556958",
    "355700": "Votorantim/SP — IBGE 3557006",
    "355710": "Votuporanga/SP — IBGE 3557105",
    "355715": "Zacarias/SP — IBGE 3557154",
    "355720": "Chavantes/SP — IBGE 3557204",
    "355730": "Estiva Gerbi/SP — IBGE 3557303",
    "410010": "Abatiá/PR — IBGE 4100103",
    "410020": "Adrianópolis/PR — IBGE 4100202",
    "410030": "Agudos do Sul/PR — IBGE 4100301",
    "410040": "Almirante Tamandaré/PR — IBGE 4100400",
    "410045": "Altamira do Paraná/PR — IBGE 4100459",
    "410050": "Altônia/PR — IBGE 4100509",
    "410060": "Alto Paraná/PR — IBGE 4100608",
    "410070": "Alto Piquiri/PR — IBGE 4100707",
    "410080": "Alvorada do Sul/PR — IBGE 4100806",
    "410090": "Amaporã/PR — IBGE 4100905",
    "410100": "Ampére/PR — IBGE 4101002",
    "410105": "Anahy/PR — IBGE 4101051",
    "410110": "Andirá/PR — IBGE 4101101",
    "410115": "Ângulo/PR — IBGE 4101150",
    "410120": "Antonina/PR — IBGE 4101200",
    "410130": "Antônio Olinto/PR — IBGE 4101309",
    "410140": "Apucarana/PR — IBGE 4101408",
    "410150": "Arapongas/PR — IBGE 4101507",
    "410160": "Arapoti/PR — IBGE 4101606",
    "410165": "Arapuã/PR — IBGE 4101655",
    "410170": "Araruna/PR — IBGE 4101705",
    "410180": "Araucária/PR — IBGE 4101804",
    "410185": "Ariranha do Ivaí/PR — IBGE 4101853",
    "410190": "Assaí/PR — IBGE 4101903",
    "410200": "Assis Chateaubriand/PR — IBGE 4102000",
    "410210": "Astorga/PR — IBGE 4102109",
    "410220": "Atalaia/PR — IBGE 4102208",
    "410230": "Balsa Nova/PR — IBGE 4102307",
    "410240": "Bandeirantes/PR — IBGE 4102406",
    "410250": "Barbosa Ferraz/PR — IBGE 4102505",
    "410260": "Barracão/PR — IBGE 4102604",
    "410270": "Barra do Jacaré/PR — IBGE 4102703",
    "410275": "Bela Vista da Caroba/PR — IBGE 4102752",
    "410280": "Bela Vista do Paraíso/PR — IBGE 4102802",
    "410290": "Bituruna/PR — IBGE 4102901",
    "410300": "Boa Esperança/PR — IBGE 4103008",
    "410302": "Boa Esperança do Iguaçu/PR — IBGE 4103024",
    "410304": "Boa Ventura de São Roque/PR — IBGE 4103040",
    "410305": "Boa Vista da Aparecida/PR — IBGE 4103057",
    "410310": "Bocaiúva do Sul/PR — IBGE 4103107",
    "410315": "Bom Jesus do Sul/PR — IBGE 4103156",
    "410320": "Bom Sucesso/PR — IBGE 4103206",
    "410322": "Bom Sucesso do Sul/PR — IBGE 4103222",
    "410330": "Borrazópolis/PR — IBGE 4103305",
    "410335": "Braganey/PR — IBGE 4103354",
    "410337": "Brasilândia do Sul/PR — IBGE 4103370",
    "410340": "Cafeara/PR — IBGE 4103404",
    "410345": "Cafelândia/PR — IBGE 4103453",
    "410347": "Cafezal do Sul/PR — IBGE 4103479",
    "410350": "Califórnia/PR — IBGE 4103503",
    "410360": "Cambará/PR — IBGE 4103602",
    "410370": "Cambé/PR — IBGE 4103701",
    "410380": "Cambira/PR — IBGE 4103800",
    "410390": "Campina da Lagoa/PR — IBGE 4103909",
    "410395": "Campina do Simão/PR — IBGE 4103958",
    "410400": "Campina Grande do Sul/PR — IBGE 4104006",
    "410405": "Campo Bonito/PR — IBGE 4104055",
    "410410": "Campo do Tenente/PR — IBGE 4104105",
    "410420": "Campo Largo/PR — IBGE 4104204",
    "410425": "Campo Magro/PR — IBGE 4104253",
    "410430": "Campo Mourão/PR — IBGE 4104303",
    "410440": "Cândido de Abreu/PR — IBGE 4104402",
    "410442": "Candói/PR — IBGE 4104428",
    "410445": "Cantagalo/PR — IBGE 4104451",
    "410450": "Capanema/PR — IBGE 4104501",
    "410460": "Capitão Leônidas Marques/PR — IBGE 4104600",
    "410465": "Carambeí/PR — IBGE 4104659",
    "410470": "Carlópolis/PR — IBGE 4104709",
    "410480": "Cascavel/PR — IBGE 4104808",
    "410490": "Castro/PR — IBGE 4104907",
    "410500": "Catanduvas/PR — IBGE 4105003",
    "410510": "Centenário do Sul/PR — IBGE 4105102",
    "410520": "Cerro Azul/PR — IBGE 4105201",
    "410530": "Céu Azul/PR — IBGE 4105300",
    "410540": "Chopinzinho/PR — IBGE 4105409",
    "410550": "Cianorte/PR — IBGE 4105508",
    "410560": "Cidade Gaúcha/PR — IBGE 4105607",
    "410570": "Clevelândia/PR — IBGE 4105706",
    "410580": "Colombo/PR — IBGE 4105805",
    "410590": "Colorado/PR — IBGE 4105904",
    "410600": "Congonhinhas/PR — IBGE 4106001",
    "410610": "Conselheiro Mairinck/PR — IBGE 4106100",
    "410620": "Contenda/PR — IBGE 4106209",
    "410630": "Corbélia/PR — IBGE 4106308",
    "410640": "Cornélio Procópio/PR — IBGE 4106407",
    "410645": "Coronel Domingos Soares/PR — IBGE 4106456",
    "410650": "Coronel Vivida/PR — IBGE 4106506",
    "410655": "Corumbataí do Sul/PR — IBGE 4106555",
    "410657": "Cruzeiro do Iguaçu/PR — IBGE 4106571",
    "410660": "Cruzeiro do Oeste/PR — IBGE 4106605",
    "410670": "Cruzeiro do Sul/PR — IBGE 4106704",
    "410680": "Cruz Machado/PR — IBGE 4106803",
    "410685": "Cruzmaltina/PR — IBGE 4106852",
    "410690": "Curitiba/PR — IBGE 4106902",
    "410700": "Curiúva/PR — IBGE 4107009",
    "410710": "Diamante do Norte/PR — IBGE 4107108",
    "410712": "Diamante do Sul/PR — IBGE 4107124",
    "410715": "Diamante D'Oeste/PR — IBGE 4107157",
    "410720": "Dois Vizinhos/PR — IBGE 4107207",
    "410725": "Douradina/PR — IBGE 4107256",
    "410730": "Doutor Camargo/PR — IBGE 4107306",
    "410740": "Enéas Marques/PR — IBGE 4107405",
    "410750": "Engenheiro Beltrão/PR — IBGE 4107504",
    "410752": "Esperança Nova/PR — IBGE 4107520",
    "410753": "Entre Rios do Oeste/PR — IBGE 4107538",
    "410754": "Espigão Alto do Iguaçu/PR — IBGE 4107546",
    "410755": "Farol/PR — IBGE 4107553",
    "410760": "Faxinal/PR — IBGE 4107603",
    "410765": "Fazenda Rio Grande/PR — IBGE 4107652",
    "410770": "Fênix/PR — IBGE 4107702",
    "410773": "Fernandes Pinheiro/PR — IBGE 4107736",
    "410775": "Figueira/PR — IBGE 4107751",
    "410780": "Floraí/PR — IBGE 4107801",
    "410785": "Flor da Serra do Sul/PR — IBGE 4107850",
    "410790": "Floresta/PR — IBGE 4107900",
    "410800": "Florestópolis/PR — IBGE 4108007",
    "410810": "Flórida/PR — IBGE 4108106",
    "410820": "Formosa do Oeste/PR — IBGE 4108205",
    "410830": "Foz do Iguaçu/PR — IBGE 4108304",
    "410832": "Francisco Alves/PR — IBGE 4108320",
    "410840": "Francisco Beltrão/PR — IBGE 4108403",
    "410845": "Foz do Jordão/PR — IBGE 4108452",
    "410850": "General Carneiro/PR — IBGE 4108502",
    "410855": "Godoy Moreira/PR — IBGE 4108551",
    "410860": "Goioerê/PR — IBGE 4108601",
    "410865": "Goioxim/PR — IBGE 4108650",
    "410870": "Grandes Rios/PR — IBGE 4108700",
    "410880": "Guaíra/PR — IBGE 4108809",
    "410890": "Guairaçá/PR — IBGE 4108908",
    "410895": "Guamiranga/PR — IBGE 4108957",
    "410900": "Guapirama/PR — IBGE 4109005",
    "410910": "Guaporema/PR — IBGE 4109104",
    "410920": "Guaraci/PR — IBGE 4109203",
    "410930": "Guaraniaçu/PR — IBGE 4109302",
    "410940": "Guarapuava/PR — IBGE 4109401",
    "410950": "Guaraqueçaba/PR — IBGE 4109500",
    "410960": "Guaratuba/PR — IBGE 4109609",
    "410965": "Honório Serpa/PR — IBGE 4109658",
    "410970": "Ibaiti/PR — IBGE 4109708",
    "410975": "Ibema/PR — IBGE 4109757",
    "410980": "Ibiporã/PR — IBGE 4109807",
    "410990": "Icaraíma/PR — IBGE 4109906",
    "411000": "Iguaraçu/PR — IBGE 4110003",
    "411005": "Iguatu/PR — IBGE 4110052",
    "411007": "Imbaú/PR — IBGE 4110078",
    "411010": "Imbituva/PR — IBGE 4110102",
    "411020": "Inácio Martins/PR — IBGE 4110201",
    "411030": "Inajá/PR — IBGE 4110300",
    "411040": "Indianópolis/PR — IBGE 4110409",
    "411050": "Ipiranga/PR — IBGE 4110508",
    "411060": "Iporã/PR — IBGE 4110607",
    "411065": "Iracema do Oeste/PR — IBGE 4110656",
    "411070": "Irati/PR — IBGE 4110706",
    "411080": "Iretama/PR — IBGE 4110805",
    "411090": "Itaguajé/PR — IBGE 4110904",
    "411095": "Itaipulândia/PR — IBGE 4110953",
    "411100": "Itambaracá/PR — IBGE 4111001",
    "411110": "Itambé/PR — IBGE 4111100",
    "411120": "Itapejara d'Oeste/PR — IBGE 4111209",
    "411125": "Itaperuçu/PR — IBGE 4111258",
    "411130": "Itaúna do Sul/PR — IBGE 4111308",
    "411140": "Ivaí/PR — IBGE 4111407",
    "411150": "Ivaiporã/PR — IBGE 4111506",
    "411155": "Ivaté/PR — IBGE 4111555",
    "411160": "Ivatuba/PR — IBGE 4111605",
    "411170": "Jaboti/PR — IBGE 4111704",
    "411180": "Jacarezinho/PR — IBGE 4111803",
    "411190": "Jaguapitã/PR — IBGE 4111902",
    "411200": "Jaguariaíva/PR — IBGE 4112009",
    "411210": "Jandaia do Sul/PR — IBGE 4112108",
    "411220": "Janiópolis/PR — IBGE 4112207",
    "411230": "Japira/PR — IBGE 4112306",
    "411240": "Japurá/PR — IBGE 4112405",
    "411250": "Jardim Alegre/PR — IBGE 4112504",
    "411260": "Jardim Olinda/PR — IBGE 4112603",
    "411270": "Jataizinho/PR — IBGE 4112702",
    "411275": "Jesuítas/PR — IBGE 4112751",
    "411280": "Joaquim Távora/PR — IBGE 4112801",
    "411290": "Jundiaí do Sul/PR — IBGE 4112900",
    "411295": "Juranda/PR — IBGE 4112959",
    "411300": "Jussara/PR — IBGE 4113007",
    "411310": "Kaloré/PR — IBGE 4113106",
    "411320": "Lapa/PR — IBGE 4113205",
    "411325": "Laranjal/PR — IBGE 4113254",
    "411330": "Laranjeiras do Sul/PR — IBGE 4113304",
    "411340": "Leópolis/PR — IBGE 4113403",
    "411342": "Lidianópolis/PR — IBGE 4113429",
    "411345": "Lindoeste/PR — IBGE 4113452",
    "411350": "Loanda/PR — IBGE 4113502",
    "411360": "Lobato/PR — IBGE 4113601",
    "411370": "Londrina/PR — IBGE 4113700",
    "411373": "Luiziana/PR — IBGE 4113734",
    "411375": "Lunardelli/PR — IBGE 4113759",
    "411380": "Lupionópolis/PR — IBGE 4113809",
    "411390": "Mallet/PR — IBGE 4113908",
    "411400": "Mamborê/PR — IBGE 4114005",
    "411410": "Mandaguaçu/PR — IBGE 4114104",
    "411420": "Mandaguari/PR — IBGE 4114203",
    "411430": "Mandirituba/PR — IBGE 4114302",
    "411435": "Manfrinópolis/PR — IBGE 4114351",
    "411440": "Mangueirinha/PR — IBGE 4114401",
    "411450": "Manoel Ribas/PR — IBGE 4114500",
    "411460": "Marechal Cândido Rondon/PR — IBGE 4114609",
    "411470": "Maria Helena/PR — IBGE 4114708",
    "411480": "Marialva/PR — IBGE 4114807",
    "411490": "Marilândia do Sul/PR — IBGE 4114906",
    "411500": "Marilena/PR — IBGE 4115002",
    "411510": "Mariluz/PR — IBGE 4115101",
    "411520": "Maringá/PR — IBGE 4115200",
    "411530": "Mariópolis/PR — IBGE 4115309",
    "411535": "Maripá/PR — IBGE 4115358",
    "411540": "Marmeleiro/PR — IBGE 4115408",
    "411545": "Marquinho/PR — IBGE 4115457",
    "411550": "Marumbi/PR — IBGE 4115507",
    "411560": "Matelândia/PR — IBGE 4115606",
    "411570": "Matinhos/PR — IBGE 4115705",
    "411573": "Mato Rico/PR — IBGE 4115739",
    "411575": "Mauá da Serra/PR — IBGE 4115754",
    "411580": "Medianeira/PR — IBGE 4115804",
    "411585": "Mercedes/PR — IBGE 4115853",
    "411590": "Mirador/PR — IBGE 4115903",
    "411600": "Miraselva/PR — IBGE 4116000",
    "411605": "Missal/PR — IBGE 4116059",
    "411610": "Moreira Sales/PR — IBGE 4116109",
    "411620": "Morretes/PR — IBGE 4116208",
    "411630": "Munhoz de Melo/PR — IBGE 4116307",
    "411640": "Nossa Senhora das Graças/PR — IBGE 4116406",
    "411650": "Nova Aliança do Ivaí/PR — IBGE 4116505",
    "411660": "Nova América da Colina/PR — IBGE 4116604",
    "411670": "Nova Aurora/PR — IBGE 4116703",
    "411680": "Nova Cantu/PR — IBGE 4116802",
    "411690": "Nova Esperança/PR — IBGE 4116901",
    "411695": "Nova Esperança do Sudoeste/PR — IBGE 4116950",
    "411700": "Nova Fátima/PR — IBGE 4117008",
    "411705": "Nova Laranjeiras/PR — IBGE 4117057",
    "411710": "Nova Londrina/PR — IBGE 4117107",
    "411720": "Nova Olímpia/PR — IBGE 4117206",
    "411721": "Nova Santa Bárbara/PR — IBGE 4117214",
    "411722": "Nova Santa Rosa/PR — IBGE 4117222",
    "411725": "Nova Prata do Iguaçu/PR — IBGE 4117255",
    "411727": "Nova Tebas/PR — IBGE 4117271",
    "411729": "Novo Itacolomi/PR — IBGE 4117297",
    "411730": "Ortigueira/PR — IBGE 4117305",
    "411740": "Ourizona/PR — IBGE 4117404",
    "411745": "Ouro Verde do Oeste/PR — IBGE 4117453",
    "411750": "Paiçandu/PR — IBGE 4117503",
    "411760": "Palmas/PR — IBGE 4117602",
    "411770": "Palmeira/PR — IBGE 4117701",
    "411780": "Palmital/PR — IBGE 4117800",
    "411790": "Palotina/PR — IBGE 4117909",
    "411800": "Paraíso do Norte/PR — IBGE 4118006",
    "411810": "Paranacity/PR — IBGE 4118105",
    "411820": "Paranaguá/PR — IBGE 4118204",
    "411830": "Paranapoema/PR — IBGE 4118303",
    "411840": "Paranavaí/PR — IBGE 4118402",
    "411845": "Pato Bragado/PR — IBGE 4118451",
    "411850": "Pato Branco/PR — IBGE 4118501",
    "411860": "Paula Freitas/PR — IBGE 4118600",
    "411870": "Paulo Frontin/PR — IBGE 4118709",
    "411880": "Peabiru/PR — IBGE 4118808",
    "411885": "Perobal/PR — IBGE 4118857",
    "411890": "Pérola/PR — IBGE 4118907",
    "411900": "Pérola d'Oeste/PR — IBGE 4119004",
    "411910": "Piên/PR — IBGE 4119103",
    "411915": "Pinhais/PR — IBGE 4119152",
    "411920": "Pinhalão/PR — IBGE 4119202",
    "411925": "Pinhal de São Bento/PR — IBGE 4119251",
    "411930": "Pinhão/PR — IBGE 4119301",
    "411940": "Piraí do Sul/PR — IBGE 4119400",
    "411950": "Piraquara/PR — IBGE 4119509",
    "411960": "Pitanga/PR — IBGE 4119608",
    "411965": "Pitangueiras/PR — IBGE 4119657",
    "411970": "Planaltina do Paraná/PR — IBGE 4119707",
    "411980": "Planalto/PR — IBGE 4119806",
    "411990": "Ponta Grossa/PR — IBGE 4119905",
    "411995": "Pontal do Paraná/PR — IBGE 4119954",
    "412000": "Porecatu/PR — IBGE 4120002",
    "412010": "Porto Amazonas/PR — IBGE 4120101",
    "412015": "Porto Barreiro/PR — IBGE 4120150",
    "412020": "Porto Rico/PR — IBGE 4120200",
    "412030": "Porto Vitória/PR — IBGE 4120309",
    "412033": "Prado Ferreira/PR — IBGE 4120333",
    "412035": "Pranchita/PR — IBGE 4120358",
    "412040": "Presidente Castelo Branco/PR — IBGE 4120408",
    "412050": "Primeiro de Maio/PR — IBGE 4120507",
    "412060": "Prudentópolis/PR — IBGE 4120606",
    "412065": "Quarto Centenário/PR — IBGE 4120655",
    "412070": "Quatiguá/PR — IBGE 4120705",
    "412080": "Quatro Barras/PR — IBGE 4120804",
    "412085": "Quatro Pontes/PR — IBGE 4120853",
    "412090": "Quedas do Iguaçu/PR — IBGE 4120903",
    "412100": "Querência do Norte/PR — IBGE 4121000",
    "412110": "Quinta do Sol/PR — IBGE 4121109",
    "412120": "Quitandinha/PR — IBGE 4121208",
    "412125": "Ramilândia/PR — IBGE 4121257",
    "412130": "Rancho Alegre/PR — IBGE 4121307",
    "412135": "Rancho Alegre D'Oeste/PR — IBGE 4121356",
    "412140": "Realeza/PR — IBGE 4121406",
    "412150": "Rebouças/PR — IBGE 4121505",
    "412160": "Renascença/PR — IBGE 4121604",
    "412170": "Reserva/PR — IBGE 4121703",
    "412175": "Reserva do Iguaçu/PR — IBGE 4121752",
    "412180": "Ribeirão Claro/PR — IBGE 4121802",
    "412190": "Ribeirão do Pinhal/PR — IBGE 4121901",
    "412200": "Rio Azul/PR — IBGE 4122008",
    "412210": "Rio Bom/PR — IBGE 4122107",
    "412215": "Rio Bonito do Iguaçu/PR — IBGE 4122156",
    "412217": "Rio Branco do Ivaí/PR — IBGE 4122172",
    "412220": "Rio Branco do Sul/PR — IBGE 4122206",
    "412230": "Rio Negro/PR — IBGE 4122305",
    "412240": "Rolândia/PR — IBGE 4122404",
    "412250": "Roncador/PR — IBGE 4122503",
    "412260": "Rondon/PR — IBGE 4122602",
    "412265": "Rosário do Ivaí/PR — IBGE 4122651",
    "412270": "Sabáudia/PR — IBGE 4122701",
    "412280": "Salgado Filho/PR — IBGE 4122800",
    "412290": "Salto do Itararé/PR — IBGE 4122909",
    "412300": "Salto do Lontra/PR — IBGE 4123006",
    "412310": "Santa Amélia/PR — IBGE 4123105",
    "412320": "Santa Cecília do Pavão/PR — IBGE 4123204",
    "412330": "Santa Cruz de Monte Castelo/PR — IBGE 4123303",
    "412340": "Santa Fé/PR — IBGE 4123402",
    "412350": "Santa Helena/PR — IBGE 4123501",
    "412360": "Santa Inês/PR — IBGE 4123600",
    "412370": "Santa Isabel do Ivaí/PR — IBGE 4123709",
    "412380": "Santa Izabel do Oeste/PR — IBGE 4123808",
    "412382": "Santa Lúcia/PR — IBGE 4123824",
    "412385": "Santa Maria do Oeste/PR — IBGE 4123857",
    "412390": "Santa Mariana/PR — IBGE 4123907",
    "412395": "Santa Mônica/PR — IBGE 4123956",
    "412400": "Santana do Itararé/PR — IBGE 4124004",
    "412402": "Santa Tereza do Oeste/PR — IBGE 4124020",
    "412405": "Santa Terezinha de Itaipu/PR — IBGE 4124053",
    "412410": "Santo Antônio da Platina/PR — IBGE 4124103",
    "412420": "Santo Antônio do Caiuá/PR — IBGE 4124202",
    "412430": "Santo Antônio do Paraíso/PR — IBGE 4124301",
    "412440": "Santo Antônio do Sudoeste/PR — IBGE 4124400",
    "412450": "Santo Inácio/PR — IBGE 4124509",
    "412460": "São Carlos do Ivaí/PR — IBGE 4124608",
    "412470": "São Jerônimo da Serra/PR — IBGE 4124707",
    "412480": "São João/PR — IBGE 4124806",
    "412490": "São João do Caiuá/PR — IBGE 4124905",
    "412500": "São João do Ivaí/PR — IBGE 4125001",
    "412510": "São João do Triunfo/PR — IBGE 4125100",
    "412520": "São Jorge d'Oeste/PR — IBGE 4125209",
    "412530": "São Jorge do Ivaí/PR — IBGE 4125308",
    "412535": "São Jorge do Patrocínio/PR — IBGE 4125357",
    "412540": "São José da Boa Vista/PR — IBGE 4125407",
    "412545": "São José das Palmeiras/PR — IBGE 4125456",
    "412550": "São José dos Pinhais/PR — IBGE 4125506",
    "412555": "São Manoel do Paraná/PR — IBGE 4125555",
    "412560": "São Mateus do Sul/PR — IBGE 4125605",
    "412570": "São Miguel do Iguaçu/PR — IBGE 4125704",
    "412575": "São Pedro do Iguaçu/PR — IBGE 4125753",
    "412580": "São Pedro do Ivaí/PR — IBGE 4125803",
    "412590": "São Pedro do Paraná/PR — IBGE 4125902",
    "412600": "São Sebastião da Amoreira/PR — IBGE 4126009",
    "412610": "São Tomé/PR — IBGE 4126108",
    "412620": "Sapopema/PR — IBGE 4126207",
    "412625": "Sarandi/PR — IBGE 4126256",
    "412627": "Saudade do Iguaçu/PR — IBGE 4126272",
    "412630": "Sengés/PR — IBGE 4126306",
    "412635": "Serranópolis do Iguaçu/PR — IBGE 4126355",
    "412640": "Sertaneja/PR — IBGE 4126405",
    "412650": "Sertanópolis/PR — IBGE 4126504",
    "412660": "Siqueira Campos/PR — IBGE 4126603",
    "412665": "Sulina/PR — IBGE 4126652",
    "412667": "Tamarana/PR — IBGE 4126678",
    "412670": "Tamboara/PR — IBGE 4126702",
    "412680": "Tapejara/PR — IBGE 4126801",
    "412690": "Tapira/PR — IBGE 4126900",
    "412700": "Teixeira Soares/PR — IBGE 4127007",
    "412710": "Telêmaco Borba/PR — IBGE 4127106",
    "412720": "Terra Boa/PR — IBGE 4127205",
    "412730": "Terra Rica/PR — IBGE 4127304",
    "412740": "Terra Roxa/PR — IBGE 4127403",
    "412750": "Tibagi/PR — IBGE 4127502",
    "412760": "Tijucas do Sul/PR — IBGE 4127601",
    "412770": "Toledo/PR — IBGE 4127700",
    "412780": "Tomazina/PR — IBGE 4127809",
    "412785": "Três Barras do Paraná/PR — IBGE 4127858",
    "412788": "Tunas do Paraná/PR — IBGE 4127882",
    "412790": "Tuneiras do Oeste/PR — IBGE 4127908",
    "412795": "Tupãssi/PR — IBGE 4127957",
    "412796": "Turvo/PR — IBGE 4127965",
    "412800": "Ubiratã/PR — IBGE 4128005",
    "412810": "Umuarama/PR — IBGE 4128104",
    "412820": "União da Vitória/PR — IBGE 4128203",
    "412830": "Uniflor/PR — IBGE 4128302",
    "412840": "Uraí/PR — IBGE 4128401",
    "412850": "Wenceslau Braz/PR — IBGE 4128500",
    "412853": "Ventania/PR — IBGE 4128534",
    "412855": "Vera Cruz do Oeste/PR — IBGE 4128559",
    "412860": "Verê/PR — IBGE 4128609",
    "412862": "Alto Paraíso/PR — IBGE 4128625",
    "412863": "Doutor Ulysses/PR — IBGE 4128633",
    "412865": "Virmond/PR — IBGE 4128658",
    "412870": "Vitorino/PR — IBGE 4128708",
    "412880": "Xambrê/PR — IBGE 4128807",
    "420005": "Abdon Batista/SC — IBGE 4200051",
    "420010": "Abelardo Luz/SC — IBGE 4200101",
    "420020": "Agrolândia/SC — IBGE 4200200",
    "420030": "Agronômica/SC — IBGE 4200309",
    "420040": "Água Doce/SC — IBGE 4200408",
    "420050": "Águas de Chapecó/SC — IBGE 4200507",
    "420055": "Águas Frias/SC — IBGE 4200556",
    "420060": "Águas Mornas/SC — IBGE 4200606",
    "420070": "Alfredo Wagner/SC — IBGE 4200705",
    "420075": "Alto Bela Vista/SC — IBGE 4200754",
    "420080": "Anchieta/SC — IBGE 4200804",
    "420090": "Angelina/SC — IBGE 4200903",
    "420100": "Anita Garibaldi/SC — IBGE 4201000",
    "420110": "Anitápolis/SC — IBGE 4201109",
    "420120": "Antônio Carlos/SC — IBGE 4201208",
    "420125": "Apiúna/SC — IBGE 4201257",
    "420127": "Arabutã/SC — IBGE 4201273",
    "420130": "Araquari/SC — IBGE 4201307",
    "420140": "Araranguá/SC — IBGE 4201406",
    "420150": "Armazém/SC — IBGE 4201505",
    "420160": "Arroio Trinta/SC — IBGE 4201604",
    "420165": "Arvoredo/SC — IBGE 4201653",
    "420170": "Ascurra/SC — IBGE 4201703",
    "420180": "Atalanta/SC — IBGE 4201802",
    "420190": "Aurora/SC — IBGE 4201901",
    "420195": "Balneário Arroio do Silva/SC — IBGE 4201950",
    "420200": "Balneário Camboriú/SC — IBGE 4202008",
    "420205": "Balneário Barra do Sul/SC — IBGE 4202057",
    "420207": "Balneário Gaivota/SC — IBGE 4202073",
    "420208": "Bandeirante/SC — IBGE 4202081",
    "420209": "Barra Bonita/SC — IBGE 4202099",
    "420210": "Barra Velha/SC — IBGE 4202107",
    "420213": "Bela Vista do Toldo/SC — IBGE 4202131",
    "420215": "Belmonte/SC — IBGE 4202156",
    "420220": "Benedito Novo/SC — IBGE 4202206",
    "420230": "Biguaçu/SC — IBGE 4202305",
    "420240": "Blumenau/SC — IBGE 4202404",
    "420243": "Bocaina do Sul/SC — IBGE 4202438",
    "420245": "Bombinhas/SC — IBGE 4202453",
    "420250": "Bom Jardim da Serra/SC — IBGE 4202503",
    "420253": "Bom Jesus/SC — IBGE 4202537",
    "420257": "Bom Jesus do Oeste/SC — IBGE 4202578",
    "420260": "Bom Retiro/SC — IBGE 4202602",
    "420270": "Botuverá/SC — IBGE 4202701",
    "420280": "Braço do Norte/SC — IBGE 4202800",
    "420285": "Braço do Trombudo/SC — IBGE 4202859",
    "420287": "Brunópolis/SC — IBGE 4202875",
    "420290": "Brusque/SC — IBGE 4202909",
    "420300": "Caçador/SC — IBGE 4203006",
    "420310": "Caibi/SC — IBGE 4203105",
    "420315": "Calmon/SC — IBGE 4203154",
    "420320": "Camboriú/SC — IBGE 4203204",
    "420325": "Capão Alto/SC — IBGE 4203253",
    "420330": "Campo Alegre/SC — IBGE 4203303",
    "420340": "Campo Belo do Sul/SC — IBGE 4203402",
    "420350": "Campo Erê/SC — IBGE 4203501",
    "420360": "Campos Novos/SC — IBGE 4203600",
    "420370": "Canelinha/SC — IBGE 4203709",
    "420380": "Canoinhas/SC — IBGE 4203808",
    "420390": "Capinzal/SC — IBGE 4203907",
    "420395": "Capivari de Baixo/SC — IBGE 4203956",
    "420400": "Catanduvas/SC — IBGE 4204004",
    "420410": "Caxambu do Sul/SC — IBGE 4204103",
    "420415": "Celso Ramos/SC — IBGE 4204152",
    "420417": "Cerro Negro/SC — IBGE 4204178",
    "420419": "Chapadão do Lageado/SC — IBGE 4204194",
    "420420": "Chapecó/SC — IBGE 4204202",
    "420425": "Cocal do Sul/SC — IBGE 4204251",
    "420430": "Concórdia/SC — IBGE 4204301",
    "420435": "Cordilheira Alta/SC — IBGE 4204350",
    "420440": "Coronel Freitas/SC — IBGE 4204400",
    "420445": "Coronel Martins/SC — IBGE 4204459",
    "420450": "Corupá/SC — IBGE 4204509",
    "420455": "Correia Pinto/SC — IBGE 4204558",
    "420460": "Criciúma/SC — IBGE 4204608",
    "420470": "Cunha Porã/SC — IBGE 4204707",
    "420475": "Cunhataí/SC — IBGE 4204756",
    "420480": "Curitibanos/SC — IBGE 4204806",
    "420490": "Descanso/SC — IBGE 4204905",
    "420500": "Dionísio Cerqueira/SC — IBGE 4205001",
    "420510": "Dona Emma/SC — IBGE 4205100",
    "420515": "Doutor Pedrinho/SC — IBGE 4205159",
    "420517": "Entre Rios/SC — IBGE 4205175",
    "420519": "Ermo/SC — IBGE 4205191",
    "420520": "Erval Velho/SC — IBGE 4205209",
    "420530": "Faxinal dos Guedes/SC — IBGE 4205308",
    "420535": "Flor do Sertão/SC — IBGE 4205357",
    "420540": "Florianópolis/SC — IBGE 4205407",
    "420543": "Formosa do Sul/SC — IBGE 4205431",
    "420545": "Forquilhinha/SC — IBGE 4205456",
    "420550": "Fraiburgo/SC — IBGE 4205506",
    "420555": "Frei Rogério/SC — IBGE 4205555",
    "420560": "Galvão/SC — IBGE 4205605",
    "420570": "Garopaba/SC — IBGE 4205704",
    "420580": "Garuva/SC — IBGE 4205803",
    "420590": "Gaspar/SC — IBGE 4205902",
    "420600": "Governador Celso Ramos/SC — IBGE 4206009",
    "420610": "Grão-Pará/SC — IBGE 4206108",
    "420620": "Gravatal/SC — IBGE 4206207",
    "420630": "Guabiruba/SC — IBGE 4206306",
    "420640": "Guaraciaba/SC — IBGE 4206405",
    "420650": "Guaramirim/SC — IBGE 4206504",
    "420660": "Guarujá do Sul/SC — IBGE 4206603",
    "420665": "Guatambú/SC — IBGE 4206652",
    "420670": "Herval d'Oeste/SC — IBGE 4206702",
    "420675": "Ibiam/SC — IBGE 4206751",
    "420680": "Ibicaré/SC — IBGE 4206801",
    "420690": "Ibirama/SC — IBGE 4206900",
    "420700": "Içara/SC — IBGE 4207007",
    "420710": "Ilhota/SC — IBGE 4207106",
    "420720": "Imaruí/SC — IBGE 4207205",
    "420730": "Imbituba/SC — IBGE 4207304",
    "420740": "Imbuia/SC — IBGE 4207403",
    "420750": "Indaial/SC — IBGE 4207502",
    "420757": "Iomerê/SC — IBGE 4207577",
    "420760": "Ipira/SC — IBGE 4207601",
    "420765": "Iporã do Oeste/SC — IBGE 4207650",
    "420768": "Ipuaçu/SC — IBGE 4207684",
    "420770": "Ipumirim/SC — IBGE 4207700",
    "420775": "Iraceminha/SC — IBGE 4207759",
    "420780": "Irani/SC — IBGE 4207809",
    "420785": "Irati/SC — IBGE 4207858",
    "420790": "Irineópolis/SC — IBGE 4207908",
    "420800": "Itá/SC — IBGE 4208005",
    "420810": "Itaiópolis/SC — IBGE 4208104",
    "420820": "Itajaí/SC — IBGE 4208203",
    "420830": "Itapema/SC — IBGE 4208302",
    "420840": "Itapiranga/SC — IBGE 4208401",
    "420845": "Itapoá/SC — IBGE 4208450",
    "420850": "Ituporanga/SC — IBGE 4208500",
    "420860": "Jaborá/SC — IBGE 4208609",
    "420870": "Jacinto Machado/SC — IBGE 4208708",
    "420880": "Jaguaruna/SC — IBGE 4208807",
    "420890": "Jaraguá do Sul/SC — IBGE 4208906",
    "420895": "Jardinópolis/SC — IBGE 4208955",
    "420900": "Joaçaba/SC — IBGE 4209003",
    "420910": "Joinville/SC — IBGE 4209102",
    "420915": "José Boiteux/SC — IBGE 4209151",
    "420917": "Jupiá/SC — IBGE 4209177",
    "420920": "Lacerdópolis/SC — IBGE 4209201",
    "420930": "Lages/SC — IBGE 4209300",
    "420940": "Laguna/SC — IBGE 4209409",
    "420945": "Lajeado Grande/SC — IBGE 4209458",
    "420950": "Laurentino/SC — IBGE 4209508",
    "420960": "Lauro Müller/SC — IBGE 4209607",
    "420970": "Lebon Régis/SC — IBGE 4209706",
    "420980": "Leoberto Leal/SC — IBGE 4209805",
    "420985": "Lindóia do Sul/SC — IBGE 4209854",
    "420990": "Lontras/SC — IBGE 4209904",
    "421000": "Luiz Alves/SC — IBGE 4210001",
    "421003": "Luzerna/SC — IBGE 4210035",
    "421005": "Macieira/SC — IBGE 4210050",
    "421010": "Mafra/SC — IBGE 4210100",
    "421020": "Major Gercino/SC — IBGE 4210209",
    "421030": "Major Vieira/SC — IBGE 4210308",
    "421040": "Maracajá/SC — IBGE 4210407",
    "421050": "Maravilha/SC — IBGE 4210506",
    "421055": "Marema/SC — IBGE 4210555",
    "421060": "Massaranduba/SC — IBGE 4210605",
    "421070": "Matos Costa/SC — IBGE 4210704",
    "421080": "Meleiro/SC — IBGE 4210803",
    "421085": "Mirim Doce/SC — IBGE 4210852",
    "421090": "Modelo/SC — IBGE 4210902",
    "421100": "Mondaí/SC — IBGE 4211009",
    "421105": "Monte Carlo/SC — IBGE 4211058",
    "421110": "Monte Castelo/SC — IBGE 4211108",
    "421120": "Morro da Fumaça/SC — IBGE 4211207",
    "421125": "Morro Grande/SC — IBGE 4211256",
    "421130": "Navegantes/SC — IBGE 4211306",
    "421140": "Nova Erechim/SC — IBGE 4211405",
    "421145": "Nova Itaberaba/SC — IBGE 4211454",
    "421150": "Nova Trento/SC — IBGE 4211504",
    "421160": "Nova Veneza/SC — IBGE 4211603",
    "421165": "Novo Horizonte/SC — IBGE 4211652",
    "421170": "Orleans/SC — IBGE 4211702",
    "421175": "Otacílio Costa/SC — IBGE 4211751",
    "421180": "Ouro/SC — IBGE 4211801",
    "421185": "Ouro Verde/SC — IBGE 4211850",
    "421187": "Paial/SC — IBGE 4211876",
    "421189": "Painel/SC — IBGE 4211892",
    "421190": "Palhoça/SC — IBGE 4211900",
    "421200": "Palma Sola/SC — IBGE 4212007",
    "421205": "Palmeira/SC — IBGE 4212056",
    "421210": "Palmitos/SC — IBGE 4212106",
    "421220": "Papanduva/SC — IBGE 4212205",
    "421223": "Paraíso/SC — IBGE 4212239",
    "421225": "Passo de Torres/SC — IBGE 4212254",
    "421227": "Passos Maia/SC — IBGE 4212270",
    "421230": "Paulo Lopes/SC — IBGE 4212304",
    "421240": "Pedras Grandes/SC — IBGE 4212403",
    "421250": "Penha/SC — IBGE 4212502",
    "421260": "Peritiba/SC — IBGE 4212601",
    "421265": "Pescaria Brava/SC — IBGE 4212650",
    "421270": "Petrolândia/SC — IBGE 4212700",
    "421280": "Balneário Piçarras/SC — IBGE 4212809",
    "421290": "Pinhalzinho/SC — IBGE 4212908",
    "421300": "Pinheiro Preto/SC — IBGE 4213005",
    "421310": "Piratuba/SC — IBGE 4213104",
    "421315": "Planalto Alegre/SC — IBGE 4213153",
    "421320": "Pomerode/SC — IBGE 4213203",
    "421330": "Ponte Alta/SC — IBGE 4213302",
    "421335": "Ponte Alta do Norte/SC — IBGE 4213351",
    "421340": "Ponte Serrada/SC — IBGE 4213401",
    "421350": "Porto Belo/SC — IBGE 4213500",
    "421360": "Porto União/SC — IBGE 4213609",
    "421370": "Pouso Redondo/SC — IBGE 4213708",
    "421380": "Praia Grande/SC — IBGE 4213807",
    "421390": "Presidente Castello Branco/SC — IBGE 4213906",
    "421400": "Presidente Getúlio/SC — IBGE 4214003",
    "421410": "Presidente Nereu/SC — IBGE 4214102",
    "421415": "Princesa/SC — IBGE 4214151",
    "421420": "Quilombo/SC — IBGE 4214201",
    "421430": "Rancho Queimado/SC — IBGE 4214300",
    "421440": "Rio das Antas/SC — IBGE 4214409",
    "421450": "Rio do Campo/SC — IBGE 4214508",
    "421460": "Rio do Oeste/SC — IBGE 4214607",
    "421470": "Rio dos Cedros/SC — IBGE 4214706",
    "421480": "Rio do Sul/SC — IBGE 4214805",
    "421490": "Rio Fortuna/SC — IBGE 4214904",
    "421500": "Rio Negrinho/SC — IBGE 4215000",
    "421505": "Rio Rufino/SC — IBGE 4215059",
    "421507": "Riqueza/SC — IBGE 4215075",
    "421510": "Rodeio/SC — IBGE 4215109",
    "421520": "Romelândia/SC — IBGE 4215208",
    "421530": "Salete/SC — IBGE 4215307",
    "421535": "Saltinho/SC — IBGE 4215356",
    "421540": "Salto Veloso/SC — IBGE 4215406",
    "421545": "Sangão/SC — IBGE 4215455",
    "421550": "Santa Cecília/SC — IBGE 4215505",
    "421555": "Santa Helena/SC — IBGE 4215554",
    "421560": "Santa Rosa de Lima/SC — IBGE 4215604",
    "421565": "Santa Rosa do Sul/SC — IBGE 4215653",
    "421567": "Santa Terezinha/SC — IBGE 4215679",
    "421568": "Santa Terezinha do Progresso/SC — IBGE 4215687",
    "421569": "Santiago do Sul/SC — IBGE 4215695",
    "421570": "Santo Amaro da Imperatriz/SC — IBGE 4215703",
    "421575": "São Bernardino/SC — IBGE 4215752",
    "421580": "São Bento do Sul/SC — IBGE 4215802",
    "421590": "São Bonifácio/SC — IBGE 4215901",
    "421600": "São Carlos/SC — IBGE 4216008",
    "421605": "São Cristóvão do Sul/SC — IBGE 4216057",
    "421610": "São Domingos/SC — IBGE 4216107",
    "421620": "São Francisco do Sul/SC — IBGE 4216206",
    "421625": "São João do Oeste/SC — IBGE 4216255",
    "421630": "São João Batista/SC — IBGE 4216305",
    "421635": "São João do Itaperiú/SC — IBGE 4216354",
    "421640": "São João do Sul/SC — IBGE 4216404",
    "421650": "São Joaquim/SC — IBGE 4216503",
    "421660": "São José/SC — IBGE 4216602",
    "421670": "São José do Cedro/SC — IBGE 4216701",
    "421680": "São José do Cerrito/SC — IBGE 4216800",
    "421690": "São Lourenço do Oeste/SC — IBGE 4216909",
    "421700": "São Ludgero/SC — IBGE 4217006",
    "421710": "São Martinho/SC — IBGE 4217105",
    "421715": "São Miguel da Boa Vista/SC — IBGE 4217154",
    "421720": "São Miguel do Oeste/SC — IBGE 4217204",
    "421725": "São Pedro de Alcântara/SC — IBGE 4217253",
    "421730": "Saudades/SC — IBGE 4217303",
    "421740": "Schroeder/SC — IBGE 4217402",
    "421750": "Seara/SC — IBGE 4217501",
    "421755": "Serra Alta/SC — IBGE 4217550",
    "421760": "Siderópolis/SC — IBGE 4217600",
    "421770": "Sombrio/SC — IBGE 4217709",
    "421775": "Sul Brasil/SC — IBGE 4217758",
    "421780": "Taió/SC — IBGE 4217808",
    "421790": "Tangará/SC — IBGE 4217907",
    "421795": "Tigrinhos/SC — IBGE 4217956",
    "421800": "Tijucas/SC — IBGE 4218004",
    "421810": "Timbé do Sul/SC — IBGE 4218103",
    "421820": "Timbó/SC — IBGE 4218202",
    "421825": "Timbó Grande/SC — IBGE 4218251",
    "421830": "Três Barras/SC — IBGE 4218301",
    "421835": "Treviso/SC — IBGE 4218350",
    "421840": "Treze de Maio/SC — IBGE 4218400",
    "421850": "Treze Tílias/SC — IBGE 4218509",
    "421860": "Trombudo Central/SC — IBGE 4218608",
    "421870": "Tubarão/SC — IBGE 4218707",
    "421875": "Tunápolis/SC — IBGE 4218756",
    "421880": "Turvo/SC — IBGE 4218806",
    "421885": "União do Oeste/SC — IBGE 4218855",
    "421890": "Urubici/SC — IBGE 4218905",
    "421895": "Urupema/SC — IBGE 4218954",
    "421900": "Urussanga/SC — IBGE 4219002",
    "421910": "Vargeão/SC — IBGE 4219101",
    "421915": "Vargem/SC — IBGE 4219150",
    "421917": "Vargem Bonita/SC — IBGE 4219176",
    "421920": "Vidal Ramos/SC — IBGE 4219200",
    "421930": "Videira/SC — IBGE 4219309",
    "421935": "Vitor Meireles/SC — IBGE 4219358",
    "421940": "Witmarsum/SC — IBGE 4219408",
    "421950": "Xanxerê/SC — IBGE 4219507",
    "421960": "Xavantina/SC — IBGE 4219606",
    "421970": "Xaxim/SC — IBGE 4219705",
    "421985": "Zortéa/SC — IBGE 4219853",
    "422000": "Balneário Rincão/SC — IBGE 4220000",
    "430003": "Aceguá/RS — IBGE 4300034",
    "430005": "Água Santa/RS — IBGE 4300059",
    "430010": "Agudo/RS — IBGE 4300109",
    "430020": "Ajuricaba/RS — IBGE 4300208",
    "430030": "Alecrim/RS — IBGE 4300307",
    "430040": "Alegrete/RS — IBGE 4300406",
    "430045": "Alegria/RS — IBGE 4300455",
    "430047": "Almirante Tamandaré do Sul/RS — IBGE 4300471",
    "430050": "Alpestre/RS — IBGE 4300505",
    "430055": "Alto Alegre/RS — IBGE 4300554",
    "430057": "Alto Feliz/RS — IBGE 4300570",
    "430060": "Alvorada/RS — IBGE 4300604",
    "430063": "Amaral Ferrador/RS — IBGE 4300638",
    "430064": "Ametista do Sul/RS — IBGE 4300646",
    "430066": "André da Rocha/RS — IBGE 4300661",
    "430070": "Anta Gorda/RS — IBGE 4300703",
    "430080": "Antônio Prado/RS — IBGE 4300802",
    "430085": "Arambaré/RS — IBGE 4300851",
    "430087": "Araricá/RS — IBGE 4300877",
    "430090": "Aratiba/RS — IBGE 4300901",
    "430100": "Arroio do Meio/RS — IBGE 4301008",
    "430105": "Arroio do Sal/RS — IBGE 4301057",
    "430107": "Arroio do Padre/RS — IBGE 4301073",
    "430110": "Arroio dos Ratos/RS — IBGE 4301107",
    "430120": "Arroio do Tigre/RS — IBGE 4301206",
    "430130": "Arroio Grande/RS — IBGE 4301305",
    "430140": "Arvorezinha/RS — IBGE 4301404",
    "430150": "Augusto Pestana/RS — IBGE 4301503",
    "430155": "Áurea/RS — IBGE 4301552",
    "430160": "Bagé/RS — IBGE 4301602",
    "430163": "Balneário Pinhal/RS — IBGE 4301636",
    "430165": "Barão/RS — IBGE 4301651",
    "430170": "Barão de Cotegipe/RS — IBGE 4301701",
    "430175": "Barão do Triunfo/RS — IBGE 4301750",
    "430180": "Barracão/RS — IBGE 4301800",
    "430185": "Barra do Guarita/RS — IBGE 4301859",
    "430187": "Barra do Quaraí/RS — IBGE 4301875",
    "430190": "Barra do Ribeiro/RS — IBGE 4301909",
    "430192": "Barra do Rio Azul/RS — IBGE 4301925",
    "430195": "Barra Funda/RS — IBGE 4301958",
    "430200": "Barros Cassal/RS — IBGE 4302006",
    "430205": "Benjamin Constant do Sul/RS — IBGE 4302055",
    "430210": "Bento Gonçalves/RS — IBGE 4302105",
    "430215": "Boa Vista das Missões/RS — IBGE 4302154",
    "430220": "Boa Vista do Buricá/RS — IBGE 4302204",
    "430222": "Boa Vista do Cadeado/RS — IBGE 4302220",
    "430223": "Boa Vista do Incra/RS — IBGE 4302238",
    "430225": "Boa Vista do Sul/RS — IBGE 4302253",
    "430230": "Bom Jesus/RS — IBGE 4302303",
    "430235": "Bom Princípio/RS — IBGE 4302352",
    "430237": "Bom Progresso/RS — IBGE 4302378",
    "430240": "Bom Retiro do Sul/RS — IBGE 4302402",
    "430245": "Boqueirão do Leão/RS — IBGE 4302451",
    "430250": "Bossoroca/RS — IBGE 4302501",
    "430258": "Bozano/RS — IBGE 4302584",
    "430260": "Braga/RS — IBGE 4302600",
    "430265": "Brochier/RS — IBGE 4302659",
    "430270": "Butiá/RS — IBGE 4302709",
    "430280": "Caçapava do Sul/RS — IBGE 4302808",
    "430290": "Cacequi/RS — IBGE 4302907",
    "430300": "Cachoeira do Sul/RS — IBGE 4303004",
    "430310": "Cachoeirinha/RS — IBGE 4303103",
    "430320": "Cacique Doble/RS — IBGE 4303202",
    "430330": "Caibaté/RS — IBGE 4303301",
    "430340": "Caiçara/RS — IBGE 4303400",
    "430350": "Camaquã/RS — IBGE 4303509",
    "430355": "Camargo/RS — IBGE 4303558",
    "430360": "Cambará do Sul/RS — IBGE 4303608",
    "430367": "Campestre da Serra/RS — IBGE 4303673",
    "430370": "Campina das Missões/RS — IBGE 4303707",
    "430380": "Campinas do Sul/RS — IBGE 4303806",
    "430390": "Campo Bom/RS — IBGE 4303905",
    "430400": "Campo Novo/RS — IBGE 4304002",
    "430410": "Campos Borges/RS — IBGE 4304101",
    "430420": "Candelária/RS — IBGE 4304200",
    "430430": "Cândido Godói/RS — IBGE 4304309",
    "430435": "Candiota/RS — IBGE 4304358",
    "430440": "Canela/RS — IBGE 4304408",
    "430450": "Canguçu/RS — IBGE 4304507",
    "430460": "Canoas/RS — IBGE 4304606",
    "430461": "Canudos do Vale/RS — IBGE 4304614",
    "430462": "Capão Bonito do Sul/RS — IBGE 4304622",
    "430463": "Capão da Canoa/RS — IBGE 4304630",
    "430465": "Capão do Cipó/RS — IBGE 4304655",
    "430466": "Capão do Leão/RS — IBGE 4304663",
    "430467": "Capivari do Sul/RS — IBGE 4304671",
    "430468": "Capela de Santana/RS — IBGE 4304689",
    "430469": "Capitão/RS — IBGE 4304697",
    "430470": "Carazinho/RS — IBGE 4304705",
    "430471": "Caraá/RS — IBGE 4304713",
    "430480": "Carlos Barbosa/RS — IBGE 4304804",
    "430485": "Carlos Gomes/RS — IBGE 4304853",
    "430490": "Casca/RS — IBGE 4304903",
    "430495": "Caseiros/RS — IBGE 4304952",
    "430500": "Catuípe/RS — IBGE 4305009",
    "430510": "Caxias do Sul/RS — IBGE 4305108",
    "430511": "Centenário/RS — IBGE 4305116",
    "430512": "Cerrito/RS — IBGE 4305124",
    "430513": "Cerro Branco/RS — IBGE 4305132",
    "430515": "Cerro Grande/RS — IBGE 4305157",
    "430517": "Cerro Grande do Sul/RS — IBGE 4305173",
    "430520": "Cerro Largo/RS — IBGE 4305207",
    "430530": "Chapada/RS — IBGE 4305306",
    "430535": "Charqueadas/RS — IBGE 4305355",
    "430537": "Charrua/RS — IBGE 4305371",
    "430540": "Chiapetta/RS — IBGE 4305405",
    "430543": "Chuí/RS — IBGE 4305439",
    "430544": "Chuvisca/RS — IBGE 4305447",
    "430545": "Cidreira/RS — IBGE 4305454",
    "430550": "Ciríaco/RS — IBGE 4305504",
    "430558": "Colinas/RS — IBGE 4305587",
    "430560": "Colorado/RS — IBGE 4305603",
    "430570": "Condor/RS — IBGE 4305702",
    "430580": "Constantina/RS — IBGE 4305801",
    "430583": "Coqueiro Baixo/RS — IBGE 4305835",
    "430585": "Coqueiros do Sul/RS — IBGE 4305850",
    "430587": "Coronel Barros/RS — IBGE 4305871",
    "430590": "Coronel Bicaco/RS — IBGE 4305900",
    "430593": "Coronel Pilar/RS — IBGE 4305934",
    "430595": "Cotiporã/RS — IBGE 4305959",
    "430597": "Coxilha/RS — IBGE 4305975",
    "430600": "Crissiumal/RS — IBGE 4306007",
    "430605": "Cristal/RS — IBGE 4306056",
    "430607": "Cristal do Sul/RS — IBGE 4306072",
    "430610": "Cruz Alta/RS — IBGE 4306106",
    "430613": "Cruzaltense/RS — IBGE 4306130",
    "430620": "Cruzeiro do Sul/RS — IBGE 4306205",
    "430630": "David Canabarro/RS — IBGE 4306304",
    "430632": "Derrubadas/RS — IBGE 4306320",
    "430635": "Dezesseis de Novembro/RS — IBGE 4306353",
    "430637": "Dilermando de Aguiar/RS — IBGE 4306379",
    "430640": "Dois Irmãos/RS — IBGE 4306403",
    "430642": "Dois Irmãos das Missões/RS — IBGE 4306429",
    "430645": "Dois Lajeados/RS — IBGE 4306452",
    "430650": "Dom Feliciano/RS — IBGE 4306502",
    "430655": "Dom Pedro de Alcântara/RS — IBGE 4306551",
    "430660": "Dom Pedrito/RS — IBGE 4306601",
    "430670": "Dona Francisca/RS — IBGE 4306700",
    "430673": "Doutor Maurício Cardoso/RS — IBGE 4306734",
    "430675": "Doutor Ricardo/RS — IBGE 4306759",
    "430676": "Eldorado do Sul/RS — IBGE 4306767",
    "430680": "Encantado/RS — IBGE 4306809",
    "430690": "Encruzilhada do Sul/RS — IBGE 4306908",
    "430692": "Engenho Velho/RS — IBGE 4306924",
    "430693": "Entre-Ijuís/RS — IBGE 4306932",
    "430695": "Entre Rios do Sul/RS — IBGE 4306957",
    "430697": "Erebango/RS — IBGE 4306973",
    "430700": "Erechim/RS — IBGE 4307005",
    "430705": "Ernestina/RS — IBGE 4307054",
    "430710": "Herval/RS — IBGE 4307104",
    "430720": "Erval Grande/RS — IBGE 4307203",
    "430730": "Erval Seco/RS — IBGE 4307302",
    "430740": "Esmeralda/RS — IBGE 4307401",
    "430745": "Esperança do Sul/RS — IBGE 4307450",
    "430750": "Espumoso/RS — IBGE 4307500",
    "430755": "Estação/RS — IBGE 4307559",
    "430760": "Estância Velha/RS — IBGE 4307609",
    "430770": "Esteio/RS — IBGE 4307708",
    "430780": "Estrela/RS — IBGE 4307807",
    "430781": "Estrela Velha/RS — IBGE 4307815",
    "430783": "Eugênio de Castro/RS — IBGE 4307831",
    "430786": "Fagundes Varela/RS — IBGE 4307864",
    "430790": "Farroupilha/RS — IBGE 4307906",
    "430800": "Faxinal do Soturno/RS — IBGE 4308003",
    "430805": "Faxinalzinho/RS — IBGE 4308052",
    "430807": "Fazenda Vilanova/RS — IBGE 4308078",
    "430810": "Feliz/RS — IBGE 4308102",
    "430820": "Flores da Cunha/RS — IBGE 4308201",
    "430825": "Floriano Peixoto/RS — IBGE 4308250",
    "430830": "Fontoura Xavier/RS — IBGE 4308300",
    "430840": "Formigueiro/RS — IBGE 4308409",
    "430843": "Forquetinha/RS — IBGE 4308433",
    "430845": "Fortaleza dos Valos/RS — IBGE 4308458",
    "430850": "Frederico Westphalen/RS — IBGE 4308508",
    "430860": "Garibaldi/RS — IBGE 4308607",
    "430865": "Garruchos/RS — IBGE 4308656",
    "430870": "Gaurama/RS — IBGE 4308706",
    "430880": "General Câmara/RS — IBGE 4308805",
    "430885": "Gentil/RS — IBGE 4308854",
    "430890": "Getúlio Vargas/RS — IBGE 4308904",
    "430900": "Giruá/RS — IBGE 4309001",
    "430905": "Glorinha/RS — IBGE 4309050",
    "430910": "Gramado/RS — IBGE 4309100",
    "430912": "Gramado dos Loureiros/RS — IBGE 4309126",
    "430915": "Gramado Xavier/RS — IBGE 4309159",
    "430920": "Gravataí/RS — IBGE 4309209",
    "430925": "Guabiju/RS — IBGE 4309258",
    "430930": "Guaíba/RS — IBGE 4309308",
    "430940": "Guaporé/RS — IBGE 4309407",
    "430950": "Guarani das Missões/RS — IBGE 4309506",
    "430955": "Harmonia/RS — IBGE 4309555",
    "430957": "Herveiras/RS — IBGE 4309571",
    "430960": "Horizontina/RS — IBGE 4309605",
    "430965": "Hulha Negra/RS — IBGE 4309654",
    "430970": "Humaitá/RS — IBGE 4309704",
    "430975": "Ibarama/RS — IBGE 4309753",
    "430980": "Ibiaçá/RS — IBGE 4309803",
    "430990": "Ibiraiaras/RS — IBGE 4309902",
    "430995": "Ibirapuitã/RS — IBGE 4309951",
    "431000": "Ibirubá/RS — IBGE 4310009",
    "431010": "Igrejinha/RS — IBGE 4310108",
    "431020": "Ijuí/RS — IBGE 4310207",
    "431030": "Ilópolis/RS — IBGE 4310306",
    "431033": "Imbé/RS — IBGE 4310330",
    "431036": "Imigrante/RS — IBGE 4310363",
    "431040": "Independência/RS — IBGE 4310405",
    "431041": "Inhacorá/RS — IBGE 4310413",
    "431043": "Ipê/RS — IBGE 4310439",
    "431046": "Ipiranga do Sul/RS — IBGE 4310462",
    "431050": "Iraí/RS — IBGE 4310504",
    "431053": "Itaara/RS — IBGE 4310538",
    "431055": "Itacurubi/RS — IBGE 4310553",
    "431057": "Itapuca/RS — IBGE 4310579",
    "431060": "Itaqui/RS — IBGE 4310603",
    "431065": "Itati/RS — IBGE 4310652",
    "431070": "Itatiba do Sul/RS — IBGE 4310702",
    "431075": "Ivorá/RS — IBGE 4310751",
    "431080": "Ivoti/RS — IBGE 4310801",
    "431085": "Jaboticaba/RS — IBGE 4310850",
    "431087": "Jacuizinho/RS — IBGE 4310876",
    "431090": "Jacutinga/RS — IBGE 4310900",
    "431100": "Jaguarão/RS — IBGE 4311007",
    "431110": "Jaguari/RS — IBGE 4311106",
    "431112": "Jaquirana/RS — IBGE 4311122",
    "431113": "Jari/RS — IBGE 4311130",
    "431115": "Jóia/RS — IBGE 4311155",
    "431120": "Júlio de Castilhos/RS — IBGE 4311205",
    "431123": "Lagoa Bonita do Sul/RS — IBGE 4311239",
    "431125": "Lagoão/RS — IBGE 4311254",
    "431127": "Lagoa dos Três Cantos/RS — IBGE 4311270",
    "431130": "Lagoa Vermelha/RS — IBGE 4311304",
    "431140": "Lajeado/RS — IBGE 4311403",
    "431142": "Lajeado do Bugre/RS — IBGE 4311429",
    "431150": "Lavras do Sul/RS — IBGE 4311502",
    "431160": "Liberato Salzano/RS — IBGE 4311601",
    "431162": "Lindolfo Collor/RS — IBGE 4311627",
    "431164": "Linha Nova/RS — IBGE 4311643",
    "431170": "Machadinho/RS — IBGE 4311700",
    "431171": "Maçambará/RS — IBGE 4311718",
    "431173": "Mampituba/RS — IBGE 4311734",
    "431175": "Manoel Viana/RS — IBGE 4311759",
    "431177": "Maquiné/RS — IBGE 4311775",
    "431179": "Maratá/RS — IBGE 4311791",
    "431180": "Marau/RS — IBGE 4311809",
    "431190": "Marcelino Ramos/RS — IBGE 4311908",
    "431198": "Mariana Pimentel/RS — IBGE 4311981",
    "431200": "Mariano Moro/RS — IBGE 4312005",
    "431205": "Marques de Souza/RS — IBGE 4312054",
    "431210": "Mata/RS — IBGE 4312104",
    "431213": "Mato Castelhano/RS — IBGE 4312138",
    "431215": "Mato Leitão/RS — IBGE 4312153",
    "431217": "Mato Queimado/RS — IBGE 4312179",
    "431220": "Maximiliano de Almeida/RS — IBGE 4312203",
    "431225": "Minas do Leão/RS — IBGE 4312252",
    "431230": "Miraguaí/RS — IBGE 4312302",
    "431235": "Montauri/RS — IBGE 4312351",
    "431237": "Monte Alegre dos Campos/RS — IBGE 4312377",
    "431238": "Monte Belo do Sul/RS — IBGE 4312385",
    "431240": "Montenegro/RS — IBGE 4312401",
    "431242": "Mormaço/RS — IBGE 4312427",
    "431244": "Morrinhos do Sul/RS — IBGE 4312443",
    "431245": "Morro Redondo/RS — IBGE 4312450",
    "431247": "Morro Reuter/RS — IBGE 4312476",
    "431250": "Mostardas/RS — IBGE 4312500",
    "431260": "Muçum/RS — IBGE 4312609",
    "431261": "Muitos Capões/RS — IBGE 4312617",
    "431262": "Muliterno/RS — IBGE 4312625",
    "431265": "Não-Me-Toque/RS — IBGE 4312658",
    "431267": "Nicolau Vergueiro/RS — IBGE 4312674",
    "431270": "Nonoai/RS — IBGE 4312708",
    "431275": "Nova Alvorada/RS — IBGE 4312757",
    "431280": "Nova Araçá/RS — IBGE 4312807",
    "431290": "Nova Bassano/RS — IBGE 4312906",
    "431295": "Nova Boa Vista/RS — IBGE 4312955",
    "431300": "Nova Bréscia/RS — IBGE 4313003",
    "431301": "Nova Candelária/RS — IBGE 4313011",
    "431303": "Nova Esperança do Sul/RS — IBGE 4313037",
    "431306": "Nova Hartz/RS — IBGE 4313060",
    "431308": "Nova Pádua/RS — IBGE 4313086",
    "431310": "Nova Palma/RS — IBGE 4313102",
    "431320": "Nova Petrópolis/RS — IBGE 4313201",
    "431330": "Nova Prata/RS — IBGE 4313300",
    "431333": "Nova Ramada/RS — IBGE 4313334",
    "431335": "Nova Roma do Sul/RS — IBGE 4313359",
    "431337": "Nova Santa Rita/RS — IBGE 4313375",
    "431339": "Novo Cabrais/RS — IBGE 4313391",
    "431340": "Novo Hamburgo/RS — IBGE 4313409",
    "431342": "Novo Machado/RS — IBGE 4313425",
    "431344": "Novo Tiradentes/RS — IBGE 4313441",
    "431346": "Novo Xingu/RS — IBGE 4313466",
    "431349": "Novo Barreiro/RS — IBGE 4313490",
    "431350": "Osório/RS — IBGE 4313508",
    "431360": "Paim Filho/RS — IBGE 4313607",
    "431365": "Palmares do Sul/RS — IBGE 4313656",
    "431370": "Palmeira das Missões/RS — IBGE 4313706",
    "431380": "Palmitinho/RS — IBGE 4313805",
    "431390": "Panambi/RS — IBGE 4313904",
    "431395": "Pantano Grande/RS — IBGE 4313953",
    "431400": "Paraí/RS — IBGE 4314001",
    "431402": "Paraíso do Sul/RS — IBGE 4314027",
    "431403": "Pareci Novo/RS — IBGE 4314035",
    "431405": "Parobé/RS — IBGE 4314050",
    "431406": "Passa Sete/RS — IBGE 4314068",
    "431407": "Passo do Sobrado/RS — IBGE 4314076",
    "431410": "Passo Fundo/RS — IBGE 4314100",
    "431413": "Paulo Bento/RS — IBGE 4314134",
    "431415": "Paverama/RS — IBGE 4314159",
    "431417": "Pedras Altas/RS — IBGE 4314175",
    "431420": "Pedro Osório/RS — IBGE 4314209",
    "431430": "Pejuçara/RS — IBGE 4314308",
    "431440": "Pelotas/RS — IBGE 4314407",
    "431442": "Picada Café/RS — IBGE 4314423",
    "431445": "Pinhal/RS — IBGE 4314456",
    "431446": "Pinhal da Serra/RS — IBGE 4314464",
    "431447": "Pinhal Grande/RS — IBGE 4314472",
    "431449": "Pinheirinho do Vale/RS — IBGE 4314498",
    "431450": "Pinheiro Machado/RS — IBGE 4314506",
    "431454": "Pinto Bandeira/RS — IBGE 4314548",
    "431455": "Pirapó/RS — IBGE 4314555",
    "431460": "Piratini/RS — IBGE 4314605",
    "431470": "Planalto/RS — IBGE 4314704",
    "431475": "Poço das Antas/RS — IBGE 4314753",
    "431477": "Pontão/RS — IBGE 4314779",
    "431478": "Ponte Preta/RS — IBGE 4314787",
    "431480": "Portão/RS — IBGE 4314803",
    "431490": "Porto Alegre/RS — IBGE 4314902",
    "431500": "Porto Lucena/RS — IBGE 4315008",
    "431505": "Porto Mauá/RS — IBGE 4315057",
    "431507": "Porto Vera Cruz/RS — IBGE 4315073",
    "431510": "Porto Xavier/RS — IBGE 4315107",
    "431513": "Pouso Novo/RS — IBGE 4315131",
    "431514": "Presidente Lucena/RS — IBGE 4315149",
    "431515": "Progresso/RS — IBGE 4315156",
    "431517": "Protásio Alves/RS — IBGE 4315172",
    "431520": "Putinga/RS — IBGE 4315206",
    "431530": "Quaraí/RS — IBGE 4315305",
    "431531": "Quatro Irmãos/RS — IBGE 4315313",
    "431532": "Quevedos/RS — IBGE 4315321",
    "431535": "Quinze de Novembro/RS — IBGE 4315354",
    "431540": "Redentora/RS — IBGE 4315404",
    "431545": "Relvado/RS — IBGE 4315453",
    "431550": "Restinga Sêca/RS — IBGE 4315503",
    "431555": "Rio dos Índios/RS — IBGE 4315552",
    "431560": "Rio Grande/RS — IBGE 4315602",
    "431570": "Rio Pardo/RS — IBGE 4315701",
    "431575": "Riozinho/RS — IBGE 4315750",
    "431580": "Roca Sales/RS — IBGE 4315800",
    "431590": "Rodeio Bonito/RS — IBGE 4315909",
    "431595": "Rolador/RS — IBGE 4315958",
    "431600": "Rolante/RS — IBGE 4316006",
    "431610": "Ronda Alta/RS — IBGE 4316105",
    "431620": "Rondinha/RS — IBGE 4316204",
    "431630": "Roque Gonzales/RS — IBGE 4316303",
    "431640": "Rosário do Sul/RS — IBGE 4316402",
    "431642": "Sagrada Família/RS — IBGE 4316428",
    "431643": "Saldanha Marinho/RS — IBGE 4316436",
    "431645": "Salto do Jacuí/RS — IBGE 4316451",
    "431647": "Salvador das Missões/RS — IBGE 4316477",
    "431650": "Salvador do Sul/RS — IBGE 4316501",
    "431660": "Sananduva/RS — IBGE 4316600",
    "431670": "Santa Bárbara do Sul/RS — IBGE 4316709",
    "431673": "Santa Cecília do Sul/RS — IBGE 4316733",
    "431675": "Santa Clara do Sul/RS — IBGE 4316758",
    "431680": "Santa Cruz do Sul/RS — IBGE 4316808",
    "431690": "Santa Maria/RS — IBGE 4316907",
    "431695": "Santa Maria do Herval/RS — IBGE 4316956",
    "431697": "Santa Margarida do Sul/RS — IBGE 4316972",
    "431700": "Santana da Boa Vista/RS — IBGE 4317004",
    "431710": "Sant'Ana do Livramento/RS — IBGE 4317103",
    "431720": "Santa Rosa/RS — IBGE 4317202",
    "431725": "Santa Tereza/RS — IBGE 4317251",
    "431730": "Santa Vitória do Palmar/RS — IBGE 4317301",
    "431740": "Santiago/RS — IBGE 4317400",
    "431750": "Santo Ângelo/RS — IBGE 4317509",
    "431755": "Santo Antônio do Palma/RS — IBGE 4317558",
    "431760": "Santo Antônio da Patrulha/RS — IBGE 4317608",
    "431770": "Santo Antônio das Missões/RS — IBGE 4317707",
    "431775": "Santo Antônio do Planalto/RS — IBGE 4317756",
    "431780": "Santo Augusto/RS — IBGE 4317806",
    "431790": "Santo Cristo/RS — IBGE 4317905",
    "431795": "Santo Expedito do Sul/RS — IBGE 4317954",
    "431800": "São Borja/RS — IBGE 4318002",
    "431805": "São Domingos do Sul/RS — IBGE 4318051",
    "431810": "São Francisco de Assis/RS — IBGE 4318101",
    "431820": "São Francisco de Paula/RS — IBGE 4318200",
    "431830": "São Gabriel/RS — IBGE 4318309",
    "431840": "São Jerônimo/RS — IBGE 4318408",
    "431842": "São João da Urtiga/RS — IBGE 4318424",
    "431843": "São João do Polêsine/RS — IBGE 4318432",
    "431844": "São Jorge/RS — IBGE 4318440",
    "431845": "São José das Missões/RS — IBGE 4318457",
    "431846": "São José do Herval/RS — IBGE 4318465",
    "431848": "São José do Hortêncio/RS — IBGE 4318481",
    "431849": "São José do Inhacorá/RS — IBGE 4318499",
    "431850": "São José do Norte/RS — IBGE 4318507",
    "431860": "São José do Ouro/RS — IBGE 4318606",
    "431861": "São José do Sul/RS — IBGE 4318614",
    "431862": "São José dos Ausentes/RS — IBGE 4318622",
    "431870": "São Leopoldo/RS — IBGE 4318705",
    "431880": "São Lourenço do Sul/RS — IBGE 4318804",
    "431890": "São Luiz Gonzaga/RS — IBGE 4318903",
    "431900": "São Marcos/RS — IBGE 4319000",
    "431910": "São Martinho/RS — IBGE 4319109",
    "431912": "São Martinho da Serra/RS — IBGE 4319125",
    "431915": "São Miguel das Missões/RS — IBGE 4319158",
    "431920": "São Nicolau/RS — IBGE 4319208",
    "431930": "São Paulo das Missões/RS — IBGE 4319307",
    "431935": "São Pedro da Serra/RS — IBGE 4319356",
    "431936": "São Pedro das Missões/RS — IBGE 4319364",
    "431937": "São Pedro do Butiá/RS — IBGE 4319372",
    "431940": "São Pedro do Sul/RS — IBGE 4319406",
    "431950": "São Sebastião do Caí/RS — IBGE 4319505",
    "431960": "São Sepé/RS — IBGE 4319604",
    "431970": "São Valentim/RS — IBGE 4319703",
    "431971": "São Valentim do Sul/RS — IBGE 4319711",
    "431973": "São Valério do Sul/RS — IBGE 4319737",
    "431975": "São Vendelino/RS — IBGE 4319752",
    "431980": "São Vicente do Sul/RS — IBGE 4319802",
    "431990": "Sapiranga/RS — IBGE 4319901",
    "432000": "Sapucaia do Sul/RS — IBGE 4320008",
    "432010": "Sarandi/RS — IBGE 4320107",
    "432020": "Seberi/RS — IBGE 4320206",
    "432023": "Sede Nova/RS — IBGE 4320230",
    "432026": "Segredo/RS — IBGE 4320263",
    "432030": "Selbach/RS — IBGE 4320305",
    "432032": "Senador Salgado Filho/RS — IBGE 4320321",
    "432035": "Sentinela do Sul/RS — IBGE 4320354",
    "432040": "Serafina Corrêa/RS — IBGE 4320404",
    "432045": "Sério/RS — IBGE 4320453",
    "432050": "Sertão/RS — IBGE 4320503",
    "432055": "Sertão Santana/RS — IBGE 4320552",
    "432057": "Sete de Setembro/RS — IBGE 4320578",
    "432060": "Severiano de Almeida/RS — IBGE 4320602",
    "432065": "Silveira Martins/RS — IBGE 4320651",
    "432067": "Sinimbu/RS — IBGE 4320677",
    "432070": "Sobradinho/RS — IBGE 4320701",
    "432080": "Soledade/RS — IBGE 4320800",
    "432085": "Tabaí/RS — IBGE 4320859",
    "432090": "Tapejara/RS — IBGE 4320909",
    "432100": "Tapera/RS — IBGE 4321006",
    "432110": "Tapes/RS — IBGE 4321105",
    "432120": "Taquara/RS — IBGE 4321204",
    "432130": "Taquari/RS — IBGE 4321303",
    "432132": "Taquaruçu do Sul/RS — IBGE 4321329",
    "432135": "Tavares/RS — IBGE 4321352",
    "432140": "Tenente Portela/RS — IBGE 4321402",
    "432143": "Terra de Areia/RS — IBGE 4321436",
    "432145": "Teutônia/RS — IBGE 4321451",
    "432146": "Tio Hugo/RS — IBGE 4321469",
    "432147": "Tiradentes do Sul/RS — IBGE 4321477",
    "432149": "Toropi/RS — IBGE 4321493",
    "432150": "Torres/RS — IBGE 4321501",
    "432160": "Tramandaí/RS — IBGE 4321600",
    "432162": "Travesseiro/RS — IBGE 4321626",
    "432163": "Três Arroios/RS — IBGE 4321634",
    "432166": "Três Cachoeiras/RS — IBGE 4321667",
    "432170": "Três Coroas/RS — IBGE 4321709",
    "432180": "Três de Maio/RS — IBGE 4321808",
    "432183": "Três Forquilhas/RS — IBGE 4321832",
    "432185": "Três Palmeiras/RS — IBGE 4321857",
    "432190": "Três Passos/RS — IBGE 4321907",
    "432195": "Trindade do Sul/RS — IBGE 4321956",
    "432200": "Triunfo/RS — IBGE 4322004",
    "432210": "Tucunduva/RS — IBGE 4322103",
    "432215": "Tunas/RS — IBGE 4322152",
    "432218": "Tupanci do Sul/RS — IBGE 4322186",
    "432220": "Tupanciretã/RS — IBGE 4322202",
    "432225": "Tupandi/RS — IBGE 4322251",
    "432230": "Tuparendi/RS — IBGE 4322301",
    "432232": "Turuçu/RS — IBGE 4322327",
    "432234": "Ubiretama/RS — IBGE 4322343",
    "432235": "União da Serra/RS — IBGE 4322350",
    "432237": "Unistalda/RS — IBGE 4322376",
    "432240": "Uruguaiana/RS — IBGE 4322400",
    "432250": "Vacaria/RS — IBGE 4322509",
    "432252": "Vale Verde/RS — IBGE 4322525",
    "432253": "Vale do Sol/RS — IBGE 4322533",
    "432254": "Vale Real/RS — IBGE 4322541",
    "432255": "Vanini/RS — IBGE 4322558",
    "432260": "Venâncio Aires/RS — IBGE 4322608",
    "432270": "Vera Cruz/RS — IBGE 4322707",
    "432280": "Veranópolis/RS — IBGE 4322806",
    "432285": "Vespasiano Corrêa/RS — IBGE 4322855",
    "432290": "Viadutos/RS — IBGE 4322905",
    "432300": "Viamão/RS — IBGE 4323002",
    "432310": "Vicente Dutra/RS — IBGE 4323101",
    "432320": "Victor Graeff/RS — IBGE 4323200",
    "432330": "Vila Flores/RS — IBGE 4323309",
    "432335": "Vila Lângaro/RS — IBGE 4323358",
    "432340": "Vila Maria/RS — IBGE 4323408",
    "432345": "Vila Nova do Sul/RS — IBGE 4323457",
    "432350": "Vista Alegre/RS — IBGE 4323507",
    "432360": "Vista Alegre do Prata/RS — IBGE 4323606",
    "432370": "Vista Gaúcha/RS — IBGE 4323705",
    "432375": "Vitória das Missões/RS — IBGE 4323754",
    "432377": "Westfália/RS — IBGE 4323770",
    "432380": "Xangri-lá/RS — IBGE 4323804",
    "500020": "Água Clara/MS — IBGE 5000203",
    "500025": "Alcinópolis/MS — IBGE 5000252",
    "500060": "Amambai/MS — IBGE 5000609",
    "500070": "Anastácio/MS — IBGE 5000708",
    "500080": "Anaurilândia/MS — IBGE 5000807",
    "500085": "Angélica/MS — IBGE 5000856",
    "500090": "Antônio João/MS — IBGE 5000906",
    "500100": "Aparecida do Taboado/MS — IBGE 5001003",
    "500110": "Aquidauana/MS — IBGE 5001102",
    "500124": "Aral Moreira/MS — IBGE 5001243",
    "500150": "Bandeirantes/MS — IBGE 5001508",
    "500190": "Bataguassu/MS — IBGE 5001904",
    "500200": "Batayporã/MS — IBGE 5002001",
    "500210": "Bela Vista/MS — IBGE 5002100",
    "500215": "Bodoquena/MS — IBGE 5002159",
    "500220": "Bonito/MS — IBGE 5002209",
    "500230": "Brasilândia/MS — IBGE 5002308",
    "500240": "Caarapó/MS — IBGE 5002407",
    "500260": "Camapuã/MS — IBGE 5002605",
    "500270": "Campo Grande/MS — IBGE 5002704",
    "500280": "Caracol/MS — IBGE 5002803",
    "500290": "Cassilândia/MS — IBGE 5002902",
    "500295": "Chapadão do Sul/MS — IBGE 5002951",
    "500310": "Corguinho/MS — IBGE 5003108",
    "500315": "Coronel Sapucaia/MS — IBGE 5003157",
    "500320": "Corumbá/MS — IBGE 5003207",
    "500325": "Costa Rica/MS — IBGE 5003256",
    "500330": "Coxim/MS — IBGE 5003306",
    "500345": "Deodápolis/MS — IBGE 5003454",
    "500348": "Dois Irmãos do Buriti/MS — IBGE 5003488",
    "500350": "Douradina/MS — IBGE 5003504",
    "500370": "Dourados/MS — IBGE 5003702",
    "500375": "Eldorado/MS — IBGE 5003751",
    "500380": "Fátima do Sul/MS — IBGE 5003801",
    "500390": "Figueirão/MS — IBGE 5003900",
    "500400": "Glória de Dourados/MS — IBGE 5004007",
    "500410": "Guia Lopes da Laguna/MS — IBGE 5004106",
    "500430": "Iguatemi/MS — IBGE 5004304",
    "500440": "Inocência/MS — IBGE 5004403",
    "500450": "Itaporã/MS — IBGE 5004502",
    "500460": "Itaquiraí/MS — IBGE 5004601",
    "500470": "Ivinhema/MS — IBGE 5004700",
    "500480": "Japorã/MS — IBGE 5004809",
    "500490": "Jaraguari/MS — IBGE 5004908",
    "500500": "Jardim/MS — IBGE 5005004",
    "500510": "Jateí/MS — IBGE 5005103",
    "500515": "Juti/MS — IBGE 5005152",
    "500520": "Ladário/MS — IBGE 5005202",
    "500525": "Laguna Carapã/MS — IBGE 5005251",
    "500540": "Maracaju/MS — IBGE 5005400",
    "500560": "Miranda/MS — IBGE 5005608",
    "500568": "Mundo Novo/MS — IBGE 5005681",
    "500570": "Naviraí/MS — IBGE 5005707",
    "500580": "Nioaque/MS — IBGE 5005806",
    "500600": "Nova Alvorada do Sul/MS — IBGE 5006002",
    "500620": "Nova Andradina/MS — IBGE 5006200",
    "500625": "Novo Horizonte do Sul/MS — IBGE 5006259",
    "500627": "Paraíso das Águas/MS — IBGE 5006275",
    "500630": "Paranaíba/MS — IBGE 5006309",
    "500635": "Paranhos/MS — IBGE 5006358",
    "500640": "Pedro Gomes/MS — IBGE 5006408",
    "500660": "Ponta Porã/MS — IBGE 5006606",
    "500690": "Porto Murtinho/MS — IBGE 5006903",
    "500710": "Ribas do Rio Pardo/MS — IBGE 5007109",
    "500720": "Rio Brilhante/MS — IBGE 5007208",
    "500730": "Rio Negro/MS — IBGE 5007307",
    "500740": "Rio Verde de Mato Grosso/MS — IBGE 5007406",
    "500750": "Rochedo/MS — IBGE 5007505",
    "500755": "Santa Rita do Pardo/MS — IBGE 5007554",
    "500769": "São Gabriel do Oeste/MS — IBGE 5007695",
    "500770": "Sete Quedas/MS — IBGE 5007703",
    "500780": "Selvíria/MS — IBGE 5007802",
    "500790": "Sidrolândia/MS — IBGE 5007901",
    "500793": "Sonora/MS — IBGE 5007935",
    "500795": "Tacuru/MS — IBGE 5007950",
    "500797": "Taquarussu/MS — IBGE 5007976",
    "500800": "Terenos/MS — IBGE 5008008",
    "500830": "Três Lagoas/MS — IBGE 5008305",
    "500840": "Vicentina/MS — IBGE 5008404",
    "510010": "Acorizal/MT — IBGE 5100102",
    "510020": "Água Boa/MT — IBGE 5100201",
    "510025": "Alta Floresta/MT — IBGE 5100250",
    "510030": "Alto Araguaia/MT — IBGE 5100300",
    "510035": "Alto Boa Vista/MT — IBGE 5100359",
    "510040": "Alto Garças/MT — IBGE 5100409",
    "510050": "Alto Paraguai/MT — IBGE 5100508",
    "510060": "Alto Taquari/MT — IBGE 5100607",
    "510080": "Apiacás/MT — IBGE 5100805",
    "510100": "Araguaiana/MT — IBGE 5101001",
    "510120": "Araguainha/MT — IBGE 5101209",
    "510125": "Araputanga/MT — IBGE 5101258",
    "510130": "Arenápolis/MT — IBGE 5101308",
    "510140": "Aripuanã/MT — IBGE 5101407",
    "510160": "Barão de Melgaço/MT — IBGE 5101605",
    "510170": "Barra do Bugres/MT — IBGE 5101704",
    "510180": "Barra do Garças/MT — IBGE 5101803",
    "510185": "Bom Jesus do Araguaia/MT — IBGE 5101852",
    "510190": "Brasnorte/MT — IBGE 5101902",
    "510250": "Cáceres/MT — IBGE 5102504",
    "510260": "Campinápolis/MT — IBGE 5102603",
    "510263": "Campo Novo do Parecis/MT — IBGE 5102637",
    "510267": "Campo Verde/MT — IBGE 5102678",
    "510268": "Campos de Júlio/MT — IBGE 5102686",
    "510269": "Canabrava do Norte/MT — IBGE 5102694",
    "510270": "Canarana/MT — IBGE 5102702",
    "510279": "Carlinda/MT — IBGE 5102793",
    "510285": "Castanheira/MT — IBGE 5102850",
    "510300": "Chapada dos Guimarães/MT — IBGE 5103007",
    "510305": "Cláudia/MT — IBGE 5103056",
    "510310": "Cocalinho/MT — IBGE 5103106",
    "510320": "Colíder/MT — IBGE 5103205",
    "510325": "Colniza/MT — IBGE 5103254",
    "510330": "Comodoro/MT — IBGE 5103304",
    "510335": "Confresa/MT — IBGE 5103353",
    "510336": "Conquista D'Oeste/MT — IBGE 5103361",
    "510337": "Cotriguaçu/MT — IBGE 5103379",
    "510340": "Cuiabá/MT — IBGE 5103403",
    "510343": "Curvelândia/MT — IBGE 5103437",
    "510345": "Denise/MT — IBGE 5103452",
    "510350": "Diamantino/MT — IBGE 5103502",
    "510360": "Dom Aquino/MT — IBGE 5103601",
    "510370": "Feliz Natal/MT — IBGE 5103700",
    "510380": "Figueirópolis D'Oeste/MT — IBGE 5103809",
    "510385": "Gaúcha do Norte/MT — IBGE 5103858",
    "510390": "General Carneiro/MT — IBGE 5103908",
    "510395": "Glória D'Oeste/MT — IBGE 5103957",
    "510410": "Guarantã do Norte/MT — IBGE 5104104",
    "510420": "Guiratinga/MT — IBGE 5104203",
    "510450": "Indiavaí/MT — IBGE 5104500",
    "510452": "Ipiranga do Norte/MT — IBGE 5104526",
    "510454": "Itanhangá/MT — IBGE 5104542",
    "510455": "Itaúba/MT — IBGE 5104559",
    "510460": "Itiquira/MT — IBGE 5104609",
    "510480": "Jaciara/MT — IBGE 5104807",
    "510490": "Jangada/MT — IBGE 5104906",
    "510500": "Jauru/MT — IBGE 5105002",
    "510510": "Juara/MT — IBGE 5105101",
    "510515": "Juína/MT — IBGE 5105150",
    "510517": "Juruena/MT — IBGE 5105176",
    "510520": "Juscimeira/MT — IBGE 5105200",
    "510523": "Lambari D'Oeste/MT — IBGE 5105234",
    "510525": "Lucas do Rio Verde/MT — IBGE 5105259",
    "510530": "Luciara/MT — IBGE 5105309",
    "510550": "Vila Bela da Santíssima Trindade/MT — IBGE 5105507",
    "510558": "Marcelândia/MT — IBGE 5105580",
    "510560": "Matupá/MT — IBGE 5105606",
    "510562": "Mirassol d'Oeste/MT — IBGE 5105622",
    "510590": "Nobres/MT — IBGE 5105903",
    "510600": "Nortelândia/MT — IBGE 5106000",
    "510610": "Nossa Senhora do Livramento/MT — IBGE 5106109",
    "510615": "Nova Bandeirantes/MT — IBGE 5106158",
    "510617": "Nova Nazaré/MT — IBGE 5106174",
    "510618": "Nova Lacerda/MT — IBGE 5106182",
    "510619": "Nova Santa Helena/MT — IBGE 5106190",
    "510620": "Nova Brasilândia/MT — IBGE 5106208",
    "510621": "Nova Canaã do Norte/MT — IBGE 5106216",
    "510622": "Nova Mutum/MT — IBGE 5106224",
    "510623": "Nova Olímpia/MT — IBGE 5106232",
    "510624": "Nova Ubiratã/MT — IBGE 5106240",
    "510625": "Nova Xavantina/MT — IBGE 5106257",
    "510626": "Novo Mundo/MT — IBGE 5106265",
    "510627": "Novo Horizonte do Norte/MT — IBGE 5106273",
    "510628": "Novo São Joaquim/MT — IBGE 5106281",
    "510629": "Paranaíta/MT — IBGE 5106299",
    "510630": "Paranatinga/MT — IBGE 5106307",
    "510631": "Novo Santo Antônio/MT — IBGE 5106315",
    "510637": "Pedra Preta/MT — IBGE 5106372",
    "510642": "Peixoto de Azevedo/MT — IBGE 5106422",
    "510645": "Planalto da Serra/MT — IBGE 5106455",
    "510650": "Poconé/MT — IBGE 5106505",
    "510665": "Pontal do Araguaia/MT — IBGE 5106653",
    "510670": "Ponte Branca/MT — IBGE 5106703",
    "510675": "Pontes e Lacerda/MT — IBGE 5106752",
    "510677": "Porto Alegre do Norte/MT — IBGE 5106778",
    "510680": "Porto dos Gaúchos/MT — IBGE 5106802",
    "510682": "Porto Esperidião/MT — IBGE 5106828",
    "510685": "Porto Estrela/MT — IBGE 5106851",
    "510700": "Poxoréu/MT — IBGE 5107008",
    "510704": "Primavera do Leste/MT — IBGE 5107040",
    "510706": "Querência/MT — IBGE 5107065",
    "510710": "São José dos Quatro Marcos/MT — IBGE 5107107",
    "510715": "Reserva do Cabaçal/MT — IBGE 5107156",
    "510718": "Ribeirão Cascalheira/MT — IBGE 5107180",
    "510719": "Ribeirãozinho/MT — IBGE 5107198",
    "510720": "Rio Branco/MT — IBGE 5107206",
    "510724": "Santa Carmem/MT — IBGE 5107248",
    "510726": "Santo Afonso/MT — IBGE 5107263",
    "510729": "São José do Povo/MT — IBGE 5107297",
    "510730": "São José do Rio Claro/MT — IBGE 5107305",
    "510735": "São José do Xingu/MT — IBGE 5107354",
    "510740": "São Pedro da Cipa/MT — IBGE 5107404",
    "510757": "Rondolândia/MT — IBGE 5107578",
    "510760": "Rondonópolis/MT — IBGE 5107602",
    "510770": "Rosário Oeste/MT — IBGE 5107701",
    "510774": "Santa Cruz do Xingu/MT — IBGE 5107743",
    "510775": "Salto do Céu/MT — IBGE 5107750",
    "510776": "Santa Rita do Trivelato/MT — IBGE 5107768",
    "510777": "Santa Terezinha/MT — IBGE 5107776",
    "510779": "Santo Antônio do Leste/MT — IBGE 5107792",
    "510780": "Santo Antônio do Leverger/MT — IBGE 5107800",
    "510785": "São Félix do Araguaia/MT — IBGE 5107859",
    "510787": "Sapezal/MT — IBGE 5107875",
    "510788": "Serra Nova Dourada/MT — IBGE 5107883",
    "510790": "Sinop/MT — IBGE 5107909",
    "510792": "Sorriso/MT — IBGE 5107925",
    "510794": "Tabaporã/MT — IBGE 5107941",
    "510795": "Tangará da Serra/MT — IBGE 5107958",
    "510800": "Tapurah/MT — IBGE 5108006",
    "510805": "Terra Nova do Norte/MT — IBGE 5108055",
    "510810": "Tesouro/MT — IBGE 5108105",
    "510820": "Torixoréu/MT — IBGE 5108204",
    "510830": "União do Sul/MT — IBGE 5108303",
    "510835": "Vale de São Domingos/MT — IBGE 5108352",
    "510840": "Várzea Grande/MT — IBGE 5108402",
    "510850": "Vera/MT — IBGE 5108501",
    "510860": "Vila Rica/MT — IBGE 5108600",
    "510880": "Nova Guarita/MT — IBGE 5108808",
    "510885": "Nova Marilândia/MT — IBGE 5108857",
    "510890": "Nova Maringá/MT — IBGE 5108907",
    "510895": "Nova Monte Verde/MT — IBGE 5108956",
    "520005": "Abadia de Goiás/GO — IBGE 5200050",
    "520010": "Abadiânia/GO — IBGE 5200100",
    "520013": "Acreúna/GO — IBGE 5200134",
    "520015": "Adelândia/GO — IBGE 5200159",
    "520017": "Água Fria de Goiás/GO — IBGE 5200175",
    "520020": "Água Limpa/GO — IBGE 5200209",
    "520025": "Águas Lindas de Goiás/GO — IBGE 5200258",
    "520030": "Alexânia/GO — IBGE 5200308",
    "520050": "Aloândia/GO — IBGE 5200506",
    "520055": "Alto Horizonte/GO — IBGE 5200555",
    "520060": "Alto Paraíso de Goiás/GO — IBGE 5200605",
    "520080": "Alvorada do Norte/GO — IBGE 5200803",
    "520082": "Amaralina/GO — IBGE 5200829",
    "520085": "Americano do Brasil/GO — IBGE 5200852",
    "520090": "Amorinópolis/GO — IBGE 5200902",
    "520110": "Anápolis/GO — IBGE 5201108",
    "520120": "Anhanguera/GO — IBGE 5201207",
    "520130": "Anicuns/GO — IBGE 5201306",
    "520140": "Aparecida de Goiânia/GO — IBGE 5201405",
    "520145": "Aparecida do Rio Doce/GO — IBGE 5201454",
    "520150": "Aporé/GO — IBGE 5201504",
    "520160": "Araçu/GO — IBGE 5201603",
    "520170": "Aragarças/GO — IBGE 5201702",
    "520180": "Aragoiânia/GO — IBGE 5201801",
    "520215": "Araguapaz/GO — IBGE 5202155",
    "520235": "Arenópolis/GO — IBGE 5202353",
    "520250": "Aruanã/GO — IBGE 5202502",
    "520260": "Aurilândia/GO — IBGE 5202601",
    "520280": "Avelinópolis/GO — IBGE 5202809",
    "520310": "Baliza/GO — IBGE 5203104",
    "520320": "Barro Alto/GO — IBGE 5203203",
    "520330": "Bela Vista de Goiás/GO — IBGE 5203302",
    "520340": "Bom Jardim de Goiás/GO — IBGE 5203401",
    "520350": "Bom Jesus de Goiás/GO — IBGE 5203500",
    "520355": "Bonfinópolis/GO — IBGE 5203559",
    "520357": "Bonópolis/GO — IBGE 5203575",
    "520360": "Brazabrantes/GO — IBGE 5203609",
    "520380": "Britânia/GO — IBGE 5203807",
    "520390": "Buriti Alegre/GO — IBGE 5203906",
    "520393": "Buriti de Goiás/GO — IBGE 5203939",
    "520396": "Buritinópolis/GO — IBGE 5203962",
    "520400": "Cabeceiras/GO — IBGE 5204003",
    "520410": "Cachoeira Alta/GO — IBGE 5204102",
    "520420": "Cachoeira de Goiás/GO — IBGE 5204201",
    "520425": "Cachoeira Dourada/GO — IBGE 5204250",
    "520430": "Caçu/GO — IBGE 5204300",
    "520440": "Caiapônia/GO — IBGE 5204409",
    "520450": "Caldas Novas/GO — IBGE 5204508",
    "520455": "Caldazinha/GO — IBGE 5204557",
    "520460": "Campestre de Goiás/GO — IBGE 5204607",
    "520465": "Campinaçu/GO — IBGE 5204656",
    "520470": "Campinorte/GO — IBGE 5204706",
    "520480": "Campo Alegre de Goiás/GO — IBGE 5204805",
    "520485": "Campo Limpo de Goiás/GO — IBGE 5204854",
    "520490": "Campos Belos/GO — IBGE 5204904",
    "520495": "Campos Verdes/GO — IBGE 5204953",
    "520500": "Carmo do Rio Verde/GO — IBGE 5205000",
    "520505": "Castelândia/GO — IBGE 5205059",
    "520510": "Catalão/GO — IBGE 5205109",
    "520520": "Caturaí/GO — IBGE 5205208",
    "520530": "Cavalcante/GO — IBGE 5205307",
    "520540": "Ceres/GO — IBGE 5205406",
    "520545": "Cezarina/GO — IBGE 5205455",
    "520547": "Chapadão do Céu/GO — IBGE 5205471",
    "520549": "Cidade Ocidental/GO — IBGE 5205497",
    "520551": "Cocalzinho de Goiás/GO — IBGE 5205513",
    "520552": "Colinas do Sul/GO — IBGE 5205521",
    "520570": "Córrego do Ouro/GO — IBGE 5205703",
    "520580": "Corumbá de Goiás/GO — IBGE 5205802",
    "520590": "Corumbaíba/GO — IBGE 5205901",
    "520620": "Cristalina/GO — IBGE 5206206",
    "520630": "Cristianópolis/GO — IBGE 5206305",
    "520640": "Crixás/GO — IBGE 5206404",
    "520650": "Cromínia/GO — IBGE 5206503",
    "520660": "Cumari/GO — IBGE 5206602",
    "520670": "Damianópolis/GO — IBGE 5206701",
    "520680": "Damolândia/GO — IBGE 5206800",
    "520690": "Davinópolis/GO — IBGE 5206909",
    "520710": "Diorama/GO — IBGE 5207105",
    "520725": "Doverlândia/GO — IBGE 5207253",
    "520735": "Edealina/GO — IBGE 5207352",
    "520740": "Edéia/GO — IBGE 5207402",
    "520750": "Estrela do Norte/GO — IBGE 5207501",
    "520753": "Faina/GO — IBGE 5207535",
    "520760": "Fazenda Nova/GO — IBGE 5207600",
    "520780": "Firminópolis/GO — IBGE 5207808",
    "520790": "Flores de Goiás/GO — IBGE 5207907",
    "520800": "Formosa/GO — IBGE 5208004",
    "520810": "Formoso/GO — IBGE 5208103",
    "520815": "Gameleira de Goiás/GO — IBGE 5208152",
    "520830": "Divinópolis de Goiás/GO — IBGE 5208301",
    "520840": "Goianápolis/GO — IBGE 5208400",
    "520850": "Goiandira/GO — IBGE 5208509",
    "520860": "Goianésia/GO — IBGE 5208608",
    "520870": "Goiânia/GO — IBGE 5208707",
    "520880": "Goianira/GO — IBGE 5208806",
    "520890": "Goiás/GO — IBGE 5208905",
    "520910": "Goiatuba/GO — IBGE 5209101",
    "520915": "Gouvelândia/GO — IBGE 5209150",
    "520920": "Guapó/GO — IBGE 5209200",
    "520929": "Guaraíta/GO — IBGE 5209291",
    "520940": "Guarani de Goiás/GO — IBGE 5209408",
    "520945": "Guarinos/GO — IBGE 5209457",
    "520960": "Heitoraí/GO — IBGE 5209606",
    "520970": "Hidrolândia/GO — IBGE 5209705",
    "520980": "Hidrolina/GO — IBGE 5209804",
    "520990": "Iaciara/GO — IBGE 5209903",
    "520993": "Inaciolândia/GO — IBGE 5209937",
    "520995": "Indiara/GO — IBGE 5209952",
    "521000": "Inhumas/GO — IBGE 5210000",
    "521010": "Ipameri/GO — IBGE 5210109",
    "521015": "Ipiranga de Goiás/GO — IBGE 5210158",
    "521020": "Iporá/GO — IBGE 5210208",
    "521030": "Israelândia/GO — IBGE 5210307",
    "521040": "Itaberaí/GO — IBGE 5210406",
    "521056": "Itaguari/GO — IBGE 5210562",
    "521060": "Itaguaru/GO — IBGE 5210604",
    "521080": "Itajá/GO — IBGE 5210802",
    "521090": "Itapaci/GO — IBGE 5210901",
    "521100": "Itapirapuã/GO — IBGE 5211008",
    "521120": "Itapuranga/GO — IBGE 5211206",
    "521130": "Itarumã/GO — IBGE 5211305",
    "521140": "Itauçu/GO — IBGE 5211404",
    "521150": "Itumbiara/GO — IBGE 5211503",
    "521160": "Ivolândia/GO — IBGE 5211602",
    "521170": "Jandaia/GO — IBGE 5211701",
    "521180": "Jaraguá/GO — IBGE 5211800",
    "521190": "Jataí/GO — IBGE 5211909",
    "521200": "Jaupaci/GO — IBGE 5212006",
    "521205": "Jesúpolis/GO — IBGE 5212055",
    "521210": "Joviânia/GO — IBGE 5212105",
    "521220": "Jussara/GO — IBGE 5212204",
    "521225": "Lagoa Santa/GO — IBGE 5212253",
    "521230": "Leopoldo de Bulhões/GO — IBGE 5212303",
    "521250": "Luziânia/GO — IBGE 5212501",
    "521260": "Mairipotaba/GO — IBGE 5212600",
    "521270": "Mambaí/GO — IBGE 5212709",
    "521280": "Mara Rosa/GO — IBGE 5212808",
    "521290": "Marzagão/GO — IBGE 5212907",
    "521295": "Matrinchã/GO — IBGE 5212956",
    "521300": "Maurilândia/GO — IBGE 5213004",
    "521305": "Mimoso de Goiás/GO — IBGE 5213053",
    "521308": "Minaçu/GO — IBGE 5213087",
    "521310": "Mineiros/GO — IBGE 5213103",
    "521340": "Moiporá/GO — IBGE 5213400",
    "521350": "Monte Alegre de Goiás/GO — IBGE 5213509",
    "521370": "Montes Claros de Goiás/GO — IBGE 5213707",
    "521375": "Montividiu/GO — IBGE 5213756",
    "521377": "Montividiu do Norte/GO — IBGE 5213772",
    "521380": "Morrinhos/GO — IBGE 5213806",
    "521385": "Morro Agudo de Goiás/GO — IBGE 5213855",
    "521390": "Mossâmedes/GO — IBGE 5213905",
    "521400": "Mozarlândia/GO — IBGE 5214002",
    "521405": "Mundo Novo/GO — IBGE 5214051",
    "521410": "Mutunópolis/GO — IBGE 5214101",
    "521440": "Nazário/GO — IBGE 5214408",
    "521450": "Nerópolis/GO — IBGE 5214507",
    "521460": "Niquelândia/GO — IBGE 5214606",
    "521470": "Nova América/GO — IBGE 5214705",
    "521480": "Nova Aurora/GO — IBGE 5214804",
    "521483": "Nova Crixás/GO — IBGE 5214838",
    "521486": "Nova Glória/GO — IBGE 5214861",
    "521487": "Nova Iguaçu de Goiás/GO — IBGE 5214879",
    "521490": "Nova Roma/GO — IBGE 5214903",
    "521500": "Nova Veneza/GO — IBGE 5215009",
    "521520": "Novo Brasil/GO — IBGE 5215207",
    "521523": "Novo Gama/GO — IBGE 5215231",
    "521525": "Novo Planalto/GO — IBGE 5215256",
    "521530": "Orizona/GO — IBGE 5215306",
    "521540": "Ouro Verde de Goiás/GO — IBGE 5215405",
    "521550": "Ouvidor/GO — IBGE 5215504",
    "521560": "Padre Bernardo/GO — IBGE 5215603",
    "521565": "Palestina de Goiás/GO — IBGE 5215652",
    "521570": "Palmeiras de Goiás/GO — IBGE 5215702",
    "521580": "Palmelo/GO — IBGE 5215801",
    "521590": "Palminópolis/GO — IBGE 5215900",
    "521600": "Panamá/GO — IBGE 5216007",
    "521630": "Paranaiguara/GO — IBGE 5216304",
    "521640": "Paraúna/GO — IBGE 5216403",
    "521645": "Perolândia/GO — IBGE 5216452",
    "521680": "Petrolina de Goiás/GO — IBGE 5216809",
    "521690": "Pilar de Goiás/GO — IBGE 5216908",
    "521710": "Piracanjuba/GO — IBGE 5217104",
    "521720": "Piranhas/GO — IBGE 5217203",
    "521730": "Pirenópolis/GO — IBGE 5217302",
    "521740": "Pires do Rio/GO — IBGE 5217401",
    "521760": "Planaltina/GO — IBGE 5217609",
    "521770": "Pontalina/GO — IBGE 5217708",
    "521800": "Porangatu/GO — IBGE 5218003",
    "521805": "Porteirão/GO — IBGE 5218052",
    "521810": "Portelândia/GO — IBGE 5218102",
    "521830": "Posse/GO — IBGE 5218300",
    "521839": "Professor Jamil/GO — IBGE 5218391",
    "521850": "Quirinópolis/GO — IBGE 5218508",
    "521860": "Rialma/GO — IBGE 5218607",
    "521870": "Rianápolis/GO — IBGE 5218706",
    "521878": "Rio Quente/GO — IBGE 5218789",
    "521880": "Rio Verde/GO — IBGE 5218805",
    "521890": "Rubiataba/GO — IBGE 5218904",
    "521900": "Sanclerlândia/GO — IBGE 5219001",
    "521910": "Santa Bárbara de Goiás/GO — IBGE 5219100",
    "521920": "Santa Cruz de Goiás/GO — IBGE 5219209",
    "521925": "Santa Fé de Goiás/GO — IBGE 5219258",
    "521930": "Santa Helena de Goiás/GO — IBGE 5219308",
    "521935": "Santa Isabel/GO — IBGE 5219357",
    "521940": "Santa Rita do Araguaia/GO — IBGE 5219407",
    "521945": "Santa Rita do Novo Destino/GO — IBGE 5219456",
    "521950": "Santa Rosa de Goiás/GO — IBGE 5219506",
    "521960": "Santa Tereza de Goiás/GO — IBGE 5219605",
    "521970": "Santa Terezinha de Goiás/GO — IBGE 5219704",
    "521971": "Santo Antônio da Barra/GO — IBGE 5219712",
    "521973": "Santo Antônio de Goiás/GO — IBGE 5219738",
    "521975": "Santo Antônio do Descoberto/GO — IBGE 5219753",
    "521980": "São Domingos/GO — IBGE 5219803",
    "521990": "São Francisco de Goiás/GO — IBGE 5219902",
    "522000": "São João d'Aliança/GO — IBGE 5220009",
    "522005": "São João da Paraúna/GO — IBGE 5220058",
    "522010": "São Luís de Montes Belos/GO — IBGE 5220108",
    "522015": "São Luiz do Norte/GO — IBGE 5220157",
    "522020": "São Miguel do Araguaia/GO — IBGE 5220207",
    "522026": "São Miguel do Passa Quatro/GO — IBGE 5220264",
    "522028": "São Patrício/GO — IBGE 5220280",
    "522040": "São Simão/GO — IBGE 5220405",
    "522045": "Senador Canedo/GO — IBGE 5220454",
    "522050": "Serranópolis/GO — IBGE 5220504",
    "522060": "Silvânia/GO — IBGE 5220603",
    "522068": "Simolândia/GO — IBGE 5220686",
    "522070": "Sítio d'Abadia/GO — IBGE 5220702",
    "522100": "Taquaral de Goiás/GO — IBGE 5221007",
    "522108": "Teresina de Goiás/GO — IBGE 5221080",
    "522119": "Terezópolis de Goiás/GO — IBGE 5221197",
    "522130": "Três Ranchos/GO — IBGE 5221304",
    "522140": "Trindade/GO — IBGE 5221403",
    "522145": "Trombas/GO — IBGE 5221452",
    "522150": "Turvânia/GO — IBGE 5221502",
    "522155": "Turvelândia/GO — IBGE 5221551",
    "522157": "Uirapuru/GO — IBGE 5221577",
    "522160": "Uruaçu/GO — IBGE 5221601",
    "522170": "Uruana/GO — IBGE 5221700",
    "522180": "Urutaí/GO — IBGE 5221809",
    "522185": "Valparaíso de Goiás/GO — IBGE 5221858",
    "522190": "Varjão/GO — IBGE 5221908",
    "522200": "Vianópolis/GO — IBGE 5222005",
    "522205": "Vicentinópolis/GO — IBGE 5222054",
    "522220": "Vila Boa/GO — IBGE 5222203",
    "522230": "Vila Propício/GO — IBGE 5222302",
    "530010": "Brasília/DF — IBGE 5300108",
}


@dataclass(frozen=True)
class SourceConfig:
    name: str
    title: str
    default_db: str
    default_table: str
    expected_period: str
    date_candidates: List[str]
    sex_candidates: List[str]
    age_candidates: List[str]
    age_unit_candidates: List[str]
    race_candidates: List[str]
    municipality_res_candidates: List[str]
    municipality_event_candidates: List[str]
    cid_candidates: List[str]
    field_notes: List[str]


SOURCE_CONFIG: Dict[str, SourceConfig] = {
    "SINAN": SourceConfig(
        name="SINAN",
        title="Notificações e investigação de casos",
        default_db="sinan_meningite_rio_estado.duckdb",
        default_table="sinan_meningite_rio_estado_data",
        expected_period="2007–2025",
        date_candidates=["DT_NOTIFIC", "DT_SIN_PRI", "DT_INVEST", "DT_ENCERRA", "DT_DIGITA"],
        sex_candidates=["CS_SEXO", "SEXO"],
        age_candidates=["NU_IDADE_N", "IDADE", "IDADE_ANOS", "IDADEANOS"],
        age_unit_candidates=[],
        race_candidates=["CS_RACA", "RACACOR"],
        municipality_res_candidates=["ID_MN_RESI", "CODMUNRES", "MUNIC_RES", "MUN_RES"],
        municipality_event_candidates=["ID_MUNICIP", "ID_MN_OCORR", "CODMUNOCOR", "MUNIC_MOV"],
        cid_candidates=["ID_AGRAVO", "CID10", "CID", "AGRAVO"],
        field_notes=[
            "No recorte enviado, o SINAN tende a ter ID_AGRAVO constante como G039.",
            "Para etiologia/forma clínica no SINAN, priorize CON_DIAGES; complemente com CLA_ME_BAC, CLA_ME_ASS, CLA_ME_ETI, CRITERIO e EVOLUCAO.",
        ],
    ),
    "SIM": SourceConfig(
        name="SIM",
        title="Óbitos e causas de morte",
        default_db="sim_do_rio_estado.duckdb",
        default_table="sim_do_rio_estado_data",
        expected_period="2007–2024",
        date_candidates=["DTOBITO", "DT_OBITO", "DTATESTADO", "DTNASC", "DT_NASC"],
        sex_candidates=["SEXO", "CS_SEXO"],
        age_candidates=["IDADE", "IDADEANOS", "IDADE_ANOS"],
        age_unit_candidates=[],
        race_candidates=["RACACOR", "CS_RACA"],
        municipality_res_candidates=["CODMUNRES", "MUNRES", "ID_MN_RESI"],
        municipality_event_candidates=["CODMUNOCOR", "MUNOCOR", "ID_MN_OCORR"],
        cid_candidates=["CAUSABAS", "CAUSABAS_O", "LINHAA", "LINHAB", "LINHAC", "LINHAD", "LINHAII", "ATESTADO", "CB_PRE"],
        field_notes=[
            "CAUSABAS/CAUSABAS_O representam causa básica; LINHAA–LINHAII e ATESTADO podem capturar menções associadas.",
            "Compare causa básica versus qualquer menção de CID de meningite para investigar concordância com o SINAN.",
        ],
    ),
    "CIHA": SourceConfig(
        name="CIHA",
        title="Atendimentos/internações informados à CIHA",
        default_db="ciha_rio_estado.duckdb",
        default_table="ciha_rio_estado_data",
        expected_period="2011–2025",
        date_candidates=["DT_ATEND", "DT_SAIDA", "DT_INTER", "DT_INTERNA", "DT_COMPET", "COMPET", "ANO_CMPT"],
        sex_candidates=["SEXO", "CS_SEXO"],
        age_candidates=["IDADE", "IDADE_ANOS", "IDADEANOS", "NU_IDADE_N"],
        age_unit_candidates=["COD_IDADE"],
        race_candidates=["RACACOR", "CS_RACA"],
        municipality_res_candidates=["MUNIC_RES", "CODMUNRES", "MUN_RES", "ID_MN_RESI"],
        municipality_event_candidates=["MUNIC_MOV", "CODMUNOCOR", "CODMUN", "MUN_MOV"],
        cid_candidates=["DIAG_PRINC", "DIAG_SECUN", "CIDPRI", "CID_PRINC", "CID", "DIAG"],
        field_notes=[
            "DIAG_PRINC é o campo CID-10 mais importante; DIAG_SECUN pode capturar diagnósticos secundários, mas costuma ter menor completude.",
            "CIHA deve ser lida como utilização de serviços/produção assistencial, não como incidência populacional.",
        ],
    ),
}


FIELD_GUIDE = {
    "SINAN": [
        ("DT_SIN_PRI", "data principal", "início dos sintomas"),
        ("DT_NOTIFIC", "data alternativa", "notificação"),
        ("CLASSI_FIN", "definição de caso", "confirmado, descartado, inconclusivo"),
        ("CON_DIAGES", "etiologia/forma", "conclusão diagnóstica específica"),
        ("EVOLUCAO", "desfecho", "alta, óbito por meningite, óbito por outra causa"),
        ("CRITERIO", "critério de confirmação", "cultura, PCR, clínico, quimiocitológico etc."),
        ("LAB_PUNCAO", "investigação", "punção laboratorial/lombar realizada"),
        ("LAB_LIQUOR", "exame", "quimiocitológico do líquor realizado"),
        ("LAB_GLICO / LAB_PROT / LAB_LEUCO", "parâmetros do LCR", "glicose, proteínas e leucócitos do exame quimiocitológico"),
        ("ID_AGRAVO", "CID bruto", "geralmente G039 neste recorte"),
    ],
    "SIM": [
        ("DTOBITO", "data principal", "data do óbito"),
        ("CAUSABAS", "CID principal", "causa básica codificada"),
        ("CAUSABAS_O", "CID complementar", "causa básica original/complementar"),
        ("LINHAA–LINHAII", "menções", "linhas da Declaração de Óbito"),
        ("IDADE", "idade", "idade codificada no padrão DATASUS"),
        ("CODMUNRES", "território", "município de residência"),
        ("CODMUNOCOR", "território", "município de ocorrência"),
        ("OBITOGRAV", "ciclo gravídico", "óbito durante gravidez/parto/aborto ou período pós-gestacional"),
        ("OBITOPUERP", "ciclo gravídico", "óbito no puerpério quando disponível"),
    ],
    "CIHA": [
        ("DT_ATEND", "data principal", "data de atendimento"),
        ("DT_SAIDA", "data alternativa", "data de saída"),
        ("DIAG_PRINC", "CID principal", "diagnóstico principal"),
        ("DIAG_SECUN", "CID complementar", "diagnóstico secundário"),
        ("MORTE", "desfecho administrativo", "morte no atendimento"),
        ("DIAS_PERM", "uso de serviço", "dias de permanência"),
        ("MODALIDADE", "uso de serviço", "hospitalar/ambulatorial"),
        ("IDADE + COD_IDADE", "idade", "idade e unidade da idade"),
    ],
}


# =============================================================================
# Utilitários SQL/texto
# =============================================================================


def normalize_name(text: object) -> str:
    text = unicodedata.normalize("NFKD", str(text or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return "".join(ch for ch in text.upper().strip() if ch.isalnum())


def qident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def qstr(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def clean_str_expr(col: str) -> str:
    return f"NULLIF(TRIM(CAST({qident(col)} AS VARCHAR)), '')"


def clean_code_expr(col: str, pad2: bool = False) -> str:
    raw = f"NULLIF(regexp_replace(UPPER(COALESCE({clean_str_expr(col)}, '')), '[^0-9A-Z]', '', 'g'), '')"
    if pad2:
        return f"CASE WHEN {raw} IS NULL THEN NULL WHEN LENGTH({raw}) = 1 THEN '0' || {raw} ELSE {raw} END"
    return raw


def case_from_mapping(code_sql: str, mapping: Dict[str, str], default: str) -> str:
    parts = [f"WHEN {qstr(k)} THEN {qstr(v)}" for k, v in mapping.items()]
    return f"CASE {code_sql} {' '.join(parts)} ELSE {qstr(default)} END"


def municipality_display_expr(col: str) -> str:
    raw = clean_str_expr(col)
    digits = f"regexp_replace(COALESCE({raw}, ''), '[^0-9]', '', 'g')"
    code6 = f"CASE WHEN LENGTH({digits}) >= 6 THEN SUBSTR({digits}, 1, 6) ELSE {digits} END"
    parts = [f"WHEN {qstr(k)} THEN {qstr(v)}" for k, v in BR_MUNICIPIOS_IBGE.items()]
    return f"""
    CASE {code6}
        {' '.join(parts)}
        ELSE CASE
            WHEN {raw} IS NULL THEN 'Sem informação'
            WHEN {digits} <> '' THEN 'Código informado: ' || {raw}
            ELSE {raw}
        END
    END
    """


def date_expr(col: str) -> str:
    txt = clean_str_expr(col)
    q = qident(col)
    return f"""
    CAST(COALESCE(
        TRY_CAST({q} AS DATE),
        CASE WHEN regexp_matches({txt}, '^\\d{{4}}-\\d{{2}}-\\d{{2}}$') THEN CAST(try_strptime({txt}, '%Y-%m-%d') AS DATE) END,
        CASE WHEN regexp_matches({txt}, '^\\d{{8}}$') AND SUBSTR({txt}, 1, 4) BETWEEN '1900' AND '2099' THEN CAST(try_strptime({txt}, '%Y%m%d') AS DATE) END,
        CASE WHEN regexp_matches({txt}, '^\\d{{8}}$') THEN CAST(try_strptime({txt}, '%d%m%Y') AS DATE) END,
        CASE WHEN regexp_matches({txt}, '^\\d{{6}}$') AND SUBSTR({txt}, 1, 4) BETWEEN '1900' AND '2099' THEN CAST(try_strptime({txt} || '01', '%Y%m%d') AS DATE) END,
        CASE WHEN regexp_matches({txt}, '^\\d{{4}}$') AND {txt} BETWEEN '1900' AND '2099' THEN CAST(try_strptime({txt} || '0101', '%Y%m%d') AS DATE) END,
        CASE WHEN regexp_matches({txt}, '^\\d{{2}}/\\d{{2}}/\\d{{4}}$') THEN CAST(try_strptime({txt}, '%d/%m/%Y') AS DATE) END,
        CASE WHEN regexp_matches({txt}, '^\\d{{2}}-\\d{{2}}-\\d{{4}}$') THEN CAST(try_strptime({txt}, '%d-%m-%Y') AS DATE) END
    ) AS DATE)
    """


def datasus_age_expr(col: str) -> str:
    txt = clean_str_expr(col)
    return f"""
    CASE
        WHEN {txt} IS NULL THEN NULL
        WHEN regexp_matches({txt}, '^\\d{{3,4}}$') AND SUBSTR({txt}, 1, 1) IN ('0','1','2','3','4','5') THEN
            CASE SUBSTR({txt}, 1, 1)
                WHEN '0' THEN TRY_CAST(SUBSTR({txt}, 2) AS DOUBLE) / (365.25 * 24)
                WHEN '1' THEN TRY_CAST(SUBSTR({txt}, 2) AS DOUBLE) / (365.25 * 24)
                WHEN '2' THEN TRY_CAST(SUBSTR({txt}, 2) AS DOUBLE) / 365.25
                WHEN '3' THEN TRY_CAST(SUBSTR({txt}, 2) AS DOUBLE) / 12
                WHEN '4' THEN TRY_CAST(SUBSTR({txt}, 2) AS DOUBLE)
                WHEN '5' THEN TRY_CAST(SUBSTR({txt}, 2) AS DOUBLE)
                ELSE NULL
            END
        WHEN regexp_matches({txt}, '^\\d{{1,3}}$') AND TRY_CAST({txt} AS DOUBLE) BETWEEN 0 AND 130 THEN TRY_CAST({txt} AS DOUBLE)
        ELSE NULL
    END
    """


def age_with_unit_expr(age_col: str, unit_col: str) -> str:
    age_txt = clean_str_expr(age_col)
    unit_txt = clean_str_expr(unit_col)
    age_num = f"TRY_CAST(REPLACE({age_txt}, ',', '.') AS DOUBLE)"
    return f"""
    CASE
        WHEN {age_txt} IS NULL OR {age_num} IS NULL THEN NULL
        WHEN {unit_txt} IN ('0', '1') THEN {age_num} / (365.25 * 24)
        WHEN {unit_txt} = '2' THEN {age_num} / 365.25
        WHEN {unit_txt} = '3' THEN {age_num} / 12
        WHEN {unit_txt} IN ('4', '5') THEN {age_num}
        ELSE {age_num}
    END
    """


def direct_age_expr(col: str) -> str:
    txt = clean_str_expr(col)
    return f"TRY_CAST(REPLACE({txt}, ',', '.') AS DOUBLE)"


def sex_expr(col: str) -> str:
    txt = clean_str_expr(col)
    return f"""
    CASE UPPER({txt})
        WHEN 'M' THEN 'Masculino'
        WHEN '1' THEN 'Masculino'
        WHEN 'F' THEN 'Feminino'
        WHEN '2' THEN 'Feminino'
        WHEN '3' THEN 'Feminino'
        WHEN 'I' THEN 'Ignorado/outro'
        WHEN '0' THEN 'Ignorado/outro'
        WHEN '9' THEN 'Ignorado/outro'
        ELSE COALESCE({txt}, 'Ignorado/outro')
    END
    """


def cid_extract_expr_for_col(col: str) -> str:
    txt = f"UPPER(COALESCE({clean_str_expr(col)}, ''))"
    raw = f"regexp_extract({txt}, '{CID_MENINGITE_REGEX}', 1)"
    return f"NULLIF(regexp_replace({raw}, '\\.', '', 'g'), '')"


def cid_extract_expr(cols: Sequence[str]) -> Optional[str]:
    exprs = [cid_extract_expr_for_col(c) for c in cols if c]
    if not exprs:
        return None
    return exprs[0] if len(exprs) == 1 else "COALESCE(" + ", ".join(exprs) + ")"


def cid_source_expr(cols: Sequence[str]) -> Optional[str]:
    tests = []
    for col in cols:
        cid = cid_extract_expr_for_col(col)
        tests.append(f"WHEN {cid} IS NOT NULL THEN {qstr(col)}")
    return None if not tests else "CASE " + " ".join(tests) + " ELSE NULL END"


def cid_group_expr(cid_sql: str) -> str:
    clauses = [f"WHEN {cid_sql} LIKE {qstr(rule['prefixo'] + '%')} THEN {qstr(rule['grupo'])}" for rule in CID_RULES]
    return f"CASE WHEN {cid_sql} IS NULL THEN 'Sem CID de meningite detectado' {' '.join(clauses)} ELSE 'Outro CID capturado' END"


def cid_type_expr(cid_sql: str) -> str:
    clauses = [f"WHEN {cid_sql} LIKE {qstr(rule['prefixo'] + '%')} THEN {qstr(rule['rotulo'])}" for rule in CID_RULES]
    return f"CASE WHEN {cid_sql} IS NULL THEN 'Sem CID de meningite detectado' {' '.join(clauses)} ELSE 'Outro CID capturado' END"


def _case_from_con_diages_attr(con_code_sql: str, attr: str, default: str) -> str:
    parts = [
        f"WHEN {qstr(code)} THEN {qstr(info[attr])}"
        for code, info in SINAN_CID10_FROM_CON_DIAGES.items()
    ]
    for code, label in SINAN_CID10_NOT_CONVERTED.items():
        parts.append(f"WHEN {qstr(code)} THEN {qstr(label)}")
    return f"CASE {con_code_sql} {' '.join(parts)} ELSE {qstr(default)} END"


def sinan_cid10_conversion_group_expr(con_code_sql: str) -> str:
    return _case_from_con_diages_attr(
        con_code_sql,
        "grupo",
        "Sem conversão — CON_DIAGES ausente ou não mapeado",
    )


def sinan_cid10_conversion_type_expr(con_code_sql: str) -> str:
    return _case_from_con_diages_attr(
        con_code_sql,
        "rotulo",
        "Sem conversão — CON_DIAGES ausente ou não mapeado",
    )


def sinan_cid10_conversion_include_expr(con_code_sql: str) -> str:
    mapped = ", ".join(qstr(code) for code in SINAN_CID10_FROM_CON_DIAGES)
    return f"CASE WHEN {con_code_sql} IN ({mapped}) THEN 'Sim' ELSE 'Não' END"


def age_band_expr(age_sql: str, width: int = 5) -> str:
    return f"FLOOR(({age_sql}) / {width}) * {width}"


def choose_candidate(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    if not columns:
        return None
    norm_to_col = {normalize_name(c): c for c in columns}
    for cand in candidates:
        if normalize_name(cand) in norm_to_col:
            return norm_to_col[normalize_name(cand)]
    cand_norms = [normalize_name(c) for c in candidates]
    for col in columns:
        ncol = normalize_name(col)
        if any(cn in ncol or ncol in cn for cn in cand_norms):
            return col
    return None


def choose_candidates(columns: Sequence[str], candidates: Sequence[str], max_items: int = 12) -> List[str]:
    result: List[str] = []
    norm_to_col = {normalize_name(c): c for c in columns}
    for cand in candidates:
        col = norm_to_col.get(normalize_name(cand))
        if col and col not in result:
            result.append(col)
    for col in columns:
        if len(result) >= max_items:
            break
        if col in result:
            continue
        ncol = normalize_name(col)
        if any(normalize_name(cand) in ncol or ncol in normalize_name(cand) for cand in candidates):
            result.append(col)
    return result[:max_items]


def sql_where(clauses: Iterable[Optional[str]]) -> str:
    valid = [c.strip() for c in clauses if c and c.strip()]
    return "" if not valid else "WHERE " + " AND ".join(f"({c})" for c in valid)


def append_clause(where_sql: str, clause: Optional[str]) -> str:
    if not clause:
        return where_sql
    if not where_sql:
        return f"WHERE ({clause})"
    return where_sql + f" AND ({clause})"


def after_where_keyword(where_sql: str) -> str:
    if not where_sql:
        return "1=1"
    return where_sql.replace("WHERE", "", 1).strip()


def pct_expr(numer: str, denom: str) -> str:
    return f"CASE WHEN {denom} > 0 THEN ROUND(100.0 * ({numer}) / ({denom}), 2) ELSE NULL END"


def first_existing_path(filename: str) -> str:
    candidates = [Path.cwd() / filename, Path("/mnt/data") / filename, Path(__file__).parent / filename]
    for p in candidates:
        if p.exists():
            return str(p)
    return filename


def safe_filename(text: str) -> str:
    n = normalize_name(text).lower()
    return n or "saida"


# =============================================================================
# Tabelas carregadas e consultas
# =============================================================================


@dataclass
class LoadedTable:
    source: str
    kind: str  # duckdb | parquet
    ref_sql: str
    db_path: Optional[str] = None
    table_name: Optional[str] = None
    parquet_paths: Optional[List[str]] = None
    label: str = ""


@st.cache_data(show_spinner=False)
def list_duckdb_tables(path: str) -> List[str]:
    con = duckdb.connect(path, read_only=True)
    try:
        return [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    finally:
        con.close()


def parquet_ref(paths: Sequence[str]) -> str:
    quoted = ", ".join(qstr(p) for p in paths)
    return f"read_parquet([{quoted}], union_by_name=true)"


def materialize_upload(upload, namespace: str) -> str:
    data = upload.getbuffer().tobytes()
    digest = hashlib.sha1(data).hexdigest()[:16]
    suffix = Path(upload.name).suffix or ".dat"
    clean_name = safe_filename(Path(upload.name).stem)
    out = Path(tempfile.gettempdir()) / f"meningite_{namespace}_{clean_name}_{digest}{suffix}"
    if not out.exists():
        out.write_bytes(data)
    return str(out)


def run_query(table: LoadedTable, sql: str) -> pd.DataFrame:
    if table.kind == "duckdb":
        con = duckdb.connect(table.db_path, read_only=True)
    else:
        con = duckdb.connect(database=":memory:")
    try:
        return con.execute(sql).df()
    finally:
        con.close()


def schema_df(table: LoadedTable) -> pd.DataFrame:
    sql = f"DESCRIBE SELECT * FROM {table.ref_sql}"
    df = run_query(table, sql)
    if "column_name" in df.columns:
        keep = [c for c in ["column_name", "column_type", "null"] if c in df.columns]
        return df[keep].rename(columns={"column_name": "coluna", "column_type": "tipo", "null": "nulo"})
    return df


def count_rows(table: LoadedTable, where_sql: str = "") -> int:
    df = run_query(table, f"SELECT COUNT(*) AS n FROM {table.ref_sql} {where_sql}")
    return int(df.iloc[0, 0]) if not df.empty else 0


def top_values(table: LoadedTable, expr: str, where_sql: str = "", limit: int = 40) -> List[str]:
    if not expr:
        return []
    clause = append_clause(where_sql, f"{expr} IS NOT NULL")
    sql = f"""
        SELECT {expr} AS valor, COUNT(*) AS n
        FROM {table.ref_sql}
        {clause}
        GROUP BY 1
        ORDER BY n DESC, valor
        LIMIT {int(limit)}
    """
    try:
        df = run_query(table, sql)
    except Exception:
        return []
    return [str(x) for x in df["valor"].dropna().tolist()]


def minmax_date(table: LoadedTable, dt_sql: Optional[str], where_sql: str = "") -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    if not dt_sql:
        return None
    clause = append_clause(where_sql, f"{dt_sql} IS NOT NULL")
    sql = f"SELECT MIN({dt_sql}) AS dt_min, MAX({dt_sql}) AS dt_max FROM {table.ref_sql} {clause}"
    df = run_query(table, sql)
    if df.empty or pd.isna(df.iloc[0, 0]) or pd.isna(df.iloc[0, 1]):
        return None
    return pd.to_datetime(df.iloc[0, 0]), pd.to_datetime(df.iloc[0, 1])


# =============================================================================
# Seleção de colunas e expressões por fonte
# =============================================================================


@dataclass
class ColumnSelection:
    date_col: Optional[str]
    sex_col: Optional[str]
    age_col: Optional[str]
    age_unit_col: Optional[str]
    race_col: Optional[str]
    municipality_res_col: Optional[str]
    municipality_event_col: Optional[str]
    cid_cols: List[str]
    age_mode: str
    # SINAN
    classi_fin_col: Optional[str] = None
    con_diages_col: Optional[str] = None
    evolucao_col: Optional[str] = None
    criterio_col: Optional[str] = None
    lab_puncao_col: Optional[str] = None
    lab_liquor_col: Optional[str] = None
    lab_hema_col: Optional[str] = None
    lab_neutro_col: Optional[str] = None
    lab_glico_col: Optional[str] = None
    lab_leuco_col: Optional[str] = None
    lab_eosi_col: Optional[str] = None
    lab_prot_col: Optional[str] = None
    lab_mono_col: Optional[str] = None
    lab_linfo_col: Optional[str] = None
    lab_clor_col: Optional[str] = None
    ate_hospit_col: Optional[str] = None
    dt_encerramento_col: Optional[str] = None
    dt_notificacao_col: Optional[str] = None
    # SIM
    causabas_col: Optional[str] = None
    causabas_o_col: Optional[str] = None
    obitograv_col: Optional[str] = None
    obitopuerp_col: Optional[str] = None
    # CIHA
    diag_princ_col: Optional[str] = None
    diag_secun_col: Optional[str] = None
    morte_col: Optional[str] = None
    dias_perm_col: Optional[str] = None
    modalidade_col: Optional[str] = None


def default_selections(source: str, columns: Sequence[str]) -> ColumnSelection:
    cfg = SOURCE_CONFIG[source]
    date_col = choose_candidate(columns, cfg.date_candidates)
    sex_col = choose_candidate(columns, cfg.sex_candidates)
    age_col = choose_candidate(columns, cfg.age_candidates)
    age_unit_col = choose_candidate(columns, cfg.age_unit_candidates)
    race_col = choose_candidate(columns, cfg.race_candidates)
    mun_res_col = choose_candidate(columns, cfg.municipality_res_candidates)
    mun_event_col = choose_candidate(columns, cfg.municipality_event_candidates)
    cid_cols = choose_candidates(columns, cfg.cid_candidates, max_items=10)
    age_mode = "Automático"
    if source == "CIHA" and age_col and age_unit_col:
        age_mode = "DATASUS com coluna de unidade"
    elif source in {"SINAN", "SIM"} and age_col:
        age_mode = "DATASUS codificada"

    sel = ColumnSelection(
        date_col=date_col,
        sex_col=sex_col,
        age_col=age_col,
        age_unit_col=age_unit_col,
        race_col=race_col,
        municipality_res_col=mun_res_col,
        municipality_event_col=mun_event_col,
        cid_cols=cid_cols,
        age_mode=age_mode,
    )
    if source == "SINAN":
        sel.classi_fin_col = choose_candidate(columns, ["CLASSI_FIN"])
        sel.con_diages_col = choose_candidate(columns, ["CON_DIAGES"])
        sel.evolucao_col = choose_candidate(columns, ["EVOLUCAO"])
        sel.criterio_col = choose_candidate(columns, ["CRITERIO"])
        sel.lab_puncao_col = choose_candidate(columns, ["LAB_PUNCAO", "PUNCAO", "PUNCAO_LCR", "PUNCAO_LOMBAR"])
        sel.lab_liquor_col = choose_candidate(columns, ["LAB_LIQUOR", "LIQUOR", "QUIMIOCITOLOGICO", "EXAME_QUIMIOCITOLOGICO", "EXAME_LIQUOR"])
        sel.lab_hema_col = choose_candidate(columns, ["LAB_HEMA", "HEMACIAS", "NU_HEMACIAS"])
        sel.lab_neutro_col = choose_candidate(columns, ["LAB_NEUTRO", "NEUTROFILOS", "NU_NEUTROFILO", "NU_NEUTROFILOS"])
        sel.lab_glico_col = choose_candidate(columns, ["LAB_GLICO", "GLICOSE", "NU_GLICOSE"])
        sel.lab_leuco_col = choose_candidate(columns, ["LAB_LEUCO", "LEUCOCITOS", "NU_LEUCOCITO", "NU_LEUCOCITOS"])
        sel.lab_eosi_col = choose_candidate(columns, ["LAB_EOSI", "EOSINOFILOS", "NU_EOSINOFILO", "NU_EOSINOFILOS"])
        sel.lab_prot_col = choose_candidate(columns, ["LAB_PROT", "PROTEINAS", "PROTEINA", "NU_PROTEINA", "NU_PROTEINAS"])
        sel.lab_mono_col = choose_candidate(columns, ["LAB_MONO", "MONOCITOS", "NU_MONOCITO", "NU_MONOCITOS"])
        sel.lab_linfo_col = choose_candidate(columns, ["LAB_LINFO", "LINFOCITOS", "NU_LINFOCITO", "NU_LINFOCITOS"])
        sel.lab_clor_col = choose_candidate(columns, ["LAB_CLOR", "CLORETO", "CLORETOS", "NU_CLORETO", "NU_CLORETOS"])
        sel.ate_hospit_col = choose_candidate(columns, ["ATE_HOSPIT"])
        sel.dt_encerramento_col = choose_candidate(columns, ["DT_ENCERRA"])
        sel.dt_notificacao_col = choose_candidate(columns, ["DT_NOTIFIC"])
    elif source == "SIM":
        sel.causabas_col = choose_candidate(columns, ["CAUSABAS"])
        sel.causabas_o_col = choose_candidate(columns, ["CAUSABAS_O"])
        sel.obitograv_col = choose_candidate(columns, ["OBITOGRAV", "OBITO_GRAV", "OBITO_GRAVIDEZ", "GRAVIDEZ"])
        sel.obitopuerp_col = choose_candidate(columns, ["OBITOPUERP", "OBITO_PUERP", "PUERPERIO", "PUERP"])
    elif source == "CIHA":
        sel.diag_princ_col = choose_candidate(columns, ["DIAG_PRINC"])
        sel.diag_secun_col = choose_candidate(columns, ["DIAG_SECUN"])
        sel.morte_col = choose_candidate(columns, ["MORTE"])
        sel.dias_perm_col = choose_candidate(columns, ["DIAS_PERM"])
        sel.modalidade_col = choose_candidate(columns, ["MODALIDADE"])
    return sel


def build_age_sql(sel: ColumnSelection) -> Optional[str]:
    if not sel.age_col:
        return None
    if sel.age_mode == "Anos diretos":
        return direct_age_expr(sel.age_col)
    if sel.age_mode == "DATASUS com coluna de unidade" and sel.age_unit_col:
        return age_with_unit_expr(sel.age_col, sel.age_unit_col)
    if sel.age_mode == "DATASUS codificada":
        return datasus_age_expr(sel.age_col)
    if sel.age_unit_col:
        return f"COALESCE({age_with_unit_expr(sel.age_col, sel.age_unit_col)}, {datasus_age_expr(sel.age_col)}, {direct_age_expr(sel.age_col)})"
    return f"COALESCE({datasus_age_expr(sel.age_col)}, {direct_age_expr(sel.age_col)})"


def build_expressions(source: str, sel: ColumnSelection) -> Dict[str, Optional[str]]:
    exprs: Dict[str, Optional[str]] = {
        "dt": date_expr(sel.date_col) if sel.date_col else None,
        "sex": sex_expr(sel.sex_col) if sel.sex_col else None,
        "age": build_age_sql(sel),
        "race": case_from_mapping(clean_code_expr(sel.race_col), RACA_COR, "Sem informação/ignorado") if sel.race_col else None,
        "mun_res": clean_str_expr(sel.municipality_res_col) if sel.municipality_res_col else None,
        "mun_event": clean_str_expr(sel.municipality_event_col) if sel.municipality_event_col else None,
        "mun_res_label": municipality_display_expr(sel.municipality_res_col) if sel.municipality_res_col else None,
        "mun_event_label": municipality_display_expr(sel.municipality_event_col) if sel.municipality_event_col else None,
        "cid": cid_extract_expr(sel.cid_cols),
        "cid_source": cid_source_expr(sel.cid_cols),
    }
    if exprs["cid"]:
        exprs["cid_group"] = cid_group_expr(exprs["cid"])
        exprs["cid_type"] = cid_type_expr(exprs["cid"])
    else:
        exprs["cid_group"] = None
        exprs["cid_type"] = None

    if source == "SINAN":
        exprs["classi_code"] = clean_code_expr(sel.classi_fin_col) if sel.classi_fin_col else None
        exprs["classi_label"] = case_from_mapping(exprs["classi_code"], SINAN_CLASSI_FIN, "Sem classificação/ignorado") if exprs["classi_code"] else None
        exprs["evol_code"] = clean_code_expr(sel.evolucao_col) if sel.evolucao_col else None
        exprs["evol_label"] = case_from_mapping(exprs["evol_code"], SINAN_EVOLUCAO, "Sem evolução/ignorado") if exprs["evol_code"] else None
        exprs["con_code"] = clean_code_expr(sel.con_diages_col, pad2=True) if sel.con_diages_col else None
        exprs["con_label"] = case_from_mapping(exprs["con_code"], SINAN_CON_DIAGES, "Sem conclusão diagnóstica/ignorado") if exprs["con_code"] else None
        exprs["con_group"] = case_from_mapping(exprs["con_code"], SINAN_CON_GROUP, "Sem conclusão diagnóstica/ignorado") if exprs["con_code"] else None
        if exprs["con_code"]:
            exprs["sinan_cid10_conversion_group"] = sinan_cid10_conversion_group_expr(exprs["con_code"])
            exprs["sinan_cid10_conversion_type"] = sinan_cid10_conversion_type_expr(exprs["con_code"])
            exprs["sinan_cid10_conversion_include"] = sinan_cid10_conversion_include_expr(exprs["con_code"])
        else:
            exprs["sinan_cid10_conversion_group"] = None
            exprs["sinan_cid10_conversion_type"] = None
            exprs["sinan_cid10_conversion_include"] = None
        exprs["criterio_code"] = clean_code_expr(sel.criterio_col) if sel.criterio_col else None
        exprs["criterio_label"] = case_from_mapping(exprs["criterio_code"], SINAN_CRITERIO, "Sem critério/ignorado") if exprs["criterio_code"] else None
        exprs["puncao_label"] = case_from_mapping(clean_code_expr(sel.lab_puncao_col), YES_NO_IGN, "Sem informação") if sel.lab_puncao_col else None
        exprs["quimio_label"] = case_from_mapping(clean_code_expr(sel.lab_liquor_col), YES_NO_IGN, "Sem informação") if sel.lab_liquor_col else None
        exprs["lab_hema"] = numeric_expr(sel.lab_hema_col) if sel.lab_hema_col else None
        exprs["lab_neutro"] = numeric_expr(sel.lab_neutro_col) if sel.lab_neutro_col else None
        exprs["lab_glico"] = numeric_expr(sel.lab_glico_col) if sel.lab_glico_col else None
        exprs["lab_leuco"] = numeric_expr(sel.lab_leuco_col) if sel.lab_leuco_col else None
        exprs["lab_eosi"] = numeric_expr(sel.lab_eosi_col) if sel.lab_eosi_col else None
        exprs["lab_prot"] = numeric_expr(sel.lab_prot_col) if sel.lab_prot_col else None
        exprs["lab_mono"] = numeric_expr(sel.lab_mono_col) if sel.lab_mono_col else None
        exprs["lab_linfo"] = numeric_expr(sel.lab_linfo_col) if sel.lab_linfo_col else None
        exprs["lab_clor"] = numeric_expr(sel.lab_clor_col) if sel.lab_clor_col else None
        exprs["hospital_label"] = case_from_mapping(clean_code_expr(sel.ate_hospit_col), YES_NO_IGN, "Sem informação") if sel.ate_hospit_col else None
        exprs["dt_encerramento"] = date_expr(sel.dt_encerramento_col) if sel.dt_encerramento_col else None
        exprs["dt_notificacao"] = date_expr(sel.dt_notificacao_col) if sel.dt_notificacao_col else None
    elif source == "SIM":
        exprs["causabas_cid"] = cid_extract_expr([sel.causabas_col] if sel.causabas_col else [])
        exprs["causabas_o_cid"] = cid_extract_expr([sel.causabas_o_col] if sel.causabas_o_col else [])
        exprs["causabas_group"] = cid_group_expr(exprs["causabas_cid"]) if exprs["causabas_cid"] else None
        exprs["causabas_type"] = cid_type_expr(exprs["causabas_cid"]) if exprs["causabas_cid"] else None
        exprs["lococor_label"] = case_from_mapping(clean_code_expr("LOCOCOR"), SIM_LOCOCOR, "Sem informação/ignorado") if "LOCOCOR" in [sel.municipality_event_col, sel.municipality_res_col] else None
        exprs["obitograv_code"] = clean_code_expr(sel.obitograv_col) if sel.obitograv_col else None
        exprs["obitograv_label"] = case_from_mapping(exprs["obitograv_code"], SIM_OBITOGRAV, "Sem informação/ignorado") if exprs.get("obitograv_code") else None
        exprs["obitopuerp_code"] = clean_code_expr(sel.obitopuerp_col) if sel.obitopuerp_col else None
        exprs["obitopuerp_label"] = case_from_mapping(exprs["obitopuerp_code"], SIM_OBITOPUERP, "Sem informação/ignorado") if exprs.get("obitopuerp_code") else None
    elif source == "CIHA":
        exprs["diag_princ_cid"] = cid_extract_expr([sel.diag_princ_col] if sel.diag_princ_col else [])
        exprs["diag_secun_cid"] = cid_extract_expr([sel.diag_secun_col] if sel.diag_secun_col else [])
        exprs["diag_princ_type"] = cid_type_expr(exprs["diag_princ_cid"]) if exprs["diag_princ_cid"] else None
        exprs["morte_code"] = clean_code_expr(sel.morte_col) if sel.morte_col else None
        exprs["dias_perm"] = direct_age_expr(sel.dias_perm_col) if sel.dias_perm_col else None
        exprs["modalidade_label"] = case_from_mapping(clean_code_expr(sel.modalidade_col, pad2=True), CIHA_MODALIDADE, "Sem modalidade/ignorado") if sel.modalidade_col else None
    return exprs


# =============================================================================
# Queries analíticas
# =============================================================================


def query_timeseries(table: LoadedTable, dt_sql: str, where_sql: str, freq: str, category_sql: Optional[str] = None) -> pd.DataFrame:
    if category_sql:
        sql = f"""
            WITH base AS (
                SELECT {dt_sql} AS dt, {category_sql} AS categoria
                FROM {table.ref_sql}
                {where_sql}
            )
            SELECT date_trunc({qstr(freq)}, dt) AS periodo, categoria, COUNT(*) AS n
            FROM base
            WHERE dt IS NOT NULL AND categoria IS NOT NULL
            GROUP BY 1, 2
            ORDER BY 1, 2
        """
    else:
        sql = f"""
            WITH base AS (
                SELECT {dt_sql} AS dt
                FROM {table.ref_sql}
                {where_sql}
            )
            SELECT date_trunc({qstr(freq)}, dt) AS periodo, COUNT(*) AS n
            FROM base
            WHERE dt IS NOT NULL
            GROUP BY 1
            ORDER BY 1
        """
    return run_query(table, sql)


def query_heatmap(table: LoadedTable, dt_sql: str, where_sql: str) -> pd.DataFrame:
    sql = f"""
        WITH base AS (
            SELECT {dt_sql} AS dt
            FROM {table.ref_sql}
            {where_sql}
        )
        SELECT EXTRACT(YEAR FROM dt) AS ano, EXTRACT(MONTH FROM dt) AS mes, COUNT(*) AS n
        FROM base
        WHERE dt IS NOT NULL
        GROUP BY 1, 2
        ORDER BY 1, 2
    """
    return run_query(table, sql)


def query_category(table: LoadedTable, category_sql: str, where_sql: str, top_n: int = 30) -> pd.DataFrame:
    clause = append_clause(where_sql, f"{category_sql} IS NOT NULL")
    sql = f"""
        SELECT {category_sql} AS categoria, COUNT(*) AS n
        FROM {table.ref_sql}
        {clause}
        GROUP BY 1
        ORDER BY n DESC, categoria
        LIMIT {int(top_n)}
    """
    df = run_query(table, sql)
    if not df.empty:
        df["pct"] = (df["n"] / df["n"].sum() * 100).round(2)
    return df


def query_category_top_with_outros(table: LoadedTable, category_sql: str, where_sql: str, top_n: int = 15, outros_label: str = "Outros municípios") -> pd.DataFrame:
    sql = f"""
        WITH counts AS (
            SELECT COALESCE({category_sql}, 'Sem informação') AS categoria, COUNT(*) AS n
            FROM {table.ref_sql}
            {where_sql}
            GROUP BY 1
        ), ranked AS (
            SELECT categoria, n,
                   ROW_NUMBER() OVER (ORDER BY n DESC, categoria) AS rn,
                   SUM(n) OVER () AS denominador
            FROM counts
        ), grouped AS (
            SELECT CASE WHEN rn <= {int(top_n)} THEN categoria ELSE {qstr(outros_label)} END AS categoria,
                   SUM(n) AS n,
                   MAX(denominador) AS denominador,
                   CASE WHEN MIN(rn) <= {int(top_n)} THEN MIN(rn) ELSE 999999 END AS ordem
            FROM ranked
            GROUP BY 1
        )
        SELECT categoria, n, denominador,
               CASE WHEN denominador > 0 THEN ROUND(100.0 * n / denominador, 2) ELSE NULL END AS pct
        FROM grouped
        ORDER BY ordem, n DESC, categoria
    """
    return run_query(table, sql)


def query_yearly_category(table: LoadedTable, dt_sql: str, category_sql: str, where_sql: str) -> pd.DataFrame:
    sql = f"""
        WITH base AS (
            SELECT {dt_sql} AS dt, COALESCE({category_sql}, 'Sem informação') AS categoria
            FROM {table.ref_sql}
            {where_sql}
        ), counts AS (
            SELECT EXTRACT(YEAR FROM dt) AS ano, categoria, COUNT(*) AS n
            FROM base
            WHERE dt IS NOT NULL
            GROUP BY 1, 2
        )
        SELECT ano, categoria, n,
               SUM(n) OVER (PARTITION BY ano) AS total_ano,
               CASE WHEN SUM(n) OVER (PARTITION BY ano) > 0
                    THEN ROUND(100.0 * n / SUM(n) OVER (PARTITION BY ano), 2)
                    ELSE NULL END AS pct
        FROM counts
        ORDER BY ano, categoria
    """
    return run_query(table, sql)


def query_age_dist(table: LoadedTable, age_sql: str, where_sql: str, sex_sql: Optional[str] = None) -> pd.DataFrame:
    if sex_sql:
        sql = f"""
            WITH base AS (
                SELECT {age_sql} AS idade, {sex_sql} AS sexo
                FROM {table.ref_sql}
                {where_sql}
            )
            SELECT sexo, FLOOR(idade / 5) * 5 AS faixa_ini, COUNT(*) AS n
            FROM base
            WHERE idade BETWEEN 0 AND 130 AND sexo IN ('Masculino', 'Feminino')
            GROUP BY 1, 2
            ORDER BY 2, 1
        """
    else:
        sql = f"""
            WITH base AS (
                SELECT {age_sql} AS idade
                FROM {table.ref_sql}
                {where_sql}
            )
            SELECT FLOOR(idade / 5) * 5 AS faixa_ini, COUNT(*) AS n
            FROM base
            WHERE idade BETWEEN 0 AND 130
            GROUP BY 1
            ORDER BY 1
        """
    return run_query(table, sql)


def query_cid_distribution(table: LoadedTable, exprs: Dict[str, Optional[str]], where_sql: str) -> pd.DataFrame:
    cid = exprs.get("cid")
    if not cid:
        return pd.DataFrame()
    source_expr = exprs.get("cid_source") or "NULL"
    sql = f"""
        WITH base AS (
            SELECT {cid} AS cid, {cid_group_expr(cid)} AS grupo, {cid_type_expr(cid)} AS tipo, {source_expr} AS coluna_origem
            FROM {table.ref_sql}
            {where_sql}
        )
        SELECT grupo, tipo, COUNT(*) AS n,
               COUNT(DISTINCT cid) AS cids_distintos,
               string_agg(DISTINCT cid, ', ' ORDER BY cid) FILTER (WHERE cid IS NOT NULL) AS cids_encontrados,
               string_agg(DISTINCT coluna_origem, ', ' ORDER BY coluna_origem) FILTER (WHERE coluna_origem IS NOT NULL) AS campos_origem
        FROM base
        GROUP BY 1, 2
        ORDER BY n DESC, grupo
    """
    df = run_query(table, sql)
    if not df.empty:
        df["pct"] = (df["n"] / df["n"].sum() * 100).round(2)
    return df


def query_sinan_cid10_conversion(table: LoadedTable, exprs: Dict[str, Optional[str]], where_sql: str) -> pd.DataFrame:
    con_code = exprs.get("con_code")
    if not con_code:
        return pd.DataFrame()
    con_label = exprs.get("con_label") or "NULL"
    con_group = exprs.get("con_group") or "NULL"
    cid_group = exprs.get("sinan_cid10_conversion_group") or sinan_cid10_conversion_group_expr(con_code)
    cid_type = exprs.get("sinan_cid10_conversion_type") or sinan_cid10_conversion_type_expr(con_code)
    include = exprs.get("sinan_cid10_conversion_include") or sinan_cid10_conversion_include_expr(con_code)
    sql = f"""
        WITH base AS (
            SELECT {con_code} AS con_code,
                   {con_label} AS conclusao_diagnostica,
                   {con_group} AS grupo_etiologico_sinan,
                   {cid_group} AS cid10_grupo,
                   {cid_type} AS cid10_classificacao,
                   {include} AS incluido_comparacao
            FROM {table.ref_sql}
            {where_sql}
        ), agg AS (
            SELECT cid10_grupo,
                   cid10_classificacao,
                   incluido_comparacao,
                   COUNT(*) AS n,
                   COUNT(DISTINCT con_code) FILTER (WHERE con_code IS NOT NULL) AS con_diages_distintos,
                   string_agg(DISTINCT conclusao_diagnostica, '; ' ORDER BY conclusao_diagnostica)
                       FILTER (WHERE conclusao_diagnostica IS NOT NULL) AS conclusoes_sinan,
                   string_agg(DISTINCT grupo_etiologico_sinan, '; ' ORDER BY grupo_etiologico_sinan)
                       FILTER (WHERE grupo_etiologico_sinan IS NOT NULL) AS grupos_sinan
            FROM base
            GROUP BY 1, 2, 3
        )
        SELECT *,
               SUM(n) OVER (PARTITION BY incluido_comparacao) AS denominador,
               CASE WHEN SUM(n) OVER (PARTITION BY incluido_comparacao) > 0
                    THEN ROUND(100.0 * n / SUM(n) OVER (PARTITION BY incluido_comparacao), 2)
                    ELSE NULL END AS pct
        FROM agg
        ORDER BY CASE WHEN incluido_comparacao = 'Sim' THEN 0 ELSE 1 END,
                 n DESC, cid10_grupo
    """
    return run_query(table, sql)


def query_ciha_death_cid_distribution(table: LoadedTable, exprs: Dict[str, Optional[str]], where_sql: str) -> pd.DataFrame:
    morte = exprs.get("morte_code")
    cid = exprs.get("cid")
    if not (morte and cid):
        return pd.DataFrame()
    death_where = append_clause(where_sql, f"{morte} = '1'")
    return query_cid_distribution(table, exprs, death_where)


def sinan_quimio_param_exprs(exprs: Dict[str, Optional[str]]) -> List[Tuple[str, str, str]]:
    params: List[Tuple[str, str, str]] = []
    for key, info in SINAN_QUIMIO_PARAMS.items():
        expr = exprs.get(f"lab_{key}")
        if expr:
            params.append((key, str(info["label"]), expr))
    return params


def query_sinan_quimio_summary(table: LoadedTable, exprs: Dict[str, Optional[str]], where_sql: str) -> pd.DataFrame:
    params = sinan_quimio_param_exprs(exprs)
    if not params:
        return pd.DataFrame()
    unions = []
    for key, label, value_expr in params:
        unions.append(
            f"""
            SELECT {qstr(key)} AS parametro_id, {qstr(label)} AS parametro, {value_expr} AS valor
            FROM {table.ref_sql}
            {where_sql}
            """
        )
    sql = f"""
        WITH valores AS (
            {' UNION ALL '.join(unions)}
        )
        SELECT parametro_id,
               parametro,
               COUNT(*) AS registros_avaliados,
               COUNT(*) FILTER (WHERE valor IS NOT NULL AND valor >= 0) AS n_valido,
               COUNT(*) FILTER (WHERE valor IS NULL OR valor < 0) AS n_sem_valor,
               ROUND(100.0 * COUNT(*) FILTER (WHERE valor IS NOT NULL AND valor >= 0) / NULLIF(COUNT(*), 0), 2) AS pct_preenchido,
               MIN(valor) FILTER (WHERE valor IS NOT NULL AND valor >= 0) AS minimo,
               quantile_cont(valor, 0.25) FILTER (WHERE valor IS NOT NULL AND valor >= 0) AS q1,
               median(valor) FILTER (WHERE valor IS NOT NULL AND valor >= 0) AS mediana,
               AVG(valor) FILTER (WHERE valor IS NOT NULL AND valor >= 0) AS media,
               quantile_cont(valor, 0.75) FILTER (WHERE valor IS NOT NULL AND valor >= 0) AS q3,
               MAX(valor) FILTER (WHERE valor IS NOT NULL AND valor >= 0) AS maximo
        FROM valores
        GROUP BY 1, 2
        ORDER BY CASE parametro_id
            WHEN 'hema' THEN 1
            WHEN 'neutro' THEN 2
            WHEN 'glico' THEN 3
            WHEN 'leuco' THEN 4
            WHEN 'eosi' THEN 5
            WHEN 'prot' THEN 6
            WHEN 'mono' THEN 7
            WHEN 'linfo' THEN 8
            WHEN 'clor' THEN 9
            ELSE 99 END
    """
    return run_query(table, sql)


def query_sinan_numeric_distribution(table: LoadedTable, value_expr: str, where_sql: str, bins: int = 30) -> pd.DataFrame:
    stats_sql = f"""
        WITH base AS (
            SELECT {value_expr} AS valor
            FROM {table.ref_sql}
            {where_sql}
        )
        SELECT COUNT(*) AS n, MIN(valor) AS minimo, MAX(valor) AS maximo
        FROM base
        WHERE valor IS NOT NULL AND valor >= 0
    """
    stats = run_query(table, stats_sql)
    if stats.empty or int(stats.iloc[0]["n"] or 0) == 0:
        return pd.DataFrame()

    n = int(stats.iloc[0]["n"])
    minimo = float(stats.iloc[0]["minimo"])
    maximo = float(stats.iloc[0]["maximo"])
    if minimo == maximo:
        return pd.DataFrame(
            {
                "faixa_inicio": [minimo],
                "faixa_fim": [maximo],
                "faixa": [f"{minimo:g}"],
                "n": [n],
                "denominador": [n],
                "pct": [100.0],
            }
        )

    bin_count = max(1, min(int(bins), n))
    width = (maximo - minimo) / bin_count
    if width <= 0:
        return pd.DataFrame()

    sql = f"""
        WITH base AS (
            SELECT {value_expr} AS valor
            FROM {table.ref_sql}
            {where_sql}
        ), validos AS (
            SELECT valor
            FROM base
            WHERE valor IS NOT NULL AND valor >= 0
        ), binned AS (
            SELECT CASE
                       WHEN valor = {maximo!r} THEN {bin_count - 1}
                       ELSE CAST(FLOOR((valor - {minimo!r}) / {width!r}) AS INTEGER)
                   END AS bin_idx
            FROM validos
        ), agg AS (
            SELECT bin_idx, COUNT(*) AS n
            FROM binned
            GROUP BY 1
        )
        SELECT {minimo!r} + bin_idx * {width!r} AS faixa_inicio,
               CASE WHEN bin_idx = {bin_count - 1} THEN {maximo!r}
                    ELSE {minimo!r} + (bin_idx + 1) * {width!r} END AS faixa_fim,
               n,
               SUM(n) OVER () AS denominador,
               ROUND(100.0 * n / SUM(n) OVER (), 2) AS pct
        FROM agg
        ORDER BY faixa_inicio
    """
    df = run_query(table, sql)
    if df.empty:
        return df

    def fmt(value: object) -> str:
        try:
            num = float(value)
        except Exception:
            return str(value)
        if abs(num) >= 100 or abs(num - round(num)) < 1e-9:
            return f"{num:.0f}"
        return f"{num:.1f}".replace(".", ",")

    df["faixa"] = [f"{fmt(a)}–{fmt(b)}" for a, b in zip(df["faixa_inicio"], df["faixa_fim"])]
    return df


def query_sinan_indicators(table: LoadedTable, exprs: Dict[str, Optional[str]], where_sql: str) -> pd.DataFrame:
    dt, classi, evol = exprs.get("dt"), exprs.get("classi_code"), exprs.get("evol_code")
    con_group = exprs.get("con_group")
    if not (dt and classi and evol):
        return pd.DataFrame()
    extra = f", {con_group} AS etiologia" if con_group else ", NULL AS etiologia"
    encerr = exprs.get("dt_encerramento")
    notif = exprs.get("dt_notificacao") or dt
    dias_encerr = f"DATEDIFF('day', {notif}, {encerr})" if encerr and notif else "NULL"
    sql = f"""
        WITH base AS (
            SELECT {dt} AS dt,
                   {classi} AS classi,
                   {evol} AS evol,
                   {dias_encerr} AS dias_encerramento
                   {extra}
            FROM {table.ref_sql}
            {where_sql}
        ), agg AS (
            SELECT EXTRACT(YEAR FROM dt) AS ano,
                   COUNT(*) AS notificacoes,
                   COUNT(*) FILTER (WHERE classi = '1') AS confirmados,
                   COUNT(*) FILTER (WHERE classi = '2') AS descartados,
                   COUNT(*) FILTER (WHERE classi = '8') AS inconclusivos,
                   COUNT(*) FILTER (WHERE classi IS NULL OR classi NOT IN ('1','2','8')) AS sem_classificacao,
                   COUNT(*) FILTER (WHERE classi = '1' AND evol = '1') AS altas_confirmados,
                   COUNT(*) FILTER (WHERE classi = '1' AND evol = '2') AS obitos_meningite_confirmados,
                   COUNT(*) FILTER (WHERE classi = '1' AND evol = '3') AS obitos_outra_causa_confirmados,
                   COUNT(*) FILTER (WHERE classi = '1' AND evol IN ('1','2','3')) AS confirmados_evolucao_conhecida,
                   COUNT(*) FILTER (WHERE classi = '1' AND (evol IS NULL OR evol = '9')) AS confirmados_evolucao_ignorada,
                   median(dias_encerramento) FILTER (WHERE dias_encerramento BETWEEN 0 AND 3650) AS mediana_dias_encerramento
            FROM base
            WHERE dt IS NOT NULL
            GROUP BY 1
        )
        SELECT *,
               {pct_expr('confirmados', 'notificacoes')} AS pct_confirmacao,
               {pct_expr('descartados', 'notificacoes')} AS pct_descarte,
               {pct_expr('inconclusivos', 'notificacoes')} AS pct_inconclusivos,
               {pct_expr('sem_classificacao', 'notificacoes')} AS pct_sem_classificacao,
               {pct_expr('obitos_meningite_confirmados', 'notificacoes')} AS pct_obitos_meningite_confirmados_notificacoes,
               {pct_expr('obitos_meningite_confirmados', 'confirmados_evolucao_conhecida')} AS letalidade_confirmados_evolucao_conhecida
        FROM agg
        ORDER BY ano
    """
    return run_query(table, sql)


def query_sinan_etiology_lethality(table: LoadedTable, exprs: Dict[str, Optional[str]], where_sql: str) -> pd.DataFrame:
    classi, evol, con_group, con_label = exprs.get("classi_code"), exprs.get("evol_code"), exprs.get("con_group"), exprs.get("con_label")
    if not (classi and evol and con_group):
        return pd.DataFrame()
    sql = f"""
        WITH base AS (
            SELECT {classi} AS classi,
                   {evol} AS evol,
                   {con_group} AS grupo_etiologico,
                   {con_label or con_group} AS conclusao_diagnostica
            FROM {table.ref_sql}
            {where_sql}
        ), agg AS (
            SELECT grupo_etiologico,
                   COUNT(*) FILTER (WHERE classi = '1') AS confirmados,
                   COUNT(*) FILTER (WHERE classi = '1' AND evol = '2') AS obitos_meningite,
                   COUNT(*) FILTER (WHERE classi = '1' AND evol = '3') AS obitos_outra_causa,
                   COUNT(*) FILTER (WHERE classi = '1' AND evol IN ('1','2','3')) AS confirmados_evolucao_conhecida,
                   COUNT(*) FILTER (WHERE classi = '1' AND (evol IS NULL OR evol = '9')) AS confirmados_evolucao_ignorada
            FROM base
            GROUP BY 1
        )
        SELECT *,
               {pct_expr('obitos_meningite', 'confirmados_evolucao_conhecida')} AS letalidade_pct
        FROM agg
        ORDER BY confirmados DESC, grupo_etiologico
    """
    return run_query(table, sql)


def query_sinan_diagnostics_by_year(table: LoadedTable, exprs: Dict[str, Optional[str]], where_sql: str) -> pd.DataFrame:
    dt, classi, con_group = exprs.get("dt"), exprs.get("classi_code"), exprs.get("con_group")
    if not (dt and classi and con_group):
        return pd.DataFrame()
    sql = f"""
        WITH base AS (
            SELECT {dt} AS dt, {classi} AS classi, {con_group} AS grupo_etiologico
            FROM {table.ref_sql}
            {where_sql}
        )
        SELECT EXTRACT(YEAR FROM dt) AS ano, grupo_etiologico,
               COUNT(*) FILTER (WHERE classi = '1') AS confirmados
        FROM base
        WHERE dt IS NOT NULL
        GROUP BY 1, 2
        ORDER BY 1, 2
    """
    return run_query(table, sql)


def query_sim_indicators(table: LoadedTable, exprs: Dict[str, Optional[str]], where_sql: str) -> pd.DataFrame:
    dt = exprs.get("dt")
    cid_any = exprs.get("cid")
    causabas = exprs.get("causabas_cid")
    if not (dt and cid_any):
        return pd.DataFrame()
    causabas_sql = causabas or "NULL"
    sql = f"""
        WITH base AS (
            SELECT {dt} AS dt, {cid_any} AS cid_mencao, {causabas_sql} AS cid_causa_basica
            FROM {table.ref_sql}
            {where_sql}
        ), agg AS (
            SELECT EXTRACT(YEAR FROM dt) AS ano,
                   COUNT(*) AS obitos_registros,
                   COUNT(*) FILTER (WHERE cid_causa_basica IS NOT NULL) AS obitos_causa_basica_meningite,
                   COUNT(*) FILTER (WHERE cid_mencao IS NOT NULL) AS obitos_com_mencao_meningite
            FROM base
            WHERE dt IS NOT NULL
            GROUP BY 1
        )
        SELECT *,
               {pct_expr('obitos_causa_basica_meningite', 'obitos_registros')} AS pct_causa_basica_meningite,
               {pct_expr('obitos_com_mencao_meningite', 'obitos_registros')} AS pct_mencao_meningite
        FROM agg
        ORDER BY ano
    """
    return run_query(table, sql)


def query_ciha_indicators(table: LoadedTable, exprs: Dict[str, Optional[str]], where_sql: str) -> pd.DataFrame:
    dt = exprs.get("dt")
    cid_any = exprs.get("cid")
    diag_princ = exprs.get("diag_princ_cid")
    morte = exprs.get("morte_code")
    dias = exprs.get("dias_perm")
    if not dt:
        return pd.DataFrame()
    cid_any_sql = cid_any or "NULL"
    diag_princ_sql = diag_princ or "NULL"
    morte_sql = morte or "NULL"
    dias_sql = dias or "NULL"
    sql = f"""
        WITH base AS (
            SELECT {dt} AS dt,
                   {cid_any_sql} AS cid_mencao,
                   {diag_princ_sql} AS cid_principal,
                   {morte_sql} AS morte,
                   {dias_sql} AS dias_perm
            FROM {table.ref_sql}
            {where_sql}
        ), agg AS (
            SELECT EXTRACT(YEAR FROM dt) AS ano,
                   COUNT(*) AS atendimentos,
                   COUNT(*) FILTER (WHERE cid_principal IS NOT NULL) AS atendimentos_diag_principal_meningite,
                   COUNT(*) FILTER (WHERE cid_mencao IS NOT NULL) AS atendimentos_qualquer_cid_meningite,
                   COUNT(*) FILTER (WHERE morte = '1') AS mortes_administrativas,
                   COUNT(*) FILTER (WHERE dias_perm = 0) AS permanencia_zero,
                   median(dias_perm) FILTER (WHERE dias_perm BETWEEN 0 AND 365) AS mediana_dias_perm
            FROM base
            WHERE dt IS NOT NULL
            GROUP BY 1
        )
        SELECT *,
               {pct_expr('mortes_administrativas', 'atendimentos')} AS pct_morte_administrativa,
               {pct_expr('permanencia_zero', 'atendimentos')} AS pct_permanencia_zero
        FROM agg
        ORDER BY ano
    """
    return run_query(table, sql)


def query_ciha_dias_perm_distribution(table: LoadedTable, exprs: Dict[str, Optional[str]], where_sql: str) -> pd.DataFrame:
    dias = exprs.get("dias_perm")
    if not dias:
        return pd.DataFrame()
    sql = f"""
        WITH base AS (
            SELECT {dias} AS dias_perm
            FROM {table.ref_sql}
            {where_sql}
        ), bucketed AS (
            SELECT
                CASE
                    WHEN dias_perm IS NULL THEN 'Sem informação'
                    WHEN dias_perm < 0 THEN 'Valor negativo/inválido'
                    WHEN dias_perm BETWEEN 0 AND 30 THEN CAST(CAST(dias_perm AS BIGINT) AS VARCHAR)
                    WHEN dias_perm BETWEEN 31 AND 60 THEN '31–60'
                    WHEN dias_perm BETWEEN 61 AND 90 THEN '61–90'
                    WHEN dias_perm > 90 THEN '91+'
                    ELSE 'Sem informação'
                END AS faixa_dias_perm,
                CASE
                    WHEN dias_perm IS NULL THEN 9998
                    WHEN dias_perm < 0 THEN 9999
                    WHEN dias_perm BETWEEN 0 AND 30 THEN CAST(dias_perm AS BIGINT)
                    WHEN dias_perm BETWEEN 31 AND 60 THEN 31
                    WHEN dias_perm BETWEEN 61 AND 90 THEN 61
                    WHEN dias_perm > 90 THEN 91
                    ELSE 9998
                END AS ordem
            FROM base
        ), counts AS (
            SELECT faixa_dias_perm, ordem, COUNT(*) AS n
            FROM bucketed
            GROUP BY 1, 2
        )
        SELECT faixa_dias_perm, ordem, n, SUM(n) OVER () AS denominador,
               CASE WHEN SUM(n) OVER () > 0 THEN ROUND(100.0 * n / SUM(n) OVER (), 2) ELSE NULL END AS pct
        FROM counts
        ORDER BY ordem, faixa_dias_perm
    """
    return run_query(table, sql)


def query_missingness(table: LoadedTable, fields: Dict[str, Optional[str]], dt_sql: Optional[str], where_sql: str) -> pd.DataFrame:
    checks = [(label, expr) for label, expr in fields.items() if expr]
    if not checks:
        return pd.DataFrame()
    select_parts = []
    for label, expr in checks:
        select_parts.append(
            f"SELECT {qstr(label)} AS campo, COUNT(*) FILTER (WHERE {expr} IS NULL) AS faltantes, COUNT(*) AS total FROM base"
        )
    sql = f"""
        WITH base AS (
            SELECT {', '.join(f'{expr} AS f_{i}' for i, (_, expr) in enumerate(checks))}
            FROM {table.ref_sql}
            {where_sql}
        )
        SELECT campo, faltantes, total, CASE WHEN total > 0 THEN ROUND(100.0 * faltantes / total, 2) ELSE NULL END AS pct_faltante
        FROM (
            {' UNION ALL '.join(
                f"SELECT {qstr(label)} AS campo, COUNT(*) FILTER (WHERE f_{i} IS NULL) AS faltantes, COUNT(*) AS total FROM base"
                for i, (label, _) in enumerate(checks)
            )}
        )
        ORDER BY pct_faltante DESC, campo
    """
    return run_query(table, sql)


def query_missingness_by_year(table: LoadedTable, fields: Dict[str, Optional[str]], dt_sql: Optional[str], where_sql: str) -> pd.DataFrame:
    if not dt_sql:
        return pd.DataFrame()
    checks = [(label, expr) for label, expr in fields.items() if expr]
    if not checks:
        return pd.DataFrame()
    field_select = ", ".join(f"{expr} AS f_{i}" for i, (_, expr) in enumerate(checks))
    union = []
    for i, (label, _) in enumerate(checks):
        union.append(
            f"""
            SELECT ano, {qstr(label)} AS campo,
                   COUNT(*) FILTER (WHERE f_{i} IS NULL) AS faltantes,
                   COUNT(*) AS total
            FROM base
            GROUP BY 1
            """
        )
    sql = f"""
        WITH base AS (
            SELECT EXTRACT(YEAR FROM {dt_sql}) AS ano, {field_select}
            FROM {table.ref_sql}
            {where_sql}
        )
        SELECT ano, campo, faltantes, total,
               CASE WHEN total > 0 THEN ROUND(100.0 * faltantes / total, 2) ELSE NULL END AS pct_faltante
        FROM ({' UNION ALL '.join(union)})
        WHERE ano IS NOT NULL
        ORDER BY ano, campo
    """
    return run_query(table, sql)


def query_enriched_preview(table: LoadedTable, sel: ColumnSelection, exprs: Dict[str, Optional[str]], where_sql: str, limit: Optional[int] = 200) -> pd.DataFrame:
    items = []
    mapping = [
        ("data_analise", exprs.get("dt")),
        ("sexo", exprs.get("sex")),
        ("idade_anos", exprs.get("age")),
        ("raca_cor", exprs.get("race")),
        ("municipio_residencia", exprs.get("mun_res_label") or exprs.get("mun_res")),
        ("municipio_evento_atendimento", exprs.get("mun_event_label") or exprs.get("mun_event")),
        ("cid_meningite_detectado", exprs.get("cid")),
        ("tipo_cid10", exprs.get("cid_type")),
        ("campo_origem_cid", exprs.get("cid_source")),
        ("sinan_classificacao_final", exprs.get("classi_label")),
        ("sinan_conclusao_diagnostica", exprs.get("con_label")),
        ("sinan_grupo_etiologico", exprs.get("con_group")),
        ("sinan_cid10_convertido_grupo", exprs.get("sinan_cid10_conversion_group")),
        ("sinan_cid10_convertido_tipo", exprs.get("sinan_cid10_conversion_type")),
        ("sinan_cid10_inclui_comparacao", exprs.get("sinan_cid10_conversion_include")),
        ("sinan_evolucao", exprs.get("evol_label")),
        ("sinan_criterio", exprs.get("criterio_label")),
        ("sinan_puncao_laboratorial", exprs.get("puncao_label")),
        ("sinan_exame_quimiocitologico", exprs.get("quimio_label")),
        ("sinan_lab_glicose", exprs.get("lab_glico")),
        ("sinan_lab_proteinas", exprs.get("lab_prot")),
        ("sinan_lab_leucocitos", exprs.get("lab_leuco")),
        ("sim_obito_gravidez", exprs.get("obitograv_label")),
        ("sim_obito_puerperio", exprs.get("obitopuerp_label")),
        ("ciha_morte", exprs.get("morte_code")),
        ("ciha_dias_perm", exprs.get("dias_perm")),
    ]
    for alias, expr in mapping:
        if expr:
            items.append(f"{expr} AS {qident(alias)}")

    raw_cols: List[str] = []
    for col in [
        sel.date_col,
        sel.sex_col,
        sel.age_col,
        sel.age_unit_col,
        sel.race_col,
        sel.municipality_res_col,
        sel.municipality_event_col,
        *sel.cid_cols,
        sel.classi_fin_col,
        sel.con_diages_col,
        sel.evolucao_col,
        sel.criterio_col,
        sel.lab_puncao_col,
        sel.lab_liquor_col,
        sel.lab_hema_col,
        sel.lab_neutro_col,
        sel.lab_glico_col,
        sel.lab_leuco_col,
        sel.lab_eosi_col,
        sel.lab_prot_col,
        sel.lab_mono_col,
        sel.lab_linfo_col,
        sel.lab_clor_col,
        sel.causabas_col,
        sel.causabas_o_col,
        sel.obitograv_col,
        sel.obitopuerp_col,
        sel.diag_princ_col,
        sel.diag_secun_col,
        sel.morte_col,
        sel.dias_perm_col,
    ]:
        if col and col not in raw_cols:
            raw_cols.append(col)
    for col in raw_cols:
        items.append(f"{qident(col)} AS {qident('raw_' + col[:45])}")
    if not items:
        items = ["*"]
    limit_sql = "" if limit is None else f" LIMIT {int(limit)}"
    sql = f"SELECT {', '.join(items)} FROM {table.ref_sql} {where_sql}{limit_sql}"
    return run_query(table, sql)


# =============================================================================
# Visualização e UI
# =============================================================================


def download_button(df: pd.DataFrame, filename: str, label: str = "Baixar CSV") -> None:
    if df is None or df.empty:
        return
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name=filename,
        mime="text/csv",
        use_container_width=False,
    )


def render_field_guide(source: str) -> None:
    st.dataframe(
        pd.DataFrame(FIELD_GUIDE[source], columns=["Campo", "Uso", "Leitura epidemiológica"]),
        use_container_width=True,
        hide_index=True,
    )
    for note in SOURCE_CONFIG[source].field_notes:
        st.caption("• " + note)


def render_cid_reference() -> None:
    st.dataframe(pd.DataFrame(CID_RULES)[["grupo", "padrao", "rotulo"]], use_container_width=True, hide_index=True)
    st.caption(
        "O app procura A17.0, A39.0, A87*, G00*, G01*, G02* e G03* nos campos selecionados. "
        "No SINAN, esse CID costuma ser apenas o agravo geral; a etiologia específica deve vir de CON_DIAGES e campos relacionados."
    )


def render_loader(source: str) -> Optional[LoadedTable]:
    cfg = SOURCE_CONFIG[source]
    st.markdown(f"### {source} — {cfg.title}")
    st.caption(f"Período esperado no arquivo enviado: {cfg.expected_period}")

    mode = st.radio(
        "Fonte de dados",
        ["DuckDB local", "Upload DuckDB", "Parquet local/glob", "Upload Parquet"],
        horizontal=True,
        key=f"load_mode_{source}",
    )

    if mode == "DuckDB local":
        default_path = first_existing_path(cfg.default_db)
        path = st.text_input("Caminho do DuckDB", value=default_path, key=f"duckdb_path_{source}")
        if not path or not Path(path).exists():
            st.warning("Informe um caminho existente para o arquivo .duckdb ou use upload.")
            return None
        try:
            tables = list_duckdb_tables(path)
        except Exception as exc:
            st.error(f"Não consegui abrir o DuckDB: {exc}")
            return None
        if not tables:
            st.error("O arquivo DuckDB não contém tabelas visíveis.")
            return None
        default_idx = tables.index(cfg.default_table) if cfg.default_table in tables else 0
        table_name = st.selectbox("Tabela", options=tables, index=default_idx, key=f"duckdb_table_{source}")
        return LoadedTable(source=source, kind="duckdb", db_path=path, table_name=table_name, ref_sql=qident(table_name), label=f"{Path(path).name}:{table_name}")

    if mode == "Upload DuckDB":
        upload = st.file_uploader("Envie um arquivo .duckdb", type=["duckdb"], key=f"upload_duckdb_{source}")
        if not upload:
            st.info("Envie o DuckDB para continuar.")
            return None
        path = materialize_upload(upload, f"{source.lower()}_duckdb")
        try:
            tables = list_duckdb_tables(path)
        except Exception as exc:
            st.error(f"Não consegui abrir o DuckDB enviado: {exc}")
            return None
        default_idx = tables.index(cfg.default_table) if cfg.default_table in tables else 0
        table_name = st.selectbox("Tabela", options=tables, index=default_idx, key=f"upload_duckdb_table_{source}")
        return LoadedTable(source=source, kind="duckdb", db_path=path, table_name=table_name, ref_sql=qident(table_name), label=f"upload:{upload.name}:{table_name}")

    if mode == "Parquet local/glob":
        glob_value = st.text_input(
            "Caminho/glob dos Parquets",
            value="",
            placeholder="Ex.: dados/sinan/*.parquet",
            key=f"parquet_glob_{source}",
        )
        paths = sorted(glob.glob(glob_value)) if glob_value else []
        if not paths:
            st.info("Informe um glob local que encontre ao menos um arquivo .parquet.")
            return None
        return LoadedTable(source=source, kind="parquet", parquet_paths=paths, ref_sql=parquet_ref(paths), label=f"{len(paths)} parquet(s)")

    uploads = st.file_uploader("Envie um ou mais Parquets", type=["parquet"], accept_multiple_files=True, key=f"upload_parquet_{source}")
    if not uploads:
        st.info("Envie Parquet(s) para continuar.")
        return None
    paths = [materialize_upload(up, f"{source.lower()}_parquet") for up in uploads]
    return LoadedTable(source=source, kind="parquet", parquet_paths=paths, ref_sql=parquet_ref(paths), label=f"{len(paths)} parquet(s) enviados")


def render_column_config(source: str, columns: Sequence[str]) -> ColumnSelection:
    cfg = SOURCE_CONFIG[source]
    defaults = default_selections(source, columns)
    age_options = ["Automático", "Anos diretos", "DATASUS codificada", "DATASUS com coluna de unidade"]

    def select(label: str, default: Optional[str], key: str, help_text: Optional[str] = None) -> Optional[str]:
        idx = columns.index(default) + 1 if default in columns else 0
        return st.selectbox(label, [None] + list(columns), index=idx, key=key, format_func=lambda x: "(não usar)" if x is None else x, help=help_text)

    with st.expander("1) Configuração de colunas", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            date_col = select("Data principal", defaults.date_col, f"date_{source}")
            sex_col = select("Sexo", defaults.sex_col, f"sex_{source}")
            race_col = select("Raça/cor", defaults.race_col, f"race_{source}")
        with c2:
            age_col = select("Idade", defaults.age_col, f"age_{source}")
            age_unit_col = select("Unidade da idade", defaults.age_unit_col, f"age_unit_{source}")
            age_mode = st.selectbox("Interpretação da idade", age_options, index=age_options.index(defaults.age_mode) if defaults.age_mode in age_options else 0, key=f"age_mode_{source}")
        with c3:
            mun_res = select("Município de residência", defaults.municipality_res_col, f"mun_res_{source}")
            mun_event = select("Município de ocorrência/atendimento/notificação", defaults.municipality_event_col, f"mun_event_{source}")
        with c4:
            cid_cols = st.multiselect(
                "Campos para localizar CID-10",
                options=list(columns),
                default=[c for c in defaults.cid_cols if c in columns],
                key=f"cid_cols_{source}",
                help="No SIM/CIHA use causa básica/diagnósticos. No SINAN, ID_AGRAVO costuma ser G039; a etiologia específica fica em CON_DIAGES.",
            )

        if source == "SINAN":
            st.markdown("**Campos específicos do SINAN**")
            s1, s2, s3, s4 = st.columns(4)
            with s1:
                classi_fin_col = select("CLASSI_FIN", defaults.classi_fin_col, f"classi_{source}")
                con_diages_col = select("CON_DIAGES", defaults.con_diages_col, f"con_{source}")
            with s2:
                evolucao_col = select("EVOLUCAO", defaults.evolucao_col, f"evol_{source}")
                criterio_col = select("CRITERIO", defaults.criterio_col, f"criterio_{source}")
            with s3:
                lab_puncao_col = select("LAB_PUNCAO — Punção Laboratorial", defaults.lab_puncao_col, f"puncao_{source}")
                lab_liquor_col = select("LAB_LIQUOR — Exame Quimiocitológico", defaults.lab_liquor_col, f"quimio_{source}")
            with s4:
                ate_hospit_col = select("ATE_HOSPIT", defaults.ate_hospit_col, f"hospit_{source}")
                dt_encerramento_col = select("DT_ENCERRA", defaults.dt_encerramento_col, f"dt_enc_{source}")
                dt_notificacao_col = select("DT_NOTIFIC", defaults.dt_notificacao_col, f"dt_notif_{source}")

            st.markdown("**Parâmetros do Exame Quimiocitológico do líquor**")
            q1, q2, q3 = st.columns(3)
            with q1:
                lab_hema_col = select("LAB_HEMA — Hemácias", defaults.lab_hema_col, f"lab_hema_{source}")
                lab_neutro_col = select("LAB_NEUTRO — Neutrófilos", defaults.lab_neutro_col, f"lab_neutro_{source}")
                lab_glico_col = select("LAB_GLICO — Glicose", defaults.lab_glico_col, f"lab_glico_{source}")
            with q2:
                lab_leuco_col = select("LAB_LEUCO — Leucócitos", defaults.lab_leuco_col, f"lab_leuco_{source}")
                lab_eosi_col = select("LAB_EOSI — Eosinófilos", defaults.lab_eosi_col, f"lab_eosi_{source}")
                lab_prot_col = select("LAB_PROT — Proteínas", defaults.lab_prot_col, f"lab_prot_{source}")
            with q3:
                lab_mono_col = select("LAB_MONO — Monócitos", defaults.lab_mono_col, f"lab_mono_{source}")
                lab_linfo_col = select("LAB_LINFO — Linfócitos", defaults.lab_linfo_col, f"lab_linfo_{source}")
                lab_clor_col = select("LAB_CLOR — Cloreto", defaults.lab_clor_col, f"lab_clor_{source}")

            return ColumnSelection(
                date_col=date_col, sex_col=sex_col, age_col=age_col, age_unit_col=age_unit_col,
                race_col=race_col, municipality_res_col=mun_res, municipality_event_col=mun_event,
                cid_cols=cid_cols, age_mode=age_mode,
                classi_fin_col=classi_fin_col, con_diages_col=con_diages_col,
                evolucao_col=evolucao_col, criterio_col=criterio_col,
                lab_puncao_col=lab_puncao_col, lab_liquor_col=lab_liquor_col,
                lab_hema_col=lab_hema_col, lab_neutro_col=lab_neutro_col,
                lab_glico_col=lab_glico_col, lab_leuco_col=lab_leuco_col,
                lab_eosi_col=lab_eosi_col, lab_prot_col=lab_prot_col,
                lab_mono_col=lab_mono_col, lab_linfo_col=lab_linfo_col, lab_clor_col=lab_clor_col,
                ate_hospit_col=ate_hospit_col, dt_encerramento_col=dt_encerramento_col,
                dt_notificacao_col=dt_notificacao_col,
            )

        if source == "SIM":
            st.markdown("**Campos específicos do SIM**")
            s1, s2, s3, s4 = st.columns(4)
            with s1:
                causabas_col = select("CAUSABAS", defaults.causabas_col, f"causabas_{source}")
            with s2:
                causabas_o_col = select("CAUSABAS_O", defaults.causabas_o_col, f"causabaso_{source}")
            with s3:
                obitograv_col = select("OBITOGRAV — óbito na gravidez", defaults.obitograv_col, f"obitograv_{source}")
            with s4:
                obitopuerp_col = select("OBITOPUERP — puerpério", defaults.obitopuerp_col, f"obitopuerp_{source}")
            return ColumnSelection(
                date_col, sex_col, age_col, age_unit_col, race_col, mun_res, mun_event, cid_cols, age_mode,
                causabas_col=causabas_col, causabas_o_col=causabas_o_col,
                obitograv_col=obitograv_col, obitopuerp_col=obitopuerp_col,
            )

        st.markdown("**Campos específicos da CIHA**")
        s1, s2, s3 = st.columns(3)
        with s1:
            diag_princ_col = select("DIAG_PRINC", defaults.diag_princ_col, f"diagp_{source}")
            diag_secun_col = select("DIAG_SECUN", defaults.diag_secun_col, f"diags_{source}")
        with s2:
            morte_col = select("MORTE", defaults.morte_col, f"morte_{source}")
            dias_perm_col = select("DIAS_PERM", defaults.dias_perm_col, f"dias_{source}")
        with s3:
            modalidade_col = select("MODALIDADE", defaults.modalidade_col, f"modalidade_{source}")
        return ColumnSelection(date_col, sex_col, age_col, age_unit_col, race_col, mun_res, mun_event, cid_cols, age_mode, diag_princ_col=diag_princ_col, diag_secun_col=diag_secun_col, morte_col=morte_col, dias_perm_col=dias_perm_col, modalidade_col=modalidade_col)


def case_definition_clause(source: str, definition: str, exprs: Dict[str, Optional[str]]) -> Optional[str]:
    if source == "SINAN":
        classi = exprs.get("classi_code")
        evol = exprs.get("evol_code")
        if definition == "Todos os registros/notificações" or not classi:
            return None
        if definition == "Somente confirmados":
            return f"{classi} = '1'"
        if definition == "Somente descartados":
            return f"{classi} = '2'"
        if definition == "Confirmados com evolução conhecida" and evol:
            return f"{classi} = '1' AND {evol} IN ('1','2','3')"
        if definition == "Óbito por meningite entre confirmados" and evol:
            return f"{classi} = '1' AND {evol} = '2'"
        return None
    if source == "SIM":
        cid = exprs.get("cid")
        cb = exprs.get("causabas_cid")
        if definition == "Todos os registros do recorte":
            return None
        if definition == "Causa básica com CID de meningite" and cb:
            return f"{cb} IS NOT NULL"
        if definition == "Menção de CID de meningite em qualquer campo" and cid:
            return f"{cid} IS NOT NULL"
        return None
    if source == "CIHA":
        cid = exprs.get("cid")
        dp = exprs.get("diag_princ_cid")
        morte = exprs.get("morte_code")
        if definition == "Todos os atendimentos do recorte":
            return None
        if definition == "Diagnóstico principal com CID de meningite" and dp:
            return f"{dp} IS NOT NULL"
        if definition == "Diagnóstico principal ou secundário com CID de meningite" and cid:
            return f"{cid} IS NOT NULL"
        if definition == "Somente registros com morte" and morte:
            return f"{morte} = '1'"
        return None
    return None


def render_filters(source: str, table: LoadedTable, exprs: Dict[str, Optional[str]]) -> Tuple[str, str, str]:
    clauses: List[str] = []
    definition_clause: Optional[str] = None

    with st.expander("2) Filtros e definição de série", expanded=True):
        if source == "SINAN":
            definitions = [
                "Todos os registros/notificações",
                "Somente confirmados",
                "Somente descartados",
                "Confirmados com evolução conhecida",
                "Óbito por meningite entre confirmados",
            ]
        elif source == "SIM":
            definitions = [
                "Todos os registros do recorte",
                "Causa básica com CID de meningite",
                "Menção de CID de meningite em qualquer campo",
            ]
        else:
            definitions = [
                "Todos os atendimentos do recorte",
                "Diagnóstico principal com CID de meningite",
                "Diagnóstico principal ou secundário com CID de meningite",
                "Somente registros com morte",
            ]
        definition = st.selectbox("Definição aplicada aos gráficos exploratórios", definitions, key=f"definition_{source}")
        definition_clause = case_definition_clause(source, definition, exprs)

        c1, c2, c3, c4 = st.columns(4)
        dt = exprs.get("dt")
        if dt:
            bounds = minmax_date(table, dt)
            if bounds:
                min_year, max_year = int(bounds[0].year), int(bounds[1].year)
                expected_years = [int(x) for x in __import__('re').findall(r'\d{4}', SOURCE_CONFIG[source].expected_period)]
                if len(expected_years) >= 2:
                    default_start = max(min_year, expected_years[0])
                    default_end = min(max_year, expected_years[-1])
                else:
                    default_start, default_end = min_year, max_year
                if default_start > default_end:
                    default_start, default_end = min_year, max_year
                with c1:
                    if min_year >= max_year:
                        st.markdown(f"**Ano:** {min_year}")
                        year_range = (min_year, max_year)
                    else:
                        year_range = st.slider("Ano", min_year, max_year, (default_start, default_end), key=f"year_{source}")
                        if min_year < default_start or max_year > default_end:
                            st.caption("Há datas fora do período esperado; o intervalo padrão usa o período operacional da base.")
                clauses.append(f"EXTRACT(YEAR FROM {dt}) BETWEEN {int(year_range[0])} AND {int(year_range[1])}")
        age = exprs.get("age")
        if age:
            with c2:
                age_range = st.slider("Idade em anos", 0, 120, (0, 120), key=f"age_filter_{source}")
            clauses.append(f"{age} BETWEEN {int(age_range[0])} AND {int(age_range[1])}")
        sex = exprs.get("sex")
        if sex:
            with c3:
                sex_opts = top_values(table, sex, limit=10)
                selected = st.multiselect("Sexo", sex_opts, default=[], key=f"sex_filter_{source}")
            if selected:
                clauses.append(f"{sex} IN ({', '.join(qstr(x) for x in selected)})")
        mun = exprs.get("mun_res_label") or exprs.get("mun_res")
        if mun:
            with c4:
                mun_opts = top_values(table, mun, limit=50)
                selected_mun = st.multiselect("Município de residência", mun_opts, default=[], key=f"mun_filter_{source}")
            if selected_mun:
                clauses.append(f"{mun} IN ({', '.join(qstr(x) for x in selected_mun)})")

        c5, c6, c7 = st.columns(3)
        cid_type = exprs.get("cid_type")
        if cid_type:
            with c5:
                cid_opts = top_values(table, cid_type, limit=20)
                selected_cid = st.multiselect("Tipo CID-10", cid_opts, default=[], key=f"cidtype_filter_{source}")
            if selected_cid:
                clauses.append(f"{cid_type} IN ({', '.join(qstr(x) for x in selected_cid)})")
        if source == "SINAN":
            if exprs.get("classi_label"):
                with c6:
                    opts = top_values(table, exprs["classi_label"], limit=10)
                    selected_classi = st.multiselect("CLASSI_FIN", opts, default=[], key=f"classi_filter_{source}")
                if selected_classi:
                    clauses.append(f"{exprs['classi_label']} IN ({', '.join(qstr(x) for x in selected_classi)})")
            if exprs.get("con_group"):
                with c7:
                    opts = top_values(table, exprs["con_group"], limit=20)
                    selected_con = st.multiselect("Grupo etiológico SINAN", opts, default=[], key=f"con_filter_{source}")
                if selected_con:
                    clauses.append(f"{exprs['con_group']} IN ({', '.join(qstr(x) for x in selected_con)})")
        elif source == "CIHA" and exprs.get("modalidade_label"):
            with c6:
                opts = top_values(table, exprs["modalidade_label"], limit=10)
                selected_mod = st.multiselect("Modalidade", opts, default=[], key=f"modalidade_filter_{source}")
            if selected_mod:
                clauses.append(f"{exprs['modalidade_label']} IN ({', '.join(qstr(x) for x in selected_mod)})")

    base_where = sql_where(clauses)
    graph_where = append_clause(base_where, definition_clause)
    return base_where, graph_where, definition


def render_kpis(table: LoadedTable, source: str, base_where: str, graph_where: str, exprs: Dict[str, Optional[str]]) -> None:
    total_base = count_rows(table, base_where)
    total_graph = count_rows(table, graph_where)
    bounds = minmax_date(table, exprs.get("dt"), graph_where) if exprs.get("dt") else None
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Registros após filtros-base", f"{total_base:,}".replace(",", "."))
    k2.metric("Registros nos gráficos", f"{total_graph:,}".replace(",", "."))
    if bounds:
        k3.metric("Data mínima", str(bounds[0].date()))
        k4.metric("Data máxima", str(bounds[1].date()))
    else:
        k3.metric("Data mínima", "—")
        k4.metric("Data máxima", "—")
    if source == "SINAN" and exprs.get("classi_code"):
        confirmed = count_rows(table, append_clause(base_where, f"{exprs['classi_code']} = '1'"))
        k5.metric("Confirmados", f"{confirmed:,}".replace(",", "."), f"{confirmed / total_base * 100:.1f}%" if total_base else None)
    elif source == "CIHA" and exprs.get("morte_code"):
        deaths = count_rows(table, append_clause(base_where, f"{exprs['morte_code']} = '1'"))
        k5.metric("Mortes CIHA", f"{deaths:,}".replace(",", "."), f"{deaths / total_base * 100:.1f}%" if total_base else None)
    elif source == "SIM" and exprs.get("causabas_cid"):
        cause = count_rows(table, append_clause(base_where, f"{exprs['causabas_cid']} IS NOT NULL"))
        k5.metric("Causa básica meningite", f"{cause:,}".replace(",", "."), f"{cause / total_base * 100:.1f}%" if total_base else None)
    else:
        k5.metric("Tipo CID-10", "configurado" if exprs.get("cid") else "não configurado")


def render_temporal_tab(table: LoadedTable, source: str, graph_where: str, exprs: Dict[str, Optional[str]]) -> None:
    dt = exprs.get("dt")
    if not dt:
        st.warning("Configure uma coluna de data para gerar a série temporal.")
        return
    c1, c2 = st.columns([1, 2])
    with c1:
        freq_label = st.selectbox("Agregação", ["Ano", "Mês", "Semana"], index=1, key=f"freq_{source}")
    freq = {"Ano": "year", "Mês": "month", "Semana": "week"}[freq_label]
    cat_options = {"Nenhuma": None}
    if exprs.get("cid_type"):
        cat_options["Tipo CID-10"] = exprs["cid_type"]
    if source == "SINAN" and exprs.get("con_group"):
        cat_options["Grupo etiológico SINAN"] = exprs["con_group"]
    if source == "SINAN" and exprs.get("sinan_cid10_conversion_type"):
        cat_options["CID-10 convertido SINAN"] = exprs["sinan_cid10_conversion_type"]
    if source == "SINAN" and exprs.get("classi_label"):
        cat_options["CLASSI_FIN"] = exprs["classi_label"]
    if exprs.get("sex"):
        cat_options["Sexo"] = exprs["sex"]
    with c2:
        cat_label = st.selectbox("Estratificar por", list(cat_options.keys()), key=f"ts_cat_{source}")
    ts = query_timeseries(table, dt, graph_where, freq, cat_options[cat_label])
    if ts.empty:
        st.info("Sem dados para a série temporal com os filtros atuais.")
    elif cat_options[cat_label]:
        fig = px.line(ts, x="periodo", y="n", color="categoria", markers=True, title="Série temporal estratificada", labels={"periodo": "Período", "n": "Registros", "categoria": cat_label})
        st.plotly_chart(fig, use_container_width=True)
        download_button(ts, f"{source.lower()}_serie_temporal_estratificada.csv")
    else:
        fig = px.line(ts, x="periodo", y="n", markers=True, title="Série temporal", labels={"periodo": "Período", "n": "Registros"})
        st.plotly_chart(fig, use_container_width=True)
        download_button(ts, f"{source.lower()}_serie_temporal.csv")

    heat = query_heatmap(table, dt, graph_where)
    if not heat.empty:
        pivot = heat.pivot(index="ano", columns="mes", values="n").fillna(0)
        pivot = pivot.reindex(sorted(pivot.index))
        pivot = pivot.reindex(columns=list(range(1, 13)), fill_value=0)
        month_labels = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
        fig = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=month_labels,
                y=[str(int(x)) for x in pivot.index],
                hovertemplate="Ano %{y}<br>Mês %{x}<br>Registros %{z}<extra></extra>",
            )
        )
        fig.update_layout(title="Sazonalidade: ano × mês", xaxis_title="Mês", yaxis_title="Ano")
        st.plotly_chart(fig, use_container_width=True)
        download_button(heat, f"{source.lower()}_heatmap_ano_mes.csv", "Baixar dados do heatmap")


def render_indicators_tab(table: LoadedTable, source: str, base_where: str, exprs: Dict[str, Optional[str]]) -> None:
    st.markdown("Os indicadores desta aba usam os filtros-base, mas **não** aplicam a definição escolhida para os gráficos exploratórios. Isso evita esconder confirmados/descartados dentro de uma mesma tabela.")

    def br_int(value: object) -> str:
        if pd.isna(value):
            return "—"
        return f"{int(value):,}".replace(",", ".")

    def br_pct(value: object) -> str:
        if pd.isna(value):
            return "—"
        return f"{float(value):.1f}%".replace(".", ",")

    def count_pct_text(n: object, pct: object = None) -> str:
        if pct is None or pd.isna(pct):
            return br_int(n)
        return f"{br_int(n)} ({br_pct(pct)})"

    def add_text_column(df: pd.DataFrame, n_col: str = "n", pct_col: str = "pct") -> pd.DataFrame:
        if df is None or df.empty:
            return df
        out = df.copy()
        out["texto"] = [count_pct_text(n, pct) for n, pct in zip(out[n_col], out[pct_col])]
        return out

    if source == "SINAN":
        ind = query_sinan_indicators(table, exprs, base_where)
        if ind.empty:
            st.warning("Não foi possível calcular indicadores do SINAN. Verifique CLASSI_FIN, EVOLUCAO e data.")
            return
        st.dataframe(ind, use_container_width=True, hide_index=True)
        download_button(ind, "sinan_indicadores_anuais.csv")

        count_specs = [
            ("notificacoes", "Notificações", None),
            ("confirmados", "Confirmados", "pct_confirmacao"),
            ("descartados", "Descartados", "pct_descarte"),
            ("obitos_meningite_confirmados", "Óbitos por meningite confirmados", "pct_obitos_meningite_confirmados_notificacoes"),
        ]
        count_rows = []
        for _, row in ind.iterrows():
            for n_col, label, pct_col in count_specs:
                pct = row[pct_col] if pct_col and pct_col in ind.columns else None
                count_rows.append({
                    "ano": row["ano"],
                    "indicador": label,
                    "n": row[n_col],
                    "pct": pct,
                    "texto": count_pct_text(row[n_col], pct),
                    "denominador_pct": row["notificacoes"] if pct_col else None,
                })
        count_long = pd.DataFrame(count_rows)
        fig = px.line(
            count_long,
            x="ano",
            y="n",
            color="indicador",
            markers=True,
            text="texto",
            title="SINAN: notificações, confirmados, descartados e óbitos",
            labels={"ano": "Ano", "n": "Registros", "indicador": "Indicador", "pct": "% das notificações"},
            hover_data={"texto": False, "pct": ":.2f", "denominador_pct": True},
        )
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, use_container_width=True)

        prop_specs = [
            ("pct_confirmacao", "Confirmados", "confirmados", "notificacoes", "% das notificações"),
            ("pct_descarte", "Descartados", "descartados", "notificacoes", "% das notificações"),
            ("pct_inconclusivos", "Inconclusivos", "inconclusivos", "notificacoes", "% das notificações"),
            ("pct_sem_classificacao", "Sem confirmação/ignorados", "sem_classificacao", "notificacoes", "% das notificações"),
            ("letalidade_confirmados_evolucao_conhecida", "Letalidade entre confirmados com evolução conhecida", "obitos_meningite_confirmados", "confirmados_evolucao_conhecida", "% dos confirmados com evolução conhecida"),
        ]
        prop_rows = []
        for _, row in ind.iterrows():
            for pct_col, label, n_col, denom_col, denom_label in prop_specs:
                pct = row[pct_col] if pct_col in ind.columns else None
                n_abs = row[n_col] if n_col in ind.columns else None
                denom = row[denom_col] if denom_col in ind.columns else None
                prop_rows.append({
                    "ano": row["ano"],
                    "indicador": label,
                    "pct": pct,
                    "n": n_abs,
                    "denominador": denom,
                    "denominador_label": denom_label,
                    "texto": f"{br_pct(pct)} (n={br_int(n_abs)})",
                })
        prop_long = pd.DataFrame(prop_rows)
        fig2 = px.line(
            prop_long,
            x="ano",
            y="pct",
            color="indicador",
            markers=True,
            text="texto",
            title="SINAN: proporções, inconclusivos, ignorados e letalidade (%)",
            labels={"ano": "Ano", "pct": "%", "indicador": "Indicador", "n": "Valor absoluto"},
            hover_data={"texto": False, "n": True, "denominador": True, "denominador_label": True},
        )
        fig2.update_traces(textposition="top center")
        st.plotly_chart(fig2, use_container_width=True)

        if exprs.get("hospital_label") and exprs.get("dt"):
            hosp = query_yearly_category(table, exprs["dt"], exprs["hospital_label"], base_where)
            if not hosp.empty:
                st.markdown("**Internação/hospitalização informada no SINAN**")
                hosp = add_text_column(hosp)
                fig_hosp = px.bar(
                    hosp,
                    x="ano",
                    y="n",
                    color="categoria",
                    text="texto",
                    title="SINAN: internação/hospitalização informada (ATE_HOSPIT)",
                    labels={"ano": "Ano", "n": "Registros", "categoria": "ATE_HOSPIT", "pct": "% no ano"},
                    hover_data={"texto": False, "pct": ":.2f", "total_ano": True},
                )
                st.plotly_chart(fig_hosp, use_container_width=True)
                st.dataframe(hosp, use_container_width=True, hide_index=True)
                download_button(hosp, "sinan_internacao_ate_hospit.csv")
        else:
            st.info("Para gerar o gráfico de internação, selecione o campo ATE_HOSPIT na configuração de colunas do SINAN.")

        etio = query_sinan_etiology_lethality(table, exprs, base_where)
        if not etio.empty:
            st.markdown("**Letalidade por grupo etiológico entre confirmados**")
            st.dataframe(etio, use_container_width=True, hide_index=True)
            etio = etio.copy()
            etio["texto"] = [f"{br_pct(p)} (n={br_int(n)})" for p, n in zip(etio["letalidade_pct"], etio["obitos_meningite"])]
            fig3 = px.bar(etio, x="letalidade_pct", y="grupo_etiologico", orientation="h", text="texto", title="Letalidade por grupo etiológico (%)", labels={"letalidade_pct": "%", "grupo_etiologico": "Grupo etiológico"})
            fig3.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig3, use_container_width=True)
            download_button(etio, "sinan_letalidade_por_etiologia.csv")
        by_year = query_sinan_diagnostics_by_year(table, exprs, base_where)
        if not by_year.empty:
            fig4 = px.area(by_year, x="ano", y="confirmados", color="grupo_etiologico", title="Confirmados por grupo etiológico — SINAN")
            st.plotly_chart(fig4, use_container_width=True)
            download_button(by_year, "sinan_confirmados_por_etiologia_ano.csv")
        return

    if source == "SIM":
        ind = query_sim_indicators(table, exprs, base_where)
        if ind.empty:
            st.warning("Não foi possível calcular indicadores principais do SIM. Verifique data, CAUSABAS e campos CID.")
        else:
            st.dataframe(ind, use_container_width=True, hide_index=True)
            download_button(ind, "sim_indicadores_anuais.csv")
            fig = px.line(ind, x="ano", y=["obitos_registros", "obitos_causa_basica_meningite", "obitos_com_mencao_meningite"], markers=True, title="SIM: óbitos por definição de CID")
            st.plotly_chart(fig, use_container_width=True)
            fig2 = px.line(ind, x="ano", y=["pct_causa_basica_meningite", "pct_mencao_meningite"], markers=True, title="SIM: percentual com causa básica/menção de meningite")
            st.plotly_chart(fig2, use_container_width=True)

        if exprs.get("dt") and exprs.get("obitograv_label"):
            grav = query_yearly_category(table, exprs["dt"], exprs["obitograv_label"], base_where)
            if not grav.empty:
                st.markdown("**Óbito na gravidez — gráfico específico**")
                grav = add_text_column(grav)
                fig_grav = px.bar(
                    grav,
                    x="ano",
                    y="n",
                    color="categoria",
                    text="texto",
                    title="SIM: óbito na gravidez (OBITOGRAV)",
                    labels={"ano": "Ano", "n": "Óbitos", "categoria": "OBITOGRAV", "pct": "% no ano"},
                    hover_data={"texto": False, "pct": ":.2f", "total_ano": True},
                )
                st.plotly_chart(fig_grav, use_container_width=True)
                st.dataframe(grav, use_container_width=True, hide_index=True)
                download_button(grav, "sim_obito_gravidez_obitograv.csv")
        else:
            st.info("Para o gráfico de óbito na gravidez, selecione o campo OBITOGRAV na configuração de colunas do SIM.")

        if exprs.get("dt") and exprs.get("obitopuerp_label"):
            puer = query_yearly_category(table, exprs["dt"], exprs["obitopuerp_label"], base_where)
            if not puer.empty:
                st.markdown("**Óbito no puerpério — gráfico específico**")
                puer = add_text_column(puer)
                fig_puer = px.bar(
                    puer,
                    x="ano",
                    y="n",
                    color="categoria",
                    text="texto",
                    title="SIM: óbito no puerpério (OBITOPUERP)",
                    labels={"ano": "Ano", "n": "Óbitos", "categoria": "OBITOPUERP", "pct": "% no ano"},
                    hover_data={"texto": False, "pct": ":.2f", "total_ano": True},
                )
                st.plotly_chart(fig_puer, use_container_width=True)
                st.dataframe(puer, use_container_width=True, hide_index=True)
                download_button(puer, "sim_obito_puerperio_obitopuerp.csv")
        return

    ind = query_ciha_indicators(table, exprs, base_where)
    st.info(
        "**Morte administrativa** é a contagem operacional do campo `MORTE = 1` na CIHA. "
        "Ela registra desfecho administrativo no atendimento e não substitui, sozinha, a causa básica do óbito do SIM.\n\n"
        "**Permanência zero** é a contagem de registros com `DIAS_PERM = 0`, isto é, sem dia completo de permanência registrado. "
        "Pode representar atendimento sem pernoite, saída no mesmo dia ou forma de preenchimento administrativo, conforme a regra da base."
    )
    if ind.empty:
        st.warning("Não foi possível calcular indicadores da CIHA. Verifique data, diagnóstico e campos MORTE/DIAS_PERM.")
    else:
        st.dataframe(ind, use_container_width=True, hide_index=True)
        download_button(ind, "ciha_indicadores_anuais.csv")
        fig = px.line(ind, x="ano", y=["atendimentos", "atendimentos_diag_principal_meningite", "mortes_administrativas"], markers=True, title="CIHA: atendimentos e mortes administrativas")
        st.plotly_chart(fig, use_container_width=True)
        fig2 = px.line(ind, x="ano", y=["pct_morte_administrativa", "pct_permanencia_zero"], markers=True, title="CIHA: morte administrativa e permanência zero (%)")
        st.plotly_chart(fig2, use_container_width=True)

    dias_dist = query_ciha_dias_perm_distribution(table, exprs, base_where)
    if not dias_dist.empty:
        st.markdown("**Distribuição dos dias de permanência**")
        dias_dist = add_text_column(dias_dist)
        fig_dias = px.bar(
            dias_dist,
            x="faixa_dias_perm",
            y="n",
            text="texto",
            title="CIHA: dias de permanência",
            labels={"faixa_dias_perm": "Dias de permanência", "n": "Atendimentos", "pct": "%"},
            hover_data={"texto": False, "pct": ":.2f", "denominador": True},
        )
        st.plotly_chart(fig_dias, use_container_width=True)
        st.dataframe(dias_dist, use_container_width=True, hide_index=True)
        download_button(dias_dist, "ciha_dias_permanencia_distribuicao.csv")
    else:
        st.info("Para gerar o gráfico de dias de permanência, selecione o campo DIAS_PERM na configuração de colunas da CIHA.")


def render_cid_tab(table: LoadedTable, source: str, graph_where: str, exprs: Dict[str, Optional[str]]) -> None:
    def br_int(value: object) -> str:
        if pd.isna(value):
            return "—"
        return f"{int(value):,}".replace(",", ".")

    def br_pct(value: object) -> str:
        if pd.isna(value):
            return "—"
        return f"{float(value):.1f}%".replace(".", ",")

    def add_text(df: pd.DataFrame, pct_col: str = "pct") -> pd.DataFrame:
        out = df.copy()
        out["texto"] = [f"{br_int(n)} ({br_pct(pct)})" for n, pct in zip(out["n"], out[pct_col])]
        return out

    if source != "SINAN":
        st.markdown("### CID-10 do registro")
        render_cid_reference()
        cid_dist = query_cid_distribution(table, exprs, graph_where)
        if cid_dist.empty:
            st.warning("Selecione ao menos um campo CID-10 válido para ativar esta análise.")
        else:
            cid_dist = add_text(cid_dist)
            fig = px.bar(
                cid_dist,
                x="n",
                y="tipo",
                orientation="h",
                text="texto",
                title="Distribuição por tipo CID-10",
                labels={"tipo": "Tipo CID-10", "n": "Registros", "pct": "%"},
                hover_data={"texto": False, "pct": ":.2f", "cids_encontrados": True, "campos_origem": True},
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(cid_dist, use_container_width=True, hide_index=True)
            download_button(cid_dist, f"{source.lower()}_cid10_distribuicao.csv")

        if source == "CIHA":
            st.markdown("### Óbitos CIHA — CID-10 destes")
            morte = exprs.get("morte_code")
            if not morte:
                st.info("Para mostrar os óbitos da CIHA e seus CID-10, selecione o campo MORTE na configuração de colunas.")
            elif not exprs.get("cid"):
                st.info("Para mostrar o CID-10 dos óbitos da CIHA, selecione ao menos um campo de diagnóstico/CID-10.")
            else:
                death_where = append_clause(graph_where, f"{morte} = '1'")
                total_deaths = count_rows(table, death_where)
                st.metric("Óbitos CIHA no recorte atual", f"{total_deaths:,}".replace(",", "."))
                if total_deaths == 0:
                    st.info("Não há registros com MORTE = 1 no recorte atual.")
                else:
                    death_cid = query_ciha_death_cid_distribution(table, exprs, graph_where)
                    if death_cid.empty:
                        st.warning("Há óbitos no recorte, mas não foi possível tabular CID-10 para esses registros.")
                    else:
                        death_cid = add_text(death_cid)
                        fig_death = px.bar(
                            death_cid,
                            x="n",
                            y="tipo",
                            orientation="h",
                            text="texto",
                            title="CIHA: CID-10 dos registros com morte administrativa",
                            labels={"tipo": "Tipo CID-10", "n": "Óbitos CIHA", "pct": "% dos óbitos"},
                            hover_data={"texto": False, "pct": ":.2f", "cids_encontrados": True, "campos_origem": True},
                        )
                        fig_death.update_layout(yaxis={"categoryorder": "total ascending"})
                        st.plotly_chart(fig_death, use_container_width=True)
                        st.dataframe(death_cid, use_container_width=True, hide_index=True)
                        download_button(death_cid, "ciha_obitos_cid10_distribuicao.csv")
        return

    st.markdown("### Classificação específica do SINAN")
    st.info(
        "No SINAN, o CID bruto do agravo pode estar como G039 para quase todos os registros. "
        "Por isso, nesta aba o gráfico bruto de distribuição por CID-10 foi substituído pela conversão de CON_DIAGES/grupo etiológico para famílias CID-10 comparáveis com SIM e CIHA."
    )

    # O gráfico CON_DIAGES detalhado foi removido porque o Grupo etiológico SINAN já é derivado do mesmo campo,
    # com agrupamento mais adequado para leitura epidemiológica.
    for label, expr in [
        ("CLASSI_FIN", exprs.get("classi_label")),
        ("Grupo etiológico SINAN", exprs.get("con_group")),
    ]:
        if expr:
            df = query_category(table, expr, graph_where, top_n=40)
            if not df.empty:
                st.markdown(f"**{label}**")
                fig = px.bar(df, x="n", y="categoria", orientation="h", text="pct", labels={"categoria": label, "n": "Registros"})
                fig.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(df, use_container_width=True, hide_index=True)

        if label == "Grupo etiológico SINAN":
            conv = query_sinan_cid10_conversion(table, exprs, graph_where)
            if not conv.empty:
                conv_yes = conv[conv["incluido_comparacao"].eq("Sim")].copy()
                conv_no = conv[~conv["incluido_comparacao"].eq("Sim")].copy()

                st.markdown("**CID-10 / classificação — conversão do grupo etiológico SINAN**")
                if conv_yes.empty:
                    st.warning("Não há registros com CON_DIAGES conversível pela regra definida.")
                else:
                    conv_yes = add_text(conv_yes)
                    fig_conv = px.bar(
                        conv_yes,
                        x="n",
                        y="cid10_classificacao",
                        orientation="h",
                        text="texto",
                        title="SINAN: conversão dos grupos etiológicos para CID-10",
                        labels={"cid10_classificacao": "CID-10 convertido", "n": "Registros", "pct": "%"},
                        hover_data={"texto": False, "pct": ":.2f", "denominador": True, "grupos_sinan": True, "conclusoes_sinan": True},
                    )
                    fig_conv.update_layout(yaxis={"categoryorder": "total ascending"})
                    st.plotly_chart(fig_conv, use_container_width=True)
                    st.dataframe(
                        conv_yes[["cid10_grupo", "cid10_classificacao", "n", "pct", "grupos_sinan", "conclusoes_sinan"]],
                        use_container_width=True,
                        hide_index=True,
                    )
                    download_button(conv_yes, "sinan_cid10_conversao_grupo_etiologico.csv")

                with st.expander("Regra usada para converter CON_DIAGES em CID-10"):
                    st.dataframe(pd.DataFrame(SINAN_CID10_MAPPING_ROWS), use_container_width=True, hide_index=True)
                    st.caption(
                        "Observação: CON_DIAGES 01 (meningococcemia isolada) fica fora da conversão; "
                        "CON_DIAGES 02 e 03 entram como A39.0. A categoria 08 foi mapeada para G02 por coerência com o CID-10, "
                        "pois no SINAN ela representa outras etiologias infecciosas/parasitárias."
                    )

                if not conv_no.empty:
                    st.caption("Registros não convertidos para a comparação CID-10, mantendo transparência da exclusão/ausência de mapeamento:")
                    st.dataframe(
                        conv_no[["cid10_grupo", "cid10_classificacao", "n", "pct", "conclusoes_sinan"]],
                        use_container_width=True,
                        hide_index=True,
                    )

    for label, expr in [
        ("EVOLUCAO", exprs.get("evol_label")),
        ("Critério de confirmação para classificação do caso", exprs.get("criterio_label")),
        ("Punção Laboratorial", exprs.get("puncao_label")),
        ("Exame Quimiocitológico", exprs.get("quimio_label")),
    ]:
        if expr:
            df = query_category(table, expr, graph_where, top_n=40)
            if not df.empty:
                df = add_text(df)
                st.markdown(f"**{label}**")
                fig = px.bar(
                    df,
                    x="n",
                    y="categoria",
                    orientation="h",
                    text="texto",
                    labels={"categoria": label, "n": "Registros", "pct": "%"},
                    hover_data={"texto": False, "pct": ":.2f"},
                )
                fig.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(df, use_container_width=True, hide_index=True)

    quimio_summary = query_sinan_quimio_summary(table, exprs, graph_where)
    if quimio_summary.empty:
        st.info(
            "Para gerar o resumo do Exame Quimiocitológico, selecione os campos laboratoriais do SINAN "
            "na configuração de colunas, como LAB_GLICO, LAB_LEUCO e LAB_PROT."
        )
    else:
        st.markdown("**Exame Quimiocitológico — valores dos parâmetros**")
        resumo_plot = quimio_summary[quimio_summary["n_valido"] > 0].copy()
        if not resumo_plot.empty:
            resumo_plot["texto"] = [
                f"mediana {float(med):.1f}".replace(".", ",") if pd.notna(med) else "—"
                for med in resumo_plot["mediana"]
            ]
            fig_quimio = px.bar(
                resumo_plot,
                x="parametro",
                y="mediana",
                text="texto",
                title="SINAN: mediana dos parâmetros do exame quimiocitológico",
                labels={"parametro": "Parâmetro", "mediana": "Mediana", "n_valido": "Registros válidos"},
                hover_data={
                    "texto": False,
                    "n_valido": True,
                    "pct_preenchido": ":.2f",
                    "minimo": ":.2f",
                    "q1": ":.2f",
                    "media": ":.2f",
                    "q3": ":.2f",
                    "maximo": ":.2f",
                },
            )
            st.plotly_chart(fig_quimio, use_container_width=True)
        st.dataframe(quimio_summary, use_container_width=True, hide_index=True)
        download_button(quimio_summary, "sinan_quimiocitologico_resumo_parametros.csv")

        for key, titulo in [("glico", "Glicose"), ("prot", "Proteínas"), ("leuco", "Leucócitos")]:
            expr = exprs.get(f"lab_{key}")
            if not expr:
                st.info(f"Para gerar a distribuição de {titulo}, selecione o campo correspondente na configuração de colunas do SINAN.")
                continue
            dist = query_sinan_numeric_distribution(table, expr, graph_where)
            if dist.empty:
                st.info(f"Não há valores numéricos válidos para {titulo} no recorte atual.")
                continue
            dist = add_text(dist)
            st.markdown(f"**Distribuição — {titulo}**")
            fig_dist = px.bar(
                dist,
                x="faixa",
                y="n",
                text="texto",
                title=f"SINAN: distribuição de {titulo}",
                labels={"faixa": titulo, "n": "Registros", "pct": "%"},
                hover_data={"texto": False, "pct": ":.2f", "denominador": True, "faixa_inicio": ":.2f", "faixa_fim": ":.2f"},
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            st.dataframe(dist, use_container_width=True, hide_index=True)
            download_button(dist, f"sinan_quimiocitologico_distribuicao_{safe_filename(titulo)}.csv")


def render_demography_tab(table: LoadedTable, source: str, graph_where: str, exprs: Dict[str, Optional[str]]) -> None:
    def br_int(value: object) -> str:
        if pd.isna(value):
            return "—"
        return f"{int(value):,}".replace(",", ".")

    def br_pct(value: object) -> str:
        if pd.isna(value):
            return "—"
        return f"{float(value):.1f}%".replace(".", ",")

    def add_text(df: pd.DataFrame, pct_col: str = "pct") -> pd.DataFrame:
        out = df.copy()
        out["texto"] = [f"{br_int(n)} ({br_pct(pct)})" for n, pct in zip(out["n"], out[pct_col])]
        return out

    age = exprs.get("age")
    if not age:
        st.warning("Configure idade para gerar os gráficos etários. As categorias territoriais ainda podem ser exibidas abaixo.")
    else:
        age_df = query_age_dist(table, age, graph_where)
        if not age_df.empty:
            age_df["faixa"] = age_df["faixa_ini"].astype(int).astype(str) + "–" + (age_df["faixa_ini"].astype(int) + 4).astype(str)
            age_df["denominador"] = int(age_df["n"].sum())
            age_df["pct"] = np.where(age_df["denominador"].gt(0), (age_df["n"] / age_df["denominador"] * 100).round(2), np.nan)
            age_df = add_text(age_df)
            fig = px.bar(
                age_df,
                x="faixa",
                y="n",
                text="texto",
                title="Distribuição por faixa etária de 5 anos",
                labels={"faixa": "Faixa etária", "n": "Registros", "pct": "%", "denominador": "Denominador"},
                hover_data={"texto": False, "pct": ":.2f", "denominador": True},
            )
            fig.update_traces(textposition="outside", cliponaxis=False)
            st.plotly_chart(fig, use_container_width=True)
            download_button(age_df, f"{source.lower()}_idade.csv")
        sex = exprs.get("sex")
        if sex:
            pyr = query_age_dist(table, age, graph_where, sex_sql=sex)
            if not pyr.empty:
                pyr["faixa"] = pyr["faixa_ini"].astype(int).astype(str) + "–" + (pyr["faixa_ini"].astype(int) + 4).astype(str)
                pyr["valor"] = np.where(pyr["sexo"].eq("Masculino"), -pyr["n"], pyr["n"])
                fig = px.bar(pyr, x="valor", y="faixa", color="sexo", orientation="h", title="Pirâmide etária por sexo", labels={"valor": "Registros", "faixa": "Faixa etária"})
                fig.update_layout(barmode="relative")
                st.plotly_chart(fig, use_container_width=True)
                download_button(pyr, f"{source.lower()}_piramide.csv")

    sex = exprs.get("sex")
    cols = []
    if sex:
        cols.append(("Sexo", sex, False, 25))
    if exprs.get("race"):
        cols.append(("Raça/cor", exprs["race"], False, 25))
    if exprs.get("mun_res_label") or exprs.get("mun_res"):
        cols.append(("Município de residência", exprs.get("mun_res_label") or exprs["mun_res"], True, 15))
    if exprs.get("mun_event_label") or exprs.get("mun_event"):
        cols.append(("Município de ocorrência/atendimento/notificação", exprs.get("mun_event_label") or exprs["mun_event"], True, 15))
    if cols:
        st.markdown("### Categorias demográficas e territoriais")
        if any(is_mun for _, _, is_mun, _ in cols):
            top_municipios = st.slider("Municípios principais no gráfico territorial", 10, 15, 15, key=f"top_municipios_{source}")
            st.caption("Nos gráficos de município, todas as categorias fora do Top N são somadas em 'Outros municípios'; assim o percentual e o denominador continuam representando 100% dos dados filtrados.")
        else:
            top_municipios = 15
        for label, expr, is_mun, top_n in cols:
            if is_mun:
                df = query_category_top_with_outros(table, expr, graph_where, top_n=top_municipios)
                filename = f"{source.lower()}_{safe_filename(label)}_top{top_municipios}_outros.csv"
            else:
                df = query_category(table, expr, graph_where, top_n=top_n)
                if not df.empty:
                    df["denominador"] = df["n"].sum()
                filename = f"{source.lower()}_{safe_filename(label)}.csv"
            if not df.empty:
                df = add_text(df)
                title = label if not is_mun else f"{label} — Top {top_municipios} + Outros municípios"
                fig = px.bar(
                    df,
                    x="n",
                    y="categoria",
                    orientation="h",
                    text="texto",
                    title=title,
                    labels={"categoria": label, "n": "Registros", "pct": "%", "denominador": "Denominador"},
                    hover_data={"texto": False, "pct": ":.2f", "denominador": True},
                )
                if is_mun:
                    fig.update_layout(yaxis={"categoryorder": "array", "categoryarray": df["categoria"].tolist()[::-1]})
                else:
                    fig.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(df, use_container_width=True, hide_index=True)
                download_button(df, filename)


def render_quality_tab(table: LoadedTable, source: str, base_where: str, exprs: Dict[str, Optional[str]]) -> None:
    st.markdown("### Campos importantes não preenchidos")
    st.caption(
        "Esta aba usa os filtros-base e mede, para cada campo-chave configurado, quantos registros estão sem preenchimento válido. "
        "Os gráficos mostram a porcentagem e também o número absoluto de registros não preenchidos sobre o total analisado."
    )

    fields = {
        "data": exprs.get("dt"),
        "sexo": exprs.get("sex"),
        "idade": exprs.get("age"),
        "raça/cor": exprs.get("race"),
        "município residência": exprs.get("mun_res"),
        "município ocorrência/atendimento": exprs.get("mun_event"),
        "CID meningite detectado": exprs.get("cid"),
    }
    if source == "SINAN":
        fields.update({
            "CLASSI_FIN": exprs.get("classi_code"),
            "CON_DIAGES": exprs.get("con_code"),
            "EVOLUCAO": exprs.get("evol_code"),
            "CRITERIO": exprs.get("criterio_code"),
            "Punção Laboratorial": exprs.get("puncao_label"),
            "Exame Quimiocitológico": exprs.get("quimio_label"),
            "Glicose": exprs.get("lab_glico"),
            "Proteínas": exprs.get("lab_prot"),
            "Leucócitos": exprs.get("lab_leuco"),
        })
    elif source == "CIHA":
        fields.update({"MORTE": exprs.get("morte_code"), "DIAS_PERM": exprs.get("dias_perm"), "MODALIDADE": exprs.get("modalidade_label")})
    elif source == "SIM":
        fields.update({"CAUSABAS CID": exprs.get("causabas_cid"), "CAUSABAS_O CID": exprs.get("causabas_o_cid")})

    def fmt_int(value: object) -> str:
        if pd.isna(value):
            return "0"
        return f"{int(value):,}".replace(",", ".")

    def fmt_pct(value: object) -> str:
        if pd.isna(value):
            return "—"
        return f"{float(value):.2f}%".replace(".", ",")

    def add_missing_text(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["texto"] = out.apply(
            lambda r: f"{fmt_int(r['faltantes'])} de {fmt_int(r['total'])} ({fmt_pct(r['pct_faltante'])})",
            axis=1,
        )
        return out

    miss = query_missingness(table, fields, exprs.get("dt"), base_where)
    if miss.empty:
        st.info("Sem campos configurados para avaliar preenchimento.")
    else:
        miss = add_missing_text(miss)
        fig = px.bar(
            miss,
            x="pct_faltante",
            y="campo",
            orientation="h",
            text="texto",
            title="Campos importantes não preenchidos — percentual e número absoluto",
            labels={
                "campo": "Campo",
                "pct_faltante": "% não preenchido",
                "faltantes": "Registros não preenchidos",
                "total": "Total analisado",
            },
            hover_data={"texto": False, "faltantes": True, "total": True, "pct_faltante": ":.2f"},
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        fig.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(miss[["campo", "faltantes", "total", "pct_faltante", "texto"]], use_container_width=True, hide_index=True)
        download_button(miss.drop(columns=["texto"], errors="ignore"), f"{source.lower()}_campos_importantes_nao_preenchidos.csv")

    by_year = query_missingness_by_year(table, fields, exprs.get("dt"), base_where)
    if not by_year.empty:
        by_year = add_missing_text(by_year)
        focus_fields = st.multiselect(
            "Campos para visualizar por ano",
            sorted(by_year["campo"].unique()),
            default=sorted(by_year["campo"].unique())[:5],
            key=f"miss_fields_{source}",
        )
        filtered = by_year[by_year["campo"].isin(focus_fields)] if focus_fields else by_year
        fig = px.line(
            filtered,
            x="ano",
            y="pct_faltante",
            color="campo",
            markers=True,
            text="texto",
            title="Campos importantes não preenchidos por ano — percentual e número absoluto",
            labels={
                "ano": "Ano",
                "pct_faltante": "% não preenchido",
                "campo": "Campo",
                "faltantes": "Registros não preenchidos",
                "total": "Total analisado",
            },
            hover_data={"texto": False, "faltantes": True, "total": True, "pct_faltante": ":.2f"},
        )
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(filtered[["ano", "campo", "faltantes", "total", "pct_faltante", "texto"]], use_container_width=True, hide_index=True)
        download_button(by_year.drop(columns=["texto"], errors="ignore"), f"{source.lower()}_campos_importantes_nao_preenchidos_por_ano.csv")

def render_sql_lab(table: LoadedTable, source: str) -> None:
    st.markdown("### Laboratório SQL")
    st.caption("Use `{{tabela}}` como placeholder para a tabela carregada. O app substituirá pelo nome/referência SQL correta.")
    example = f"""
    SELECT COUNT(*) AS registros
    FROM {{tabela}};
    """
    if source == "SINAN":
        example = """
        SELECT
          EXTRACT(YEAR FROM DT_SIN_PRI) AS ano,
          CLASSI_FIN,
          CON_DIAGES,
          EVOLUCAO,
          COUNT(*) AS n
        FROM {tabela}
        GROUP BY 1, 2, 3, 4
        ORDER BY 1, 2, 3, 4;
        """
    elif source == "SIM":
        example = """
        SELECT
          SUBSTR(DTOBITO, 1, 4) AS ano,
          CAUSABAS,
          COUNT(*) AS n
        FROM {tabela}
        GROUP BY 1, 2
        ORDER BY 1, n DESC;
        """
    elif source == "CIHA":
        example = """
        SELECT
          ANO_CMPT AS ano,
          DIAG_PRINC,
          MORTE,
          COUNT(*) AS n
        FROM {tabela}
        GROUP BY 1, 2, 3
        ORDER BY 1, n DESC;
        """
    sql_text = st.text_area("SQL", value=textwrap.dedent(example).strip(), height=220, key=f"sql_lab_{source}")
    if st.button("Executar SQL", key=f"run_sql_{source}"):
        sql = sql_text.replace("{tabela}", table.ref_sql).replace("{{tabela}}", table.ref_sql)
        try:
            df = run_query(table, sql)
            st.dataframe(df, use_container_width=True, hide_index=True)
            download_button(df, f"{source.lower()}_sql_lab.csv", "Baixar resultado")
        except Exception as exc:
            st.error(f"Erro ao executar SQL: {exc}")


def render_source(source: str) -> Optional[Dict[str, object]]:
    table = render_loader(source)
    if table is None:
        return None
    try:
        schema = schema_df(table)
    except Exception as exc:
        st.error(f"Não foi possível ler o schema: {exc}")
        return None
    columns = schema["coluna"].astype(str).tolist()

    st.success(f"Dados carregados: {table.label}")

    sel = render_column_config(source, columns)
    exprs = build_expressions(source, sel)
    base_where, graph_where, definition = render_filters(source, table, exprs)
    render_kpis(table, source, base_where, graph_where, exprs)

    tabs = st.tabs(["Indicadores", "Temporal", "CID-10 / classificação", "Demografia e território", "Campos importantes não preenchidos", "Prévia", "SQL Lab"])
    with tabs[0]:
        render_indicators_tab(table, source, base_where, exprs)
    with tabs[1]:
        render_temporal_tab(table, source, graph_where, exprs)
    with tabs[2]:
        render_cid_tab(table, source, graph_where, exprs)
    with tabs[3]:
        render_demography_tab(table, source, graph_where, exprs)
    with tabs[4]:
        render_quality_tab(table, source, base_where, exprs)
    with tabs[5]:
        st.markdown("### Prévia enriquecida")
        limit = st.slider("Número de linhas", 50, 20000, 200, step=50, key=f"preview_limit_{source}")
        try:
            df_prev = query_enriched_preview(table, sel, exprs, graph_where, limit)
            st.dataframe(df_prev, use_container_width=True)
            download_button(df_prev, f"{source.lower()}_previa_enriquecida.csv")
        except Exception as exc:
            st.error(f"Erro ao montar prévia: {exc}")

        st.markdown("### Exportação completa dos casos filtrados")
        st.caption("Gera CSV com todos os casos que passam pelos filtros atuais da aba. Use esta opção quando a prévia de 20 mil linhas não for suficiente.")
        if st.button("Gerar CSV completo dos casos filtrados", key=f"full_export_{source}"):
            try:
                df_full = query_enriched_preview(table, sel, exprs, graph_where, limit=None)
                st.success(f"Exportação preparada com {len(df_full):,} linhas.".replace(",", "."))
                download_button(df_full, f"{source.lower()}_casos_filtrados_completos.csv", "Baixar CSV completo")
            except Exception as exc:
                st.error(f"Erro ao gerar exportação completa: {exc}")
    with tabs[6]:
        render_sql_lab(table, source)

    return {"source": source, "table": table, "sel": sel, "exprs": exprs, "base_where": base_where, "graph_where": graph_where, "definition": definition}


def render_comparison(loaded: Sequence[Dict[str, object]]) -> None:
    st.markdown("### Comparação entre bases")
    available = [x for x in loaded if x and x.get("exprs", {}).get("dt")]
    if len(available) < 2:
        st.info("Carregue ao menos duas bases com data configurada para comparar séries.")
        return
    source_names = [x["source"] for x in available]
    chosen = st.multiselect("Bases", source_names, default=source_names, key="comp_sources")
    freq_label = st.selectbox("Agregação", ["Ano", "Mês", "Semana"], index=1, key="comp_freq")
    freq = {"Ano": "year", "Mês": "month", "Semana": "week"}[freq_label]
    normalize = st.checkbox("Normalizar em índice 100 no primeiro período não-zero", value=False, key="comp_norm")
    stratify_cid = st.checkbox("Estratificar por tipo CID-10 quando disponível", value=False, key="comp_cid")
    st.caption("Na comparação, o SINAN entra sempre como casos confirmados (CLASSI_FIN = 1), independentemente da definição exploratória escolhida na aba SINAN. Na agregação mensal, meses sem registros são mantidos com valor zero.")

    frames = []
    for item in available:
        source_name = item["source"]
        if source_name not in chosen:
            continue
        table: LoadedTable = item["table"]
        exprs = item["exprs"]
        cat = exprs.get("cid_type") if stratify_cid else None
        series_where = item["graph_where"]
        series_label = item.get("definition", "")
        if source_name == "SINAN":
            classi = exprs.get("classi_code")
            if not classi:
                st.warning("SINAN foi ignorado na comparação porque CLASSI_FIN não está configurado; não é possível isolar confirmados.")
                continue
            series_where = append_clause(item["base_where"], f"{classi} = '1'")
            series_label = "Confirmados (CLASSI_FIN = 1)"
        try:
            ts = query_timeseries(table, exprs["dt"], series_where, freq, cat)
        except Exception as exc:
            st.warning(f"Falha na série de {source_name}: {exc}")
            continue
        if ts.empty:
            continue
        if cat:
            ts["serie"] = source_name + " — " + series_label + " — " + ts["categoria"].astype(str)
        else:
            ts["serie"] = source_name + " — " + series_label
        ts = ts.rename(columns={"n": "valor"})
        frames.append(ts[["periodo", "serie", "valor"]])
    if not frames:
        st.warning("Nenhuma série gerada.")
        return
    comp = pd.concat(frames, ignore_index=True)
    comp["periodo"] = pd.to_datetime(comp["periodo"])

    if freq == "month" and not comp.empty:
        comp["periodo"] = comp["periodo"].dt.to_period("M").dt.to_timestamp()
        full_months = pd.date_range(comp["periodo"].min(), comp["periodo"].max(), freq="MS")
        series_values = comp["serie"].dropna().unique().tolist()
        full_index = pd.MultiIndex.from_product([full_months, series_values], names=["periodo", "serie"])
        comp = (
            comp.groupby(["periodo", "serie"], as_index=False)["valor"].sum()
            .set_index(["periodo", "serie"])
            .reindex(full_index, fill_value=0)
            .reset_index()
        )

    if normalize:
        comp = comp.sort_values("periodo")
        for s in comp["serie"].unique():
            idx = comp["serie"].eq(s)
            nonzero = comp.loc[idx & comp["valor"].gt(0), "valor"]
            if not nonzero.empty:
                comp.loc[idx, "valor"] = comp.loc[idx, "valor"] / nonzero.iloc[0] * 100

    fig = px.line(comp, x="periodo", y="valor", color="serie", markers=True, title="Comparação de tendências", labels={"valor": "Índice" if normalize else "Registros", "periodo": "Período", "serie": "Série"})
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(comp, use_container_width=True, hide_index=True)
    download_button(comp, "comparacao_series_bases.csv")

    st.markdown("**Cuidados de leitura**")
    st.write(
        "SINAN mede notificações/investigações; SIM mede óbitos; CIHA mede utilização de serviços. "
        "Compare tendências, composição e concordância agregada, mas evite interpretar contagens brutas entre bases como o mesmo fenômeno sem linkage e denominadores populacionais."
    )


def render_methodology() -> None:
    st.markdown("### Como usar este app para investigação epidemiológica")
    st.markdown(
        """
        1. Comece pela aba **Indicadores** do SINAN para separar notificações, confirmados, descartados e óbitos.
        2. Use **CID-10 / classificação** para comparar o CID bruto com a classificação específica. No SINAN, dê prioridade a `CON_DIAGES`.
        3. Use **Temporal** para verificar queda, recuperação e sazonalidade.
        4. Use **Demografia e território** para levantar hipóteses por idade, sexo, residência e atendimento.
        5. Use **Prévia** para inspecionar casos filtrados e exportar a planilha completa quando necessário.
        6. Use **SQL Lab** para transformar a hipótese em uma consulta reprodutível.
        """
    )
    st.markdown("### Definições principais usadas")
    st.dataframe(
        pd.DataFrame(
            [
                ["SINAN — notificação", "todos os registros após filtros"],
                ["SINAN — confirmado", "CLASSI_FIN = 1"],
                ["SINAN — descartado", "CLASSI_FIN = 2"],
                ["SINAN — óbito por meningite", "CLASSI_FIN = 1 e EVOLUCAO = 2"],
                ["SIM — causa básica", "CID de meningite detectado em CAUSABAS"],
                ["SIM — menção", "CID de meningite detectado em CAUSABAS, linhas da DO ou ATESTADO"],
                ["SIM — óbito na gravidez", "campo OBITOGRAV quando disponível"],
                ["CIHA — atendimento", "registro administrativo com data/diagnóstico"],
                ["CIHA — morte administrativa", "MORTE = 1"],
                ["CIHA — permanência zero", "DIAS_PERM = 0"],
            ],
            columns=["Conceito", "Regra operacional"],
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("### Referência CID-10")
    render_cid_reference()


def main() -> None:
    st.title("Painel epidemiológico de meningite — SINAN, SIM e CIHA")
    st.caption(f"Versão {APP_VERSION}. Lê DuckDB ou Parquet e mantém regras analíticas explícitas.")

    with st.sidebar:
        st.header("Orientação")
        st.write("Carregue cada base em sua aba. Para os arquivos enviados, use o modo **DuckDB local** se eles estiverem na mesma pasta do app.")
        st.write("O painel é exploratório. Para conclusões finais, exporte as tabelas e valide as regras em SQL/Python.")

    tabs = st.tabs(["Metodologia", "SINAN", "SIM", "CIHA", "Comparação"])
    loaded: List[Optional[Dict[str, object]]] = []
    with tabs[0]:
        render_methodology()
    with tabs[1]:
        loaded.append(render_source("SINAN"))
    with tabs[2]:
        loaded.append(render_source("SIM"))
    with tabs[3]:
        loaded.append(render_source("CIHA"))
    with tabs[4]:
        render_comparison([x for x in loaded if x])


if __name__ == "__main__":
    main()
