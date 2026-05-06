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

APP_VERSION = "2026-05-02"


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
        ("CRITERIO", "critério", "cultura, PCR, clínico, quimiocitológico etc."),
        ("LAB_PUNCAO", "investigação", "punção lombar realizada"),
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
    ate_hospit_col: Optional[str] = None
    dt_encerramento_col: Optional[str] = None
    dt_notificacao_col: Optional[str] = None
    # SIM
    causabas_col: Optional[str] = None
    causabas_o_col: Optional[str] = None
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
        sel.lab_puncao_col = choose_candidate(columns, ["LAB_PUNCAO"])
        sel.ate_hospit_col = choose_candidate(columns, ["ATE_HOSPIT"])
        sel.dt_encerramento_col = choose_candidate(columns, ["DT_ENCERRA"])
        sel.dt_notificacao_col = choose_candidate(columns, ["DT_NOTIFIC"])
    elif source == "SIM":
        sel.causabas_col = choose_candidate(columns, ["CAUSABAS"])
        sel.causabas_o_col = choose_candidate(columns, ["CAUSABAS_O"])
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
        exprs["criterio_code"] = clean_code_expr(sel.criterio_col) if sel.criterio_col else None
        exprs["criterio_label"] = case_from_mapping(exprs["criterio_code"], SINAN_CRITERIO, "Sem critério/ignorado") if exprs["criterio_code"] else None
        exprs["puncao_label"] = case_from_mapping(clean_code_expr(sel.lab_puncao_col), YES_NO_IGN, "Sem informação") if sel.lab_puncao_col else None
        exprs["hospital_label"] = case_from_mapping(clean_code_expr(sel.ate_hospit_col), YES_NO_IGN, "Sem informação") if sel.ate_hospit_col else None
        exprs["dt_encerramento"] = date_expr(sel.dt_encerramento_col) if sel.dt_encerramento_col else None
        exprs["dt_notificacao"] = date_expr(sel.dt_notificacao_col) if sel.dt_notificacao_col else None
    elif source == "SIM":
        exprs["causabas_cid"] = cid_extract_expr([sel.causabas_col] if sel.causabas_col else [])
        exprs["causabas_o_cid"] = cid_extract_expr([sel.causabas_o_col] if sel.causabas_o_col else [])
        exprs["causabas_group"] = cid_group_expr(exprs["causabas_cid"]) if exprs["causabas_cid"] else None
        exprs["causabas_type"] = cid_type_expr(exprs["causabas_cid"]) if exprs["causabas_cid"] else None
        exprs["lococor_label"] = case_from_mapping(clean_code_expr("LOCOCOR"), SIM_LOCOCOR, "Sem informação/ignorado") if "LOCOCOR" in [sel.municipality_event_col, sel.municipality_res_col] else None
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


def query_enriched_preview(table: LoadedTable, sel: ColumnSelection, exprs: Dict[str, Optional[str]], where_sql: str, limit: int = 200) -> pd.DataFrame:
    items = []
    mapping = [
        ("data_analise", exprs.get("dt")),
        ("sexo", exprs.get("sex")),
        ("idade_anos", exprs.get("age")),
        ("raca_cor", exprs.get("race")),
        ("municipio_residencia", exprs.get("mun_res")),
        ("municipio_evento_atendimento", exprs.get("mun_event")),
        ("cid_meningite_detectado", exprs.get("cid")),
        ("tipo_cid10", exprs.get("cid_type")),
        ("campo_origem_cid", exprs.get("cid_source")),
        ("sinan_classificacao_final", exprs.get("classi_label")),
        ("sinan_conclusao_diagnostica", exprs.get("con_label")),
        ("sinan_grupo_etiologico", exprs.get("con_group")),
        ("sinan_evolucao", exprs.get("evol_label")),
        ("sinan_criterio", exprs.get("criterio_label")),
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
        sel.causabas_col,
        sel.causabas_o_col,
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
    sql = f"SELECT {', '.join(items)} FROM {table.ref_sql} {where_sql} LIMIT {int(limit)}"
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
                lab_puncao_col = select("LAB_PUNCAO", defaults.lab_puncao_col, f"puncao_{source}")
                ate_hospit_col = select("ATE_HOSPIT", defaults.ate_hospit_col, f"hospit_{source}")
            with s4:
                dt_encerramento_col = select("DT_ENCERRA", defaults.dt_encerramento_col, f"dt_enc_{source}")
                dt_notificacao_col = select("DT_NOTIFIC", defaults.dt_notificacao_col, f"dt_notif_{source}")
            return ColumnSelection(date_col, sex_col, age_col, age_unit_col, race_col, mun_res, mun_event, cid_cols, age_mode, classi_fin_col, con_diages_col, evolucao_col, criterio_col, lab_puncao_col, ate_hospit_col, dt_encerramento_col, dt_notificacao_col)

        if source == "SIM":
            st.markdown("**Campos específicos do SIM**")
            s1, s2 = st.columns(2)
            with s1:
                causabas_col = select("CAUSABAS", defaults.causabas_col, f"causabas_{source}")
            with s2:
                causabas_o_col = select("CAUSABAS_O", defaults.causabas_o_col, f"causabaso_{source}")
            return ColumnSelection(date_col, sex_col, age_col, age_unit_col, race_col, mun_res, mun_event, cid_cols, age_mode, causabas_col=causabas_col, causabas_o_col=causabas_o_col)

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
        mun = exprs.get("mun_res")
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
    if source == "SINAN":
        ind = query_sinan_indicators(table, exprs, base_where)
        if ind.empty:
            st.warning("Não foi possível calcular indicadores do SINAN. Verifique CLASSI_FIN, EVOLUCAO e data.")
            return
        st.dataframe(ind, use_container_width=True, hide_index=True)
        download_button(ind, "sinan_indicadores_anuais.csv")
        fig = px.line(ind, x="ano", y=["notificacoes", "confirmados", "descartados", "obitos_meningite_confirmados"], markers=True, title="SINAN: notificações, confirmados, descartados e óbitos")
        st.plotly_chart(fig, use_container_width=True)
        fig2 = px.line(ind, x="ano", y=["pct_confirmacao", "pct_descarte", "letalidade_confirmados_evolucao_conhecida"], markers=True, title="SINAN: proporções e letalidade (%)")
        st.plotly_chart(fig2, use_container_width=True)
        etio = query_sinan_etiology_lethality(table, exprs, base_where)
        if not etio.empty:
            st.markdown("**Letalidade por grupo etiológico entre confirmados**")
            st.dataframe(etio, use_container_width=True, hide_index=True)
            fig3 = px.bar(etio, x="letalidade_pct", y="grupo_etiologico", orientation="h", text="letalidade_pct", title="Letalidade por grupo etiológico (%)")
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
            st.warning("Não foi possível calcular indicadores do SIM. Verifique data, CAUSABAS e campos CID.")
            return
        st.dataframe(ind, use_container_width=True, hide_index=True)
        download_button(ind, "sim_indicadores_anuais.csv")
        fig = px.line(ind, x="ano", y=["obitos_registros", "obitos_causa_basica_meningite", "obitos_com_mencao_meningite"], markers=True, title="SIM: óbitos por definição de CID")
        st.plotly_chart(fig, use_container_width=True)
        fig2 = px.line(ind, x="ano", y=["pct_causa_basica_meningite", "pct_mencao_meningite"], markers=True, title="SIM: percentual com causa básica/menção de meningite")
        st.plotly_chart(fig2, use_container_width=True)
        return

    ind = query_ciha_indicators(table, exprs, base_where)
    if ind.empty:
        st.warning("Não foi possível calcular indicadores da CIHA. Verifique data, diagnóstico e campos MORTE/DIAS_PERM.")
        return
    st.dataframe(ind, use_container_width=True, hide_index=True)
    download_button(ind, "ciha_indicadores_anuais.csv")
    fig = px.line(ind, x="ano", y=["atendimentos", "atendimentos_diag_principal_meningite", "mortes_administrativas"], markers=True, title="CIHA: atendimentos e mortes administrativas")
    st.plotly_chart(fig, use_container_width=True)
    fig2 = px.line(ind, x="ano", y=["pct_morte_administrativa", "pct_permanencia_zero"], markers=True, title="CIHA: morte administrativa e permanência zero (%)")
    st.plotly_chart(fig2, use_container_width=True)


def render_cid_tab(table: LoadedTable, source: str, graph_where: str, exprs: Dict[str, Optional[str]]) -> None:
    st.markdown("### CID-10 do registro")
    render_cid_reference()
    cid_dist = query_cid_distribution(table, exprs, graph_where)
    if cid_dist.empty:
        st.warning("Selecione ao menos um campo CID-10 válido para ativar esta análise.")
    else:
        fig = px.bar(cid_dist, x="n", y="tipo", orientation="h", text="pct", title="Distribuição por tipo CID-10")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(cid_dist, use_container_width=True, hide_index=True)
        download_button(cid_dist, f"{source.lower()}_cid10_distribuicao.csv")

    if source == "SINAN":
        st.markdown("### Classificação específica do SINAN")
        st.info("No SINAN, use esta seção para interpretar a forma/etiologia. O CID bruto ID_AGRAVO pode estar como G039 para todos os registros.")
        for label, expr in [
            ("CLASSI_FIN", exprs.get("classi_label")),
            ("CON_DIAGES — conclusão diagnóstica", exprs.get("con_label")),
            ("Grupo etiológico SINAN", exprs.get("con_group")),
            ("EVOLUCAO", exprs.get("evol_label")),
            ("CRITERIO", exprs.get("criterio_label")),
            ("LAB_PUNCAO", exprs.get("puncao_label")),
        ]:
            if expr:
                df = query_category(table, expr, graph_where, top_n=40)
                if not df.empty:
                    st.markdown(f"**{label}**")
                    fig = px.bar(df, x="n", y="categoria", orientation="h", text="pct", labels={"categoria": label, "n": "Registros"})
                    fig.update_layout(yaxis={"categoryorder": "total ascending"})
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(df, use_container_width=True, hide_index=True)


def render_demography_tab(table: LoadedTable, source: str, graph_where: str, exprs: Dict[str, Optional[str]]) -> None:
    age = exprs.get("age")
    if not age:
        st.warning("Configure idade para gerar demografia.")
        return
    age_df = query_age_dist(table, age, graph_where)
    if not age_df.empty:
        age_df["faixa"] = age_df["faixa_ini"].astype(int).astype(str) + "–" + (age_df["faixa_ini"].astype(int) + 4).astype(str)
        fig = px.bar(age_df, x="faixa", y="n", title="Distribuição por faixa etária de 5 anos", labels={"faixa": "Faixa etária", "n": "Registros"})
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

    cols = []
    if sex:
        cols.append(("Sexo", sex))
    if exprs.get("race"):
        cols.append(("Raça/cor", exprs["race"]))
    if exprs.get("mun_res"):
        cols.append(("Município de residência", exprs["mun_res"]))
    if exprs.get("mun_event"):
        cols.append(("Município de ocorrência/atendimento", exprs["mun_event"]))
    if cols:
        st.markdown("### Categorias demográficas e territoriais")
        for label, expr in cols:
            df = query_category(table, expr, graph_where, top_n=25)
            if not df.empty:
                fig = px.bar(df, x="n", y="categoria", orientation="h", text="pct", title=label)
                fig.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)


def render_quality_tab(table: LoadedTable, source: str, base_where: str, exprs: Dict[str, Optional[str]]) -> None:
    fields = {
        "data": exprs.get("dt"),
        "sexo": exprs.get("sex"),
        "idade": exprs.get("age"),
        "raça/cor": exprs.get("race"),
        "município residência": exprs.get("mun_res"),
        "município evento/atendimento": exprs.get("mun_event"),
        "CID meningite detectado": exprs.get("cid"),
    }
    if source == "SINAN":
        fields.update({
            "CLASSI_FIN": exprs.get("classi_code"),
            "CON_DIAGES": exprs.get("con_code"),
            "EVOLUCAO": exprs.get("evol_code"),
            "CRITERIO": exprs.get("criterio_code"),
            "LAB_PUNCAO": exprs.get("puncao_label"),
        })
    elif source == "CIHA":
        fields.update({"MORTE": exprs.get("morte_code"), "DIAS_PERM": exprs.get("dias_perm"), "MODALIDADE": exprs.get("modalidade_label")})
    elif source == "SIM":
        fields.update({"CAUSABAS CID": exprs.get("causabas_cid"), "CAUSABAS_O CID": exprs.get("causabas_o_cid")})
    miss = query_missingness(table, fields, exprs.get("dt"), base_where)
    if miss.empty:
        st.info("Sem campos configurados para avaliar completude.")
    else:
        fig = px.bar(miss, x="pct_faltante", y="campo", orientation="h", text="pct_faltante", title="Faltantes/indisponíveis nos campos-chave")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(miss, use_container_width=True, hide_index=True)
        download_button(miss, f"{source.lower()}_completude.csv")

    by_year = query_missingness_by_year(table, fields, exprs.get("dt"), base_where)
    if not by_year.empty:
        focus_fields = st.multiselect("Campos para visualizar por ano", sorted(by_year["campo"].unique()), default=sorted(by_year["campo"].unique())[:5], key=f"miss_fields_{source}")
        filtered = by_year[by_year["campo"].isin(focus_fields)] if focus_fields else by_year
        fig = px.line(filtered, x="ano", y="pct_faltante", color="campo", markers=True, title="Percentual faltante por ano")
        st.plotly_chart(fig, use_container_width=True)
        download_button(by_year, f"{source.lower()}_completude_por_ano.csv")


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
    with st.expander("Dicionário operacional e campos prioritários", expanded=False):
        render_field_guide(source)

    sel = render_column_config(source, columns)
    exprs = build_expressions(source, sel)
    base_where, graph_where, definition = render_filters(source, table, exprs)
    render_kpis(table, source, base_where, graph_where, exprs)

    tabs = st.tabs(["Indicadores", "Temporal", "CID-10 / classificação", "Demografia e território", "Qualidade", "Prévia", "SQL Lab"])
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
        limit = st.slider("Número de linhas", 50, 5000, 200, step=50, key=f"preview_limit_{source}")
        try:
            df_prev = query_enriched_preview(table, sel, exprs, graph_where, limit)
            st.dataframe(df_prev, use_container_width=True)
            download_button(df_prev, f"{source.lower()}_previa_enriquecida.csv")
        except Exception as exc:
            st.error(f"Erro ao montar prévia: {exc}")
        st.markdown("### Schema")
        st.dataframe(schema, use_container_width=True, hide_index=True)
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
    freq_label = st.selectbox("Agregação", ["Ano", "Mês", "Semana"], index=0, key="comp_freq")
    freq = {"Ano": "year", "Mês": "month", "Semana": "week"}[freq_label]
    normalize = st.checkbox("Normalizar em índice 100 no primeiro período não-zero", value=False, key="comp_norm")
    stratify_cid = st.checkbox("Estratificar por tipo CID-10 quando disponível", value=False, key="comp_cid")

    frames = []
    for item in available:
        if item["source"] not in chosen:
            continue
        table: LoadedTable = item["table"]
        exprs = item["exprs"]
        cat = exprs.get("cid_type") if stratify_cid else None
        try:
            ts = query_timeseries(table, exprs["dt"], item["graph_where"], freq, cat)
        except Exception as exc:
            st.warning(f"Falha na série de {item['source']}: {exc}")
            continue
        if ts.empty:
            continue
        if cat:
            ts["serie"] = item["source"] + " — " + ts["categoria"].astype(str)
        else:
            ts["serie"] = item["source"] + " — " + item.get("definition", "")
        ts = ts.rename(columns={"n": "valor"})
        if normalize:
            ts = ts.sort_values("periodo")
            for s in ts["serie"].unique():
                idx = ts["serie"].eq(s)
                nonzero = ts.loc[idx & ts["valor"].gt(0), "valor"]
                if not nonzero.empty:
                    ts.loc[idx, "valor"] = ts.loc[idx, "valor"] / nonzero.iloc[0] * 100
        frames.append(ts[["periodo", "serie", "valor"]])
    if not frames:
        st.warning("Nenhuma série gerada.")
        return
    comp = pd.concat(frames, ignore_index=True)
    fig = px.line(comp, x="periodo", y="valor", color="serie", markers=True, title="Comparação de tendências", labels={"valor": "Índice" if normalize else "Registros"})
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
        5. Use **Qualidade** para checar se uma tendência pode ser artefato de preenchimento.
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
                ["CIHA — atendimento", "registro administrativo com data/diagnóstico"],
                ["CIHA — morte administrativa", "MORTE = 1"],
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
