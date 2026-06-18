# -*- coding: utf-8 -*-
"""
Painel epidemiológico para meningite — SINAN, SIM e CIHA

O app aceita upload de DuckDB, upload de Parquet ou bancos hospedados no github em Parquet, calcula indicadores descritivos e separa:
- CID-10 bruto do caso/óbito/atendimento;
- classificação epidemiológica específica do SINAN, especialmente CON_DIAGES;
- definições operacionais de série: notificações, confirmados, descartados, óbitos etc.

Executar:
    streamlit run app_meningite_epidemiologico.py

Dependências:
    pip install streamlit duckdb pandas numpy plotly fastparquet
"""

from __future__ import annotations

import hashlib
import html as html_lib
import json
import re
import tempfile
import threading
import textwrap
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    import fastparquet as fp
except Exception:  # fallback defensivo: o app segue com DuckDB nativo se fastparquet não estiver instalado
    fp = None


# =============================================================================
# Configuração geral
# =============================================================================

st.set_page_config(
    page_title="Meningite — SINAN, SIM e CIHA",
    page_icon="🧫",
    layout="wide",
)

APP_VERSION = "2026-06-18-v26-fastparquet-sinan-cid10-municipios-csv"

# =============================================================================
# Controles de desempenho e limites defensivos
# =============================================================================

DEFAULT_MAX_PARQUET_FILES_PER_LOAD = 20
DEFAULT_DISPLAY_ROW_LIMIT = 1000
DEFAULT_COPY_ROW_LIMIT = 300
DEFAULT_DOWNLOAD_ROW_LIMIT = 50000
DEFAULT_PREVIEW_ROW_LIMIT = 200
DEFAULT_MAX_PREVIEW_ROWS = 5000
DEFAULT_SQL_LAB_ROW_LIMIT = 5000
DEFAULT_FULL_EXPORT_ROW_LIMIT = 100000
UPLOAD_CHUNK_SIZE = 1024 * 1024
DEFAULT_DUCKDB_MEMORY_LIMIT = "2GB"
DEFAULT_DUCKDB_THREADS = 2
DEFAULT_QUERY_CACHE_MAX_ENTRIES = 128
DEFAULT_FASTPARQUET_ROW_LIMIT = 1500000
DUCKDB_TEMP_SUBDIR = "meningite_duckdb_tmp"
DEATH_RED = "#D62728"
LETHALITY_RED = DEATH_RED
LETHALITY_LABEL = "Letalidade — óbitos por meningite / confirmados"
PLOTLY_DEFAULT_BLUE = "#636EFA"
APP_COLOR_SEQUENCE = (
    "#F2C500",  # amarelo
    "#FF7F0E",  # laranja
    "#1F77B4",  # azul
    "#2CA02C",  # verde
    "#000000",  # preto
    "#9467BD",  # roxo
    "#8C564B",  # marrom
    "#17BECF",  # ciano
    "#7F7F7F",  # cinza
    "#BCBD22",  # oliva
)
DEATH_COLOR_TERMS = (
    "obit",
    "obito",
    "obitos",
    "morte",
    "mortes",
    "mortal",
    "mortalidade",
    "letal",
    "letalidade",
    "fatal",
    "fatalidade",
    "death",
    "deaths",
)
PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


# =============================================================================
# Aparência geral da aplicação e dos gráficos
# =============================================================================

def _norm_ui_text(value: object) -> str:
    """Normaliza texto apenas para testes internos de rótulos/cores."""
    text = str(value or "")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return text.lower()


def _text_mentions_death(value: object) -> bool:
    text = _norm_ui_text(value)
    return any(term in text for term in DEATH_COLOR_TERMS)


def render_app_css() -> None:
    """Aplica ajustes discretos de legibilidade sem alterar a navegação do app."""
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2.5rem;
            max-width: 1480px;
        }
        h1, h2, h3 {
            letter-spacing: -0.015em;
            line-height: 1.18;
        }
        div[data-testid="stCaptionContainer"] p {
            line-height: 1.45;
        }
        div[data-testid="stMetric"] {
            background: rgba(250, 250, 250, 0.72);
            border: 1px solid rgba(49, 51, 63, 0.10);
            border-radius: 0.75rem;
            padding: 0.65rem 0.8rem;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(49, 51, 63, 0.10);
            border-radius: 0.55rem;
        }
        section[data-testid="stSidebar"] .stRadio label,
        section[data-testid="stSidebar"] .stCheckbox label {
            line-height: 1.3;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _axis_text_values(fig: go.Figure) -> List[str]:
    values: List[str] = []
    for axis_name in ("xaxis", "yaxis"):
        try:
            axis = getattr(fig.layout, axis_name)
            values.append(str(axis.title.text or ""))
        except Exception:
            pass
    return values


def _layout_text_values(fig: go.Figure) -> List[str]:
    values: List[str] = []
    try:
        values.append(str(fig.layout.title.text or ""))
    except Exception:
        pass
    values.extend(_axis_text_values(fig))
    try:
        values.append(str(fig.layout.legend.title.text or ""))
    except Exception:
        pass
    return values


def _figure_axis_mentions_death(fig: go.Figure) -> bool:
    return any(_text_mentions_death(value) for value in _axis_text_values(fig))


def _figure_context_mentions_death(fig: go.Figure) -> bool:
    return any(_text_mentions_death(value) for value in _layout_text_values(fig))


def _sequence_values(values: object) -> List[object]:
    if values is None:
        return []
    if isinstance(values, (str, bytes)):
        return [values]
    try:
        return list(values)
    except Exception:
        return []


def _expand_marker_colors(existing_color: object, n: int) -> List[object]:
    if n <= 0:
        return []
    if isinstance(existing_color, (str, bytes)):
        return [existing_color] * n
    values = _sequence_values(existing_color)
    if values:
        if len(values) >= n:
            return values[:n]
        return values + [values[-1]] * (n - len(values))
    return [PLOTLY_DEFAULT_BLUE] * n


def _bar_category_values(trace: go.BaseTraceType) -> List[object]:
    x_vals = _sequence_values(getattr(trace, "x", None))
    y_vals = _sequence_values(getattr(trace, "y", None))
    orientation = str(getattr(trace, "orientation", "") or "")
    if orientation == "h" and y_vals:
        return y_vals
    if orientation != "h" and x_vals:
        return x_vals
    return y_vals or x_vals


def _apply_death_red_to_bar_points(trace: go.BaseTraceType) -> bool:
    if str(getattr(trace, "type", "")) != "bar":
        return False
    category_values = _bar_category_values(trace)
    if not category_values:
        return False
    mask = [_text_mentions_death(value) for value in category_values]
    if not any(mask):
        return False
    try:
        existing_color = trace.marker.color
    except Exception:
        existing_color = None
    base_colors = _expand_marker_colors(existing_color, len(mask))
    colors = [DEATH_RED if is_death else base_colors[idx] for idx, is_death in enumerate(mask)]
    try:
        trace.update(marker_color=colors)
        return True
    except Exception:
        return False


def _trace_mentions_death(trace: go.BaseTraceType) -> bool:
    values = [
        getattr(trace, "name", ""),
        getattr(trace, "legendgroup", ""),
        getattr(trace, "hovertemplate", ""),
    ]
    return any(_text_mentions_death(value) for value in values)



def _set_trace_color(trace: go.BaseTraceType, color: str) -> None:
    """Aplica cor coerente em linhas, marcadores e barras sem depender do default do Plotly."""
    for update_kwargs in (
        {"line_color": color},
        {"marker_color": color},
    ):
        try:
            trace.update(**update_kwargs)
        except Exception:
            pass


def _apply_distinct_trace_colors(fig: go.Figure) -> None:
    """Usa uma paleta de alto contraste quando há múltiplas séries no mesmo gráfico."""
    data = list(getattr(fig, "data", []) or [])
    if len(data) <= 1:
        return
    color_idx = 0
    for trace in data:
        if _trace_mentions_death(trace):
            _set_trace_color(trace, DEATH_RED)
            continue
        color = APP_COLOR_SEQUENCE[color_idx % len(APP_COLOR_SEQUENCE)]
        color_idx += 1
        _set_trace_color(trace, color)


def enforce_death_related_red(fig: go.Figure) -> None:
    """Garante vermelho para traços, barras ou gráficos identificados como óbito/morte/letalidade."""
    if fig is None:
        return
    data = list(getattr(fig, "data", []) or [])
    axis_mentions_death = _figure_axis_mentions_death(fig)
    context_mentions_death = _figure_context_mentions_death(fig)
    for trace in data:
        point_level_colored = _apply_death_red_to_bar_points(trace)
        is_death_trace = (
            axis_mentions_death
            or _trace_mentions_death(trace)
            or (context_mentions_death and len(data) == 1 and not point_level_colored)
        )
        if not is_death_trace:
            continue
        for update_kwargs in (
            {"line_color": DEATH_RED},
            {"marker_color": DEATH_RED},
        ):
            try:
                trace.update(**update_kwargs)
            except Exception:
                pass


def style_plotly_figure(fig: go.Figure) -> go.Figure:
    """Padroniza margem, legenda, fonte e cor de óbito/letalidade em todos os gráficos."""
    if fig is None:
        return fig
    _apply_distinct_trace_colors(fig)
    enforce_death_related_red(fig)
    trace_types = {str(getattr(trace, "type", "")) for trace in (getattr(fig, "data", []) or [])}
    is_line_like = bool(trace_types) and trace_types.issubset({"scatter", "scattergl"})
    fig.update_layout(
        template="plotly_white",
        margin={"l": 36, "r": 28, "t": 104, "b": 72},
        font={"size": 13},
        title={"x": 0.0, "xanchor": "left", "y": 0.98, "yanchor": "top", "pad": {"b": 18}},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.04, "xanchor": "left", "x": 0, "itemsizing": "constant"},
        hoverlabel={"align": "left"},
        hovermode="x unified" if is_line_like else "closest",
    )
    fig.update_xaxes(automargin=True, showgrid=True, zeroline=False)
    fig.update_yaxes(automargin=True, zeroline=False)
    return fig


def render_plotly_chart(fig: go.Figure) -> None:
    """Renderiza Plotly com configuração leve e consistente."""
    st.plotly_chart(style_plotly_figure(fig), width="stretch", config=PLOTLY_CONFIG)


def _session_int(key: str, default: int) -> int:
    """Lê um inteiro de session_state com fallback seguro."""
    try:
        value = int(st.session_state.get(key, default))
    except Exception:
        return default
    return max(0, value)


def perf_int(key: str, default: int) -> int:
    """Atalho para parâmetros de desempenho configuráveis na barra lateral."""
    return _session_int(key, default)


def render_performance_controls() -> None:
    """Expõe limites para evitar carregamento, renderização e exportação excessivos."""
    with st.expander("Desempenho e memória", expanded=False):
        st.number_input(
            "Máximo de Parquets por carregamento",
            min_value=1,
            max_value=60,
            value=perf_int("perf_max_parquet_files", DEFAULT_MAX_PARQUET_FILES_PER_LOAD),
            step=1,
            key="perf_max_parquet_files",
            help="Evita abrir muitos arquivos/anos de uma vez. Aumente gradualmente se o ambiente suportar.",
        )
        st.text_input(
            "Limite de memória do DuckDB",
            value=str(st.session_state.get("perf_duckdb_memory_limit", DEFAULT_DUCKDB_MEMORY_LIMIT)),
            key="perf_duckdb_memory_limit",
            help="Exemplos válidos: 1GB, 2GB, 4096MB ou 75%. O DuckDB fará spill para disco quando possível.",
        )
        st.number_input(
            "Threads do DuckDB",
            min_value=1,
            max_value=8,
            value=perf_int("perf_duckdb_threads", DEFAULT_DUCKDB_THREADS),
            step=1,
            key="perf_duckdb_threads",
            help="Menos threads reduzem picos de memória; mais threads podem acelerar consultas em máquinas com RAM suficiente.",
        )
        st.number_input(
            "Máximo de linhas renderizadas em tabelas",
            min_value=100,
            max_value=20000,
            value=perf_int("perf_display_row_limit", DEFAULT_DISPLAY_ROW_LIMIT),
            step=100,
            key="perf_display_row_limit",
            help="A tabela na tela é truncada para proteger o navegador.",
        )
        st.number_input(
            "Máximo de linhas no botão copiar",
            min_value=50,
            max_value=5000,
            value=perf_int("perf_copy_row_limit", DEFAULT_COPY_ROW_LIMIT),
            step=50,
            key="perf_copy_row_limit",
            help="O botão de cópia injeta HTML/TSV no navegador; mantenha baixo para tabelas grandes.",
        )
        st.number_input(
            "Máximo de linhas em downloads genéricos",
            min_value=1000,
            max_value=500000,
            value=perf_int("perf_download_row_limit", DEFAULT_DOWNLOAD_ROW_LIMIT),
            step=1000,
            key="perf_download_row_limit",
            help="Downloads de tabelas agregadas normalmente ficam muito abaixo deste limite.",
        )
        st.number_input(
            "Máximo de linhas por página na prévia",
            min_value=100,
            max_value=50000,
            value=perf_int("perf_max_preview_rows", DEFAULT_MAX_PREVIEW_ROWS),
            step=100,
            key="perf_max_preview_rows",
            help="A prévia é paginada. Evite enviar muitas linhas ao frontend.",
        )
        st.number_input(
            "Máximo de linhas no SQL Lab",
            min_value=100,
            max_value=100000,
            value=perf_int("perf_sql_lab_row_limit", DEFAULT_SQL_LAB_ROW_LIMIT),
            step=100,
            key="perf_sql_lab_row_limit",
            help="O SQL Lab sempre encapsula SELECT/WITH em LIMIT para evitar resultados gigantes.",
        )
        st.number_input(
            "Máximo de linhas na exportação completa",
            min_value=1000,
            max_value=1000000,
            value=perf_int("perf_full_export_row_limit", DEFAULT_FULL_EXPORT_ROW_LIMIT),
            step=1000,
            key="perf_full_export_row_limit",
            help="A exportação completa só é habilitada quando os filtros reduzem o total para este limite.",
        )
        st.checkbox(
            "Materializar bases Parquet em memória (mais rápido)",
            value=bool(st.session_state.get("perf_materialize_tables", True)),
            key="perf_materialize_tables",
            help=(
                "Lê cada Parquet uma única vez para uma tabela nativa do DuckDB; as consultas "
                "seguintes não reprocessam o Parquet. Desmarque para usar VIEW (lazy) em ambientes "
                "com pouca memória — ainda assim a conexão é reaproveitada entre consultas."
            ),
        )
        st.checkbox(
            "Usar fastparquet na materialização dos Parquets",
            value=bool(st.session_state.get("perf_use_fastparquet", True)),
            key="perf_use_fastparquet",
            help=(
                "Quando a base está materializada, o app tenta ler os arquivos com fastparquet, "
                "registrar o DataFrame no DuckDB e só então executar as consultas SQL. Se a leitura "
                "falhar, se fastparquet não estiver instalado ou se o volume exceder o limite abaixo, "
                "o app usa automaticamente o leitor nativo read_parquet do DuckDB."
            ),
        )
        st.number_input(
            "Limite estimado de linhas para materialização via fastparquet",
            min_value=1000,
            max_value=10000000,
            value=perf_int("perf_fastparquet_row_limit", DEFAULT_FASTPARQUET_ROW_LIMIT),
            step=10000,
            key="perf_fastparquet_row_limit",
            help="Acima deste limite, o app preserva memória e usa DuckDB read_parquet diretamente.",
        )
        st.caption(fastparquet_status())
        st.text_input(
            "CSV externo de municípios IBGE",
            value=str(st.session_state.get("municipios_ibge_csv_source", MUNICIPIOS_IBGE_CSV_URL or MUNICIPIOS_IBGE_CSV_FILENAME)),
            key="municipios_ibge_csv_source",
            help=(
                "Informe uma URL raw do GitHub ou mantenha municipios_ibge.csv no mesmo diretório do script. "
                "O arquivo substitui o dicionário de municípios que antes ficava embutido no Python."
            ),
        )
        st.caption(municipios_ibge_csv_status())
        if st.button("Limpar cache de consultas", key="clear_query_cache"):
            st.cache_data.clear()
            try:
                st.cache_resource.clear()
            except Exception:
                pass
            st.success("Cache limpo. As próximas consultas serão recalculadas.")


# =============================================================================
# Integração GitHub Release — Parquets empacotados com o painel
# =============================================================================

GITHUB_RELEASE_OWNER = "borbito123"
GITHUB_RELEASE_REPO = "Teste---Dados-Epidemiol-gicos-para-meningite-SINAN-CIHA-SIM---Rio-de-Janeiro"
GITHUB_RELEASE_TAG = "Release1"
GITHUB_HOSTED_PARQUETS_LABEL = "Bancos hospedados no github (Parquets)"
GITHUB_RELEASE_PAGE_URL = (
    f"https://github.com/{GITHUB_RELEASE_OWNER}/{GITHUB_RELEASE_REPO}/releases/tag/{GITHUB_RELEASE_TAG}"
)
GITHUB_RELEASE_EXPANDED_ASSETS_URL = (
    f"https://github.com/{GITHUB_RELEASE_OWNER}/{GITHUB_RELEASE_REPO}/releases/expanded_assets/{GITHUB_RELEASE_TAG}"
)
GITHUB_RELEASE_API_URL = (
    "https://api.github.com/repos/"
    f"{GITHUB_RELEASE_OWNER}/{urllib.parse.quote(GITHUB_RELEASE_REPO, safe='')}/releases/tags/{GITHUB_RELEASE_TAG}"
)
GITHUB_RELEASE_SOURCE_PREFIX = {
    "SINAN": "SINAN_MENINGITE_RIO_ESTADO_",
    "SIM": "SIM_DO_RIO_ESTADO_",
    "CIHA": "CIHA_RIO_ESTADO_",
}
GITHUB_RELEASE_FALLBACK_PARQUETS = (
    [f"CIHA_RIO_ESTADO_{year}.parquet" for year in range(2011, 2026)]
    + [f"SIM_DO_RIO_ESTADO_{year}.parquet" for year in range(2007, 2025)]
    + [f"SINAN_MENINGITE_RIO_ESTADO_{year}.parquet" for year in range(2007, 2026)]
)


CID_RULES = [
    {
        "grupo": "A17.0",
        "prefixo": "A170",
        "rotulo": "A17.0 — meningite tuberculosa",
        "padrao": "A17.0",
    },
    {
        "grupo": "A22.8",
        "prefixo": "A228",
        "rotulo": "A22.8 — meningite por carbúnculo",
        "padrao": "A22.8",
    },
    {
        "grupo": "A32.1",
        "prefixo": "A321",
        "rotulo": "A32.1 — meningite e meningoencefalite por listéria",
        "padrao": "A32.1",
    },
    {
        "grupo": "A39.0",
        "prefixo": "A390",
        "rotulo": "A39.0 — meningite meningocócica",
        "padrao": "A39.0",
    },
    {
        "grupo": "A83",
        "prefixo": "A83",
        "rotulo": "A83 — encefalite por vírus transmitidos por mosquitos",
        "padrao": "A83*",
    },
    {
        "grupo": "A84",
        "prefixo": "A84",
        "rotulo": "A84 — encefalite por vírus transmitido por carrapatos",
        "padrao": "A84*",
    },
    {
        "grupo": "A85",
        "prefixo": "A85",
        "rotulo": "A85 — outras encefalites virais, não classificadas em outra parte",
        "padrao": "A85*",
    },
    {
        "grupo": "A86",
        "prefixo": "A86",
        "rotulo": "A86 — encefalite viral não especificada",
        "padrao": "A86*",
    },
    {
        "grupo": "A87",
        "prefixo": "A87",
        "rotulo": "A87 — meningite viral",
        "padrao": "A87*",
    },
    {
        "grupo": "B00.3",
        "prefixo": "B003",
        "rotulo": "B00.3 — meningite devida ao vírus do herpes",
        "padrao": "B00.3",
    },
    {
        "grupo": "B00.4",
        "prefixo": "B004",
        "rotulo": "B00.4 — encefalite devida ao vírus do herpes",
        "padrao": "B00.4",
    },
    {
        "grupo": "B01.0",
        "prefixo": "B010",
        "rotulo": "B01.0 — meningite por varicela",
        "padrao": "B01.0",
    },
    {
        "grupo": "B01.1",
        "prefixo": "B011",
        "rotulo": "B01.1 — encefalite por varicela",
        "padrao": "B01.1",
    },
    {
        "grupo": "B02.0",
        "prefixo": "B020",
        "rotulo": "B02.0 — encefalite pelo vírus do herpes zoster",
        "padrao": "B02.0",
    },
    {
        "grupo": "B02.1",
        "prefixo": "B021",
        "rotulo": "B02.1 — meningite pelo vírus do herpes zoster",
        "padrao": "B02.1",
    },
    {
        "grupo": "B05.0",
        "prefixo": "B050",
        "rotulo": "B05.0 — sarampo complicado por encefalite",
        "padrao": "B05.0",
    },
    {
        "grupo": "B05.1",
        "prefixo": "B051",
        "rotulo": "B05.1 — sarampo complicado por meningite",
        "padrao": "B05.1",
    },
    {
        "grupo": "B06",
        "prefixo": "B06",
        "rotulo": "B06 — rubéola com complicações neurológicas",
        "padrao": "B06*",
    },
    {
        "grupo": "B26.1",
        "prefixo": "B261",
        "rotulo": "B26.1 — meningite por caxumba / parotidite epidêmica",
        "padrao": "B26.1",
    },
    {
        "grupo": "B26.2",
        "prefixo": "B262",
        "rotulo": "B26.2 — encefalite por caxumba / parotidite epidêmica",
        "padrao": "B26.2",
    },
    {
        "grupo": "B37.5",
        "prefixo": "B375",
        "rotulo": "B37.5 — meningite por Candida",
        "padrao": "B37.5",
    },
    {
        "grupo": "B38.4",
        "prefixo": "B384",
        "rotulo": "B38.4 — meningite por coccidioidomicose",
        "padrao": "B38.4",
    },
    {
        "grupo": "B45.1",
        "prefixo": "B451",
        "rotulo": "B45.1 — criptococose cerebral",
        "padrao": "B45.1",
    },
    {
        "grupo": "B57.4",
        "prefixo": "B574",
        "rotulo": "B57.4 — doença de Chagas crônica com comprometimento do sistema nervoso",
        "padrao": "B57.4",
    },
    {
        "grupo": "B58.2",
        "prefixo": "B582",
        "rotulo": "B58.2 — meningoencefalite por Toxoplasma",
        "padrao": "B58.2",
    },
    {
        "grupo": "B60.2",
        "prefixo": "B602",
        "rotulo": "B60.2 — naegleríase",
        "padrao": "B60.2",
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
    {
        "grupo": "G04",
        "prefixo": "G04",
        "rotulo": "G04 — encefalite, mielite e encefalomielite",
        "padrao": "G04*",
    },
    {
        "grupo": "G05",
        "prefixo": "G05",
        "rotulo": "G05 — encefalite, mielite e encefalomielite em doenças classificadas em outra parte",
        "padrao": "G05*",
    },
]

# Aceita CIDs com ponto, sem ponto, precedidos de * e dentro de campos compostos.
# G04 e G05 são tratados como prefixos; os códigos A/B abaixo ampliam o recorte para encefalite/meningoencefalite.
CID_MENINGITE_REGEX = (
    r"(A17[\.]?0|A22[\.]?8|A32[\.]?1|A39[\.]?0|A83[\.]?[0-9A-Z]?|"
    r"A84[\.]?[0-9A-Z]?|A85[\.]?[0-9A-Z]?|A86[\.]?[0-9A-Z]?|A87[\.]?[0-9A-Z]?|"
    r"B00[\.]?[34]|B01[\.]?[01]|B02[\.]?[01]|B05[\.]?[01]|B06[\.]?[0-9A-Z]?|"
    r"B26[\.]?[12]|B37[\.]?5|B38[\.]?4|B45[\.]?1|B57[\.]?4|B58[\.]?2|B60[\.]?2|"
    r"G00[\.]?[0-9A-Z]?|G01[\.]?[0-9A-Z]?|G02[\.]?[0-9A-Z]?|G03[\.]?[0-9A-Z]?|"
    r"G04[\.]?[0-9A-Z]?|G05[\.]?[0-9A-Z]?)"
)

CID_G01_PRESENT_REGEX = r"\*?G01[\.]?[0-9A-Z]?\*?"
CID_G02_PRESENT_REGEX = r"\*?G02[\.]?[0-9A-Z]?\*?"

CID10_ADEQUACY_TARGET_LABELS = {
    "G01": "G01 — meningite bacteriana em doença classificada em outra parte",
    "G02": "G02 — meningite em outras doenças infecciosas/parasitárias",
    "G05": "G05 — encefalite, mielite e encefalomielite em doenças classificadas em outra parte",
}

CID10_ADEQUACY_CONVERSION_RULES = [
    {
        "origem_grupo": "A22.8", "origem_prefixo": "A228", "origem_padrao": "A22.8", "match": "exact",
        "origem_rotulo": "A22.8 — meningite por carbúnculo",
        "destino_grupo": "G01", "destino_rotulo": CID10_ADEQUACY_TARGET_LABELS["G01"],
        "observacao": "A22.8 — meningite por carbúnculo convertida para G01.",
    },
    {
        "origem_grupo": "A32.1", "origem_prefixo": "A321", "origem_padrao": "A32.1", "match": "exact",
        "origem_rotulo": "A32.1 — meningite e meningoencefalite por listéria",
        "destino_grupo": "G05", "destino_rotulo": CID10_ADEQUACY_TARGET_LABELS["G05"],
        "observacao": "A32.1 — meningite/meningoencefalite por listéria convertida para G05.",
    },
    {
        "origem_grupo": "A83", "origem_prefixo": "A83", "origem_padrao": "A83*", "match": "prefix",
        "origem_rotulo": "A83 — encefalite por vírus transmitidos por mosquitos",
        "destino_grupo": "G05", "destino_rotulo": CID10_ADEQUACY_TARGET_LABELS["G05"],
        "observacao": "A83* — encefalite por vírus transmitidos por mosquitos convertida para G05.",
    },
    {
        "origem_grupo": "A84", "origem_prefixo": "A84", "origem_padrao": "A84*", "match": "prefix",
        "origem_rotulo": "A84 — encefalite por vírus transmitido por carrapatos",
        "destino_grupo": "G05", "destino_rotulo": CID10_ADEQUACY_TARGET_LABELS["G05"],
        "observacao": "A84* — encefalite por vírus transmitido por carrapatos convertida para G05.",
    },
    {
        "origem_grupo": "A85", "origem_prefixo": "A85", "origem_padrao": "A85*", "match": "prefix",
        "origem_rotulo": "A85 — outras encefalites virais, não classificadas em outra parte",
        "destino_grupo": "G05", "destino_rotulo": CID10_ADEQUACY_TARGET_LABELS["G05"],
        "observacao": "A85* — outras encefalites virais convertidas para G05.",
    },
    {
        "origem_grupo": "A86", "origem_prefixo": "A86", "origem_padrao": "A86*", "match": "prefix",
        "origem_rotulo": "A86 — encefalite viral não especificada",
        "destino_grupo": "G05", "destino_rotulo": CID10_ADEQUACY_TARGET_LABELS["G05"],
        "observacao": "A86* — encefalite viral não especificada convertida para G05.",
    },
    {
        "origem_grupo": "B00.3", "origem_prefixo": "B003", "origem_padrao": "B00.3", "match": "exact",
        "origem_rotulo": "B00.3 — meningite devida ao vírus do herpes",
        "destino_grupo": "G02", "destino_rotulo": CID10_ADEQUACY_TARGET_LABELS["G02"],
        "observacao": "B00.3 — meningite devida ao vírus do herpes convertida para G02.",
    },
    {
        "origem_grupo": "B00.4", "origem_prefixo": "B004", "origem_padrao": "B00.4", "match": "exact",
        "origem_rotulo": "B00.4 — encefalite devida ao vírus do herpes",
        "destino_grupo": "G02", "destino_rotulo": CID10_ADEQUACY_TARGET_LABELS["G02"],
        "observacao": "B00.4 — encefalite devida ao vírus do herpes convertida para G02.",
    },
    {
        "origem_grupo": "B01.0", "origem_prefixo": "B010", "origem_padrao": "B01.0", "match": "exact",
        "origem_rotulo": "B01.0 — meningite por varicela",
        "destino_grupo": "G02", "destino_rotulo": CID10_ADEQUACY_TARGET_LABELS["G02"],
        "observacao": "B01.0 — meningite por varicela convertida para G02.",
    },
    {
        "origem_grupo": "B01.1", "origem_prefixo": "B011", "origem_padrao": "B01.1", "match": "exact",
        "origem_rotulo": "B01.1 — encefalite por varicela",
        "destino_grupo": "G05", "destino_rotulo": CID10_ADEQUACY_TARGET_LABELS["G05"],
        "observacao": "B01.1 — encefalite por varicela convertida para G05.",
    },
    {
        "origem_grupo": "B02.0", "origem_prefixo": "B020", "origem_padrao": "B02.0", "match": "exact",
        "origem_rotulo": "B02.0 — encefalite pelo vírus do herpes zoster",
        "destino_grupo": "G05", "destino_rotulo": CID10_ADEQUACY_TARGET_LABELS["G05"],
        "observacao": "B02.0 — encefalite pelo vírus do herpes zoster convertida para G05.",
    },
    {
        "origem_grupo": "B02.1", "origem_prefixo": "B021", "origem_padrao": "B02.1", "match": "exact",
        "origem_rotulo": "B02.1 — meningite pelo vírus do herpes zoster",
        "destino_grupo": "G05", "destino_rotulo": CID10_ADEQUACY_TARGET_LABELS["G05"],
        "observacao": "B02.1 — meningite pelo vírus do herpes zoster convertida para G05.",
    },
    {
        "origem_grupo": "B05.0", "origem_prefixo": "B050", "origem_padrao": "B05.0", "match": "exact",
        "origem_rotulo": "B05.0 — sarampo complicado por encefalite",
        "destino_grupo": "G05", "destino_rotulo": CID10_ADEQUACY_TARGET_LABELS["G05"],
        "observacao": "B05.0 — sarampo complicado por encefalite convertido para G05.",
    },
    {
        "origem_grupo": "B05.1", "origem_prefixo": "B051", "origem_padrao": "B05.1", "match": "exact",
        "origem_rotulo": "B05.1 — sarampo complicado por meningite",
        "destino_grupo": "G02", "destino_rotulo": CID10_ADEQUACY_TARGET_LABELS["G02"],
        "observacao": "B05.1 — sarampo complicado por meningite convertido para G02.",
    },
    {
        "origem_grupo": "B06", "origem_prefixo": "B06", "origem_padrao": "B06*", "match": "prefix",
        "origem_rotulo": "B06 — rubéola com complicações neurológicas",
        "destino_grupo": "G05", "destino_rotulo": CID10_ADEQUACY_TARGET_LABELS["G05"],
        "observacao": "B06* — rubéola com complicações neurológicas convertida para G05.",
    },
    {
        "origem_grupo": "B26.1", "origem_prefixo": "B261", "origem_padrao": "B26.1", "match": "exact",
        "origem_rotulo": "B26.1 — meningite por caxumba / parotidite epidêmica",
        "destino_grupo": "G02", "destino_rotulo": CID10_ADEQUACY_TARGET_LABELS["G02"],
        "observacao": "B26.1 — meningite por caxumba convertida para G02.",
    },
    {
        "origem_grupo": "B26.2", "origem_prefixo": "B262", "origem_padrao": "B26.2", "match": "exact",
        "origem_rotulo": "B26.2 — encefalite por caxumba / parotidite epidêmica",
        "destino_grupo": "G05", "destino_rotulo": CID10_ADEQUACY_TARGET_LABELS["G05"],
        "observacao": "B26.2 — encefalite por caxumba convertida para G05.",
    },
    {
        "origem_grupo": "B37.5", "origem_prefixo": "B375", "origem_padrao": "B37.5", "match": "exact",
        "origem_rotulo": "B37.5 — meningite por Candida",
        "destino_grupo": "G02", "destino_rotulo": CID10_ADEQUACY_TARGET_LABELS["G02"],
        "observacao": "B37.5 — meningite por Candida convertida para G02.",
    },
    {
        "origem_grupo": "B38.4", "origem_prefixo": "B384", "origem_padrao": "B38.4", "match": "exact",
        "origem_rotulo": "B38.4 — meningite por coccidioidomicose",
        "destino_grupo": "G02", "destino_rotulo": CID10_ADEQUACY_TARGET_LABELS["G02"],
        "observacao": "B38.4 — meningite por coccidioidomicose convertida para G02.",
    },
    {
        "origem_grupo": "B45.1", "origem_prefixo": "B451", "origem_padrao": "B45.1", "match": "exact",
        "origem_rotulo": "B45.1 — criptococose cerebral",
        "destino_grupo": "G02", "destino_rotulo": CID10_ADEQUACY_TARGET_LABELS["G02"],
        "observacao": "B45.1 — criptococose cerebral convertida para G02.",
    },
    {
        "origem_grupo": "B58.2", "origem_prefixo": "B582", "origem_padrao": "B58.2", "match": "exact",
        "origem_rotulo": "B58.2 — meningoencefalite por Toxoplasma",
        "destino_grupo": "G05", "destino_rotulo": CID10_ADEQUACY_TARGET_LABELS["G05"],
        "observacao": "B58.2 — meningoencefalite por Toxoplasma convertida para G05.",
    },
    {
        "origem_grupo": "B57.4", "origem_prefixo": "B574", "origem_padrao": "B57.4", "match": "exact",
        "origem_rotulo": "B57.4 — doença de Chagas crônica com comprometimento do sistema nervoso",
        "destino_grupo": "G05", "destino_rotulo": CID10_ADEQUACY_TARGET_LABELS["G05"],
        "observacao": "B57.4 — doença de Chagas crônica com comprometimento do sistema nervoso convertida para G05.",
    },
    {
        "origem_grupo": "B60.2", "origem_prefixo": "B602", "origem_padrao": "B60.2", "match": "exact",
        "origem_rotulo": "B60.2 — naegleríase",
        "destino_grupo": "G05", "destino_rotulo": CID10_ADEQUACY_TARGET_LABELS["G05"],
        "observacao": "B60.2 — naegleríase convertida para G05.",
    },
]

CID10_ADEQUACY_MAPPING_ROWS = [
    {
        "CID-10 original": rule["origem_padrao"],
        "Descrição original": rule["origem_rotulo"],
        "CID-10 convertido": rule["destino_grupo"],
        "Categoria convertida": rule["destino_rotulo"],
        "Observação": rule["observacao"],
    }
    for rule in CID10_ADEQUACY_CONVERSION_RULES
]

CID10_ADEQUACY_OBSERVATION = (
    "Observação: A22.8 é convertido para G01; A32.1, A83*, A84*, A85*, A86*, "
    "B01.1, B02.0, B02.1, B05.0, B06*, B26.2, B57.4, B58.2 e B60.2 são convertidos para G05; "
    "B00.3, B00.4, B01.0, B05.1, B26.1, B37.5, B38.4 e B45.1 são convertidos para G02. "
    "Os demais CID-10 detectados ficam fora da conversão e permanecem no denominador para preservar o total de casos do recorte/ano."
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
        "CON_DIAGES": "05",
        "Grupo SINAN": "Meningite por outras bactérias",
        "CID-10 convertido": "G00 ou G01",
        "Observação": "Regra corrigida: não converter automaticamente para G04.2. Usa G01 quando o agente/doença é classificado em outra parte; caso contrário, usa G00, incluindo bactéria não especificada.",
    },
    {
        "CON_DIAGES": "09, 10",
        "Grupo SINAN": "Haemophilus influenzae; pneumocócica",
        "CID-10 convertido": "G00",
        "Observação": "Mantém Haemophilus influenzae e pneumocócica agregadas em G00.0/G00.1 para comparação por família CID.",
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


# Especificações auxiliares do SINAN para refinar CON_DIAGES.
# CLA_ME_BAC vem do Quadro II do dicionário SINAN NET Meningite.
SINAN_CLA_ME_BAC = {
    "09": "09 — Shigella sp",
    "10": "10 — Staphylococcus (aureus, sp, epidermidis)",
    "11": "11 — Salmonella sp",
    "12": "12 — Escherichia coli",
    "13": "13 — Klebsiella (sp, pneumoniae)",
    "14": "14 — Streptococcus (sp, pyogenes, agalactiae)",
    "15": "15 — Enterococcus",
    "16": "16 — Pseudomonas (aeruginosa, sp)",
    "18": "18 — Serratia (marcescens, sp)",
    "19": "19 — Alcaligenes (sp, faecalis)",
    "20": "20 — Proteus (sp, vulgaris, mirabilis)",
    "21": "21 — Listeria monocytogenes",
    "22": "22 — Enterobacter (sp, cloacae)",
    "23": "23 — Acinetobacter (sp, baumannii)",
    "26": "26 — Neisseria sp",
    "28": "28 — outras bactérias",
    "45": "45 — Treponema pallidum",
    "46": "46 — Rickettsiae",
    "49": "49 — Leptospira",
    "81": "81 — bactéria não especificada",
}

# Códigos de CLA_ME_BAC que, para a finalidade de comparação por família CID-10,
# são mais compatíveis com G01 por remeterem a doenças bacterianas classificadas em outra parte.
SINAN_CLA_ME_BAC_G01_CODES = {"11", "21", "45", "49"}

SINAN_CLA_ME_ASS = {
    "37": "37 — caxumba",
    "38": "38 — sarampo",
    "39": "39 — herpes simples",
    "40": "40 — varicela/catapora/herpes zoster",
    "41": "41 — rubéola",
    "55": "55 — influenza",
    "56": "56 — echovirus",
    "59": "59 — outros enterovírus",
    "63": "63 — coxsackie",
    "70": "70 — adenovírus",
    "71": "71 — vírus do Nilo Ocidental",
    "72": "72 — dengue",
    "73": "73 — outros arbovírus",
    "74": "74 — outros vírus",
    "75": "75 — não identificado",
}

SINAN_CLA_ME_ETI = {
    "42": "42 — outros fungos",
    "43": "43 — Cryptococcus/Torula",
    "44": "44 — Candida albicans/sp",
    "47": "47 — Trypanosoma cruzi",
    "48": "48 — Toxoplasma gondii/sp",
    "50": "50 — cisticerco",
    "52": "52 — outros parasitas",
    "64": "64 — Aspergillus",
    "76": "76 — Plasmodium sp",
    "77": "77 — Taenia solium",
}

SINAN_OTHER_BACTERIA_CID10_RULE_ROWS = [
    {
        "Cenário": "CON_DIAGES 05 + CLA_ME_BAC 11, 21, 45 ou 49; ou texto compatível com salmonela, listeriose, neurossífilis/sífilis ou leptospirose",
        "CID-10 convertido": "G01",
        "Justificativa": "Meningite em doença bacteriana classificada em outra parte.",
    },
    {
        "Cenário": "CON_DIAGES 05 + texto compatível com carbúnculo/antraz, Lyme/Borrelia, febre tifóide ou gonocócica",
        "CID-10 convertido": "G01",
        "Justificativa": "A doença bacteriana de base tem código próprio; a meningite entra como manifestação associada.",
    },
    {
        "Cenário": "CON_DIAGES 05 + Streptococcus, Staphylococcus, Escherichia coli, Klebsiella/Friedländer ou demais bactérias do Quadro II não remetidas a G01",
        "CID-10 convertido": "G00",
        "Justificativa": "Meningite bacteriana não classificada em outra parte; usar subcategoria específica quando disponível.",
    },
    {
        "Cenário": "CON_DIAGES 05 sem bactéria especificada ou CLA_ME_BAC 81",
        "CID-10 convertido": "G00",
        "Justificativa": "Equivale operacionalmente a meningite bacteriana não especificada/pyogênica/purulenta/supurativa SOE.",
    },
]


SINAN_G01_BASE_DISEASE_REFERENCE_ROWS = [
    {
        "Critério no SINAN": "CLA_ME_BAC 11 ou texto com Salmonella/salmonela",
        "Doença de base provável": "Infecção por Salmonella sp / salmonelose invasiva",
        "CID-10 da doença de base": "A02.2†",
        "CID-10 da manifestação": "G01*",
    },
    {
        "Critério no SINAN": "CLA_ME_BAC 21 ou texto com Listeria/listeriose",
        "Doença de base provável": "Listeriose / Listeria monocytogenes",
        "CID-10 da doença de base": "A32.1†",
        "CID-10 da manifestação": "G01*",
    },
    {
        "Critério no SINAN": "CLA_ME_BAC 45 ou texto com Treponema, sífilis ou neurossífilis",
        "Doença de base provável": "Sífilis / neurossífilis",
        "CID-10 da doença de base": "A52.1†; avaliar A50.4†/A51.4† conforme contexto",
        "CID-10 da manifestação": "G01*",
    },
    {
        "Critério no SINAN": "CLA_ME_BAC 49 ou texto com Leptospira/leptospirose",
        "Doença de base provável": "Leptospirose",
        "CID-10 da doença de base": "A27.-†",
        "CID-10 da manifestação": "G01*",
    },
    {
        "Critério no SINAN": "texto com carbúnculo/antraz",
        "Doença de base provável": "Carbúnculo / antraz",
        "CID-10 da doença de base": "A22.8†",
        "CID-10 da manifestação": "G01*",
    },
    {
        "Critério no SINAN": "texto com Lyme/Borrelia",
        "Doença de base provável": "Doença de Lyme / borreliose",
        "CID-10 da doença de base": "A69.2†",
        "CID-10 da manifestação": "G01*",
    },
    {
        "Critério no SINAN": "texto com febre tifóide/typhoid",
        "Doença de base provável": "Febre tifóide",
        "CID-10 da doença de base": "A01.0†",
        "CID-10 da manifestação": "G01*",
    },
    {
        "Critério no SINAN": "texto com gonococo/gonocócica",
        "Doença de base provável": "Infecção gonocócica",
        "CID-10 da doença de base": "A54.8†",
        "CID-10 da manifestação": "G01*",
    },
]

# Regex operacional para pistas textuais. Usado somente em campos auxiliares detectados automaticamente.
SINAN_G01_DETAIL_REGEX = (
    r"CARB[UÚ]NCULO|ANTRAZ|ANTHRAX|LYME|BORREL|TIF[OÓ]IDE|TYPHOID|"
    r"GONOCOC|GONOCOCO|SALMONEL|LEPTOSPI|LISTERI|NEUROSS[ÍI]FIL|NEUROSYPH|"
    r"S[ÍI]FIL|SYPHIL|TREPONEMA"
)

# Campos que o app tenta selecionar automaticamente como auxiliares para refino etiológico/textual.
SINAN_AUXILIARY_CID10_CANDIDATES = [
    "CLA_ME_BAC", "CLA_ME_ASS", "CLA_ME_ETI", "DS_OBSERVACAO", "OBSERVACAO", "OBSERVACOES",
    "OUTROS_SINTOMAS", "OUTRO_SINTOMA", "OUTR_SINT", "SIN_OUT", "SINTOMAS", "SINAIS",
    "DIAGNOSTICO", "DIAG_FINAL", "CLASSIFICACAO", "EVOLUCAO", "CID", "ID_AGRAVO",
]

SINAN_QUIMIO_INTERPRETATION_ROWS = [
    {
        "Parâmetro": "Hemácias",
        "Bacteriana": "Geralmente ausentes/baixas; elevação sugere punção traumática, hemorragia ou quadro necro-hemorrágico associado.",
        "Viral/asséptica": "Geralmente ausentes/baixas; herpes e alguns quadros encefalíticos podem cursar com hemácias.",
        "Fúngica/TB": "Sem padrão específico; interpretar com aspecto do LCR e demais marcadores.",
        "Helmintos/parasitária eosinofílica": "Sem padrão específico; pode haver pleocitose eosinofílica sem hemácias importantes.",
        "Protozoários": "Naegleria/PAM pode simular bacteriana e pode ter LCR turvo/hemorrágico.",
    },
    {
        "Parâmetro": "Neutrófilos",
        "Bacteriana": "Predomínio neutrofílico, frequentemente muito aumentado.",
        "Viral/asséptica": "Predomínio linfocitário; pode haver neutrófilos nas primeiras 24-48h.",
        "Fúngica/TB": "Frequentemente linfocítica ou mista; TB pode ser mista no início.",
        "Helmintos/parasitária eosinofílica": "Pode haver mistura celular; eosinófilos são a pista principal.",
        "Protozoários": "PAM/Naegleria costuma ter pleocitose neutrofílica intensa, semelhante à bacteriana.",
    },
    {
        "Parâmetro": "Glicose",
        "Bacteriana": "Baixa ou muito baixa, muitas vezes <50% da glicemia sérica.",
        "Viral/asséptica": "Usualmente normal.",
        "Fúngica/TB": "Frequentemente baixa, sobretudo TB/criptococose.",
        "Helmintos/parasitária eosinofílica": "Usualmente normal, mas pode variar em doença grave.",
        "Protozoários": "Pode ser baixa, especialmente na PAM/Naegleria.",
    },
    {
        "Parâmetro": "Leucócitos",
        "Bacteriana": "Geralmente muito elevados; valores >1.000-2.000 células/µL favorecem bacteriana.",
        "Viral/asséptica": "Elevação menor/moderada, tipicamente <300-1.000 células/µL conforme referência.",
        "Fúngica/TB": "Elevação variável, geralmente menor que bacteriana aguda, mas pode ser importante.",
        "Helmintos/parasitária eosinofílica": "Pleocitose com fração eosinofílica relevante.",
        "Protozoários": "Pode ser muito alta na PAM/Naegleria.",
    },
    {
        "Parâmetro": "Eosinófilos",
        "Bacteriana": "Não costuma predominar.",
        "Viral/asséptica": "Não costuma predominar.",
        "Fúngica/TB": "Pode ocorrer em algumas micoses, mas não é o padrão principal.",
        "Helmintos/parasitária eosinofílica": "Marcador-chave: eosinofilia no LCR sugere meningite eosinofílica, frequentemente helmíntica.",
        "Protozoários": "Geralmente não é o marcador principal.",
    },
    {
        "Parâmetro": "Proteínas",
        "Bacteriana": "Elevadas; valores >220 mg/dL favorecem fortemente bacteriana em alguns critérios.",
        "Viral/asséptica": "Normais a moderadamente elevadas.",
        "Fúngica/TB": "Elevadas, muitas vezes de forma persistente/subaguda.",
        "Helmintos/parasitária eosinofílica": "Normais a elevadas; podem aumentar com inflamação intensa.",
        "Protozoários": "Frequentemente elevadas em PAM/Naegleria.",
    },
    {
        "Parâmetro": "Monócitos",
        "Bacteriana": "Menor participação no padrão típico agudo; pode aparecer após tratamento/evolução.",
        "Viral/asséptica": "Pode compor o predomínio mononuclear.",
        "Fúngica/TB": "Comum em padrões crônicos/subagudos mononucleares.",
        "Helmintos/parasitária eosinofílica": "Pode compor a pleocitose junto a linfócitos/eosinófilos.",
        "Protozoários": "Variável; não é a pista principal na PAM.",
    },
    {
        "Parâmetro": "Linfócitos",
        "Bacteriana": "Não é o padrão típico inicial, mas pode predominar em fases iniciais/atípicas ou após antibiótico.",
        "Viral/asséptica": "Predomínio linfocitário é o padrão clássico.",
        "Fúngica/TB": "Predomínio linfocitário ou misto é frequente.",
        "Helmintos/parasitária eosinofílica": "Pode estar elevado junto a eosinófilos.",
        "Protozoários": "PAM tende mais a neutrófilos; outros protozoários podem variar.",
    },
    {
        "Parâmetro": "Cloreto",
        "Bacteriana": "Pode estar reduzido, mas tem baixa especificidade isolada.",
        "Viral/asséptica": "Geralmente preservado.",
        "Fúngica/TB": "Redução de cloreto é classicamente descrita em TB, mas deve ser interpretada com cautela.",
        "Helmintos/parasitária eosinofílica": "Sem padrão útil isolado.",
        "Protozoários": "Sem padrão útil isolado.",
    },
]


SINAN_QUIMIO_REFERENCES = [
    {
        "Referência": "Bennett JE, Dolin R, Blaser MJ, eds. Mandell, Douglas, and Bennett's Principles and Practice of Infectious Diseases. Elsevier.",
        "Uso no painel": "Base clínica geral para padrões de LCR em meningites bacterianas, virais, fúngicas, tuberculosas e parasitárias.",
    },
    {
        "Referência": "Tunkel AR et al. Practice Guidelines for the Management of Bacterial Meningitis. Clinical Infectious Diseases. 2004;39:1267-1284.",
        "Uso no painel": "Interpretação de meningite bacteriana e limitações dos marcadores de LCR isolados.",
    },
    {
        "Referência": "MSD/Merck Manual Professional. Cerebrospinal Fluid Findings in Meningitis; Cerebrospinal Fluid Abnormalities in Various Disorders.",
        "Uso no painel": "Resumo prático de predominância celular, proteína e glicose por síndrome/etiologia.",
    },
    {
        "Referência": "Tunkel AR et al. 2017 IDSA Clinical Practice Guidelines for Healthcare-Associated Ventriculitis and Meningitis.",
        "Uso no painel": "Ressalva de que normalidade de celularidade, glicose ou proteína não exclui infecção em contextos associados à assistência/neurocirurgia.",
    },
    {
        "Referência": "WHO. Guidelines on meningitis diagnosis, treatment and care. 2025.",
        "Uso no painel": "Reforço do papel de glicose, proteína, contagem total e diferencial de leucócitos, hemácias e Gram como investigação inicial do LCR.",
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


SINAN_SYMPTOM_FIELDS = [
    ("CLI_CEFALE", "Cefaleia"),
    ("CLI_FEBRE", "Febre"),
    ("CLI_VOMITO", "Vômitos"),
    ("CLI_CONVUL", "Convulsões"),
    ("CLI_RIGIDE", "Rigidez de nuca"),
    ("CLI_KERNIG", "Kernig/Brudzinski"),
    ("CLI_ABAULA", "Abaulamento de fontanela"),
    ("CLI_PETEQU", "Petequias/sufusões hemorrágicas"),
    ("CLI_COMA", "Coma"),
    ("CLI_OUTRAS", "Outras manifestações"),
]

SINAN_VACCINE_FIELDS = [
    ("ANT_AC", "Polissacarídica A/C"),
    ("ANT_BC", "Polissacarídica B/C"),
    ("ANT_CONJ_C", "Conjugada meningo C"),
    ("ANT_BCG", "BCG"),
    ("ANT_TRIPLI", "Tríplice viral"),
    ("ANT_HEMO_T", "Hemófilo/Tetravalente ou Hib"),
    ("ANT_PNEUMO", "Pneumococo"),
    ("ANT_OUTRA", "Outra vacina"),
]

# Campos do exame quimiocitológico do líquor no SINAN.
# Os nomes abaixo seguem o dicionário SINAN NET para meningite; os seletores do app
# também aceitam variações próximas caso o banco venha renomeado.
SINAN_QUIMIO_MATERIAL = "Líquor (LCR)"
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

SINAN_ESCOLARIDADE = {
    "0": "0 — analfabeto",
    "1": "1 — 1ª a 4ª série incompleta do EF",
    "2": "2 — 4ª série completa do EF",
    "3": "3 — 5ª à 8ª série incompleta do EF",
    "4": "4 — ensino fundamental completo",
    "5": "5 — ensino médio incompleto",
    "6": "6 — ensino médio completo",
    "7": "7 — educação superior incompleta",
    "8": "8 — educação superior completa",
    "9": "9 — ignorado",
    "10": "10 — não se aplica",
    "NA": "10 — não se aplica",
}

SIM_ESCOLARIDADE_2010 = {
    "0": "0 — sem escolaridade",
    "1": "1 — ensino fundamental I",
    "2": "2 — ensino fundamental II",
    "3": "3 — ensino médio",
    "4": "4 — superior incompleto",
    "5": "5 — superior completo",
    "9": "9 — ignorado",
}

SIM_ESCOLARIDADE_ANTIGA = {
    "0": "0 — sem escolaridade",
    "1": "1 — nenhuma",
    "2": "2 — 1 a 3 anos de estudo",
    "3": "3 — 4 a 7 anos de estudo",
    "4": "4 — 8 a 11 anos de estudo",
    "5": "5 — 12 anos ou mais de estudo",
    "9": "9 — ignorado",
}

SIM_ESCOLARIDADE_AGREGADA = {
    "0": "0 — sem escolaridade",
    "1": "1 — ensino fundamental I",
    "2": "2 — ensino fundamental II",
    "3": "3 — ensino médio",
    "4": "4 — superior",
    "5": "5 — superior",
    "9": "9 — ignorado",
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


# Dicionário externo de municípios brasileiros (IBGE)
# -----------------------------------------------------------------------------
# O mapeamento nacional de municípios não fica mais embutido no script. Coloque
# `municipios_ibge.csv` no mesmo diretório deste arquivo ou informe uma URL raw
# do GitHub no controle "CSV externo de municípios IBGE".
MUNICIPIOS_IBGE_CSV_FILENAME = "municipios_ibge.csv"
MUNICIPIOS_IBGE_CSV_URL = ""
MUNICIPIOS_IBGE_VIEW_NAME = "municipios_ibge"


def _is_url(value: object) -> bool:
    text = str(value or "").strip().lower()
    return text.startswith("http://") or text.startswith("https://")


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def _download_municipios_ibge_csv_cached(url: str) -> str:
    """Baixa o CSV de municípios para cache local, útil para URL raw do GitHub."""
    digest = hashlib.sha1(str(url).encode("utf-8")).hexdigest()[:16]
    out_dir = Path(tempfile.gettempdir()) / "meningite_municipios_ibge"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"municipios_ibge_{digest}.csv"
    if out.exists() and out.stat().st_size > 0:
        return str(out)
    req = urllib.request.Request(
        str(url),
        headers={"User-Agent": "meningite-streamlit-dashboard"},
    )
    tmp = out.with_suffix(".csv.download")
    try:
        with urllib.request.urlopen(req, timeout=45) as resp, tmp.open("wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        if tmp.stat().st_size == 0:
            raise ValueError("CSV de municípios vazio.")
        tmp.replace(out)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        raise
    return str(out)


def _local_municipios_ibge_csv_candidates(configured: str = "") -> List[Path]:
    candidates: List[Path] = []
    if configured and not _is_url(configured):
        candidates.append(Path(configured).expanduser())
    try:
        candidates.append(Path(__file__).resolve().with_name(MUNICIPIOS_IBGE_CSV_FILENAME))
    except Exception:
        pass
    candidates.append(Path.cwd() / MUNICIPIOS_IBGE_CSV_FILENAME)
    candidates.append(Path(tempfile.gettempdir()) / MUNICIPIOS_IBGE_CSV_FILENAME)
    # Remove duplicatas preservando ordem.
    out: List[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            out.append(candidate)
    return out


def _configured_municipios_ibge_csv_source() -> str:
    default_source = MUNICIPIOS_IBGE_CSV_URL or MUNICIPIOS_IBGE_CSV_FILENAME
    try:
        return str(st.session_state.get("municipios_ibge_csv_source", default_source) or "").strip()
    except Exception:
        return default_source


def _resolve_municipios_ibge_csv_source() -> Optional[str]:
    configured = _configured_municipios_ibge_csv_source()
    if configured and _is_url(configured):
        try:
            return _download_municipios_ibge_csv_cached(configured)
        except Exception:
            return None
    for candidate in _local_municipios_ibge_csv_candidates(configured):
        try:
            if candidate.exists() and candidate.is_file() and candidate.stat().st_size > 0:
                return str(candidate)
        except Exception:
            continue
    return None


def municipios_ibge_csv_status() -> str:
    source = _resolve_municipios_ibge_csv_source()
    if source:
        return f"CSV de municípios IBGE ativo: {source}"
    return (
        "CSV de municípios IBGE não localizado. Coloque `municipios_ibge.csv` junto ao script "
        "ou informe uma URL raw do GitHub; enquanto isso, o app preserva o código municipal informado."
    )


def _empty_municipios_ibge_view_sql() -> str:
    return f"""
    CREATE OR REPLACE TEMP VIEW {qident(MUNICIPIOS_IBGE_VIEW_NAME)} AS
    SELECT
        CAST(NULL AS VARCHAR) AS codigo_ibge_6,
        CAST(NULL AS VARCHAR) AS municipio,
        CAST(NULL AS VARCHAR) AS uf,
        CAST(NULL AS VARCHAR) AS codigo_ibge_7,
        CAST(NULL AS VARCHAR) AS label
    WHERE FALSE
    """


def _ensure_municipios_ibge_view(shared: "_SharedDB") -> None:
    """Registra o CSV externo de municípios como VIEW temporária no DuckDB."""
    source = _resolve_municipios_ibge_csv_source()
    token = ("municipios_ibge", _file_fingerprint(source) if source else None)
    if shared.registry.get("__municipios_ibge_view__") == str(token):
        return
    with shared.lock:
        if shared.registry.get("__municipios_ibge_view__") == str(token):
            return
        con = shared.con
        if not source:
            try:
                con.execute(_empty_municipios_ibge_view_sql())
            except Exception:
                pass
            shared.registry["__municipios_ibge_view__"] = str(token)
            return
        try:
            con.execute(
                f"""
                CREATE OR REPLACE TEMP VIEW {qident(MUNICIPIOS_IBGE_VIEW_NAME)} AS
                WITH raw AS (
                    SELECT *
                    FROM read_csv_auto({qstr(source)}, header=true, all_varchar=true)
                )
                SELECT
                    RIGHT('000000' || regexp_replace(COALESCE(codigo_ibge_6, ''), '[^0-9]', '', 'g'), 6) AS codigo_ibge_6,
                    NULLIF(TRIM(CAST(municipio AS VARCHAR)), '') AS municipio,
                    NULLIF(TRIM(CAST(uf AS VARCHAR)), '') AS uf,
                    NULLIF(TRIM(CAST(codigo_ibge_7 AS VARCHAR)), '') AS codigo_ibge_7,
                    COALESCE(
                        NULLIF(TRIM(CAST(label AS VARCHAR)), ''),
                        NULLIF(TRIM(CAST(municipio AS VARCHAR)), '') || '/' || NULLIF(TRIM(CAST(uf AS VARCHAR)), '') || ' — IBGE ' || NULLIF(TRIM(CAST(codigo_ibge_7 AS VARCHAR)), '')
                    ) AS label
                FROM raw
                WHERE regexp_replace(COALESCE(codigo_ibge_6, ''), '[^0-9]', '', 'g') <> ''
                """
            )
        except Exception:
            # Não interrompe o painel se o CSV estiver ausente, inválido ou temporariamente indisponível.
            try:
                con.execute(_empty_municipios_ibge_view_sql())
            except Exception:
                pass
        shared.registry["__municipios_ibge_view__"] = str(token)



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
            "G04.2 não é inferido por CON_DIAGES=05; a classificação de 'outras bactérias' é refinada como G00 ou G01 conforme CLA_ME_BAC e campos complementares.",
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
        ("NU_NOTIFIC", "identificador operacional", "número da notificação; usado para verificar sobreposição e possíveis duplicidades"),
        ("CLASSI_FIN", "definição de caso", "confirmado, descartado, inconclusivo"),
        ("CON_DIAGES", "etiologia/forma", "conclusão diagnóstica específica"),
        ("CLA_ME_BAC", "bactéria em outras bacterianas", "refina CON_DIAGES=05 em G00 ou G01"),
        ("CLA_ME_ASS", "agente viral/asséptico", "detalha meningite asséptica/viral"),
        ("CLA_ME_ETI", "outra etiologia", "detalha fungos, protozoários e parasitas"),
        ("EVOLUCAO", "desfecho", "alta, óbito por meningite, óbito por outra causa"),
        ("CRITERIO", "critério de confirmação", "cultura, PCR, clínico, quimiocitológico etc."),
        ("LAB_PUNCAO", "investigação", "punção laboratorial/lombar realizada"),
        ("LAB_LIQUOR", "exame", "quimiocitológico do líquor (LCR) realizado"),
        ("LAB_HEMA / LAB_NEUTRO / LAB_GLICO / LAB_LEUCO / LAB_EOSI / LAB_PROT / LAB_MONO / LAB_LINFO / LAB_CLOR", "parâmetros do LCR", "hemácias, diferenciais celulares, glicose, proteínas e cloreto"),
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
        ("PROC_REA / PROCEDIMENTO", "procedimento", "procedimento informado e sua quantidade"),
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
    """Normaliza campos de códigos categóricos do DATASUS.

    Alguns arquivos DuckDB/Parquet chegam com códigos como 1.0, 07.0 ou 5,0
    depois da conversão de tipos. Se apenas removêssemos pontuação, 1.0 viraria
    10 e 07.0 viraria 070, quebrando CLASSI_FIN, CON_DIAGES, EVOLUCAO etc.
    """
    txt = f"UPPER(COALESCE({clean_str_expr(col)}, ''))"
    numeric_like = f"regexp_matches({txt}, '^\\s*[0-9]+([\\.,]0+)?\\s*$')"
    numeric_code = f"regexp_replace(TRIM({txt}), '[\\.,]0+$', '')"
    alnum_code = f"regexp_replace({txt}, '[^0-9A-Z]', '', 'g')"
    code = f"NULLIF(CASE WHEN {numeric_like} THEN {numeric_code} ELSE {alnum_code} END, '')"
    if pad2:
        return f"CASE WHEN {code} IS NULL THEN NULL WHEN LENGTH({code}) = 1 THEN '0' || {code} ELSE {code} END"
    return code



def sqlsafe(expr: object) -> str:
    if expr is None:
        return "NULL"
    return str(expr)


def case_from_mapping(code_sql: str, mapping: Dict[str, str], default: str) -> str:
    parts = [f"WHEN {qstr(k)} THEN {qstr(v)}" for k, v in mapping.items()]
    return f"CASE {code_sql} {' '.join(parts)} ELSE {qstr(default)} END"


def education_label_expr(source: str, col: str) -> str:
    code = clean_code_expr(col)
    if source == "SINAN":
        return case_from_mapping(code, SINAN_ESCOLARIDADE, "Sem informação/ignorado")
    if source == "SIM":
        col_norm = normalize_name(col)
        if "2010" in col_norm:
            mapping = SIM_ESCOLARIDADE_2010
        elif "AGR" in col_norm:
            mapping = SIM_ESCOLARIDADE_AGREGADA
        else:
            mapping = SIM_ESCOLARIDADE_ANTIGA
        return case_from_mapping(code, mapping, "Sem informação/ignorado")
    return clean_str_expr(col)


def unique_mapping_labels(mapping: Dict[str, str]) -> List[str]:
    """Retorna rótulos categóricos preservando a ordem e removendo duplicatas."""
    labels: List[str] = []
    seen: set[str] = set()
    for label in mapping.values():
        if label not in seen:
            labels.append(label)
            seen.add(label)
    return labels


def education_category_labels(source: str, col_or_expr: Optional[str] = None, include_missing: bool = True) -> List[str]:
    """Lista todas as categorias operacionais de escolaridade esperadas para o campo detectado."""
    if source == "SINAN":
        labels = unique_mapping_labels(SINAN_ESCOLARIDADE)
    elif source == "SIM":
        col_norm = normalize_name(col_or_expr or "")
        if "2010" in col_norm:
            labels = unique_mapping_labels(SIM_ESCOLARIDADE_2010)
        elif "AGR" in col_norm:
            labels = unique_mapping_labels(SIM_ESCOLARIDADE_AGREGADA)
        else:
            labels = unique_mapping_labels(SIM_ESCOLARIDADE_ANTIGA)
    else:
        labels = []
    if include_missing and "Sem informação/ignorado" not in labels:
        labels.append("Sem informação/ignorado")
    return labels


def values_cte_from_labels(labels: Sequence[str], label_col: str, order_col: str) -> str:
    """Constrói uma lista VALUES segura para CTEs de categorias categóricas."""
    if not labels:
        return f"SELECT {qstr('Sem informação/ignorado')} AS {label_col}, 1 AS {order_col}"
    values = ", ".join(f"({qstr(label)}, {idx})" for idx, label in enumerate(labels, start=1))
    return f"SELECT * FROM (VALUES {values}) AS t({label_col}, {order_col})"


def category_label_expr(category_sql: str, default: str = "Sem informação") -> str:
    """Normaliza categorias textuais para evitar rótulos vazios/Undefined nos gráficos."""
    cleaned = f"NULLIF(TRIM(CAST(({category_sql}) AS VARCHAR)), '')"
    return f"""
    CASE
        WHEN {cleaned} IS NULL THEN {qstr(default)}
        WHEN UPPER({cleaned}) IN ('UNDEFINED', 'NONE', 'NULL', 'NAN') THEN {qstr(default)}
        ELSE {cleaned}
    END
    """


def municipality_display_expr(col: str) -> str:
    raw = clean_str_expr(col)
    digits = f"regexp_replace(COALESCE({raw}, ''), '[^0-9]', '', 'g')"
    code6 = f"CASE WHEN LENGTH({digits}) >= 6 THEN SUBSTR({digits}, 1, 6) ELSE {digits} END"
    if _resolve_municipios_ibge_csv_source():
        lookup = (
            f"(SELECT label FROM {qident(MUNICIPIOS_IBGE_VIEW_NAME)} "
            f"WHERE codigo_ibge_6 = {code6} LIMIT 1)"
        )
        return f"""
        CASE
            WHEN {raw} IS NULL THEN 'Sem informação'
            WHEN {digits} <> '' THEN COALESCE({lookup}, 'Código informado: ' || {raw})
            ELSE {raw}
        END
        """


# ================================
# OTIMIZAÇÃO: MUNICÍPIOS (ANTI-CRASH)
# ================================

def _municipality_code_expr(col: str) -> str:
    raw = clean_str_expr(col)
    digits = f"regexp_replace(COALESCE({raw}, ''), '[^0-9]', '', 'g')"
    return f"""
    CASE
        WHEN {raw} IS NULL THEN NULL
        WHEN LENGTH({digits}) >= 6 THEN SUBSTR({digits}, 1, 6)
        ELSE NULL
    END
    """


def query_municipality_top(table, col, where_sql, top_n=15):
    code_expr = _municipality_code_expr(col)

    sql = f"""
        WITH base AS (
            SELECT {code_expr} AS codigo_ibge_6
            FROM {table.ref_sql}
            {where_sql}
        ),
        agg AS (
            SELECT
                COALESCE(m.label, 'Código não mapeado') AS categoria,
                COUNT(*) AS n
            FROM base b
            LEFT JOIN {MUNICIPIOS_IBGE_VIEW_NAME} m
                ON b.codigo_ibge_6 = m.codigo_ibge_6
            GROUP BY 1
        ),
        ranked AS (
            SELECT *,
                   ROW_NUMBER() OVER (ORDER BY n DESC) AS rn,
                   SUM(n) OVER () AS total
            FROM agg
        ),
        final AS (
            SELECT
                CASE WHEN rn <= {int(top_n)} THEN categoria ELSE 'Outros municípios' END AS categoria,
                SUM(n) AS n,
                MAX(total) AS denominador,
                MIN(rn) AS ordem
            FROM ranked
            GROUP BY 1
        )
        SELECT categoria, n, denominador,
               ROUND(100.0 * n / denominador, 2) AS pct
        FROM final
        ORDER BY ordem, n DESC
    """
    return run_query(table, sql)

    return f"""
    CASE
        WHEN {raw} IS NULL THEN 'Sem informação'
        WHEN {digits} <> '' THEN 'Código informado: ' || {raw}
        ELSE {raw}
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


def numeric_expr(col: str) -> str:
    """Expressão SQL segura para converter campos numéricos DATASUS em DOUBLE.

    Alguns campos laboratoriais chegam como inteiros, outros como texto, e alguns
    podem vir com vírgula decimal. TRY_CAST evita que valores sujos interrompam
    a execução do painel.
    """
    txt = clean_str_expr(col)
    cleaned = f"regexp_replace({txt}, '[^0-9,\\.\\-+]', '', 'g')"
    return f"""
    CASE
        WHEN {txt} IS NULL THEN NULL
        WHEN regexp_matches({txt}, '^\\s*[-+]?\\d{{1,3}}(\\.\\d{{3}})+(,\\d+)?\\s*$')
            THEN TRY_CAST(REPLACE(REPLACE({txt}, '.', ''), ',', '.') AS DOUBLE)
        WHEN regexp_matches({txt}, '^\\s*[-+]?\\d+(,\\d+)?\\s*$')
            THEN TRY_CAST(REPLACE({txt}, ',', '.') AS DOUBLE)
        ELSE TRY_CAST(REPLACE({cleaned}, ',', '.') AS DOUBLE)
    END
    """


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


def cid_presence_expr(cols: Sequence[str], pattern: str) -> Optional[str]:
    tests = []
    for col in cols:
        txt = f"UPPER(COALESCE({clean_str_expr(col)}, ''))"
        tests.append(f"regexp_matches({txt}, {qstr(pattern)})")
    if not tests:
        return None
    return " OR ".join(f"({t})" for t in tests)


def cid_group_expr(cid_sql: str) -> str:
    clauses = [f"WHEN {cid_sql} LIKE {qstr(rule['prefixo'] + '%')} THEN {qstr(rule['grupo'])}" for rule in CID_RULES]
    return f"CASE WHEN {cid_sql} IS NULL THEN 'Sem CID de meningite detectado' {' '.join(clauses)} ELSE 'Outro CID capturado' END"


def cid_type_expr(cid_sql: str) -> str:
    clauses = [f"WHEN {cid_sql} LIKE {qstr(rule['prefixo'] + '%')} THEN {qstr(rule['rotulo'])}" for rule in CID_RULES]
    return f"CASE WHEN {cid_sql} IS NULL THEN 'Sem CID de meningite detectado' {' '.join(clauses)} ELSE 'Outro CID capturado' END"


def _cid10_adequacy_condition(cid_sql: str, rule: Dict[str, str]) -> str:
    if rule.get("match") == "prefix":
        return f"{cid_sql} LIKE {qstr(rule['origem_prefixo'] + '%')}"
    return f"{cid_sql} = {qstr(rule['origem_prefixo'])}"


def cid10_adequacy_original_display_expr(cid_sql: str) -> str:
    clauses = [
        f"WHEN {_cid10_adequacy_condition(cid_sql, rule)} THEN {qstr(rule['origem_padrao'])}"
        for rule in CID10_ADEQUACY_CONVERSION_RULES
    ]
    return f"CASE WHEN {cid_sql} IS NULL THEN NULL {' '.join(clauses)} ELSE {cid_group_expr(cid_sql)} END"


def cid10_adequacy_group_expr(cid_sql: str) -> str:
    clauses = [
        f"WHEN {_cid10_adequacy_condition(cid_sql, rule)} THEN {qstr(rule['destino_grupo'])}"
        for rule in CID10_ADEQUACY_CONVERSION_RULES
    ]
    return f"CASE WHEN {cid_sql} IS NULL THEN 'Sem CID de meningite/encefalite detectado' {' '.join(clauses)} ELSE {cid_group_expr(cid_sql)} END"


def cid10_adequacy_type_expr(cid_sql: str) -> str:
    clauses = [
        f"WHEN {_cid10_adequacy_condition(cid_sql, rule)} THEN {qstr(rule['destino_rotulo'])}"
        for rule in CID10_ADEQUACY_CONVERSION_RULES
    ]
    return f"CASE WHEN {cid_sql} IS NULL THEN 'Sem CID de meningite/encefalite detectado' {' '.join(clauses)} ELSE {cid_type_expr(cid_sql)} END"


def cid10_adequacy_status_expr(cid_sql: str) -> str:
    clauses = [
        f"WHEN {_cid10_adequacy_condition(cid_sql, rule)} THEN 'Convertido'"
        for rule in CID10_ADEQUACY_CONVERSION_RULES
    ]
    return f"CASE WHEN {cid_sql} IS NULL THEN 'Sem CID detectado' {' '.join(clauses)} ELSE 'Fora da conversão — mantido no total' END"


def cid10_adequacy_reason_expr(cid_sql: str) -> str:
    clauses = [
        f"WHEN {_cid10_adequacy_condition(cid_sql, rule)} THEN {qstr(rule['observacao'])}"
        for rule in CID10_ADEQUACY_CONVERSION_RULES
    ]
    return (
        f"CASE WHEN {cid_sql} IS NULL THEN 'Sem CID de meningite/encefalite detectado.' "
        f"{' '.join(clauses)} "
        f"ELSE CONCAT('Fora da conversão operacional: ', {cid_type_expr(cid_sql)}, ' foi mantido no total como CID-10 detectado/não mapeado.') END"
    )


def cid10_adequacy_plot_label_expr(cid_sql: str) -> str:
    """Categoria usada no gráfico resumido de adequação e na comparação.

    Retorna o CID-10 adequado prefixado final: códigos presentes na tabela de
    conversão são deslocados para o destino; códigos já prefixados e não
    convertidos permanecem em seu próprio grupo CID-10. Registros sem CID
    retornam NULL para não entrar no gráfico nem na comparação estratificada.
    """
    converted_or_original_group = cid10_adequacy_group_expr(cid_sql)
    return f"CASE WHEN {cid_sql} IS NULL THEN NULL ELSE {converted_or_original_group} END"


def text_concat_expr(cols: Sequence[str]) -> Optional[str]:
    """Concatena campos textuais detectados automaticamente em uma expressão SQL única.

    Usado para procurar nomes de agentes e doenças de base que não aparecem de forma estruturada em CON_DIAGES.
    """
    clean_cols = [c for c in cols if c]
    if not clean_cols:
        return None
    parts = [f"COALESCE({clean_str_expr(c)}, '')" for c in clean_cols]
    if len(parts) == 1:
        return f"UPPER({parts[0]})"
    return "UPPER(" + " || ' | ' || ".join(parts) + ")"


def _regex_bool_expr(text_sql: Optional[str], pattern: str) -> str:
    if not text_sql:
        return "FALSE"
    return f"regexp_matches(COALESCE({text_sql}, ''), {qstr(pattern)})"


def _sinan_other_bacteria_g01_condition(bacteria_code_sql: Optional[str] = None, aux_text_sql: Optional[str] = None) -> str:
    conditions: List[str] = []
    if bacteria_code_sql:
        g01_codes = ", ".join(qstr(code) for code in sorted(SINAN_CLA_ME_BAC_G01_CODES))
        conditions.append(f"{bacteria_code_sql} IN ({g01_codes})")
    if aux_text_sql:
        conditions.append(_regex_bool_expr(aux_text_sql, SINAN_G01_DETAIL_REGEX))
    return " OR ".join(f"({c})" for c in conditions) if conditions else "FALSE"


def _sinan_con05_detail_expr(bacteria_code_sql: Optional[str] = None, aux_text_sql: Optional[str] = None) -> str:
    g01_condition = _sinan_other_bacteria_g01_condition(bacteria_code_sql, aux_text_sql)
    return f"""
    CASE
        WHEN ({g01_condition}) THEN 'CON_DIAGES=05 refinado como G01 por CLA_ME_BAC/texto: agente/doença bacteriana classificada em outra parte.'
        WHEN {bacteria_code_sql or 'NULL'} IS NULL THEN 'CON_DIAGES=05 sem CLA_ME_BAC detectado/preenchido: classificado conservadoramente como G00.'
        WHEN {bacteria_code_sql or 'NULL'} IN ('81') THEN 'CON_DIAGES=05 com bactéria não especificada: G00.'
        ELSE 'CON_DIAGES=05 com bactéria comum/outra bactéria: G00.'
    END
    """


def sinan_cid10_conversion_group_expr(con_code_sql: str, bacteria_code_sql: Optional[str] = None, aux_text_sql: Optional[str] = None) -> str:
    g01_condition = _sinan_other_bacteria_g01_condition(bacteria_code_sql, aux_text_sql)
    return f"""
    CASE
        WHEN {con_code_sql} IN ('02', '03') THEN 'A39.0'
        WHEN {con_code_sql} = '04' THEN 'A17.0'
        WHEN {con_code_sql} = '05' AND ({g01_condition}) THEN 'G01'
        WHEN {con_code_sql} = '05' THEN 'G00'
        WHEN {con_code_sql} = '06' THEN 'G03'
        WHEN {con_code_sql} = '07' THEN 'A87'
        WHEN {con_code_sql} = '08' THEN 'G02'
        WHEN {con_code_sql} IN ('09', '10') THEN 'G00'
        WHEN {con_code_sql} = '01' THEN 'Não convertido — meningococcemia isolada'
        ELSE 'Sem conversão — CON_DIAGES ausente ou não mapeado'
    END
    """


def sinan_cid10_conversion_type_expr(con_code_sql: str, bacteria_code_sql: Optional[str] = None, aux_text_sql: Optional[str] = None) -> str:
    g01_condition = _sinan_other_bacteria_g01_condition(bacteria_code_sql, aux_text_sql)
    return f"""
    CASE
        WHEN {con_code_sql} IN ('02', '03') THEN 'A39.0 — meningite meningocócica'
        WHEN {con_code_sql} = '04' THEN 'A17.0 — meningite tuberculosa'
        WHEN {con_code_sql} = '05' AND ({g01_condition}) THEN 'G01 — meningite bacteriana em doença classificada em outra parte'
        WHEN {con_code_sql} = '05' THEN 'G00 — meningite bacteriana não classificada em outra parte'
        WHEN {con_code_sql} = '06' THEN 'G03 — meningite por outras causas / não especificada'
        WHEN {con_code_sql} = '07' THEN 'A87 — meningite viral'
        WHEN {con_code_sql} = '08' THEN 'G02 — meningite em outras doenças infecciosas/parasitárias'
        WHEN {con_code_sql} IN ('09', '10') THEN 'G00 — meningite bacteriana não classificada em outra parte'
        WHEN {con_code_sql} = '01' THEN 'Não convertido — meningococcemia isolada'
        ELSE 'Sem conversão — CON_DIAGES ausente ou não mapeado'
    END
    """


def sinan_cid10_conversion_reason_expr(con_code_sql: str, bacteria_code_sql: Optional[str] = None, aux_text_sql: Optional[str] = None) -> str:
    con05_reason = _sinan_con05_detail_expr(bacteria_code_sql, aux_text_sql)
    return f"""
    CASE
        WHEN {con_code_sql} IN ('02', '03') THEN 'Forma meningítica meningocócica: A39.0.'
        WHEN {con_code_sql} = '04' THEN 'Meningite tuberculosa: A17.0.'
        WHEN {con_code_sql} = '05' THEN ({con05_reason})
        WHEN {con_code_sql} = '06' THEN 'Meningite não especificada: G03.'
        WHEN {con_code_sql} = '07' THEN 'Meningite asséptica no SINAN: A87 como leitura operacional viral; validar etiologia quando disponível.'
        WHEN {con_code_sql} = '08' THEN 'Outra etiologia infecciosa/parasitária: G02 como família comparável; validar CLA_ME_ETI.'
        WHEN {con_code_sql} IN ('09', '10') THEN 'Haemophilus influenzae/pneumocócica: G00.'
        WHEN {con_code_sql} = '01' THEN 'Meningococcemia isolada: fora da comparação de meningite.'
        ELSE 'CON_DIAGES ausente ou sem regra.'
    END
    """


def sinan_cid10_conversion_include_expr(con_code_sql: str) -> str:
    mapped = ", ".join(qstr(code) for code in SINAN_CID10_FROM_CON_DIAGES)
    return f"CASE WHEN {con_code_sql} IN ({mapped}) THEN 'Sim' ELSE 'Não' END"



def sinan_g01_base_disease_expr(bacteria_code_sql: Optional[str] = None, aux_text_sql: Optional[str] = None) -> str:
    """Retorna a doença bacteriana de base provável quando a conversão do SINAN cai em G01.

    G01 é uma manifestação em doença bacteriana classificada em outra parte. Assim, esta
    expressão não tenta transformar todos os casos bacterianos em G01; ela só descreve a
    doença de base provável quando os mesmos sinais usados no refinamento G01 aparecem em
    CLA_ME_BAC ou em campos textuais/auxiliares.
    """
    bacteria = bacteria_code_sql or "NULL"
    salmonella_text = _regex_bool_expr(aux_text_sql, r"SALMONEL")
    listeria_text = _regex_bool_expr(aux_text_sql, r"LISTERI")
    syphilis_text = _regex_bool_expr(aux_text_sql, r"NEUROSS[ÍI]FIL|NEUROSYPH|S[ÍI]FIL|SYPHIL|TREPONEMA")
    leptospirosis_text = _regex_bool_expr(aux_text_sql, r"LEPTOSPI")
    anthrax_text = _regex_bool_expr(aux_text_sql, r"CARB[UÚ]NCULO|ANTRAZ|ANTHRAX")
    lyme_text = _regex_bool_expr(aux_text_sql, r"LYME|BORREL")
    typhoid_text = _regex_bool_expr(aux_text_sql, r"TIF[OÓ]IDE|TYPHOID")
    gonococcal_text = _regex_bool_expr(aux_text_sql, r"GONOCOC|GONOCOCO")
    return f"""
    CASE
        WHEN {bacteria} = '11' OR ({salmonella_text}) THEN 'Infecção por Salmonella sp / salmonelose invasiva — A02.2†'
        WHEN {bacteria} = '21' OR ({listeria_text}) THEN 'Listeriose / Listeria monocytogenes — A32.1†'
        WHEN {bacteria} = '45' OR ({syphilis_text}) THEN 'Sífilis / neurossífilis — A52.1†; avaliar A50.4†/A51.4† conforme contexto'
        WHEN {bacteria} = '49' OR ({leptospirosis_text}) THEN 'Leptospirose — A27.-†'
        WHEN ({anthrax_text}) THEN 'Carbúnculo / antraz — A22.8†'
        WHEN ({lyme_text}) THEN 'Doença de Lyme / borreliose — A69.2†'
        WHEN ({typhoid_text}) THEN 'Febre tifóide — A01.0†'
        WHEN ({gonococcal_text}) THEN 'Infecção gonocócica — A54.8†'
        ELSE 'G01 sem doença de base provável identificada nos campos disponíveis'
    END
    """


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
# Integração com assets Parquet da release do GitHub
# =============================================================================


def github_release_download_url(asset_name: str) -> str:
    encoded_name = urllib.parse.quote(asset_name, safe="")
    return (
        f"https://github.com/{GITHUB_RELEASE_OWNER}/{GITHUB_RELEASE_REPO}/"
        f"releases/download/{GITHUB_RELEASE_TAG}/{encoded_name}"
    )


def _asset_source(asset_name: str) -> str:
    upper = asset_name.upper()
    for source, prefix in GITHUB_RELEASE_SOURCE_PREFIX.items():
        if upper.startswith(prefix.upper()):
            return source
    return "OUTROS"


def _asset_year(asset_name: str) -> Optional[int]:
    match = re.search(r"(?:19|20)\d{2}", asset_name)
    if not match:
        return None
    return int(match.group(0))


def _format_bytes(size: object) -> str:
    try:
        value = float(size)
    except (TypeError, ValueError):
        return ""
    units = ["B", "KB", "MB", "GB"]
    idx = 0
    while value >= 1024 and idx < len(units) - 1:
        value /= 1024
        idx += 1
    if idx == 0:
        return f"{int(value)} {units[idx]}"
    return f"{value:.1f} {units[idx]}"


def _github_request(url: str, accept: str = "application/vnd.github+json") -> str:
    req = urllib.request.Request(
        url,
        headers={
            "Accept": accept,
            "User-Agent": "meningite-streamlit-dashboard",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(req, timeout=45) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _normalise_github_asset(raw: Dict[str, object]) -> Dict[str, object]:
    name = str(raw.get("name") or "").strip()
    return {
        "name": name,
        "source": _asset_source(name),
        "year": _asset_year(name),
        "size": raw.get("size"),
        "digest": raw.get("digest") or raw.get("sha256") or "",
        "download_url": raw.get("browser_download_url") or github_release_download_url(name),
        "updated_at": raw.get("updated_at") or raw.get("created_at") or "",
    }


@st.cache_data(show_spinner=False, ttl=3600)
def list_github_release_parquets() -> List[Dict[str, object]]:
    """Lista os Parquets da release GitHub.

    Usa a API pública do GitHub quando disponível, tenta a página HTML expandida
    como segunda opção e, por fim, usa a lista esperada da Release1. A lista de
    fallback evita que o painel quebre em ambiente com rate limit temporário da API.
    """
    assets: List[Dict[str, object]] = []

    try:
        payload = json.loads(_github_request(GITHUB_RELEASE_API_URL))
        for item in payload.get("assets", []):
            name = str(item.get("name") or "")
            if name.lower().endswith(".parquet"):
                assets.append(_normalise_github_asset(item))
    except Exception:
        assets = []

    if not assets:
        try:
            html = _github_request(GITHUB_RELEASE_EXPANDED_ASSETS_URL, accept="text/html")
            seen = set()
            pattern = r'href="([^"]*/releases/download/[^"]+?\.parquet)"[^>]*>\s*([^<]+\.parquet)\s*</a>'
            for href, raw_name in re.findall(pattern, html, flags=re.IGNORECASE):
                name = html_lib.unescape(raw_name).strip()
                if not name or name in seen:
                    continue
                seen.add(name)
                url = href if href.startswith("http") else f"https://github.com{href}"
                assets.append(_normalise_github_asset({"name": name, "browser_download_url": url}))
        except Exception:
            assets = []

    if not assets:
        assets = [_normalise_github_asset({"name": name}) for name in GITHUB_RELEASE_FALLBACK_PARQUETS]

    source_order = {"SINAN": 0, "SIM": 1, "CIHA": 2, "OUTROS": 9}
    return sorted(
        assets,
        key=lambda asset: (
            source_order.get(str(asset.get("source")), 9),
            asset.get("year") or 9999,
            str(asset.get("name")),
        ),
    )


def github_asset_label(asset: Dict[str, object]) -> str:
    year = asset.get("year")
    name = str(asset.get("name") or "")
    size = _format_bytes(asset.get("size"))
    prefix = f"{year} — " if year else ""
    suffix = f" ({size})" if size else ""
    return f"{prefix}{name}{suffix}"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().lower()


def download_github_release_asset_to_path(asset_name: str, download_url: str, out_path: Path, digest: str = "") -> None:
    """Baixa um asset para disco em streaming, sem manter o Parquet inteiro em memória."""
    req = urllib.request.Request(
        download_url,
        headers={
            "Accept": "application/octet-stream",
            "User-Agent": "meningite-streamlit-dashboard",
        },
    )
    expected_sha256 = ""
    if digest and str(digest).startswith("sha256:"):
        expected_sha256 = str(digest).split(":", 1)[1].lower()

    tmp_path = out_path.with_suffix(out_path.suffix + ".download")
    h = hashlib.sha256() if expected_sha256 else None
    total = 0
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, tmp_path.open("wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                total += len(chunk)
                if h is not None:
                    h.update(chunk)

        if total == 0:
            raise ValueError(f"Download vazio para {asset_name}.")

        if expected_sha256:
            observed = h.hexdigest().lower() if h is not None else ""
            if observed != expected_sha256:
                raise ValueError(
                    f"SHA-256 divergente para {asset_name}: esperado {expected_sha256}, obtido {observed}."
                )

        tmp_path.replace(out_path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def materialize_github_release_asset_cached(asset_name: str, download_url: str, digest: str = "") -> str:
    digest_key = hashlib.sha1(f"{GITHUB_RELEASE_TAG}|{download_url}|{digest}".encode("utf-8")).hexdigest()[:16]
    out_dir = Path(tempfile.gettempdir()) / "meningite_github_release"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{safe_filename(Path(asset_name).stem)}_{digest_key}.parquet"

    should_download = not out.exists() or out.stat().st_size == 0
    if not should_download and digest and str(digest).startswith("sha256:"):
        expected = str(digest).split(":", 1)[1].lower()
        should_download = _sha256_file(out) != expected

    if should_download:
        download_github_release_asset_to_path(asset_name, download_url, out, digest)
    return str(out)


def materialize_github_release_asset(asset: Dict[str, object]) -> str:
    name = str(asset.get("name") or "")
    if not name:
        raise ValueError("Asset GitHub sem nome.")
    url = str(asset.get("download_url") or github_release_download_url(name))
    digest = str(asset.get("digest") or "")
    return materialize_github_release_asset_cached(name, url, digest)


def github_selection_summary(assets: Sequence[Dict[str, object]]) -> str:
    years = sorted({a.get("year") for a in assets if a.get("year")})
    if not years:
        return f"{len(assets)} parquet(s)"
    if len(years) == 1:
        return f"{len(assets)} parquet(s), ano {years[0]}"
    return f"{len(assets)} parquet(s), {years[0]}–{years[-1]}"


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
    runtime_settings = (
        DEFAULT_DUCKDB_MEMORY_LIMIT,
        DEFAULT_DUCKDB_THREADS,
        str(Path(tempfile.gettempdir()) / DUCKDB_TEMP_SUBDIR),
    )
    con = open_duckdb_connection(path, read_only=True, runtime_settings=runtime_settings)
    try:
        return [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    finally:
        con.close()


def parquet_ref(paths: Sequence[str]) -> str:
    quoted = ", ".join(qstr(p) for p in paths)
    return f"read_parquet([{quoted}], union_by_name=true)"


def materialize_upload(upload, namespace: str) -> str:
    """Materializa upload em arquivo temporário sem duplicar todo o conteúdo em memória."""
    upload_id = str(getattr(upload, "file_id", "") or "")
    upload_size = str(getattr(upload, "size", "") or "")
    session_key = ""
    if upload_id:
        session_key = f"materialized_upload::{namespace}::{upload_id}::{upload.name}::{upload_size}"
        cached_path = st.session_state.get(session_key)
        if cached_path and Path(str(cached_path)).exists():
            return str(cached_path)

    suffix = Path(upload.name).suffix or ".dat"
    clean_name = safe_filename(Path(upload.name).stem)
    temp_dir = Path(tempfile.gettempdir())
    temp_dir.mkdir(parents=True, exist_ok=True)

    digest_obj = hashlib.sha1()
    tmp = tempfile.NamedTemporaryFile(
        prefix=f"meningite_{namespace}_{clean_name}_",
        suffix=f"{suffix}.tmp",
        dir=temp_dir,
        delete=False,
    )
    tmp_path = Path(tmp.name)

    try:
        upload.seek(0)
        with tmp:
            while True:
                chunk = upload.read(UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                digest_obj.update(chunk)
                tmp.write(chunk)
        upload.seek(0)

        digest = digest_obj.hexdigest()[:16]
        out = temp_dir / f"meningite_{namespace}_{clean_name}_{digest}{suffix}"
        if out.exists():
            tmp_path.unlink(missing_ok=True)
        else:
            tmp_path.replace(out)
        if session_key:
            st.session_state[session_key] = str(out)
        return str(out)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        try:
            upload.seek(0)
        except Exception:
            pass
        raise


def _safe_duckdb_memory_limit(value: object) -> str:
    """Normaliza limite de memória aceito pelo DuckDB, com fallback seguro."""
    text = str(value or DEFAULT_DUCKDB_MEMORY_LIMIT).strip().upper().replace(" ", "")
    if re.fullmatch(r"(?:[1-9]\d*)(?:\.\d+)?(?:KB|MB|GB|TB)|(?:[1-9]\d?|100)%", text):
        return text
    return DEFAULT_DUCKDB_MEMORY_LIMIT


def duckdb_runtime_settings() -> Tuple[str, int, str]:
    """Configuração leve para reduzir picos de memória em consultas Parquet/DuckDB."""
    memory_limit = _safe_duckdb_memory_limit(
        st.session_state.get("perf_duckdb_memory_limit", DEFAULT_DUCKDB_MEMORY_LIMIT)
    )
    threads = max(1, perf_int("perf_duckdb_threads", DEFAULT_DUCKDB_THREADS))
    temp_dir = str(Path(tempfile.gettempdir()) / DUCKDB_TEMP_SUBDIR)
    return memory_limit, threads, temp_dir


def configure_duckdb_connection(
    con: duckdb.DuckDBPyConnection,
    runtime_settings: Optional[Tuple[str, int, str]] = None,
) -> None:
    """Aplica limites defensivos ao DuckDB sem interromper o app se uma opção falhar."""
    memory_limit, threads, temp_dir = runtime_settings or duckdb_runtime_settings()
    temp_path = Path(temp_dir)
    try:
        temp_path.mkdir(parents=True, exist_ok=True)
    except Exception:
        temp_path = Path(tempfile.gettempdir())

    statements = [
        f"SET memory_limit={qstr(memory_limit)}",
        f"SET threads={int(max(1, threads))}",
        f"SET temp_directory={qstr(str(temp_path))}",
        "SET preserve_insertion_order=false",
        "SET enable_object_cache=true",
    ]
    for statement in statements:
        try:
            con.execute(statement)
        except Exception:
            # Algumas versões/ambientes bloqueiam opções específicas; a consulta deve seguir funcionando.
            pass


def open_duckdb_connection(
    db_path: Optional[str] = None,
    read_only: bool = False,
    runtime_settings: Optional[Tuple[str, int, str]] = None,
) -> duckdb.DuckDBPyConnection:
    """Abre conexão DuckDB com spill para disco e limite de memória configurável."""
    if db_path:
        con = duckdb.connect(db_path, read_only=read_only)
    else:
        con = duckdb.connect(database=":memory:")
    configure_duckdb_connection(con, runtime_settings)
    return con


def _file_fingerprint(path: Optional[str]) -> Tuple[str, Optional[int], Optional[int]]:
    if not path:
        return "", None, None
    try:
        stat = Path(path).stat()
        return str(path), int(stat.st_size), int(stat.st_mtime_ns)
    except OSError:
        return str(path), None, None


def table_cache_key(table: LoadedTable) -> Tuple[object, ...]:
    """Chave leve para invalidar cache quando arquivos mudam."""
    parquet_meta = tuple(_file_fingerprint(path) for path in (table.parquet_paths or []))
    duckdb_meta = _file_fingerprint(table.db_path)
    return (table.kind, table.ref_sql, duckdb_meta, parquet_meta)


# =============================================================================
# Conexão DuckDB persistente + materialização de Parquets
# -----------------------------------------------------------------------------
# Antes, cada consulta abria uma conexão :memory: nova e embutia
# read_parquet([...]) no FROM, re-decodificando os mesmos Parquets a cada
# query (dezenas por render). Agora mantemos UMA conexão in-memory viva entre
# reruns (cache_resource) e materializamos cada base Parquet em uma tabela
# nativa do DuckDB uma única vez. As consultas seguintes leem armazenamento
# colunar nativo (sem reparse de Parquet, com predicate pushdown e zone maps).
# =============================================================================

class _SharedDB:
    """Encapsula a conexão in-memory persistente, o registro de objetos já
    materializados e o lock de DDL — todos com o mesmo ciclo de vida.

    Usar um único objeto cacheado evita atribuir atributos no tipo C do DuckDB
    e garante que conexão e registro nunca fiquem dessincronizados (ex.: se um
    fosse despejado do cache e o outro não).
    """

    def __init__(self, con: duckdb.DuckDBPyConnection) -> None:
        self.con = con
        self.registry: Dict[str, str] = {}
        self.lock = threading.Lock()


@st.cache_resource(show_spinner=False)
def get_duckdb_file_db(
    db_path: str,
    runtime_settings: Tuple[str, int, str],
    file_fingerprint: Tuple[str, Optional[int], Optional[int]],
) -> "_SharedDB":
    """Mantém conexão read-only de DuckDB entre consultas da mesma base.

    O fingerprint entra na chave do cache; se o arquivo mudar, a conexão antiga
    deixa de ser reutilizada para as novas consultas.
    """
    con = duckdb.connect(db_path, read_only=True)
    configure_duckdb_connection(con, runtime_settings)
    return _SharedDB(con)


def _query_duckdb_file(db_path: Optional[str], sql: str, runtime_settings: Tuple[str, int, str]) -> pd.DataFrame:
    if not db_path:
        return pd.DataFrame()
    shared = get_duckdb_file_db(db_path, runtime_settings, _file_fingerprint(db_path))
    _ensure_municipios_ibge_view(shared)
    with shared.lock:
        cur = shared.con.cursor()
        try:
            return cur.execute(sql).df()
        finally:
            cur.close()


def parquet_object_name(source: str, paths: Sequence[str]) -> str:
    """Identificador estável da base Parquet, derivado do conteúdo dos arquivos.

    Inclui caminho + tamanho + mtime no hash: se qualquer arquivo mudar, o nome
    muda, forçando a recriação da tabela materializada e invalidando o cache.
    """
    fingerprints = [_file_fingerprint(p) for p in (paths or [])]
    raw = json.dumps([source, fingerprints], sort_keys=True, default=str)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    safe_source = re.sub(r"[^0-9A-Za-z]+", "_", str(source)) or "src"
    return f"pq_{safe_source}_{digest}"



def _fastparquet_available() -> bool:
    """Retorna se fastparquet foi importado com sucesso no ambiente atual."""
    return fp is not None


def _should_use_fastparquet() -> bool:
    """Controla o uso de fastparquet na etapa de materialização Parquet -> DuckDB."""
    return bool(st.session_state.get("perf_use_fastparquet", True))


def _fastparquet_row_limit() -> int:
    """Limite defensivo de linhas para evitar carregar Parquets grandes demais em pandas."""
    limit = perf_int("perf_fastparquet_row_limit", DEFAULT_FASTPARQUET_ROW_LIMIT)
    return max(1000, int(limit))


def _fastparquet_file_rows(path: str) -> Optional[int]:
    """Lê apenas metadados de um arquivo Parquet com fastparquet para estimar linhas."""
    if fp is None:
        return None
    try:
        pf = fp.ParquetFile(path)
        row_groups = getattr(pf, "row_groups", None) or []
        total = 0
        for row_group in row_groups:
            total += int(getattr(row_group, "num_rows", 0) or 0)
        if total > 0:
            return total
        info = getattr(pf, "info", None)
        if isinstance(info, dict):
            rows = info.get("rows") or info.get("num_rows")
            if rows is not None:
                return int(rows)
    except Exception:
        return None
    return None


def _fastparquet_total_rows(paths: Sequence[str]) -> Optional[int]:
    """Soma metadados de linhas quando disponíveis; retorna None se a estimativa falhar."""
    total = 0
    for path in paths:
        rows = _fastparquet_file_rows(str(path))
        if rows is None:
            return None
        total += int(rows)
    return total


def fastparquet_status() -> str:
    """Texto curto exibido na barra lateral sobre o motor de leitura Parquet."""
    if not _should_use_fastparquet():
        return "fastparquet desativado: Parquets serão materializados pelo leitor nativo do DuckDB ou usados como VIEW."
    if not _fastparquet_available():
        return "fastparquet não está instalado neste ambiente; o app fará fallback automático para DuckDB read_parquet."
    return (
        "fastparquet ativo para materialização de Parquets até "
        f"{_fastparquet_row_limit():,} linhas estimadas; DuckDB permanece como motor SQL."
    ).replace(",", ".")


def _load_parquets_fastparquet(paths: Sequence[str]) -> pd.DataFrame:
    """Lê Parquets com fastparquet e une por nome de coluna, preservando fallback externo."""
    if fp is None:
        raise RuntimeError("fastparquet não está instalado")
    frames: List[pd.DataFrame] = []
    for path in paths:
        parquet_file = fp.ParquetFile(str(path))
        frames.append(parquet_file.to_pandas())
    if not frames:
        return pd.DataFrame()
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, ignore_index=True, sort=False, copy=False)


def _can_try_fastparquet(paths: Sequence[str]) -> bool:
    """Decide se vale tentar fastparquet antes do leitor nativo do DuckDB."""
    if not _should_use_fastparquet() or not _fastparquet_available():
        return False
    total_rows = _fastparquet_total_rows(paths)
    return total_rows is None or total_rows <= _fastparquet_row_limit()

def _should_materialize() -> bool:
    """Materializar Parquet em tabela nativa (padrão) ou usar VIEW (lazy)."""
    return bool(st.session_state.get("perf_materialize_tables", True))


@st.cache_resource(show_spinner=False)
def get_shared_db(runtime_settings: Tuple[str, int, str]) -> "_SharedDB":
    """Conexão in-memory única, reutilizada entre reruns e consultas.

    Persistir a conexão evita reconectar, reconfigurar PRAGMAs e relistar
    arquivos a cada query, além de manter o catálogo (tabelas materializadas)
    e o cache de metadados de Parquet aquecidos.
    """
    con = duckdb.connect(database=":memory:")
    configure_duckdb_connection(con, runtime_settings)
    return _SharedDB(con)


def _ensure_parquet_object(
    shared: "_SharedDB",
    name: str,
    paths: Sequence[str],
    materialize: bool,
) -> None:
    """Garante que `name` exista na conexão como tabela ou VIEW.

    Se materialização estiver ativa, tenta fastparquet primeiro. O fastparquet
    decodifica os Parquets para pandas; o DataFrame é registrado no DuckDB e
    transformado em tabela nativa. Se houver incompatibilidade de ambiente,
    tipo, volume ou memória, o código cai para o read_parquet nativo do DuckDB.
    """
    desired_kind = (
        "table_fastparquet" if materialize and _should_use_fastparquet()
        else "table_duckdb" if materialize
        else "view"
    )
    if shared.registry.get(name) == desired_kind:
        return
    with shared.lock:
        if shared.registry.get(name) == desired_kind:
            return
        con = shared.con
        ident = qident(name)
        src = parquet_ref(paths)
        # Remove qualquer objeto anterior de mesmo nome (ex.: troca tabela<->view/fastparquet).
        for drop_stmt in (f"DROP VIEW IF EXISTS {ident}", f"DROP TABLE IF EXISTS {ident}"):
            try:
                con.execute(drop_stmt)
            except Exception:
                pass

        made: Optional[str] = None
        if materialize and _can_try_fastparquet(paths):
            temp_view = f"__fp_{hashlib.sha1(name.encode('utf-8')).hexdigest()[:12]}"
            df_fastparquet: Optional[pd.DataFrame] = None
            try:
                df_fastparquet = _load_parquets_fastparquet(paths)
                con.register(temp_view, df_fastparquet)
                con.execute(f"CREATE TABLE {ident} AS SELECT * FROM {qident(temp_view)}")
                made = "table_fastparquet"
            except Exception as exc:
                st.session_state[f"fastparquet_fallback::{name}"] = str(exc)
                made = None
            finally:
                try:
                    con.unregister(temp_view)
                except Exception:
                    pass
                try:
                    del df_fastparquet
                except Exception:
                    pass

        if materialize and made is None:
            try:
                con.execute(f"CREATE TABLE {ident} AS SELECT * FROM {src}")
                made = "table_duckdb"
            except Exception:
                made = None  # fallback para VIEW abaixo
        if made is None:
            con.execute(f"CREATE VIEW {ident} AS SELECT * FROM {src}")
            made = "view"

        # O registro usa a configuração desejada para evitar reprocessar a mesma base.
        # Se fastparquet falhou e houve fallback válido, a tabela/VIEW permanece pronta.
        shared.registry[name] = desired_kind if materialize else made

def _prepare_parquet(table: LoadedTable, runtime_settings: Tuple[str, int, str]) -> None:
    """Registra a base Parquet na conexão persistente antes de consultar."""
    if table.kind == "parquet" and table.parquet_paths:
        shared = get_shared_db(runtime_settings)
        _ensure_parquet_object(shared, table.ref_sql, table.parquet_paths, _should_materialize())


def _query_shared(sql: str, runtime_settings: Tuple[str, int, str]) -> pd.DataFrame:
    """Executa na conexão persistente usando um cursor isolado (seguro p/ threads)."""
    shared = get_shared_db(runtime_settings)
    _ensure_municipios_ibge_view(shared)
    cur = shared.con.cursor()
    try:
        return cur.execute(sql).df()
    finally:
        cur.close()


def _execute_query_uncached(table: LoadedTable, sql: str) -> pd.DataFrame:
    runtime_settings = duckdb_runtime_settings()
    if table.kind == "duckdb":
        return _query_duckdb_file(table.db_path, sql, runtime_settings)
    return _query_shared(sql, runtime_settings)


@st.cache_data(show_spinner=False, ttl=1800, max_entries=DEFAULT_QUERY_CACHE_MAX_ENTRIES)
def _run_query_cached(
    table_key: Tuple[object, ...],
    kind: str,
    db_path: Optional[str],
    sql: str,
    runtime_settings: Tuple[str, int, str],
) -> pd.DataFrame:
    if kind == "duckdb":
        return _query_duckdb_file(db_path, sql, runtime_settings)
    return _query_shared(sql, runtime_settings)


def run_query(table: LoadedTable, sql: str, cache: bool = True) -> pd.DataFrame:
    """Executa SQL; por padrão cacheia apenas resultados de consulta agregada/pequena."""
    runtime_settings = duckdb_runtime_settings()
    _prepare_parquet(table, runtime_settings)
    if cache:
        return _run_query_cached(
            table_cache_key(table),
            table.kind,
            table.db_path,
            sql,
            runtime_settings,
        )
    return _execute_query_uncached(table, sql)


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
    education_col: Optional[str] = None
    # SINAN
    classi_fin_col: Optional[str] = None
    con_diages_col: Optional[str] = None
    cla_me_bac_col: Optional[str] = None
    cla_me_ass_col: Optional[str] = None
    cla_me_eti_col: Optional[str] = None
    sinan_auxiliary_cid10_cols: Optional[List[str]] = None
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
    procedimento_col: Optional[str] = None


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
    education_candidates = {
        "SINAN": ["CS_ESCOL_N", "ESCOLARIDADE", "ESCOLARI", "CS_ESCOL", "ESCOL_N"],
        "SIM": ["ESC2010", "ESC", "ESCOLARIDADE", "ESCOLARI", "ESCFALAGR1"],
    }.get(source, [])
    sel.education_col = choose_candidate(columns, education_candidates)
    if source == "SINAN":
        sel.classi_fin_col = choose_candidate(columns, ["CLASSI_FIN"])
        sel.con_diages_col = choose_candidate(columns, ["CON_DIAGES"])
        sel.cla_me_bac_col = choose_candidate(columns, ["CLA_ME_BAC", "CLASSIFICACAO_BACTERIA", "CLASS_BACTERIA"])
        sel.cla_me_ass_col = choose_candidate(columns, ["CLA_ME_ASS", "CLASSIFICACAO_ASSEPTICA", "CLASS_ASSEPTICA"])
        sel.cla_me_eti_col = choose_candidate(columns, ["CLA_ME_ETI", "CLASSIFICACAO_ETIOLOGIA", "CLASS_ETIOLOGIA"])
        sel.sinan_auxiliary_cid10_cols = choose_candidates(columns, SINAN_AUXILIARY_CID10_CANDIDATES, max_items=12)
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
        sel.procedimento_col = choose_candidate(columns, ["PROC_REA", "PROC_REALIZADO", "PROCEDIMENTO", "PROCED", "COD_PROC", "COD_PROCEDIMENTO", "PROC_SOLIC", "PROC_ID", "PROC_PRINC", "PROCEDIMENTO_PRINCIPAL"])
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
        "education": education_label_expr(source, sel.education_col) if sel.education_col else None,
        "mun_res": clean_str_expr(sel.municipality_res_col) if sel.municipality_res_col else None,
        "mun_event": clean_str_expr(sel.municipality_event_col) if sel.municipality_event_col else None,
        "mun_res_label": municipality_display_expr(sel.municipality_res_col) if sel.municipality_res_col else None,
        "mun_event_label": municipality_display_expr(sel.municipality_event_col) if sel.municipality_event_col else None,
        "cid": cid_extract_expr(sel.cid_cols),
        "cid_source": cid_source_expr(sel.cid_cols),
        "cid_g01_present": cid_presence_expr(sel.cid_cols, CID_G01_PRESENT_REGEX),
        "cid_g02_present": cid_presence_expr(sel.cid_cols, CID_G02_PRESENT_REGEX),
    }
    if exprs["cid"]:
        exprs["cid_group"] = cid_group_expr(exprs["cid"])
        exprs["cid_type"] = cid_type_expr(exprs["cid"])
        exprs["cid10_adequacy_group"] = cid10_adequacy_group_expr(exprs["cid"])
        exprs["cid10_adequacy_type"] = cid10_adequacy_type_expr(exprs["cid"])
        exprs["cid10_adequacy_status"] = cid10_adequacy_status_expr(exprs["cid"])
        exprs["cid10_adequacy_reason"] = cid10_adequacy_reason_expr(exprs["cid"])
        exprs["cid10_adequacy_plot_label"] = cid10_adequacy_plot_label_expr(exprs["cid"])
    else:
        exprs["cid_group"] = None
        exprs["cid_type"] = None
        exprs["cid10_adequacy_group"] = None
        exprs["cid10_adequacy_type"] = None
        exprs["cid10_adequacy_status"] = None
        exprs["cid10_adequacy_reason"] = None
        exprs["cid10_adequacy_plot_label"] = None

    if source == "SINAN":
        exprs["classi_code"] = clean_code_expr(sel.classi_fin_col) if sel.classi_fin_col else None
        exprs["classi_label"] = case_from_mapping(exprs["classi_code"], SINAN_CLASSI_FIN, "Sem classificação/ignorado") if exprs["classi_code"] else None
        exprs["evol_code"] = clean_code_expr(sel.evolucao_col) if sel.evolucao_col else None
        exprs["evol_label"] = case_from_mapping(exprs["evol_code"], SINAN_EVOLUCAO, "Sem evolução/ignorado") if exprs["evol_code"] else None
        exprs["con_code"] = clean_code_expr(sel.con_diages_col, pad2=True) if sel.con_diages_col else None
        exprs["con_label"] = case_from_mapping(exprs["con_code"], SINAN_CON_DIAGES, "Sem conclusão diagnóstica/ignorado") if exprs["con_code"] else None
        exprs["con_group"] = case_from_mapping(exprs["con_code"], SINAN_CON_GROUP, "Sem conclusão diagnóstica/ignorado") if exprs["con_code"] else None
        exprs["cla_me_bac_code"] = clean_code_expr(sel.cla_me_bac_col, pad2=True) if sel.cla_me_bac_col else None
        exprs["cla_me_bac_label"] = case_from_mapping(exprs["cla_me_bac_code"], SINAN_CLA_ME_BAC, "Sem bactéria especificada/ignorado") if exprs["cla_me_bac_code"] else None
        exprs["cla_me_ass_code"] = clean_code_expr(sel.cla_me_ass_col, pad2=True) if sel.cla_me_ass_col else None
        exprs["cla_me_ass_label"] = case_from_mapping(exprs["cla_me_ass_code"], SINAN_CLA_ME_ASS, "Sem agente viral/asséptico especificado") if exprs["cla_me_ass_code"] else None
        exprs["cla_me_eti_code"] = clean_code_expr(sel.cla_me_eti_col, pad2=True) if sel.cla_me_eti_col else None
        exprs["cla_me_eti_label"] = case_from_mapping(exprs["cla_me_eti_code"], SINAN_CLA_ME_ETI, "Sem outra etiologia especificada") if exprs["cla_me_eti_code"] else None
        exprs["sinan_aux_text"] = text_concat_expr(sel.sinan_auxiliary_cid10_cols or [])
        if exprs["con_code"]:
            exprs["sinan_cid10_conversion_group"] = sinan_cid10_conversion_group_expr(exprs["con_code"], exprs.get("cla_me_bac_code"), exprs.get("sinan_aux_text"))
            exprs["sinan_cid10_conversion_type"] = sinan_cid10_conversion_type_expr(exprs["con_code"], exprs.get("cla_me_bac_code"), exprs.get("sinan_aux_text"))
            exprs["sinan_cid10_conversion_reason"] = sinan_cid10_conversion_reason_expr(exprs["con_code"], exprs.get("cla_me_bac_code"), exprs.get("sinan_aux_text"))
            exprs["sinan_cid10_conversion_include"] = sinan_cid10_conversion_include_expr(exprs["con_code"])
            exprs["sinan_g01_base_disease"] = sinan_g01_base_disease_expr(exprs.get("cla_me_bac_code"), exprs.get("sinan_aux_text"))
        else:
            exprs["sinan_cid10_conversion_group"] = None
            exprs["sinan_cid10_conversion_type"] = None
            exprs["sinan_cid10_conversion_reason"] = None
            exprs["sinan_cid10_conversion_include"] = None
            exprs["sinan_g01_base_disease"] = None
        exprs["criterio_code"] = clean_code_expr(sel.criterio_col) if sel.criterio_col else None
        exprs["criterio_label"] = case_from_mapping(exprs["criterio_code"], SINAN_CRITERIO, "Sem critério/ignorado") if exprs["criterio_code"] else None
        exprs["puncao_code"] = clean_code_expr(sel.lab_puncao_col) if sel.lab_puncao_col else None
        exprs["puncao_label"] = case_from_mapping(exprs["puncao_code"], YES_NO_IGN, "Sem informação") if exprs["puncao_code"] else None
        exprs["quimio_code"] = clean_code_expr(sel.lab_liquor_col) if sel.lab_liquor_col else None
        exprs["quimio_label"] = case_from_mapping(exprs["quimio_code"], YES_NO_IGN, "Sem informação") if exprs["quimio_code"] else None
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
        exprs["procedimento_label"] = clean_str_expr(sel.procedimento_col) if sel.procedimento_col else None
    return exprs


# =============================================================================
# Queries analíticas
# =============================================================================


def query_timeseries(table: LoadedTable, dt_sql: str, where_sql: str, freq: str, category_sql: Optional[str] = None) -> pd.DataFrame:
    if category_sql:
        cat_sql = category_label_expr(category_sql)
        sql = f"""
            WITH base AS (
                SELECT {dt_sql} AS dt, {cat_sql} AS categoria
                FROM {table.ref_sql}
                {where_sql}
            )
            SELECT date_trunc({qstr(freq)}, dt) AS periodo, categoria, COUNT(*) AS n
            FROM base
            WHERE dt IS NOT NULL
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
    cat_sql = category_label_expr(category_sql)
    sql = f"""
        SELECT {cat_sql} AS categoria, COUNT(*) AS n
        FROM {table.ref_sql}
        {where_sql}
        GROUP BY 1
        ORDER BY n DESC, categoria
        LIMIT {int(top_n)}
    """
    df = run_query(table, sql)
    if not df.empty:
        df["pct"] = (df["n"] / df["n"].sum() * 100).round(2)
    return df


def query_category_top_with_outros(table: LoadedTable, category_sql: str, where_sql: str, top_n: int = 15, outros_label: str = "Outros municípios") -> pd.DataFrame:
    cat_sql = category_label_expr(category_sql)
    sql = f"""
        WITH counts AS (
            SELECT {cat_sql} AS categoria, COUNT(*) AS n
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


def query_sinan_education_outcomes(
    table: LoadedTable,
    education_sql: str,
    classi_sql: str,
    evol_sql: str,
    where_sql: str,
) -> pd.DataFrame:
    escolaridade_labels = education_category_labels("SINAN", include_missing=True)
    cats_cte = values_cte_from_labels(escolaridade_labels, "escolaridade", "ordem_escolaridade")
    group_values = ", ".join(
        f"({qstr(label)}, {idx})"
        for idx, label in enumerate(["Casos confirmados", "Óbitos por meningite", "Óbitos por outra causa"], start=1)
    )
    sql = f"""
        WITH base AS (
            SELECT
                COALESCE({education_sql}, 'Sem informação/ignorado') AS escolaridade,
                {classi_sql} AS classi,
                {evol_sql} AS evol
            FROM {table.ref_sql}
            {where_sql}
        ), categorias AS (
            {cats_cte}
        ), grupos_ref AS (
            SELECT * FROM (VALUES {group_values}) AS t(grupo, ordem_grupo)
        ), grid AS (
            SELECT g.grupo, g.ordem_grupo, c.escolaridade, c.ordem_escolaridade
            FROM grupos_ref g
            CROSS JOIN categorias c
        ), grupos AS (
            SELECT 'Casos confirmados' AS grupo, escolaridade
            FROM base
            WHERE classi = '1'
            UNION ALL
            SELECT 'Óbitos por meningite' AS grupo, escolaridade
            FROM base
            WHERE classi = '1' AND evol = '2'
            UNION ALL
            SELECT 'Óbitos por outra causa' AS grupo, escolaridade
            FROM base
            WHERE classi = '1' AND evol = '3'
        ), counts AS (
            SELECT grupo, escolaridade, COUNT(*) AS n
            FROM grupos
            GROUP BY 1, 2
        ), total_confirmados AS (
            SELECT COUNT(*) FILTER (WHERE classi = '1') AS denominador
            FROM base
        )
        SELECT
            grid.grupo,
            grid.escolaridade,
            COALESCE(counts.n, 0) AS n,
            total_confirmados.denominador,
            CASE WHEN total_confirmados.denominador > 0
                 THEN ROUND(100.0 * COALESCE(counts.n, 0) / total_confirmados.denominador, 2)
                 ELSE NULL END AS pct,
            grid.ordem_escolaridade,
            grid.ordem_grupo
        FROM grid
        CROSS JOIN total_confirmados
        LEFT JOIN counts
          ON counts.grupo = grid.grupo
         AND counts.escolaridade = grid.escolaridade
        ORDER BY grid.ordem_escolaridade, grid.ordem_grupo
    """
    return run_query(table, sql)


def query_education_distribution_all_categories(
    table: LoadedTable,
    source: str,
    education_sql: str,
    where_sql: str,
) -> pd.DataFrame:
    labels = education_category_labels(source, education_sql, include_missing=True)
    cats_cte = values_cte_from_labels(labels, "categoria", "ordem_categoria")
    sql = f"""
        WITH categorias AS (
            {cats_cte}
        ), base AS (
            SELECT COALESCE({education_sql}, 'Sem informação/ignorado') AS categoria
            FROM {table.ref_sql}
            {where_sql}
        ), counts AS (
            SELECT categoria, COUNT(*) AS n
            FROM base
            GROUP BY 1
        ), total AS (
            SELECT COUNT(*) AS denominador
            FROM base
        )
        SELECT
            categorias.categoria,
            COALESCE(counts.n, 0) AS n,
            total.denominador,
            CASE WHEN total.denominador > 0
                 THEN ROUND(100.0 * COALESCE(counts.n, 0) / total.denominador, 2)
                 ELSE NULL END AS pct,
            categorias.ordem_categoria
        FROM categorias
        CROSS JOIN total
        LEFT JOIN counts USING (categoria)
        ORDER BY categorias.ordem_categoria
    """
    return run_query(table, sql)

def query_yearly_category(table: LoadedTable, dt_sql: str, category_sql: str, where_sql: str) -> pd.DataFrame:
    cat_sql = category_label_expr(category_sql)
    sql = f"""
        WITH base AS (
            SELECT {dt_sql} AS dt, {cat_sql} AS categoria
            FROM {table.ref_sql}
            {where_sql}
        ), counts AS (
            SELECT EXTRACT(YEAR FROM dt) AS ano, categoria, COUNT(*) AS n
            FROM base
            WHERE dt IS NOT NULL
            GROUP BY 1, 2
        ), with_totals AS (
            SELECT ano, categoria, n,
                   SUM(n) OVER (PARTITION BY ano) AS total_ano
            FROM counts
        )
        SELECT ano, categoria, n, total_ano,
               CASE WHEN total_ano > 0
                    THEN ROUND(100.0 * n / total_ano, 2)
                    ELSE NULL END AS pct
        FROM with_totals
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


def query_g01_g02_cid_distribution(table: LoadedTable, exprs: Dict[str, Optional[str]], where_sql: str) -> pd.DataFrame:
    """Tabula G01/G02 em SIM/CIHA a partir do CID-10 bruto detectado.

    A verificação usa presença de G01*/G02* em qualquer campo CID detectado, não apenas
    o primeiro CID priorizado pela distribuição geral. Assim, G01/G02 não são perdidos
    quando aparecem como diagnóstico/menção associado após outro CID de meningite.
    """
    g01 = exprs.get("cid_g01_present")
    g02 = exprs.get("cid_g02_present")
    if not (g01 or g02):
        return pd.DataFrame()
    g01_sql = g01 or "FALSE"
    g02_sql = g02 or "FALSE"
    sql = f"""
        WITH base AS (
            SELECT ({g01_sql}) AS tem_g01,
                   ({g02_sql}) AS tem_g02
            FROM {table.ref_sql}
            {where_sql}
        ), agg AS (
            SELECT 'G01' AS grupo,
                   'G01 — meningite bacteriana em doença classificada em outra parte' AS tipo,
                   COUNT(*) FILTER (WHERE tem_g01) AS n
            FROM base
            UNION ALL
            SELECT 'G02' AS grupo,
                   'G02 — meningite em outras doenças infecciosas/parasitárias' AS tipo,
                   COUNT(*) FILTER (WHERE tem_g02) AS n
            FROM base
        ), with_totals AS (
            SELECT *, SUM(n) OVER () AS denominador
            FROM agg
        )
        SELECT *,
               CASE WHEN denominador > 0 THEN ROUND(100.0 * n / denominador, 2) ELSE NULL END AS pct
        FROM with_totals
        WHERE n > 0
        ORDER BY grupo
    """
    return run_query(table, sql)



def query_cid10_adequacy_conversion(table: LoadedTable, exprs: Dict[str, Optional[str]], where_sql: str) -> pd.DataFrame:
    cid = exprs.get("cid")
    if not cid:
        return pd.DataFrame()
    source_expr = exprs.get("cid_source") or "NULL"
    original_display = cid10_adequacy_original_display_expr(cid)
    original_type = cid_type_expr(cid)
    converted_group = exprs.get("cid10_adequacy_group") or cid10_adequacy_group_expr(cid)
    converted_type = exprs.get("cid10_adequacy_type") or cid10_adequacy_type_expr(cid)
    status = exprs.get("cid10_adequacy_status") or cid10_adequacy_status_expr(cid)
    reason = exprs.get("cid10_adequacy_reason") or cid10_adequacy_reason_expr(cid)
    plot_label = exprs.get("cid10_adequacy_plot_label") or cid10_adequacy_plot_label_expr(cid)
    sql = f"""
        WITH base AS (
            SELECT {cid} AS cid10_detectado,
                   {original_display} AS cid10_original,
                   {original_type} AS cid10_original_classificacao,
                   {converted_group} AS cid10_adequado_grupo,
                   {converted_type} AS cid10_adequado_classificacao,
                   {status} AS status_conversao,
                   {reason} AS observacao_conversao,
                   {plot_label} AS categoria_grafico,
                   {source_expr} AS coluna_origem
            FROM {table.ref_sql}
            {where_sql}
        ), agg AS (
            SELECT cid10_original,
                   cid10_adequado_grupo,
                   cid10_adequado_classificacao,
                   status_conversao,
                   categoria_grafico,
                   COUNT(*) AS n,
                   COUNT(DISTINCT cid10_detectado) AS cids_distintos,
                   string_agg(DISTINCT cid10_detectado, ', ' ORDER BY cid10_detectado)
                       FILTER (WHERE cid10_detectado IS NOT NULL) AS cids_detectados,
                   string_agg(DISTINCT cid10_original_classificacao, '; ' ORDER BY cid10_original_classificacao)
                       FILTER (WHERE cid10_original_classificacao IS NOT NULL) AS classificacoes_originais,
                   string_agg(DISTINCT observacao_conversao, '; ' ORDER BY observacao_conversao)
                       FILTER (WHERE observacao_conversao IS NOT NULL) AS observacoes,
                   string_agg(DISTINCT coluna_origem, ', ' ORDER BY coluna_origem)
                       FILTER (WHERE coluna_origem IS NOT NULL) AS campos_origem
            FROM base
            WHERE cid10_detectado IS NOT NULL
            GROUP BY 1, 2, 3, 4, 5
        ), with_totals AS (
            SELECT *, SUM(n) OVER () AS denominador
            FROM agg
        )
        SELECT *,
               CASE WHEN denominador > 0 THEN ROUND(100.0 * n / denominador, 2) ELSE NULL END AS pct
        FROM with_totals
        ORDER BY CASE WHEN status_conversao = 'Convertido' THEN 0 ELSE 1 END,
                 n DESC, cid10_original, cid10_adequado_grupo
    """
    return run_query(table, sql)

def _join_unique_text(values: pd.Series, sep: str = ", ") -> Optional[str]:
    """Une valores textuais únicos preservando a ordem de aparecimento."""
    seen: List[str] = []
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.append(text)
    return sep.join(seen) if seen else None


def _format_br_int(value: object) -> str:
    if pd.isna(value):
        return "—"
    try:
        return f"{int(round(float(value))):,}".replace(",", ".")
    except Exception:
        return str(value)


def _format_br_pct(value: object) -> str:
    if pd.isna(value):
        return "—"
    try:
        return f"{float(value):.1f}%".replace(".", ",")
    except Exception:
        return str(value)


def build_cid10_adequacy_conversion_note(df: pd.DataFrame) -> str:
    """Resume, em texto curto, somente o que foi efetivamente convertido."""
    required = {"status_conversao", "cid10_original", "cid10_adequado_grupo", "n", "denominador"}
    if df.empty or not required.issubset(df.columns):
        return "Conversão efetiva no recorte: não foi possível calcular o resumo de conversões."

    denom = pd.to_numeric(df["denominador"], errors="coerce").max()
    if pd.isna(denom) or denom <= 0:
        denom = pd.to_numeric(df["n"], errors="coerce").sum()

    converted = df[df["status_conversao"].eq("Convertido")].copy()
    if converted.empty:
        return (
            "Conversão efetiva no recorte: nenhum registro foi convertido; "
            f"o gráfico mostra somente CID-10 prefixados já presentes no SIM/CIHA "
            f"({_format_br_int(denom)} registros com CID-10 detectado)."
        )

    converted["n"] = pd.to_numeric(converted["n"], errors="coerce").fillna(0)
    total_converted = converted["n"].sum()
    pct_converted = (100.0 * total_converted / denom) if denom and denom > 0 else np.nan

    agg_kwargs = {"n": ("n", "sum")}
    if "cids_detectados" in converted.columns:
        agg_kwargs["cids_detectados"] = ("cids_detectados", lambda s: _join_unique_text(s, ", "))

    detail = (
        converted
        .groupby(["cid10_original", "cid10_adequado_grupo"], dropna=False, as_index=False)
        .agg(**agg_kwargs)
        .sort_values(["cid10_adequado_grupo", "n", "cid10_original"], ascending=[True, False, True])
    )

    parts: List[str] = []
    for _, row in detail.iterrows():
        origem = row.get("cid10_original")
        destino = row.get("cid10_adequado_grupo")
        origem = "CID original não identificado" if pd.isna(origem) or not str(origem).strip() else str(origem)
        destino = "destino não identificado" if pd.isna(destino) or not str(destino).strip() else str(destino)
        piece = f"{origem} → {destino}: {_format_br_int(row.get('n'))}"
        cids = row.get("cids_detectados") if "cids_detectados" in detail.columns else None
        if isinstance(cids, str) and cids.strip():
            piece += f" (CID detectado: {cids})"
        parts.append(piece)

    return (
        f"Conversão efetiva no recorte: {_format_br_int(total_converted)} registros "
        f"({_format_br_pct(pct_converted)} do total com CID-10 detectado) foram convertidos. "
        f"Detalhe: {'; '.join(parts)}."
    )


def summarize_cid10_adequacy_plot(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega o gráfico no nível do CID-10 adequado final.

    O gráfico soma os códigos efetivamente convertidos no destino adequado e
    também mantém os CID-10 prefixados já presentes no SIM/CIHA. A tabela
    detalhada continua separando códigos originais e status de conversão.
    """
    required = {
        "categoria_grafico",
        "cid10_adequado_grupo",
        "cid10_adequado_classificacao",
        "n",
        "denominador",
    }
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame()

    plot_df = df[
        df["categoria_grafico"].notna()
        & df["cid10_adequado_grupo"].notna()
    ].copy()
    if plot_df.empty:
        return pd.DataFrame()

    agg = (
        plot_df
        .groupby(["categoria_grafico", "cid10_adequado_grupo", "cid10_adequado_classificacao"], dropna=False, as_index=False)
        .agg(
            n=("n", "sum"),
            denominador=("denominador", "max"),
            status_conversao=("status_conversao", lambda s: _join_unique_text(s, ", ") if "status_conversao" in plot_df.columns else None),
            cid10_originais=("cid10_original", lambda s: _join_unique_text(s, ", ") if "cid10_original" in plot_df.columns else None),
            cids_detectados=("cids_detectados", lambda s: _join_unique_text(s, ", ") if "cids_detectados" in plot_df.columns else None),
            classificacoes_originais=("classificacoes_originais", lambda s: _join_unique_text(s, "; ") if "classificacoes_originais" in plot_df.columns else None),
            observacoes=("observacoes", lambda s: _join_unique_text(s, "; ") if "observacoes" in plot_df.columns else None),
            campos_origem=("campos_origem", lambda s: _join_unique_text(s, ", ") if "campos_origem" in plot_df.columns else None),
        )
    )
    denom = agg["denominador"].replace({0: np.nan})
    agg["pct"] = (100.0 * agg["n"] / denom).round(2)
    return agg.sort_values(["n", "categoria_grafico"], ascending=[False, True]).reset_index(drop=True)


def query_sinan_cid10_conversion(table: LoadedTable, exprs: Dict[str, Optional[str]], where_sql: str) -> pd.DataFrame:
    con_code = exprs.get("con_code")
    if not con_code:
        return pd.DataFrame()
    con_label = exprs.get("con_label") or "NULL"
    con_group = exprs.get("con_group") or "NULL"
    bac_label = exprs.get("cla_me_bac_label") or "NULL"
    ass_label = exprs.get("cla_me_ass_label") or "NULL"
    eti_label = exprs.get("cla_me_eti_label") or "NULL"
    cid_group = exprs.get("sinan_cid10_conversion_group") or sinan_cid10_conversion_group_expr(con_code)
    cid_type = exprs.get("sinan_cid10_conversion_type") or sinan_cid10_conversion_type_expr(con_code)
    cid_reason = exprs.get("sinan_cid10_conversion_reason") or "NULL"
    include = exprs.get("sinan_cid10_conversion_include") or sinan_cid10_conversion_include_expr(con_code)
    g01_base = exprs.get("sinan_g01_base_disease") or "NULL"
    sql = f"""
        WITH base AS (
            SELECT {con_code} AS con_code,
                   {con_label} AS conclusao_diagnostica,
                   {con_group} AS grupo_etiologico_sinan,
                   {bac_label} AS bacteria_sinan,
                   {ass_label} AS agente_asseptica_sinan,
                   {eti_label} AS outra_etiologia_sinan,
                   {cid_group} AS cid10_grupo,
                   {cid_type} AS cid10_classificacao,
                   {cid_reason} AS justificativa_cid10,
                   {g01_base} AS doenca_base_g01_provavel,
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
                       FILTER (WHERE grupo_etiologico_sinan IS NOT NULL) AS grupos_sinan,
                   string_agg(DISTINCT bacteria_sinan, '; ' ORDER BY bacteria_sinan)
                       FILTER (WHERE bacteria_sinan IS NOT NULL) AS bacterias_sinan,
                   string_agg(DISTINCT agente_asseptica_sinan, '; ' ORDER BY agente_asseptica_sinan)
                       FILTER (WHERE agente_asseptica_sinan IS NOT NULL) AS agentes_asseptica_sinan,
                   string_agg(DISTINCT outra_etiologia_sinan, '; ' ORDER BY outra_etiologia_sinan)
                       FILTER (WHERE outra_etiologia_sinan IS NOT NULL) AS outras_etiologias_sinan,
                   string_agg(DISTINCT doenca_base_g01_provavel, '; ' ORDER BY doenca_base_g01_provavel)
                       FILTER (WHERE cid10_grupo = 'G01' AND doenca_base_g01_provavel IS NOT NULL) AS doencas_base_g01_provaveis,
                   string_agg(DISTINCT justificativa_cid10, '; ' ORDER BY justificativa_cid10)
                       FILTER (WHERE justificativa_cid10 IS NOT NULL) AS justificativas
            FROM base
            GROUP BY 1, 2, 3
        ), with_totals AS (
            SELECT *,
                   SUM(n) OVER () AS denominador
            FROM agg
        )
        SELECT *,
               CASE WHEN denominador > 0
                    THEN ROUND(100.0 * n / denominador, 2)
                    ELSE NULL END AS pct
        FROM with_totals
        ORDER BY CASE WHEN incluido_comparacao = 'Sim' THEN 0 ELSE 1 END,
                 n DESC, cid10_grupo
    """
    return run_query(table, sql)



def query_sinan_g01_base_disease(table: LoadedTable, exprs: Dict[str, Optional[str]], where_sql: str) -> pd.DataFrame:
    con_code = exprs.get("con_code")
    cid_group = exprs.get("sinan_cid10_conversion_group")
    include = exprs.get("sinan_cid10_conversion_include")
    base_disease = exprs.get("sinan_g01_base_disease")
    con_label = exprs.get("con_label") or "NULL"
    bacteria = exprs.get("cla_me_bac_label") or "NULL"
    reason = exprs.get("sinan_cid10_conversion_reason") or "NULL"
    if not (con_code and cid_group and include and base_disease):
        return pd.DataFrame()
    sql = f"""
        WITH base AS (
            SELECT {cid_group} AS cid10_grupo,
                   {include} AS incluido_comparacao,
                   {base_disease} AS doenca_base_provavel,
                   {con_label} AS conclusao_sinan,
                   {bacteria} AS bacteria_sinan,
                   {reason} AS justificativa
            FROM {table.ref_sql}
            {where_sql}
        ), agg AS (
            SELECT doenca_base_provavel,
                   COUNT(*) AS n,
                   string_agg(DISTINCT conclusao_sinan, '; ' ORDER BY conclusao_sinan)
                       FILTER (WHERE conclusao_sinan IS NOT NULL) AS conclusoes_sinan,
                   string_agg(DISTINCT bacteria_sinan, '; ' ORDER BY bacteria_sinan)
                       FILTER (WHERE bacteria_sinan IS NOT NULL) AS bacterias_sinan,
                   string_agg(DISTINCT justificativa, '; ' ORDER BY justificativa)
                       FILTER (WHERE justificativa IS NOT NULL) AS justificativas
            FROM base
            WHERE cid10_grupo = 'G01' AND incluido_comparacao = 'Sim'
            GROUP BY 1
        ), with_totals AS (
            SELECT *, SUM(n) OVER () AS denominador
            FROM agg
        )
        SELECT *,
               CASE WHEN denominador > 0 THEN ROUND(100.0 * n / denominador, 2) ELSE NULL END AS pct
        FROM with_totals
        ORDER BY n DESC, doenca_base_provavel
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
            SELECT {qstr(key)} AS parametro_id, {qstr(SINAN_QUIMIO_MATERIAL)} AS material_analisado, {qstr(label)} AS parametro, {value_expr} AS valor
            FROM {table.ref_sql}
            {where_sql}
            """
        )
    sql = f"""
        WITH valores AS (
            {' UNION ALL '.join(unions)}
        )
        SELECT parametro_id,
               material_analisado,
               parametro,
               COUNT(*) AS registros_avaliados,
               COUNT(*) FILTER (WHERE valor IS NOT NULL AND valor >= 0) AS n_valido,
               COUNT(*) FILTER (WHERE valor IS NULL OR valor < 0) AS n_sem_valor,
               ROUND(100.0 * COUNT(*) FILTER (WHERE valor IS NOT NULL AND valor >= 0) / NULLIF(COUNT(*), 0), 2) AS pct_preenchido,
               MIN(valor) FILTER (WHERE valor IS NOT NULL AND valor >= 0) AS minimo,
               quantile_cont(valor, 0.25) FILTER (WHERE valor IS NOT NULL AND valor >= 0) AS q1,
               median(valor) FILTER (WHERE valor IS NOT NULL AND valor >= 0) AS mediana,
               ROUND(AVG(valor) FILTER (WHERE valor IS NOT NULL AND valor >= 0), 2) AS media,
               quantile_cont(valor, 0.75) FILTER (WHERE valor IS NOT NULL AND valor >= 0) AS q3,
               MAX(valor) FILTER (WHERE valor IS NOT NULL AND valor >= 0) AS maximo
        FROM valores
        GROUP BY 1, 2, 3
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
        ), with_totals AS (
            SELECT bin_idx, n, SUM(n) OVER () AS denominador
            FROM agg
        )
        SELECT {minimo!r} + bin_idx * {width!r} AS faixa_inicio,
               CASE WHEN bin_idx = {bin_count - 1} THEN {maximo!r}
                    ELSE {minimo!r} + (bin_idx + 1) * {width!r} END AS faixa_fim,
               n,
               denominador,
               CASE WHEN denominador > 0 THEN ROUND(100.0 * n / denominador, 2) ELSE NULL END AS pct
        FROM with_totals
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
               {pct_expr('obitos_meningite_confirmados', 'confirmados')} AS letalidade_confirmados,
               {pct_expr('obitos_meningite_confirmados', 'confirmados_evolucao_conhecida')} AS letalidade_confirmados_evolucao_conhecida
        FROM agg
        ORDER BY ano
    """
    return run_query(table, sql)



def _available_column_specs(columns: Sequence[str], specs: Sequence[Tuple[str, str]]) -> List[Tuple[str, str]]:
    available: List[Tuple[str, str]] = []
    for default_col, label in specs:
        col = choose_candidate(columns, [default_col])
        if col:
            available.append((col, label))
    return available


def query_sinan_symptom_prevalence(table: LoadedTable, exprs: Dict[str, Optional[str]], where_sql: str, symptom_specs: Sequence[Tuple[str, str]]) -> pd.DataFrame:
    dt = exprs.get("dt")
    classi = exprs.get("classi_code")
    if not (dt and classi and symptom_specs):
        return pd.DataFrame()

    unions = []
    for col, label in symptom_specs:
        unions.append(
            f"""
            SELECT {dt} AS dt,
                   {classi} AS classi,
                   {qstr(label)} AS sintoma,
                   {clean_code_expr(col)} AS sintoma_codigo
            FROM {table.ref_sql}
            {where_sql}
            """
        )

    sql = f"""
        WITH long AS (
            {' UNION ALL '.join(unions)}
        ), base AS (
            SELECT EXTRACT(YEAR FROM dt) AS ano,
                   sintoma,
                   sintoma_codigo
            FROM long
            WHERE dt IS NOT NULL
              AND classi = '1'
        ), agg AS (
            SELECT ano,
                   sintoma,
                   COUNT(*) AS confirmados,
                   COUNT(*) FILTER (WHERE sintoma_codigo = '1') AS sintoma_sim,
                   COUNT(*) FILTER (WHERE sintoma_codigo = '2') AS sintoma_nao,
                   COUNT(*) FILTER (WHERE sintoma_codigo IS NULL OR sintoma_codigo NOT IN ('1','2')) AS sintoma_ignorado
            FROM base
            GROUP BY 1, 2
        )
        SELECT *,
               {pct_expr('sintoma_sim', 'confirmados')} AS pct_sintoma_confirmados
        FROM agg
        ORDER BY ano, sintoma
    """
    return run_query(table, sql)


def query_sinan_hospitalization_internment(
    table: LoadedTable,
    exprs: Dict[str, Optional[str]],
    where_sql: str,
    hospital_col: Optional[str],
    internment_col: Optional[str] = None,
) -> pd.DataFrame:
    dt = exprs.get("dt")
    classi = exprs.get("classi_code")
    if not (dt and classi and hospital_col):
        return pd.DataFrame()

    hospital_code = clean_code_expr(hospital_col)
    sql = f"""
        WITH base AS (
            SELECT EXTRACT(YEAR FROM {dt}) AS ano,
                   {classi} AS classi,
                   {hospital_code} AS hospitalizacao
            FROM {table.ref_sql}
            {where_sql}
        ), scoped AS (
            SELECT ano,
                   'Suspeitos/notificados' AS grupo_caso,
                   1 AS ordem_grupo,
                   COUNT(*) AS denominador,
                   COUNT(*) FILTER (WHERE hospitalizacao = '1') AS n
            FROM base
            WHERE ano IS NOT NULL
            GROUP BY 1
            UNION ALL
            SELECT ano,
                   'Confirmados' AS grupo_caso,
                   2 AS ordem_grupo,
                   COUNT(*) AS denominador,
                   COUNT(*) FILTER (WHERE hospitalizacao = '1') AS n
            FROM base
            WHERE ano IS NOT NULL
              AND classi = '1'
            GROUP BY 1
            UNION ALL
            SELECT ano,
                   'Descartados' AS grupo_caso,
                   3 AS ordem_grupo,
                   COUNT(*) AS denominador,
                   COUNT(*) FILTER (WHERE hospitalizacao = '1') AS n
            FROM base
            WHERE ano IS NOT NULL
              AND classi = '2'
            GROUP BY 1
        )
        SELECT ano,
               grupo_caso,
               'Hospitalização informada (ATE_HOSPIT = Sim)' AS indicador,
               n,
               denominador,
               {pct_expr('n', 'denominador')} AS pct,
               ordem_grupo
        FROM scoped
        ORDER BY ano, ordem_grupo
    """
    return run_query(table, sql)

def query_sinan_communicants_prophylaxis(
    table: LoadedTable,
    exprs: Dict[str, Optional[str]],
    where_sql: str,
    communicants_col: Optional[str],
    prophylaxis_col: Optional[str],
) -> pd.DataFrame:
    dt = exprs.get("dt")
    if not (dt and (communicants_col or prophylaxis_col)):
        return pd.DataFrame()

    communicants = numeric_expr(communicants_col) if communicants_col else "CAST(NULL AS DOUBLE)"
    prophylaxis = case_from_mapping(clean_code_expr(prophylaxis_col), YES_NO_IGN, "Sem informação") if prophylaxis_col else qstr("Sem informação")
    sql = f"""
        WITH base AS (
            SELECT EXTRACT(YEAR FROM {dt}) AS ano,
                   {communicants} AS comunicantes,
                   {prophylaxis} AS quimioprofilaxia
            FROM {table.ref_sql}
            {where_sql}
        ), agg AS (
            SELECT ano,
                   quimioprofilaxia,
                   COUNT(*) AS registros,
                   COUNT(*) FILTER (WHERE comunicantes IS NOT NULL AND comunicantes >= 0) AS registros_com_comunicantes,
                   SUM(CASE WHEN comunicantes IS NOT NULL AND comunicantes >= 0 THEN comunicantes ELSE 0 END) AS comunicantes_total,
                   ROUND(AVG(comunicantes) FILTER (WHERE comunicantes IS NOT NULL AND comunicantes >= 0), 2) AS media_comunicantes
            FROM base
            WHERE ano IS NOT NULL
              AND (comunicantes IS NOT NULL OR quimioprofilaxia <> 'Sem informação')
            GROUP BY 1, 2
        ), with_totals AS (
            SELECT *,
                   SUM(registros) OVER (PARTITION BY ano) AS total_registros_ano,
                   SUM(comunicantes_total) OVER (PARTITION BY ano) AS total_comunicantes_ano
            FROM agg
        )
        SELECT *,
               {pct_expr('registros', 'total_registros_ano')} AS pct_registros_ano,
               {pct_expr('comunicantes_total', 'total_comunicantes_ano')} AS pct_comunicantes_ano
        FROM with_totals
        ORDER BY ano, quimioprofilaxia
    """
    return run_query(table, sql)


def query_sinan_vaccination_by_classification(table: LoadedTable, exprs: Dict[str, Optional[str]], where_sql: str, vaccine_specs: Sequence[Tuple[str, str]]) -> pd.DataFrame:
    classi = exprs.get("classi_code")
    if not (classi and vaccine_specs):
        return pd.DataFrame()

    unions = []
    for col, label in vaccine_specs:
        unions.append(
            f"""
            SELECT {qstr(label)} AS vacina,
                   {classi} AS classi,
                   {clean_code_expr(col)} AS vacina_codigo
            FROM {table.ref_sql}
            {where_sql}
            """
        )

    sql = f"""
        WITH long AS (
            {' UNION ALL '.join(unions)}
        ), base AS (
            SELECT vacina,
                   CASE
                       WHEN classi = '1' THEN 'Confirmados'
                       WHEN classi IN ('2','8') THEN 'Descartados / inconclusivos'
                       ELSE NULL
                   END AS grupo_classificacao,
                   vacina_codigo
            FROM long
            WHERE classi IN ('1','2','8')
        ), agg AS (
            SELECT vacina,
                   grupo_classificacao,
                   COUNT(*) AS denominador,
                   COUNT(*) FILTER (WHERE vacina_codigo = '1') AS vacinados_sim,
                   COUNT(*) FILTER (WHERE vacina_codigo = '2') AS vacinados_nao,
                   COUNT(*) FILTER (WHERE vacina_codigo IS NULL OR vacina_codigo NOT IN ('1','2')) AS vacinacao_ignorada
            FROM base
            WHERE grupo_classificacao IS NOT NULL
            GROUP BY 1, 2
        )
        SELECT *,
               {pct_expr('vacinados_sim', 'denominador')} AS pct_vacinados_sim
        FROM agg
        ORDER BY vacina,
                 CASE WHEN grupo_classificacao = 'Confirmados' THEN 0 ELSE 1 END
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
               {pct_expr('obitos_meningite', 'confirmados')} AS letalidade_pct,
               {pct_expr('obitos_meningite', 'confirmados_evolucao_conhecida')} AS letalidade_evolucao_conhecida_pct
        FROM agg
        WHERE confirmados > 0
        ORDER BY confirmados DESC, grupo_etiologico
    """
    return run_query(table, sql)


def query_sinan_diagnostics_by_year(table: LoadedTable, exprs: Dict[str, Optional[str]], where_sql: str) -> pd.DataFrame:
    dt = exprs.get("dt")
    classi = exprs.get("classi_code")
    cid_group = exprs.get("sinan_cid10_conversion_group")
    cid_type = exprs.get("sinan_cid10_conversion_type")
    include = exprs.get("sinan_cid10_conversion_include")
    if not (dt and classi and cid_group and cid_type and include):
        return pd.DataFrame()
    ordem = """
        CASE cid10_grupo
            WHEN 'A17.0' THEN 1
            WHEN 'A39.0' THEN 2
            WHEN 'A87' THEN 3
            WHEN 'G00' THEN 4
            WHEN 'G01' THEN 5
            WHEN 'G02' THEN 6
            WHEN 'G03' THEN 7
            WHEN 'G04' THEN 8
            WHEN 'G05' THEN 9
            ELSE 99
        END
    """
    sql = f"""
        WITH base AS (
            SELECT CAST(EXTRACT(YEAR FROM {dt}) AS INTEGER) AS ano,
                   {classi} AS classi,
                   {cid_group} AS cid10_grupo,
                   {cid_type} AS grupo_etiologico
            FROM {table.ref_sql}
            {append_clause(where_sql, f"{classi} = '1' AND {include} = 'Sim'")}
        ), agg AS (
            SELECT ano,
                   cid10_grupo,
                   grupo_etiologico,
                   COUNT(*) AS confirmados
            FROM base
            WHERE ano IS NOT NULL
              AND cid10_grupo IS NOT NULL
              AND grupo_etiologico IS NOT NULL
            GROUP BY 1, 2, 3
        ), totais AS (
            SELECT ano, SUM(confirmados) AS total_ano
            FROM agg
            GROUP BY 1
        )
        SELECT agg.ano,
               agg.cid10_grupo,
               agg.grupo_etiologico,
               agg.confirmados,
               totais.total_ano,
               CASE WHEN totais.total_ano > 0 THEN ROUND(100.0 * agg.confirmados / totais.total_ano, 2) ELSE NULL END AS pct_ano
        FROM agg
        JOIN totais USING (ano)
        WHERE agg.confirmados > 0
        ORDER BY agg.ano, {ordem}, agg.grupo_etiologico
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
               {pct_expr('atendimentos_diag_principal_meningite', 'atendimentos')} AS pct_atendimentos_diag_principal_meningite,
               {pct_expr('atendimentos_qualquer_cid_meningite', 'atendimentos')} AS pct_atendimentos_qualquer_cid_meningite,
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
        ), with_totals AS (
            SELECT faixa_dias_perm, ordem, n, SUM(n) OVER () AS denominador
            FROM counts
        )
        SELECT faixa_dias_perm, ordem, n, denominador,
               CASE WHEN denominador > 0 THEN ROUND(100.0 * n / denominador, 2) ELSE NULL END AS pct
        FROM with_totals
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


def query_enriched_preview(table: LoadedTable, sel: ColumnSelection, exprs: Dict[str, Optional[str]], where_sql: str, limit: Optional[int] = 200, offset: int = 0) -> pd.DataFrame:
    items = []
    mapping = [
        ("data_analise", exprs.get("dt")),
        ("sexo", exprs.get("sex")),
        ("idade_anos", exprs.get("age")),
        ("raca_cor", exprs.get("race")),
        ("municipio_residencia", exprs.get("mun_res_label") or exprs.get("mun_res")),
        ("municipio_evento_atendimento", exprs.get("mun_event_label") or exprs.get("mun_event")),
        ("cid_meningite_encefalite_detectado", exprs.get("cid")),
        ("tipo_cid10", exprs.get("cid_type")),
        ("cid10_adequado_grupo", exprs.get("cid10_adequacy_group")),
        ("cid10_adequado_tipo", exprs.get("cid10_adequacy_type")),
        ("cid10_status_conversao", exprs.get("cid10_adequacy_status")),
        ("cid10_observacao_conversao", exprs.get("cid10_adequacy_reason")),
        ("campo_origem_cid", exprs.get("cid_source")),
        ("sinan_classificacao_final", exprs.get("classi_label")),
        ("sinan_conclusao_diagnostica", exprs.get("con_label")),
        ("sinan_grupo_etiologico", exprs.get("con_group")),
        ("sinan_cla_me_bac", exprs.get("cla_me_bac_label")),
        ("sinan_cla_me_ass", exprs.get("cla_me_ass_label")),
        ("sinan_cla_me_eti", exprs.get("cla_me_eti_label")),
        ("sinan_cid10_convertido_grupo", exprs.get("sinan_cid10_conversion_group")),
        ("sinan_cid10_convertido_tipo", exprs.get("sinan_cid10_conversion_type")),
        ("sinan_cid10_justificativa", exprs.get("sinan_cid10_conversion_reason")),
        ("sinan_cid10_inclui_comparacao", exprs.get("sinan_cid10_conversion_include")),
        ("sinan_g01_doenca_base_provavel", exprs.get("sinan_g01_base_disease")),
        ("sinan_evolucao", exprs.get("evol_label")),
        ("sinan_criterio", exprs.get("criterio_label")),
        ("sinan_puncao_laboratorial", exprs.get("puncao_label")),
        ("sinan_exame_quimiocitologico_liquor_lcr", exprs.get("quimio_label")),
        ("sinan_lab_hemacias", exprs.get("lab_hema")),
        ("sinan_lab_neutrofilos", exprs.get("lab_neutro")),
        ("sinan_lab_glicose", exprs.get("lab_glico")),
        ("sinan_lab_leucocitos", exprs.get("lab_leuco")),
        ("sinan_lab_eosinofilos", exprs.get("lab_eosi")),
        ("sinan_lab_proteinas", exprs.get("lab_prot")),
        ("sinan_lab_monocitos", exprs.get("lab_mono")),
        ("sinan_lab_linfocitos", exprs.get("lab_linfo")),
        ("sinan_lab_cloreto", exprs.get("lab_clor")),
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
        sel.cla_me_bac_col,
        sel.cla_me_ass_col,
        sel.cla_me_eti_col,
        *(sel.sinan_auxiliary_cid10_cols or []),
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
        sel.modalidade_col,
        sel.procedimento_col,
    ]:
        if col and col not in raw_cols:
            raw_cols.append(col)
    for col in raw_cols:
        items.append(f"{qident(col)} AS {qident('raw_' + col[:45])}")
    if not items:
        items = ["*"]
    limit_sql = "" if limit is None else f" LIMIT {int(limit)} OFFSET {int(max(offset, 0))}"
    sql = f"SELECT {', '.join(items)} FROM {table.ref_sql} {where_sql}{limit_sql}"
    return run_query(table, sql, cache=False)


# =============================================================================
# Visualização e UI
# =============================================================================


def download_button(df: pd.DataFrame, filename: str, label: str = "Baixar CSV", max_rows: Optional[int] = None) -> None:
    if df is None or df.empty:
        return
    row_limit = perf_int("perf_download_row_limit", DEFAULT_DOWNLOAD_ROW_LIMIT) if max_rows is None else int(max_rows)
    out = df
    if row_limit > 0 and len(df) > row_limit:
        out = df.head(row_limit).copy()
        st.caption(
            f"Download limitado às primeiras {row_limit:,} linhas de {len(df):,} para evitar excesso de memória."
            .replace(",", ".")
        )
    st.download_button(
        label=label,
        data=out.to_csv(index=False).encode("utf-8-sig"),
        file_name=filename,
        mime="text/csv",
        width="content",
    )


def _copyable_table_payload(df: pd.DataFrame) -> Tuple[str, str]:
    """Gera versões HTML e TSV para colagem em Google Docs/editores."""
    if df is None or df.empty:
        return "", ""
    out = df.copy()
    out = out.where(pd.notna(out), "")
    html_table = out.to_html(index=False, escape=True, border=1)
    tsv_table = out.to_csv(index=False, sep="\t", lineterminator="\n")
    return html_table, tsv_table


def copy_table_button(df: pd.DataFrame, label: str = "Copiar tabela para Google Docs/editores") -> None:
    """Renderiza botão de cópia com HTML de tabela e fallback em TSV."""
    if df is None or df.empty:
        return
    html_table, tsv_table = _copyable_table_payload(df)
    if not html_table:
        return
    uid = hashlib.sha1((html_table[:4000] + str(df.shape)).encode("utf-8", errors="ignore")).hexdigest()[:12]
    button_id = f"copy-table-{uid}"
    status_id = f"copy-status-{uid}"
    html_component = f"""
    <div style="font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;">
      <button id="{button_id}"
              style="border: 1px solid #d0d7de; border-radius: 8px; background: #f6f8fa; padding: 6px 10px; cursor: pointer; font-size: 0.9rem;">
        {html_lib.escape(label)}
      </button>
      <span id="{status_id}" style="margin-left: 8px; color: #57606a; font-size: 0.85rem;"></span>
    </div>
    <script>
    const htmlTable_{uid} = {json.dumps(html_table, ensure_ascii=False)};
    const plainText_{uid} = {json.dumps(tsv_table, ensure_ascii=False)};
    const button_{uid} = document.getElementById({json.dumps(button_id)});
    const status_{uid} = document.getElementById({json.dumps(status_id)});

    async function copyRichTable_{uid}() {{
      try {{
        if (navigator.clipboard && window.ClipboardItem) {{
          const item = new ClipboardItem({{
            'text/html': new Blob([htmlTable_{uid}], {{type: 'text/html'}}),
            'text/plain': new Blob([plainText_{uid}], {{type: 'text/plain'}})
          }});
          await navigator.clipboard.write([item]);
        }} else if (navigator.clipboard) {{
          await navigator.clipboard.writeText(plainText_{uid});
        }} else {{
          const textarea = document.createElement('textarea');
          textarea.value = plainText_{uid};
          textarea.style.position = 'fixed';
          textarea.style.left = '-9999px';
          document.body.appendChild(textarea);
          textarea.focus();
          textarea.select();
          document.execCommand('copy');
          textarea.remove();
        }}
        status_{uid}.textContent = 'Tabela copiada.';
      }} catch (err) {{
        const textarea = document.createElement('textarea');
        textarea.value = plainText_{uid};
        textarea.style.position = 'fixed';
        textarea.style.left = '-9999px';
        document.body.appendChild(textarea);
        textarea.focus();
        textarea.select();
        const ok = document.execCommand('copy');
        textarea.remove();
        status_{uid}.textContent = ok ? 'Tabela copiada em texto tabulado.' : 'Copie manualmente pela tabela acima.';
      }}
    }}
    button_{uid}.addEventListener('click', copyRichTable_{uid});
    </script>
    """
    st.iframe(html_component, height=42, width="stretch")


def copyable_dataframe(df: pd.DataFrame, *args, **kwargs) -> None:
    if df is None:
        return

    display_limit = perf_int("perf_display_row_limit", DEFAULT_DISPLAY_ROW_LIMIT)
    copy_limit = perf_int("perf_copy_row_limit", DEFAULT_COPY_ROW_LIMIT)

    display_df = df
    if display_limit > 0 and len(df) > display_limit:
        display_df = df.head(display_limit).copy()
        st.caption(
            f"Tabela renderizada com {display_limit:,} de {len(df):,} linhas. "
            "Use filtros, paginação ou download para volumes maiores."
            .replace(",", ".")
        )

    st.dataframe(display_df, *args, **kwargs)

    copy_df = df
    if copy_limit > 0 and len(df) > copy_limit:
        copy_df = df.head(copy_limit).copy()
        st.caption(
            f"Botão de cópia limitado às primeiras {copy_limit:,} linhas para não sobrecarregar o navegador."
            .replace(",", ".")
        )
    copy_table_button(copy_df)




def format_int_br(value: object) -> str:
    """Formata inteiros em padrão brasileiro para captions de gráficos."""
    if pd.isna(value):
        return "—"
    try:
        return f"{int(round(float(value))):,}".replace(",", ".")
    except Exception:
        return str(value)


def format_pct_br(value: object) -> str:
    """Formata percentuais em padrão brasileiro para captions de gráficos."""
    if pd.isna(value):
        return "—"
    try:
        return f"{float(value):.1f}%".replace(".", ",")
    except Exception:
        return str(value)


def render_interval_total(
    df: pd.DataFrame,
    value_col: str = "n",
    by_col: Optional[str] = None,
    denominator_col: Optional[str] = None,
    value_label: str = "registros",
    denominator_label: str = "denominador",
    prefix: str = "Somatória no intervalo filtrado",
    max_items: int = 8,
) -> None:
    """Exibe a somatória representada pelo gráfico no recorte de tempo/filtros atual.

    O recorte é o que chegou ao dataframe do gráfico: filtros-base, definição exploratória
    quando aplicável e os anos/parquets efetivamente carregados pelo usuário.
    """
    if df is None or df.empty or value_col not in df.columns:
        return
    tmp = df.copy()
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce").fillna(0)

    def build_piece(label: object, part: pd.DataFrame) -> str:
        total = part[value_col].sum()
        piece = f"{label}: {format_int_br(total)} {value_label}"
        if denominator_col and denominator_col in part.columns:
            denom = pd.to_numeric(part[denominator_col], errors="coerce").fillna(0).sum()
            if denom > 0:
                pct = 100.0 * total / denom
                piece += f" de {format_int_br(denom)} {denominator_label} ({format_pct_br(pct)})"
        return piece

    if by_col and by_col in tmp.columns:
        grouped = (
            tmp.groupby(by_col, dropna=False, as_index=False)[value_col]
            .sum()
            .sort_values(value_col, ascending=False)
        )
        labels = grouped[by_col].tolist()
        pieces = []
        for label in labels[:max_items]:
            part = tmp[tmp[by_col].isna()] if pd.isna(label) else tmp[tmp[by_col].eq(label)]
            pieces.append(build_piece(label if pd.notna(label) else "Sem informação", part))
        if len(labels) > max_items:
            pieces.append(f"+{len(labels) - max_items} categorias")
        st.caption(f"{prefix}: " + "; ".join(pieces) + ".")
        return

    total = tmp[value_col].sum()
    text = f"{prefix}: {format_int_br(total)} {value_label}"
    if denominator_col and denominator_col in tmp.columns:
        denom = pd.to_numeric(tmp[denominator_col], errors="coerce").fillna(0).sum()
        if denom > 0:
            pct = 100.0 * total / denom
            text += f" de {format_int_br(denom)} {denominator_label} ({format_pct_br(pct)})"
    st.caption(text + ".")

def render_field_guide(source: str) -> None:
    copyable_dataframe(
        pd.DataFrame(FIELD_GUIDE[source], columns=["Campo", "Uso", "Leitura epidemiológica"]),
        width="stretch",
        hide_index=True,
    )
    for note in SOURCE_CONFIG[source].field_notes:
        st.caption("• " + note)


def render_cid_reference() -> None:
    copyable_dataframe(pd.DataFrame(CID_RULES)[["grupo", "padrao", "rotulo"]], width="stretch", hide_index=True)
    st.caption(
        "O app procura os padrões CID-10 listados acima nos campos de diagnóstico/causa. "
        "G04* e G05* são tratados como prefixos. A22.8, A32.1, A83*, A84*, A85*, A86*, B00.3, B00.4, B01.0, B01.1, B02.0, B02.1, B05.0, B05.1, B06*, B26.1, B26.2, B37.5, B38.4, B45.1, B57.4, B58.2 e B60.2 foram adicionados ao recorte de meningite/encefalite/meningoencefalite. "
        "No SINAN, a etiologia específica continua derivada de CON_DIAGES e campos complementares; CON_DIAGES=05 não é convertido para G04.2."
    )


def render_quimio_interpretation() -> None:
    st.markdown("### Interpretação usual do exame quimiocitológico do líquor (LCR)")
    st.caption(
        "Padrões de LCR ajudam a levantar hipóteses, mas não substituem cultura, PCR, Gram, tinta nanquim, sorologia, "
        "epidemiologia e avaliação clínica. Os limites variam com idade, coleta traumática, antibiótico prévio e laboratório."
    )
    copyable_dataframe(pd.DataFrame(SINAN_QUIMIO_INTERPRETATION_ROWS), width="stretch", hide_index=True)
    st.markdown("#### Referências bibliográficas usadas para esta síntese")
    copyable_dataframe(pd.DataFrame(SINAN_QUIMIO_REFERENCES), width="stretch", hide_index=True)


def render_loader(source: str) -> Optional[LoadedTable]:
    cfg = SOURCE_CONFIG[source]
    st.markdown(f"### {source} — {cfg.title}")
    st.caption(f"Período esperado no arquivo enviado: {cfg.expected_period}")

    load_modes = [GITHUB_HOSTED_PARQUETS_LABEL, "Upload DuckDB", "Upload Parquet"]
    load_mode_key = f"load_mode_{source}"
    if st.session_state.get(load_mode_key) not in (None, *load_modes):
        st.session_state.pop(load_mode_key, None)
    mode = st.radio(
        "Fonte de dados",
        load_modes,
        horizontal=True,
        key=load_mode_key,
    )

    if mode == GITHUB_HOSTED_PARQUETS_LABEL:
        st.caption(f"Fonte padrão: {GITHUB_RELEASE_PAGE_URL}")
        st.info(
            "Nenhum banco hospedado no github é carregado automaticamente. "
            "Marque manualmente os Parquets desejados e clique em **Carregar/atualizar seleção**."
        )
        try:
            release_assets = list_github_release_parquets()
        except Exception as exc:
            st.error(f"Não consegui listar os assets da release do GitHub: {exc}")
            return None

        source_assets = [asset for asset in release_assets if asset.get("source") == source]
        if not source_assets:
            st.error(f"Não encontrei Parquets da base {source} nos bancos hospedados no github.")
            return None

        max_files = perf_int("perf_max_parquet_files", DEFAULT_MAX_PARQUET_FILES_PER_LOAD)
        visible_assets = source_assets

        label_to_asset = {github_asset_label(asset): asset for asset in visible_assets}
        name_to_asset = {str(asset.get("name") or ""): asset for asset in source_assets}
        labels = list(label_to_asset.keys())
        selected_labels = st.multiselect(
            "Escolha manualmente os Parquets da release para carregar",
            options=labels,
            default=[],
            key=f"github_release_assets_{source}",
            help=f"Nada é pré-selecionado. Limite defensivo atual: {max_files} arquivo(s) por carregamento.",
        )
        selected_names = [str(label_to_asset[label].get("name") or "") for label in selected_labels]
        loaded_key = f"github_release_loaded_asset_names_{source}"

        c_load, c_clear = st.columns([2, 1])
        with c_load:
            load_clicked = st.button(
                "Carregar/atualizar seleção",
                key=f"github_release_load_selected_{source}",
                type="primary",
                disabled=not selected_names,
                width="stretch",
            )
        with c_clear:
            clear_clicked = st.button(
                "Descarregar base",
                key=f"github_release_clear_selected_{source}",
                disabled=not st.session_state.get(loaded_key),
                width="stretch",
            )

        if load_clicked:
            if len(selected_names) > max_files:
                st.error(
                    f"Seleção bloqueada: {len(selected_names)} Parquets excedem o limite atual de {max_files}. "
                    "Reduza os anos/arquivos ou aumente o limite em Desempenho e memória."
                )
            else:
                st.session_state[loaded_key] = selected_names
        if clear_clicked:
            st.session_state.pop(loaded_key, None)

        loaded_names = list(st.session_state.get(loaded_key, []))
        if not loaded_names:
            st.info("Selecione um ou mais Parquets e clique em **Carregar/atualizar seleção** para iniciar a análise.")
            return None
        if len(loaded_names) > max_files:
            st.error(
                f"A seleção carregada contém {len(loaded_names)} Parquets, acima do limite atual de {max_files}. "
                "Clique em **Descarregar base**, reduza a seleção ou aumente o limite em Desempenho e memória."
            )
            return None

        selected_assets = [name_to_asset[name] for name in loaded_names if name in name_to_asset]
        missing_assets = [name for name in loaded_names if name not in name_to_asset]
        if missing_assets:
            st.warning(
                "Alguns arquivos carregados anteriormente não aparecem mais na release e foram ignorados: "
                + ", ".join(missing_assets)
            )
        if not selected_assets:
            st.info("A seleção carregada não contém Parquets válidos. Escolha novamente os arquivos da release.")
            return None

        if set(selected_names) != set(loaded_names):
            st.warning(
                "A lista marcada na tela é diferente da seleção atualmente carregada. "
                "Clique em **Carregar/atualizar seleção** para aplicar a nova escolha."
            )

        with st.expander("Parquets atualmente carregados", expanded=False):
            st.write("\n".join(f"- {name}" for name in loaded_names))

        try:
            with st.spinner(f"Preparando {len(selected_assets)} parquet(s) selecionado(s) dos bancos hospedados no github..."):
                paths = [materialize_github_release_asset(asset) for asset in selected_assets]
        except Exception as exc:
            st.error(f"Não consegui baixar os Parquets selecionados dos bancos hospedados no github: {exc}")
            st.info("Como alternativa, use Upload Parquet.")
            return None

        st.success(f"{github_selection_summary(selected_assets)} carregado(s) dos bancos hospedados no github.")
        return LoadedTable(
            source=source,
            kind="parquet",
            parquet_paths=paths,
            ref_sql=parquet_object_name(source, paths),
            label=f"Bancos hospedados no github: {github_selection_summary(selected_assets)}",
        )

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

    uploads = st.file_uploader("Envie um ou mais Parquets", type=["parquet"], accept_multiple_files=True, key=f"upload_parquet_{source}")
    if not uploads:
        st.info("Envie Parquet(s) para continuar.")
        return None
    max_files = perf_int("perf_max_parquet_files", DEFAULT_MAX_PARQUET_FILES_PER_LOAD)
    if len(uploads) > max_files:
        st.error(
            f"Foram enviados {len(uploads)} Parquets, acima do limite atual de {max_files}. "
            "Reduza a seleção ou aumente o limite em Desempenho e memória."
        )
        return None
    paths = [materialize_upload(up, f"{source.lower()}_parquet") for up in uploads]
    return LoadedTable(source=source, kind="parquet", parquet_paths=paths, ref_sql=parquet_object_name(source, paths), label=f"{len(paths)} parquet(s) enviados")


def render_column_config(source: str, columns: Sequence[str]) -> ColumnSelection:
    """Detecta automaticamente as colunas esperadas para cada base.

    A interface manual de seleção de colunas foi omitida para manter o painel mais limpo.
    Quando uma coluna não é encontrada, as abas correspondentes exibem avisos operacionais.
    """
    return default_selections(source, columns)


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
        if source == "SINAN":
            sinan_cid_type = exprs.get("sinan_cid10_conversion_type")
            sinan_include = exprs.get("sinan_cid10_conversion_include")
            if sinan_cid_type:
                with c5:
                    opt_where = append_clause("", f"{sinan_include} = 'Sim'") if sinan_include else ""
                    cid_opts = top_values(table, sinan_cid_type, opt_where, limit=20)
                    selected_cid = st.multiselect(
                        "CID-10 convertido (SINAN)",
                        cid_opts,
                        default=[],
                        key=f"sinan_cid10_convertido_filter_{source}",
                    )
                    st.caption("Filtro baseado em CON_DIAGES/CLA_ME_BAC/campos complementares, não no ID_AGRAVO bruto.")
                if selected_cid:
                    clauses.append(f"{sinan_cid_type} IN ({', '.join(qstr(x) for x in selected_cid)})")
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
        else:
            cid_type = exprs.get("cid10_adequacy_type") or exprs.get("cid_type")
            if cid_type:
                with c5:
                    cid_opts = top_values(table, cid_type, limit=25)
                    selected_cid = st.multiselect("CID-10 adequado/conversão", cid_opts, default=[], key=f"cidtype_filter_{source}")
                    st.caption("Filtro baseado na conversão de adequação quando aplicável; os CID-10 fora da conversão permanecem como categoria original.")
                if selected_cid:
                    clauses.append(f"{cid_type} IN ({', '.join(qstr(x) for x in selected_cid)})")
            if source == "CIHA" and exprs.get("modalidade_label"):
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
        k5.metric("Tipo CID-10", "detectado" if exprs.get("cid") else "não detectado")


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
    if source == "SINAN":
        if exprs.get("sinan_cid10_conversion_type"):
            cat_options["CID-10 convertido SINAN"] = exprs["sinan_cid10_conversion_type"]
        if exprs.get("con_group"):
            cat_options["Grupo etiológico SINAN"] = exprs["con_group"]
        if exprs.get("classi_label"):
            cat_options["CLASSI_FIN"] = exprs["classi_label"]
    elif exprs.get("cid_type"):
        cat_options["Tipo CID-10"] = exprs["cid_type"]
    if exprs.get("sex"):
        cat_options["Sexo"] = exprs["sex"]
    with c2:
        cat_label = st.selectbox("Estratificar por", list(cat_options.keys()), key=f"ts_cat_{source}")
    ts = query_timeseries(table, dt, graph_where, freq, cat_options[cat_label])
    if ts.empty:
        st.info("Sem dados para a série temporal com os filtros atuais.")
    elif cat_options[cat_label]:
        fig = px.line(ts, x="periodo", y="n", color="categoria", markers=True, title="Série temporal estratificada", labels={"periodo": "Período", "n": "Registros", "categoria": cat_label})
        render_plotly_chart(fig)
        render_interval_total(ts, value_col="n", by_col="categoria")
        download_button(ts, f"{source.lower()}_serie_temporal_estratificada.csv")
    else:
        fig = px.line(ts, x="periodo", y="n", markers=True, title="Série temporal", labels={"periodo": "Período", "n": "Registros"})
        render_plotly_chart(fig)
        render_interval_total(ts, value_col="n")
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
        render_plotly_chart(fig)
        render_interval_total(heat, value_col="n")
        download_button(heat, f"{source.lower()}_heatmap_ano_mes.csv", "Baixar dados do heatmap")



def render_sinan_lcr_indicators(table: LoadedTable, exprs: Dict[str, Optional[str]], graph_where: str) -> None:
    """Renderiza punção e parâmetros do LCR no bloco de principais indicadores."""
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

    st.markdown("### Punção laboratorial e exame quimiocitológico do líquor")
    st.caption(
        "Estes gráficos usam o recorte exploratório atual, o qual é determinado pelo filtro definido pelo usuário (quando aplicável). "
        f"Material analisado no bloco quimiocitológico: {SINAN_QUIMIO_MATERIAL}."
    )

    for label, expr in [
        ("Punção Laboratorial", exprs.get("puncao_label")),
        ("Exame Quimiocitológico do líquor (LCR)", exprs.get("quimio_label")),
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
                render_plotly_chart(fig)
                render_interval_total(df, value_col="n")
                copyable_dataframe(df, width="stretch", hide_index=True)

    with st.expander("Como os parâmetros do LCR costumam se comportar por etiologia"):
        render_quimio_interpretation()

    quimio_summary = query_sinan_quimio_summary(table, exprs, graph_where)
    if quimio_summary.empty:
        st.info(
            "Para gerar o resumo do Exame Quimiocitológico do líquor (LCR), os campos laboratoriais do SINAN precisam existir "
            "e ser detectados automaticamente, como LAB_GLICO, LAB_LEUCO, LAB_NEUTRO e LAB_PROT."
        )
        return

    for key, titulo in [("glico", "Glicose"), ("prot", "Proteínas"), ("neutro", "Neutrófilos"), ("leuco", "Leucócitos")]:
        expr = exprs.get(f"lab_{key}")
        if not expr:
            st.info(f"Para gerar a distribuição de {titulo}, o campo correspondente precisa existir no SINAN e ser detectado automaticamente.")
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
        render_plotly_chart(fig_dist)
        render_interval_total(dist, value_col="n")
        copyable_dataframe(dist, width="stretch", hide_index=True)
        download_button(dist, f"sinan_quimiocitologico_distribuicao_{safe_filename(titulo)}.csv")

    st.markdown("**Exame Quimiocitológico do líquor (LCR) — valores médios dos parâmetros**")
    resumo_plot = quimio_summary[quimio_summary["n_valido"] > 0].copy()
    if not resumo_plot.empty:
        resumo_plot["texto"] = [
            f"média {float(media):.1f}".replace(".", ",") if pd.notna(media) else "—"
            for media in resumo_plot["media"]
        ]
        fig_quimio = px.bar(
            resumo_plot,
            x="parametro",
            y="media",
            text="texto",
            title="SINAN: média dos parâmetros do exame quimiocitológico do líquor (LCR)",
            labels={
                "parametro": "Parâmetro",
                "material_analisado": "Material analisado",
                "media": "Média",
                "n_valido": "Registros válidos",
            },
            hover_data={
                "texto": False,
                "material_analisado": True,
                "n_valido": True,
                "pct_preenchido": ":.2f",
                "minimo": ":.2f",
                "q1": ":.2f",
                "mediana": ":.2f",
                "q3": ":.2f",
                "maximo": ":.2f",
            },
        )
        render_plotly_chart(fig_quimio)
        render_interval_total(resumo_plot, value_col="n_valido", by_col="parametro", value_label="registros válidos")
    copyable_dataframe(quimio_summary, width="stretch", hide_index=True)
    download_button(quimio_summary, "sinan_quimiocitologico_liquor_resumo_parametros.csv")



def query_sinan_nu_notific_overlap_summary(
    table: LoadedTable,
    nu_notific_col: str,
    where_sql: str,
) -> pd.DataFrame:
    nu_expr = clean_str_expr(nu_notific_col)
    sql = f"""
        WITH base AS (
            SELECT {nu_expr} AS nu_notific
            FROM {table.ref_sql}
            {where_sql}
        ), counts AS (
            SELECT nu_notific, COUNT(*) AS n
            FROM base
            WHERE nu_notific IS NOT NULL
            GROUP BY 1
        ), totals AS (
            SELECT
                COUNT(*) AS total_registros,
                COUNT(*) FILTER (WHERE nu_notific IS NOT NULL) AS registros_com_nu_notific,
                COUNT(*) FILTER (WHERE nu_notific IS NULL) AS registros_sem_nu_notific
            FROM base
        )
        SELECT
            totals.total_registros,
            totals.registros_com_nu_notific,
            totals.registros_sem_nu_notific,
            COALESCE((SELECT COUNT(*) FROM counts), 0) AS nu_notific_distintos,
            COALESCE((SELECT COUNT(*) FROM counts WHERE n > 1), 0) AS nu_notific_com_sobreposicao,
            COALESCE((SELECT SUM(n) FROM counts WHERE n > 1), 0) AS registros_em_sobreposicao,
            CASE WHEN totals.registros_com_nu_notific > 0
                 THEN ROUND(100.0 * COALESCE((SELECT SUM(n) FROM counts WHERE n > 1), 0) / totals.registros_com_nu_notific, 2)
                 ELSE NULL END AS pct_registros_com_sobreposicao
        FROM totals
    """
    return run_query(table, sql)


def query_sinan_nu_notific_overlap_details(
    table: LoadedTable,
    nu_notific_col: str,
    where_sql: str,
    exprs: Dict[str, Optional[str]],
    limit: int = 200,
) -> pd.DataFrame:
    nu_expr = clean_str_expr(nu_notific_col)
    dt_expr = exprs.get("dt")
    classi_expr = exprs.get("classi_label")
    evol_expr = exprs.get("evol_label")
    con_expr = exprs.get("con_group")
    select_bits = [f"{nu_expr} AS nu_notific"]
    if dt_expr:
        select_bits.append(f"{dt_expr} AS data_referencia")
    if classi_expr:
        select_bits.append(f"{classi_expr} AS classificacao_final")
    if evol_expr:
        select_bits.append(f"{evol_expr} AS evolucao")
    if con_expr:
        select_bits.append(f"{con_expr} AS grupo_etiologico")
    select_sql = ",\n                ".join(select_bits)
    optional_cols = []
    if dt_expr:
        optional_cols.extend([
            "MIN(data_referencia) AS primeira_data",
            "MAX(data_referencia) AS ultima_data",
        ])
    if classi_expr:
        optional_cols.append("STRING_AGG(DISTINCT classificacao_final, '; ' ORDER BY classificacao_final) AS classificacoes")
    if evol_expr:
        optional_cols.append("STRING_AGG(DISTINCT evolucao, '; ' ORDER BY evolucao) AS evolucoes")
    if con_expr:
        optional_cols.append("STRING_AGG(DISTINCT grupo_etiologico, '; ' ORDER BY grupo_etiologico) AS grupos_etiologicos")
    optional_sql = (",\n            " + ",\n            ".join(optional_cols)) if optional_cols else ""
    sql = f"""
        WITH base AS (
            SELECT
                {select_sql}
            FROM {table.ref_sql}
            {where_sql}
        ), counts AS (
            SELECT nu_notific, COUNT(*) AS registros
            FROM base
            WHERE nu_notific IS NOT NULL
            GROUP BY 1
            HAVING COUNT(*) > 1
        )
        SELECT
            base.nu_notific,
            counts.registros{optional_sql}
        FROM base
        JOIN counts USING (nu_notific)
        GROUP BY base.nu_notific, counts.registros
        ORDER BY counts.registros DESC, base.nu_notific
        LIMIT {int(limit)}
    """
    return run_query(table, sql)


def render_sinan_overlap_tab(table: LoadedTable, base_where: str, exprs: Dict[str, Optional[str]]) -> None:
    st.markdown("### Sobreposição de `NU_NOTIFIC`")
    st.caption(
        "Esta análise verifica se o mesmo número de notificação aparece em mais de um registro após os filtros-base. "
        "Sobreposição de `NU_NOTIFIC` é um sinal operacional de possível duplicidade ou repetição de caso; a revisão final deve considerar datas, classificação e evolução."
    )
    schema = schema_df(table)
    columns = schema["coluna"].astype(str).tolist() if "coluna" in schema.columns else []
    nu_col = choose_candidate(columns, ["NU_NOTIFIC", "NUM_NOTIFIC", "NUNOTIFIC", "NU_NOTIF"])
    if not nu_col:
        st.warning("Não localizei o campo `NU_NOTIFIC` no SINAN carregado.")
        return

    summary = query_sinan_nu_notific_overlap_summary(table, nu_col, base_where)
    if summary.empty:
        st.info("Sem registros para avaliar `NU_NOTIFIC` com os filtros atuais.")
        return
    row = summary.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros avaliados", f"{int(row['total_registros']):,}".replace(",", "."))
    c2.metric("NU_NOTIFIC preenchidos", f"{int(row['registros_com_nu_notific']):,}".replace(",", "."))
    c3.metric("NU_NOTIFIC sobrepostos", f"{int(row['nu_notific_com_sobreposicao']):,}".replace(",", "."))
    pct = row.get("pct_registros_com_sobreposicao")
    c4.metric(
        "Registros em sobreposição",
        f"{int(row['registros_em_sobreposicao']):,}".replace(",", "."),
        None if pd.isna(pct) else f"{float(pct):.2f}%".replace(".", ","),
    )
    copyable_dataframe(summary, width="stretch", hide_index=True)
    download_button(summary, "sinan_sobreposicao_nu_notific_resumo.csv")

    details = query_sinan_nu_notific_overlap_details(table, nu_col, base_where, exprs, limit=200)
    if details.empty:
        st.success("Não há `NU_NOTIFIC` repetido no recorte atual.")
        return
    plot_df = details.head(30).copy()
    plot_df["texto"] = plot_df["registros"].astype(int).astype(str)
    fig = px.bar(
        plot_df,
        x="registros",
        y="nu_notific",
        orientation="h",
        text="texto",
        title="SINAN: principais NU_NOTIFIC com sobreposição",
        labels={"nu_notific": "NU_NOTIFIC", "registros": "Registros"},
    )
    fig.update_layout(yaxis={"categoryorder": "array", "categoryarray": plot_df["nu_notific"].tolist()[::-1]})
    render_plotly_chart(fig)
    copyable_dataframe(details, width="stretch", hide_index=True)
    download_button(details, "sinan_sobreposicao_nu_notific_detalhes.csv")

def render_indicators_tab(table: LoadedTable, source: str, base_where: str, graph_where: str, exprs: Dict[str, Optional[str]]) -> None:
    st.markdown("Os principais indicadores epidemiológicos usam os filtros-base. Os blocos laboratoriais movidos para esta área usam o recorte exploratório atual para preservar a leitura original dos gráficos.")

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

        sinan_schema = schema_df(table)
        sinan_columns = sinan_schema["coluna"].astype(str).tolist() if "coluna" in sinan_schema.columns else []
        symptom_specs = _available_column_specs(sinan_columns, SINAN_SYMPTOM_FIELDS)
        vaccine_specs = _available_column_specs(sinan_columns, SINAN_VACCINE_FIELDS)
        hospital_col = choose_candidate(sinan_columns, ["ATE_HOSPIT"])
        communicants_col = choose_candidate(sinan_columns, ["MED_NUCOMU", "NU_COMUNICANTES", "NUM_COMUNICANTES", "COMUNICANTES"])
        prophylaxis_col = choose_candidate(sinan_columns, ["MED_QUIMIO", "QUIMIOPROFILAXIA", "PROFILAXIA", "QUIMIO"])

        copyable_dataframe(ind, width="stretch", hide_index=True)
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
        render_plotly_chart(fig)
        render_interval_total(count_long, value_col="n", by_col="indicador")

        if exprs.get("evol_label") and exprs.get("dt") and exprs.get("classi_code"):
            evol_confirmados = query_yearly_category(
                table,
                exprs["dt"],
                exprs["evol_label"],
                append_clause(base_where, f"{exprs['classi_code']} = '1'"),
            )
            if not evol_confirmados.empty:
                evol_confirmados = add_text_column(evol_confirmados)
                fig_evol_confirmados = px.line(
                    evol_confirmados,
                    x="ano",
                    y="n",
                    color="categoria",
                    markers=True,
                    text="texto",
                    title="SINAN: Evolução de casos confirmados",
                    labels={"ano": "Ano", "n": "Confirmados", "categoria": "Evolução", "pct": "% no ano"},
                    hover_data={"texto": False, "pct": ":.2f", "total_ano": True},
                )
                fig_evol_confirmados.update_traces(textposition="top center")
                render_plotly_chart(fig_evol_confirmados)
                render_interval_total(evol_confirmados, value_col="n", by_col="categoria")
                copyable_dataframe(evol_confirmados, width="stretch", hide_index=True)
                download_button(evol_confirmados, "sinan_evolucao_confirmados.csv")
        else:
            st.info("Para gerar o gráfico de evolução dos confirmados, CLASSI_FIN, EVOLUCAO e data precisam existir no SINAN.")

        prop_specs = [
            ("pct_confirmacao", "Confirmados", "confirmados", "notificacoes", "% das notificações"),
            ("pct_descarte", "Descartados", "descartados", "notificacoes", "% das notificações"),
            ("pct_inconclusivos", "Inconclusivos", "inconclusivos", "notificacoes", "% das notificações"),
            ("pct_sem_classificacao", "Sem confirmação/ignorados", "sem_classificacao", "notificacoes", "% das notificações"),
            ("letalidade_confirmados", "Letalidade — óbitos por meningite / confirmados", "obitos_meningite_confirmados", "confirmados", "% dos casos confirmados"),
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
            color_discrete_map={LETHALITY_LABEL: LETHALITY_RED},
        )
        fig2.update_traces(textposition="top center")
        render_plotly_chart(fig2)
        render_interval_total(prop_long, value_col="n", by_col="indicador", denominator_col="denominador", denominator_label="registros do denominador")

        assistencia = query_sinan_hospitalization_internment(table, exprs, base_where, hospital_col)
        if not assistencia.empty:
            assistencia = assistencia.copy()
            assistencia["texto"] = [f"{br_pct(p)} (n={br_int(n)})" for p, n in zip(assistencia["pct"], assistencia["n"])]
            grupo_hosp_order = ["Suspeitos/notificados", "Confirmados", "Descartados"]
            fig_assistencia = px.bar(
                assistencia,
                x="ano",
                y="pct",
                color="grupo_caso",
                barmode="group",
                text="texto",
                title="SINAN: ocorrência de hospitalização por definição de caso",
                labels={"ano": "Ano", "pct": "% no grupo", "grupo_caso": "Grupo", "n": "Registros", "denominador": "Denominador"},
                hover_data={"texto": False, "n": True, "denominador": True},
                category_orders={"grupo_caso": grupo_hosp_order},
            )
            render_plotly_chart(fig_assistencia)
            render_interval_total(assistencia, value_col="n", by_col="grupo_caso", denominator_col="denominador", denominator_label="registros do grupo")
            copyable_dataframe(assistencia.drop(columns=["ordem_grupo"], errors="ignore"), width="stretch", hide_index=True)
            download_button(assistencia.drop(columns=["ordem_grupo"], errors="ignore"), "sinan_hospitalizacao_suspeitos_confirmados_descartados.csv")
        else:
            st.info("Para gerar o gráfico comparativo de hospitalização, CLASSI_FIN, data e ATE_HOSPIT precisam existir no SINAN.")

        etio = query_sinan_etiology_lethality(table, exprs, base_where)
        if not etio.empty:
            st.markdown("**Letalidade por grupo etiológico entre confirmados**")
            st.caption("Denominador do gráfico: casos confirmados do respectivo grupo etiológico (CLASSI_FIN = 1). Fórmula: óbitos por meningite entre confirmados / casos confirmados do grupo.")
            copyable_dataframe(etio, width="stretch", hide_index=True)
            etio = etio.copy()
            etio["denominador_letalidade"] = etio["confirmados"]
            etio["texto"] = [
                f"{br_pct(p)} ({br_int(o)}/{br_int(c)})"
                for p, o, c in zip(etio["letalidade_pct"], etio["obitos_meningite"], etio["denominador_letalidade"])
            ]
            fig3 = px.bar(
                etio,
                x="letalidade_pct",
                y="grupo_etiologico",
                orientation="h",
                text="texto",
                title="Letalidade por grupo etiológico — denominador: casos confirmados",
                labels={"letalidade_pct": "Óbitos por meningite / casos confirmados (%)", "grupo_etiologico": "Grupo etiológico", "denominador_letalidade": "Casos confirmados do grupo"},
                hover_data={"obitos_meningite": True, "denominador_letalidade": True},
                color_discrete_sequence=[DEATH_RED],
            )
            fig3.update_layout(yaxis={"categoryorder": "total ascending"})
            render_plotly_chart(fig3)
            render_interval_total(etio, value_col="obitos_meningite", denominator_col="confirmados", value_label="óbitos por meningite", denominator_label="casos confirmados")
            download_button(etio, "sinan_letalidade_por_etiologia.csv")

        if symptom_specs and exprs.get("classi_code") and exprs.get("dt"):
            sintomas = query_sinan_symptom_prevalence(table, exprs, base_where, symptom_specs)
            if not sintomas.empty:
                st.markdown('<h2 style="font-size: 2.0rem; line-height: 1.18; margin: 1.1rem 0 0.65rem 0;">Prevalência de sinais e sintomas entre casos confirmados</h2>', unsafe_allow_html=True)
                opcoes_sintomas = sorted(sintomas["sintoma"].dropna().unique().tolist())
                sintoma_sel = st.selectbox(
                    "Escolha o sintoma para se analisar a curva anual entre confirmados",
                    options=opcoes_sintomas,
                    key="sinan_indicadores_sintoma_prevalencia",
                )
                sintomas_sel = sintomas[sintomas["sintoma"].eq(sintoma_sel)].copy()
                sintomas_sel["texto"] = [f"{br_pct(p)} (n={br_int(n)})" for p, n in zip(sintomas_sel["pct_sintoma_confirmados"], sintomas_sel["sintoma_sim"])]
                fig_sintoma = px.line(
                    sintomas_sel,
                    x="ano",
                    y="pct_sintoma_confirmados",
                    markers=True,
                    text="texto",
                    title=f"SINAN: prevalência anual de {sintoma_sel} entre casos confirmados",
                    labels={"ano": "Ano", "pct_sintoma_confirmados": "% dos confirmados", "sintoma_sim": "Confirmados com sintoma"},
                    hover_data={"texto": False, "sintoma_sim": True, "confirmados": True, "sintoma_nao": True, "sintoma_ignorado": True},
                )
                fig_sintoma.update_traces(textposition="top center")
                render_plotly_chart(fig_sintoma)
                render_interval_total(sintomas_sel, value_col="sintoma_sim", denominator_col="confirmados", value_label="confirmados com sintoma", denominator_label="casos confirmados")

                sintomas_resumo = (
                    sintomas
                    .groupby("sintoma", dropna=False, as_index=False)
                    .agg(
                        confirmados=("confirmados", "sum"),
                        sintoma_sim=("sintoma_sim", "sum"),
                        sintoma_nao=("sintoma_nao", "sum"),
                        sintoma_ignorado=("sintoma_ignorado", "sum"),
                    )
                )
                sintomas_resumo["pct_sintoma_confirmados"] = (100.0 * sintomas_resumo["sintoma_sim"] / sintomas_resumo["confirmados"].replace({0: np.nan})).round(2)
                sintomas_resumo = sintomas_resumo.sort_values("pct_sintoma_confirmados", ascending=True)
                sintomas_resumo["texto"] = [f"{br_pct(p)} (n={br_int(n)})" for p, n in zip(sintomas_resumo["pct_sintoma_confirmados"], sintomas_resumo["sintoma_sim"])]
                fig_sintomas_resumo = px.bar(
                    sintomas_resumo,
                    x="pct_sintoma_confirmados",
                    y="sintoma",
                    orientation="h",
                    text="texto",
                    title="SINAN: prevalência acumulada dos sinais e sintomas entre confirmados",
                    labels={"pct_sintoma_confirmados": "% dos confirmados", "sintoma": "Sinal/sintoma"},
                    hover_data={"texto": False, "sintoma_sim": True, "confirmados": True, "sintoma_nao": True, "sintoma_ignorado": True},
                )
                render_plotly_chart(fig_sintomas_resumo)
                render_interval_total(sintomas_resumo, value_col="sintoma_sim", by_col="sintoma")
                copyable_dataframe(sintomas, width="stretch", hide_index=True)
                download_button(sintomas, "sinan_prevalencia_sintomas_confirmados.csv")
        else:
            st.info("Para gerar a prevalência de sintomas, CLASSI_FIN, data e os campos clínicos CLI_* precisam existir no SINAN.")

        comunicantes = query_sinan_communicants_prophylaxis(table, exprs, base_where, communicants_col, prophylaxis_col)
        if not comunicantes.empty:
            st.markdown("**Comunicantes e quimioprofilaxia**")
            comunicantes = comunicantes.copy()
            comunicantes["texto_comunicantes"] = [br_int(v) for v in comunicantes["comunicantes_total"]]
            fig_comunicantes = px.line(
                comunicantes,
                x="ano",
                y="comunicantes_total",
                color="quimioprofilaxia",
                markers=True,
                text="texto_comunicantes",
                title="SINAN: número de comunicantes por realização de quimioprofilaxia",
                labels={"ano": "Ano", "comunicantes_total": "Comunicantes", "quimioprofilaxia": "Quimioprofilaxia"},
                hover_data={"texto_comunicantes": False, "registros": True, "registros_com_comunicantes": True, "media_comunicantes": True, "pct_comunicantes_ano": ":.2f"},
            )
            fig_comunicantes.update_traces(textposition="top center")
            render_plotly_chart(fig_comunicantes)
            render_interval_total(comunicantes, value_col="comunicantes_total", by_col="quimioprofilaxia", value_label="comunicantes")
            copyable_dataframe(comunicantes, width="stretch", hide_index=True)
            download_button(comunicantes, "sinan_comunicantes_quimioprofilaxia.csv")
        else:
            st.info("Para gerar o gráfico de comunicantes/profilaxia, MED_NUCOMU e/ou MED_QUIMIO precisam existir no SINAN.")

        vacinacao = query_sinan_vaccination_by_classification(table, exprs, base_where, vaccine_specs)
        if not vacinacao.empty:
            st.markdown("**Vacinação por classificação final do caso**")
            vacinacao = vacinacao.copy()
            vacinacao["texto"] = [f"{br_pct(p)} (n={br_int(n)})" for p, n in zip(vacinacao["pct_vacinados_sim"], vacinacao["vacinados_sim"])]
            fig_vacinacao = px.bar(
                vacinacao,
                x="vacina",
                y="pct_vacinados_sim",
                color="grupo_classificacao",
                barmode="group",
                text="texto",
                title="SINAN: vacinação informada como 'Sim' em confirmados vs descartados/inconclusivos",
                labels={"vacina": "Vacina", "pct_vacinados_sim": "% com vacinação = Sim", "grupo_classificacao": "Classificação", "denominador": "Denominador"},
                hover_data={"texto": False, "vacinados_sim": True, "vacinados_nao": True, "vacinacao_ignorada": True, "denominador": True},
            )
            fig_vacinacao.update_xaxes(tickangle=-30)
            render_plotly_chart(fig_vacinacao)
            render_interval_total(vacinacao, value_col="vacinados_sim", by_col="vacina", value_label="vacinados com informação = Sim")
            copyable_dataframe(vacinacao, width="stretch", hide_index=True)
            download_button(vacinacao, "sinan_vacinacao_confirmados_descartados_inconclusivos.csv")
        else:
            st.info("Para gerar o gráfico de vacinação, CLASSI_FIN e campos ANT_* de vacinação precisam existir no SINAN.")

        render_sinan_lcr_indicators(table, exprs, graph_where)

        return

    if source == "SIM":
        ind = query_sim_indicators(table, exprs, base_where)
        if ind.empty:
            st.warning("Não foi possível calcular indicadores principais do SIM. Verifique data, CAUSABAS e campos CID.")
        else:
            copyable_dataframe(ind, width="stretch", hide_index=True)
            download_button(ind, "sim_indicadores_anuais.csv")

            sim_cid_specs = [
                ("obitos_com_mencao_meningite", "Meningite mencionada", "pct_mencao_meningite"),
                ("obitos_causa_basica_meningite", "Meningite como causa básica", "pct_causa_basica_meningite"),
            ]
            sim_cid_rows = []
            for _, row in ind.iterrows():
                for n_col, label, pct_col in sim_cid_specs:
                    pct = row[pct_col] if pct_col in ind.columns else None
                    sim_cid_rows.append({
                        "ano": row["ano"],
                        "definicao": label,
                        "n": row[n_col],
                        "pct": pct,
                        "denominador": row["obitos_registros"],
                        "texto": count_pct_text(row[n_col], pct),
                    })
            sim_cid_long = pd.DataFrame(sim_cid_rows)
            fig = px.line(
                sim_cid_long,
                x="ano",
                y="n",
                color="definicao",
                markers=True,
                text="texto",
                title="SIM: Óbitos com meningite sendo mencionada ou como causa básica",
                labels={
                    "ano": "Ano",
                    "n": "Óbitos",
                    "definicao": "Definição de CID",
                    "pct": "% dos óbitos no recorte",
                    "denominador": "Óbitos no recorte",
                },
                hover_data={"texto": False, "pct": ":.2f", "denominador": True},
            )
            fig.update_traces(textposition="top center")
            render_plotly_chart(fig)
            render_interval_total(sim_cid_long, value_col="n", by_col="definicao")

        def render_sim_cycle_chart(
            category_sql: Optional[str],
            field_label: str,
            where_sql: Optional[str],
            markdown_title: str,
            figure_title: str,
            caption: str,
            filename: str,
        ) -> None:
            if not (exprs.get("dt") and category_sql and where_sql):
                return
            df = query_yearly_category(table, exprs["dt"], category_sql, where_sql)
            if df.empty:
                st.info(f"Sem dados para {markdown_title.lower()} com esta definição de meningite.")
                return
            st.markdown(f"**{markdown_title}**")
            st.caption(caption)
            df = add_text_column(df)
            fig_cycle = px.bar(
                df,
                x="ano",
                y="n",
                color="categoria",
                text="texto",
                title=figure_title,
                labels={"ano": "Ano", "n": "Óbitos", "categoria": field_label, "pct": "% no ano"},
                hover_data={"texto": False, "pct": ":.2f", "total_ano": True},
            )
            render_plotly_chart(fig_cycle)
            render_interval_total(df, value_col="n", by_col="categoria")
            copyable_dataframe(df, width="stretch", hide_index=True)
            download_button(df, filename)

        cid_any = exprs.get("cid")
        causabas = exprs.get("causabas_cid")
        mention_where = append_clause(base_where, f"{cid_any} IS NOT NULL") if cid_any else None
        primary_cause_where = append_clause(base_where, f"{causabas} IS NOT NULL") if causabas else None

        if exprs.get("dt") and exprs.get("obitograv_label"):
            if mention_where:
                render_sim_cycle_chart(
                    exprs["obitograv_label"],
                    "OBITOGRAV",
                    mention_where,
                    "Óbito na gravidez — menção de meningite",
                    "SIM: óbito na gravidez (OBITOGRAV) — óbitos com menção de meningite",
                    "Observação: este gráfico foi construído com base nos óbitos em que houve menção de CID de meningite em qualquer campo do SIM.",
                    "sim_obito_gravidez_obitograv_mencao_meningite.csv",
                )
            else:
                st.info("Para gerar o gráfico de gravidez por menção de meningite, é necessário detectar algum campo CID no SIM.")
            if primary_cause_where:
                render_sim_cycle_chart(
                    exprs["obitograv_label"],
                    "OBITOGRAV",
                    primary_cause_where,
                    "Óbito na gravidez — meningite como causa primária/básica",
                    "SIM: óbito na gravidez (OBITOGRAV) — meningite como causa primária/básica",
                    "Este gráfico foi construído apenas com óbitos cuja causa primária/básica contém CID de meningite em CAUSABAS.",
                    "sim_obito_gravidez_obitograv_causa_basica_meningite.csv",
                )
            else:
                st.info("Para gerar o gráfico de gravidez por causa primária/básica, o campo CAUSABAS precisa existir no SIM e ser detectado automaticamente.")
        else:
            st.info("Para o gráfico de óbito na gravidez, o campo OBITOGRAV precisa existir no SIM e ser detectado automaticamente.")

        if exprs.get("dt") and exprs.get("obitopuerp_label"):
            if mention_where:
                render_sim_cycle_chart(
                    exprs["obitopuerp_label"],
                    "OBITOPUERP",
                    mention_where,
                    "Óbito no puerpério — menção de meningite",
                    "SIM: óbito no puerpério (OBITOPUERP) — óbitos com menção de meningite",
                    "Observação: este gráfico foi construído com base nos óbitos em que houve menção de CID de meningite em qualquer campo do SIM.",
                    "sim_obito_puerperio_obitopuerp_mencao_meningite.csv",
                )
            else:
                st.info("Para gerar o gráfico de puerpério por menção de meningite, é necessário detectar algum campo CID no SIM.")
            if primary_cause_where:
                render_sim_cycle_chart(
                    exprs["obitopuerp_label"],
                    "OBITOPUERP",
                    primary_cause_where,
                    "Óbito no puerpério — meningite como causa primária/básica",
                    "SIM: óbito no puerpério (OBITOPUERP) — meningite como causa primária/básica",
                    "Este gráfico foi construído apenas com óbitos cuja causa primária/básica contém CID de meningite em CAUSABAS.",
                    "sim_obito_puerperio_obitopuerp_causa_basica_meningite.csv",
                )
            else:
                st.info("Para gerar o gráfico de puerpério por causa primária/básica, o campo CAUSABAS precisa existir no SIM e ser detectado automaticamente.")
        else:
            st.info("Para o gráfico de óbito no puerpério, o campo OBITOPUERP precisa existir no SIM e ser detectado automaticamente.")
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
        copyable_dataframe(ind, width="stretch", hide_index=True)
        download_button(ind, "ciha_indicadores_anuais.csv")
        ciha_count_specs = [
            ("atendimentos", "Atendimentos", None),
            ("atendimentos_diag_principal_meningite", "Diagnóstico principal de meningite", "pct_atendimentos_diag_principal_meningite"),
            ("mortes_administrativas", "Mortes administrativas", "pct_morte_administrativa"),
        ]
        ciha_count_rows = []
        for _, row in ind.iterrows():
            for n_col, label, pct_col in ciha_count_specs:
                pct = 100.0 if pct_col is None else row[pct_col]
                ciha_count_rows.append({
                    "ano": row["ano"],
                    "indicador": label,
                    "n": row[n_col],
                    "pct": pct,
                    "denominador": row["atendimentos"],
                    "texto": count_pct_text(row[n_col], pct),
                })
        ciha_count_long = pd.DataFrame(ciha_count_rows)
        fig = px.line(
            ciha_count_long,
            x="ano",
            y="n",
            color="indicador",
            markers=True,
            text="texto",
            title="CIHA: atendimentos e mortes administrativas",
            labels={"ano": "Ano", "n": "Atendimentos/registros", "indicador": "Indicador", "pct": "% dos atendimentos"},
            hover_data={"texto": False, "pct": ":.2f", "denominador": True},
        )
        fig.update_traces(textposition="top center")
        render_plotly_chart(fig)
        render_interval_total(ciha_count_long, value_col="n", by_col="indicador")

    if exprs.get("dt") and exprs.get("modalidade_label"):
        modalidade = query_yearly_category(table, exprs["dt"], exprs["modalidade_label"], base_where)
        if not modalidade.empty:
            st.markdown("**Modalidade do atendimento — hospitalar vs ambulatorial**")
            modalidade = add_text_column(modalidade)
            fig_modalidade = px.bar(
                modalidade,
                x="ano",
                y="n",
                color="categoria",
                text="texto",
                title="CIHA: atendimentos por modalidade hospitalar e ambulatorial",
                labels={"ano": "Ano", "n": "Atendimentos", "categoria": "Modalidade", "pct": "% no ano"},
                hover_data={"texto": False, "pct": ":.2f", "total_ano": True},
            )
            fig_modalidade.update_layout(barmode="stack")
            render_plotly_chart(fig_modalidade)
            render_interval_total(modalidade, value_col="n", by_col="categoria")
            copyable_dataframe(modalidade, width="stretch", hide_index=True)
            download_button(modalidade, "ciha_modalidade_hospitalar_ambulatorial.csv")
        else:
            st.info("Sem dados de modalidade no recorte atual da CIHA.")
    else:
        st.info("Para gerar o gráfico de hospitalar vs ambulatorial, os campos de data e MODALIDADE precisam existir na CIHA e ser detectados automaticamente.")

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
        render_plotly_chart(fig_dias)
        render_interval_total(dias_dist, value_col="n")
        copyable_dataframe(dias_dist, width="stretch", hide_index=True)
        download_button(dias_dist, "ciha_dias_permanencia_distribuicao.csv")
    else:
        st.info("Para gerar o gráfico de dias de permanência, o campo DIAS_PERM precisa existir na CIHA e ser detectado automaticamente.")


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
            st.warning("Não localizei campo CID-10 válido pela detecção automática para ativar esta análise.")
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
            render_plotly_chart(fig)
            render_interval_total(cid_dist, value_col="n")
            copyable_dataframe(cid_dist, width="stretch", hide_index=True)
            download_button(cid_dist, f"{source.lower()}_cid10_distribuicao.csv")

            conv_adequacy = query_cid10_adequacy_conversion(table, exprs, graph_where)
            if not conv_adequacy.empty:
                conv_adequacy = add_text(conv_adequacy)
                conv_adequacy_plot = summarize_cid10_adequacy_plot(conv_adequacy)
                st.markdown("### Conversão para adequação ao CID-10 de meningite / encefalite")
                st.caption(CID10_ADEQUACY_OBSERVATION)
                if conv_adequacy_plot.empty:
                    st.info("Não houve CID-10 detectado para exibir no gráfico de adequação no recorte atual.")
                else:
                    conv_adequacy_plot = add_text(conv_adequacy_plot)
                    fig_conv = px.bar(
                        conv_adequacy_plot,
                        x="n",
                        y="categoria_grafico",
                        orientation="h",
                        text="texto",
                        title=f"{source}: Conversão para adequação ao CID-10 de meningite / encefalite",
                        labels={"categoria_grafico": "CID-10 adequado (prefixo)", "n": "Registros", "pct": "% do total detectado"},
                        hover_data={
                            "texto": False,
                            "pct": ":.2f",
                            "denominador": True,
                            "cid10_adequado_classificacao": True,
                            "status_conversao": True,
                            "cid10_originais": True,
                            "cids_detectados": True,
                            "campos_origem": True,
                        },
                    )
                    fig_conv.update_layout(yaxis={"categoryorder": "total ascending"})
                    render_plotly_chart(fig_conv)
                    render_interval_total(conv_adequacy_plot, value_col="n")
                    st.caption(
                        "Gráfico agregado pelo CID-10 adequado final: CID-10 convertidos somam no destino "
                        "e CID-10 prefixados já presentes permanecem em seu próprio grupo."
                    )
                    st.caption(build_cid10_adequacy_conversion_note(conv_adequacy))
                display_cols = [
                    c for c in [
                        "cid10_original", "cid10_adequado_grupo", "cid10_adequado_classificacao",
                        "status_conversao", "n", "pct", "denominador", "cids_detectados",
                        "classificacoes_originais", "observacoes", "campos_origem",
                    ]
                    if c in conv_adequacy.columns
                ]
                copyable_dataframe(conv_adequacy[display_cols], width="stretch", hide_index=True)
                download_button(conv_adequacy, f"{source.lower()}_cid10_conversao_adequacao_meningite_encefalite.csv")
                with st.expander("Regra usada para a conversão de adequação"):
                    copyable_dataframe(pd.DataFrame(CID10_ADEQUACY_MAPPING_ROWS), width="stretch", hide_index=True)
                    st.caption(CID10_ADEQUACY_OBSERVATION)

            g01_g02 = query_g01_g02_cid_distribution(table, exprs, graph_where)
            if not g01_g02.empty:
                g01_g02 = add_text(g01_g02)
                st.markdown("**Verificação específica — G01 e G02 em SIM/CIHA**")
                st.caption(
                    "Este bloco usa apenas CID-10 bruto informado no próprio SIM/CIHA. "
                    "Não há conversão automática da doença de base para G01/G02 quando o código G01/G02 não aparece no campo CID."
                )
                fig_g01_g02 = px.bar(
                    g01_g02,
                    x="n",
                    y="tipo",
                    orientation="h",
                    text="texto",
                    title=f"{source}: registros classificados como G01 ou G02",
                    labels={"tipo": "Tipo CID-10", "n": "Registros", "pct": "%"},
                    hover_data={"texto": False, "pct": ":.2f", "denominador": True},
                )
                fig_g01_g02.update_layout(yaxis={"categoryorder": "total ascending"})
                render_plotly_chart(fig_g01_g02)
                render_interval_total(g01_g02, value_col="n")
                copyable_dataframe(g01_g02, width="stretch", hide_index=True)
                download_button(g01_g02, f"{source.lower()}_verificacao_g01_g02.csv")

        if source == "CIHA":
            st.markdown("### Óbitos CIHA — CID-10 destes")
            morte = exprs.get("morte_code")
            if not morte:
                st.info("Para mostrar os óbitos da CIHA e seus CID-10, o campo MORTE precisa existir na CIHA e ser detectado automaticamente.")
            elif not exprs.get("cid"):
                st.info("Para mostrar o CID-10 dos óbitos da CIHA, ao menos um campo de diagnóstico/CID-10 precisa existir e ser detectado automaticamente.")
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
                        render_plotly_chart(fig_death)
                        render_interval_total(death_cid, value_col="n", value_label="óbitos CIHA")
                        copyable_dataframe(death_cid, width="stretch", hide_index=True)
                        download_button(death_cid, "ciha_obitos_cid10_distribuicao.csv")
        return

    st.markdown("### Classificação específica do SINAN")
    st.info(
        "No SINAN, o CID bruto do agravo pode estar como G039 para muitos registros. "
        "Por isso, esta aba prioriza CON_DIAGES e os campos complementares CLA_ME_BAC, CLA_ME_ASS e CLA_ME_ETI. "
        "CON_DIAGES=05 deixou de ser convertido automaticamente para G04.2; ele é refinado como G00 ou G01 quando há informação suficiente."
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
                render_plotly_chart(fig)
                render_interval_total(df, value_col="n")
                copyable_dataframe(df, width="stretch", hide_index=True)


        if label == "Grupo etiológico SINAN":
            by_year = query_sinan_diagnostics_by_year(table, exprs, graph_where)
            if not by_year.empty:
                st.markdown("**Confirmados por grupo etiológico convertido em CID-10**")
                st.caption(
                    "Usa a classificação CID-10 derivada de CON_DIAGES, CLA_ME_BAC e campos complementares do SINAN, "
                    "restrita a casos confirmados. O ID_AGRAVO/CID bruto do SINAN não é usado para estratificar este gráfico."
                )
                plot_by_year = by_year.copy()
                plot_by_year["ano"] = plot_by_year["ano"].astype(int).astype(str)
                fig4 = px.bar(
                    plot_by_year,
                    x="ano",
                    y="confirmados",
                    color="grupo_etiologico",
                    title="Confirmados por grupo etiológico convertido em CID-10 — SINAN",
                    labels={
                        "grupo_etiologico": "CID-10 convertido / grupo etiológico",
                        "confirmados": "Confirmados",
                        "ano": "Ano",
                        "cid10_grupo": "Família CID-10",
                        "total_ano": "Total no ano",
                        "pct_ano": "% no ano",
                    },
                    hover_data={"cid10_grupo": True, "total_ano": True, "pct_ano": ":.2f"},
                )
                fig4.update_layout(barmode="stack")
                fig4.update_xaxes(type="category")
                render_plotly_chart(fig4)
                render_interval_total(by_year, value_col="confirmados", by_col="grupo_etiologico")
                copyable_dataframe(by_year, width="stretch", hide_index=True)
                download_button(by_year, "sinan_confirmados_por_cid10_convertido_ano.csv")

            conv = query_sinan_cid10_conversion(table, exprs, graph_where)
            if not conv.empty:
                conv_yes = conv[conv["incluido_comparacao"].eq("Sim")].copy()
                conv_no = conv[~conv["incluido_comparacao"].eq("Sim")].copy()

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
                        title="SINAN: classificação etiológica convertida para CID-10",
                        labels={"cid10_classificacao": "CID-10 convertido", "n": "Registros", "pct": "%"},
                        hover_data={"texto": False, "pct": ":.2f", "denominador": True, "grupos_sinan": True, "conclusoes_sinan": True},
                    )
                    fig_conv.update_layout(yaxis={"categoryorder": "total ascending"})
                    render_plotly_chart(fig_conv)
                    render_interval_total(conv_yes, value_col="n")
                    display_cols = [
                        c for c in [
                            "cid10_grupo", "cid10_classificacao", "n", "pct",
                            "grupos_sinan", "conclusoes_sinan", "bacterias_sinan",
                            "agentes_asseptica_sinan", "outras_etiologias_sinan",
                            "doencas_base_g01_provaveis", "justificativas",
                        ]
                        if c in conv_yes.columns
                    ]
                    copyable_dataframe(conv_yes[display_cols], width="stretch", hide_index=True)
                    download_button(conv_yes, "sinan_cid10_conversao_grupo_etiologico.csv")

                with st.expander("Regra usada para converter CON_DIAGES em CID-10"):
                    st.markdown("**Conversão principal CON_DIAGES -> família CID-10**")
                    copyable_dataframe(pd.DataFrame(SINAN_CID10_MAPPING_ROWS), width="stretch", hide_index=True)
                    st.markdown("**Refinamento específico para CON_DIAGES=05 — meningite por outras bactérias**")
                    copyable_dataframe(pd.DataFrame(SINAN_OTHER_BACTERIA_CID10_RULE_ROWS), width="stretch", hide_index=True)
                    st.markdown("**G01 — doença bacteriana de base provável**")
                    copyable_dataframe(pd.DataFrame(SINAN_G01_BASE_DISEASE_REFERENCE_ROWS), width="stretch", hide_index=True)
                    st.caption(
                        "Observação: CON_DIAGES 01 (meningococcemia isolada) fica fora da conversão; "
                        "CON_DIAGES 02 e 03 entram como A39.0; CON_DIAGES 05 entra como G00 por padrão e como G01 quando CLA_ME_BAC/texto sugerem doença bacteriana classificada em outra parte."
                    )

                if not conv_no.empty:
                    st.caption("Registros não convertidos para a comparação CID-10, mantendo transparência da exclusão/ausência de mapeamento:")
                    conv_no_cols = [c for c in ["cid10_grupo", "cid10_classificacao", "n", "pct", "conclusoes_sinan", "justificativas"] if c in conv_no.columns]
                    copyable_dataframe(conv_no[conv_no_cols], width="stretch", hide_index=True)


                g01_base = query_sinan_g01_base_disease(table, exprs, graph_where)
                if not g01_base.empty:
                    g01_base = add_text(g01_base)
                    st.markdown("**G01 + doença de base provável**")
                    st.caption(
                        "Este gráfico mostra apenas registros cuja conversão operacional resultou em G01. "
                        "A doença de base provável é inferida de CLA_ME_BAC e dos campos textuais/auxiliares detectados automaticamente."
                    )
                    fig_g01 = px.bar(
                        g01_base,
                        x="n",
                        y="doenca_base_provavel",
                        orientation="h",
                        text="texto",
                        title="SINAN: G01 + doença bacteriana de base provável",
                        labels={"doenca_base_provavel": "Doença de base provável", "n": "Registros", "pct": "%"},
                        hover_data={"texto": False, "pct": ":.2f", "denominador": True, "conclusoes_sinan": True, "bacterias_sinan": True},
                    )
                    fig_g01.update_layout(yaxis={"categoryorder": "total ascending"})
                    render_plotly_chart(fig_g01)
                    render_interval_total(g01_base, value_col="n")
                    copyable_dataframe(g01_base, width="stretch", hide_index=True)
                    download_button(g01_base, "sinan_g01_doenca_base_provavel.csv")


    for label, expr in [
        ("EVOLUCAO", exprs.get("evol_label")),
        ("Critério de confirmação para classificação etiológica do caso", exprs.get("criterio_label")),
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
                render_plotly_chart(fig)
                render_interval_total(df, value_col="n")
                copyable_dataframe(df, width="stretch", hide_index=True)



def render_demography_tab(table: LoadedTable, source: str, graph_where: str, exprs: Dict[str, Optional[str]], base_where: Optional[str] = None) -> None:
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
            render_plotly_chart(fig)
            render_interval_total(age_df, value_col="n")
            download_button(age_df, f"{source.lower()}_idade.csv")
        sex = exprs.get("sex")
        if sex:
            pyr = query_age_dist(table, age, graph_where, sex_sql=sex)
            if not pyr.empty:
                pyr["faixa"] = pyr["faixa_ini"].astype(int).astype(str) + "–" + (pyr["faixa_ini"].astype(int) + 4).astype(str)
                pyr["valor"] = np.where(pyr["sexo"].eq("Masculino"), -pyr["n"], pyr["n"])
                fig = px.bar(pyr, x="valor", y="faixa", color="sexo", orientation="h", title="Pirâmide etária por sexo", labels={"valor": "Registros", "faixa": "Faixa etária"})
                fig.update_layout(barmode="relative")
                render_plotly_chart(fig)
                render_interval_total(pyr, value_col="n", by_col="sexo")
                download_button(pyr, f"{source.lower()}_piramide.csv")

    education = exprs.get("education")
    if source == "SINAN":
        st.markdown("### Escolaridade")
        if not education:
            st.info("Para gerar o gráfico de escolaridade no SINAN, o campo CS_ESCOL_N/ESCOLARIDADE precisa existir e ser detectado automaticamente.")
        elif not (exprs.get("classi_code") and exprs.get("evol_code")):
            st.info("Para gerar a escolaridade por confirmados e óbitos no SINAN, os campos CLASSI_FIN e EVOLUCAO precisam existir e ser detectados automaticamente.")
        else:
            schooling_where = base_where if base_where is not None else graph_where
            edu_df = query_sinan_education_outcomes(
                table,
                education,
                exprs["classi_code"],
                exprs["evol_code"],
                schooling_where,
            )
            if edu_df.empty or pd.to_numeric(edu_df.get("denominador"), errors="coerce").fillna(0).max() <= 0:
                st.info("Sem casos confirmados para calcular a escolaridade com os filtros atuais.")
            else:
                edu_df = add_text(edu_df)
                grupo_order = ["Casos confirmados", "Óbitos por meningite", "Óbitos por outra causa"]
                categoria_order = education_category_labels("SINAN", education, include_missing=True)
                edu_df = edu_df.sort_values(["ordem_escolaridade", "ordem_grupo", "grupo"]).reset_index(drop=True)
                fig_edu = px.bar(
                    edu_df,
                    x="n",
                    y="escolaridade",
                    color="grupo",
                    orientation="h",
                    barmode="group",
                    text="texto",
                    title="SINAN: escolaridade — confirmados e óbitos",
                    labels={
                        "escolaridade": "Escolaridade",
                        "n": "Registros",
                        "grupo": "Grupo",
                        "pct": "% dos casos confirmados",
                        "denominador": "Total de casos confirmados",
                    },
                    hover_data={"texto": False, "pct": ":.2f", "denominador": True},
                    category_orders={"escolaridade": categoria_order, "grupo": grupo_order},
                )
                fig_edu.update_layout(yaxis={"categoryorder": "array", "categoryarray": categoria_order[::-1]})
                st.caption("O gráfico exibe todas as categorias operacionais de escolaridade do SINAN; os percentuais usam o total de casos confirmados como denominador comum.")
                render_plotly_chart(fig_edu)
                render_interval_total(edu_df, value_col="n", by_col="grupo")
                edu_out = edu_df.drop(columns=["ordem_escolaridade", "ordem_grupo"], errors="ignore")
                copyable_dataframe(edu_out, width="stretch", hide_index=True)
                download_button(edu_out, "sinan_escolaridade_confirmados_obitos.csv")
    elif source == "SIM":
        st.markdown("### Escolaridade")
        if not education:
            st.info("Para gerar o gráfico de escolaridade no SIM, o campo ESC2010/ESC precisa existir e ser detectado automaticamente.")
        else:
            edu_df = query_education_distribution_all_categories(table, "SIM", education, graph_where)
            if edu_df.empty or pd.to_numeric(edu_df.get("denominador"), errors="coerce").fillna(0).max() <= 0:
                st.info("Sem dados de escolaridade no SIM com os filtros atuais.")
            else:
                edu_df = add_text(edu_df)
                categoria_order = education_category_labels("SIM", education, include_missing=True)
                edu_df = edu_df.sort_values("ordem_categoria").reset_index(drop=True)
                fig_edu = px.bar(
                    edu_df,
                    x="n",
                    y="categoria",
                    orientation="h",
                    text="texto",
                    title="SIM: distribuição por escolaridade",
                    labels={"categoria": "Escolaridade", "n": "Óbitos", "pct": "% do total filtrado", "denominador": "Total filtrado"},
                    hover_data={"texto": False, "pct": ":.2f", "denominador": True},
                    category_orders={"categoria": categoria_order},
                )
                fig_edu.update_layout(yaxis={"categoryorder": "array", "categoryarray": categoria_order[::-1]})
                st.caption("O gráfico exibe todas as categorias operacionais de escolaridade detectadas para o campo do SIM; os percentuais usam o total de registros filtrados como denominador.")
                render_plotly_chart(fig_edu)
                render_interval_total(edu_df, value_col="n")
                edu_out = edu_df.drop(columns=["ordem_categoria"], errors="ignore")
                copyable_dataframe(edu_out, width="stretch", hide_index=True)
                download_button(edu_out, "sim_escolaridade.csv")

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
            st.caption("Dicionário IBGE: os rótulos de município são lidos do CSV externo `municipios_ibge.csv` ou da URL configurada em Desempenho e memória; hospede esse CSV no GitHub como arquivo raw para manter o script sem dicionário embutido.")
        else:
            top_municipios = 15
        for label, expr, is_mun, top_n in cols:
            if is_mun:
                df = query_municipality_top(table, expr, graph_where, top_n=top_municipios)
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
                render_plotly_chart(fig)
                render_interval_total(df, value_col="n")
                copyable_dataframe(df, width="stretch", hide_index=True)
                download_button(df, filename)


def render_quality_tab(table: LoadedTable, source: str, base_where: str, exprs: Dict[str, Optional[str]]) -> None:
    st.markdown("### Campos importantes não preenchidos")
    st.caption(
        "Esta aba usa os filtros-base e mede, para cada campo-chave detectado, quantos registros estão sem preenchimento válido. "
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
            "CLA_ME_BAC": exprs.get("cla_me_bac_code"),
            "CLA_ME_ASS": exprs.get("cla_me_ass_code"),
            "CLA_ME_ETI": exprs.get("cla_me_eti_code"),
            "EVOLUCAO": exprs.get("evol_code"),
            "CRITERIO": exprs.get("criterio_code"),
            "Punção Laboratorial": exprs.get("puncao_code"),
            "Exame Quimiocitológico do líquor (LCR)": exprs.get("quimio_code"),
            "Hemácias": exprs.get("lab_hema"),
            "Neutrófilos": exprs.get("lab_neutro"),
            "Glicose": exprs.get("lab_glico"),
            "Leucócitos": exprs.get("lab_leuco"),
            "Eosinófilos": exprs.get("lab_eosi"),
            "Proteínas": exprs.get("lab_prot"),
            "Monócitos": exprs.get("lab_mono"),
            "Linfócitos": exprs.get("lab_linfo"),
            "Cloreto": exprs.get("lab_clor"),
        })
    elif source == "CIHA":
        fields.update({"MORTE": exprs.get("morte_code"), "DIAS_PERM": exprs.get("dias_perm"), "MODALIDADE": exprs.get("modalidade_label"), "PROCEDIMENTO": exprs.get("procedimento_label")})
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
        st.info("Sem campos detectados automaticamente para avaliar preenchimento.")
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
        render_plotly_chart(fig)
        st.caption("Total no intervalo filtrado: " + format_int_br(pd.to_numeric(miss["total"], errors="coerce").max()) + " registros analisados; faltantes são contados por campo.")
        copyable_dataframe(miss[["campo", "faltantes", "total", "pct_faltante", "texto"]], width="stretch", hide_index=True)
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
        render_plotly_chart(fig)
        render_interval_total(filtered, value_col="faltantes", by_col="campo", value_label="registros não preenchidos")
        copyable_dataframe(filtered[["ano", "campo", "faltantes", "total", "pct_faltante", "texto"]], width="stretch", hide_index=True)
        download_button(by_year.drop(columns=["texto"], errors="ignore"), f"{source.lower()}_campos_importantes_nao_preenchidos_por_ano.csv")

def render_sql_lab(table: LoadedTable, source: str) -> None:
    st.markdown("### Laboratório SQL")
    st.caption("Use `{{tabela}}` como placeholder para a tabela carregada. O app substituirá pelo nome/referência SQL correta.")
    example = """
    SELECT COUNT(*) AS registros
    FROM {tabela};
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
    sql_limit = perf_int("perf_sql_lab_row_limit", DEFAULT_SQL_LAB_ROW_LIMIT)
    st.caption(f"O resultado do SQL Lab será limitado a {sql_limit:,} linhas.".replace(",", "."))
    if st.button("Executar SQL", key=f"run_sql_{source}"):
        sql = sql_text.replace("{tabela}", table.ref_sql).replace("{{tabela}}", table.ref_sql)
        sql_clean = sql.strip().rstrip(";")
        if not re.match(r"^(SELECT|WITH)\b", sql_clean, flags=re.IGNORECASE):
            st.error("Por segurança e desempenho, o SQL Lab aceita apenas consultas SELECT/WITH.")
            return
        try:
            limited_sql = f"SELECT * FROM ({sql_clean}) AS _sql_lab_result LIMIT {int(sql_limit)}"
            df = run_query(table, limited_sql, cache=False)
            copyable_dataframe(df, width="stretch", hide_index=True)
            download_button(df, f"{source.lower()}_sql_lab.csv", "Baixar resultado", max_rows=sql_limit)
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

    analysis_sections = [
        "Principais indicadores epidemiológicos",
        "Temporal",
        "Análise etiológica e CID-10",
        "Demografia e território",
    ]
    if source == "SINAN":
        analysis_sections.append("Sobreposição NU_NOTIFIC")
    analysis_sections.extend([
        "Campos importantes não preenchidos",
        "Prévia",
        "SQL Lab",
    ])
    analysis_section_key = f"analysis_section_{source}"
    if st.session_state.get(analysis_section_key) not in (None, *analysis_sections):
        st.session_state.pop(analysis_section_key, None)
    selected_section = st.radio(
        "Área de análise",
        analysis_sections,
        horizontal=True,
        key=analysis_section_key,
        help="Somente a área selecionada é calculada nesta execução para reduzir memória e tempo de rerun.",
    )

    if selected_section == "Principais indicadores epidemiológicos":
        render_indicators_tab(table, source, base_where, graph_where, exprs)
    elif selected_section == "Temporal":
        render_temporal_tab(table, source, graph_where, exprs)
    elif selected_section == "Análise etiológica e CID-10":
        render_cid_tab(table, source, graph_where, exprs)
    elif selected_section == "Demografia e território":
        render_demography_tab(table, source, graph_where, exprs, base_where=base_where)
    elif selected_section == "Sobreposição NU_NOTIFIC" and source == "SINAN":
        render_sinan_overlap_tab(table, base_where, exprs)
    elif selected_section == "Campos importantes não preenchidos":
        render_quality_tab(table, source, base_where, exprs)
    elif selected_section == "Prévia":
        st.markdown("### Prévia enriquecida")
        total_preview = count_rows(table, graph_where)
        max_preview_rows = max(50, perf_int("perf_max_preview_rows", DEFAULT_MAX_PREVIEW_ROWS))
        default_preview = min(DEFAULT_PREVIEW_ROW_LIMIT, max_preview_rows)
        page_size = st.slider(
            "Linhas por página",
            50,
            int(max_preview_rows),
            int(default_preview),
            step=50,
            key=f"preview_limit_{source}",
        )
        max_page = max(1, int(np.ceil(total_preview / page_size))) if page_size else 1
        page = st.number_input("Página", min_value=1, max_value=max_page, value=1, step=1, key=f"preview_page_{source}")
        offset = (int(page) - 1) * int(page_size)
        st.caption(
            f"Exibindo página {int(page):,} de {max_page:,}; total filtrado: {total_preview:,} registros."
            .replace(",", ".")
        )
        try:
            df_prev = query_enriched_preview(table, sel, exprs, graph_where, int(page_size), offset=offset)
            copyable_dataframe(df_prev, width="stretch")
            download_button(df_prev, f"{source.lower()}_previa_enriquecida_pagina_{int(page)}.csv", max_rows=int(page_size))
        except Exception as exc:
            st.error(f"Erro ao montar prévia: {exc}")

        st.markdown("### Exportação completa dos casos filtrados")
        full_export_limit = perf_int("perf_full_export_row_limit", DEFAULT_FULL_EXPORT_ROW_LIMIT)
        if total_preview > full_export_limit:
            st.warning(
                f"Exportação completa bloqueada: {total_preview:,} registros excedem o limite atual de {full_export_limit:,}. "
                "Aplique filtros adicionais ou aumente o limite em Desempenho e memória se o ambiente suportar."
                .replace(",", ".")
            )
        else:
            st.caption(
                "A exportação completa é habilitada somente quando o total filtrado está dentro do limite defensivo configurado."
            )
            if st.button("Gerar CSV completo dos casos filtrados", key=f"full_export_{source}"):
                try:
                    df_full = query_enriched_preview(table, sel, exprs, graph_where, limit=None)
                    st.success(f"Exportação preparada com {len(df_full):,} linhas.".replace(",", "."))
                    download_button(
                        df_full,
                        f"{source.lower()}_casos_filtrados_completos.csv",
                        "Baixar CSV completo",
                        max_rows=max(1, len(df_full)),
                    )
                except Exception as exc:
                    st.error(f"Erro ao gerar exportação completa: {exc}")
    elif selected_section == "SQL Lab":
        render_sql_lab(table, source)

    context = {"source": source, "table": table, "sel": sel, "exprs": exprs, "base_where": base_where, "graph_where": graph_where, "definition": definition}
    st.session_state[f"loaded_context_{source}"] = context
    return context


def render_comparison(loaded: Sequence[Dict[str, object]]) -> None:
    st.markdown("### Comparação de bancos de dados")
    available = [x for x in loaded if x and x.get("exprs", {}).get("dt")]
    if len(available) < 2:
        st.info("Carregue ao menos duas bases com data detectada para comparar séries.")
        return
    source_names = [x["source"] for x in available]
    chosen = st.multiselect("Bases", source_names, default=source_names, key="comp_sources")
    freq_label = st.selectbox("Agregação", ["Ano", "Mês", "Semana"], index=1, key="comp_freq")
    freq = {"Ano": "year", "Mês": "month", "Semana": "week"}[freq_label]
    normalize = st.checkbox("Normalizar em índice 100 no primeiro período não-zero", value=False, key="comp_norm")
    stratify_cid = st.checkbox("Estratificar por tipo CID-10 quando disponível", value=False, key="comp_cid")
    st.caption("Na comparação, o SINAN entra sempre como casos confirmados (CLASSI_FIN = 1), independentemente da definição exploratória escolhida na aba SINAN. Quando há estratificação por CID-10, o SINAN usa a conversão de CON_DIAGES; SIM/CIHA usam os mesmos CID-10 adequados prefixados do gráfico de conversão: os códigos convertidos somam no destino e os CID-10 prefixados já presentes permanecem em seu próprio grupo. Na agregação mensal, meses sem registros são mantidos com valor zero.")

    frames = []
    comparison_conversion_notes: List[str] = []
    for item in available:
        source_name = item["source"]
        if source_name not in chosen:
            continue
        table: LoadedTable = item["table"]
        exprs = item["exprs"]
        if stratify_cid:
            if source_name == "SINAN" and exprs.get("sinan_cid10_conversion_type"):
                cat = exprs.get("sinan_cid10_conversion_type")
            elif source_name in {"SIM", "CIHA"} and exprs.get("cid10_adequacy_plot_label"):
                cat = exprs.get("cid10_adequacy_plot_label")
            else:
                cat = exprs.get("cid_type")
        else:
            cat = None
        series_where = item["graph_where"]
        series_label = item.get("definition", "")
        if source_name == "SINAN":
            classi = exprs.get("classi_code")
            if not classi:
                st.warning("SINAN foi ignorado na comparação porque CLASSI_FIN não foi detectado automaticamente; não é possível isolar confirmados.")
                continue
            series_where = append_clause(item["base_where"], f"{classi} = '1'")
            series_label = "Confirmados (CLASSI_FIN = 1)"
        try:
            ts = query_timeseries(table, exprs["dt"], series_where, freq, cat)
        except Exception as exc:
            st.warning(f"Falha na série de {source_name}: {exc}")
            continue
        if stratify_cid and source_name in {"SIM", "CIHA"} and exprs.get("cid10_adequacy_plot_label"):
            try:
                conv_note_df = query_cid10_adequacy_conversion(table, exprs, series_where)
                if not conv_note_df.empty:
                    comparison_conversion_notes.append(f"{source_name}: {build_cid10_adequacy_conversion_note(conv_note_df)}")
            except Exception as exc:
                comparison_conversion_notes.append(f"{source_name}: não foi possível calcular a observação de conversão ({exc}).")
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

    fig = px.line(comp, x="periodo", y="valor", color="serie", markers=True, title="Comparação de bancos de dados — tendências", labels={"valor": "Índice" if normalize else "Registros", "periodo": "Período", "serie": "Série"})
    render_plotly_chart(fig)
    if not normalize:
        render_interval_total(comp, value_col="valor", by_col="serie")
    if comparison_conversion_notes:
        st.caption("Observação da conversão usada na comparação estratificada: " + " ".join(comparison_conversion_notes))
    copyable_dataframe(comp, width="stretch", hide_index=True)
    download_button(comp, "comparacao_series_bases.csv")

    st.markdown("**Cuidados de leitura**")
    st.write(
        "SINAN mede notificações/investigações; SIM mede óbitos; CIHA mede utilização de serviços. "
        "Compare tendências, composição e concordância agregada, mas evite interpretar contagens brutas entre bases como o mesmo fenômeno sem linkage e denominadores populacionais."
    )


def render_methodology():
    st.divider() -> None:
    st.markdown("### Como usar este app para investigação epidemiológica")
    st.markdown(
        """
        1. Comece pela aba **Principais indicadores epidemiológicos** do SINAN para separar notificações, confirmados, descartados e óbitos.
        2. Use **Análise etiológica e CID-10** para comparar o CID bruto com a classificação específica. No SINAN, dê prioridade a `CON_DIAGES`, `CLA_ME_BAC`, `CLA_ME_ASS` e `CLA_ME_ETI`.
        3. Use **Sobreposição NU_NOTIFIC** no SINAN para verificar repetição do número de notificação e levantar possíveis duplicidades.
        4. Use **Temporal** para verificar queda, recuperação e sazonalidade.
        5. Use **Demografia e território** para levantar hipóteses por idade, sexo, residência e atendimento.
        6. Use **Prévia** para inspecionar casos filtrados e exportar a planilha completa quando necessário.
        7. Use **SQL Lab** para transformar a hipótese em uma consulta reprodutível.
        """
    )
    st.info(
        "Adendo de atualização: espere a consolidação dos bancos antes de interpretar anos mais recentes como definitivos. "
        "SINAN pode mudar após investigação/encerramento do caso; SIM pode mudar após codificação e qualificação da causa básica; CIHA pode sofrer recomposição por competência e processamento administrativo."
    )
    st.markdown("### Dicionário externo de municípios IBGE")
    st.write(
        "O dicionário nacional de municípios foi retirado do script Python e deve ser mantido como CSV externo. "
        "Hospede `municipios_ibge.csv` no GitHub e informe a URL raw no controle **CSV externo de municípios IBGE**, ou mantenha o arquivo no mesmo diretório do script."
    )
    st.caption(municipios_ibge_csv_status())
    st.markdown("### Guia de campos por base")
    guide_rows = [
        {"Base": base, "Campo": field, "Papel": role, "Uso no painel": use}
        for base, rows in FIELD_GUIDE.items()
        for field, role, use in rows
    ]
    copyable_dataframe(pd.DataFrame(guide_rows), width="stretch", hide_index=True)
    st.markdown("### Definições principais usadas")
    copyable_dataframe(
        pd.DataFrame(
            [
                ["SINAN — notificação", "todos os registros após filtros"],
                ["SINAN — confirmado", "CLASSI_FIN = 1"],
                ["SINAN — descartado", "CLASSI_FIN = 2"],
                ["SINAN — óbito por meningite", "CLASSI_FIN = 1 e EVOLUCAO = 2"],
                ["SIM — causa básica", "CID de meningite/encefalite detectado em CAUSABAS"],
                ["SIM — menção", "CID de meningite/encefalite detectado em CAUSABAS, linhas da DO ou ATESTADO"],
                ["SIM — óbito na gravidez", "campo OBITOGRAV quando disponível"],
                ["CIHA — atendimento", "registro administrativo com data/diagnóstico, incluindo os novos CID-10 de meningite/encefalite quando detectados"],
                ["CIHA — morte administrativa", "MORTE = 1"],
                ["CIHA — permanência zero", "DIAS_PERM = 0"],
            ],
            columns=["Conceito", "Regra operacional"],
        ),
        width="stretch",
        hide_index=True,
    )
    st.markdown("### Referência CID-10")
    render_cid_reference()
    render_quimio_interpretation()


def main() -> None:
    render_app_css()
    st.title("Painel epidemiológico de meningite — SINAN, SIM e CIHA")
    st.caption(f"Versão {APP_VERSION}. Lê upload de DuckDB, upload de Parquet ou bancos hospedados no github em Parquet e mantém regras analíticas explícitas.")

    with st.sidebar:
        render_performance_controls()
        main_sections = ["Metodologia", "SINAN", "SIM", "CIHA", "Comparação de bancos de dados"]
        main_section_key = "main_section"
        if st.session_state.get(main_section_key) not in (None, *main_sections):
            st.session_state.pop(main_section_key, None)
        section = st.radio(
            "Seção",
            main_sections,
            key=main_section_key,
        )

    if section in {"SINAN", "SIM", "CIHA"}:
        st.divider()
    render_source(section)
    elif section == "Metodologia":
        render_methodology()
    st.divider()
    else:
        loaded = [
            st.session_state.get(f"loaded_context_{src}")
            for src in ["SINAN", "SIM", "CIHA"]
            if st.session_state.get(f"loaded_context_{src}")
        ]
        st.caption(
            "A Comparação de bancos de dados usa as bases já carregadas nas seções SINAN/SIM/CIHA. "
            "Carregue cada base separadamente antes de comparar para evitar sobrecarga."
        )
        render_comparison([x for x in loaded if x])
    st.divider()


if __name__ == "__main__":
    main()
