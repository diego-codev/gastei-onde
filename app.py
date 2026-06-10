"""App Streamlit do gastei-onde: sobe um extrato, categoriza e destaca gastos atípicos.

Decisões de arquitetura:
- O modelo é **treinado no startup** (cacheado com `@st.cache_resource`) a partir do CSV
  sintético versionado — sem artefato `.joblib` no repo. Treinar é rápido e garante que o
  modelo é sempre reproduzível a partir de código + dados versionados.
- A categorização e a anomalia reusam exatamente o código de `src/` (mesma transformação do
  treino), pra o que roda aqui ser o que foi validado nos notebooks.
- Processamento **100% em memória**: o CSV do usuário nunca é salvo nem enviado a lugar nenhum.

Rodar localmente:  streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.models.anomaly import marcar_anomalias
from src.models.train import prever_com_confianca, treinar

CSV_EXEMPLO = Path(__file__).parent / "data/sample/extrato_exemplo.csv"

# Limiar de confiança abaixo do qual o app sugere revisão manual (ver notebook 02: os erros
# do modelo se concentram na baixa confiança). É deliberadamente conservador.
LIMIAR_CONFIANCA = 0.60

# Mapeamento de nomes de coluna de exportações reais -> esquema canônico (data/descricao/valor).
# Bancos exportam com cabeçalhos variados; aceito os mais comuns pra não exigir CSV "perfeito".
SINONIMOS_COLUNAS: dict[str, tuple[str, ...]] = {
    "data": ("data", "date", "data lancamento", "data do lancamento", "data mov"),
    "descricao": ("descricao", "descrição", "historico", "histórico", "lancamento",
                  "lançamento", "estabelecimento", "memo", "detalhe"),
    "valor": ("valor", "value", "amount", "valor (r$)", "valor r$", "montante"),
}


def mapear_colunas(df: pd.DataFrame) -> pd.DataFrame:
    """Renomeia colunas conhecidas pro esquema canônico (case/acento-insensível no cabeçalho)."""
    normaliza = lambda s: str(s).strip().lower()
    atual = {normaliza(c): c for c in df.columns}
    renomear = {}
    for alvo, sinonimos in SINONIMOS_COLUNAS.items():
        for s in sinonimos:
            if s in atual:
                renomear[atual[s]] = alvo
                break
    return df.rename(columns=renomear)


def parsear_valor(serie: pd.Series) -> pd.Series:
    """Converte a coluna de valor pra float, tolerando formato BR (R$ 1.234,56) e sinais."""
    if pd.api.types.is_numeric_dtype(serie):
        return serie.astype(float)

    def _um(v: object) -> float:
        t = str(v).strip().replace("R$", "").replace(" ", "")
        if not t or t.lower() in {"nan", "none"}:
            return float("nan")
        # Se tem ',' e '.', o último separador é o decimal (BR: '.'=milhar; intl: ','=milhar).
        if "," in t and "." in t:
            if t.rfind(",") > t.rfind("."):       # 1.234,56 -> BR
                t = t.replace(".", "").replace(",", ".")
            else:                                  # 1,234.56 -> intl
                t = t.replace(",", "")
        elif "," in t:                             # só vírgula: decimal BR
            t = t.replace(",", ".")
        try:
            return float(t)
        except ValueError:
            return float("nan")

    return serie.map(_um)


def preparar_dados(df_bruto: pd.DataFrame) -> pd.DataFrame:
    """Valida e normaliza o CSV cru pro esquema canônico. Levanta ValueError se faltar coluna.

    `data` é opcional (alguns extratos não trazem); `descricao` e `valor` são obrigatórios —
    sem eles não há o que categorizar.
    """
    df = mapear_colunas(df_bruto)
    faltando = [c for c in ("descricao", "valor") if c not in df.columns]
    if faltando:
        raise ValueError(
            f"Não encontrei as colunas {faltando}. O CSV precisa ter ao menos descrição e "
            f"valor (aceito nomes comuns como Histórico/Valor). Colunas vistas: "
            f"{list(df_bruto.columns)}"
        )
    df = df.copy()
    df["valor"] = parsear_valor(df["valor"])
    df = df.dropna(subset=["valor"])               # linhas sem valor legível não entram
    df["descricao"] = df["descricao"].fillna("").astype(str)
    if "data" not in df.columns:
        df["data"] = ""
    return df.reset_index(drop=True)


def classificar(df: pd.DataFrame, modelo) -> pd.DataFrame:
    """Aplica modelo + anomalia. A anomalia é medida DENTRO da categoria PREVISTA pelo modelo."""
    pred = prever_com_confianca(modelo, df)
    out = df.copy()
    out["categoria"] = pred["categoria_prevista"].to_numpy()
    out["confianca"] = pred["confianca"].to_numpy()
    out = marcar_anomalias(out, col_categoria="categoria")  # usa a categoria prevista
    return out


@st.cache_resource(show_spinner="Treinando o modelo (uma vez por sessão)...")
def carregar_modelo():
    """Treina o pipeline sobre o CSV sintético versionado. Cacheado: roda uma vez por sessão."""
    df = pd.read_csv(CSV_EXEMPLO)
    return treinar(df)


@st.cache_data
def carregar_exemplo() -> pd.DataFrame:
    """CSV de exemplo cru (cacheado) pro botão 'usar dados de exemplo'."""
    return pd.read_csv(CSV_EXEMPLO)


# --------------------------------------------------------------------------- UI


def _brl(x: float) -> str:
    """Formata em reais no padrão BR (1.234,56), que `{:,.2f}` (padrão US) não dá."""
    return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _formatar_tabela(df: pd.DataFrame):
    """Monta a tabela exibida, destacando anomalias (vermelho) e baixa confiança (âmbar)."""
    vis = pd.DataFrame(
        {
            "Data": df["data"],
            "Descrição": df["descricao"],
            "Valor (R$)": df["valor"],
            "Categoria": df["categoria"],
            "Confiança": df["confianca"],
            "Atípico": df["eh_anomalia"].map({True: "⚠️", False: ""}),
        }
    )

    def _cor(row):
        i = row.name
        if df.loc[i, "eh_anomalia"]:
            return ["background-color: #f8d7da"] * len(row)        # vermelho claro
        if df.loc[i, "confianca"] < LIMIAR_CONFIANCA:
            return ["background-color: #fff3cd"] * len(row)        # âmbar
        return [""] * len(row)

    return (
        vis.style.apply(_cor, axis=1)
        .format({"Valor (R$)": _brl, "Confiança": "{:.0%}"})
    )


def main() -> None:
    st.set_page_config(page_title="gastei-onde", page_icon="💸", layout="wide")
    st.title("gastei-onde 💸")
    st.caption("Sobe o extrato, eu categorizo cada transação e aponto os gastos fora da curva.")

    fonte = st.radio(
        "De onde vêm os dados?",
        ["Usar dados de exemplo", "Subir meu CSV"],
        horizontal=True,
    )

    df_bruto = None
    if fonte == "Usar dados de exemplo":
        df_bruto = carregar_exemplo()
        st.info("Usando um extrato **sintético** de exemplo (nenhum dado real).")
    else:
        arquivo = st.file_uploader("CSV do extrato (colunas como Data, Histórico, Valor)", "csv")
        if arquivo is not None:
            try:
                df_bruto = pd.read_csv(arquivo)
            except Exception as e:  # CSV corrompido / não-CSV
                st.error(f"Não consegui ler o arquivo como CSV: {e}")
                st.stop()

    if df_bruto is None:
        st.stop()  # ainda sem dados: espera o upload

    try:
        df = preparar_dados(df_bruto)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    if df.empty:
        st.warning("O arquivo não tem nenhuma transação com valor legível.")
        st.stop()

    resultado = classificar(df, carregar_modelo())

    # --- Resumo ---------------------------------------------------------------
    gastos = resultado[resultado["valor"] < 0]
    entradas = resultado[resultado["valor"] >= 0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transações", len(resultado))
    c2.metric("Total gasto", f"R$ {_brl(gastos['valor'].abs().sum())}")
    c3.metric("Total recebido", f"R$ {_brl(entradas['valor'].sum())}")
    c4.metric("Gastos atípicos", int(resultado["eh_anomalia"].sum()))

    # --- Dashboard por categoria ---------------------------------------------
    if not gastos.empty:
        por_cat = (
            gastos.assign(gasto=gastos["valor"].abs())
            .groupby("categoria")["gasto"].sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        fig = px.bar(por_cat, x="categoria", y="gasto", title="Gastos por categoria")
        fig.update_layout(xaxis_title="", yaxis_title="R$")
        st.plotly_chart(fig, width="stretch")

    # --- Tabela detalhada -----------------------------------------------------
    n_baixa = int((resultado["confianca"] < LIMIAR_CONFIANCA).sum())
    st.subheader("Transações")
    st.caption(
        f"⚠️ = gasto atípico para a categoria · linhas em âmbar = baixa confiança "
        f"(< {LIMIAR_CONFIANCA:.0%}), vale revisar ({n_baixa} transações)."
    )
    st.dataframe(_formatar_tabela(resultado), width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
