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

import csv
import io
import re
import unicodedata
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


# Palavras-chave (sem acento) pra achar colunas em cabeçalho bagunçado de extrato. Uso radicais
# ('hist', 'descr') pra casar variações; 'lanc' fica FORA de descrição porque casaria com a
# própria coluna "Data Lançamento".
_KW_VALOR = ("valor", "value", "amount", "montante")
_KW_DESCRICAO = ("hist", "descr", "estabelec", "memo", "detalhe")
_KW_DATA = ("data", "date")
_RE_DATA = re.compile(r"^\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s*$")
# Inteiro com separador de milhar opcional (48, -784, 2.042) — usado pra detectar valor partido.
_RE_INTEIRO = re.compile(r"^[+-]?\d{1,3}(?:\.\d{3})+$|^[+-]?\d+$")
_RE_CENTAVOS = re.compile(r"^\d{1,2}$")


def _sem_acento(texto: object) -> str:
    s = unicodedata.normalize("NFKD", str(texto))
    return "".join(c for c in s if not unicodedata.combining(c)).strip().lower()


def _bytes_de(arquivo) -> bytes:
    if hasattr(arquivo, "getvalue"):       # UploadedFile do Streamlit / BytesIO
        return arquivo.getvalue()
    if hasattr(arquivo, "read"):           # file handle aberto
        return arquivo.read()
    return arquivo                          # já são bytes


def _decodificar(dados: bytes) -> str | None:
    for enc in ("utf-8-sig", "latin-1"):    # latin-1 sempre decodifica; utf-8-sig engole BOM
        try:
            return dados.decode(enc)
        except UnicodeDecodeError:
            continue
    return None


def _tem_essenciais(df: pd.DataFrame) -> bool:
    """True se, após mapear sinônimos, o df tem ao menos descrição e valor."""
    return {"descricao", "valor"}.issubset(mapear_colunas(df).columns)


def _ler_tabular(dados: bytes) -> pd.DataFrame | None:
    """CSV 'bem-comportado': descobre delimitador (`;`/`,`) e encoding. Devolve o melhor df lido,
    priorizando um que já tenha as colunas essenciais."""
    melhor = None
    for encoding in ("utf-8-sig", "latin-1"):
        for sep in (None, ";", ","):        # sep=None -> csv.Sniffer adivinha
            try:
                df = pd.read_csv(io.BytesIO(dados), sep=sep, engine="python", encoding=encoding)
            except Exception:  # noqa: BLE001 - separador/encoding errado: tenta o próximo
                continue
            if df.shape[1] < 2:
                continue
            if _tem_essenciais(df):
                return df
            melhor = melhor if melhor is not None else df
    return melhor


def _combina_valor_partido(inteiro: str, centavos: str) -> float | None:
    """Junta '48' + '42' -> 48.42 (o decimal virou coluna por causa da vírgula-delimitadora)."""
    inteiro, centavos = inteiro.strip(), centavos.strip()
    sinal = "-" if inteiro.startswith("-") else ""
    digitos = inteiro.lstrip("+-").replace(".", "")     # tira separador de milhar
    if not digitos.isdigit() or not centavos.isdigit():
        return None
    return float(f"{sinal}{digitos}.{centavos}")


def _ler_extrato_br(dados: bytes) -> pd.DataFrame | None:
    """Extrato BR 'sujo': preâmbulo antes do cabeçalho e/ou valor partido pela vírgula-decimal.

    Tenta `;` antes de `,`: é o delimitador típico de export de banco BR (a vírgula fica livre
    pro decimal). Com `;` errado num arquivo de vírgulas, a linha vira célula única e a checagem
    de data descarta tudo — então a ordem é segura, não silenciosamente errada.
    """
    texto = _decodificar(dados)
    if texto is None:
        return None
    for delim in (";", ","):
        df = _parsear_linhas_extrato(list(csv.reader(texto.splitlines(), delimiter=delim)))
        if df is not None:
            return df
    return None


def _parsear_linhas_extrato(linhas: list[list[str]]) -> pd.DataFrame | None:
    """Acha o cabeçalho (linha com coluna de valor E de descrição), localiza as colunas por
    palavra-chave, detecta se o valor está partido (coluna inteira seguida de 1-2 dígitos) e
    remonta. Pula tudo que não começa com uma data — descarta preâmbulo e rodapé.
    """
    cabecalho_idx = idx_valor = idx_desc = idx_data = None
    for i, linha in enumerate(linhas):
        celulas = [_sem_acento(c) for c in linha]
        achar = lambda kws: next((j for j, c in enumerate(celulas)
                                  if any(k in c for k in kws)), None)
        iv, idesc, idata = achar(_KW_VALOR), achar(_KW_DESCRICAO), achar(_KW_DATA)
        if iv is not None and idesc is not None:        # achou o cabeçalho real
            cabecalho_idx, idx_valor, idx_desc, idx_data = i, iv, idesc, idata
            break
    if cabecalho_idx is None:
        return None

    dados_linhas = [ln for ln in linhas[cabecalho_idx + 1:]
                    if len(ln) > idx_valor and (idx_data is None
                                                or _RE_DATA.match(ln[idx_data] if idx_data < len(ln) else ""))]
    if not dados_linhas:
        return None

    # Valor partido? Maioria das linhas tem inteiro em idx_valor e centavos (1-2 díg.) na seguinte.
    partido = sum(
        bool(_RE_INTEIRO.match(ln[idx_valor].strip())
             and idx_valor + 1 < len(ln) and _RE_CENTAVOS.match(ln[idx_valor + 1].strip()))
        for ln in dados_linhas
    ) > len(dados_linhas) * 0.6

    registros = []
    for ln in dados_linhas:
        data = ln[idx_data].strip() if idx_data is not None and idx_data < len(ln) else ""
        # Descrição = todas as colunas de texto entre a descrição e o valor (histórico + lojista).
        fim_texto = idx_valor if idx_valor > idx_desc else idx_desc + 1
        descricao = " ".join(ln[j].strip() for j in range(idx_desc, min(fim_texto, len(ln)))
                             if ln[j].strip())
        if partido:
            valor = (_combina_valor_partido(ln[idx_valor], ln[idx_valor + 1])
                     if idx_valor + 1 < len(ln) else None)
        else:
            valor = parsear_valor(pd.Series([ln[idx_valor]])).iloc[0]
        if valor is not None and pd.notna(valor):
            registros.append((data, descricao, float(valor)))

    return pd.DataFrame(registros, columns=["data", "descricao", "valor"]) if registros else None


def ler_csv_upload(arquivo) -> pd.DataFrame:
    """Lê o CSV do upload em duas camadas: primeiro como CSV normal; se não der, como extrato
    BR 'sujo' (preâmbulo + valor partido). Reusa `preparar_dados` pra validação final."""
    dados = _bytes_de(arquivo)

    df = _ler_tabular(dados)
    if df is not None and _tem_essenciais(df):
        return df

    extrato = _ler_extrato_br(dados)
    if extrato is not None:
        return extrato

    if df is not None:           # leu algo, mas sem as colunas certas: erro de coluna (claro)
        return df
    raise ValueError(
        "Não consegui interpretar o arquivo como CSV. Confira se é um CSV de extrato com "
        "colunas como Data, Histórico/Descrição e Valor."
    )


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
    # Linha de saldo ("Saldo do dia", "Saldo Anterior", "S A L D O") não é transação: alguns
    # bancos a intercalam no extrato e, como saldo é sempre alto pro padrão da conta, o detector
    # de atípicos flagava todas (apontado por usuário em teste real). Comparo sem acento e sem
    # espaço porque há banco que grafa "S A L D O" espaçado.
    eh_saldo = (
        df["descricao"].map(_sem_acento).str.lower().str.replace(" ", "").str.startswith("saldo")
    )
    df = df[~eh_saldo]
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
        # Fixo fundo E texto: no tema escuro do Streamlit o texto padrão é claro e sumiria
        # sobre esses fundos pastéis (pares de contraste do Bootstrap, legíveis nos dois temas).
        i = row.name
        if df.loc[i, "eh_anomalia"]:
            return ["background-color: #f8d7da; color: #842029"] * len(row)   # vermelho claro
        if df.loc[i, "confianca"] < LIMIAR_CONFIANCA:
            return ["background-color: #fff3cd; color: #664d03"] * len(row)   # âmbar
        return [""] * len(row)

    return (
        vis.style.apply(_cor, axis=1)
        .format({"Valor (R$)": _brl, "Confiança": "{:.0%}"})
    )


def main() -> None:
    st.set_page_config(page_title="gastei-onde", page_icon="💸", layout="wide")
    st.title("gastei-onde 💸")
    st.caption("Sobe o extrato, eu categorizo cada transação e aponto os gastos fora da curva.")

    # Banner sempre visível (não atrás de clique): quem sobe extrato real precisa ler isso
    # ANTES de decidir subir — é dado financeiro pessoal, e o compromisso é parte do produto.
    st.info(
        "🔒 **Privacidade:** seu extrato é processado 100% em memória, só nesta sessão. "
        "Nada é salvo em disco, nada é enviado a terceiros — fechou a aba, os dados se foram.",
    )

    fonte = st.radio(
        "De onde vêm os dados?",
        ["Usar dados de exemplo", "Subir meu CSV"],
        horizontal=True,
    )

    df_bruto = None
    if fonte == "Usar dados de exemplo":
        df_bruto = carregar_exemplo()
        st.info(
            "Usando um extrato **sintético** de exemplo (nenhum dado real) — bom pra "
            "conhecer o app antes de subir o seu."
        )
    else:
        arquivo = st.file_uploader("CSV do extrato (colunas como Data, Histórico, Valor)", "csv")
        if arquivo is not None:
            try:
                df_bruto = ler_csv_upload(arquivo)
            except ValueError as e:  # CSV corrompido / não-CSV / ilegível
                st.error(str(e))
                st.stop()

    if df_bruto is None:
        # Estado vazio do upload: orienta em vez de só esperar — o público-alvo não é técnico.
        st.markdown(
            "👆 No app ou site do seu banco, procure **exportar extrato** e escolha o formato "
            "**CSV** (não PDF). Depois é só arrastar o arquivo aqui — aceito os layouts mais "
            "comuns, mesmo com cabeçalho bagunçado. Se quiser só espiar antes, use os dados "
            "de exemplo ali em cima."
        )
        st.stop()

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
    else:
        # Extrato só com entradas (ex.: conta-salário): sem gasto não há gráfico — explica
        # em vez de sumir com a seção.
        st.caption("Nenhum gasto (valor negativo) no arquivo — o gráfico por categoria fica de fora.")

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
