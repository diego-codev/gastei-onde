"""Classificador multiclasse: TF-IDF (n-grama de caractere) + features numéricas -> LogReg.

Por que este modelo existe, se o baseline por regras já acerta ~94%? O dado sintético é
**circular**: as descrições nascem das mesmas keywords que as regras consultam, então
in-distribution o ML apenas EMPATA com as regras — e empate não justifica ML.

O valor real aparece na **generalização** para grafias que as regras NÃO catalogaram. Extrato
de verdade escreve o mesmo lojista de N formas ('IFOOD', 'IFD', 'IFOOOD'); a regra por keyword
exata erra a variante não prevista, enquanto o n-grama de caractere a reconhece pela
sobreposição parcial ('ifoood' compartilha 'ifoo', 'food' com 'ifood'). É isso que
`experimento_generalizacao` mede — e é o argumento central do projeto.

Limite honesto (`experimento_limite`): marca **totalmente** inédita não tem sinal de texto pra
ninguém; aí o ML cai ao nível do acaso, sustentado só pelo valor. Reconhecer onde o modelo
NÃO ajuda é parte da história.

Escolhas de modelagem:
- **LogReg**: forte em texto esparso de alta dimensão, interpretável, e com `predict_proba`
  que vira confiança calibrável (Etapa 7). Árvores/boosting raramente ganham em TF-IDF
  esparso e perdem em interpretabilidade.
- **class_weight="balanced"**: saúde/educação são raras (ver EDA). Sem reponderar, o modelo
  maximizaria accuracy ignorando as classes raras — justamente as de maior custo de erro.
- **ColumnTransformer**: junta o TF-IDF da descrição com as features do valor num único
  Pipeline, garantindo que treino e app apliquem exatamente a mesma transformação.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.data.generate import CATEGORIAS, gerar_transacoes
from src.features.text import construir_vetorizador_texto, features_numericas
from src.models.rules import classificar_lote

# Colunas brutas que o pipeline consome. Mantê-las como DataFrame (não arrays soltos) é o que
# permite o ColumnTransformer endereçar cada feature por nome — e o app passar o mesmo formato.
COLUNAS_X = ["descricao", "valor"]


def construir_pipeline() -> Pipeline:
    """Monta o pipeline TF-IDF (descrição) + features numéricas (valor) -> LogReg."""
    pre = ColumnTransformer(
        transformers=[
            # TF-IDF char_wb sobre a descrição: o vetorizador já normaliza via preprocessor.
            ("texto", construir_vetorizador_texto(), "descricao"),
            # FunctionTransformer reusa exatamente o `features_numericas` da feature eng.,
            # então não há divergência entre o que o notebook valida e o que o modelo treina.
            ("num", FunctionTransformer(features_numericas), ["valor"]),
        ],
    )
    # C=10: o espaço char-n-gram é grande e esparso; um pouco mais de folga que o default
    # (C=1) deixa as features de texto se expressarem sem overfitting perceptível (in-distrib.
    # e generalização sobem juntas — sinal de que não é memorização).
    # Não passo `multi_class`: o lbfgs já resolve multinomial e o parâmetro saiu do sklearn.
    clf = LogisticRegression(class_weight="balanced", max_iter=2000, C=10.0)
    return Pipeline([("features", pre), ("clf", clf)])


def treinar(df: pd.DataFrame) -> Pipeline:
    """Treina o pipeline na base inteira e devolve o modelo ajustado (uso do app/experimentos)."""
    pipe = construir_pipeline()
    pipe.fit(df[COLUNAS_X], df["categoria"])
    return pipe


def avaliar_holdout(df: pd.DataFrame, seed: int = 42, test_size: float = 0.25) -> dict:
    """Split estratificado: mede o ML in-distribution e compara com o baseline por regras.

    Estratificado de propósito — com classes desbalanceadas, um split aleatório puro poderia
    deixar saúde/educação quase fora do teste e tornar a métrica delas instável/enganosa.
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        df[COLUNAS_X], df["categoria"],
        test_size=test_size, stratify=df["categoria"], random_state=seed,
    )
    pipe = construir_pipeline().fit(X_tr, y_tr)
    pred_ml = pipe.predict(X_te)
    pred_regras = classificar_lote(X_te["descricao"])  # mesma base de teste, comparação justa
    return {
        "pipeline": pipe,
        "y_true": y_te,
        "pred_ml": pred_ml,
        "pred_regras": pred_regras,
        "acc_ml": accuracy_score(y_te, pred_ml),
        "acc_regras": accuracy_score(y_te, pred_regras),
    }


def _gerar_de_modelos(modelos_por_cat: dict[str, list[str]], n: int, seed: int) -> pd.DataFrame:
    """Gera uma base de teste reusando o gerador, mas com modelos de descrição alternativos.

    Herda peso/mediana/sigma/sinal de cada categoria real (só troca os textos) e desliga
    label noise/anomalia: estes experimentos medem generalização pura, não tolerância a ruído.
    """
    cats = {}
    for categoria, modelos in modelos_por_cat.items():
        cfg = dict(CATEGORIAS[categoria])
        cfg["descricoes"] = modelos
        cats[categoria] = cfg
    return gerar_transacoes(
        n=n, seed=seed, categorias=cats, frac_anomalia=0.0, frac_label_noise=0.0,
    )


# ---------------------------------------------------------------------------
# Experimento de generalização (HEADLINE) — variações de grafia de lojistas CONHECIDOS.
#
# Mesma marca, escrita de um jeito que a lista de keywords não previu: abreviações, letras a
# mais/menos, fonética. Conferidas uma a uma para NÃO conter a keyword exata como substring
# (senão a regra acertaria e o teste perderia o sentido), mas mantendo sobreposição de
# caracteres com o nome catalogado — que é o sinal que o char-n-gram aproveita.
# É o caso de uso real do char_wb: extrato bancário é uma sopa de grafias inconsistentes.
# ---------------------------------------------------------------------------
VARIANTES_LOJISTAS: dict[str, list[str]] = {
    "alimentacao": ["IFOOOD", "BURGUER KING", "RAPPY", "MACDONALD", "HABBIBS", "SUBWEY"],
    "mercado": ["CAREFOUR", "CARREFOR", "ATACADAUM", "ATACADON", "ASAI", "ATACAD {nome}"],
    "transporte": ["UBR TRIP", "SHEL BOX", "IPIRANG", "ESTAPR", "POSTU {nome}", "UBR VIAGEM"],
    "contas_fixas": ["SABSP", "COMGAZ", "VIVU FIXO", "CLARU NET", "CONDOMINI {nome}",
                     "VIVU FIBRA"],
    "lazer": ["NETFLX", "SPOTFY", "DSNEY PLUS", "PLAYSTATIO", "CINEMRK", "YOUTUB PREMIUM"],
    "saude": ["DROGSIL", "DROGRIA {nome}", "FARMCIA {nome}", "UNMED", "CLINIC {nome}",
              "FARMCIA SP"],
    "educacao": ["UDEMI", "ALUR {nome}", "CURSERA", "HOTMRT", "FACULDAD {nome}", "KINDL EDU"],
}


def experimento_generalizacao(df_treino: pd.DataFrame, n: int = 240, seed: int = 7) -> dict:
    """ML vs regras em grafias não catalogadas de lojistas conhecidos.

    Treina o ML em TODA a base sintética e avalia em variantes que as regras não previram.
    Aqui o ML supera as regras com folga — é onde o projeto se justifica.
    """
    pipe = treinar(df_treino)
    df_var = _gerar_de_modelos(VARIANTES_LOJISTAS, n=n, seed=seed)
    return {
        "df": df_var,
        "pred_ml": (pred_ml := pipe.predict(df_var[COLUNAS_X])),
        "pred_regras": (pred_regras := classificar_lote(df_var["descricao"])),
        "acc_ml": accuracy_score(df_var["categoria"], pred_ml),
        "acc_regras": accuracy_score(df_var["categoria"], pred_regras),
    }


# ---------------------------------------------------------------------------
# Limite honesto — lojistas 100% inéditos (marcas reais ausentes do treino E das regras).
#
# Sem sobreposição de caracteres com o que o modelo viu, não há sinal de TEXTO pra ninguém:
# as regras caem em 'outros' e o ML fica no nível do acaso, segurado apenas pela distribuição
# de valor (acerta contas_fixas, que são caras; erra o resto). Mostrar este teto é honestidade
# — define a fronteira do que o modelo resolve e motiva trabalho futuro (dados de lojista,
# embeddings). Categorias genéricas (transferência/'outros') ficam de fora: não são lojistas.
# ---------------------------------------------------------------------------
LOJISTAS_INEDITOS: dict[str, list[str]] = {
    "alimentacao": ["OUTBACK", "GIRAFFAS", "SPOLETO", "MADERO", "DIVINO FOGAO", "CHINA IN BOX"],
    "mercado": ["ZAFFARI", "ANGELONI", "BIG BOMPRECO", "SONDA", "MAMBO", "ST MARCHE"],
    "transporte": ["CABIFY", "BR MANIA", "ALE COMBUSTIVEIS", "TEXACO", "MULTIPARK", "CPTM SP"],
    "contas_fixas": ["CPFL ENERGIA", "CEMIG DISTRIB", "COPEL", "OI FIBRA", "AGUAS DO BRASIL",
                     "ALGAR TELECOM"],
    "lazer": ["HBO MAX", "GLOBOPLAY", "DEEZER", "PARAMOUNT", "XBOX LIVE", "CINEPOLIS"],
    "saude": ["PANVEL", "VENANCIO", "AMIL SAUDE", "HAPVIDA", "HOSPITAL SIRIO", "BIO MUNDO"],
    "educacao": ["DUOLINGO", "DOMESTIKA", "SKILLSHARE", "DATACAMP", "EBAC ONLINE", "PUC MINAS"],
}


def experimento_limite(df_treino: pd.DataFrame, n: int = 240, seed: int = 99) -> dict:
    """ML vs regras em marcas 100% inéditas — expõe o teto da abordagem (ML ≈ acaso)."""
    pipe = treinar(df_treino)
    df_novos = _gerar_de_modelos(LOJISTAS_INEDITOS, n=n, seed=seed)
    return {
        "df": df_novos,
        "pred_ml": (pred_ml := pipe.predict(df_novos[COLUNAS_X])),
        "pred_regras": (pred_regras := classificar_lote(df_novos["descricao"])),
        "acc_ml": accuracy_score(df_novos["categoria"], pred_ml),
        "acc_regras": accuracy_score(df_novos["categoria"], pred_regras),
    }


def _demo() -> None:
    """Roda os três cenários no terminal pra conferência rápida (o notebook 02 detalha)."""
    from pathlib import Path

    df = pd.read_csv(Path("data/sample/extrato_exemplo.csv"))

    h = avaliar_holdout(df)
    print("1) In-distribution (holdout estratificado 25%)")
    print(f"   regras: {h['acc_regras']:.3f}   ML: {h['acc_ml']:.3f}")
    print("   -> empate: dado circular, a regra barata já 'conhece' esses lojistas.\n")

    g = experimento_generalizacao(df)
    print("2) Generalização — grafias não catalogadas de lojistas conhecidos")
    print(f"   regras: {g['acc_regras']:.3f}   ML: {g['acc_ml']:.3f}")
    print("   -> ML ganha com folga: o char-n-gram reconhece a variante; é o valor do ML.\n")

    lim = experimento_limite(df)
    print("3) Limite — lojistas 100% inéditos (sem sinal de texto)")
    print(f"   regras: {lim['acc_regras']:.3f}   ML: {lim['acc_ml']:.3f}")
    print("   -> ML ≈ acaso (só o valor sustenta); fronteira honesta do que o modelo resolve.")


if __name__ == "__main__":
    _demo()
