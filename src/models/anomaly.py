"""Detecção de gastos atípicos — por categoria, com estatística robusta (mediana + MAD).

Por que **por categoria**: R$300 em mercado é rotina; R$300 em lazer é fora da curva. Uma
régua global ignoraria o contexto e apontaria as categorias caras como "anômalas" o tempo
todo. O desvio é medido DENTRO de cada categoria.

Por que **mediana + MAD** (e não média + desvio-padrão): a métrica de dispersão não pode ser
contaminada pelos próprios outliers que estou caçando. Média e desvio são puxados pelo gasto
gigante; mediana e MAD (desvio absoluto mediano) quase não se mexem. Uso o z-score modificado
de Iglewicz-Hoaglin: `0.6745*(x - mediana)/MAD`, com limiar 3.5 (corte recomendado na
literatura).

Por que **não IsolationForest** aqui: o sinal é 1-D (o valor) e as classes raras têm poucas
amostras; uma floresta seria caixa-preta e instável nesse regime. O MAD é interpretável ("este
gasto está N desvios robustos acima do típico da categoria") — e o app consegue explicar o
flag pro usuário, que é parte do produto.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Fator de Iglewicz-Hoaglin: deixa o z modificado na mesma escala do z normal (a MAD de uma
# normal ≈ 0.6745*sigma). Limiar 3.5 é o ponto de corte que eles recomendam.
_FATOR_MAD = 0.6745
# Fator equivalente quando caímos no desvio absoluto MÉDIO (E[|N(0,1)|] ≈ 1/1.2533).
_FATOR_MEAN_AD = 1.2533
LIMIAR_PADRAO = 3.5


def _z_robusto(valores: pd.Series) -> pd.Series:
    """z-score modificado de UM grupo (uma categoria), sobre o |valor|.

    Trabalho com o valor absoluto porque o que importa é a magnitude do gasto, não o sinal
    (débito negativo / crédito positivo) — assim um salário atipicamente alto também é pego.
    """
    x = valores.abs().astype(float)
    mediana = x.median()
    mad = (x - mediana).abs().median()
    if mad > 0:
        return _FATOR_MAD * (x - mediana) / mad
    # MAD zero (valores quase idênticos ou grupo minúsculo): caio pro desvio absoluto MÉDIO,
    # ainda mais robusto que o desvio-padrão. Se também for zero, não há dispersão -> z=0.
    mean_ad = (x - mediana).abs().mean()
    if mean_ad > 0:
        return (x - mediana) / (_FATOR_MEAN_AD * mean_ad)
    return pd.Series(0.0, index=valores.index)


def escore_anomalia(valores, categorias) -> pd.Series:
    """z-score robusto POR categoria: quanto cada gasto desvia do típico da SUA categoria."""
    s = pd.Series(np.asarray(valores, dtype=float)).reset_index(drop=True)
    c = pd.Series(np.asarray(categorias)).reset_index(drop=True)
    # transform aplica `_z_robusto` em cada grupo e remonta alinhado ao índice original.
    return s.groupby(c).transform(_z_robusto)


def marcar_anomalias(
    df: pd.DataFrame,
    col_valor: str = "valor",
    col_categoria: str = "categoria",
    limiar: float = LIMIAR_PADRAO,
) -> pd.DataFrame:
    """Devolve cópia do df com `score_anomalia` (z robusto) e `eh_anomalia` (|z| >= limiar).

    Uso o limiar sobre o valor ABSOLUTO do score: um gasto muito acima OU muito abaixo do
    padrão da categoria é atípico — embora, na prática, o que aparece são os picos altos.
    A categoria usada é a que vier no df (no app, a **prevista** pelo modelo).
    """
    score = escore_anomalia(df[col_valor].to_numpy(), df[col_categoria].to_numpy())
    out = df.copy()
    out["score_anomalia"] = score.to_numpy().round(2)
    out["eh_anomalia"] = np.abs(score.to_numpy()) >= limiar
    return out


def _demo() -> None:
    """Roda sobre o CSV de exemplo e mostra os gastos atípicos pegos em cada categoria."""
    from pathlib import Path

    df = pd.read_csv(Path("data/sample/extrato_exemplo.csv"))
    marcado = marcar_anomalias(df)
    anom = marcado[marcado["eh_anomalia"]].sort_values("score_anomalia", ascending=False)

    taxa = marcado["eh_anomalia"].mean()
    print(f"Anomalias: {len(anom)}/{len(df)} ({taxa:.1%})  [gerador injeta ~3%]\n")

    # Mostra, por anomalia, o valor vs a mediana da categoria — pra dar pra "ver" o atípico.
    medianas = df.groupby("categoria")["valor"].apply(lambda s: s.abs().median())
    for _, r in anom.head(12).iterrows():
        med = medianas[r["categoria"]]
        print(f"  {r['categoria']:<14} R$ {abs(r['valor']):>9.2f}  "
              f"(mediana ~R$ {med:>7.2f}, z={r['score_anomalia']:.1f})  {r['descricao']}")


if __name__ == "__main__":
    _demo()
