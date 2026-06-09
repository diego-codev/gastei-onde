"""Baseline de classificação por regras de palavra-chave.

Um dicionário `palavra-chave -> categoria` aplicado sobre a descrição **normalizada**. É
deliberadamente simples: serve de **piso**. O modelo de ML da próxima etapa só se justifica
se superar este baseline com folga — senão, regra boa e barata resolveria.

Reutiliza `normalizar_descricao` pra enxergar exatamente o mesmo texto limpo que o modelo vê.
"""

from __future__ import annotations

from collections.abc import Iterable

from src.features.text import normalizar_descricao

# Regras em ORDEM DE PRIORIDADE (dict do Python preserva ordem de inserção): a primeira
# categoria com alguma palavra-chave casada vence. A ordem resolve ambiguidades — ex.: um
# "PIX*" que na verdade era tarifa cai em transferência porque ela vem antes de "outros".
# As palavras-chave são casadas como SUBSTRING do texto normalizado e foram escolhidas
# longas o bastante pra evitar colisão (ex.: uso "netflix", não "net", que pegaria a conta
# de internet "claro net").
REGRAS: dict[str, list[str]] = {
    "alimentacao": ["ifood", "ifd", "rappi", "eats", "donalds", "burger", "bk", "habibs",
                    "subway", "padaria", "pizzaria", "rest", "bar", "delivery"],
    "mercado": ["carrefour", "pao", "acucar", "extra", "assai", "atacadao", "superm",
                "hortifruti", "sacolao", "mercado"],
    "transporte": ["uber", "posto", "shell", "ipiranga", "estapar", "metro", "via quatro",
                   "bilhete", "recarga", "pop"],
    "contas_fixas": ["enel", "sabesp", "comgas", "vivo", "claro", "tim", "condominio",
                     "aluguel", "iptu", "internet"],
    "lazer": ["netflix", "spotify", "disney", "prime", "steam", "playstation", "cinemark",
              "ingresso", "youtube"],
    "saude": ["drogasil", "drogaria", "droga", "raia", "pague", "farma", "clinica",
              "unimed", "lab"],
    "educacao": ["udemy", "alura", "coursera", "hotmart", "escola", "faculdade", "livraria",
                 "kindle"],
    "transferencia": ["pix", "ted", "doc", "transf", "salario", "recebido", "recebida"],
}

# Categoria de fallback quando nenhuma regra casa — mesma usada como "resto" no gerador.
FALLBACK = "outros"


def classificar_por_regras(descricao: str) -> str:
    """Classifica UMA descrição pela primeira palavra-chave que casar (ordem de prioridade)."""
    texto = normalizar_descricao(descricao)
    for categoria, palavras in REGRAS.items():
        if any(p in texto for p in palavras):
            return categoria
    return FALLBACK


def classificar_lote(descricoes: Iterable[str]) -> list[str]:
    """Versão em lote pra aplicar numa coluna inteira de uma vez."""
    return [classificar_por_regras(d) for d in descricoes]


def _avaliar() -> None:
    """Mede o piso: roda o baseline sobre o CSV de exemplo e imprime as métricas."""
    from pathlib import Path

    import pandas as pd
    from sklearn.metrics import accuracy_score, classification_report

    df = pd.read_csv(Path("data/sample/extrato_exemplo.csv"))
    pred = classificar_lote(df["descricao"])

    acc = accuracy_score(df["categoria"], pred)
    # Cobertura: quantas transações alguma regra de fato classificou (não caíram no fallback).
    cobertura = sum(p != FALLBACK for p in pred) / len(pred)
    print(f"Baseline por regras — accuracy: {acc:.3f} | cobertura: {cobertura:.1%}\n")
    print(classification_report(df["categoria"], pred, zero_division=0))


if __name__ == "__main__":
    _avaliar()
