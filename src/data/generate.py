"""Gerador de extrato bancário sintético.

Produz um CSV que *imita* um extrato real: descrições bagunçadas (`PAG*IFD`, `UBER *TRIP`),
categorias desbalanceadas, gastos atípicos e uma pequena taxa de rótulo errado.

A intenção é deliberada: dado sintético "limpo demais" deixa a classificação trivial e as
métricas viram 99% sem significado. Aqui o ruído faz o feature engineering e o modelo
trabalharem de verdade — e as métricas contam uma história honesta.

Uso:
    python -m src.data.generate                 # gera data/sample/extrato_exemplo.csv
    python -m src.data.generate --n 800 --seed 7
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# Caminho padrão do CSV versionado que alimenta os notebooks, o treino e o app.
DEFAULT_OUT = Path("data/sample/extrato_exemplo.csv")

# Período fixo (não "hoje") de propósito: assim regenerar o CSV com o mesmo seed produz
# exatamente o mesmo arquivo, mantendo o artefato versionado reprodutível.
DEFAULT_INICIO = date(2026, 2, 1)
DEFAULT_FIM = date(2026, 5, 31)


# ---------------------------------------------------------------------------
# Configuração por categoria
#
# `peso`    -> frequência relativa (desbalanceamento PROPOSITAL: alimentação/mercado são
#              comuns, saúde/educação são raras — espelha gasto real e dá motivo concreto
#              pra usar class_weight e métricas por classe lá na frente).
# `mediana` -> valor típico em R$ (mediana da lognormal; gasto financeiro é assimétrico,
#              com muitos valores pequenos e poucos grandes — lognormal modela isso melhor
#              que uma normal).
# `sigma`   -> dispersão em escala log.
# `sinal`   -> -1 para despesa (débito), +1 para entrada (crédito).
# `descricoes` -> modelos de texto crus, já no estilo "sopa de letrinhas" de extrato.
#                 `{n}`/`{nome}` viram ruído aleatório (número de loja, sufixo).
# ---------------------------------------------------------------------------
CATEGORIAS: dict[str, dict] = {
    "alimentacao": {
        "peso": 0.22,
        "mediana": 38.0,
        "sigma": 0.55,
        "sinal": -1,
        "descricoes": [
            "IFD*IFOOD", "IFOOD *PEDIDO", "PAG*IFOOD", "RAPPI*BRASIL", "PAG*RAPPI",
            "UBER *EATS", "MC DONALDS", "BURGER KING", "BK BRASIL {n}", "HABIBS",
            "SUBWAY {n}", "PADARIA {nome}", "REST {nome}", "PIZZARIA {nome}",
            "BAR DO {nome}", "ZE DELIVERY",
        ],
    },
    "mercado": {
        "peso": 0.16,
        "mediana": 120.0,
        "sigma": 0.65,
        "sinal": -1,
        "descricoes": [
            "PAG*CARREFOUR", "CARREFOUR {n}", "PAO DE ACUCAR", "EXTRA SUPER {n}",
            "ASSAI ATACAD", "ATACADAO {n}", "DIA SUPERMERCADO", "SUPERM {nome}",
            "HORTIFRUTI {nome}", "SACOLAO {nome}",
        ],
    },
    "transporte": {
        "peso": 0.15,
        "mediana": 19.0,
        "sigma": 0.7,
        "sinal": -1,
        "descricoes": [
            "UBER *TRIP", "UBER* VIAGEM", "99*APP", "99POP {n}", "TEF UBER",
            "POSTO IPIRANGA", "POSTO SHELL", "SHELL BOX", "AUTO POSTO {n}",
            "ESTAPAR ESTAC", "METRO SP", "VIA QUATRO", "RECARGA BU {n}",
        ],
    },
    "contas_fixas": {
        "peso": 0.12,
        "mediana": 115.0,
        "sigma": 0.5,
        "sinal": -1,
        "descricoes": [
            "ENEL SP", "ENEL DISTRIB", "SABESP", "COMGAS", "VIVO FIXO", "CLARO NET",
            "TIM*FATURA", "CONDOMINIO {n}", "ALUGUEL {nome}", "IPTU {nome}",
            "INTERNET VIVO",
        ],
    },
    "lazer": {
        "peso": 0.11,
        "mediana": 32.0,
        "sigma": 0.75,
        "sinal": -1,
        "descricoes": [
            "NETFLIX.COM", "SPOTIFY", "SPOTIFY BR", "DISNEY PLUS", "AMAZON PRIME",
            "STEAMGAMES.COM", "PLAYSTATION NET", "CINEMARK {n}", "INGRESSO.COM",
            "PAG*BAR {nome}", "YOUTUBEPREMIUM",
        ],
    },
    "transferencia": {
        "peso": 0.10,
        "mediana": 250.0,
        "sigma": 1.0,
        "sinal": -1,  # sobrescrito caso a caso: entradas viram positivas (ver gerador)
        "descricoes": [
            # Entradas (crédito) — viram valor positivo no gerador:
            "SALARIO", "PIX RECEBIDO", "TED RECEBIDA", "TRANSF RECEBIDA",
            "PAGAMENTO RECEBIDO",
            # Saídas (débito) — genuinamente ambíguas (poderiam ser qualquer coisa);
            # essa ambiguidade é intencional, vai confundir o modelo de propósito:
            "PIX ENVIADO", "TED ENVIADA", "PIX*{nome}", "DOC ENVIADO", "TRANSF ENVIADA",
        ],
    },
    "saude": {
        "peso": 0.06,
        "mediana": 65.0,
        "sigma": 0.8,
        "sinal": -1,
        "descricoes": [
            "DROGASIL {n}", "DROGARIA SP", "RAIA DROGASIL", "PAGUE MENOS", "DROGA RAIA",
            "FARMA {nome}", "CLINICA {nome}", "LAB {nome}", "UNIMED", "PAG*FARMACIA",
        ],
    },
    "educacao": {
        "peso": 0.04,
        "mediana": 95.0,
        "sigma": 0.6,
        "sinal": -1,
        "descricoes": [
            "UDEMY", "ALURA CURSOS", "PAG*ALURA", "COURSERA", "HOTMART*CURSO",
            "PAG*ESCOLA", "FACULDADE {nome}", "LIVRARIA {nome}", "KINDLE EDU",
        ],
    },
    "outros": {
        "peso": 0.04,
        "mediana": 28.0,
        "sigma": 1.0,
        "sinal": -1,
        "descricoes": [
            "SAQUE 24HORAS", "SAQUE BANCO24H", "TARIFA MENSAL", "TARIFA PACOTE",
            "ANUIDADE CARTAO", "PAG BOLETO", "IOF", "PIX*{nome}",
        ],
    },
}

# Entradas de crédito dentro de "transferencia": casam pelo prefixo da descrição.
_PREFIXOS_CREDITO = ("SALARIO", "PIX RECEBIDO", "TED RECEBIDA", "TRANSF RECEBIDA",
                     "PAGAMENTO RECEBIDO")

# Sufixos/“nomes” genéricos pra preencher {nome} e dar variedade textual.
_NOMES = ["SP", "RJ", "LTDA", "ME", "JOAO", "MARIA", "CENTRO", "JARDINS", "01", "BR",
          "COMERCIO", "SILVA", "EIRELI"]


def _aplica_ruido(texto: str, rng: np.random.Generator) -> str:
    """Suja a descrição como um extrato real: caixa inconsistente, prefixos de gateway e
    códigos de loja no fim.

    Por quê: bancos e adquirentes concatenam prefixo do gateway + nome do lojista +
    identificadores. O modelo precisa aprender a ignorar esse lixo e focar no sinal — então
    o dado de treino tem que conter o lixo.
    """
    # Preenche placeholders {n} (número) e {nome} (sufixo) com ruído aleatório.
    texto = texto.replace("{n}", str(rng.integers(1, 9999)))
    texto = texto.replace("{nome}", rng.choice(_NOMES))

    # Prefixo de adquirente ocasional (PAG*, TEF, C*...) quando ainda não há um.
    if rng.random() < 0.25 and not texto.startswith(("PAG", "TEF", "IFD", "PIX")):
        texto = rng.choice(["PAG*", "TEF ", "C*", "DL*", "COMPRA "]) + texto

    # Código numérico residual no fim (terminal/loja), como aparece em fatura.
    if rng.random() < 0.30:
        texto = f"{texto} {rng.integers(100, 999999)}"

    # Caixa inconsistente: extrato real mistura tudo maiúsculo com trechos minúsculos.
    sorteio = rng.random()
    if sorteio < 0.7:
        texto = texto.upper()
    elif sorteio < 0.85:
        texto = texto.lower()
    # senão, mantém como está (mixed case)

    return texto.strip()


def _gera_valor(cfg: dict, rng: np.random.Generator) -> float:
    """Valor em R$ via lognormal (cauda à direita: poucos gastos grandes, muitos pequenos)."""
    mu = np.log(cfg["mediana"])  # mediana da lognormal = exp(mu)
    valor = float(rng.lognormal(mean=mu, sigma=cfg["sigma"]))
    return round(valor, 2)


def gerar_transacoes(
    n: int,
    seed: int,
    inicio: date = DEFAULT_INICIO,
    fim: date = DEFAULT_FIM,
    frac_anomalia: float = 0.03,
    frac_label_noise: float = 0.03,
) -> pd.DataFrame:
    """Gera `n` transações sintéticas com colunas: data, descricao, valor, categoria."""
    rng = np.random.default_rng(seed)

    nomes_cat = list(CATEGORIAS.keys())
    pesos = np.array([CATEGORIAS[c]["peso"] for c in nomes_cat])
    pesos = pesos / pesos.sum()  # normaliza pra somar 1 (os pesos brutos são aproximados)

    # Sorteia a categoria de cada transação segundo os pesos -> gera o desbalanceamento.
    categorias = rng.choice(nomes_cat, size=n, p=pesos)

    # Datas uniformes no período; ordenar depois deixa o extrato com cara de cronológico.
    dias_periodo = (fim - inicio).days
    offsets = rng.integers(0, dias_periodo + 1, size=n)
    datas = [pd.Timestamp(inicio) + pd.Timedelta(days=int(o)) for o in offsets]

    descricoes, valores = [], []
    for cat in categorias:
        cfg = CATEGORIAS[cat]
        modelo = rng.choice(cfg["descricoes"])
        descricao = _aplica_ruido(modelo, rng)
        valor = _gera_valor(cfg, rng)

        # Entradas de crédito (salário, PIX recebido) entram positivas; o resto, negativo.
        sinal = 1 if modelo.startswith(_PREFIXOS_CREDITO) else cfg["sinal"]
        valores.append(round(sinal * valor, 2))
        descricoes.append(descricao)

    df = pd.DataFrame(
        {
            "data": [d.date().isoformat() for d in datas],
            "descricao": descricoes,
            "valor": valores,
            "categoria": categorias,
        }
    )

    # --- Gastos atípicos (anomalias) -------------------------------------------------
    # Multiplica o valor de uma fração pequena por um fator alto, criando outliers DENTRO
    # da categoria (um delivery de R$ 600 é anômalo; um aluguel de R$ 600 não). É isso que
    # o detector de anomalia vai precisar pegar mais pra frente.
    n_anom = int(round(frac_anomalia * n))
    if n_anom:
        idx_anom = rng.choice(df.index, size=n_anom, replace=False)
        fatores = rng.uniform(5.0, 14.0, size=n_anom)
        df.loc[idx_anom, "valor"] = (df.loc[idx_anom, "valor"] * fatores).round(2)

    # --- Ruído de rótulo (label noise) -----------------------------------------------
    # Troca o rótulo de uma fração por outra categoria aleatória. Rótulo de dado real é
    # imperfeito (categorização humana erra); embutir isso evita métricas otimistas demais
    # e força o modelo a tolerar um teto de ruído.
    n_noise = int(round(frac_label_noise * n))
    if n_noise:
        idx_noise = rng.choice(df.index, size=n_noise, replace=False)
        for i in idx_noise:
            outras = [c for c in nomes_cat if c != df.at[i, "categoria"]]
            df.at[i, "categoria"] = rng.choice(outras)

    # Ordena por data: extrato de verdade vem em ordem cronológica.
    df = df.sort_values("data").reset_index(drop=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera um extrato bancário sintético (CSV).")
    parser.add_argument("--n", type=int, default=600, help="número de transações")
    parser.add_argument("--seed", type=int, default=42, help="semente (reprodutibilidade)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="caminho de saída")
    args = parser.parse_args()

    df = gerar_transacoes(n=args.n, seed=args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)  # cria data/sample/ se não existir
    df.to_csv(args.out, index=False)

    # Resumo no terminal pra conferência rápida de que o desbalanceamento "saiu".
    print(f"Gerado: {args.out}  ({len(df)} transações)")
    print("\nDistribuição por categoria:")
    print(df["categoria"].value_counts())


if __name__ == "__main__":
    main()
