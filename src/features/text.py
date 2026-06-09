"""Feature engineering das descrições de transação.

Duas frentes:
1. **Texto** — normaliza a descrição crua (`PAG*IFD 1234` -> `ifd`) e vetoriza com TF-IDF
   de n-grama de caractere.
2. **Numéricas** — sinais simples derivados do valor que ajudam a desambiguar categorias.

A normalização e o vetorizador são reaproveitados tanto no treino do modelo quanto no app,
pra garantir que o que o modelo vê em produção é exatamente o que viu no treino.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Tokens de adquirente/gateway que aparecem como prefixo e não carregam sinal de categoria.
# Removê-los reduz a dimensionalidade e deixa as features interpretáveis. (O TF-IDF já
# penalizaria via IDF por serem comuns, mas tirar explicitamente é mais limpo e barato.)
_TOKENS_RUIDO = frozenset({"pag", "tef", "compra", "c", "dl", "ec"})

# Sequências de dígitos = código de terminal/loja/data; nenhum sinal de categoria.
_RE_DIGITOS = re.compile(r"\d+")
# Separadores de gateway (asterisco, barra) viram espaço pra soltar os tokens grudados.
_RE_SEPARADORES = re.compile(r"[*/]+")
_RE_ESPACOS = re.compile(r"\s+")


def normalizar_descricao(texto: str) -> str:
    """Limpa a descrição crua mantendo só o sinal do lojista.

    Ex.: 'COMPRA IFOOD *PEDIDO 26499' -> 'ifood pedido'
         'PAG*CARREFOUR 61397'        -> 'carrefour'
    """
    texto = str(texto).lower()
    texto = _RE_SEPARADORES.sub(" ", texto)   # 'pag*ifood' -> 'pag ifood'
    texto = _RE_DIGITOS.sub(" ", texto)        # remove códigos numéricos
    # Descarta os tokens de gateway; o que sobra é o nome do lojista (o que importa).
    tokens = [t for t in texto.split() if t not in _TOKENS_RUIDO]
    return _RE_ESPACOS.sub(" ", " ".join(tokens)).strip()


def construir_vetorizador_texto() -> TfidfVectorizer:
    """TF-IDF de n-grama de CARACTERE sobre a descrição normalizada.

    `analyzer='char_wb'` gera n-gramas dentro de cada palavra (respeitando a fronteira).
    Escolhi caractere e não palavra porque variações como 'ifd' e 'ifood' compartilham
    subsequências ('ifo', 'foo') mas não tokens inteiros — n-grama de palavra perderia esse
    parentesco. `min_df=2` corta n-gramas que aparecem uma única vez (provável ruído).
    """
    return TfidfVectorizer(
        preprocessor=normalizar_descricao,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
    )


def features_numericas(df: pd.DataFrame) -> pd.DataFrame:
    """Sinais numéricos simples derivados do valor.

    - `valor_log`: log1p do valor absoluto. O valor tem cauda longa (ver EDA); o log
      comprime a escala e ajuda a separar categorias caras (contas fixas, mercado) das
      baratas (transporte, alimentação).
    - `eh_credito`: 1 se entrada (valor >= 0). Crédito é quase sempre salário/transferência
      recebida — sinal forte e barato pra essa categoria.

    Deixei DE FORA dia-da-semana/fim-de-semana de propósito: o gerador sintético não embute
    padrão temporal por categoria, então essas features seriam ruído. Em dado real valeriam
    o teste (ex.: lazer concentra no fim de semana).
    """
    valor = df["valor"].astype(float)
    return pd.DataFrame(
        {
            "valor_log": np.log1p(valor.abs()),
            "eh_credito": (valor >= 0).astype(int),
        },
        index=df.index,
    )


def _demo() -> None:
    """Mostra a normalização em ação sobre algumas descrições do CSV de exemplo."""
    from pathlib import Path

    csv = Path("data/sample/extrato_exemplo.csv")
    df = pd.read_csv(csv).sample(10, random_state=2)
    print("descrição crua -> normalizada")
    for _, row in df.iterrows():
        print(f"  {row['descricao']:<34} -> {normalizar_descricao(row['descricao'])!r}")


if __name__ == "__main__":
    _demo()
