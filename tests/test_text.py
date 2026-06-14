"""Testes do feature engineering de texto.

Foco na `normalizar_descricao`: é o ponto onde treino e produção têm que enxergar o MESMO
texto limpo (o app e o modelo dependem disso). Se a normalização mudar sem querer, o modelo
passa a ver algo diferente do que treinou — esses testes travam esse contrato.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.text import (
    construir_vetorizador_texto,
    features_numericas,
    normalizar_descricao,
)


class TestNormalizarDescricao:
    def test_exemplos_documentados(self):
        # Os dois casos que a própria docstring promete — se quebrarem, a doc mente.
        assert normalizar_descricao("COMPRA IFOOD *PEDIDO 26499") == "ifood pedido"
        assert normalizar_descricao("PAG*CARREFOUR 61397") == "carrefour"

    def test_remove_digitos(self):
        # Sequências numéricas são código de terminal/loja/data, zero sinal de categoria.
        assert normalizar_descricao("UBER 123 456") == "uber"

    def test_remove_tokens_de_gateway(self):
        # 'pag', 'tef', 'compra' etc. são ruído de adquirente — somem.
        assert normalizar_descricao("TEF COMPRA SPOTIFY") == "spotify"

    def test_separadores_de_gateway_viram_espaco(self):
        # '*' e '/' grudam tokens; viram espaço pra soltar o nome do lojista.
        assert normalizar_descricao("NETFLIX*ASSINATURA/MENSAL") == "netflix assinatura mensal"

    def test_caixa_e_espacos(self):
        # Lower + colapso de espaços + strip: saída canônica sem depender da grafia da entrada.
        assert normalizar_descricao("  PADARIA   Do   BAIRRO  ") == "padaria do bairro"

    def test_so_ruido_vira_vazio(self):
        # Descrição que é puro gateway + dígitos não sobra nada — caso de borda real.
        assert normalizar_descricao("PAG*TEF 0000") == ""

    def test_aceita_nao_string(self):
        # A função faz str(texto): valores não-string (NaN vira 'nan') não podem explodir.
        assert normalizar_descricao(12345) == ""  # vira "12345", dígitos somem


class TestFeaturesNumericas:
    def test_valor_log_e_credito(self):
        df = pd.DataFrame({"valor": [-100.0, 50.0, -1.0]})
        feats = features_numericas(df)
        # valor_log usa |valor| (magnitude importa, sinal não) via log1p.
        assert feats["valor_log"].to_list() == pytest.approx(
            [np.log1p(100), np.log1p(50), np.log1p(1)]
        )
        # eh_credito: 1 só pra entradas (valor >= 0).
        assert feats["eh_credito"].to_list() == [0, 1, 0]

    def test_preserva_indice(self):
        # As features são concatenadas ao df original no pipeline; o índice tem que bater.
        df = pd.DataFrame({"valor": [10.0, 20.0]}, index=[7, 99])
        feats = features_numericas(df)
        assert feats.index.to_list() == [7, 99]


class TestVetorizador:
    def test_configuracao_char_wb(self):
        # char_wb 3-5 com min_df=2 é a escolha registrada; o pipeline depende desses params.
        vec = construir_vetorizador_texto()
        assert vec.analyzer == "char_wb"
        assert vec.ngram_range == (3, 5)
        assert vec.min_df == 2

    def test_fit_transform_funciona(self):
        # Smoke: precisa fitar e gerar matriz sobre descrições com n-grama compartilhado
        # (min_df=2 exige n-grama em >=2 documentos — 'ifood' aparece em dois).
        vec = construir_vetorizador_texto()
        m = vec.fit_transform(["IFOOD PEDIDO", "IFOOD LOJA", "CARREFOUR"])
        assert m.shape[0] == 3
        assert len(vec.vocabulary_) > 0
