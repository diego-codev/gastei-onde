"""Testes do baseline por regras.

Duas propriedades que importam: (1) a ORDEM do dict é prioridade — a primeira categoria que
casa vence, e é assim que ambiguidades se resolvem; (2) o casamento é por SUBSTRING do texto
já normalizado. Os testes fixam esses dois comportamentos.
"""

from __future__ import annotations

from src.models.rules import FALLBACK, classificar_lote, classificar_por_regras


class TestClassificarPorRegras:
    def test_palavra_chave_direta(self):
        assert classificar_por_regras("NETFLIX.COM") == "lazer"
        assert classificar_por_regras("UBER *TRIP 123") == "transporte"

    def test_casamento_por_substring(self):
        # "droga" é substring de "drogasil" — a regra de saúde pega por pedaço, de propósito.
        assert classificar_por_regras("DROGASIL FILIAL 88") == "saude"

    def test_prioridade_resolve_ambiguidade(self):
        # "PIX IFOOD" casaria transferência (pix) E alimentação (ifood). alimentacao vem antes
        # no dict, então vence — é o desempate por ordem que a docstring promete.
        assert classificar_por_regras("PIX IFOOD PEDIDO") == "alimentacao"

    def test_fallback_quando_nada_casa(self):
        # Sem palavra-chave conhecida -> "outros" (mesmo "resto" do gerador).
        assert classificar_por_regras("LOJA XPTO QUALQUER") == FALLBACK

    def test_normaliza_antes_de_casar(self):
        # Reusa normalizar_descricao: gateway/dígitos somem antes do casamento.
        assert classificar_por_regras("PAG*SPOTIFY 9931") == "lazer"


class TestClassificarLote:
    def test_alinha_saida_com_entrada(self):
        descricoes = ["IFOOD", "UBER", "NADA AQUI"]
        assert classificar_lote(descricoes) == ["alimentacao", "transporte", FALLBACK]

    def test_lista_vazia(self):
        assert classificar_lote([]) == []
