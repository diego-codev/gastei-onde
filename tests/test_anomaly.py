"""Testes da detecção de anomalia.

O comportamento central é "por categoria": o MESMO valor pode ser rotina numa categoria e
fora da curva em outra. E o estimador de dispersão (MAD) tem que ser robusto — não pode ser
contaminado pelos outliers que ele caça. Os testes fixam os dois pontos e os fallbacks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.anomaly import escore_anomalia, marcar_anomalias


class TestEscorePorCategoria:
    def test_mesmo_valor_atipico_so_na_categoria_certa(self):
        # R$300 é rotina em mercado (todos ~300) mas dispara em lazer (todos ~25). O detector
        # tem que pegar SÓ o 300 do lazer — é o sentido de medir desvio dentro da categoria.
        valores =    [300, 310, 290, 305, 295,  20, 25, 30, 22, 300]
        categorias = ["mercado"] * 5 + ["lazer"] * 5
        scores = escore_anomalia(valores, categorias)
        assert abs(scores.iloc[0]) < 3.5     # 300 em mercado: normal
        assert abs(scores.iloc[9]) >= 3.5    # 300 em lazer: atípico

    def test_grupo_homogeneo_com_um_pico_usa_fallback_mad_zero(self):
        # MAD = 0 quando a maioria é idêntica; o código cai pro desvio absoluto médio em vez de
        # dividir por zero. O pico tem que continuar sendo pego.
        valores = [5, 5, 5, 5, 100]
        scores = escore_anomalia(valores, ["x"] * 5)
        assert abs(scores.iloc[4]) >= 3.5

    def test_valores_identicos_sem_dispersao_score_zero(self):
        # Sem nenhuma variação não há atípico — z = 0, não NaN nem divisão por zero.
        scores = escore_anomalia([10, 10, 10], ["x"] * 3)
        assert (scores == 0).all()


class TestMarcarAnomalias:
    def test_colunas_e_flag(self):
        df = pd.DataFrame({
            "valor": [20, 25, 30, 22, 300],
            "categoria": ["lazer"] * 5,
        })
        out = marcar_anomalias(df)
        assert "score_anomalia" in out.columns
        assert "eh_anomalia" in out.columns
        assert out["eh_anomalia"].to_list() == [False, False, False, False, True]
        # score arredondado a 2 casas (é o que vai pra UI).
        assert out["score_anomalia"].iloc[4] == round(out["score_anomalia"].iloc[4], 2)

    def test_nao_muta_o_df_original(self):
        # marcar_anomalias devolve cópia; o df de entrada não pode ganhar colunas.
        df = pd.DataFrame({"valor": [1, 2, 3], "categoria": ["x"] * 3})
        marcar_anomalias(df)
        assert list(df.columns) == ["valor", "categoria"]

    def test_usa_o_valor_absoluto(self):
        # Magnitude importa, sinal não: um crédito gigante (positivo) também é atípico.
        df = pd.DataFrame({
            "valor": [-10, -12, -9, -11, 5000],
            "categoria": ["x"] * 5,
        })
        out = marcar_anomalias(df)
        assert bool(out["eh_anomalia"].iloc[4]) is True

    def test_limiar_customizavel(self):
        # Limiar mais alto deixa passar o que o padrão pegaria — parâmetro de fato muda o corte.
        df = pd.DataFrame({"valor": [5, 5, 5, 5, 100], "categoria": ["x"] * 5})
        assert marcar_anomalias(df, limiar=3.5)["eh_anomalia"].iloc[4] == np.True_
        assert marcar_anomalias(df, limiar=100)["eh_anomalia"].iloc[4] == np.False_
