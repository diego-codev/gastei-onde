"""Testes de regressão da ingestão de CSV.

Cada caso aqui nasceu de um arquivo real que quebrou — vários vieram do teste com usuários.
A intenção é travar esses comportamentos pra que um refactor futuro não ressuscite o bug:
- filtro de linhas de saldo (não são transação);
- descrição partida em duas colunas (Lançamento + Detalhes);
- valor em formato BR e valor "partido" pela vírgula-decimal;
- escolha de delimitador `;` antes de `,`.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app import (
    _ler_extrato_br,
    ler_csv_upload,
    mapear_colunas,
    parsear_valor,
    preparar_dados,
)


class TestParsearValor:
    def test_formato_br(self):
        # '.' = milhar, ',' = decimal — o caso nativo de banco brasileiro.
        assert parsear_valor(pd.Series(["1.234,56"])).iloc[0] == pytest.approx(1234.56)

    def test_formato_internacional(self):
        # ',' = milhar, '.' = decimal — desambiguado pela posição do último separador.
        assert parsear_valor(pd.Series(["1,234.56"])).iloc[0] == pytest.approx(1234.56)

    def test_so_virgula_decimal(self):
        assert parsear_valor(pd.Series(["45,90"])).iloc[0] == pytest.approx(45.90)

    def test_prefixo_e_sinal(self):
        # 'R$', espaços e sinal negativo não podem atrapalhar o parse.
        assert parsear_valor(pd.Series(["-R$ 1.000,00"])).iloc[0] == pytest.approx(-1000.0)

    def test_numerico_passa_direto(self):
        # Coluna já numérica (CSV bem-comportado) só vira float, sem reparse de string.
        out = parsear_valor(pd.Series([1.5, -2]))
        assert out.to_list() == [1.5, -2.0]

    def test_vazio_e_invalido_viram_nan(self):
        out = parsear_valor(pd.Series(["", "nan", "abc"]))
        assert out.isna().all()


class TestMapearColunas:
    def test_sinonimos_para_canonico(self):
        df = pd.DataFrame(columns=["Data", "Histórico", "Valor"])
        assert set(mapear_colunas(df).columns) == {"data", "descricao", "valor"}

    def test_insensivel_a_caixa(self):
        df = pd.DataFrame(columns=["DESCRIÇÃO", "VALOR"])
        assert set(mapear_colunas(df).columns) == {"descricao", "valor"}

    def test_guard_detalhe_nao_e_reivindicado_duas_vezes(self):
        # "Lançamento" vira descricao e "Detalhes" vira detalhes — colunas distintas. O guard
        # impede que "detalhe" (sinônimo dos DOIS alvos) faça um roubar a coluna do outro.
        df = pd.DataFrame(columns=["Data", "Lançamento", "Detalhes", "Valor"])
        cols = set(mapear_colunas(df).columns)
        assert "descricao" in cols and "detalhes" in cols

    def test_coluna_detalhe_sozinha_vira_descricao(self):
        # Só "Detalhe" no arquivo: descricao vem antes no dict e vence -> não some o texto.
        df = pd.DataFrame(columns=["Data", "Detalhe", "Valor"])
        assert "descricao" in mapear_colunas(df).columns


class TestPrepararDados:
    def test_filtra_linhas_de_saldo(self):
        # BUG achado em teste real: linha de saldo do dia entrava como transação e era flagada
        # como atípica. Todas as variações de grafia de saldo têm que cair fora.
        df = pd.DataFrame({
            "descricao": ["Saldo Anterior", "PIX FULANO", "Saldo do dia", "S A L D O", "IFOOD"],
            "valor": ["100,00", "-50,00", "1.500,00", "1.450,00", "-30,00"],
        })
        out = preparar_dados(df)
        assert out["descricao"].to_list() == ["PIX FULANO", "IFOOD"]

    def test_concatena_detalhes_na_descricao(self):
        # Extrato que parte a descrição: o modelo precisa ver o favorecido, não só "Pix-Enviado".
        df = pd.DataFrame({
            "lançamento": ["Pix - Enviado"],
            "detalhes": ["FULANO DE TAL"],
            "valor": ["-100,00"],
        })
        out = preparar_dados(df)
        assert out["descricao"].iloc[0] == "Pix - Enviado FULANO DE TAL"

    def test_detalhes_sozinho_vira_descricao(self):
        df = pd.DataFrame({"detalhes": ["MERCADO X"], "valor": ["-50,00"]})
        assert preparar_dados(df)["descricao"].iloc[0] == "MERCADO X"

    def test_erro_claro_sem_colunas_essenciais(self):
        df = pd.DataFrame({"foo": [1], "bar": [2]})
        with pytest.raises(ValueError, match="descricao"):
            preparar_dados(df)

    def test_data_e_opcional(self):
        # Alguns extratos não trazem data; não pode quebrar, só vira coluna vazia.
        df = pd.DataFrame({"descricao": ["IFOOD"], "valor": ["-30,00"]})
        out = preparar_dados(df)
        assert "data" in out.columns

    def test_descarta_valor_ilegivel(self):
        df = pd.DataFrame({"descricao": ["A", "B"], "valor": ["-30,00", "xxx"]})
        out = preparar_dados(df)
        assert len(out) == 1


class TestLerCsvUpload:
    def test_csv_ponto_e_virgula(self):
        dados = "Data;Histórico;Valor\n01/05/2026;IFOOD;-45,90\n".encode("utf-8")
        out = preparar_dados(ler_csv_upload(dados))
        assert out["descricao"].iloc[0] == "IFOOD"
        assert out["valor"].iloc[0] == pytest.approx(-45.90)

    def test_csv_virgula(self):
        dados = "data,descricao,valor\n01/05/2026,UBER TRIP,-12.50\n".encode("utf-8")
        out = preparar_dados(ler_csv_upload(dados))
        assert out["descricao"].iloc[0] == "UBER TRIP"
        assert out["valor"].iloc[0] == pytest.approx(-12.50)

    def test_extrato_br_sujo_com_preambulo(self):
        # Preâmbulo antes do cabeçalho + delimitador `;`: o caso que originou o fix do
        # delimitador. Tem que pular o preâmbulo e recuperar as transações.
        texto = (
            "Banco Exemplo - Extrato\n"
            "Conta 0001 / Agência 1234\n"
            "\n"
            "Data;Histórico;Valor\n"
            "01/05/2026;PIX ENVIADO;-100,00\n"
            "02/05/2026;IFOOD PEDIDO;-45,90\n"
        )
        out = preparar_dados(ler_csv_upload(texto.encode("utf-8")))
        assert len(out) == 2
        assert out["valor"].to_list() == pytest.approx([-100.0, -45.90])

    def test_arquivo_ilegivel_da_erro_claro(self):
        # Um arquivo que não é extrato passa "cru" pelo leitor leniente, mas o pipeline tem que
        # barrar com erro claro de coluna faltando — o usuário não recebe lixo silencioso.
        lixo = b"isto nao e um csv de extrato\nlinha solta sem nada\n"
        with pytest.raises(ValueError):
            preparar_dados(ler_csv_upload(lixo))

    def test_decodifica_latin1(self):
        # Bancos exportam em latin-1; o leitor tem que decodificar sem perder acento.
        dados = "Data;Histórico;Valor\n01/05/2026;PADARIA;-10,00\n".encode("latin-1")
        out = preparar_dados(ler_csv_upload(dados))
        assert out["descricao"].iloc[0] == "PADARIA"


class TestLerExtratoBrDelimitador:
    def test_ponto_e_virgula_e_virgula_ambos_funcionam(self):
        # O fix: tentar `;` antes de `,`. Os dois layouts do mesmo extrato têm que ler igual.
        ponto_virgula = "Data;Histórico;Valor\n01/05/2026;IFOOD;-45,90\n".encode("utf-8")
        virgula = "Data,Histórico,Valor\n01/05/2026,IFOOD,-45.90\n".encode("utf-8")
        df_pv = _ler_extrato_br(ponto_virgula)
        df_v = _ler_extrato_br(virgula)
        assert df_pv is not None and df_v is not None
        assert df_pv["valor"].iloc[0] == pytest.approx(-45.90)
        assert df_v["valor"].iloc[0] == pytest.approx(-45.90)

    def test_remonta_valor_partido_pela_virgula(self):
        # Arquivo `,`-delimitado em que a vírgula-decimal partiu "-45,90" em duas células.
        # _ler_extrato_br detecta o padrão (inteiro + 1-2 dígitos) e remonta -45.90. Testo aqui
        # direto porque, sem preâmbulo, o _ler_tabular "resolveria" antes (com índice implícito)
        # e a remontagem nunca rodaria — em extrato real o preâmbulo joga pra cá.
        texto = (
            "Data,Histórico,Valor\n"
            "01/05/2026,IFOOD,-45,90\n"
            "02/05/2026,UBER,-12,50\n"
        ).encode("utf-8")
        df = _ler_extrato_br(texto)
        assert df is not None
        assert df["valor"].to_list() == pytest.approx([-45.90, -12.50])
