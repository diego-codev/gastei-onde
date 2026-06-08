# gastei-onde 💸

> Cansei de não saber pra onde foi meu dinheiro no fim do mês, então fiz isso.
> Classificação de gastos com Python + Streamlit.

**Status:**  em construção

Você sobe o CSV do seu extrato bancário, o app **categoriza cada transação**
(alimentação, transporte, lazer, contas fixas, ...) e **destaca os gastos atípicos** —
aquilo que fugiu do seu padrão. Tudo num dashboard simples.

## O problema

Extrato de banco é muito bagunçado: `PAG*IFD`, `TEF UBER`, `PIX ENVIADO`...
Difícil bater o olho e entender pra onde o dinheiro foi. Esse projeto transforma essa
bagunça em categorias legíveis e aponta os gastos fora da curva.

## Stack

- **Python** + **pandas** — manipulação de dados
- **scikit-learn** — classificação multiclasse e detecção de anomalia
- **Streamlit** — app web e dashboard

## Privacidade

O app processa o CSV **100% em memória, durante a sua sessão**. Nada é salvo, nada é
enviado pra lugar nenhum, nada é registrado. Quando você fecha a aba, os dados somem.
Este repositório **não contém dados reais** — apenas dados sintéticos de exemplo,
gerados por código.

## Como rodar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Roadmap

- [ ] Gerador de dados sintéticos
- [ ] Exploração dos dados (EDA)
- [ ] Feature engineering das descrições
- [ ] Baseline por regras
- [ ] Classificador multiclasse (ML)
- [ ] Métricas (precision/recall/AUC) e confiança por predição
- [ ] Detecção de gastos atípicos
- [ ] App Streamlit
- [ ] Testes + CI
- [ ] Deploy no Streamlit Community Cloud

## Licença

MIT — veja [LICENSE](LICENSE).
