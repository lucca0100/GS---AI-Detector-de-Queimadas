# Detector de Queimadas com Dados de Satélite e Inteligência Artificial

## 1. Contexto do problema

Queimadas são eventos ambientais extremos que causam impactos sociais, econômicos e ambientais. Elas afetam a qualidade do ar, a biodiversidade, a agricultura, a saúde pública e o planejamento de ações de defesa civil.

Este projeto utiliza dados relacionados à Economia Espacial para desenvolver um pipeline completo de Inteligência Artificial capaz de classificar uma amostra geográfica como **normal** ou com **indício de queimada**.

## 2. Relação com Economia Espacial

A Economia Espacial envolve o uso de tecnologias, dados e serviços derivados do setor espacial para gerar valor econômico e social. Neste projeto, são utilizados dados de satélite da NASA para monitoramento ambiental de focos de calor.

A principal fonte utilizada foi:

- **NASA FIRMS**: dados de focos de calor e anomalias térmicas detectadas por satélites.

Também foram utilizadas amostras normais geradas dentro de uma área geográfica definida para compor a classe sem indício de queimada.

## 3. Objetivo

Desenvolver um pipeline completo de Machine Learning para classificar amostras em duas classes:

- **Normal**
- **Queimada**

O objetivo é demonstrar, de forma prática, as etapas de coleta de dados, pré-processamento, engenharia de atributos, treinamento, validação, interpretabilidade e deploy de uma solução de IA aplicada a dados espaciais.

## 4. Fontes dos dados

### NASA FIRMS

A NASA FIRMS foi utilizada para coletar dados de focos de calor associados a queimadas. A base contém informações como latitude, longitude, brilho térmico, FRP, sensor, data e confiança da detecção.

### Amostras normais

Foram geradas amostras aleatórias dentro de uma área geográfica definida, representando locais sem associação direta com os focos de calor coletados na API utilizada.

## 5. Metodologia utilizada

A metodologia do projeto foi organizada como um pipeline completo de Machine Learning, contendo as seguintes etapas:

1. Coleta de dados de queimadas por meio da API NASA FIRMS.
2. Geração de amostras normais para compor a classe sem indício de queimada.
3. Padronização das colunas das diferentes fontes de dados.
4. Tratamento de valores ausentes e conversão de variáveis categóricas.
5. Engenharia de atributos com criação de variáveis temporais e indicadores térmicos.
6. Balanceamento das classes para reduzir viés no treinamento.
7. Treinamento de três modelos de classificação: Logistic Regression, Random Forest e XGBoost.
8. Avaliação dos modelos com métricas de desempenho.
9. Escolha do melhor modelo com base no F1-score da classe queimada.
10. Aplicação de SHAP para interpretabilidade.
11. Desenvolvimento e deploy da aplicação com Streamlit.

## 6. Estrutura do dataset

O dataset final foi balanceado com:

- 500 amostras da classe normal
- 500 amostras da classe queimada

Total:

- 1.000 linhas
- Mais de 10 colunas

Principais variáveis utilizadas:

- latitude
- longitude
- bright_ti4
- bright_ti5
- frp
- scan
- track
- confidence_num
- daynight_num
- ano
- mes
- dia
- dia_do_ano
- risco_fogo
- diferenca_brilho
- frp_por_brilho
- periodo_seco

## 7. Explicação das colunas do dataset

| Coluna | Descrição |
|---|---|
| `latitude` | Coordenada geográfica de latitude do evento ou amostra. |
| `longitude` | Coordenada geográfica de longitude do evento ou amostra. |
| `data` | Data da observação ou da amostra. |
| `bright_ti4` | Temperatura de brilho detectada pelo sensor VIIRS na banda I4, associada à intensidade térmica do foco de calor. |
| `bright_ti5` | Temperatura de brilho detectada pelo sensor VIIRS na banda I5, usada como apoio na análise térmica. |
| `frp` | Fire Radiative Power, medida da potência radiativa do fogo. Valores maiores indicam maior intensidade do foco de calor. |
| `scan` | Medida relacionada à largura do pixel observado pelo satélite. |
| `track` | Medida relacionada ao comprimento do pixel observado pelo satélite. |
| `confidence_num` | Valor numérico criado a partir da confiança da detecção da FIRMS. Valores como low, nominal e high foram convertidos para números. |
| `daynight_num` | Indica se a observação ocorreu durante o dia ou à noite. Foi codificada como 1 para dia e 0 para noite. |
| `ano` | Ano extraído da data da observação. |
| `mes` | Mês extraído da data da observação. |
| `dia` | Dia do mês extraído da data da observação. |
| `dia_do_ano` | Dia correspondente dentro do ano, variando de 1 a 365 ou 366. |
| `risco_fogo` | Variável criada pela multiplicação entre `bright_ti4` e `confidence_num`, representando um indicador derivado de risco térmico. |
| `diferenca_brilho` | Diferença entre `bright_ti4` e `bright_ti5`, usada para capturar variações térmicas entre bandas. |
| `frp_por_brilho` | Relação entre a potência radiativa do fogo e o brilho térmico. |
| `periodo_seco` | Indicador criado a partir do mês da observação. Recebe 1 para meses de julho a outubro e 0 para os demais meses. |
| `classe` | Variável alvo do modelo. Pode assumir os valores `normal` ou `queimada`. |

## 8. Pré-processamento

As etapas de pré-processamento incluíram:

- Leitura dos dados brutos da API NASA FIRMS.
- Geração de amostras normais.
- Padronização dos nomes das colunas.
- Conversão de datas.
- Conversão da variável de confiança da FIRMS para formato numérico.
- Tratamento de valores ausentes.
- Balanceamento das classes.
- Criação do dataset processado.

A coluna de confiança da FIRMS podia conter valores categóricos, como `l`, `n` e `h`. Esses valores foram convertidos para números:

- `l` ou low: 30
- `n` ou nominal: 60
- `h` ou high: 90

Para as amostras normais, as variáveis térmicas foram preenchidas com zero, pois essas amostras representam pontos sem indício direto de foco de calor na base utilizada.

## 9. Engenharia de atributos

Foram criadas variáveis derivadas para melhorar a capacidade preditiva dos modelos:

- `risco_fogo = bright_ti4 * confidence_num`
- `diferenca_brilho = bright_ti4 - bright_ti5`
- `frp_por_brilho = frp / (bright_ti4 + 1)`
- `periodo_seco = 1` para meses de julho a outubro; caso contrário, `0`
- `ano`, `mes`, `dia` e `dia_do_ano` extraídos da data

Essas variáveis ajudam o modelo a capturar padrões térmicos e temporais relacionados aos focos de calor.

## 10. Modelos utilizados

Foram treinados e comparados três modelos de Machine Learning:

- Logistic Regression
- Random Forest
- XGBoost

O objetivo foi comparar um modelo linear simples com modelos baseados em árvores e ensembles.

## 11. Resultados

Os resultados obtidos foram:

| Modelo | Accuracy | F1 Queimada | Precision Queimada | Recall Queimada | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| XGBoost | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

O melhor modelo salvo para a aplicação foi o **Random Forest**, por ser um modelo robusto, compatível com SHAP e adequado para dados tabulares.

## 12. Interpretabilidade com SHAP

Foi utilizado SHAP para analisar a importância das variáveis no modelo final.

As variáveis mais influentes foram:

- bright_ti5
- track
- frp_por_brilho
- frp
- risco_fogo
- bright_ti4
- scan
- confidence_num
- diferenca_brilho

A análise mostrou que o modelo utiliza principalmente variáveis térmicas e de intensidade do foco de calor para identificar a classe queimada.

## 13. Aplicação Streamlit

Foi desenvolvida uma aplicação interativa com Streamlit, permitindo:

- Inserir dados manualmente.
- Classificar uma região como normal ou queimada.
- Visualizar probabilidades por classe.
- Enviar um CSV para classificação em lote.
- Baixar o resultado das previsões.

## 14. Como utilizar a aplicação

### Previsão manual

Na aba **Previsão manual**, o usuário pode preencher os valores das variáveis utilizadas pelo modelo, como latitude, longitude, brilho térmico, FRP, confiança, data e período da observação.

Após preencher os campos, basta clicar em **Classificar região**. O app retorna:

- Classe prevista: normal ou queimada.
- Probabilidade de cada classe.
- Gráfico de probabilidades.

### Previsão por CSV

Na aba **Previsão por CSV**, o usuário pode enviar um arquivo contendo várias amostras para classificação em lote.

O CSV deve conter as seguintes colunas:

```text
latitude, longitude, bright_ti4, bright_ti5, frp, scan, track, confidence_num, daynight_num, ano, mes, dia, dia_do_ano, risco_fogo, diferenca_brilho, frp_por_brilho, periodo_seco
```

As colunas derivadas `risco_fogo`, `diferenca_brilho`, `frp_por_brilho` e `periodo_seco` também podem ser calculadas automaticamente pelo app quando as colunas de origem estiverem presentes.

Após o envio do arquivo, o app gera uma tabela com:

- Dados originais enviados.
- Classe prevista para cada linha.
- Probabilidade da classe normal.
- Probabilidade da classe queimada.

Também é possível baixar o resultado em um novo arquivo CSV.

## 15. Como executar o projeto localmente

Clone o repositório:

```bash
git clone LINK_DO_REPOSITORIO
cd NOME_DA_PASTA_DO_PROJETO
```

Crie o ambiente virtual:

```bash
python -m venv venv
```

Ative o ambiente virtual:

No Windows PowerShell/CMD:

```bash
venv\Scripts\activate
```

No Git Bash:

```bash
source venv/Scripts/activate
```

No Linux/Mac:

```bash
source venv/bin/activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Execute a aplicação:

```bash
streamlit run app/streamlit_app.py
```

Acesse no navegador:

```text
http://localhost:8501
```

## 16. Estrutura do projeto

```text
.
├── app/
│   └── streamlit_app.py
├── data/
│   └── processed/
│       └── dataset_queimadas.csv
├── models/
│   ├── modelo_final.pkl
│   ├── label_encoder.pkl
│   └── colunas_modelo.pkl
├── reports/
│   ├── resultados_modelos.csv
│   ├── shap_importancia_global.csv
│   ├── shap_importancia_global.png
│   └── shap_summary_queimada.png
├── src/
│   ├── testar_firms.py
│   ├── gerar_normais.py
│   ├── montar_dataset.py
│   ├── treinar_modelos.py
│   ├── interpretabilidade_shap.py
│   └── verificar_dados.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 17. Limitações

Este projeto possui finalidade acadêmica e demonstrativa. A aplicação não substitui sistemas oficiais de monitoramento ambiental, defesa civil ou análise especializada.

Como as amostras normais foram geradas artificialmente e tiveram variáveis térmicas preenchidas com zero, os modelos conseguiram separar as classes com desempenho muito alto. Isso indica que o problema, na forma atual, é fortemente separável. Em uma aplicação operacional real, seria necessário validar a classe normal contra bases oficiais e incluir amostras mais difíceis, como regiões com calor urbano, solo exposto ou atividades industriais.

## 18. Trabalhos futuros

Como melhorias futuras, podem ser consideradas as seguintes evoluções:

- Enriquecer o dataset com dados climáticos, como chuva, temperatura, umidade e vento.
- Validar melhor a classe normal com dados espaciais e temporais de ausência de foco de calor.
- Utilizar imagens reais de satélite, como Sentinel-2 ou Landsat.
- Aplicar redes neurais convolucionais para classificação direta de imagens.
- Criar mapas interativos para visualização espacial dos focos de queimada.
- Incluir dados históricos de diferentes períodos do ano e diferentes regiões.

## 19. Links

- Link do repositório: adicionar aqui
- Link da aplicação em funcionamento: adicionar aqui
