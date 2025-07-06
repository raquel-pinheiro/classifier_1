# Sales Opportunity Classifier

Este projeto foi desenvolvido para classificar oportunidades de vendas com base em dados históricos e variáveis comerciais, com o objetivo de ajudar o time de vendas a priorizar as oportunidades mais promissoras.

## Objetivo

Classificar oportunidades de vendas como "Potential" (promissoras) ou "Unlikely" (improváveis) com base em dados internos como região, cliente, setor, status da oportunidade, entre outros. A solução foi integrada a dashboards no Power BI para auxiliar decisões comerciais.

## Metodologia

O pipeline de modelagem inclui:

- Tratamento de outliers com Z-Score
- Imputação de dados com KNN
- Codificação de variáveis categóricas
- Interações entre variáveis de negócio
- Normalização dos dados
- Regressão logística com validação cruzada estratificada
- Métricas de avaliação (precision, recall, F1, matriz de confusão)

## Arquivos

- `model_pipeline.py`: script principal com todo o pipeline de preparação, modelagem e avaliação.
- `data_schema.md`: arquivo com as descrições esperadas das colunas do dataset (dados não incluídos por confidencialidade).

## Resultados

- **Accuracy média (cross-validation)**: ~0.9
- **Precision**: ~0.9
- **Recall**: ~0.9
- **F1-Score**: ~0.9

## Observação sobre os dados

Por se tratar de um projeto real aplicado em contexto corporativo, o conjunto de dados original é confidencial e não está incluído neste repositório.  

## Tecnologias utilizadas

- Python (pandas, scikit-learn, statsmodels, numpy, etc.)
- Power BI (integração do resultado final)
- DBT + Snowflake (no pipeline completo, fora do escopo deste repositório)

## Estrutura do Repositório

<pre>
  sales-opportunity-classifier/
  │
  ├── README.md
  ├── model_pipeline.py
  ├── requirements.txt
  └── data_schema.md
</pre>

## Contato

Desenvolvido por [Raquel Pinheiro](https://www.linkedin.com/in/raquel-s-pinheiro/)
Mais projetos em: [github.com/raquel-pinheiro](https://github.com/raquel-pinheiro)
