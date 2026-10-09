# INF-8239 · Unidad 02 · Clasificación de sentimiento en tweets en español

Pipeline reproducible para **descargar, auditar y clasificar** un corpus de tweets en español con TF-IDF, Complement Naive Bayes y regresión logística, más un análisis de errores del modelo seleccionado.

Corresponde a los laboratorios **LAB04** (búsqueda, descarga y auditoría de un corpus) y **LAB05** (TF-IDF, Naive Bayes y modelo lineal) de la Unidad 02: *Procesamiento de Lenguaje Natural*.

\---

## Tabla de contenidos

1. [Requisitos](#requisitos)
2. [Instalación](#instalación)
3. [Configuración](#configuración)
4. [Corpus](#corpus)
5. [Ejecución](#ejecución)
6. [Resultados](#resultados)
7. [Análisis de errores](#análisis-de-errores)
8. [Limitaciones](#limitaciones)
9. [Estructura del repositorio](#estructura-del-repositorio)
10. [Solución de problemas](#solución-de-problemas)
11. [Próximos pasos](#próximos-pasos)
12. [Referencias](#referencias)

\---

## Requisitos

|Componente|Versión / detalle|
|-|-|
|Sistema operativo|Windows 11 (PowerShell)|
|Python|**3.13** (ver nota)|
|Gestor de entorno|[`uv`](https://docs.astral.sh/uv/)|
|Control de versiones|Git|
|Dependencias principales|pandas, pyarrow, scikit-learn, requests, python-dotenv, joblib, matplotlib, networkx, gensim|

> \\\*\\\*Nota sobre Python 3.14:\\\*\\\* `gensim` 4.4.0 solo publica wheels para Windows hasta Python 3.13. Con Python 3.14, pip intenta compilarlo desde el código fuente y falla en `longintrepr.h`. Por eso el proyecto usa Python 3.13.

## Instalación

```powershell
git clone https://github.com/<usuario>/DSC2.git
cd DSC2
uv sync
```

`uv sync` crea el entorno virtual e instala las versiones exactas registradas en `uv.lock`.

## Configuración

El proyecto se configura mediante un archivo `.env` en la raíz. Ese archivo **no se versiona**; se crea a partir de la plantilla:

```powershell
Copy-Item .env.example .env
```

|Variable|Valor usado|Descripción|
|-|-|-|
|`DATA\\\_SOURCE`|`url`|`url` descarga el corpus; `local` usa un archivo existente|
|`DATASET\\\_PATH`|`data/raw/dataset.csv`|Archivo que leen los scripts de auditoría y entrenamiento|
|`DATASET\\\_URL`|Split `train` (Parquet)|Fuente principal del corpus|
|`DATASET\\\_URL\\\_VALIDATION`|Split `validation` (Parquet)|Partición oficial de validación|
|`DATASET\\\_URL\\\_TEST`|Split `test` (Parquet)|Partición oficial de prueba; se reserva para la evaluación final|
|`TEXT\\\_COLUMN`|`text`|Columna de texto|
|`TARGET\\\_COLUMN`|`label`|Columna objetivo|
|`RANDOM\\\_STATE`|`42`|Semilla para todas las particiones y modelos|

Las URLs apuntan a la conversión Parquet de Hugging Face:
`https://huggingface.co/api/datasets/cardiffnlp/tweet\\\_sentiment\\\_multilingual/parquet/spanish/<split>/0.parquet`

## Corpus

**Tweet Sentiment Multilingual**, subconjunto `spanish` ([`cardiffnlp/tweet\\\_sentiment\\\_multilingual`](https://huggingface.co/datasets/cardiffnlp/tweet_sentiment_multilingual)).

|Campo|Valor|
|-|-|
|Idioma|Español (sin variante regional específica)|
|Tamaño|3,030 tweets: 1,838 train · 323 validation · 869 test|
|Etiquetas|`0` = negativo · `1` = neutral · `2` = positivo|
|Origen|Campañas SemEval/TASS anteriores a 2022|
|Anonimización|Usuarios reemplazados por `@user`; enlaces por `http`|
|Licencia|CC BY 3.0, sujeta además a los términos de servicio de Twitter/X|
|Cita|Barbieri, Espinosa Anke y Camacho-Collados (2022). *XLM-T*. LREC 2022|

**Procedencia.** `download\\\_data.py` guarda la copia original en Parquet y su SHA-256 como evidencia, y la convierte a CSV (UTF-8) para que la lean los demás scripts.

## Ejecución

Ejecutar en este orden desde la raíz del repositorio:

```powershell
# 1. Verificar el entorno
uv sync
uv run pytest -q

# 2. Descargar el corpus (train, validation y test)
uv run python scripts/download\\\_data.py

# 3. Auditar: nulos, duplicados, distribución de clases y longitudes
uv run python scripts/audit\\\_data.py

# 4. Entrenar y evaluar los tres pipelines
uv run python scripts/train\\\_text.py
```

`train\\\_text.py` crea **una única partición estratificada 75/25 del split `train`** con semilla 42 y entrena tres pipelines. El vectorizador TF-IDF (unigramas y bigramas) permanece **dentro de cada pipeline** para evitar fuga de información:

|Pipeline|Rol|
|-|-|
|`DummyClassifier(strategy="most\\\_frequent")`|Baseline|
|`ComplementNB`|Modelo probabilístico|
|`LogisticRegression(class\\\_weight="balanced")`|Modelo lineal; es el que se guarda|

**Artefactos generados**

|Archivo|Contenido|
|-|-|
|`reports/text\\\_metrics.csv`|F1 macro de cada modelo|
|`reports/confusion\\\_text.png`|Matriz de confusión de la regresión logística|
|`reports/error\\\_analysis.csv`|Errores de la regresión logística (se sobrescribe en cada ejecución)|
|`models/text\\\_model.joblib`|Pipeline entrenado de regresión logística|

## Resultados

Evaluación sobre 460 tweets (25% del split `train`).

### F1 macro

|Modelo|F1 macro|Accuracy|
|-|-|-|
|Dummy (baseline)|0.166|0.33|
|Complement Naive Bayes|0.502|0.51|
|**Regresión logística**|**0.520**|**0.53**|

### Métricas por clase (regresión logística)

|Clase|Precision|Recall|F1|Support|
|-|-|-|-|-|
|0 · negativo|0.54|0.59|0.57|153|
|1 · neutral|**0.47**|**0.37**|**0.41**|153|
|2 · positivo|0.55|0.62|0.58|154|

### Matriz de confusión (regresión logística)

Filas = clase real · columnas = clase predicha.

||Pred. negativo|Pred. neutral|Pred. positivo|
|-|-|-|-|
|**Real negativo**|**91**|32|30|
|**Real neutral**|48|**56**|49|
|**Real positivo**|28|31|**95**|

### Interpretación

* **¿Supera el baseline?** Sí. Ambos modelos triplican el F1 macro del baseline. La ventaja de la regresión logística sobre Naive Bayes (+0.018) es pequeña para una sola partición de 460 tweets y no basta para afirmar que sea mejor.
* **¿Qué clase se pierde?** La neutral: recall de 0.37, con **97 falsos negativos** (48 predichos como negativos y 49 como positivos).
* **¿Qué predicción es poco confiable?** "Neutral", con precision de 0.47: de 119 predicciones neutrales, 63 son erróneas. Además, cerca del 28% de las predicciones positivas y del 29% de las negativas son en realidad tweets neutrales.
* **¿Existe desbalance?** No. El support es 153 / 153 / 154; por eso el accuracy y el F1 macro casi coinciden.
* **¿Puede generalizar?** Solo dentro del dominio del corpus. Ver [Limitaciones](#limitaciones).

> La regresión logística obtuvo el mejor F1 macro (0.520). Sin embargo, la clase \\\*\\\*neutral\\\*\\\* presentó el menor recall (0.37), con \\\*\\\*97 falsos negativos\\\*\\\*. En este contexto, ese error implica que el modelo atribuye opiniones a tweets que solo informan, lo que infla artificialmente tanto el apoyo como el rechazo. Por ello proponemos analizar los errores de la clase neutral, comparar los modelos mediante validación cruzada repetida y evaluar el modelo elegido una sola vez sobre el test oficial antes de publicar.

## Análisis de errores

Se clasificaron **los primeros 40 de los 218 errores** de `reports/error\\\_analysis.csv`. Se tomó un bloque fijo para no elegir casos a mano. **No se modificó ninguna etiqueta ni predicción.**

|Categoría|n|%|IC 95% (Wilson)|
|-|-|-|-|
|**Negación**|10|25.0%|14.2% – 40.2%|
|Ambigüedad|9|22.5%|12.3% – 37.5%|
|Ironía|6|15.0%|7.1% – 29.1%|
|Etiqueta discutible|5|12.5%|5.5% – 26.1%|
|Señal clara no aprendida|4|10.0%|4.0% – 23.1%|
|Escritura informal|2|5.0%|1.4% – 16.5%|
|Dialecto|1|2.5%|0.4% – 12.9%|
|Texto insuficiente|1|2.5%|0.4% – 12.9%|
|Fuera del dominio|1|2.5%|0.4% – 12.9%|
|Sesgo temático|1|2.5%|0.4% – 12.9%|

**Hallazgos**

1. **Negación (hipótesis principal).** En **9 de 10** casos el modelo predijo *negativo* aunque la frase negara algo malo o expresara una duda. Ejemplo: *"Me hace sentir tan bien… no sentir absolutamente nada por él"* (real: positivo).
2. **Escritura informal.** El patrón por defecto de `TfidfVectorizer` (`\\\\b\\\\w\\\\w+\\\\b`) descarta los emojis y no normaliza alargamientos como "siiiii".
3. **Ironía, ambigüedad y dialecto.** Requieren contexto o conocimiento del mundo; son una limitación de los modelos de bolsa de palabras.
4. **Etiqueta discutible.** Posible ruido de anotación; se reporta y no se corrige.

**Archivos:** `reports/error\\\_analysis\\\_clasificado.csv` (detalle de los 40 casos) y `reports/analisis\\\_errores\\\_LAB05.xlsx` (resumen con fórmulas, hipótesis y metodología).

## Limitaciones

* Los resultados aplican a **tweets en español** de campañas SemEval/TASS anteriores a 2022. No representan a la población general, a una variante regional concreta (por ejemplo, el español dominicano), a otros registros de texto ni al lenguaje posterior a la recolección.
* La evaluación usa **una sola partición interna** de 460 tweets. El **test oficial (869 tweets) no se ha utilizado**.
* Con un F1 macro de 0.52, el modelo no es apto para decisiones automáticas.
* El análisis de errores se basa en 40 casos y en el criterio de un solo anotador.

## Estructura del repositorio

```
DSC2/
├── .env.example              # Plantilla de configuración (el .env real no se versiona)
├── pyproject.toml
├── uv.lock
├── src/
│   └── inf8239\\\_u02/
│       ├── \\\_\\\_init\\\_\\\_.py
│       ├── config.py         # Carga .env y expone `settings`
│       ├── data.py           # sha256, load\\\_dataset (CSV/Parquet), validate\\\_dataframe
│       └── modeling.py       # Pipelines TF-IDF + Dummy / ComplementNB / LogisticRegression
├── scripts/
│   ├── download\\\_data.py      # Descarga splits, guarda Parquet + CSV y SHA-256
│   ├── audit\\\_data.py         # Auditoría del corpus
│   ├── train\\\_text.py         # Entrenamiento y evaluación
│   └── embeddings\\\_network.py # Word2Vec y red de demostración (LAB06)
├── tests/
├── data/raw/                 # Generado por download\\\_data.py (no versionado)
├── models/                   # Generado por train\\\_text.py
└── reports/                  # Métricas, matriz de confusión y análisis de errores
```

## Solución de problemas

|Síntoma|Causa|Solución|
|-|-|-|
|`Activate.ps1 cannot be loaded because running scripts is disabled`|Política de ejecución de PowerShell|`Set-ExecutionPolicy RemoteSigned -Scope CurrentUser`, o usar `uv run` / `.\\\\.venv\\\\Scripts\\\\python.exe` sin activar|
|`Failed building wheel for gensim`|Python 3.14 sin wheel de gensim|Usar Python 3.13|
|`git : The term 'git' is not recognized`|La terminal conserva el `PATH` anterior a la instalación|Cerrar y reabrir la terminal (o VS Code)|
|`requirements.txt` ilegible tras `pip freeze > ...`|PowerShell 5.1 escribe en UTF-16|`pip freeze \| Out-File -Encoding utf8 requirements.txt`|
|Tweets como "NA" o "None" aparecen como nulos|`pd.read\\\_csv` los interpreta como faltantes|`load\\\_dataset()` usa `keep\\\_default\\\_na=False, na\\\_values=\\\[""]`|

## Próximos pasos

* \[ ] Probar la hipótesis de negación: marcar su alcance en el preprocesamiento y comparar F1 macro con validación cruzada repetida en `train`.
* \[ ] Probar un tokenizador que conserve emojis y normalice alargamientos.
* \[ ] Comparar regresión logística y Naive Bayes con mediciones repetidas.
* \[ ] Evaluar el modelo elegido **una sola vez** sobre el test oficial.
* \[ ] LAB06: embeddings y análisis de red con `scripts/embeddings\\\_network.py`.

## Referencias

* Barbieri, F., Espinosa Anke, L. y Camacho-Collados, J. (2022). *XLM-T: Multilingual Language Models in Twitter for Sentiment Analysis and Beyond*. Proceedings of LREC 2022, 258–266. https://aclanthology.org/2022.lrec-1.27
* Dataset: https://huggingface.co/datasets/cardiffnlp/tweet\_sentiment\_multilingual

\---

**Autor:** Hairo Fermin (Asistido de IA Claude \[Antrhopic]) · Universidad Autónoma de Santo Domingo (UASD) · INF-8239, Unidad 02

