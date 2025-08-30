# IGCP — Demo de Predicción de Demanda Académica (ML.NET)

Este proyecto implementa un **prototipo end‑to‑end** para estimar:

* **Probabilidad de inscripción** para cada par **(estudiante, curso, período)**.
* **Demanda esperada por curso**, sumando las probabilidades individuales.
* **Plan de secciones sugeridas**, a partir de la demanda esperada y la capacidad por sección.

> En pocas palabras: un modelo **micro** (alumno–curso) + una **capa de agregación** (macro) para ayudar a coordinación a decidir cuántas secciones abrir.

---

## 1) Tecnologías

* **C# / .NET 7**
* **ML.NET 4.x** (paquetes: `Microsoft.ML`, `Microsoft.ML.LightGbm`)
* **System.Data.SqlClient** para prueba rápida de conexión a BD
* **VS Code / CLI**

---

## 2) Requisitos

* .NET 7 SDK instalado
* (Opcional) SQL Server LocalDB (instancias típicas: `MSSQLLocalDB`, `VerbTables`)

Paquetes NuGet (si no están ya en el `.csproj`):

```bat
cd IGCP.ML.Demo
dotnet add package Microsoft.ML
dotnet add package Microsoft.ML.LightGbm
dotnet add package System.Data.SqlClient
```

---

## 3) Estructura del repo (esperada)

```
IGCP-ML-Demo/
├─ IGCP.sln
├─ IGCP.ML.Demo/
│  ├─ IGCP.ML.Demo.csproj
│  ├─ Program.cs
│  ├─ data/
│  │  ├─ train.csv
│  │  └─ candidates_next_term.csv
│  └─ models/ (se crea al correr)
└─ README.md
```

---

## 4) Datos de entrada

### 4.1 `data/train.csv` (supervisado)

Cada fila es un **(student\_id, course\_id, term)** **elegible** con su **label** (1 si se inscribió, 0 si no):

| Columna                | Tipo   | Descripción                                        |
| ---------------------- | ------ | -------------------------------------------------- |
| `student_id`           | string | Id de estudiante (no se usa como feature directa). |
| `course_id`            | string | Id de curso (BDI, REDI, BDII…).                    |
| `term`                 | string | Período académico (ej. `2025-1`).                  |
| `credits`              | float  | Créditos del curso.                                |
| `prereq_count`         | float  | Cantidad de prerrequisitos.                        |
| `is_morning`           | bool   | `true` si (sección/preferencia) es en la mañana.   |
| `is_online`            | bool   | `true` si modalidad online.                        |
| `lag_course_demand_t1` | float  | Demanda histórica del curso en *t‑1*.              |
| `lag_student_load_t1`  | float  | Carga del estudiante en *t‑1* (cursos/créditos).   |
| `label`                | bool   | **Objetivo**: 1 si se inscribió, 0 si no.          |

> **Importante**: incluir **negativos reales** (cursos que podía tomar y **no** tomó) para que el modelo aprenda.

### 4.2 `data/candidates_next_term.csv` (a puntuar)

Mismo layout **sin** la columna `label` — todas las combinaciones **elegibles** del próximo período.

---

## 5) Cómo funciona el modelo

### 5.1 Pipeline (ML.NET)

* **Categóricas**: `course_id`, `term` → `OneHotHashEncoding`.
* **Booleans**: `is_morning`, `is_online` → convertidas a `Single`.
* **Concatenate**: todas las features numéricas/codificadas → columna `Features`.
* **Modelo**: `LightGbm` binario (objetivo logloss) → predice `label`.
* **Split estratificado** train/test: garantiza positivos y negativos en ambos sets.

### 5.2 Predicción

El modelo calcula $p = P(\text{inscribirse}=1 \mid \text{features})$ por fila **(alumno, curso, período)**.

### 5.3 Agregación (macro)

* **Demanda esperada por curso** = suma de probabilidades individuales por `(course_id, term)`.
* **Secciones sugeridas** = `ceil(expected_enrollment / capacidad_por_seccion)` si supera un umbral mínimo.

---

## 6) Instalación y ejecución

```bat
:: 1) Clonar y abrir el proyecto
cd IGCP-ML-Demo
:: 2) (Opcional) Ver proyectos en la solución
dotnet sln IGCP.sln list
:: 3) Entrar al proyecto y restaurar
cd IGCP.ML.Demo
dotnet restore
:: 4) Ejecutar
dotnet run
```

> Si ejecutas desde la raíz del repo: `dotnet run --project .\IGCP.ML.Demo\IGCP.ML.Demo.csproj`

**Notas**

* Si solo cambiaste CSVs: `dotnet run --no-build`.
* Si aparece "Couldn't find a project to run": ejecuta dentro de `IGCP.ML.Demo` o usa `--project`.
* Si `train.csv` está **abierto en Excel**, ciérralo (Windows bloquea el archivo y ML.NET no puede leerlo).

---

## 7) Salidas generadas

1. **`predictions_by_student.csv`**

   * Columnas: `student_id, course_id, term, probability`.
   * `probability` = probabilidad estimada de que el **estudiante** se inscriba en ese **curso** en ese **período**.

2. **`demand_by_course.csv`**

   * Columnas: `course_id, term, expected_enrollment`.
   * `expected_enrollment` = suma de probabilidades individuales (puede ser **decimal**).

3. **`sections_plan.csv`** *(si está habilitado en el código)*

   * Columnas: `course_id, term, expected_enrollment, sections_suggested`.
   * Regla ejemplo: capacidad = **35**, mínimo para abrir = **12**.

4. **`models/enroll_model.zip`**

   * Modelo entrenado para versionar y reusar en inferencia.

---

## 8) Interpretación rápida

* **Predicción micro** (por alumno–curso): ranking por probabilidad; puedes usar umbral (p. ej., 0.5) o Top‑N.
* **Demanda macro** (por curso): usar `expected_enrollment` para planificar **secciones**.
* **Decimales** en demanda = valor **esperado** (p. ej., 83.2 ≈ "esperamos \~83–84").
* Con pocos datos/features, las probabilidades se acercan a la **tasa base** del dataset.

---

## 9) Conexión a Base de Datos (opcional en la demo)

El proyecto incluye una **prueba de conexión** a SQL Server LocalDB.

* Ver instancias disponibles:

```bat
sqllocaldb info
```

* Cadenas típicas:

```text
Server=(localdb)\VerbTables;Database=VerbTables;Trusted_Connection=True;TrustServerCertificate=True;
Server=(localdb)\MSSQLLocalDB;Database=VerbTables;Trusted_Connection=True;TrustServerCertificate=True;
```

* Variable de entorno para sobreescribir sin tocar código:

```bat
setx ML_DEMO_CONNSTR "Server=(localdb)\VerbTables;Database=VerbTables;Trusted_Connection=True;TrustServerCertificate=True;"
```

Si no existe la base en la instancia, créala en SSMS o con:

```sql
CREATE DATABASE VerbTables;
```

---

## 10) Prueba rápida (smoke test)

1. Ver que imprime métricas (AUC‑PR, F1, Accuracy) sin errores.
2. Confirmar archivos generados: `predictions_by_student.csv`, `demand_by_course.csv` (y `sections_plan.csv` si aplica).
3. Comprobar `[BD OK]` o mensaje `[BD ERROR]` informativo.

---

## 11) Troubleshooting

* **`File ... is being used by another process`** → Cierra Excel/editores; Windows bloquea el CSV.
* **LightGBM con pocos datos se cae** (mínimo por hoja) →

  * Solución A: más filas (recomendado),
  * Solución B: bajar `MinimumExampleCountPerLeaf` o usar SDCA temporalmente.
* **AUC no definido (sin positivos/negativos en test)** → usar **split estratificado** (incluido).
* **No conecta a BD** → verifica instancia (`sqllocaldb info`), prueba sin `Database=` y lista `sys.databases`.

---

## 12) Posibles Siguientes pasos

* Generar `candidates_next_term.csv` con **todas** las combinaciones elegibles reales.
* Añadir features: **profesor**, **patrón de días (LM/MJ/S)**, **franja (mañana/tarde/noche)**, **choques de horario**, **historial académico** (GPA, créditos aprobados, repitencias), **modalidad**.
* Integrar con BD real mediante **vistas** (`vw_ml_train`, `vw_ml_candidates`) que emitan exactamente estas columnas.
* Con más histórico: volver a **cross‑validation**, **tuning** de LightGBM, y comparar contra SDCA/FastForest.
* Exponer un **WebAPI** y un **dashboard** para coordinación.
* **Backtesting**: comparar `expected_enrollment` vs. inscritos reales (MAPE por curso) y ajustar reglas/umbral.

---
 
