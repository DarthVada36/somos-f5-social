# Proyecto de Inclusión Económica: Predicción de Bajos Ingresos

## 🌍 Introducción

A veces, los datos antiguos nos hablan de problemas muy actuales.
El Adult Income Dataset —registrado en 1994— refleja una sociedad donde factores como el género, la educación o el tipo de trabajo determinaban el acceso a mejores ingresos.
Han pasado más de 30 años, pero las desigualdades que muestra no pertenecen solo al pasado: aún se repiten en distintas formas y rincones del mundo.

Este proyecto, “Explorador de Sesgos: Predicción de Bajos Ingresos”, nace con un propósito doble:
enseñar cómo aplicar Machine Learning ético y, al mismo tiempo, invitar a reflexionar sobre la justicia social en los datos.

Inspirado por la misión de Somos F5, busca demostrar que la Inteligencia Artificial tiene sentido solo cuando amplía las oportunidades de quienes históricamente han tenido menos acceso al mundo digital.

A través del análisis de este dataset, propongo no solo predecir quiénes podrían estar en riesgo de bajos ingresos, sino también imaginar cómo la formación, la educación y el acompañamiento pueden cambiar esas trayectorias.
Porque el aprendizaje —cuando se da con propósito— puede ser la herramienta más poderosa para cerrar brechas y abrir futuros.



[![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.2-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Descripción del Proyecto

Este proyecto utiliza **Machine Learning ético** para identificar personas en riesgo de **bajos ingresos persistentes** (≤50K USD/año) y proponer **intervenciones personalizadas de inclusión social** basadas en datos.

### 🎯 Objetivos

1. **Predecir** qué factores determinan el acceso a ingresos altos
2. **Detectar** brechas de desigualdad (género, edad, ocupación, educación)
3. **Proponer** intervenciones personalizadas (becas, upskilling, reconversión profesional)
4. **Evaluar** el impacto de programas de inclusión mediante A/B testing simulado
5. **Garantizar** equidad algorítmica (análisis de fairness y mitigación de sesgos)

---

## 📊 Dataset

**Nombre**: Adult Income Dataset  
**Fuente**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult)  
**Licencia**: CC BY 4.0 (uso libre con atribución)  
**Tamaño**: 32,561 registros × 15 columnas  
**Origen**: Censo de EE.UU. (1994)

### Columnas principales:

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `age` | Numérica | Edad de la persona |
| `education_num` | Numérica | Años de educación formal |
| `occupation` | Categórica | Tipo de trabajo (14 categorías) |
| `sex` | Categórica | Género (Male/Female) |
| `hours_per_week` | Numérica | Horas trabajadas por semana |
| `income` | **Target** | Ingresos anuales: ≤50K o >50K USD |

### Características del dataset:
- ✅ Dataset real de censo poblacional
- ✅ Variables socioeconómicas relevantes para inclusión
- ⚠️ Desbalanceado: ~76% clase ≤50K, ~24% clase >50K
- ⚠️ Datos de 1994 (considerar contexto histórico)

---

## 🚀 Instalación y Configuración

### Prerrequisitos

- Python 3.11 o superior
- pip (gestor de paquetes)
- Git
- [Dataset y modelos](https://drive.google.com/drive/folders/18sHN9cfRIVyeIsLApy1SSzLycMCkmXxB?usp=sharing)  

### Pasos de instalación

1. **Clonar el repositorio**
```bash
git clone https://github.com/DarthVada36/somos-f5-social.git
cd somos-f5-social
```

2. **Crear entorno virtual** (recomendado)
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **(Opcional) Configurar Groq API para LLM real**
```bash
# Ver instrucciones completas en GROQ_SETUP.md
cp .env.example .env
# Edita .env y añade tu API key de https://console.groq.com/keys
```

5. **Verificar instalación**
```bash
python -c "import sklearn; import pandas; print('✓ Todo instalado correctamente')"
```

---

## 📂 Estructura del Proyecto

```
somos-f5-social/
├── README.md                        # Este archivo (con guía Groq integrada)
├── requirements.txt                 # Dependencias del proyecto
├── CHECKLIST_PROYECTO.md            # Validación de requisitos
├── app.py                           # � Aplicación web Flask (Mock o Groq LLM)
├── .env.example                     # 🔑 Plantilla para API keys
├── templates/                       # � HTML para Flask
│   ├── index.html                   # Formulario de entrada
│   ├── resultado.html               # Resultados + LLM chatbot
│   ├── sobre.html                   # Info del proyecto
│   └── error.html                   # Página de error
├── eda/
│   ├── eda_somos.ipynb             # 📊 Análisis Exploratorio (EDA)
│   ├── model_somos.ipynb           # 🤖 Modelado + Fairness + A/B Test
│   ├── models/                      # 💾 Modelos guardados (joblib)
│   │   ├── best_model_gradient_boosting.pkl
│   │   └── model_metadata_gradient_boosting.pkl
│   └── data/
│       ├── adult.csv               # Dataset original
│       └── processed/
│           ├── adult_clean.csv     # Datos limpios (legible)
│           └── adult_clean_model.csv  # Datos para ML (one-hot encoded)
```

---

## 🎓 ¿Qué aprenderás ejecutando este proyecto?

### 1️⃣ **Análisis Exploratorio de Datos (EDA)**
- Cómo limpiar y preparar datos reales con valores faltantes
- Técnicas de feature engineering (agrupación de ocupaciones, flags de missing)
- Storytelling con datos: visualizaciones que comunican inequidad
- Identificación de patrones de desigualdad (género, educación, ocupación)

### 2️⃣ **Machine Learning Ético**
- Entrenar múltiples modelos (Logistic Regression, Random Forest, Gradient Boosting)
- Manejo de clases desbalanceadas (class_weight='balanced', stratify)
- Selección de modelos con métricas apropiadas (AUC-ROC, F1-score)
- Interpretación de resultados (matrices de confusión, ROC curves)

### 3️⃣ **Fairness Analysis (Equidad Algorítmica)**
- Detectar sesgos en modelos ML (brecha de género, discriminación etaria)
- Calcular métricas de fairness (TPR, FPR, Demographic Parity)
- Proponer estrategias de mitigación (re-weighting, threshold optimization)
- Garantizar que la IA no perpetúe discriminación histórica

### 4️⃣ **Sistema de Recomendaciones**
- Diseñar motor basado en reglas para intervenciones personalizadas
- Priorizar recursos según nivel de riesgo
- Conectar predicciones con acciones concretas (becas, formación, mentoría)

### 5️⃣ **A/B Testing y Evaluación de Impacto**
- Simular impacto de programas sociales antes de implementarlos
- Calcular métricas de negocio (lift, conversiones, ROI)
- Visualizar diferencias entre grupo control y tratamiento

---

## ▶️ Cómo Ejecutar el Proyecto

### Opción 1: Jupyter Notebook (Recomendado)

1. **Iniciar Jupyter**
```bash
jupyter notebook
```

2. **Abrir notebooks en orden:**
   - `eda/eda_somos.ipynb` → EDA completo con storytelling
   - `eda/model_somos.ipynb` → Modelado + Fairness + Recomendaciones

3. **Ejecutar celdas secuencialmente** (Shift + Enter)

### Opción 2: VS Code con Jupyter Extension

1. Abrir VS Code en la carpeta del proyecto
2. Instalar extensión "Jupyter" si no la tienes
3. Abrir `.ipynb` y ejecutar celdas con el botón ▶️

### Opción 3: Google Colab

1. Subir notebooks a Google Drive
2. Abrir con Google Colab
3. Subir `adult.csv` cuando se solicite

### 🌐 Opción 4: Aplicación Web Flask (Demo Interactivo)

**¡NUEVA! Interfaz web profesional con predicciones + LLM (Mock o Groq API)**

#### 📋 Prerequisitos

1. **Entrenar el modelo primero**
```bash
# Ejecuta model_somos.ipynb hasta la sección de guardar modelo
# Esto genera: eda/models/best_model_gradient_boosting.pkl
```

2. **Instalar dependencias**
```bash
pip install flask markdown2 groq python-dotenv
```

---

#### 🚀 Inicio Rápido (Modo Mock LLM)

```bash
python app.py
# Abrir en navegador: http://localhost:5000
```

**Funcionalidades:**
- ✅ Formulario interactivo (edad, educación, género, ocupación, horas)
- ✅ Predicción de riesgo con modelo ML real
- ✅ Recomendaciones personalizadas con Mock LLM (por defecto)
- ✅ Formato chatbot profesional con burbujas de conversación
- ✅ Interfaz elegante con gradientes y animaciones
- ✅ Página "Sobre" con info del proyecto

---

#### 🤖 Modo Avanzado: Groq API (LLM Real)

La aplicación soporta **dos modos de operación**:

| Modo | Configuración | Calidad | Internet | Costo |
|------|--------------|---------|----------|-------|
| **Mock LLM** | ✅ Ninguna | Reglas Python | ❌ No requiere | 💰 Gratis |
| **Groq API** | 🔑 API Key | 🧠 llama-3.3-70b | ✅ Requiere | 💸 Gratis* |

*Groq es gratis: 30 req/min, 6,000 req/día, sin tarjeta de crédito.

##### 🔧 Configurar Groq API (Opcional):

**Paso 1: Obtener API Key**
1. Ve a https://console.groq.com/
2. Crea cuenta (email + contraseña)
3. Genera tu API key en https://console.groq.com/keys
4. Copia la clave (empieza con `gsk_...`)

**Paso 2: Configurar en la App**

**Opción A - Archivo .env (recomendado):**
```bash
# 1. Copia el ejemplo
cp .env.example .env

# 2. Edita .env y añade tu clave
GROQ_API_KEY=gsk_tu_clave_real_aqui

# 3. Reinicia la app
python app.py
```

**Opción B - Variable de entorno temporal:**

Windows (PowerShell):
```powershell
$env:GROQ_API_KEY="gsk_tu_clave_aqui"
python app.py
```

Windows (CMD):
```cmd
set GROQ_API_KEY=gsk_tu_clave_aqui
python app.py
```

Linux/Mac:
```bash
export GROQ_API_KEY="gsk_tu_clave_aqui"
python app.py
```

**Verificación:**

Cuando inicies la app, verás:
- ✅ **Con Groq**: `"✅ Groq API configurado correctamente"`
- ⚠️ **Sin Groq**: `"⚠️ Usando Mock LLM (sin API real)"`

En la página web (resultado):
- **Groq activo**: Badge verde "✅ LLM Real: Groq API (llama-3.3-70b-versatile)"
- **Mock activo**: Badge amarillo "ℹ️ Modo: Respuestas generadas con lógica personalizada"

##### 🔒 Seguridad

- ✅ `.env` está en `.gitignore` (no se sube a GitHub)
- ✅ API key se carga desde variable de entorno
- ✅ Sin base de datos = sin riesgo de fugas de datos
- ❌ **Nunca** hardcodear la API key en el código
- ❌ **Nunca** subir `.env` a repositorios públicos

##### 🐛 Troubleshooting

| Problema | Solución |
|----------|----------|
| Error 401 "Invalid API Key" | Verifica que la clave sea correcta y esté en `.env` |
| Mock en vez de Groq | Reinicia la app después de configurar `.env` |
| `groq` no instalado | `pip install groq python-dotenv` |
| Sin respuesta del LLM | Revisa límites (30 req/min). Espera 1 minuto |

##### 📊 Comparación Mock vs Groq

**Cuándo usar Mock LLM:**
- ✅ Demos rápidas y pruebas locales
- ✅ Sin conexión a internet
- ✅ Proyectos académicos (suficiente para demostrar concepto)
- ✅ No requieres respuestas muy naturales

**Cuándo usar Groq API:**
- ⭐ Presentaciones profesionales
- ⭐ Evaluaciones técnicas
- ⭐ Necesitas respuestas contextuales y naturales
- ⭐ Quieres demostrar integración con LLMs reales

---

## ⚙️ Pipeline del Proyecto

### 📊 Fase 1: EDA (`eda_somos.ipynb`)

**Input**: `adult.csv` (32,561 filas)  
**Output**: `adult_clean.csv`, `adult_clean_model.csv`

**Pasos ejecutados:**
1. Carga y exploración inicial
2. Limpieza de valores faltantes y normalización
3. Feature engineering de ocupación (occ_group, occupation_missing, occ_freq)
4. One-hot encoding para modelado
5. **5 visualizaciones con storytelling:**
   - Brecha de género en ingresos
   - Impacto de la educación
   - Desigualdad ocupacional
   - Trampa del trabajo intensivo
   - Población sin ocupación (vulnerable)

**Hallazgos clave:**
- 📉 Brecha de género: ~15 puntos porcentuales
- 📚 Educación universitaria aumenta probabilidad de >50K en +40%
- 💼 Sectores Service/Manual concentran bajos ingresos
- ⏰ Más horas trabajadas ≠ mayores ingresos (calidad > cantidad)

---

### 🤖 Fase 2: Modelado (`model_somos.ipynb`)

**Input**: `adult_clean_model.csv`  
**Output**: Modelos entrenados, métricas, recomendaciones

#### Sección 1: Entrenamiento de Modelos
- 3 algoritmos comparados:
  - Logistic Regression (baseline interpretable)
  - Random Forest (ensemble robusto)
  - Gradient Boosting (máximo rendimiento)
- Train/test split: 80/20 con `stratify=y`
- Semilla fija: `random_state=42`

#### Sección 2: Fairness Analysis
- Métricas de equidad por género, edad y ocupación
- Detección de sesgos (TPR, FPR, Demographic Parity)
- Visualización de brechas algorítmicas
- **Resultado**: Sesgos detectados → requiere mitigación antes de deployment

#### Sección 3: Sistema de Recomendaciones
- Clasificación de riesgo: Alto / Medio / Bajo
- Motor basado en reglas con 7 tipos de intervenciones:
  - 📚 Becas educativas
  - 🔧 Reconversión profesional
  - 💼 Inserción laboral
  - 👩‍💼 Mentoría para mujeres
  - 💻 Formación digital (50+)
  - 🚀 Prácticas profesionales (jóvenes)
  - ⏰ Consultoría de productividad
- Output: 10 ejemplos personalizados

#### Sección 4: A/B Testing Simulado
- Población: Grupo de alto riesgo
- Intervención: +2 años de educación
- Métricas: Lift, conversiones, diferencia absoluta
- Visualización: Distribución de probabilidades Control vs Tratamiento

---

## 📈 Métricas y Evaluación

### Por qué usamos estas métricas:

| Métrica | Justificación | Valor objetivo |
|---------|---------------|----------------|
| **AUC-ROC** | Capacidad de discriminar entre clases (≤50K vs >50K), robusto ante desbalance | > 0.75 |
| **F1-Score (weighted)** | Balance entre precisión y recall, ponderado por tamaño de clase | > 0.70 |
| **Accuracy** | Métrica general (pero insuficiente con desbalance) | > 0.80 |
| **TPR (True Positive Rate)** | Recall: cuántos casos reales de >50K detectamos | > 0.60 |
| **FPR (False Positive Rate)** | Errores: personas ≤50K clasificadas como >50K | < 0.20 |

### ⚠️ Por qué NO usamos solo Accuracy:

Con clases desbalanceadas (76% vs 24%), un modelo que predice "siempre ≤50K" tendría:
- Accuracy = 76% (¡parece bueno!)
- Pero Recall para >50K = 0% (¡inútil!)

**Solución**: Usar F1-score weighted que penaliza modelos sesgados.

---

## 🛠️ Troubleshooting

### ❌ Problema: `ModuleNotFoundError: No module named 'sklearn'`
**Causa**: scikit-learn no instalado  
**Solución**:
```bash
pip install scikit-learn
```

### ❌ Problema: `FileNotFoundError: adult.csv not found`
**Causa**: Ruta incorrecta al archivo  
**Solución**:
1. Verificar que `adult.csv` esté en `eda/data/`
2. Ajustar ruta en el notebook:
```python
data_path = Path("eda/data/adult.csv")  # Relativa
# O usar ruta absoluta
data_path = Path(r"C:\Users\usuario\Desktop\bcia\somosf5\eda\data\adult.csv")
```

### ❌ Problema: `KeyError: 'income_bin'`
**Causa**: Ejecutar `model_somos.ipynb` sin ejecutar `eda_somos.ipynb` primero  
**Solución**: 
1. Ejecutar primero `eda_somos.ipynb` completo (genera archivos procesados)
2. Luego ejecutar `model_somos.ipynb`

### ❌ Problema: `ValueError: could not convert string to float`
**Causa**: CSV sin limpiar o con valores faltantes  
**Solución**: Ejecutar celdas de limpieza en `eda_somos.ipynb`

### ❌ Problema: Warnings de pandas/numpy
**Causa**: Versiones de librerías  
**Solución**: Actualizar dependencias:
```bash
pip install --upgrade pandas numpy scikit-learn
```

### ❌ Problema: Kernel muerto en Jupyter
**Causa**: Falta de memoria RAM (dataset grande)  
**Solución**:
- Cerrar otros programas
- Reiniciar kernel: Kernel → Restart
- Reducir tamaño de muestra en visualizaciones

---

## 🔒 Ética y Sesgo

### Sesgos Identificados

1. **Brecha de género**: El modelo tiene TPR mayor para hombres que para mujeres
   - **Causa**: Datos históricos reflejan discriminación salarial real
   - **Detección**: Calculamos TPR por género y comparamos
   - **Mitigación propuesta**: Re-weighting de muestras, threshold optimization por grupo

2. **Discriminación etaria**: Personas >60 años tienen menor TPR
   - **Causa**: Prejuicio sobre "empleabilidad" en datos de entrenamiento
   - **Detección**: Análisis de fairness por age_group
   - **Mitigación propuesta**: Auditoría manual de casos mayores, re-balanceo

3. **Segregación ocupacional**: Sectores Manual/Service penalizados
   - **Causa**: Correlación histórica ocupación-ingresos
   - **Detección**: Comparar TPR entre occ_groups
   - **Mitigación propuesta**: Feature engineering, intervenciones prioritarias

### Limitaciones del Dataset

⚠️ **Advertencias importantes:**
- Datos de 1994: pueden no reflejar realidad actual (30+ años)
- Solo EE.UU.: no generalizable a otros países
- Censo oficial: puede sub-representar población vulnerable (indocumentados, sin hogar)
- Binarización de género: no incluye identidades no binarias

### Compromiso Ético

✅ **Este proyecto NO debe ser usado en producción sin:**
1. Auditoría ética externa
2. Implementación de mitigación de sesgos
3. Monitoreo continuo de métricas de fairness
4. Proceso de apelación para decisiones automatizadas
5. Transparencia con usuarios sobre cómo funciona el modelo

---

## 📚 Recursos y Referencias

### Documentación Técnica
- [scikit-learn](https://scikit-learn.org/stable/): Documentación de ML
- [pandas](https://pandas.pydata.org/docs/): Manipulación de datos
- [Fairness Indicators](https://www.tensorflow.org/responsible_ai/fairness_indicators/guide): Métricas de equidad

### Papers y Artículos
- [Fairness and Machine Learning](https://fairmlbook.org/) (libro gratuito)
- "Fairness through awareness" (Dwork et al., 2012)
- "Algorithmic Fairness" (Mehrabi et al., 2021)

### Dataset
- [UCI Adult Income](https://archive.ics.uci.edu/ml/datasets/adult)
- Donación original: Ronny Kohavi y Barry Becker (1996)

---

## 👥 Contribuciones

¿Quieres mejorar el proyecto? ¡Contribuciones bienvenidas!

1. Fork el repositorio
2. Crea una rama: `git checkout -b feature/mejora`
3. Commit: `git commit -m 'Agregar nueva métrica de fairness'`
4. Push: `git push origin feature/mejora`
5. Abre un Pull Request

---

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver archivo `LICENSE` para más detalles.

El dataset Adult Income está bajo licencia CC BY 4.0 (UCI ML Repository).

---

## ✉️ Contacto

**Repositorio**: [github.com/DarthVada36/somos-f5-social](https://github.com/DarthVada36/somos-f5-social)  
**Branch principal**: `dev`

---

## 🎯 Roadmap Futuro

- [ ] Implementar mitigación de sesgos (re-weighting, threshold opt)
- [ ] Demo interactiva con Streamlit
- [ ] Actualizar dataset con datos más recientes
- [ ] Expandir sistema de recomendaciones con ML
- [ ] API REST para predicciones en tiempo real
- [ ] Dashboard de monitoreo de fairness
- [ ] Tests unitarios y CI/CD

---

**✨ Proyecto desarrollado con enfoque en ML ético y reducción de desigualdad económica ✨**

*Última actualización: Octubre 2025*
