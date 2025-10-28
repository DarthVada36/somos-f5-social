from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from pathlib import Path
import os
import warnings
warnings.filterwarnings('ignore')

try:
    import markdown2
    MARKDOWN_AVAILABLE = True
except ImportError:
    print("⚠️  markdown2 no instalado. Para mejor formato: pip install markdown2")
    MARKDOWN_AVAILABLE = False

try:
    import joblib
except ImportError:
    print("❌ Error: Instalar joblib con: pip install joblib")
    exit(1)

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    print("⚠️  groq no instalado. Usando Mock LLM. Para instalar: pip install groq")
    GROQ_AVAILABLE = False

# Intentar cargar variables de entorno desde .env
try:
    from dotenv import load_dotenv
    load_dotenv()  # Carga variables desde .env
    print("✅ Variables de entorno cargadas desde .env")
except ImportError:
    print("⚠️  python-dotenv no instalado (opcional)")
    pass


# ============================================
# CONFIGURACIÓN DE FLASK
# ============================================

app = Flask(__name__)
app.config['SECRET_KEY'] = 'proyecto-inclusion-economica-2025'

# Rutas
PROJECT_DIR = Path(__file__).parent
MODEL_PATH = PROJECT_DIR / "eda" / "models" / "best_model_gradient_boosting.pkl"
METADATA_PATH = PROJECT_DIR / "eda" / "models" / "model_metadata_gradient_boosting.pkl"

# Configurar Groq (si API KEY está disponible)
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
USE_REAL_LLM = GROQ_AVAILABLE and GROQ_API_KEY is not None

if USE_REAL_LLM:
    groq_client = Groq(api_key=GROQ_API_KEY)
    print("✅ Groq API configurado correctamente")
else:
    groq_client = None
    print("⚠️  Usando Mock LLM (sin API real)")


# ============================================
# CARGAR MODELO AL INICIAR LA APP
# ============================================

print("🚀 Iniciando aplicación...")
print(f"📦 Cargando modelo desde: {MODEL_PATH}")

try:
    modelo = joblib.load(MODEL_PATH)
    print("✅ Modelo cargado correctamente")
    
    if METADATA_PATH.exists():
        metadata = joblib.load(METADATA_PATH)
        feature_names = metadata['feature_names']
        print(f"✅ Metadatos cargados: {metadata['model_type']}, AUC={metadata['auc_roc']:.4f}")
    else:
        feature_names = None
        print("⚠️ Metadatos no encontrados")
        
except Exception as e:
    print(f"❌ Error al cargar modelo: {e}")
    print("\n📝 Asegúrate de ejecutar model_somos.ipynb primero para generar el modelo")
    exit(1)


# ============================================
# FUNCIONES AUXILIARES
# ============================================

def formatear_respuesta_markdown(texto):
    """
    Convierte markdown a HTML y mejora el formato para chatbot
    """
    if not MARKDOWN_AVAILABLE:
        # Fallback sin markdown2: solo formato básico
        html = texto.replace('\n\n', '</p><p>')
        html = f'<p>{html}</p>'
        html = html.replace('**', '<strong>').replace('**', '</strong>')
        html = html.replace('###', '<h3>').replace('\n', '</h3>', 1)
        return html
    
    # Convertir markdown a HTML con markdown2
    html = markdown2.markdown(texto, extras=[
        'fenced-code-blocks',
        'tables',
        'break-on-newline',
        'cuddled-lists'
    ])
    
    return html


def preparar_datos(form_data):
    """
    Convierte datos del formulario en DataFrame para el modelo
    """
    # Valores del formulario
    edad = int(form_data.get('edad', 35))
    educacion = int(form_data.get('educacion', 12))
    horas = int(form_data.get('horas', 40))
    genero = form_data.get('genero', 'female')
    ocupacion = form_data.get('ocupacion', 'service')
    capital_gain = int(form_data.get('capital_gain', 0))
    
    # Crear perfil
    perfil = {
        'age': edad,
        'education_num': educacion,
        'hours_per_week': horas,
        'capital_gain': capital_gain,
        'capital_loss': 0,
        'sex_Male': 1 if genero == 'male' else 0,
        'occ_group_Management': 1 if ocupacion == 'management' else 0,
        'occ_group_Manual': 1 if ocupacion == 'manual' else 0,
        'occ_group_Military': 0,
        'occ_group_Professional': 1 if ocupacion == 'professional' else 0,
        'occ_group_Sales': 1 if ocupacion == 'sales' else 0,
        'occ_group_Service': 1 if ocupacion == 'service' else 0,
        'occ_group_Skilled-labor': 1 if ocupacion == 'skilled' else 0,
        'occ_group_Technical': 1 if ocupacion == 'technical' else 0,
    }
    
    X = pd.DataFrame([perfil])
    
    # Agregar columnas faltantes si existen metadatos
    if feature_names:
        for col in feature_names:
            if col not in X.columns and col != 'income_bin':
                X[col] = 0
        X = X[[c for c in feature_names if c in X.columns]]
    
    return X, perfil


def generar_prompt_llm(perfil, probabilidad, nivel_riesgo):
    """
    Genera el prompt para el LLM con variables parametrizadas
    """
    genero_texto = "Masculino" if perfil['sex_Male'] == 1 else "Femenino"
    ocupacion_texto = next((k.replace('occ_group_', '') for k, v in perfil.items() 
                           if k.startswith('occ_group_') and v == 1), 'Unknown')
    
    prompt = f"""
Eres un asistente de IA especializado en **inclusión económica y social**.

---

📊 **CONTEXTO DEL PROYECTO**

**Objetivo**: Reducir la brecha de ingresos y promover movilidad social
**Métrica principal**: F1-Score weighted (balance precision-recall)
**Riesgo de sesgo identificado**: Brecha de género detectada (mujeres tienen menor TPR)

---

👤 **PERFIL DE LA PERSONA**

- **Edad**: {perfil['age']} años
- **Género**: {genero_texto}
- **Educación**: {perfil['education_num']} años de escolaridad
- **Ocupación actual**: {ocupacion_texto}
- **Horas trabajadas/semana**: {perfil['hours_per_week']}
- **Probabilidad de bajos ingresos**: {probabilidad:.1f}%
- **Nivel de riesgo**: {nivel_riesgo}

---

🎯 **TU TAREA**

Genera recomendaciones concretas que incluyan:

1. **Diagnóstico de riesgo**: Por qué está en situación vulnerable
2. **3 Intervenciones prioritarias**: Programas/formación específicos
3. **Recursos disponibles**: Becas, organizaciones, plataformas
4. **Plan de seguimiento**: Medir progreso en 3-6 meses
"""
    return prompt.strip()


def obtener_respuesta_groq(prompt_llm, perfil, probabilidad, nivel_riesgo):
    """
    Obtiene respuesta REAL del LLM usando Groq API
    
    Args:
        prompt_llm: El prompt completo generado
        perfil: Dict con datos del perfil
        probabilidad: % de riesgo
        nivel_riesgo: ALTO/MEDIO/BAJO
    
    Returns:
        str: Respuesta del LLM o error
    """
    if not USE_REAL_LLM:
        return None
    
    try:
        # Llamada a Groq API
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": """Eres un experto en inclusión económica y políticas sociales. 
Tu tarea es analizar perfiles de riesgo de bajos ingresos y proponer intervenciones 
personalizadas basadas en evidencia. Sé específico, empático y práctico.

Estructura tu respuesta en:
1. DIAGNÓSTICO (factores de riesgo identificados)
2. INTERVENCIONES RECOMENDADAS (mínimo 3, con recursos reales)
3. PLAN DE SEGUIMIENTO (cronograma 6 meses con métricas)
4. ESTIMACIÓN DE IMPACTO (ROI, % aumento salarial esperado)
5. NOTA ÉTICA (sesgos detectados en el modelo ML)

Usa formato markdown con emojis. Incluye enlaces a recursos reales."""
                },
                {
                    "role": "user",
                    "content": prompt_llm
                }
            ],
            model="llama-3.3-70b-versatile",  # Modelo de Groq
            temperature=0.7,
            max_tokens=2000,
            top_p=0.9
        )
        
        respuesta = chat_completion.choices[0].message.content
        return respuesta
        
    except Exception as e:
        print(f"❌ Error llamando a Groq API: {e}")
        return None


def simular_respuesta_llm(perfil, probabilidad, nivel_riesgo):
    """
    Simula la respuesta de un LLM (GPT-4, Claude, etc.)
    FALLBACK cuando no hay API real disponible
    """
    edad = perfil['age']
    educacion = perfil['education_num']
    genero = "Femenino" if perfil['sex_Male'] == 0 else "Masculino"
    horas = perfil['hours_per_week']
    
    ocupacion = next((k.replace('occ_group_', '') for k, v in perfil.items() 
                      if k.startswith('occ_group_') and v == 1), 'Unknown')
    
    respuesta = f"""
### 📋 ANÁLISIS DE RIESGO - NIVEL {nivel_riesgo}

**Perfil**: {genero}, {edad} años, {educacion} años de educación, sector {ocupacion}

**Probabilidad de mantener bajos ingresos**: {probabilidad:.1f}%

---

### 🔍 DIAGNÓSTICO

"""
    
    # Identificar factores de riesgo
    factores = []
    
    if educacion < 12:
        factores.append(f"""
**Factor 1: Educación básica incompleta** ({educacion} años < 12)
- Limita acceso a trabajos cualificados
- Salarios promedio: -35% vs secundaria completa
""")
    
    if ocupacion in ['Manual', 'Service', 'Unknown']:
        factores.append(f"""
**Factor 2: Sector laboral de bajos ingresos** ({ocupacion})
- Trabajos con alta demanda física, baja remuneración
- Escasa movilidad salarial dentro del sector
""")
    
    if genero == "Femenino" and probabilidad > 50:
        factores.append("""
**Factor 3: Brecha de género**
- Discriminación salarial documentada: -15-20% vs hombres
- Menor acceso a promociones (⚠️ detectado en nuestro modelo)
""")
    
    if horas > 45:
        factores.append(f"""
**Factor 4: Sobretrabajo sin recompensa** ({horas}h/semana)
- Muchas horas NO garantizan buenos ingresos
- Indica trabajo de bajo valor agregado
""")
    
    if factores:
        respuesta += "\n".join(factores)
    else:
        respuesta += "Situación relativamente estable, pero con margen de mejora.\n"
    
    # Recomendaciones
    respuesta += """

---

### ✅ INTERVENCIONES RECOMENDADAS

"""
    
    if educacion < 12:
        respuesta += """
#### 1. 📚 PROGRAMA GED + CERTIFICACIÓN TÉCNICA

**Qué es**: Completar secundaria + formación técnica

**Duración**: 6-12 meses (clases nocturnas disponibles)

**Sectores recomendados**:
- Healthcare Assistant → Salario: $35K-45K
- IT Support → Salario: $40K-55K
- HVAC Technician → Salario: $45K-60K

**Recursos**:
- [GED.com](https://ged.com) - Plataforma oficial
- Community Colleges - Becas Pell Grant

**ROI**: +40-60% incremento salarial en 2 años

"""
    
    if ocupacion in ['Manual', 'Service', 'Unknown']:
        respuesta += """
#### 2. 🔄 RECONVERSIÓN PROFESIONAL

**Programa**: Workforce Innovation & Opportunity Act (WIOA)

**Beneficios**:
- Formación 100% financiada
- Prácticas pagadas ($12-15/hora)
- 85% tasa de éxito en colocación

**Rutas de transición**:
- Service → Healthcare (6 meses)
- Manual → Skilled Trade (1 año)
- Unknown → IT Support (3-4 meses)

"""
    
    if genero == "Femenino":
        respuesta += """
#### 3. 👩‍💼 MENTORÍA + NEGOCIACIÓN SALARIAL

**Qué es**: Empoderamiento económico para mujeres

**Componentes**:
- Mentoría 1-a-1 (3 meses)
- Workshop de negociación salarial
- Networking en sectores masculinizados

**Organizaciones**:
- [Dress for Success](https://dressforsuccess.org)
- [AAUW Work Smart](https://salary.aauw.org) - Curso gratis

**Impacto**: +$500K en vida laboral por negociar salario inicial

"""
    
    respuesta += """

---

### 📅 PLAN DE SEGUIMIENTO

**Mes 1-2**: 
- ✅ Inscripción en programa de formación
- 📊 Métrica: Asistencia >80%

**Mes 3-4**:
- ✅ Completar certificación
- 📊 Métrica: 2+ entrevistas laborales

**Mes 5-6**:
- ✅ Nueva oferta laboral
- 📊 Métrica: +30% incremento salarial

---

### 🎯 IMPACTO ESPERADO

Con **80% de adherencia** al plan:

- 📈 **Probabilidad de >50K**: Mejora a 45-55% en 12 meses
- 💰 **Incremento salarial**: +35-50%
- 🚀 **ROI a 5 años**: +$50,000-80,000

---

### ⚠️ NOTA ÉTICA

Nuestro modelo detectó **sesgo de género sistemático**. Las mujeres son penalizadas 
estadísticamente incluso con cualificaciones equivalentes.

**Esto NO es tu culpa. Es una falla estructural del mercado laboral.**

---

💪 **Próximo paso**: Visita [CareerOneStop.org](https://careeronestop.org) para encontrar 
recursos en tu área local.
"""
    
    return respuesta.strip()


# ============================================
# RUTAS DE LA APLICACIÓN
# ============================================

@app.route('/')
def index():
    """Página principal con formulario"""
    return render_template('index.html')


@app.route('/predecir', methods=['POST'])
def predecir():
    """Endpoint para hacer predicción"""
    try:
        # Obtener datos del formulario
        form_data = request.form
        
        # Preparar datos para el modelo
        X, perfil = preparar_datos(form_data)
        
        # Predecir
        prediccion = modelo.predict(X)[0]
        probabilidades = modelo.predict_proba(X)[0]
        
        prob_bajo = probabilidades[0] * 100
        prob_alto = probabilidades[1] * 100
        
        # Clasificar riesgo
        if prob_bajo > 70:
            nivel_riesgo = "ALTO"
            color_riesgo = "danger"
        elif prob_bajo > 30:
            nivel_riesgo = "MEDIO"
            color_riesgo = "warning"
        else:
            nivel_riesgo = "BAJO"
            color_riesgo = "success"
        
        # Generar prompt para LLM
        prompt_llm = generar_prompt_llm(perfil, prob_bajo, nivel_riesgo)
        
        # Intentar usar Groq API (si está configurado), sino usar mock
        if USE_REAL_LLM:
            print("🤖 Llamando a Groq API...")
            respuesta_llm_raw = obtener_respuesta_groq(prompt_llm, perfil, prob_bajo, nivel_riesgo)
            
            # Si falla Groq, usar mock como fallback
            if respuesta_llm_raw is None:
                print("⚠️ Groq falló, usando Mock LLM como fallback")
                respuesta_llm_raw = simular_respuesta_llm(perfil, prob_bajo, nivel_riesgo)
            else:
                print("✅ Respuesta obtenida de Groq")
        else:
            print("🎭 Usando Mock LLM (sin API)")
            respuesta_llm_raw = simular_respuesta_llm(perfil, prob_bajo, nivel_riesgo)
        
        # Formatear respuesta de markdown a HTML
        respuesta_llm_html = formatear_respuesta_markdown(respuesta_llm_raw)
        
        # Preparar resultados
        resultado = {
            'prediccion': '≤50K (Bajos ingresos)' if prediccion == 0 else '>50K (Ingresos altos)',
            'prob_bajo': round(prob_bajo, 1),
            'prob_alto': round(prob_alto, 1),
            'nivel_riesgo': nivel_riesgo,
            'color_riesgo': color_riesgo,
            'prompt_llm': prompt_llm,
            'respuesta_llm': respuesta_llm_html,
            'usando_api_real': USE_REAL_LLM,
            'perfil': {
                'edad': perfil['age'],
                'educacion': perfil['education_num'],
                'horas': perfil['hours_per_week'],
                'genero': 'Masculino' if perfil['sex_Male'] == 1 else 'Femenino',
                'ocupacion': next((k.replace('occ_group_', '') for k, v in perfil.items() 
                                  if k.startswith('occ_group_') and v == 1), 'Unknown')
            }
        }
        
        return render_template('resultado.html', resultado=resultado)
        
    except Exception as e:
        return render_template('error.html', error=str(e))


@app.route('/api/predecir', methods=['POST'])
def api_predecir():
    """API REST para predicciones (JSON)"""
    try:
        data = request.get_json()
        
        # Preparar datos
        X, perfil = preparar_datos(data)
        
        # Predecir
        prediccion = modelo.predict(X)[0]
        probabilidades = modelo.predict_proba(X)[0]
        
        prob_bajo = float(probabilidades[0] * 100)
        
        # Clasificar riesgo
        if prob_bajo > 70:
            nivel_riesgo = "ALTO"
        elif prob_bajo > 30:
            nivel_riesgo = "MEDIO"
        else:
            nivel_riesgo = "BAJO"
        
        return jsonify({
            'success': True,
            'prediccion': int(prediccion),
            'probabilidad_bajos_ingresos': round(prob_bajo, 2),
            'nivel_riesgo': nivel_riesgo
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/sobre')
def sobre():
    """Página informativa sobre el proyecto"""
    return render_template('sobre.html')


# ============================================
# EJECUTAR APLICACIÓN
# ============================================

if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("🌐 APLICACIÓN WEB: PREDICTOR DE RIESGO + MOCK LLM")
    print("=" * 80)
    print("\n✅ Servidor iniciado correctamente")
    print("\n🔗 Abrir en navegador: http://localhost:5000")
    print("\n📝 Endpoints disponibles:")
    print("   - /           → Formulario principal")
    print("   - /predecir   → Resultado de predicción")
    print("   - /api/predecir → API REST (JSON)")
    print("   - /sobre      → Información del proyecto")
    print("\n⏹️  Presiona Ctrl+C para detener el servidor")
    print("=" * 80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
