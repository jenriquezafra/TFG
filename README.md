# TFG: **Dinámica estocástica de poblaciones**

Durante la última década se ha reconocido que la organización de las comunidades biológicas no puede entenderse dejando al margen la aleatoriedad ambiental, aunque la mayoría de los modelos cuantitativos siguen anclados en extensiones deterministas del clásico Lotka-Volterra. Aquí abordamos la cuestión de qué manera las fluctuaciones estocásticas modifican la estabilidad, el caos y la biodiversidad en un sistema multiespecie con regulación por densidad y, al incorporar ruido blanco multiplicativo a un modelo mínimo capaz de mostrar caos determinista, demostramos que incluso intensidades moderadas desvían las trayectorias poblacionales hacia varios atractores robustos y que un incremento adicional desencadena transiciones intermitentes rematadas por extinciones escalonadas. Este comportamiento contradice las predicciones deterministas y revela que el ruido puede desempeñar un doble papel —desestabilizador cuando amplifica inestabilidades latentes y estabilizador cuando abre rutas inesperadas de coexistencia—, ofreciendo un vínculo operativo entre la magnitud del exponente de Lyapunov y la pérdida de biodiversidad. El marco propuesto legitima la combinación de enfoques deterministas y estocásticos para describir ecosistemas sometidos a presiones externas variables y resulta exportable a dominios tan dispares como la epidemiología, la dinámica de redes o los mercados energéticos, donde la aleatoriedad también cincela la resiliencia y el comportamiento colectivo.

## Estructura del repositorio

.
├── .vscode/
│   └── settings.json           Configuraciones de VSCode (formato, linting, etc.).
│
├── data/
│   ├── barrido_asim.npy        Resultados de simulaciones asíncronas.
│   ├── barrido_exts_random.npy Trayectorias con extinciones aleatorias.
│   ├── barrido_LCEs_random.npy Cálculo de exponentes de Lyapunov con métodos aleatorios.
│   └── barrido_sim.npy         Resultados de simulaciones síncronas.
│
├── notebooks/
│   ├── comparations.ipynb           Comparativa de modelos y métricas de error.
│   ├── deterministic_system.ipynb   Análisis del sistema determinista.
│   ├── parameters_exploration.ipynb Estudio de sensibilidad a parámetros.
│   ├── regulated_model.ipynb        Sistema con regulación de retroalimentación.
│   ├── stochastic_system.ipynb      Análisis del sistema estocástico.
│   └── tests.ipynb                  Validación y pruebas de consistencia.
│
├── outputs/
│   ├── barrido_LCEs_exts_random.png  Gráfica de LCE frente a extinciones.
│   ├── comp_trayectorias_error.png   Trayectorias con error absoluto.
│   ├── comp_trayectorias_rel.png     Trayectorias normalizadas.
│   ├── comp_trayectorias_ruido.png   Efecto del ruido en las trayectorias.
│   ├── comp_trayectorias.png         Comparativa general de trayectorias.
│   ├── doble_asim.png                Dinámica en esquemas asíncronos múltiples.
│   ├── espacio_fasico_4especies.png  Espacio de fases para cuatro especies.
│   ├── espacio_fasico_4especies_ruido1.png  Espacio de fases con perturbación.
│   ├── evol_temp_4especies.png       Evolución temporal de cuatro especies.
│   ├── evol_temp_4especies_ruido1.png Evolución temporal con ruido.
│   ├── LCE_4especies.png             Exponentes de Lyapunov en cuatro especies.
│   ├── lces_ruido_comunes.png        LCE frente a ruido común.
│   └── wolf_det.png                  Modelo de presa-depredador determinista.
│
├── src/
│   ├── __init__.py       Inicialización del paquete.
│   ├── det_system.py     Implementación del sistema determinista.
│   ├── stoc_system.py    Implementación del sistema estocástico.
│   ├── functions.py      Funciones auxiliares (métricas, utilidades).
│   └── README.md         Documentación del módulo fuente.
│
├── .gitignore            Definición de archivos y directorios ignorados.
├── LICENSE               Licencia de uso y distribución (MIT).
├── requirements.txt      Dependencias de Python.
└── start.sh              Script de ejecución: preprocesamiento y generación de resultados.


## Ejecución

## Módulos principales
