# TFG: **Dinámica estocástica de poblaciones**

Durante la última década se ha reconocido que la organización de las comunidades biológicas no puede entenderse dejando al margen la aleatoriedad ambiental, aunque la mayoría de los modelos cuantitativos siguen anclados en extensiones deterministas del clásico Lotka-Volterra. Aquí abordamos la cuestión de qué manera las fluctuaciones estocásticas modifican la estabilidad, el caos y la biodiversidad en un sistema multiespecie con regulación por densidad y, al incorporar ruido blanco multiplicativo a un modelo mínimo capaz de mostrar caos determinista, demostramos que incluso intensidades moderadas desvían las trayectorias poblacionales hacia varios atractores robustos y que un incremento adicional desencadena transiciones intermitentes rematadas por extinciones escalonadas. Este comportamiento contradice las predicciones deterministas y revela que el ruido puede desempeñar un doble papel —desestabilizador cuando amplifica inestabilidades latentes y estabilizador cuando abre rutas inesperadas de coexistencia—, ofreciendo un vínculo operativo entre la magnitud del exponente de Lyapunov y la pérdida de biodiversidad. El marco propuesto legitima la combinación de enfoques deterministas y estocásticos para describir ecosistemas sometidos a presiones externas variables y resulta exportable a dominios tan dispares como la epidemiología, la dinámica de redes o los mercados energéticos, donde la aleatoriedad también cincela la resiliencia y el comportamiento colectivo.

## Estructura del repositorio

```bash
.
├── .vscode/
│   └── settings.json           Configuraciones de VSCode (formato, linting, etc.).
│
├── data/
│   ├── barrido_asim.npy            Resultados del barrido asimétrico (deprecated)
│   ├── barrido_exts_random.npy     Resultados del barrido de extinciones con x0 aleatorios
│   ├── barrido_LCEs_random.npy     Resultados del barrido de LCEs con x0 aleatorios
│   └── barrido_sim.npy             Resultados del barrido simétrico (deprecated)
│
├── notebooks/
│   ├── comparations.ipynb               Comparativa de modelos y métricas de error.
│   ├── deterministic_system.ipynb       Análisis del sistema determinista.
│   ├── parameters_exploration.ipynb     Estudio de sensibilidad a parámetros.
│   ├── stochastic_system.ipynb          Análisis del sistema estocástico.
│   └── tests.ipynb                  
│
├── outputs/
│   ├── barrido_LCEs_exts_random.png  
│   ├── comp_trayectorias_error.png   
│   ├── comp_trayectorias_rel.png     
│   ├── comp_trayectorias_ruido.png   
│   ├── comp_trayectorias.png         
│   ├── doble_asim.png                
│   ├── espacio_fasico_4especies.png  
│   ├── espacio_fasico_4especies_ruido1.png  
│   ├── evol_temp_4especies.png       
│   ├── evol_temp_4especies_ruido1.png 
│   ├── LCE_4especies.png            
│   ├── lces_ruido_comunes.png       
│   └── wolf_det.png                  
│
├── src/
│   ├── __init__.py       
│   ├── det_system.py     Implementación del sistema determinista
│   ├── stoc_system.py    Implementación del sistema estocástico
│   ├── functions.py      Funciones auxiliares 
│   └── README.md         
│
├── .gitignore            
├── LICENSE               
├── requirements.txt      
└── start.sh              

```


## Ejecución


## Módulos principales
