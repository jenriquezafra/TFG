import numpy as np
from scipy.integrate import solve_ivp
from src.stoc_system import StochasticSystem

def ccr(x, y):
    """
    Función que calcula el coeficiente de correlación cruzada entre dos variables.
    :param x: array de datos de la variable 1
    :param y: array de datos de la variable 2
    :return: coeficiente de correlación cruzada entre las dos variables
    """
    
    x = np.array(x)
    y = np.array(y)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    num = np.sum((x - x_mean) * (y - y_mean))
    den = np.sqrt(np.sum((x - x_mean)**2)) * np.sqrt(np.sum((y - y_mean)**2))
    return num / den

def entropy(x):
    """
    Función que calcula la entropía de Shannon de una variable.
    :param x: array de datos de la variable
    :return: entropía de Shannon de la variable
    """

    # usamos la regla de sturges para el numero de bins optimo
    bins = int(np.log(len(x)) + 1)
    # print('bins = ',bins)
    
    # hallamos el histograma
    counts, _ = np.histogram(x, bins=bins)
    
    # eliminamos bins vacíos para evitar log(0)
    counts = counts[counts>0]
    
    # normalizamos para que sea la distribución de probabilidad
    p = counts/np.sum(counts)
    
    # hallamos la entropía de shannon
    H = -np.sum(p*np.log(p))
    # plt.plot(x, p) #comentar el filtro para que funcione
    return H

def joint_entropy(x1, x2, x3, x4):
    """Función que calcula la entropía conjunta de cuatro variables.
    :param x1: array de datos de la variable 1
    :param x2: array de datos de la variable 2
    :param x3: array de datos de la variable 3
    :param x4: array de datos de la variable 4
    :return: entropía conjunta de las cuatro variables"""

    bins = int(np.log(len(x1)) + 1)
    
    # creamos un solo array
    data = np.hstack((x1, x2, x3, x4))
    
    # hallamos el histograma multidimensional
    hist, _ = np.histogramdd(data, bins=bins)
    
    # aplanar y eliminar bins con 0s
    hist = hist.flatten() # para que sea un solo vector
    hist_nonzero = hist[hist>0]
    
    # hallamos la distribución de probabilidad
    p = hist_nonzero/np.sum(hist_nonzero)
    
    # hallamos entropía conjunta
    H = -np.sum(p*np.log(p))
    return H


########################## Barridos de LCEs ##########################

def barrido_LCEs(total_time=5000, N_points=1e5, max_iters=50, save=False, save_path=None, save_name=None):
    """Función que realiza un barrido de diferentes intensidades de ruido del sistema y devuelve un array de NumPy con los resultados.
    :param total_time: Tiempo total de simulación
    :param N_points: Número de puntos para el método de Euler–Maruyama (se convertirá a entero)
    :param max_iters: Número máximo de iteraciones para el barrido (se convertirá a entero)
    :param save: Si True, guarda los resultados en un archivo CSV
    :param save_path: Ruta donde se guardará el archivo CSV
    :param save_name: Nombre del archivo CSV (sin extensión)
    :return: NumPy array de forma (max_iters, 3) con columnas [sigma, mean, std]
    """
    import os

    # aseguramos tipos enteros donde corresponda
    N_points = int(N_points)
    max_iters = int(max_iters)
    np.random.seed(2)

    # parámetros del sistema
    r = np.array([1, 0.72, 1.53, 1.27])
    a = np.array([
        [1, 1.09, 1.52, 0],
        [0, 1, 0.44, 1.36],
        [2.33, 0, 1, 0.47],
        [1.21, 0.51, 0.35, 1]
    ])
    x0 = np.array([0.5, 0.5, 0.5, 0.5])

    # valores de sigma a barrer
    array_sigmas = np.linspace(start=0.0, stop=1.5, num=max_iters)
    start_indices = [int(N_points * f) for f in (0.25, 0.5, 0.75)]

    # creamos un array para guardar los resultados
    resultados = np.zeros((max_iters, 3), dtype=float)

    for idx, sigma in enumerate(array_sigmas):
        print(f"{idx+1}/{max_iters}  sigma={sigma:.3f}...", end='\r')

        # creamos el sistema estocástico con un ruido asimétrico
        vect_ruido = sigma * np.array([0.8, 0.9, 1.0, 1.1])
        model = StochasticSystem(
            r, a, x0,
            total_time=total_time,
            dt=total_time / N_points,
            sigma=vect_ruido
        )

        # lo resolvemos 
        _, X = model.euler_maruyama()

        # hallamos su LCE
        _, vect_lambdas = model.estimate_LCE1_Wolf(
            X, evol_time=500,
            min_separation=180,
            max_replacements=500,
            start_indices=start_indices
        )

        # hallamos su media y std
        mean_val = np.nanmean(vect_lambdas)
        std_val = np.nanstd(vect_lambdas, ddof=1)

        resultados[idx, 0] = sigma
        resultados[idx, 1] = mean_val
        resultados[idx, 2] = std_val

    # guardado opcional en CSV
    if save and save_path and save_name:
        os.makedirs(save_path, exist_ok=True)
        filepath = os.path.join(save_path, f"{save_name}.csv")
        header = 'sigma,mean,std'
        np.savetxt(filepath, resultados, delimiter=',', header=header, comments='')

    return resultados


def barrido_LCEs_random(total_time=5000, N_points=1e5, max_iters=50):
    """Función que realiza un barrido de diferentes intensidades de ruido del sistema y condiciones iniciales aleatorias y devuelve un array de NumPy con los resultados.
    :param total_time: Tiempo total de simulación
    :param N_points: Número de puntos para el método de Euler–Maruyama (se convertirá a entero)
    :param max_iters: Número máximo de iteraciones para el barrido (se convertirá a entero)
    :return: NumPy array de forma (max_iters, 3) con columnas [sigma, mean, std]
    """

    # aseguramos tipos enteros donde corresponda
    N_points = int(N_points)
    max_iters = int(max_iters)
    np.random.seed(2)

    # parámetros del sistema
    r = np.array([1, 0.72, 1.53, 1.27])
    a = np.array([
        [1, 1.09, 1.52, 0],
        [0, 1, 0.44, 1.36],
        [2.33, 0, 1, 0.47],
        [1.21, 0.51, 0.35, 1]
    ])

    # valores de sigma a barrer
    array_sigmas = np.linspace(start=0.0, stop=1.5, num=max_iters)
    start_indices = [int(N_points * f) for f in (0.25, 0.5, 0.75)]

    # creamos un array para guardar los resultados
    resultados = np.zeros((max_iters, 3), dtype=float)

    for idx, sigma in enumerate(array_sigmas):
        print(f"{idx+1}/{max_iters}  sigma={sigma:.3f}...", end='\r')

        # generamos x0 aleatorio
        rng = np.random.default_rng(idx) # para que sea reproducible
        x0 = np.array([rng.random() for _ in range(4)])

        # creamos el sistema estocástico con un ruido asimétrico
        vect_ruido = sigma * np.array([0.8, 0.9, 1.0, 1.1])
        model = StochasticSystem(
            r, a, x0,
            total_time=total_time,
            dt=total_time / N_points,
            sigma=vect_ruido
        )

        # lo resolvemos 
        _, X = model.euler_maruyama()

        # hallamos su LCE
        _, vect_lambdas = model.estimate_LCE1_Wolf(
            X, evol_time=500,
            min_separation=180,
            max_replacements=500,
            start_indices=start_indices
        )

        # hallamos su media y std
        mean_val = np.nanmean(vect_lambdas)
        std_val = np.nanstd(vect_lambdas, ddof=1)

        resultados[idx, 0] = sigma
        resultados[idx, 1] = mean_val
        resultados[idx, 2] = std_val

    return resultados


def barrido_extinciones_random(total_time=5000, N_points=1e5, max_iters=50):
    """Función que realiza un barrido de diferentes intensidades de ruido del sistema y condiciones iniciales aleatorias y devuelve un array de NumPy con los resultados.
    :param total_time: Tiempo total de simulación
    :param N_points: Número de puntos para el método de Euler–Maruyama (se convertirá a entero)
    :param max_iters: Número máximo de iteraciones para el barrido (se convertirá a entero)
    :return: NumPy array de forma (max_iters, 2) con columnas [sigma, extinciones]
    """

    # aseguramos tipos enteros donde corresponda
    N_points = int(N_points)
    max_iters = int(max_iters)
    np.random.seed(2)

    # parámetros del sistema
    r = np.array([1, 0.72, 1.53, 1.27])
    a = np.array([
        [1, 1.09, 1.52, 0],
        [0, 1, 0.44, 1.36],
        [2.33, 0, 1, 0.47],
        [1.21, 0.51, 0.35, 1]
    ])

    # valores de sigma a barrer
    array_sigmas = np.linspace(start=0.0, stop=1.5, num=max_iters)

    # creamos un array para guardar los resultados
    resultados = np.zeros((max_iters, 3), dtype=float)

    for idx, sigma in enumerate(array_sigmas):
        print(f"{idx+1}/{max_iters}  sigma={sigma:.3f}...", end='\r')

        # generamos x0 aleatorio
        rng = np.random.default_rng(idx) # para que sea reproducible
        x0 = np.array([rng.random() for _ in range(4)])

        # creamos el sistema estocástico con un ruido asimétrico
        vect_ruido = sigma * np.array([0.8, 0.9, 1.0, 1.1])
        system = StochasticSystem(
            r, a, x0,
            total_time=total_time,
            dt=total_time / N_points,
            sigma=vect_ruido
        )

        # hallamos sus extinciones
        n_ext = system.extintions()
        
        resultados[idx, 0] = sigma
        resultados[idx, 1] = n_ext

    return resultados


def barrido_tiempos_extinciones_random(total_time = 5000, N_points = 1e5, max_iters = 50):
    """
    Función para hacer un barrido de la intensidad de ruido y el tiempo medio de extinción para cada uno.
    :param total_time: Tiempo total de la simulación.
    :param N_points: Número de puntos a simular.
    :param max_iters: Número máximo de iteraciones para la simulación.
    :return: Un array de numpy con los resultados del barrido.
    """

    # aseguramos tipos enteros donde corresponda
    N_points = int(N_points)
    max_iters = int(max_iters)
    np.random.seed(2)

    # parámetros del sistema
    r = np.array([1, 0.72, 1.53, 1.27])
    a = np.array([
        [1, 1.09, 1.52, 0],
        [0, 1, 0.44, 1.36],
        [2.33, 0, 1, 0.47],
        [1.21, 0.51, 0.35, 1]
    ])

    # valores de sigma a barrer
    array_sigmas = np.linspace(start=0.0, stop=1.5, num=max_iters)

    # creamos un array para guardar los resultados
    resultados = np.zeros((max_iters, 3), dtype=float)

    for idx, sigma in enumerate(array_sigmas):
        print(f"{idx+1}/{max_iters}  sigma={sigma:.3f}...", end='\r')

        # generamos x0 aleatorio
        rng = np.random.default_rng(idx) # para que sea reproducible
        x0 = np.array([rng.random() for _ in range(4)])

        # creamos el sistema estocástico con un ruido asimétrico
        vect_ruido = sigma * np.array([0.8, 0.9, 1.0, 1.1])
        system = StochasticSystem(
            r, a, x0,
            total_time=total_time,
            dt=total_time / N_points,
            sigma=vect_ruido
        )

        # hallamos los tiempos de extinción
        t_ext, std_ext = system.extintion_time()
        
        resultados[idx, 0] = sigma
        resultados[idx, 1] = t_ext
        resultados[idx, 2] = std_ext

    return resultados