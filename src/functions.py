import numpy as np
from scipy.integrate import solve_ivp


########################## Parte determinista ###########################
def system(t,x,r,a):
    '''
    :param x: vector de longitud N
    :param r: vector de tasa de crecimiento (r_i)
    :param a: matriz (N,N) con los coeficientes a_ij
    :return: numpy array
    '''
    N = len(x)
    dxdt = np.zeros(N)
    for i in range(N):  
        interaction = np.sum(a[i,:]*x)
        dxdt[i] = r[i] * x[i] * (1 - interaction)
    return dxdt

def jacobiano(x, r, a):
    """
    Calcula el jacobiano J(x).
    """
    N = len(r)
    J = np.zeros((N, N))
    Ax = a.dot(x)
    for i in range(N):
        sum_aijxj = Ax[i]
        for k in range(N):
            if i == k:
                J[i, k] = r[i] * ((1 - sum_aijxj) - x[i]*a[i, i])
            else:
                J[i, k] = -r[i]*x[i]*a[i, k]
    return J

def sistema_dinamico(x, r, a, dt):
    N = len(x)
    dxdt = np.zeros(N)
    for i in range(N):
        interaction = np.sum(a[i,:] * x)
        dxdt[i] = r[i] * x[i] * (1 - interaction)
    return x + dxdt * dt


def compute_LCEs(x0, r, a, max_iter):
    # creamos el paso de tiempo
    dt = 500/max_iter  # usamos un tiempo total de 500

    # inicializamos las matrices necesarias
    Q= np. eye(len(x0))
    sum_logs_R = np.zeros(len(x0))
    
    # guardamos para cada iteracion
    LCEs = np.zeros((max_iter, len(x0))) 
    all_logs = LCEs.copy()
    
    for i in range(max_iter):
        # hacemos la descomposicion QR
        J = jacobiano(x0, r, a)
        J_star = J @ Q
        Q, R = np.linalg.qr(J_star)
        
        # nos quedamos con los elementos diagonales
        sum_logs_R += np.log(np.abs(np.diag(R))) 
        all_logs[i] = sum_logs_R
        LCEs[i] = sum_logs_R / (1+i)
        
        # actualizamos el sistema
        sol = solve_ivp(system, [0, dt], x0, args=(r, a), method='RK45')
        x0 = sol.y[:,-1]  
        # x0 = sistema_dinamico(x0, r, a, dt)
        

    return LCEs, all_logs

def ccr(x, y):
    x = np.array(x)
    y = np.array(y)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    num = np.sum((x - x_mean) * (y - y_mean))
    den = np.sqrt(np.sum((x - x_mean)**2)) * np.sqrt(np.sum((y - y_mean)**2))
    return num / den

def entropy(x):
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

########################### Parte estocástica ###########################

def euler_maruyama(model, x0, t_span, dt, r, a, sigma):
    """
    Euler maruyama method
    :param system:  función definida antes
    :param x0: condición inicial (array de len N)
    :param t0: tiempo inicial (float)
    :param t_max: tiempo final (float)
    :param dt: step del tiempo (float)
    :param r: array con r_i
    :param a: matriz con a_ij
    :param sigma: array con sigma_i
    :return: (t_vals, x_vals), con x_vals un array (num_steps+1, N)
    """
    
    np.random.seed(2) 
    N=len(x0) # en nuestro caso será 4
    t0 = t_span[0]
    t_max = t_span[-1]
    num_steps = int((t_max-t0)/dt)
    
    # guardamos los valores 
    t_vals = np.linspace(t0, t_max, num_steps+1)
    x_vals = np.zeros((num_steps+1,N))
    
    x_vals[0, :] = x0
    x = x0.copy()
    
    for n in range(num_steps): # para cada paso temporal
        t = t_vals[n]
        
        # parte determinista
        F = model(t,x,r,a) # también array de 4

        # vector del ruido
        for i in range(N): # para cada especie
            # introducimos la variable aletoria
            zetta = np.random.normal(0, 1)
            
            # hallamos el siguiente valor
            x[i] = x[i]+ F[i]*dt + sigma[i]*x[i]*np.sqrt(dt)*zetta
            
            # si baja del umbral, suponemos que se extingue
            if x[i] <= 1e-9:
                x[i] = 0
                
            x_vals[n+1,i] = x[i]    
    return t_vals, x_vals
    
def nearest_neighbour_4D(data_4d, idx, min_separation = 50):
    '''
    Busca el punto más cercano para el punto data_4d[idx],
    excluyendo los puntos que están muy próximos en el tiempo.
    
    :param data_4d: ndarray, shape (N,4)
    :param idx: int
        Índice del punto de referencia
    :param min_separation: int
        Número mínimo de índices (pasos de tiempo) para evitar vecinos de la misma órbita
    :return: nn_idx: int
        Índice del vecino más cercano
    :return dist_min: float
        Distancia mínima encontrada
    '''
    
    ref_point = data_4d[idx]
    
    # hallamos todas las distancias 
    dists = np.linalg.norm(data_4d - ref_point, axis = 1)
    # excluimos el propio punto
    dists[idx] = np.inf
    
    # excluimos puntos en un rango temporal muy cercano
    start_excl = max(0, idx - min_separation)
    end_excl = min(len(data_4d), idx + min_separation)
    dists[start_excl:end_excl] = np.inf
    
    nn_idx = np.argmin(dists)
    dist_min = dists[nn_idx]
    return nn_idx, dist_min

def estimate_LCE1_Wolf(data_4d, evol_time, min_separation, dt, max_replacements, start_indices):
    '''
    
    Estimamos el mayor exponentes de Lyapunov usando el método 
    de Wolf  (en fixed time) para datos en R^4
    
    :param data_4d: ndarray, shape (N,4)
        Cada fila es un vector (x1, x2, x3, x4) en el instante de muestreo
    :param evol_time: int
        Número de pasos que se deja evolucionar cada par antes de renormalizar
    :param min_separation: int
        Número mínimo de pasos para excluir puntos muy próximos temporalmente
    :param df: float 
        Intervalo de muestreo
    :param max_replacements: int
        Número máximo de renormalizaciones a realizar
    :param start_indices: list
        Lista de índices iniciales a usar como puntos de referencia para promediar la estimación
        
    :return: lambda1: float
        Estimación del mayor LCE 
        
    :return: lambda_estimates: array
        Vector con la estimación de lamba1 en cada punto inicial
    '''
    
    N = len(data_4d)
        
    # almacenamos la estimación de cada punto
    lambda_estimates = [] 
    
    ## guardamos en una matriz todos los valores
    ## son nan para hacer la media mejor a que sean 0s
    all_vals = np.full(
        (len(start_indices), max_replacements), np.nan)
    
    for i, init_idx in enumerate(start_indices):
        idx_f = init_idx
        log_sum = 0.0
        count = 0
        
        # iteramos mientras se pueda evolucionar el punto de referencia y sin superar el máximo de iteraciones
        while (idx_f + evol_time < N) and (count < max_replacements):
            # buscamos vecino más cercano para el de referencia más cercano
            idx_n, dist_init = nearest_neighbour_4D(data_4d, idx_f, min_separation)
            
            # vemos que la distancia inicial es valida
            if np.isinf(dist_init) or dist_init < 1e-12:
                break
                
            # definimos los índices después de la evolución
            idx_f_next = idx_f + evol_time
            idx_n_next = idx_n + evol_time
            
            if idx_f_next >= N or idx_n_next >= N:
                break
            
            # calculamos la separación final
            dist_final = np.linalg.norm(data_4d[idx_f_next] - data_4d[idx_n_next])
            
            # acumulamos el logaritmo del crecimiento
            log_sum += np.log(dist_final / dist_init)
            count += 1
            
            ## valor parcial acumulado
            all_vals[i, count-1] = log_sum/(count*evol_time*dt)
            
            # renormalizamos
            idx_f = idx_f_next
        
        if count > 0:
            lambda_local = (log_sum / (count * evol_time * dt))
            
        else:
            lambda_local =np.nan
        
        # guardamos el valor a la lista       
        lambda_estimates.append(lambda_local)

    if len(lambda_estimates) > 0:
        # hacemos la media (quitando los nans)
        lambda1 = np.nanmean(lambda_estimates)

    else:
        lambda1 = 0
    
    return lambda1, lambda_estimates, all_vals