import numpy as np
from scipy.integrate import solve_ivp

class StochasticSystem:
    """
    Clase para definir nuestro sistema con ruido y realizar cálculos sobre él.
    """
    def __init__(self, r, a, x0, total_time, dt, sigma):
        """
        :param r: vector de tasas de crecimiento (r_i)
        :param a: matriz (N,N) con los coeficientes a_ij
        :param total_time: tiempo total de la simulación
        :param dt: paso temporal
        :param sigma: vector de intensidades de ruido
        """
        self.r = np.array(r, dtype = float)
        self.a = np.array(a, dtype = float)
        self.x0 = np.array(x0, dtype = float)
        self.N = self.r.shape[0]
        self.total_time = total_time
        self.dt = dt 
        self.sigma = np.array(sigma, dtype = float)


    def system(self, t, x):
        """
        Ecuaciones diferenciales del sistema
        :param t: tiempo
        :param x: vector de longitud N
        :return: dx/dt (numpy array)
        """
        x = np.array(x, dtype = float)
        dxdt = np.zeros(self.N)
        for i in range(self.N):  
            interaction = np.sum(self.a[i,:]*x)
            dxdt[i] = self.r[i] * x[i] * (1 - interaction)
        return dxdt


    def euler_maruyama(self):
        """
        Método de Euler-Maruyama para resolver la ecuación de Languevin de nuestro sistema.
        :param dt: paso de tiempo
        :param sigma: vector de intensidad de ruido
        """
        np.random.seed(2)  # para reproducibilidad
        n_steps = int(self.total_time/self.dt)

        # guardamos los valores
        t_vals = np.linspace(0, self.total_time, n_steps + 1)
        x_vals = np.zeros((n_steps + 1, self.N))

        x_vals[0] = self.x0
        x = self.x0.copy()

        for n in range(n_steps): # iteraciones para cada tiempo
            t = t_vals[n]

            # sistema original
            F = self.system(t, x) 

            # vector del ruido
            for i in range(self.N):  # para cada especie
                # creamos una variable aleatoria
                zetta = np.random.normal(0, 1)

                # hallamos el siguiente valor
                x[i] = x[i] + F[i]*self.dt + self.sigma[i]*x[i]*np.sqrt(self.dt)*zetta

                # definimos un umbral de extinción
                if x[i] <= 1e-9:
                    x[i] = 0.0

                x_vals[n+1, i] = x[i]

        return t_vals, x_vals


    def nearest_neighbour(self, X_vals, index, min_sep):
        """
        Busca el punto más cercano para los puntos dados en ...[index],
        excluyendo los punto que están muy próximos en el tiempo.
        Solo se llama a esta función dentro del método de Wolf.

        :param data: valores X del sistema resuelto (array)
        :param index: índice del punto de referencia
        :param min_sep: distancia mínima para considerar un punto cercano
        :return nn_index: índice del punto más cercano (int)
        :return dist_min: distancia mínima (float)
        """
        data = X_vals
        ref_points = data[index] # QUIZAS HAY QUE TRANSPONER LA MATRIZ

        # hallamos todas las distancias
        dists = np.linalg.norm(X_vals - ref_points, axis = 1)
        dists[index] = np.inf  # excluimos el propio punto de referencia

        # excluimos puntos en un rango temporal muy cercano
        start_excl = max(0, index-min_sep)
        end_excl = min(len(X_vals), index + min_sep)
        dists[start_excl:end_excl] = np.inf

        nn_index = np.argmin(dists)
        dist_min = dists[nn_index]
        return nn_index, dist_min


    def estimate_LCE1_Wolf(self, evol_time, min_separation, max_replacements, start_indices):
        """
        Estima el mayor exponente de Lyapunov (LCE) por el método de Wolf en tiempo fijo.
        Internamente genera la trayectoria mediante Euler-Maruyama.

        :param evol_time: pasos de evolución antes de renormalizar
        :param min_separation: exclusión temporal mínima de vecinos
        :param max_replacements: número máximo de renormalizaciones
        :param start_indices: lista de índices iniciales en la trayectoria para promediar
        :return: lambda1 (float), lambda_estimates (list), all_vals (ndarray)
        """
        # resolvemos la dinámica estocástica con Euler-Maruyama
        _, data = self.euler_maruyama()
        N = len(data)

        # almacenamos la estimación de cada punto
        lambda_estimates = []   

        # iteramos sobre cada punto de referencia inicial
        for i, init_idx in enumerate(start_indices):
            idx_f = init_idx
            log_sum = 0.0
            count = 0

            # iteramos mientras se pueda evolucionar el punto de referencia
            # y sin superar el máximo de iteraciones
            while (idx_f + evol_time < N) and (count < max_replacements):
                # buscamos el vecino más cercano
                idx_n, dist_init = self.nearest_neighbour(data, idx_f, min_separation)

                # comprobamos que la distancia inicial es válida
                if np.isinf(dist_init) or dist_init < 1e-12:
                    break

                # definimos los indices después de la evolución
                idx_f_next = idx_f + evol_time
                idx_n_next = idx_n + evol_time

                # comprobamos si hemos terminado de recorrer los índices
                if idx_f_next >= N or idx_n_next >= N: 
                    break

                # hallamos la separación final
                dist_final = np.linalg.norm(data[idx_f_next] - data[idx_n_next])

                # acumulamos el logaritmo del crecimiento
                log_sum += np.log(dist_final / dist_init)
                count += 1

                # renormalización: movemos el punto de referencia
                idx_f = idx_f_next

            # hallamos el max LCE estimado para este punto de referencia
            lambda_local = (log_sum / (count * evol_time * self.dt)) if count > 0 else np.nan
            lambda_estimates.append(lambda_local)

        # hacemos la media de los LCEs estimados
        lambda1 = np.nanmean(lambda_estimates) if lambda_estimates else 0.0
        return lambda1, lambda_estimates



