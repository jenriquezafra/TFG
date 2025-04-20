import numpy as np
from scipy.integrate import solve_ivp

class DeterministicSystem:
    """
    Clase para definir nuestro sistema dinámico sin ruido y realizar cálculos sobre el mismo.
    """
    def __init__(self, r, a, x0, total_time):
        """
        :param r: vector de tasas de crecimiento (r_i)
        :param a: matriz (N,N) con los coeficientes a_ij
        :param total_time: tiempo total de la simulación
        """
        self.r = np.array(r, dtype = float)
        self.a = np.array(a, dtype = float)
        self.x0 = np.array(x0, dtype = float)
        self.N = self.r.shape[0]
        self.total_time = total_time

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
    
    def RK45(self, n_steps = 1e5, method = 'RK45'):
        """
        Para resolver el sistema
        :param x0: vector inicial de longitud N
        :param n_steps: número de pasos de tiempo
        :param method: método de integración
        :return: (t, X) con t array de tiempos y X matriz (len(t), N)
        """
        t_eval = np.linspace(0, self.total_time, int(n_steps))
        sol = solve_ivp(
            self.system, [0, self.total_time], self.x0, t_eval = t_eval, method = method)
        X = sol.y # cada fila es un estado  en un tiempo dado
        t = sol.t
        return t, X
    
    def jacobian(self, x):
        """
        Cálculo del Jacobiano J(x).
        :param x: vector de longitud N
        :return: numpy array 
        """
        x = np.array(x, dtype=float)
        J = np.zeros((self.N, self.N))
        Ax = self.a.dot(x)
        for i in range(self.N):
            sum_aijxj = Ax[i]
            for k in range(self.N):
                if i == k:
                    J[i, k] = self.r[i] * ((1 - sum_aijxj) - x[i]*self.a[i, i])
                else:
                    J[i, k] = -self.r[i]*x[i]*self.a[i, k]
        return J

    def dynamical_system(self, x, dt):
        """
        TODO: Cambiar el nombre de la función para que sea más descriptivo
        :param x: vector de longitud N
        :param dt: paso de tiempo
        :return: numpy array
        """
        dxdt = self.system(0, x)
        return x + dxdt * dt
    
    def compute_LCEs(self, max_iter):
        """
        Calcula los exponentes de Lyapunov (LCEs) usando ortonormalización QR
        :param x0: vector inicial de longitud N
        :param max_iter: número máximo de iteraciones
        :return: tuple (LCEs, all_logs)
            - LCEs: matriz (max_iter, N) con los LCEs
            - all_logs: matriz (max_iter, N) con los logs de los valores propios
        """
        x = np.array(self.x0, dtype=float) 
        dt = self.total_time/max_iter # paso de tiempo 
        
        # creamos las matrices necesarias
        Q = np.eye(self.N)
        sum_logs_R = np.zeros(self.N)

        # guardamos para cada iteración
        LCEs = np.zeros((max_iter, self.N))
        all_logs = np.zeros_like(LCEs)

        for i in range(max_iter):
            # hacemos la descomposición QR
            J = self.jacobian(x)
            J_star = J @ Q
            Q, R = np.linalg.qr(J_star)

            # nos quedamos con los elementos diagonales
            sum_logs_R += np.log(np.abs(np.diag(R)))
            all_logs[i] = sum_logs_R
            LCEs[i] = sum_logs_R /(1+i)

            # actualizamos el sistema
            sol = solve_ivp(self.system, [0, dt], x, method = 'RK45')
            x = sol.y[:,-1]

        return LCEs, all_logs
