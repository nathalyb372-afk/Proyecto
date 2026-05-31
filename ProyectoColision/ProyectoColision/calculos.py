
import numpy as np

G = 9.81 

def pos_proyectil_B(t, D, h, v, phi):
 x = D - v * np.cos(phi) * t
 y = h + v * np.sin(phi) * t - 0.5 * G * t ** 2
 return x, y


def pos_proyectil_A(t, u, theta, T):
    """Posicion (x, y) del proyectil A (sale de (0,0) en t=T). None si aun no salio."""
    tau = t - T
    if tau < 0:
        return None
    x = u * np.cos(theta) * tau
    y = u * np.sin(theta) * tau - 0.5 * G * tau ** 2
    return x, y


def tiempo_vuelo_proyectil_B(h, v, phi):
    """
    Tiempo que tarda el proyectil B en tocar el suelo (y = 0).
    Resuelve  h + v*sin(phi)*t - 0.5*g*t^2 = 0  (formula cuadratica).
    """
    a = 0.5 * G
    b = -v * np.sin(phi)
    c = -h
    disc = b ** 2 - 4 * a * c
    if disc < 0:
        return 0.0
    return (-b + np.sqrt(disc)) / (2 * a)


# ---------------------------------------------------------------------------
#  SOLUCION ANALITICA (referencia)
# ---------------------------------------------------------------------------
def resolver_analitico(D, h, v, phi, T, t_estrella):
    """Solucion exacta de u y theta. Sirve para validar los metodos numericos."""
    tau = t_estrella - T
    x1, y1 = pos_proyectil_B(t_estrella, D, h, v, phi)
    P = x1 / tau
    Q = (y1 + 0.5 * G * tau ** 2) / tau
    theta = np.arctan2(Q, P)
    u = np.hypot(P, Q)
    return u, theta, (x1, y1)


# ---------------------------------------------------------------------------
#  ELECCION AUTOMATICA DEL INSTANTE DE COLISION t*
# ---------------------------------------------------------------------------
def elegir_t_estrella(D, h, v, phi, T):

    t_suelo = tiempo_vuelo_proyectil_B(h, v, phi)
    if t_suelo <= T:
        t_suelo = T + 3.0

    candidatos = np.linspace(T + 0.3, t_suelo - 0.05, 400)

    # altura minima del choque: 20% de la altura del apice del proyectil B
    vy = v * np.sin(phi)
    y_apex = h + (vy ** 2) / (2 * G)
    y_min_choque = 0.20 * y_apex

    mejor_t = None
    mejor_u = np.inf
    for t in candidatos:
        x1, y1 = pos_proyectil_B(t, D, h, v, phi)
        if y1 < y_min_choque or x1 < 0:
            continue
        tau = t - T
        ux = x1 / tau
        uy = (y1 + 0.5 * G * tau ** 2) / tau
        u = np.hypot(ux, uy)
        if u < mejor_u:
            mejor_u = u
            mejor_t = t

    # si la restriccion de altura fue muy estricta, relajamos a y1 >= 0
    if mejor_t is None:
        for t in candidatos:
            x1, y1 = pos_proyectil_B(t, D, h, v, phi)
            if y1 < 0 or x1 < 0:
                continue
            tau = t - T
            ux = x1 / tau
            uy = (y1 + 0.5 * G * tau ** 2) / tau
            u = np.hypot(ux, uy)
            if u < mejor_u:
                mejor_u = u
                mejor_t = t

    if mejor_t is None:
        mejor_t = T + 1.0
    return mejor_t


# ---------------------------------------------------------------------------
#  FUNCION OBJETIVO (reduce el problema a una sola incognita: theta)
# ---------------------------------------------------------------------------
def construir_funcion_objetivo(D, h, v, phi, T, t_estrella):
    """
    Se fija u*cos(theta) = x1/tau (velocidad horizontal del proyectil A para
    llegar a x1 en el tiempo tau) y se exige que su altura iguale la del
    proyectil B en t*:   F(theta) = y_A - y_B = 0.
    """
    tau = t_estrella - T
    x1, y1 = pos_proyectil_B(t_estrella, D, h, v, phi)

    def F(theta):
        c = np.cos(theta)
        if abs(c) < 1e-9:
            return 1e9
        u = (x1 / tau) / c
        y_A = u * np.sin(theta) * tau - 0.5 * G * tau ** 2
        return y_A - y1

    return F


def u_desde_theta(theta, D, h, v, phi, T, t_estrella):
    """Recupera la rapidez u una vez hallado el angulo theta."""
    tau = t_estrella - T
    x1, _ = pos_proyectil_B(t_estrella, D, h, v, phi)
    return (x1 / tau) / np.cos(theta)


# ---------------------------------------------------------------------------
#  METODOS NUMERICOS de busqueda de raiz para resolver F(theta) = 0
# ---------------------------------------------------------------------------
def biseccion(F, a, b, tol=1e-8, max_iter=200):
    """Robusto: parte el intervalo a la mitad. Necesita cambio de signo."""
    historial = []
    fa, fb = F(a), F(b)
    if fa * fb > 0:
        raise ValueError("Biseccion: F(a) y F(b) deben tener signos opuestos.")
    c = a
    for i in range(1, max_iter + 1):
        c = (a + b) / 2.0
        fc = F(c)
        historial.append((i, c, fc))
        if abs(fc) < tol or (b - a) / 2.0 < tol:
            return c, {"metodo": "Biseccion", "n_iter": i,
                       "error_final": abs(fc), "historial": historial,
                       "convergio": True}
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return c, {"metodo": "Biseccion", "n_iter": max_iter,
               "error_final": abs(F(c)), "historial": historial,
               "convergio": False}


def newton_raphson(F, x0, tol=1e-8, max_iter=200, dx=1e-6):
    """El mas rapido: usa la derivada (aproximada por diferencia central)."""
    historial = []
    x = x0
    for i in range(1, max_iter + 1):
        fx = F(x)
        dfx = (F(x + dx) - F(x - dx)) / (2 * dx)
        if abs(dfx) < 1e-14:
            break
        x_new = x - fx / dfx
        f_new = F(x_new)
        historial.append((i, x_new, f_new))
        if abs(f_new) < tol or abs(x_new - x) < tol:
            return x_new, {"metodo": "Newton-Raphson", "n_iter": i,
                           "error_final": abs(f_new), "historial": historial,
                           "convergio": True}
        x = x_new
    return x, {"metodo": "Newton-Raphson", "n_iter": len(historial),
               "error_final": abs(F(x)), "historial": historial,
               "convergio": False}


def comparar_metodos(F, intervalo, semilla):
    """Corre los DOS metodos (Biseccion y Newton) sobre la misma F."""
    a, b = intervalo
    resultados = []
    try:
        r, info = biseccion(F, a, b)
        info["raiz"] = r
        resultados.append(info)
    except ValueError as e:
        resultados.append({"metodo": "Biseccion", "convergio": False,
                           "error_msg": str(e), "n_iter": 0,
                           "error_final": None, "raiz": None, "historial": []})
    r, info = newton_raphson(F, semilla)
    info["raiz"] = r
    resultados.append(info)
    return resultados
