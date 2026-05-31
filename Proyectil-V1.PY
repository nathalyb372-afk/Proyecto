

import sys
import numpy as np

from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.animation as animation

G = 9.81  # aceleracion de la gravedad (m/s^2)


# ===========================================================================
#  1) FISICA: ecuaciones de movimiento
# ===========================================================================
def pos_proyectil1(t, D, h, v, phi):
    """
    Posicion (x, y) del primer proyectil en el instante global t.
    El proyectil viaja hacia el origen, por eso la componente horizontal
    es negativa: vx = -v*cos(phi). La vertical sube: vy = +v*sin(phi).
    """
    x = D - v * np.cos(phi) * t
    y = h + v * np.sin(phi) * t - 0.5 * G * t ** 2
    return x, y


def pos_proyectil2(t, u, theta, T):
    """Posicion (x, y) del segundo proyectil en t (None si aun no salio)."""
    tau = t - T
    if tau < 0:
        return None
    x = u * np.cos(theta) * tau
    y = u * np.sin(theta) * tau - 0.5 * G * tau ** 2
    return x, y


def resolver_analitico(D, h, v, phi, T, t_estrella):
    """Solucion exacta de u y theta (referencia para validar los metodos)."""
    tau = t_estrella - T
    x1, y1 = pos_proyectil1(t_estrella, D, h, v, phi)
    P = x1 / tau
    Q = (y1 + 0.5 * G * tau ** 2) / tau
    theta = np.arctan2(Q, P)
    u = np.hypot(P, Q)
    return u, theta, (x1, y1)


def tiempo_vuelo_proyectil1(h, v, phi):
    """
    Tiempo que tarda el primer proyectil en tocar el suelo (y = 0).
    Resuelve  h + v*sin(phi)*t - 0.5*g*t^2 = 0  (formula cuadratica).
    """
    a = 0.5 * G
    b = -v * np.sin(phi)
    c = -h
    disc = b ** 2 - 4 * a * c
    if disc < 0:
        return 0.0
    return (-b + np.sqrt(disc)) / (2 * a)


def elegir_t_estrella(D, h, v, phi, T):
    """
    Elige automaticamente un instante de colision t* > T valido.

    Criterio: se busca un t* que deje al segundo proyectil suficiente tiempo
    de vuelo (tau = t* - T) para describir una PARABOLA clara y necesitar una
    rapidez u razonable, evitando que tau sea muy pequeno (lo que disparria u
    y aplanaria la trayectoria de B).

    De entre los instantes validos (punto de choque sobre el suelo y delante
    del origen), se elige aquel cuya rapidez u requerida sea MINIMA. Ese punto
    da la interceptacion mas "natural" y siempre genera una parabola visible.
    """
    # tiempo total de vuelo del proyectil 1 (hasta tocar el suelo)
    t_suelo = tiempo_vuelo_proyectil1(h, v, phi)
    if t_suelo <= T:
        t_suelo = T + 3.0  # margen por si el proyectil 1 cae antes de T

    # exploramos instantes candidatos entre T (un poco despues) y t_suelo
    candidatos = np.linspace(T + 0.3, t_suelo - 0.05, 400)

    # altura de referencia: exigimos que el choque ocurra por encima de este
    # valor para que no quede pegado al suelo (20% de la altura del apice de A)
    vy = v * np.sin(phi)
    y_apex = h + (vy ** 2) / (2 * G)
    y_min_choque = 0.20 * y_apex

    mejor_t = None
    mejor_u = np.inf
    for t in candidatos:
        x1, y1 = pos_proyectil1(t, D, h, v, phi)
        if y1 < y_min_choque or x1 < 0:   # punto de choque invalido o muy bajo
            continue
        tau = t - T
        # rapidez u necesaria para que B llegue a (x1, y1) en tau
        ux = x1 / tau
        uy = (y1 + 0.5 * G * tau ** 2) / tau
        u = np.hypot(ux, uy)
        if u < mejor_u:             # nos quedamos con la u minima
            mejor_u = u
            mejor_t = t

    # si la restriccion de altura fue muy estricta, relajamos a y1 >= 0
    if mejor_t is None:
        for t in candidatos:
            x1, y1 = pos_proyectil1(t, D, h, v, phi)
            if y1 < 0 or x1 < 0:
                continue
            tau = t - T
            ux = x1 / tau
            uy = (y1 + 0.5 * G * tau ** 2) / tau
            u = np.hypot(ux, uy)
            if u < mejor_u:
                mejor_u = u
                mejor_t = t

    # si nada fue valido, usamos un instante por defecto
    if mejor_t is None:
        mejor_t = T + 1.0
    return mejor_t


def construir_funcion_objetivo(D, h, v, phi, T, t_estrella):
    """
    Reduce el problema a UNA incognita (theta).
    Se fija u*cos(theta) = x1/tau (la velocidad horizontal del proyectil 2
    para llegar a x1 en el tiempo tau) y se exige que su altura iguale la
    del proyectil 1 en t*:   F(theta) = y2 - y1 = 0.
    """
    tau = t_estrella - T
    x1, y1 = pos_proyectil1(t_estrella, D, h, v, phi)

    def F(theta):
        c = np.cos(theta)
        if abs(c) < 1e-9:
            return 1e9  # evita division por cero (theta ~ 90 grados)
        u = (x1 / tau) / c
        y2 = u * np.sin(theta) * tau - 0.5 * G * tau ** 2
        return y2 - y1

    return F


def u_desde_theta(theta, D, h, v, phi, T, t_estrella):
    """Recupera la rapidez u una vez hallado el angulo theta."""
    tau = t_estrella - T
    x1, _ = pos_proyectil1(t_estrella, D, h, v, phi)
    return (x1 / tau) / np.cos(theta)


# ===========================================================================
#  2) METODOS NUMERICOS de busqueda de raiz para resolver F(theta) = 0
# ===========================================================================
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


# ===========================================================================
#  3) VIENTO: trayectorias con ruido blanco + deteccion de colision real
# ===========================================================================
def simular_dos_proyectiles(D, h, v, phi, T, u, theta, t_total,
                            sigma=0.0, dt=0.01, dist_colision=2.0, semilla=None):
    """
    Integra (metodo de Euler) AMBOS proyectiles paso a paso, sumando ruido
    blanco a la velocidad en cada paso (efecto del viento). Detecta la
    colision REAL cuando la distancia entre los dos proyectiles es menor que
    dist_colision.

    Devuelve un diccionario con las trayectorias y los datos de la colision
    real (posicion y tiempo), util para comparar contra el valor teorico.
    """
    rng = np.random.default_rng(semilla)

    # estado inicial del proyectil 1 (viaja hacia el origen: vx negativo)
    pos1 = np.array([float(D), float(h)])
    vel1 = np.array([-v * np.cos(phi), v * np.sin(phi)])

    # el proyectil 2 aun no ha salido
    pos2 = np.array([0.0, 0.0])
    vel2 = np.array([0.0, 0.0])
    lanzado2 = False

    x1s, y1s, x2s, y2s, ts = [], [], [], [], []
    colision = False
    pos_colision = None
    t_colision = None

    n_pasos = int(t_total / dt)
    for k in range(n_pasos + 1):
        t = k * dt

        # ruido del viento (cero si sigma = 0)
        ruido1 = rng.normal(0, sigma, 2) * dt if sigma > 0 else np.zeros(2)
        ruido2 = rng.normal(0, sigma, 2) * dt if (sigma > 0 and lanzado2) else np.zeros(2)

        # integracion del proyectil 1
        vel1 += np.array([0.0, -G]) * dt + ruido1
        pos1 += vel1 * dt

        # integracion del proyectil 2 (solo despues de T)
        if t >= T:
            if not lanzado2:
                vel2 = np.array([u * np.cos(theta), u * np.sin(theta)])
                lanzado2 = True
            vel2 += np.array([0.0, -G]) * dt + ruido2
            pos2 += vel2 * dt

        x1s.append(pos1[0]); y1s.append(pos1[1])
        x2s.append(pos2[0]); y2s.append(pos2[1])
        ts.append(t)

        # deteccion de colision real
        if lanzado2 and not colision:
            d = np.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])
            if d < dist_colision:
                colision = True
                pos_colision = ((pos1[0] + pos2[0]) / 2.0, (pos1[1] + pos2[1]) / 2.0)
                t_colision = t

        if pos1[1] < 0:  # el proyectil 1 toco el suelo, detenemos
            break

    return {
        "x1": np.array(x1s), "y1": np.array(y1s),
        "x2": np.array(x2s), "y2": np.array(y2s),
        "t": np.array(ts),
        "colision": colision,
        "pos_colision": pos_colision,
        "t_colision": t_colision,
    }


# ===========================================================================
#  4) INTERFAZ GRAFICA (PyQt5) + ANIMACION
# ===========================================================================
class VentanaPrincipal(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Colision de dos proyectiles - Metodos Numericos")
        self.resize(1200, 760)
        self._anim = None
        self._datos = None

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        layout.addWidget(self._panel_controles(), 0)
        layout.addWidget(self._panel_grafico(), 1)

    # ---------------- panel izquierdo: controles ----------------
    def _panel_controles(self):
        panel = QtWidgets.QWidget()
        panel.setMaximumWidth(440)
        v = QtWidgets.QVBoxLayout(panel)

        g1 = QtWidgets.QGroupBox("Parametros del proyectil B (sale de D, h)")
        form = QtWidgets.QFormLayout(g1)
        self.in_D = QtWidgets.QDoubleSpinBox(); self.in_D.setRange(0, 2000); self.in_D.setValue(120)
        self.in_h = QtWidgets.QDoubleSpinBox(); self.in_h.setRange(0, 2000); self.in_h.setValue(20)
        self.in_v = QtWidgets.QDoubleSpinBox(); self.in_v.setRange(0.1, 500); self.in_v.setValue(25)
        self.in_phi = QtWidgets.QDoubleSpinBox(); self.in_phi.setRange(1, 89); self.in_phi.setValue(45)
        self.in_T = QtWidgets.QDoubleSpinBox(); self.in_T.setRange(0, 60); self.in_T.setSingleStep(0.1); self.in_T.setValue(2.0)
        form.addRow("D (m):", self.in_D)
        form.addRow("h (m):", self.in_h)
        form.addRow("v (m/s):", self.in_v)
        form.addRow("phi (grados, 1-89):", self.in_phi)
        form.addRow("T (s):", self.in_T)
        v.addWidget(g1)

        g2 = QtWidgets.QGroupBox("Viento (ruido blanco)")
        form2 = QtWidgets.QFormLayout(g2)
        self.in_sigma = QtWidgets.QDoubleSpinBox(); self.in_sigma.setRange(0, 20); self.in_sigma.setSingleStep(0.1); self.in_sigma.setValue(0.0)
        self.in_dt = QtWidgets.QDoubleSpinBox(); self.in_dt.setRange(0.005, 0.5); self.in_dt.setSingleStep(0.005); self.in_dt.setDecimals(3); self.in_dt.setValue(0.02)
        form2.addRow("Intensidad sigma:", self.in_sigma)
        form2.addRow("dt integracion (s):", self.in_dt)
        v.addWidget(g2)

        self.btn = QtWidgets.QPushButton("Calcular y animar")
        self.btn.clicked.connect(self.calcular)
        v.addWidget(self.btn)

        g3 = QtWidgets.QGroupBox("Resultado (proyectil A, sale de 0,0)")
        form3 = QtWidgets.QFormLayout(g3)
        self.out_t = QtWidgets.QLabel("-")
        self.out_u = QtWidgets.QLabel("-")
        self.out_theta = QtWidgets.QLabel("-")
        self.out_punto = QtWidgets.QLabel("-")
        for lab in (self.out_u, self.out_theta):
            lab.setStyleSheet("font-weight:bold; color:#185FA5; font-size:14px;")
        form3.addRow("t* (auto):", self.out_t)
        form3.addRow("u (m/s):", self.out_u)
        form3.addRow("theta (grados):", self.out_theta)
        form3.addRow("Punto choque teorico:", self.out_punto)
        v.addWidget(g3)

        g4 = QtWidgets.QGroupBox("Comparacion de metodos numericos")
        v4 = QtWidgets.QVBoxLayout(g4)
        self.tabla = QtWidgets.QTableWidget(2, 4)
        self.tabla.setHorizontalHeaderLabels(["Metodo", "Iter", "theta (deg)", "Error"])
        self.tabla.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.tabla.verticalHeader().setVisible(False)
        v4.addWidget(self.tabla)
        v.addWidget(g4)

        g5 = QtWidgets.QGroupBox("Colision real (con viento)")
        form5 = QtWidgets.QFormLayout(g5)
        self.out_real = QtWidgets.QLabel("-")
        self.out_error = QtWidgets.QLabel("-")
        form5.addRow("Posicion real:", self.out_real)
        form5.addRow("Error vs teorico:", self.out_error)
        v.addWidget(g5)

        v.addStretch(1)
        return panel

    # ---------------- panel derecho: grafico ----------------
    def _panel_grafico(self):
        panel = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(panel)
        self.fig = Figure(figsize=(6.5, 5.5))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("x (m)")
        self.ax.set_ylabel("y (m)")
        self.ax.set_title("Trayectorias")
        v.addWidget(self.canvas)
        return panel

    # ---------------- logica principal ----------------
    def calcular(self):
        D = self.in_D.value()
        h = self.in_h.value()
        v = self.in_v.value()
        phi = np.radians(self.in_phi.value())
        T = self.in_T.value()
        sigma = self.in_sigma.value()
        dt = self.in_dt.value()

        # 1) elegir t* automaticamente
        t_est = elegir_t_estrella(D, h, v, phi, T)

        # 2) construir F(theta) y resolver con los tres metodos
        Ffun = construir_funcion_objetivo(D, h, v, phi, T, t_est)
        a, b = np.radians(5), np.radians(85)
        try:
            resultados = comparar_metodos(Ffun, (a, b), np.radians(45))
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", str(e))
            return

        # 3) tomar como definitiva la solucion de Newton (o la que converja)
        solucion = None
        for r in resultados:
            if r.get("convergio") and r.get("raiz") is not None:
                solucion = r
                if r["metodo"] == "Newton-Raphson":
                    break
        if solucion is None:
            QtWidgets.QMessageBox.warning(self, "Error",
                "Ningun metodo convergio. Prueba otro angulo phi o velocidad v.")
            return

        theta = solucion["raiz"]
        u = u_desde_theta(theta, D, h, v, phi, T, t_est)
        x1, y1 = pos_proyectil1(t_est, D, h, v, phi)

        # 4) mostrar resultados
        self.out_t.setText(f"{t_est:.3f} s")
        self.out_u.setText(f"{u:.3f}")
        self.out_theta.setText(f"{np.degrees(theta):.3f}")
        self.out_punto.setText(f"({x1:.2f}, {y1:.2f})")
        self._llenar_tabla(resultados)

        # 5) guardar datos y animar
        self._datos = dict(D=D, h=h, v=v, phi=phi, T=T, u=u, theta=theta,
                           t_est=t_est, sigma=sigma, dt=dt, punto=(x1, y1))
        self._animar()

    def _llenar_tabla(self, resultados):
        for fila, r in enumerate(resultados):
            raiz = r.get("raiz")
            err = r.get("error_final")
            theta_txt = f"{np.degrees(raiz):.4f}" if (raiz is not None and r.get('convergio')) else "—"
            err_txt = f"{err:.2e}" if (err is not None) else "—"
            valores = [r.get("metodo", "?"), str(r.get("n_iter", 0)), theta_txt, err_txt]
            for col, val in enumerate(valores):
                item = QtWidgets.QTableWidgetItem(val)
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.tabla.setItem(fila, col, item)

    # ---------------- animacion ----------------
    def _animar(self):
        d = self._datos
        if self._anim is not None:
            self._anim.event_source.stop()

        self.ax.clear()
        self.ax.set_xlabel("x (m)")
        self.ax.set_ylabel("y (m)")
        self.ax.set_title("Trayectorias hasta la colision")
        self.ax.grid(True, alpha=0.3)

        t_est = d["t_est"]
        t_total = t_est + 0.5  # un poco mas alla del choque teorico

        # simulamos ambos proyectiles paso a paso (con o sin viento)
        sim = simular_dos_proyectiles(
            d["D"], d["h"], d["v"], d["phi"], d["T"], d["u"], d["theta"],
            t_total, sigma=d["sigma"], dt=d["dt"], semilla=1)

        x1, y1 = sim["x1"], sim["y1"]
        x2, y2 = sim["x2"], sim["y2"]
        t_arr = sim["t"]

        # solo mostramos el proyectil 2 a partir de T
        mask2 = t_arr >= d["T"]

        # rastros que se iran DIBUJANDO progresivamente (empiezan vacios)
        rastro1, = self.ax.plot([], [], "-", color="#E24B4A", alpha=0.7, lw=2, label="Proyectil B (D, h)")
        rastro2, = self.ax.plot([], [], "-", color="#185FA5", alpha=0.7, lw=2, label="Proyectil A (0, 0)")

        # punto de colision teorico
        self.ax.plot(*d["punto"], "*", color="#1E8E3E", markersize=16,
                     label=f"Colision teorica ({d['punto'][0]:.1f}, {d['punto'][1]:.1f})")

        # punto de colision real (si se detecto)
        if sim["colision"]:
            pr = sim["pos_colision"]
            self.ax.plot(pr[0], pr[1], "o", color="#EF9F27", markersize=12,
                         markerfacecolor="none", markeredgewidth=2.5,
                         label=f"Colision real ({pr[0]:.1f}, {pr[1]:.1f})")
            ex = abs(pr[0] - d["punto"][0])
            ey = abs(pr[1] - d["punto"][1])
            self.out_real.setText(f"({pr[0]:.2f}, {pr[1]:.2f})")
            self.out_error.setText(f"dx={ex:.2f} m, dy={ey:.2f} m")
        else:
            self.out_real.setText("no detectada")
            self.out_error.setText("—  (sube sigma o ajusta parametros)")

        # puntos moviles (las pelotas)
        p1, = self.ax.plot([], [], "o", color="#E24B4A", markersize=11)
        p2, = self.ax.plot([], [], "o", color="#185FA5", markersize=11)
        self.ax.legend(loc="upper right", fontsize=9)

        # fijamos los limites de los ejes desde el inicio (con las trayectorias
        # completas) para que la escala no salte mientras se dibuja
        all_x = np.concatenate([x1, x2[mask2], [d["punto"][0]]])
        all_y = np.concatenate([y1, y2[mask2], [d["punto"][1]]])
        self.ax.set_xlim(min(all_x.min(), 0) - 5, all_x.max() + 5)
        self.ax.set_ylim(0, all_y.max() + 5)

        # frames de la animacion (submuestreamos para que no sea lentisimo)
        idx = np.arange(0, len(t_arr), max(1, len(t_arr) // 120))

        def init():
            rastro1.set_data([], [])
            rastro2.set_data([], [])
            p1.set_data([], [])
            p2.set_data([], [])
            return rastro1, rastro2, p1, p2

        def update(j):
            i = idx[j]
            # el rastro del proyectil B crece desde el inicio hasta i
            rastro1.set_data(x1[:i + 1], y1[:i + 1])
            p1.set_data([x1[i]], [y1[i]])
            # el proyectil A solo aparece (y deja rastro) a partir de t = T
            if t_arr[i] >= d["T"]:
                ini2 = np.searchsorted(t_arr, d["T"])  # indice donde empieza A
                rastro2.set_data(x2[ini2:i + 1], y2[ini2:i + 1])
                p2.set_data([x2[i]], [y2[i]])
            return rastro1, rastro2, p1, p2

        self._anim = animation.FuncAnimation(
            self.fig, update, frames=len(idx), init_func=init,
            interval=40, blit=True, repeat=False)
        self.canvas.draw()


# ===========================================================================
#  5) PUNTO DE ENTRADA
# ===========================================================================
def main():
    app = QtWidgets.QApplication(sys.argv)
    win = VentanaPrincipal()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
