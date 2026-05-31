

import numpy as np

from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import calculos
import simulacion


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
        t_est = calculos.elegir_t_estrella(D, h, v, phi, T)

        # 2) construir F(theta) y resolver con los dos metodos
        Ffun = calculos.construir_funcion_objetivo(D, h, v, phi, T, t_est)
        a, b = np.radians(5), np.radians(85)
        try:
            resultados = calculos.comparar_metodos(Ffun, (a, b), np.radians(45))
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
        u = calculos.u_desde_theta(theta, D, h, v, phi, T, t_est)
        x1, y1 = calculos.pos_proyectil_B(t_est, D, h, v, phi)

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

    # ---------------- animacion (delegada al modulo simulacion) ----------------
    def _animar(self):
        if self._anim is not None:
            self._anim.event_source.stop()

        self._anim, info = simulacion.crear_animacion(self.fig, self.ax, self._datos)

        if info["colision"]:
            pr = info["pos"]
            ex, ey = info["error"]
            self.out_real.setText(f"({pr[0]:.2f}, {pr[1]:.2f})")
            self.out_error.setText(f"dx={ex:.2f} m, dy={ey:.2f} m")
        else:
            self.out_real.setText("no detectada")
            self.out_error.setText("—  (sube sigma o ajusta parametros)")

        self.canvas.draw()
