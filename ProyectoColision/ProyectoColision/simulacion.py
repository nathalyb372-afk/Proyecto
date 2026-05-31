

import numpy as np
import matplotlib.animation as animation

import calculos

G = 9.81


# ---------------------------------------------------------------------------
#  SIMULACION CON VIENTO + DETECCION DE COLISION REAL
# ---------------------------------------------------------------------------
def simular_dos_proyectiles(D, h, v, phi, T, u, theta, t_total,
                            sigma=0.0, dt=0.01, dist_colision=2.0, semilla=None):

    rng = np.random.default_rng(semilla)

    # estado inicial del proyectil B (viaja hacia el origen)
    posB = np.array([float(D), float(h)])
    velB = np.array([-v * np.cos(phi), v * np.sin(phi)])

    # el proyectil A aun no ha salido
    posA = np.array([0.0, 0.0])
    velA = np.array([0.0, 0.0])
    lanzadoA = False

    xB, yB, xA, yA, ts = [], [], [], [], []
    colision = False
    pos_colision = None
    t_colision = None

    n_pasos = int(t_total / dt)
    for k in range(n_pasos + 1):
        t = k * dt

        # ruido del viento (cero si sigma = 0)
        ruidoB = rng.normal(0, sigma, 2) * dt if sigma > 0 else np.zeros(2)
        ruidoA = rng.normal(0, sigma, 2) * dt if (sigma > 0 and lanzadoA) else np.zeros(2)

        # integracion del proyectil B (objetivo)
        velB += np.array([0.0, -G]) * dt + ruidoB
        posB += velB * dt

        # integracion del proyectil A (interceptor, solo despues de T)
        if t >= T:
            if not lanzadoA:
                velA = np.array([u * np.cos(theta), u * np.sin(theta)])
                lanzadoA = True
            velA += np.array([0.0, -G]) * dt + ruidoA
            posA += velA * dt

        xB.append(posB[0]); yB.append(posB[1])
        xA.append(posA[0]); yA.append(posA[1])
        ts.append(t)

        # deteccion de colision real
        if lanzadoA and not colision:
            d = np.hypot(posB[0] - posA[0], posB[1] - posA[1])
            if d < dist_colision:
                colision = True
                pos_colision = ((posB[0] + posA[0]) / 2.0, (posB[1] + posA[1]) / 2.0)
                t_colision = t

        if posB[1] < 0:  # el proyectil B toco el suelo, detenemos
            break

    return {
        "xB": np.array(xB), "yB": np.array(yB),
        "xA": np.array(xA), "yA": np.array(yA),
        "t": np.array(ts),
        "colision": colision,
        "pos_colision": pos_colision,
        "t_colision": t_colision,
    }


# ---------------------------------------------------------------------------
#  ANIMACION (las trayectorias se van dibujando progresivamente)
# ---------------------------------------------------------------------------
def crear_animacion(fig, ax, datos):
    
    d = datos
    ax.clear()
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Trayectorias hasta la colision")
    ax.grid(True, alpha=0.3)

    t_est = d["t_est"]
    t_total = t_est + 0.5

    sim = simular_dos_proyectiles(
        d["D"], d["h"], d["v"], d["phi"], d["T"], d["u"], d["theta"],
        t_total, sigma=d["sigma"], dt=d["dt"], semilla=1)

    xB, yB = sim["xB"], sim["yB"]
    xA, yA = sim["xA"], sim["yA"]
    t_arr = sim["t"]
    mask_A = t_arr >= d["T"]

    # rastros que se iran DIBUJANDO progresivamente (empiezan vacios)
    rastroB, = ax.plot([], [], "-", color="#E24B4A", alpha=0.7, lw=2, label="Proyectil B (D, h)")
    rastroA, = ax.plot([], [], "-", color="#185FA5", alpha=0.7, lw=2, label="Proyectil A (0, 0)")

    # punto de colision teorico
    ax.plot(*d["punto"], "*", color="#1E8E3E", markersize=16,
            label=f"Colision teorica ({d['punto'][0]:.1f}, {d['punto'][1]:.1f})")

    info_colision = {"colision": sim["colision"], "pos": sim["pos_colision"]}

    # punto de colision real (si se detecto)
    if sim["colision"]:
        pr = sim["pos_colision"]
        ax.plot(pr[0], pr[1], "o", color="#EF9F27", markersize=12,
                markerfacecolor="none", markeredgewidth=2.5,
                label=f"Colision real ({pr[0]:.1f}, {pr[1]:.1f})")
        info_colision["error"] = (abs(pr[0] - d["punto"][0]), abs(pr[1] - d["punto"][1]))

    # puntos moviles (las pelotas)
    pB, = ax.plot([], [], "o", color="#E24B4A", markersize=11)
    pA, = ax.plot([], [], "o", color="#185FA5", markersize=11)
    ax.legend(loc="upper right", fontsize=9)

    # limites de ejes fijos desde el inicio (para que la escala no salte)
    all_x = np.concatenate([xB, xA[mask_A], [d["punto"][0]]])
    all_y = np.concatenate([yB, yA[mask_A], [d["punto"][1]]])
    ax.set_xlim(min(all_x.min(), 0) - 5, all_x.max() + 5)
    ax.set_ylim(0, all_y.max() + 5)

    # submuestreo para que la animacion no sea lentisima
    idx = np.arange(0, len(t_arr), max(1, len(t_arr) // 120))
    ini_A = np.searchsorted(t_arr, d["T"])  # indice donde empieza A

    def init():
        rastroB.set_data([], [])
        rastroA.set_data([], [])
        pB.set_data([], [])
        pA.set_data([], [])
        return rastroB, rastroA, pB, pA

    def update(j):
        i = idx[j]
        # el rastro del proyectil B crece desde el inicio hasta i
        rastroB.set_data(xB[:i + 1], yB[:i + 1])
        pB.set_data([xB[i]], [yB[i]])
        # el proyectil A solo aparece (y deja rastro) a partir de t = T
        if t_arr[i] >= d["T"]:
            rastroA.set_data(xA[ini_A:i + 1], yA[ini_A:i + 1])
            pA.set_data([xA[i]], [yA[i]])
        return rastroB, rastroA, pB, pA

    anim = animation.FuncAnimation(
        fig, update, frames=len(idx), init_func=init,
        interval=40, blit=True, repeat=False)

    return anim, info_colision
