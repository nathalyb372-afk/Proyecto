# Colisión de dos proyectiles — Proyecto modular

Proyecto en Python (PyQt5) que calcula la velocidad `u` y el ángulo `θ` del
proyectil A para que colisione en el aire con el proyectil B, usando dos
métodos numéricos (Bisección y Newton-Raphson), con simulación de viento y
animación progresiva de las trayectorias.

## Estructura (en módulos)

| Archivo | Qué contiene |
|---------|--------------|
| `main.py` | Punto de entrada. **Ejecuta este archivo.** |
| `calculos.py` | Física, elección automática de t*, solución analítica y los dos métodos numéricos. |
| `simulacion.py` | Viento (ruido blanco), detección de colisión real y animación. |
| `interfaz.py` | Interfaz gráfica (PyQt5). |

## Instalación

```bash
pip install PyQt5 numpy matplotlib
```

## Cómo ejecutar

Abre una terminal en esta carpeta y ejecuta:

```bash
python main.py
```

Se abre la ventana. Ingresa los parámetros del proyectil B (`D`, `h`, `v`,
`φ`) y el tiempo `T`. Opcionalmente ajusta el viento. Pulsa
**"Calcular y animar"**.

## Nombres de los proyectiles

- **Proyectil B** = el de `(D, h)`. Sale primero (en t=0). Es el objetivo.
- **Proyectil A** = el de `(0, 0)`. Sale con retraso T. Sus `u` y `θ` se calculan.

(La física respeta el enunciado: el de (D,h) siempre sale primero.)
