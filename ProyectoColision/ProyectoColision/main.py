

import sys
from PyQt5 import QtWidgets

from interfaz import VentanaPrincipal


def main():
    """Funcion principal: crea la aplicacion y muestra la ventana."""
    app = QtWidgets.QApplication(sys.argv)
    ventana = VentanaPrincipal()
    ventana.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
