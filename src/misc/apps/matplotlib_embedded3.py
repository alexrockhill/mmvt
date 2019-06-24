import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget,\
    QPushButton, QSlider
from PyQt5.QtGui import QIcon
#******************************************************************
import matplotlib
matplotlib.use("Qt5Agg")
#******************************************************************
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import random
#******************************************************************


class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.left = 10
        self.top = 10
        self.title = 'PyQt5 matplotlib example - pythonspot.com'
        self.width = 640
        self.height = 400
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.m = m = PlotCanvas(self, width=5, height=4)
        m.move(0,0)

        sld = QSlider(Qt.Vertical, self)
        sld.setFocusPolicy(Qt.NoFocus)
        sld.setGeometry(500, 40, 100, 90)
        sld.valueChanged[int].connect(self.changeValue)
        # button.setToolTip('This s an example button')
        # button.move(500,0)
        # button.resize(140,100)

        self.show()

    def changeValue(self, value):
        print(value)
        self.m.plot(value)


class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        self.ax = self.figure.add_subplot(111)
        self.powers = np.load('/home/npeled/meg/nmr01325/norm_powers_max.npy')
        self.plot(2)


    def plot(self, threshold):
        powers = self.powers.copy()
        powers[np.where(np.abs(powers) < threshold)] = 0
        _plot_powers(self.powers, self.ax)
        self.ax.set_title('PyQt Matplotlib Example')
        self.draw()


def _plot_powers(powers, ax, xaxis=None, high_gamma_max=300):
    # from src.utils import color_maps_utils as cmu
    # BuPu_YlOrRd_cm = cmu.create_BuPu_YlOrRd_cm()

    def extents(f):
        delta = f[1] - f[0]
        return [f[0] - delta / 2, f[-1] + delta / 2]

    times = np.arange(powers.shape[1]) if xaxis is None else xaxis
    freqs = np.concatenate([np.arange(1, 30), np.arange(31, 60, 3), np.arange(60, high_gamma_max + 5, 5)])
    if powers.shape[0] != len(freqs):
        print('powers.shape[0] != len(freqs)!!!')
        return

    if isinstance(powers, np.ndarray):
        vmax, vmin = np.max(powers), np.min(powers)
    else:
        vmax, vmin = np.ma.masked_array.max(powers), np.ma.masked_array.min(powers)
    if vmin > 0:
        cmap = matplotlib.cm.YlOrRd
        # cmap = 'YlOrRd'
    elif vmax < 0:
        cmap = matplotlib.cm.BuPu
        # cmap = 'BuPu'
    else:
        cmap = matplotlib.cm.coolwarm # BuPu_YlOrRd_cm
        maxmin = max(map(abs, [vmax, vmin]))
        vmin, vmax = -maxmin, maxmin
    # powers[np.where(powers == 0)] = np.nan
    # cmap.set_bad(color='white')
    im = ax.imshow(np.flip(powers, 0), vmin=vmin, vmax=vmax, aspect='auto', interpolation='nearest',
               extent=extents(times) + extents(freqs), cmap=cmap)
    return im



if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
