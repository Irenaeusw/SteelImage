from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys

def window(top_left_position, window_dim):
    app = QApplication(sys.argv)
    win = QMainWindow()
    win.setGeometry(top_left_position[0], top_left_position[1], window_dim[0], window_dim[1])
    win.setWindowTitle("Steel Image Microstructure Analysis")

    # Create a label
    label = QtWidgets.QLabel(win)
    label.setText("Pearlites are wack")
    label.move(50,50)

    win.show()
    sys.exit(app.exec_())

window((200,200), (500,500))
