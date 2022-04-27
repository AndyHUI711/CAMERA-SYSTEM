##Copyright
##written by HUI, CHEUNG YUEN
##Student of HKUST
##FYP, FINAL YEAR PROJECT
from PyQt5 import QtWidgets

from login import MainWindow_controller, win_Login, win_Register, MainWindow
from lib.share import shareInfo



if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    #shareInfo.loginWin = MainWindow_controller()
    shareInfo.loginWin = win_Login()
    shareInfo.loginWin.show()
    sys.exit(app.exec_())

#pip install PyQt5
#pip install PyQt5-tools
#pyuic5 -x UI.ui -o UI.py
#pip install opencv-contrib-python
#pip install opencv-python
