# coding: utf-8
from PyQt5.QtCore import QUrl, QSize
from PyQt5.QtGui import QIcon, QColor
from PyQt5.QtWidgets import QApplication

from qfluentwidgets import NavigationItemPosition, FluentWindow, SplashScreen
from qfluentwidgets import FluentIcon as FIF

from .focus_interface import FocusInterface
from .stop_watch_interface import StopWatchInterface
from .camera_interface import CameraInterface
from .setting_interface import SettingInterface
from ..common.config import cfg
from ..common.icon import Icon
from ..common.signal_bus import signalBus
from ..common import resource


class MainWindow(FluentWindow):

    def __init__(self):
        super().__init__()
        self.initWindow()

        # TODO: create sub interface
        # self.homeInterface = HomeInterface(sel f)
        self.focusInterface = FocusInterface(self)
        self.stopWatchInterface = StopWatchInterface(self)
        self.settingInterface = SettingInterface(self)
        self.cameraInterface = CameraInterface(self)

        self.connectSignalToSlot()

        # add items to navigation interface
        self.initNavigation()

    def connectSignalToSlot(self):
        signalBus.micaEnableChanged.connect(self.setMicaEffectEnabled)

    def initNavigation(self):
        # self.navigationInterface.setAcrylicEnabled(True)

        # TODO: add navigation items
        # self.addSubInterface(self.homeInterface, FIF.HOME, self.tr('Home'))

        # add custom widget to bottom
        self.addSubInterface(self.cameraInterface, FIF.CAMERA, '摄像头')
        self.addSubInterface(self.focusInterface, FIF.RINGER, '专注时段')
        self.addSubInterface(self.stopWatchInterface, FIF.STOP_WATCH, '秒表')

        self.addSubInterface(
            self.settingInterface, FIF.SETTING, self.tr('Settings'), NavigationItemPosition.BOTTOM)

        self.splashScreen.finish()

    def initWindow(self):
        self.resize(960, 780)
        self.setMinimumWidth(400)
        self.setMinimumHeight(200)
        self.setWindowIcon(QIcon(':/app/images/logo.png'))
        self.setWindowTitle('Position Helper')

        self.setCustomBackgroundColor(QColor(240, 244, 249), QColor(32, 32, 32))
        self.setMicaEffectEnabled(cfg.get(cfg.micaEnabled))

        # create splash screen
        self.splashScreen = SplashScreen(self.windowIcon(), self)
        self.splashScreen.setIconSize(QSize(106, 106))
        self.splashScreen.raise_()

        desktop = QApplication.primaryScreen().availableGeometry()
        w, h = desktop.width(), desktop.height()
        self.move(w//2 - self.width()//2, h//2 - self.height()//2)
        self.show()
        QApplication.processEvents()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if hasattr(self, 'splashScreen'):
            self.splashScreen.resize(self.size())