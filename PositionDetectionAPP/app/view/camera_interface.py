# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'LoginWindowOVvuAC.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import cv2
from PositionDetectionModel.run import process_frame

frame_list = []
non_correct_posture_count = 0

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(963, 595)
        Form.setMinimumSize(QSize(700, 500))

        # 创建并设置统一的字体对象
        font = QFont()
        font.setBold(False)  # 设置不加粗
        font.setWeight(50)
        font.setPointSize(14)  # 设置字号为 14，可以根据需要调整

        # Setup label to display the camera feed (视频显示区域)
        self.label = QLabel(Form)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(20, 80, 531, 401))  # 设置位置和大小
        self.label.setFont(font)  # 设置字体
        self.label.setFrameShape(QFrame.WinPanel)
        self.label.setFrameShadow(QFrame.Sunken)
        self.label.setLineWidth(0)
        self.label.setMidLineWidth(0)
        self.label.setAlignment(Qt.AlignLeading | Qt.AlignLeft | Qt.AlignVCenter)
        self.label.setWordWrap(False)

        # 设置 label 为圆角
        self.label.setStyleSheet("QLabel { border-radius: 15px; border: 2px solid #aaa; background: transparent; }")

        # Setup button 3 (打开摄像头按钮)
        self.pushButton_3 = QPushButton(Form)
        self.pushButton_3.setObjectName(u"pushButton_3")
        self.pushButton_3.setGeometry(QRect(140, 500, 150, 46))
        self.pushButton_3.setFont(font)  # 设置字体
        self.pushButton_3.setStyleSheet("QPushButton { "
                                        "border-radius: 25px; "
                                        "border: 2px solid #aaa; "
                                        "background-color: #4CAF50; "
                                        "color: white; }")  # 设置按钮圆角和颜色
        self.pushButton_3.setText("打开摄像头")  # 设置按钮初始文本
        self.pushButton_3.clicked.connect(self.start_camera)  # 按钮点击事件

        # Setup button 4 (关闭摄像头按钮)
        self.pushButton_4 = QPushButton(Form)
        self.pushButton_4.setObjectName(u"pushButton_4")
        self.pushButton_4.setGeometry(QRect(302, 500, 150, 46))
        self.pushButton_4.setFont(font)  # 设置字体
        self.pushButton_4.setStyleSheet("QPushButton { "
                                        "border-radius: 25px; "
                                        "border: 2px solid #aaa; "
                                        "background-color: #f44336; "
                                        "color: white; }")  # 设置按钮圆角和颜色
        self.pushButton_4.setText("关闭摄像头")  # 设置按钮初始文本
        self.pushButton_4.clicked.connect(self.stop_camera)  # 按钮点击事件

        # Setup label 2
        self.label_2 = QLabel(Form)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(30, 30, 511, 24))
        self.label_2.setScaledContents(True)
        self.label_2.setAlignment(Qt.AlignCenter)
        self.label_2.setOpenExternalLinks(False)
        self.label_2.setFont(font)  # 设置字体

        # Setup label 3
        self.label_3 = QLabel(Form)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(480, 30, 511, 24))
        self.label_3.setScaledContents(True)
        self.label_3.setAlignment(Qt.AlignCenter)
        self.label_3.setOpenExternalLinks(False)
        self.label_3.setFont(font)  # 设置字体

        # Setup label 4 (确保label_4有圆角外框)
        self.label_4 = QLabel(Form)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setGeometry(QRect(590, 80, 300, 401))  # 设置位置和大小
        self.label_4.setStyleSheet("QLabel { border-radius: 15px; border: 2px solid #aaa; background: transparent; }")

        # 定时器，定时读取视频帧
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)  # 每次定时器超时时调用 update_frame 函数

        self.cap = None  # 摄像头对象
        self.running = False  # 摄像头是否正在运行

        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)

    def start_camera(self):
        """打开摄像头，启动视频流"""
        if not self.running:
            self.cap = cv2.VideoCapture(0)  # 打开默认摄像头
            if not self.cap.isOpened():
                print("无法打开摄像头")
                return

            # 设置摄像头分辨率
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            self.running = True
            self.timer.start(50)  # 启动定时器，每50毫秒读取一帧

    def update_frame(self):
        """获取并显示摄像头的最新帧"""
        global frame_list
        global non_correct_posture_count
        frame, frame_list1, non_correct_posture_count1, text = process_frame(self.cap, frame_list, non_correct_posture_count)
        frame_list = frame_list1
        non_correct_posture_count = non_correct_posture_count1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(q_image))

        current_text = self.label_4.text()
        # Append the new text to the existing text
        updated_text = current_text + "\n" + text
        # Set the updated text back to label_4
        self.label_4.setText(updated_text)

    def stop_camera(self):
        """关闭摄像头，停止视频流并恢复透明背景"""
        if self.running:
            self.running = False
            self.timer.stop()  # 停止定时器

            if self.cap is not None:
                self.cap.release()  # 释放摄像头资源
                self.cap = None  # 重置摄像头对象

            self.label.clear()  # 清除显示的图像
            self.label.setStyleSheet("QLabel { border-radius: 15px; border: 2px solid #aaa; background: transparent; }")
            self.label_4.setStyleSheet("QLabel { border-radius: 15px; border: 2px solid #aaa; background: transparent; }")

    def closeEvent(self, event):
        # 关闭窗口时停止视频流
        self.stop_camera()
        event.accept()

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.label.setText("")  # 清除 label 上的文本
        self.pushButton_3.setText(QCoreApplication.translate("Form", u"打开摄像头", None))
        self.pushButton_4.setText(QCoreApplication.translate("Form", u"关闭摄像头", None))
        self.label_2.setText(QCoreApplication.translate("Form", u"显示界面", None))
        self.label_3.setText(QCoreApplication.translate("Form", u"检测结果", None))


class CameraInterface(Ui_Form, QWidget):

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setupUi(self)