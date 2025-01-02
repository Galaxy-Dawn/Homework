import cv2
import pyttsx3
import threading
from PositionDetectionModel.Model import Mobile_LSTMModel
import torch
import numpy as np
import time
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image


# 加载字体
font_path = r"C:\Users\10557\PycharmProjects\PositionDetection\PositionDetectionModel\jianti.ttf"  # 替换为jianti.ttf的路径
font_size = 32
font = ImageFont.truetype(font_path, font_size)


# 标签映射
labels_mapping = {
    0: '趴下',
    1: '身体左倾',
    2: '身体右倾',
    3: '头部前倾',
    4: '头部左倾',
    5: '头部右倾',
    6: '左手撑头',
    7: '正坐',
    8: '右手撑头'
}

# 加载模型
model = Mobile_LSTMModel(num_classes=9)
model.load_state_dict(torch.load(r'C:\Users\10557\PycharmProjects\PositionDetection\PositionDetectionModel\mobile_lstm_LS_best_model.pth', map_location=torch.device('cpu')))
model.eval()  # 切换到评估模式

# 初始化姿势估计模型
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


# def process_frame(cap, frame_list, non_correct_posture_count):
#     if cap.isOpened():
#         success, image = cap.read()
#
#         # 创建img副本，用于姿势估计
#         img = image.copy()
#
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#         # 姿势估计处理
#         joint = pose.process(img)
#
#         predicted_label_name = ' '
#
#         joint_lst = []
#         if joint.pose_landmarks is not None:
#             for idx, lm in enumerate(joint.pose_landmarks.landmark):
#                 joint_lst.append([lm.x, lm.y, lm.z, lm.visibility])
#
#             joint_lst = joint_lst[:25]
#
#             joint_img = np.zeros((720, 1280), dtype=np.uint8)
#             connections = [[0, 1], [1, 2], [2, 3], [4, 5], [5, 6], [3, 7], [6, 8], [9, 10], [11, 12], [11, 13],
#                            [13, 15], [12, 14], [14, 16], [15, 17], [15, 21], [17, 19], [15, 19], [16, 22], [16, 20],
#                            [18, 20], [16, 18], [12, 24], [11, 23]]
#
#             for connection in connections:
#                 start_point = (int(joint_lst[connection[0]][0] * 1280), int(joint_lst[connection[0]][1] * 720))
#                 end_point = (int(joint_lst[connection[1]][0] * 1280), int(joint_lst[connection[1]][1] * 720))
#                 cv2.line(joint_img, start_point, end_point, 1, thickness=1)
#
#             joint_img = cv2.resize(joint_img, (720, 1280))
#             joint_img = np.tile(joint_img, (3, 1, 1))  # 复制为 [3, 720, 1280]
#
#         # 更新frame_list，确保只保存最新的3帧
#         if len(frame_list) == 3:
#             frame_list.pop(0)
#         frame_list.append(joint_lst)
#
#         # 进行坐姿检测
#         if len(frame_list) == 3:
#             frame_array = np.array(frame_list)
#             frame_tensor = torch.tensor(frame_array, dtype=torch.float32).unsqueeze(0)
#             frame_tensor = frame_tensor.view(1, 3, -1)
#             frame_tensor = frame_tensor.permute(0, 2, 1)
#
#             joint_img = np.expand_dims(joint_img, axis=0)
#             joint_img = np.transpose(joint_img, (0, 1, 2, 3))
#             joint_img_tensor = torch.tensor(joint_img, dtype=torch.float32)
#
#             with torch.no_grad():
#                 output = model(frame_tensor, joint_img_tensor)
#                 _, predicted_label = torch.max(output, 1)
#                 # 初始化状态监控变量
#                 if cap.isOpened():
#                     success, image = cap.read()
#
#                     # 创建img副本，用于姿势估计
#                     img = image.copy()
#                     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#                     # 姿势估计处理
#                     joint = pose.process(img)
#
#                     predicted_label_name = ' '
#
#                     joint_lst = []
#                     if joint.pose_landmarks is not None:
#                         for idx, lm in enumerate(joint.pose_landmarks.landmark):
#                             joint_lst.append([lm.x, lm.y, lm.z, lm.visibility])
#
#                         joint_lst = joint_lst[:25]
#
#                         joint_img = np.zeros((720, 1280), dtype=np.uint8)
#                         connections = [[0, 1], [1, 2], [2, 3], [4, 5], [5, 6], [3, 7], [6, 8], [9, 10], [11, 12],
#                                        [11, 13],
#                                        [13, 15], [12, 14], [14, 16], [15, 17], [15, 21], [17, 19], [15, 19], [16, 22],
#                                        [16, 20],
#                                        [18, 20], [16, 18], [12, 24], [11, 23]]
#
#                         for connection in connections:
#                             start_point = (
#                             int(joint_lst[connection[0]][0] * 1280), int(joint_lst[connection[0]][1] * 720))
#                             end_point = (
#                             int(joint_lst[connection[1]][0] * 1280), int(joint_lst[connection[1]][1] * 720))
#                             cv2.line(joint_img, start_point, end_point, 1, thickness=1)
#
#                         joint_img = cv2.resize(joint_img, (720, 1280))
#                         joint_img = np.tile(joint_img, (3, 1, 1))  # 复制为 [3, 720, 1280]
#
#                     # 更新frame_list，确保只保存最新的3帧
#                     if len(frame_list) == 3:
#                         frame_list.pop(0)
#                     frame_list.append(joint_lst)
#
#                     # 进行坐姿检测
#                     if len(frame_list) == 3:
#                         frame_array = np.array(frame_list)
#                         frame_tensor = torch.tensor(frame_array, dtype=torch.float32).unsqueeze(0)
#                         frame_tensor = frame_tensor.view(1, 3, -1)
#                         frame_tensor = frame_tensor.permute(0, 2, 1)
#
#                         joint_img = np.expand_dims(joint_img, axis=0)
#                         joint_img = np.transpose(joint_img, (0, 1, 2, 3))
#                         joint_img_tensor = torch.tensor(joint_img, dtype=torch.float32)
#
#                         with torch.no_grad():
#                             output = model(frame_tensor, joint_img_tensor)
#                             _, predicted_label = torch.max(output, 1)
#                             predicted_label_name = labels_mapping[predicted_label[0].item()]
#
#                     # 根据预测标签名称设置字体颜色
#                     if predicted_label_name == '正坐':
#                         font_color = (0, 255, 0)  # 绿色
#                         non_correct_posture_count = 0  # 重置计数
#                     else:
#                         font_color = (0, 0, 255)  # 红色
#                         non_correct_posture_count += 1  # 增加非正坐计数
#
#                         # 仅在连续三次检测到非正坐时进行播报
#                         if non_correct_posture_count >= 5:
#                             pass
#                     # 在图像上使用PIL绘制文本
#                     pil_image = Image.fromarray(image)
#                     draw = ImageDraw.Draw(pil_image)
#                     draw.text((10, 30), predicted_label_name, font=font, fill=font_color)
#
#                     # 将PIL图像转换回OpenCV格式
#                     image = np.array(pil_image)
#     return image, frame_list, non_correct_posture_count


def process_frame(cap, frame_list, non_correct_posture_count):
    if cap.isOpened():
        text = ""
        success, image = cap.read()
        # 创建img副本，用于姿势估计
        img = image.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 姿势估计处理
        joint = pose.process(img)
        joint_lst = []
        if joint.pose_landmarks is not None:
            for idx, lm in enumerate(joint.pose_landmarks.landmark):
                joint_lst.append([lm.x, lm.y, lm.z, lm.visibility])

            joint_lst = joint_lst[:25]

            joint_img = np.zeros((720, 1280), dtype=np.uint8)
            connections = [[0, 1], [1, 2], [2, 3], [4, 5], [5, 6], [3, 7], [6, 8], [9, 10], [11, 12], [11, 13],
                           [13, 15], [12, 14], [14, 16], [15, 17], [15, 21], [17, 19], [15, 19], [16, 22], [16, 20],
                           [18, 20], [16, 18], [12, 24], [11, 23]]

            for connection in connections:
                start_point = (int(joint_lst[connection[0]][0] * 1280), int(joint_lst[connection[0]][1] * 720))
                end_point = (int(joint_lst[connection[1]][0] * 1280), int(joint_lst[connection[1]][1] * 720))
                cv2.line(joint_img, start_point, end_point, 1, thickness=1)

            joint_img = cv2.resize(joint_img, (720, 1280))
            joint_img = np.tile(joint_img, (3, 1, 1))  # 复制为 [3, 720, 1280]

        # 更新frame_list，确保只保存最新的3帧
        if len(frame_list) == 3:
            frame_list.pop(0)
        frame_list.append(joint_lst)

        # 进行坐姿检测
        if len(frame_list) == 3:
            frame_array = np.array(frame_list)
            frame_tensor = torch.tensor(frame_array, dtype=torch.float32).unsqueeze(0)
            frame_tensor = frame_tensor.view(1, 3, -1)
            frame_tensor = frame_tensor.permute(0, 2, 1)

            joint_img = np.expand_dims(joint_img, axis=0)
            joint_img = np.transpose(joint_img, (0, 1, 2, 3))
            joint_img_tensor = torch.tensor(joint_img, dtype=torch.float32)

            with torch.no_grad():
                output = model(frame_tensor, joint_img_tensor)
                _, predicted_label = torch.max(output, 1)
                predicted_label_name = labels_mapping[predicted_label[0].item()]
                # 根据预测标签名称设置字体颜色
                if predicted_label_name == '正坐':
                    font_color = (0, 255, 0)  # 绿色
                    non_correct_posture_count = 0  # 重置计数
                else:
                    font_color = (0, 0, 255)  # 红色
                    non_correct_posture_count += 1  # 增加非正坐计数

                    if non_correct_posture_count >= 25:
                        text = "检测到五秒非正坐，请调整坐姿"
                        non_correct_posture_count = 0
                # 在图像上使用PIL绘制文本
                pil_image = Image.fromarray(image)
                draw = ImageDraw.Draw(pil_image)
                draw.text((10, 30), predicted_label_name, font=font, fill=font_color)

                # 将PIL图像转换回OpenCV格式
                image = np.array(pil_image)
    return image, frame_list, non_correct_posture_count, text





