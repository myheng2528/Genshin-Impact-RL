import time
from typing import Optional, Union, List


import numpy as np
import torch
from util import *
from torchvision import transforms
from Binary_classification import cnn
class GenshinImpact():
    def __init__(self, goal_velocity=0):
        # self.action_space = spaces.Discrete(8)  # 普攻 重击 E技能 Q技能 （W A S D）四个方向的闪避
        # 4帧图像 每帧为宽640高为360的三通道图像（1080*1920）缩小三倍
        # self.observation_space = spaces.Box(low=0, high=255, shape=(4, 360, 640, 3), dtype=np.float32)
        self.reward_system = RewardSystem()
        self.stamina_model = torch.load("stamina.pkl")
        self.CD_model = torch.load("cd.pkl")
        self.transform_cd = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # 归一化
        ])
        self.transform_stamina = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # 归一化
        ])
        # self.seed()



    # 输入动作 执行完动作后返回自身的状态，环境的观测，以及动作奖励
    def step(self, action):

        # 执行动作
        time.sleep(0.02)
        self.act(action)
        # 获取四帧图像
        screen = []
        for i in range(4):
            screen.append(get_screen()[0])
        a, raw_image, screenshot = get_screen()
        screen.append(a)

        ####################################################################
        # 鼠标对准目标

        results = yolo(raw_image)

        cls = results[0].boxes.cls.tolist()
        conf = results[0].boxes.conf.tolist()
        # 如果检测到目标并且可信度大于0.3
        if len(cls) != 0 and conf[0] > 0.3:
            xyxy = results[0].boxes.xyxy.tolist()[0]
            l_x, l_y, r_x, r_y = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
            x_ = (l_x + r_x) / 2
            y_ = (l_y + r_y) / 2
            time.sleep(0.2)
            move_mouse_to_relative(x_, y_)

            # ctypes.windll.user32.mouse_event(MOUSEEVENTF_MOVE, 960-x_, 540-y_, 0, 0)
        #######################################################################

        # 四帧图像拼接起来得到观测 (4, 360, 640, 3)
        observation = np.stack([screen[0], screen[1], screen[2], screen[3], screen[4]], axis=0)
        # 将数组形状变为(C,D,H,W)
        observation = np.transpose(observation, (3, 0, 1, 2))

        # 根据观测的图像提取自身血量和怪物血量
        target_HP = get_HP_E(raw_image)
        self_HP = get_HP_S(raw_image)
        # 得到体力值和EQ技能CD
        E_CD, Q_CD = self.get_CD(screenshot)
        stamina = self.get_Stamina(screenshot)
        # 计算动作奖励
        reward = self.reward_system.get_reward(self_HP, target_HP, action, stamina, E_CD, Q_CD)
        done = self_HP < 50 or target_HP < 0

        return self_HP, target_HP, observation, reward, done


    def reset(self):
        self.reward_system.pre_self_HP,self.reward_system.pre_target_HP = 1000, 1000
        restart()
        # 抽取四帧
        screen = []
        for i in range(5):
            screen.append(get_screen()[0])
        observation = np.stack([screen[0], screen[1], screen[2], screen[3], screen[4]], axis=0)
        # cv2.imwrite("D:\\123.png", screen[0])

        # 将数组形状变为(C,D,H,W)
        transposed_array = np.transpose(observation, (3, 0, 1, 2))
        return transposed_array

    # 普攻，重击，短E，长E，Q技能，闪避
    def act(self, action):
        if action == 0:
            normal_attack()
        elif action == 1:
            whack()
        elif action == 2:
            short_E()
        elif action == 3:
            long_E()
        elif action == 4:
            Q_attack()
        elif action == 5:
            move_F()
        elif action == 6:
            miss_B()
        elif action == 7:
            miss_L()
        elif action == 8:
            miss_R()
    def get_CD(self, screenshot):
        E_img = screenshot.crop((1655, 953, 1730, 1028))
        Q_img = screenshot.crop((1760, 912, 1878, 1026))
        E_img = self.transform_cd(E_img)
        Q_img = self.transform_cd(Q_img)
        E_img = torch.unsqueeze(E_img, dim=0).cuda()
        Q_img = torch.unsqueeze(Q_img, dim=0).cuda()
        outputs_E = self.CD_model(E_img)[0]
        predicted = torch.max(outputs_E.data, 1)[1].data
        E_CD = predicted.item()
        outputs_Q = self.CD_model(Q_img)[0]
        predicted = torch.max(outputs_Q.data, 1)[1].data
        Q_CD = predicted.item()
        return E_CD, Q_CD


    def get_Stamina(self, screenshot):
        img = screenshot.crop((895, 520, 1300, 720))
        img = self.transform_stamina(img)
        img = torch.unsqueeze(img, dim=0).cuda()
        outputs = self.stamina_model(img)[0]
        predicted = torch.max(outputs.data, 1)[1].data
        stamina = predicted.item()
        return stamina



