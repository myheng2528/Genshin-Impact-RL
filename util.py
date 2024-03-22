import cv2
import numpy as np
import pyautogui
import torch
import win32con
import win32api
from win32api import GetAsyncKeyState, GetCursorPos, Sleep
from win32con import VK_F12, MOUSEEVENTF_MOVE
import time
import ctypes
import os
import sys
from parsers import args
from ultralytics import YOLO

# 定义POINT结构体
class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long),
                ("y", ctypes.c_long)]

# 获取当前鼠标位置
def get_cursor_pos():
    cursor_pos = POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(cursor_pos))
    return cursor_pos.x, cursor_pos.y


yolo = YOLO("best.pt")
def get_screen():
    capture_width, capture_height = args.capture_width, args.capture_height
    screen_width, screen_height = pyautogui.size()
    # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    # out = cv2.VideoWriter('screen_capture.avi', fourcc, 20.0, (capture_width, capture_height))
    screenshot = pyautogui.screenshot()
    raw_image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGR2RGB)
    screenshot_ = screenshot.resize((capture_width, capture_height))
    # screenshot.show()
    # (H,W,C)
    # x,y = get_cursor_pos()
    #print("x:"+str(x))
    #print("y:"+str(y))
    open_cv_image = cv2.cvtColor(np.array(screenshot_), cv2.COLOR_BGR2RGB)
    # out.write(open_cv_image)

    # yolo = YOLO("best.pt")
    # results = yolo(raw_image)
    #
    #
    # cls = results[0].boxes.cls.tolist()
    # conf = results[0].boxes.conf.tolist()
    # # 如果检测到目标并且可信度大于0.3
    # if len(cls)!=0 and conf[0] > 0.3:
    #     xyxy = results[0].boxes.xyxy.tolist()[0]
    #     l_x, l_y, r_x, r_y = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
    #     x_ = (l_x+r_x)/2
    #     y_ = (l_y+r_y)/2
    #     move_mouse_to(x_,y_)


    return open_cv_image, raw_image, screenshot



class RewardSystem:
    def __init__(self):
        self.stamina = 224  # 耐力条，难以从图像读取，我很不道德的把判定放在了这里，反正通过动作执行能计算。你说不准？，那你要不教一下我那个弧线的当中每隔100有一个线的耐力条怎么读
        self.E_CD = 0  # 哈哈哈哈我把技能冷却也放在这个类里了，更不道德了
        self.steps = 0  # 计步，一段时间之内都没扣血，就给自己奖励一下
        self.flag = 0  # 闪避也有CD，期间不能在冲刺也不回体力
        self.pre_self_HP = 1000
        self.pre_target_HP = 1000
        self.reward_history = list()  # reward 的积累过程

    # 获取奖励
    def get_reward(self, self_HP, target_HP, action, stamina_flag, E_CD_flag, Q_CD_flag):
        reward = 0
        # if target_HP == 0:  # 怪物死亡获得大量奖励
        #     reward = 20
        #     return reward
        # elif self_HP == 0:  # 角色死亡获得惩罚
        #     reward = -20
        #     return reward
        if self_HP < self.pre_self_HP:  # 自身掉血惩罚  保持血量奖励
            reward = reward - 2 + (self_HP-self.pre_self_HP)/25
            self.steps = 0
        else:
            reward = reward + 2 + (self_HP-self.pre_self_HP)/50
            self.steps += 1
        if target_HP < self.pre_target_HP:  # 怪物掉血奖励  怪物回血惩罚
            reward = reward + 2 + (self.pre_target_HP-target_HP)/50
        else:
            reward = reward - 2 + (self.pre_target_HP-target_HP)/50
        if self.steps == 30:  # 30步不掉血获得大量奖励
            reward += 20
            self.steps = 0
        # if action == 1 or 4 <= action <= 7:  # 对消耗体力的动作给予惩罚
        #     reward -= 1
        if action == 2 or action == 3:  # 对使用技能的动作给予奖励
            reward += 5
        # 体力机制的奖励
        # 体力值为空时进行闪避或重击，给予惩罚
        if (action == 1 or action == 5) and stamina_flag==0:
            reward-=20
        if (action == 2 or action == 3) and E_CD_flag == 0:
            reward-=1
        elif (action == 2 or action == 3) and E_CD_flag == 1:
            reward+=5
        if action == 4 and Q_CD_flag == 0:
            reward-=1
        elif action == 4 and Q_CD_flag == 1:
            reward+=8

        self.pre_self_HP, self.pre_target_HP = self_HP, target_HP

        self.reward_history.append(reward)
        return reward


def roi(image, x, x_w, y, y_h):
    return image[y:y_h, x:x_w]


def get_HP_E(img):
    img_roi = roi(img, x=723, x_w=1195, y=87, y_h=88)
    # cv2.imshow("image", img_roi)  # 显示图片，后面会讲解
    # cv2.waitKey(0)  # 等待按键
    b, g, r = cv2.split(img_roi)
    #   retval,img_th=cv2.threshold(b, 100, 255, cv2.THRESH_TOZERO_INV)
    retval, img_th = cv2.threshold(r, 230, 255, cv2.THRESH_TOZERO)
    target_img = img_th[0]
    if 0 in target_img:
        Target_HP = np.argmax(target_img < 230)
    else:
        Target_HP = len(target_img)
    # 血量固定缩放为1000
    return (Target_HP * 1000) / len(target_img)


def get_HP_S(img):
    img_roi = roi(img, x=812, x_w=1108, y=1010, y_h=1011)
    # cv2.imshow("image", img_roi)  # 显示图片，后面会讲解
    # cv2.waitKey(0)  # 等待按键
    b, g, r = cv2.split(img_roi)
    retval, img_th = cv2.threshold(r, 100, 255, cv2.THRESH_TOZERO)
    target_img = img_th[0]
    if 0 in target_img:
        Self_HP = np.argmax(target_img < 90)  # 返回第一个小于90的索引
    else:
        Self_HP = len(target_img)
    # 血量固定缩放为1000
    return (Self_HP * 1000) / len(target_img)


key_map = {
    "0": 48, "1": 49, "2": 50, "3": 51, "4": 52, "5": 53, "6": 54, "7": 55, "8": 56, "9": 57,
    "A": 65, "B": 66, "C": 67, "D": 68, "E": 69, "F": 70, "G": 71, "H": 72, "I": 73, "J": 74,
    "K": 75, "L": 76, "M": 77, "N": 78, "O": 79, "P": 80, "Q": 81, "R": 82, "S": 83, "T": 84,
    "U": 85, "V": 86, "W": 87, "X": 88, "Y": 89, "Z": 90, " ": 32, "ESC": 27
}
MapVirtualKey = ctypes.windll.user32.MapVirtualKeyA


def key_down(key):
    key = key.upper()
    vk_code = key_map[key]
    win32api.keybd_event(vk_code, MapVirtualKey(vk_code, 0), 0, 0)


def key_up(key):
    key = key.upper()
    vk_code = key_map[key]
    win32api.keybd_event(vk_code, MapVirtualKey(vk_code, 0), win32con.KEYEVENTF_KEYUP, 0)


def long_E():
    key_down("E")
    time.sleep(0.4)
    key_up("E")


def short_E():
    key_down("E")
    time.sleep(0.02)
    key_up("E")


def Q_attack():
    key_down("Q")
    time.sleep(0.02)
    key_up("Q")


# 普通攻击
def normal_attack():
    # key_down("E")
    # time.sleep(0.02)
    # key_up("E")
    left_click()
    time.sleep(0.02)
    left_click()
    time.sleep(0.02)
    left_click()
    time.sleep(0.02)
    left_click()


# 重击
def whack():
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    time.sleep(0.3)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


def charged_attack():
    key_down("Q")
    time.sleep(0.02)
    key_up("Q")
    left_click()


def rapid_move_L():
    key_down("A")
    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
    time.sleep(0.02)
    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
    time.sleep(0.05)
    key_up("A")
    time.sleep(0.05)


def rapid_move_R():
    key_down("D")
    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
    time.sleep(0.02)
    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
    time.sleep(0.05)
    key_up("D")
    time.sleep(0.05)


def rapid_move_B():
    key_down("S")
    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
    time.sleep(0.02)
    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
    time.sleep(0.05)
    key_up("S")
    time.sleep(0.05)


def rapid_move_F():
    key_down("W")
    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
    time.sleep(0.02)
    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
    time.sleep(0.05)
    key_up("W")
    time.sleep(0.05)


def move_L():
    key_down("A")
    time.sleep(0.1)
    key_up("A")


def move_R():
    key_down("D")
    time.sleep(0.1)
    key_up("D")


def move_B():
    key_down("S")
    time.sleep(0.1)
    key_up("s")


def move_F():
    key_down("W")
    time.sleep(0.02)
    key_up("W")


def miss_L():
    key_down("S")
    time.sleep(0.05)
    right_click()
    time.sleep(0.1)
    key_up("S")


def miss_R():
    key_down("D")
    time.sleep(0.05)
    right_click()
    time.sleep(0.1)
    key_up("D")

def miss_F():
    key_down("W")
    time.sleep(0.05)
    right_click()
    time.sleep(0.1)
    key_up("W")
def miss_B():
    key_down("S")
    time.sleep(0.05)
    right_click()
    time.sleep(0.1)
    key_up("S")

def miss():
    key_up("A")
    key_up("D")
    key_up("S")
    key_up("W")
    # 随机闪避方向
    direction = np.random.randint(2)

    if direction == 0:
        miss_L()
    elif direction == 1:
        miss_R()






# 定义Windows API函数签名
# 参考：https://docs.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-mouse_event


def mouse_event(dwFlags, dx, dy, dwData, dwExtraInfo):
    ctypes.windll.user32.mouse_event(dwFlags, dx, dy, dwData, dwExtraInfo)


# 定义鼠标事件标志
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
# 获取屏幕尺寸
SCREEN_WIDTH = ctypes.windll.user32.GetSystemMetrics(0)
SCREEN_HEIGHT = ctypes.windll.user32.GetSystemMetrics(1)


# 移动鼠标到指定位置


def move_mouse_to(x, y):
    # 计算绝对坐标
    absolute_x = int(x * 65535 / SCREEN_WIDTH)
    absolute_y = int(y * 65535 / SCREEN_HEIGHT)
    # 发送鼠标移动事件
    mouse_event(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, absolute_x, absolute_y, 0, 0)
def move_mouse_to_relative(x, y):
    # 计算绝对坐标
    absolute_x = int(x * 65535 / SCREEN_WIDTH)
    absolute_y = int(y * 65535 / SCREEN_HEIGHT)
    target_x = int(960 * 65535 / SCREEN_WIDTH)
    target_y = int(540 * 65535 / SCREEN_HEIGHT)
    # 发送鼠标移动事件
    mouse_event(MOUSEEVENTF_MOVE, int((absolute_x-target_x)*0.02), int((absolute_y-target_y)*0.02), 0, 0)



def left_click():
    mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    time.sleep(0.02)  # 为了稳定起见，可以添加短暂延迟
    mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


def right_click():
    mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
    time.sleep(0.1)  # 为了稳定起见，可以添加短暂延迟
    mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)


# 深渊重开
def restart():
    time.sleep(0.1)
    key_down("ESC")
    time.sleep(0.02)
    key_up("ESC")
    # (710, 764)重新开始
    time.sleep(0.1)
    move_mouse_to(710, 764)
    time.sleep(0.8)
    left_click()
    time.sleep(6)
    move_mouse_to(940, 700)
    left_click()
    time.sleep(1)
    # (940，994)确认
    move_mouse_to(940, 994)
    left_click()
    time.sleep(0.1)
    key_down("w")
    time.sleep(args.shaonv)
    key_up("w")
    time.sleep(0.1)
    key_down("F")
    time.sleep(0.1)
    key_up("F")
    # 等待5秒
    time.sleep(5)




# 获取窗口句柄，从而可以切换窗口
FindWindow = ctypes.windll.user32.FindWindowW
# 设置焦点到指定窗口
SetForegroundWindow = ctypes.windll.user32.SetForegroundWindow


# 查找窗口句柄
def find_window(class_name, window_name):
    return FindWindow(class_name, window_name)


# 设置窗口焦点
def set_foreground_window(hwnd):
    SetForegroundWindow(hwnd)



# 判断是否具有管理员权限，并获取管理员权限
def is_admin():
    # 由于win32api中没有IsUserAnAdmin函数,所以用了这种方法
    try:
        # 在c:\windows目录下新建一个文件test01.txt
        testfile=os.path.join(os.getenv("windir"),"test01.txt")
        open(testfile,"w").close()
    except OSError: # 不成功
        return False
    else: # 成功
        os.remove(testfile) # 删除文件
        return True




if __name__ == '__main__':
    # screen = []
    # for i in range(5):
    #     screen.append(get_screen()[0])
    # observation = np.stack([screen[0], screen[1], screen[2], screen[3], screen[4]], axis=0)
    # transposed_array = np.transpose(observation, (3, 0, 1, 2))
    dect = torch.load("mymodel\\1_last_model.pth")
    a=1


    #get_screen()
    # if is_admin():
    #     ys = find_window(None, "原神")
    #     set_foreground_window(ys)
    #     time.sleep(2)
    #     mouse_event(MOUSEEVENTF_MOVE, 200, 200, 0, 0)
    #
    #
    # # 主程序写在这里
    # else:
    #     # 以管理员权限重新运行程序
    #     win32api.ShellExecute(None, "runas", sys.executable, __file__, None, 1)

