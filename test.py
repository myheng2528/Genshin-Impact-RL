from Genshin_Impact import GenshinImpact
from dqn_model import DeepQNetwork
from parsers import args
from util import *
import logging
import sys
import torch
from Binary_classification import cnn


def excepthook_handler(exc_type, exc_value, exc_traceback):
    """
    处理未捕获的异常并将其记录到日志中
    """
    logging.error("未捕获的异常:", exc_info=(exc_type, exc_value, exc_traceback))


# 配置日志记录器
logging.basicConfig(filename='example.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# 设置异常处理钩子
sys.excepthook = excepthook_handler


def run():
        observation = env.reset()
        step=0
        while True:
            print("step%d" % step)
            # 每10步矫正方向
            if step%10==0:
                move_F()
                time.sleep(0.01)
            # RL choose action based on observation
            action = model.choose_action(observation)

            # RL take action and get next observation and reward
            _, _, observation_, reward, done = env.step(action)

            # !! restore transition

            # 超过200条transition之后每隔5步学习一次

            # swap observation
            observation = observation_
            # break while loop when end of this episode
            if done:
                break
            step += 1

    # end of game
        print("game over")


if __name__ == "__main__":
    # 配置日志记录器

    env = GenshinImpact()
    model = DeepQNetwork(
        n_actions=args.n_actions,
        n_features=args.n_features,
        learning_rate=args.learning_rate,
        reward_decay=args.reward_decay,
        e_greedy=args.e_greedy,
        replace_target_iter=args.replace_target_iter,
        memory_size=args.memory_size,
        batch_size=args.batch_size,
        e_greedy_increment=args.e_greedy_increment
    )

    model.eval_net.load_state_dict(torch.load("mymodel\\190_model.pth"))
    model.eval_net.eval()
    if is_admin():
        ys = find_window(None, "原神")
        logging.error(ys)
        set_foreground_window(ys)
        run()
    # 主程序写在这里

    else:
        # 以管理员权限重新运行程序
        win32api.ShellExecute(None, "runas", sys.executable, __file__, None, 1)
