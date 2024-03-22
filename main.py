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
    step = 0  # 为了记录走到第几步，记忆录中积累经验（也就是积累一些transition）之后再开始学习
    for episode in range(200):
        # initial observation
        observation = env.reset()

        while True:
            print("step%d" % step)
            # 每6步矫正方向
            if step%6==0:
                move_F()
                time.sleep(0.01)
            # RL choose action based on observation
            action = model.choose_action(observation)

            # RL take action and get next observation and reward
            _, _, observation_, reward, done = env.step(action)

            # !! restore transition
            model.store_transition(observation, action, reward, observation_)
            print("添加transition")

            # 超过200条transition之后每隔5步学习一次
            if (step > 200) and (step % 5 == 0):
                model.learn()
                print("训练")

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1
        if episode>=120 and episode%10==0:
            torch.save(model.eval_net.state_dict(), "mymodel\\"+str(episode)+"_model.pth")

    # end of game
    print("game over")
    torch.save(model.eval_net.state_dict(), "mymodel/last_model.pth")


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
    # 可以接着上次训练的权重继续训练
    # model.eval_net.load_state_dict(torch.load("mymodel\\last_model.pth"))

    if is_admin():
        ys = find_window(None, "原神")
        logging.error(ys)
        set_foreground_window(ys)
        run()
    # 主程序写在这里

    else:
        # 以管理员权限重新运行程序
        win32api.ShellExecute(None, "runas", sys.executable, __file__, None, 1)
