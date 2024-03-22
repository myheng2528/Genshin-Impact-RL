import argparse

parser = argparse.ArgumentParser()


parser.add_argument("--learning_rate",type=float,default=0.01,help="学习率")
parser.add_argument("--n_actions",type=int,default=9,help="动作空间")
parser.add_argument("--n_features",type=int,default=16,help="过度特征维数")
parser.add_argument("--reward_decay",type=float,default=0.9)
parser.add_argument("--e_greedy",type=float,default=1.0)
parser.add_argument("--replace_target_iter",type=int,default=300)
parser.add_argument("--memory_size",type=int,default=2000)
parser.add_argument("--e_greedy_increment",type=float,default=None,help="e_greedy增长率")
parser.add_argument("--capture_width",type=int,default=640,help="宽")
parser.add_argument("--capture_height",type=int,default=360,help="高")
parser.add_argument("--batch_size",type=int,default=8)
parser.add_argument("--shaonian",type=int,default=5.5,help="少年")
parser.add_argument("--chengnv",type=int,default=6,help="成女")
parser.add_argument("--shaonv",type=int,default=8,help="少女")

args = parser.parse_args()

# def get_args():
#
#     return parser.parse_args()