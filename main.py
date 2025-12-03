import logging

import coloredlogs

from Coach import Coach
# 引用你修改好的六子棋游戏逻辑
from connect6.GobangGame import GobangGame as Game
# 引用你刚才复制过去的通用神经网络
from connect6.pytorch.NNet import NNetWrapper as nn
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 1000,
    'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': True,
    'load_folder_file': ('./temp','best.pth.tar'),
    'numItersForTrainExamplesHistory': 10,

})


def main():
    log.info('Loading %s...', Game.__name__)
    g = Game(19)

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info("Loading checkpoint (Weights Only)...")
        # 只要这行还在，模型智商就在
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
        
        # ！！！把下面这两行注释掉！！！
        # log.info("Loading 'trainExamples' from file...")
        # c.loadTrainExamples() 
        
        print("⚠️ 已跳过历史棋谱加载，将基于现有模型智商开启新的训练。")
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()
