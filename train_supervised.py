import sys
import os
import pickle
import numpy as np
from utils import dotdict

# 引用项目路径
sys.path.append(os.getcwd())
from connect6.GobangGame import GobangGame as Game
from connect6.pytorch.NNet import NNetWrapper as nn
# 关键：导入 NNet 模块里的 args，以便我们动态修改训练轮数
from connect6.pytorch.NNet import args as nnet_args

# === 1. 训练配置 ===
# 这里我们覆盖默认配置
TRAIN_CONFIG = {
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 50,      # 你的要求：只跑 50 轮
    'batch_size': 64,  # 显存小改 32
    'cuda': True,
    'num_channels': 64 # 必须与你 NNet.py 里写的一致！
}

# 数据文件路径 (gen_data.py 生成的那个)
DATA_FILE = "./temp/checkpoint_0.pth.tar"
OUTPUT_FILE = "best.pth.tar"

def train_supervised():
    print("🔄 正在初始化游戏与神经网络...")
    g = Game(19)
    nnet = nn(g)

    # === 强行覆盖训练参数 ===
    # 这一步很关键，确保 NNet 真的跑 50 轮，而不是默认的 10 轮
    for key, value in TRAIN_CONFIG.items():
        nnet_args[key] = value
    
    # 再次确认通道数是否匹配
    if nnet_args.num_channels != 64:
        print(f"⚠️ 警告：当前配置 num_channels={nnet_args.num_channels}，建议改为 64 以匹配之前的优化。")

    # === 加载数据 ===
    if not os.path.exists(DATA_FILE):
        print(f"❌ 错误：找不到数据文件 {DATA_FILE}")
        print("   请先运行 python gen_data.py 生成数据！")
        return

    print(f"📂 正在加载贪心算法生成的棋谱: {DATA_FILE} ...")
    with open(DATA_FILE, "rb") as f:
        trainExamples = pickle.load(f)
    
    print(f"✅ 加载成功！共 {len(trainExamples)} 个样本。")
    
    # === 开始训练 ===
    print(f"🚀 开始监督学习训练 (Target Epochs: {TRAIN_CONFIG['epochs']})...")
    print("   目标：让 Loss_pi 降到 1.0 以下，越低越好。")
    
    # 直接调用 NNet 的 train 方法
    # 这个方法内部会打印进度条
    nnet.train(trainExamples)
    
    # === 保存模型 ===
    print(f"💾 训练完成！正在保存模型到 temp/{OUTPUT_FILE} ...")
    nnet.save_checkpoint(folder='./temp/', filename=OUTPUT_FILE)
    print("✅ 全部完成！现在你可以运行 play_final.py 来验收成果了。")

if __name__ == "__main__":
    train_supervised()
