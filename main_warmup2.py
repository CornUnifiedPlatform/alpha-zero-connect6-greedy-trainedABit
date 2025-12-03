import logging
import coloredlogs
import pickle
import os
import sys
import numpy as np
import random
import time
import traceback
import builtins # 用于修改系统 input

# === 1. 黑魔法：自动绕过 Coach.py 的 [y/n] 询问 ===
# 无论谁调用 input()，永远自动返回 'y'，防止云端卡死
builtins.input = lambda prompt="": 'y'

from multiprocessing import Pool, cpu_count
from Coach import Coach
from connect6.GobangGame import GobangGame as Game
from connect6.pytorch.NNet import NNetWrapper as nn
from utils import dotdict

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')

# === 2. 全局配置 ===
ARGS = dotdict({
    'numIters': 1000,
    'numEps': 30,              # 每代 30 局 (极速迭代)
    'tempThreshold': 15,
    'updateThreshold': 0.50,   # 门槛 0.5 (不输就更新)
    'maxlenOfQueue': 50000,    # 内存保护
    'numMCTSSims': 50,         # 搜索 50 次 (平衡)
    'arenaCompare': 14,        # 竞技场 14 局
    'cpuct': 1,
    'checkpoint': './temp/',
    'load_model': False,       # 默认先由逻辑判断
    'load_folder_file': ('./temp/', 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 5,
})

# 数据生成配置
GEN_GAMES = 500
GEN_FILE = os.path.join(ARGS.checkpoint, "warmup_greedy_500.pth.tar")
USE_CORES = 8 # 本地/云端并行核心数

# === 3. 贪心算法逻辑 (V3.0 强力版) ===
class SmartGreedyPlayer:
    def __init__(self, game):
        self.game = game
        self.n = game.n
        self.p_scores = np.zeros((self.n, self.n))
        c = self.n // 2
        for r in range(self.n):
            for col in range(self.n):
                self.p_scores[r][col] = (c - max(abs(r-c), abs(col-c))) * 0.1

    def play(self, board):
        rows, cols = np.where(board != 0)
        if len(rows) == 0: return (self.n * self.n) // 2
        
        min_r, max_r = max(0, min(rows)-3), min(self.n, max(rows)+4)
        min_c, max_c = max(0, min(cols)-3), min(self.n, max(cols)+4)
        
        best_a = -1
        best_s = -float('inf')
        candidates = []
        
        for r in range(min_r, max_r):
            for c in range(min_c, max_c):
                if board[r][c] != 0: continue
                a = r * self.n + c
                candidates.append(a)
                
                # A. 绝杀
                next_b, _ = self.game.getNextState(board, 1, a)
                if self.game.getGameEnded(next_b, 1) == 1: return a 
                
                # B. 必救
                score = 0
                next_b_opp, _ = self.game.getNextState(board, -1, a)
                if self.game.getGameEnded(next_b_opp, 1) == -1:
                    score = 500000 
                
                # C. 评分 (进攻1.0 防守2.0 -> 稳健)
                score += self.smart_eval(board, r, c, 1) * 1.0
                score += self.smart_eval(board, r, c, -1) * 2.0
                score += self.p_scores[r][c] + np.random.random() * 0.5 

                if score > best_s: best_s = score; best_a = a
        
        # 5% 概率手滑
        if best_s < 10000 and random.random() < 0.05 and len(candidates) > 0:
            return np.random.choice(candidates)
        return best_a if best_a != -1 else np.random.choice(candidates)

    def smart_eval(self, board, r, c, color):
        score = 0
        for dr, dc in [(0,1), (1,0), (1,1), (1,-1)]:
            fs, bs = 0, 0
            for k in range(1, 5): # 前探4格
                nr, nc = r + k*dr, c + k*dc
                if 0<=nr<self.n and 0<=nc<self.n:
                    if board[nr][nc] == color: fs += 1
                    elif board[nr][nc] != 0: break
                else: break
            for k in range(1, 5): # 后探4格
                nr, nc = r - k*dr, c - k*dc
                if 0<=nr<self.n and 0<=nc<self.n:
                    if board[nr][nc] == color: bs += 1
                    elif board[nr][nc] != 0: break
                else: break
            
            total = 1 + fs + bs
            if total >= 6: score += 100000
            elif total == 5: score += 5000
            elif total == 4: score += 500
            elif total == 3: score += 50
            elif total == 2: score += 10
        return score

# === 4. 并行生成器 ===
def worker_sim(seed):
    try:
        np.random.seed(int(time.time()*1000)%100000 + seed)
        g = Game(19)
        p = SmartGreedyPlayer(g)
        b, cur = g.getInitBoard(), 1
        ep, step = [], 0
        while True:
            step += 1
            can = g.getCanonicalForm(b, cur)
            a = p.play(can)
            pi = np.zeros(g.getActionSize()); pi[a] = 1
            for sb, sp in g.getSymmetries(can, pi): ep.append([sb, cur, sp])
            b, cur = g.getNextState(b, cur, a)
            r = g.getGameEnded(b, 1)
            if r!=0 or step>150:
                if r!=0 and r!=1e-4:
                    res = []
                    for d in ep: res.append([d[0], d[2], 1 if r==d[1] else -1])
                    return (True, res)
                return (False, [])
    except: return (False, [])

def generate_data_if_needed():
    if not os.path.exists(ARGS.checkpoint): os.makedirs(ARGS.checkpoint)
    
    if os.path.exists(GEN_FILE):
        print(f"✅ 检测到现成的数据: {GEN_FILE}")
        return True

    print(f"🚀 未找到数据，开始生成 {GEN_GAMES} 局热启动样本...")
    pool = Pool(min(USE_CORES, cpu_count()))
    examples = []
    completed = 0
    
    # 提交任务
    results = [pool.apply_async(worker_sim, (i,)) for i in range(int(GEN_GAMES*1.5))]
    
    for res in results:
        if completed >= GEN_GAMES: break
        try:
            ok, data = res.get(timeout=300)
            if ok:
                examples.extend(data)
                completed += 1
                print(f"生成进度: {completed}/{GEN_GAMES}", end="\r")
        except: pass
    
    pool.terminate()
    print(f"\n💾 生成完成！共 {len(examples)} 样本")
    with open(GEN_FILE, "wb+") as f: pickle.dump(examples, f)
    return True

# === 5. 主程序逻辑 ===
def main():
    log.info(f"Initializing Game (19x19)...")
    g = Game(19)
    nnet = nn(g)

    # 1. 检查是否存在模型 (断点续传)
    model_path = os.path.join(ARGS.checkpoint, "best.pth.tar")
    if os.path.exists(model_path):
        log.info(f"🔄 发现现有模型 {model_path}，正在加载继续训练...")
        ARGS.load_model = True
        # 即使这里找不到 .examples 文件，第一行的 monkey patch 会自动按 'y' 跳过
        # 所以不需要担心卡死
    else:
        log.info("🆕 未发现模型，准备从零开始 (热启动模式)...")
        ARGS.load_model = False
        
        # 2. 如果是从零开始，确保有数据
        generate_data_if_needed()

    # 3. 启动 Coach
    c = Coach(g, nnet, ARGS)

    # 4. 如果是热启动 (load_model=False)，手动注入数据
    if not ARGS.load_model:
        if os.path.exists(GEN_FILE):
            log.info("🔥 正在注入热启动数据...")
            with open(GEN_FILE, "rb") as f:
                trainExamples = pickle.load(f)
            c.trainExamplesHistory.append(trainExamples)
            log.info(f"🔥 注入成功！AI 将基于 {len(trainExamples)} 个样本开始 Iter 1。")
        else:
            log.error("❌ 严重错误：数据生成失败或文件丢失！")
            return

    # 5. 开始/继续 训练
    c.learn()

if __name__ == "__main__":
    main()
