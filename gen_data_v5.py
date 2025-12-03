import pickle
import numpy as np
import os
import sys
import random
import time
from multiprocessing import Pool, cpu_count

sys.path.append(os.getcwd())
try:
    from connect6.GobangGame import GobangGame as Game
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from connect6.GobangGame import GobangGame as Game

# === 配置 ===
TOTAL_GAMES = 500  # 总目标局数
OUTPUT_FILE = "checkpoint_0.pth.tar"
FOLDER = "./temp/"

class DrunkenPlayer:
    def __init__(self, game):
        self.game = game
        self.n = game.n
        # 预计算位置分
        self.p_scores = np.zeros((self.n, self.n))
        c = self.n // 2
        for r in range(self.n):
            for col in range(self.n):
                self.p_scores[r][col] = c - max(abs(r-c), abs(col-c))

    def play(self, board):
        # 恢复 V3.0 的逻辑：带 Masking + 显式手滑
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
                
                # A. 绝杀 (Win)
                next_b, _ = self.game.getNextState(board, 1, a)
                if self.game.getGameEnded(next_b, 1) == 1: return a 
                
                # B. 必救 (Block) - 稍微降低权重，给进攻留机会
                score = 0
                next_b_opp, _ = self.game.getNextState(board, -1, a)
                if self.game.getGameEnded(next_b_opp, 1) == -1:
                    score = 20000 
                
                # C. 评分
                score += self.eval_lines(board, r, c, 1)
                score += self.eval_lines(board, r, c, -1) * 0.9
                score += self.p_scores[r][c] * 0.5
                score += np.random.normal(0, 5) # 噪声

                if score > best_s:
                    best_s = score
                    best_a = a
        
        # 关键：20% 概率手滑 (制造非平局)
        # 除非有绝杀或必救 (分数特别高)，否则允许手滑
        if best_s < 10000 and random.random() < 0.2 and len(candidates) > 0:
            return np.random.choice(candidates)
            
        return best_a if best_a != -1 else np.random.choice(candidates)

    def eval_lines(self, board, r, c, color):
        score = 0
        dirs = [(0,1), (1,0), (1,1), (1,-1)]
        for dr, dc in dirs:
            cnt = 1
            k=1
            while 0<=r+k*dr<self.n and 0<=c+k*dc<self.n and board[r+k*dr][c+k*dc]==color: cnt+=1; k+=1
            k=1
            while 0<=r-k*dr<self.n and 0<=c-k*dc<self.n and board[r-k*dr][c-k*dc]==color: cnt+=1; k+=1
            
            if cnt >= 6: score += 100000
            elif cnt == 5: score += 8000
            elif cnt == 4: score += 500
            elif cnt == 3: score += 50
        return score

def worker_play_game(game_id):
    """ 单个进程执行的函数：跑一局游戏，返回数据 """
    # 每个进程需要独立初始化 Game，避免内存共享冲突
    game = Game(19)
    player = DrunkenPlayer(game)
    
    board = game.getInitBoard()
    curPlayer = 1
    ep_data = []
    step = 0
    
    while True:
        step += 1
        canonical = game.getCanonicalForm(board, curPlayer)
        action = player.play(canonical)
        
        pi = np.zeros(game.getActionSize())
        pi[action] = 1
        sym = game.getSymmetries(canonical, pi)
        for b, p in sym: ep_data.append([b, curPlayer, p])
        
        board, curPlayer = game.getNextState(board, curPlayer, action)
        r = game.getGameEnded(board, 1)
        
        # 限制步数，或者分出胜负
        if r != 0 or step > 150:
            if r != 0 and r != 1e-4:
                result_data = []
                for d in ep_data:
                    v = 1 if r == d[1] else -1
                    result_data.append([d[0], d[2], v])
                return (True, result_data, step)
            else:
                return (False, [], step) # 平局，废弃

def main():
    if not os.path.exists(FOLDER): os.makedirs(FOLDER)
    
    # 自动检测核心数
    cores = max(1, cpu_count() - 2) # 留2个核给系统，其他的全跑满
    print(f"🚀 [多核极速版] 启用 {cores} 个核心并行生成 {TOTAL_GAMES} 局数据...")
    
    all_examples = []
    completed = 0
    
    pool = Pool(processes=cores)
    
    # 异步提交任务
    results = []
    for i in range(int(TOTAL_GAMES * 1.5)): # 多提交通50%的任务，防止平局不够数
        results.append(pool.apply_async(worker_play_game, (i,)))
    
    start_time = time.time()
    
    # 收集结果
    for res in results:
        if completed >= TOTAL_GAMES:
            break
            
        success, data, step = res.get()
        if success:
            all_examples.extend(data)
            completed += 1
            elapsed = time.time() - start_time
            print(f"[{completed}/{TOTAL_GAMES}] ✅ 完成 (步数:{step}) | 耗时: {elapsed:.1f}s", end="\r")
        else:
            print(f"[{completed}/{TOTAL_GAMES}] ⚠️ 平局 (跳过)      ", end="\r")
            
    pool.terminate()
    
    print(f"\n\n💾 生成完毕！有效局数: {completed}")
    print(f"总样本数: {len(all_examples)}")
    
    filepath = os.path.join(FOLDER, OUTPUT_FILE)
    with open(filepath, "wb+") as f:
        pickle.dump(all_examples, f)
    print(f"已保存至 {filepath}")

if __name__ == "__main__":
    # Windows下多进程必须放在 __main__ 保护块内
    main()
