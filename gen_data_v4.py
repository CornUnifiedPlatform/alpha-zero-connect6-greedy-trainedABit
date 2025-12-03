import pickle
import numpy as np
import os
import sys
import time
import random

sys.path.append(os.getcwd())
try:
    from connect6.GobangGame import GobangGame as Game
except ImportError:
    # 兼容 Kaggle 路径
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from connect6.GobangGame import GobangGame as Game

# === 配置 ===
NUM_GAMES = 100  # 先跑100局，确保能快速出结果
OUTPUT_FILE = "checkpoint_0.pth.tar"
FOLDER = "./temp/"

class AgileGreedyPlayer:
    def __init__(self, game):
        self.game = game
        self.n = game.n
        # 预计算位置分
        self.position_scores = np.zeros((self.n, self.n))
        c = self.n // 2
        for r in range(self.n):
            for col in range(self.n):
                self.position_scores[r][col] = (c - max(abs(r-c), abs(col-c))) * 0.1

    def play(self, board):
        # 1. 极速 Masking: 只看有子周围 2 格 (范围缩小，速度提升)
        rows, cols = np.where(board != 0)
        if len(rows) == 0: return (self.n * self.n) // 2
        
        min_r, max_r = max(0, min(rows)-2), min(self.n, max(rows)+3)
        min_c, max_c = max(0, min(cols)-2), min(self.n, max(cols)+3)
        
        best_a = -1
        best_s = -float('inf')
        
        # 2. 遍历
        for r in range(min_r, max_r):
            for c in range(min_c, max_c):
                if board[r][c] != 0: continue
                a = r * self.n + c
                
                # --- 智能判定 ---
                
                # A. 绝杀 (Win in 1)
                next_b, _ = self.game.getNextState(board, 1, a)
                if self.game.getGameEnded(next_b, 1) == 1: return a 
                
                # B. 必救 (Block Win)
                next_b_opp, _ = self.game.getNextState(board, -1, a)
                if self.game.getGameEnded(next_b_opp, 1) == -1:
                    # 发现必救点，给予高分，但不直接返回，
                    # 因为可能存在既能必救又能进攻的双重好点
                    score = 50000 
                else:
                    score = 0

                # C. 快速线性评分
                # 进攻权重 1.2，防守权重 0.8 -> 鼓励进攻，防止平局
                score += self.fast_evaluate(board, r, c, 1) * 1.2
                score += self.fast_evaluate(board, r, c, -1) * 0.8
                
                # D. 软随机 (Soft Noise)
                # 不乱下，而是给好棋加一点点波动，让它每次选不一样的套路
                score += self.position_scores[r][c]
                score += np.random.normal(0, 2.0) 

                if score > best_s:
                    best_s = score
                    best_a = a
        
        # 兜底
        if best_a == -1:
            valid = self.game.getValidMoves(board, 1)
            return np.random.choice(np.where(valid==1)[0])
            
        return best_a

    def fast_evaluate(self, board, r, c, color):
        # 优化版的评分：只看能不能连
        score = 0
        directions = [(0,1), (1,0), (1,1), (1,-1)]
        for dr, dc in directions:
            count = 1
            # 正向延伸
            nr, nc = r+dr, c+dc
            while 0<=nr<self.n and 0<=nc<self.n and board[nr][nc] == color:
                count += 1; nr += dr; nc += dc
            # 反向延伸
            nr, nc = r-dr, c-dc
            while 0<=nr<self.n and 0<=nc<self.n and board[nr][nc] == color:
                count += 1; nr -= dr; nc -= dc
            
            # 只有连成一定规模才给分，减少无效计算
            if count >= 6: score += 100000
            elif count == 5: score += 5000
            elif count == 4: score += 500
            elif count == 3: score += 50
            # 2个以下的忽略，提速
            
        return score

def generate():
    if not os.path.exists(FOLDER): os.makedirs(FOLDER)
    print(f"🚀 [V4.0 极速修正版] 开始生成 {NUM_GAMES} 局数据...")
    
    # 再次确认：必须是 19！
    game = Game(19) 
    player = AgileGreedyPlayer(game)
    all_examples = []
    
    start_total = time.time()
    
    for i in range(NUM_GAMES):
        print(f"生成第 {i+1}/{NUM_GAMES} 局: ", end="")
        board = game.getInitBoard()
        curPlayer = 1
        ep_data = []
        step = 0
        
        while True:
            step += 1
            if step % 10 == 0: print(".", end="", flush=True) # 心跳
            
            canonical = game.getCanonicalForm(board, curPlayer)
            action = player.play(canonical)
            
            pi = np.zeros(game.getActionSize())
            pi[action] = 1
            sym = game.getSymmetries(canonical, pi)
            for b, p in sym: ep_data.append([b, curPlayer, p])
            
            board, curPlayer = game.getNextState(board, curPlayer, action)
            r = game.getGameEnded(board, 1)
            
            # 限制 120 手，防止死局
            if r != 0 or step > 120:
                if r != 0 and r != 1e-4:
                    # 分出胜负了
                    for d in ep_data:
                        v = 1 if r == d[1] else -1
                        all_examples.append([d[0], d[2], v])
                    print(f" 完成! ({step}手)")
                else:
                    # 平局或超时，扔掉
                    print(f" 平局 (跳过)")
                break
                
    filepath = os.path.join(FOLDER, OUTPUT_FILE)
    with open(filepath, "wb+") as f:
        pickle.dump(all_examples, f)
    
    print(f"\n✅ 全部完成！耗时: {time.time()-start_total:.1f}s")
    print(f"有效样本数: {len(all_examples)}")

if __name__ == "__main__":
    generate()
