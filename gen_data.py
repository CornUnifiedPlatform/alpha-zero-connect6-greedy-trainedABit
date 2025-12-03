# gen_data.py (基于 Linear Greedy V2.0)
import pickle
import numpy as np
import os
import sys

# 引用路径确保正确
sys.path.append(os.getcwd())
from connect6.GobangGame import GobangGame as Game

# === 配置 ===
# 建议生成 500 局，配合 19x19 的 Masking 优化，速度很快
NUM_GAMES = 500 
OUTPUT_FILE = "checkpoint_0.pth.tar"
FOLDER = "./temp/"

class LinearGreedyPlayer:
    def __init__(self, game):
        self.game = game
        self.n = game.n
        # 基础位置分
        self.position_scores = np.zeros((self.n, self.n))
        c = self.n // 2
        for r in range(self.n):
            for col in range(self.n):
                self.position_scores[r][col] = c - max(abs(r-c), abs(col-c))

    def play(self, board):
        # 1. 缩小搜索范围 (Masking)
        rows, cols = np.where(board != 0)
        if len(rows) == 0: return (self.n * self.n) // 2
        
        min_r, max_r = max(0, min(rows)-3), min(self.n, max(rows)+4)
        min_c, max_c = max(0, min(cols)-3), min(self.n, max(cols)+4)
        
        best_a = -1
        best_s = -float('inf')
        
        for r in range(min_r, max_r):
            for c in range(min_c, max_c):
                a = r * self.n + c
                if board[r][c] != 0: continue
                
                # --- V2.0 强力逻辑 ---
                
                # A. 绝杀判定
                next_b, _ = self.game.getNextState(board, 1, a)
                if self.game.getGameEnded(next_b, 1) == 1:
                    return a 
                
                # B. 必救判定
                next_b_opp, _ = self.game.getNextState(board, -1, a)
                if self.game.getGameEnded(next_b_opp, 1) == -1:
                    score = 200000 
                else:
                    score = 0

                # C. 线性进攻评分
                score += self.evaluate_lines(board, r, c, 1)
                # 防守评分 (权重 0.8)
                score += self.evaluate_lines(board, r, c, -1) * 0.8
                
                # D. 位置与随机
                score += self.position_scores[r][c] * 0.5
                score += np.random.normal(0, 0.5) # 增加随机性，让棋谱更多样化

                if score > best_s:
                    best_s = score
                    best_a = a
                    
        # 兜底
        if best_a == -1:
            valid = self.game.getValidMoves(board, 1)
            return np.random.choice(np.where(valid==1)[0])
            
        return best_a

    def evaluate_lines(self, board, r, c, color):
        score = 0
        directions = [(0,1), (1,0), (1,1), (1,-1)]
        for dr, dc in directions:
            count = 1
            k = 1
            while 0 <= r + k*dr < self.n and 0 <= c + k*dc < self.n and board[r+k*dr][c+k*dc] == color:
                count += 1; k += 1
            k = 1
            while 0 <= r - k*dr < self.n and 0 <= c - k*dc < self.n and board[r-k*dr][c-k*dc] == color:
                count += 1; k += 1
            
            # 指数级评分
            if count >= 6: score += 100000
            elif count == 5: score += 8000
            elif count == 4: score += 1000
            elif count == 3: score += 200
            elif count == 2: score += 20
        return score

def generate():
    if not os.path.exists(FOLDER): os.makedirs(FOLDER)
    print(f"🚀 [GitHub版] 开始生成 {NUM_GAMES} 局高质量 V2.0 数据...")
    
    game = Game(19)
    player = LinearGreedyPlayer(game)
    all_examples = []
    
    for i in range(NUM_GAMES):
        if i % 10 == 0: print(f"生成进度: {i}/{NUM_GAMES}...", end="\r")
        board = game.getInitBoard()
        curPlayer = 1
        ep_data = []
        
        while True:
            canonical = game.getCanonicalForm(board, curPlayer)
            action = player.play(canonical)
            
            pi = np.zeros(game.getActionSize())
            pi[action] = 1
            sym = game.getSymmetries(canonical, pi)
            for b, p in sym: ep_data.append([b, curPlayer, p])
            
            board, curPlayer = game.getNextState(board, curPlayer, action)
            r = game.getGameEnded(board, 1)
            if r != 0:
                for d in ep_data:
                    v = 1 if r == d[1] else -1
                    all_examples.append([d[0], d[2], v])
                break
                
    filepath = os.path.join(FOLDER, OUTPUT_FILE)
    with open(filepath, "wb+") as f:
        pickle.dump(all_examples, f)
    print(f"\n✅ 生成完毕！保存至 {filepath}")

if __name__ == "__main__":
    generate()
