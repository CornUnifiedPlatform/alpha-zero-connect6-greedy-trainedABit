import pickle
import numpy as np
import os
import sys
import time

sys.path.append(os.getcwd())
from connect6.GobangGame import GobangGame as Game

# === ⚡ 极速配置 ===
NUM_GAMES = 50   # 改为50局，足够热启动验证了！
OUTPUT_FILE = "checkpoint_0.pth.tar"
FOLDER = "./temp/"

class FastGreedyPlayer:
    def __init__(self, game):
        self.game = game
        self.n = game.n
        # 预计算中心分数，避免重复计算
        self.position_scores = np.zeros((self.n, self.n))
        c = self.n // 2
        for r in range(self.n):
            for col in range(self.n):
                self.position_scores[r][col] = (c - max(abs(r-c), abs(col-c))) * 0.1

    def play(self, board):
        # 1. 极速 Masking: 只看有子周围 2 格 (从3改2，提速明显)
        rows, cols = np.where(board != 0)
        if len(rows) == 0: return (self.n * self.n) // 2
        
        min_r, max_r = max(0, min(rows)-2), min(self.n, max(rows)+3)
        min_c, max_c = max(0, min(cols)-2), min(self.n, max(cols)+3)
        
        best_a = -1
        best_s = -float('inf')
        
        # 2. 遍历优化
        # 直接获取该区域的切片，避免全盘遍历
        for r in range(min_r, max_r):
            for c in range(min_c, max_c):
                if board[r][c] != 0: continue # 有子跳过
                a = r * self.n + c
                
                # --- 智能判定 (保持智商) ---
                
                # A. 绝杀 (Win)
                next_b, _ = self.game.getNextState(board, 1, a)
                if self.game.getGameEnded(next_b, 1) == 1: return a 
                
                # B. 必救 (Block)
                next_b_opp, _ = self.game.getNextState(board, -1, a)
                if self.game.getGameEnded(next_b_opp, 1) == -1:
                    score = 10000 
                else:
                    # C. 快速评分 (只算进攻，防守弱化以提速)
                    score = self.fast_evaluate(board, r, c, 1)
                    score += self.position_scores[r][c]
                    score += np.random.random() * 0.5

                if score > best_s:
                    best_s = score
                    best_a = a
        
        # 兜底
        if best_a == -1:
            valid = self.game.getValidMoves(board, 1)
            return np.random.choice(np.where(valid==1)[0])
            
        return best_a

    def fast_evaluate(self, board, r, c, color):
        # 简化版评分：只看连在一起的，不看跳棋，提速！
        score = 0
        directions = [(0,1), (1,0), (1,1), (1,-1)]
        for dr, dc in directions:
            count = 1
            # 正向
            nr, nc = r+dr, c+dc
            if 0<=nr<self.n and 0<=nc<self.n and board[nr][nc] == color:
                count += 1
                # 再看远一点
                nr+=dr; nc+=dc
                if 0<=nr<self.n and 0<=nc<self.n and board[nr][nc] == color:
                    count += 1
            # 反向
            nr, nc = r-dr, c-dc
            if 0<=nr<self.n and 0<=nc<self.n and board[nr][nc] == color:
                count += 1
                nr-=dr; nc-=dc
                if 0<=nr<self.n and 0<=nc<self.n and board[nr][nc] == color:
                    count += 1
            
            if count >= 4: score += 1000 # 只要能连4个就很好了
            elif count == 3: score += 100
            elif count == 2: score += 10
        return score

def generate():
    if not os.path.exists(FOLDER): os.makedirs(FOLDER)
    print(f"⚡ [极速版] 开始生成 {NUM_GAMES} 局数据...")
    
    game = Game(19)
    player = FastGreedyPlayer(game)
    all_examples = []
    
    start_total = time.time()
    for i in range(NUM_GAMES):
        start_time = time.time()
        print(f"生成第 {i+1}/{NUM_GAMES} 局: ", end="")
        
        board = game.getInitBoard()
        curPlayer = 1
        ep_data = []
        step = 0
        
        while True:
            step += 1
            if step % 10 == 0: print(".", end="", flush=True) # 心跳包
            
            canonical = game.getCanonicalForm(board, curPlayer)
            action = player.play(canonical)
            
            pi = np.zeros(game.getActionSize())
            pi[action] = 1
            sym = game.getSymmetries(canonical, pi)
            for b, p in sym: ep_data.append([b, curPlayer, p])
            
            board, curPlayer = game.getNextState(board, curPlayer, action)
            r = game.getGameEnded(board, 1)
            
            # 防止死循环：如果超过150手还没赢，强行平局结束
            if r != 0 or step > 150:
                # 只有分出胜负才记录，平局扔掉(为了让AI学怎么赢)
                if r != 0 and r != 1e-4:
                    for d in ep_data:
                        v = 1 if r == d[1] else -1
                        all_examples.append([d[0], d[2], v])
                    print(f" 完成! ({step}手, {time.time()-start_time:.1f}s)")
                else:
                    print(f" 平局/超时 (扔掉)")
                break
                
    filepath = os.path.join(FOLDER, OUTPUT_FILE)
    with open(filepath, "wb+") as f:
        pickle.dump(all_examples, f)
    
    print(f"\n✅ 全部完成！耗时: {time.time()-start_total:.1f}s")
    print(f"有效样本数: {len(all_examples)}")

if __name__ == "__main__":
    generate()
