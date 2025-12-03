from __future__ import print_function
import sys
import numpy as np
# 注意：如果你把文件夹改名了，这里要相应修改，比如 from .Connect6Logic import Board
from .GobangLogic import Board 
from Game import Game

class GobangGame(Game):
    def __init__(self, n=19, nir=6):
        # 修改点1：默认大小改为19x19，连珠数改为6
        self.n = n
        self.n_in_row = nir

    def getInitBoard(self):
        b = Board(self.n)
        return np.array(b.pieces)

    def getBoardSize(self):
        return (self.n, self.n)

    def getActionSize(self):
        return self.n * self.n + 1

    def getNextState(self, board, player, action):
        # 修改点2：实现六子棋特殊的落子规则
        if action == self.n * self.n:
            return (board, -player)
        
        b = Board(self.n)
        b.pieces = np.copy(board)
        move = (int(action / self.n), action % self.n)
        b.execute_move(move, player)
        
        # --- 六子棋核心逻辑开始 ---
        # 统计棋盘上目前有多少颗子（非0的元素个数）
        # 这一步下完后的总子数
        pieces_count = np.count_nonzero(b.pieces)
        
        # 规则推导：
        # 第1手(黑): 下完后盘面1子 -> 奇数 -> 换人(白)
        # 第2手(白): 下完后盘面2子 -> 偶数 -> 不换人(白继续)
        # 第3手(白): 下完后盘面3子 -> 奇数 -> 换人(黑)
        # 第4手(黑): 下完后盘面4子 -> 偶数 -> 不换人(黑继续)
        # 结论：下完后如果总子数是奇数，则换人；如果是偶数，则继续。
        
        if pieces_count % 2 != 0:
            return (b.pieces, -player) # 换人
        else:
            return (b.pieces, player)  # 不换人
        # --- 六子棋核心逻辑结束 ---

# connect6/GobangGame.py 中的 getValidMoves 方法

    def getValidMoves(self, board, player):
        # 初始化全为 0 (不可走)
        valids = [0] * self.getActionSize()
        b = Board(self.n)
        b.pieces = np.copy(board)
        legalMoves = b.get_legal_moves(player)
        
        if len(legalMoves) == 0:
            valids[-1] = 1
            return np.array(valids)

        # === 核心修改：增加索敌范围限制 (Masking) ===
        # 1. 找到棋盘上所有已存在的棋子
        # board 是 numpy 数组，非 0 处即有子
        existing_pieces = np.argwhere(board != 0)
        
        # 2. 如果棋盘是空的（第一手），强制只能下天元附近
        if len(existing_pieces) == 0:
            center = self.n // 2
            # 允许天元周围 3x3 范围
            for x in range(center-1, center+2):
                for y in range(center-1, center+2):
                    valids[self.n * x + y] = 1
            return np.array(valids)

        # 3. 计算有效搜索框 (ROI)
        # 只允许在现有棋子周围 3 格范围内落子
        # 这一步能把 361 个搜索点减少到 20-50 个！
        rows, cols = existing_pieces[:, 0], existing_pieces[:, 1]
        min_r, max_r = max(0, min(rows) - 3), min(self.n, max(rows) + 4)
        min_c, max_c = max(0, min(cols) - 3), min(self.n, max(cols) + 4)

        # 4. 填充 valids
        for x, y in legalMoves:
            # 只有在 ROI 范围内的合法点才设为 1
            if min_r <= x < max_r and min_c <= y < max_c:
                valids[self.n * x + y] = 1
        
        # 兜底：如果算完发现没路走了（极端情况），则允许所有合法步
        if sum(valids) == 0:
            for x, y in legalMoves:
                valids[self.n * x + y] = 1
                
        return np.array(valids)

    def getGameEnded(self, board, player):
        # 胜利判断逻辑
        # 因为我们在 __init__ 里设置了 self.n_in_row = 6
        # 所以这里的逻辑会自动变成判断 6 子连珠，不需要改代码，只需要改参数
        b = Board(self.n)
        b.pieces = np.copy(board)
        n = self.n_in_row

        for w in range(self.n):
            for h in range(self.n):
                # 检查横向
                if (w in range(self.n - n + 1) and board[w][h] != 0 and
                        len(set(board[i][h] for i in range(w, w + n))) == 1):
                    return board[w][h]
                # 检查纵向
                if (h in range(self.n - n + 1) and board[w][h] != 0 and
                        len(set(board[w][j] for j in range(h, h + n))) == 1):
                    return board[w][h]
                # 检查左上到右下斜向
                if (w in range(self.n - n + 1) and h in range(self.n - n + 1) and board[w][h] != 0 and
                        len(set(board[w + k][h + k] for k in range(n))) == 1):
                    return board[w][h]
                # 检查左下到右上斜向
                if (w in range(self.n - n + 1) and h in range(n - 1, self.n) and board[w][h] != 0 and
                        len(set(board[w + l][h - l] for l in range(n))) == 1):
                    return board[w][h]
        
        if b.has_legal_moves():
            return 0
        return 1e-4

    def getCanonicalForm(self, board, player):
        return player * board

    def getSymmetries(self, board, pi):
        assert(len(pi) == self.n**2 + 1)
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return l

    def stringRepresentation(self, board):
        return board.tobytes()

    @staticmethod
    def display(board):
        # 简单的控制台打印，方便你调试
        n = board.shape[0]
        print("   ", end="")
        for y in range(n):
            print(f"{y:2}", end=" ")
        print("")
        print("  " + "-" * (n * 3))
        for y in range(n):
            print(f"{y:2}|", end="") 
            for x in range(n):
                piece = board[y][x] 
                if piece == -1: print(" B ", end="")
                elif piece == 1: print(" W ", end="")
                else: print(" . ", end="")
            print("|")
        print("  " + "-" * (n * 3))