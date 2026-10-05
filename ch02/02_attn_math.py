import torch
import torch.nn.functional as F

K = torch.tensor([
    [8,2,3], # アクション重視の映画
    [3,9,1], # ドラマ重視の映画
    [1,2,9], # コメディ重視の映画
    [5,5,5], # バランスの取れた映画
    [7,6,2], # アクションドラマ
    [2,7,6], # コメディドラマ
    [9,1,1], # 純粋なアクション
], dtype=torch.float32)

V = torch.tensor([
    [85],
    [70],
    [60],
    [75],
    [80],
    [65],
    [90]
], dtype=torch.float32)

Q = torch.tensor([
    [6,4,5],
    [2,8,3],
    [4,3,7],
], dtype=torch.float32)


def attention(Q, K, V):
    similarity = torch.matmul(Q, K.t())
    weights = F.softmax(similarity, dim=1)
    output = torch.matmul(weights, V)
    return output, weights

predicted_ratings, weights = attention(Q, K, V)

for movie, rating in zip(Q, predicted_ratings):
    print(f"映画 {movie.numpy()} の予測評価: {rating.item():.2f}")