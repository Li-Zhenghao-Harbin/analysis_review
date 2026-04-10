import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# ========= 1. 加载数据 =========
print("加载特征...")
img_feat = np.load("04_image_feat_aligned.npy")
txt_feat = np.load("04_text_feat_aligned.npy")

x = np.concatenate([img_feat, txt_feat], axis=1)  # (N, 512)
x = torch.tensor(x, dtype=torch.float)

print("加载图结构...")
edges = np.load("05_joint_knn_edges.npz")
row = edges["row"]
col = edges["col"]
full_edge_index = torch.tensor([row, col], dtype=torch.long)  # (2, E)

num_nodes = x.shape[0]
E = full_edge_index.shape[1]

# ========= 2. 划分训练边和验证边 =========
perm = torch.randperm(E)
train_mask = perm[:int(0.8 * E)]
val_mask   = perm[int(0.8 * E):]

train_edge_index = full_edge_index[:, train_mask]
val_edge_index   = full_edge_index[:, val_mask]

print(f"总边数: {E}, 训练边数: {train_edge_index.shape[1]}, 验证边数: {val_edge_index.shape[1]}")

# 训练时使用训练图
data = Data(x=x, edge_index=train_edge_index)

# ========= 3. GraphSAGE 模型（与之前相同）=========
class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

model = GraphSAGE(512, 256, 128)

# ========= 4. BPR Loss =========
def bpr_loss(user_emb, pos_emb, neg_emb):
    pos_score = (user_emb * pos_emb).sum(dim=1)
    neg_score = (user_emb * neg_emb).sum(dim=1)
    return -torch.mean(torch.log(torch.sigmoid(pos_score - neg_score)))

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ========= 5. 负采样（支持传入边索引）=========
def sample_batch(edge_index, num_nodes, batch_size):
    """
    从给定的边索引中随机采样 batch_size 条边，返回 (user, pos, neg)
    """
    edge_index_np = edge_index.numpy()
    idx = np.random.randint(0, edge_index_np.shape[1], batch_size)
    u = torch.tensor(edge_index_np[0, idx], dtype=torch.long)
    pos = torch.tensor(edge_index_np[1, idx], dtype=torch.long)
    neg = torch.randint(0, num_nodes, (batch_size,))
    return u, pos, neg

# ========= 6. 验证函数 =========
def validate(model, data, val_edge_index, num_nodes, batch_size=1024):
    model.eval()
    with torch.no_grad():
        emb = model(data.x, data.edge_index)   # 使用训练图计算 embedding
        u, pos, neg = sample_batch(val_edge_index, num_nodes, batch_size)
        loss = bpr_loss(emb[u], emb[pos], emb[neg])
    return loss.item()

# ========= 7. 训练 =========
EPOCHS = 10
BATCH_SIZE = 1024
STEPS_PER_EPOCH = 100   # 每个 epoch 采样的 batch 数（可根据训练边数调整）

train_loss_history = []
val_loss_history = []

print("\n开始训练 GNN（带验证集）...\n")

for epoch in range(EPOCHS):
    model.train()
    epoch_train_loss = 0.0

    pbar = tqdm(range(STEPS_PER_EPOCH), desc=f"Epoch {epoch+1}")

    for step in pbar:
        optimizer.zero_grad()

        # 全图前向（使用训练图）
        emb = model(data.x, data.edge_index)

        # 从训练边中采样 batch
        u, pos, neg = sample_batch(train_edge_index, num_nodes, BATCH_SIZE)

        loss = bpr_loss(emb[u], emb[pos], emb[neg])

        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_train_loss = epoch_train_loss / STEPS_PER_EPOCH
    train_loss_history.append(avg_train_loss)

    # 验证
    avg_val_loss = validate(model, data, val_edge_index, num_nodes, BATCH_SIZE)
    val_loss_history.append(avg_val_loss)

    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}\n")

# ========= 8. 保存最终 embedding =========
# 使用全图（或训练图）生成最终表示，这里使用训练图
model.eval()
with torch.no_grad():
    final_emb = model(data.x, data.edge_index)
torch.save(final_emb, "gnn_item_embedding_mhm.pt")
print("✅ GNN训练完成，embedding已保存！")

# ========= 9. 绘制训练和验证 Loss 曲线 =========
plt.figure(figsize=(8, 5))
plt.plot(train_loss_history, label="Train Loss", marker='o')
plt.plot(val_loss_history, label="Validation Loss", marker='s')
plt.xlabel("Epoch")
plt.ylabel("BPR Loss")
plt.title("Training and Validation Loss Curve")
plt.legend()
plt.grid(True)
plt.savefig("loss_curve.png", dpi=150)
plt.show()

# 简单判断是否过拟合
if len(val_loss_history) >= 3:
    if val_loss_history[-1] > val_loss_history[-2] > val_loss_history[-3]:
        print("⚠️ 警告: 验证 loss 连续上升，可能存在过拟合！")
    else:
        print("✅ 验证 loss 未明显上升，模型训练良好。")
else:
    print("训练完成，请观察 loss 曲线判断是否过拟合。")