import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm
import importlib

# ========== 动态导入 06_gnn_model ==========
try:
    gnn_module = importlib.import_module("06_gnn_model")
    InductiveGraphSAGE = gnn_module.InductiveGraphSAGE
    print("成功导入 06_gnn_model.py 中的 GraphSAGE 模型")
except ModuleNotFoundError:
    raise Exception("❌ 找不到 06_gnn_model.py 文件，请确保它与此脚本在同一目录下！")

# ========== 1. 数据集：返回用户、正样本、多个负样本 ==========
class BPRMultiNegDataset(Dataset):
    def __init__(self, interaction_file, num_items, num_neg=5):
        """
        :param interaction_file: 交互文件路径 (user, item, rating, timestamp)
        :param num_items: 物品总数
        :param num_neg: 每个正样本对应的负样本数量
        """
        print(f"正在加载交互数据: {interaction_file}")
        self.data = pd.read_csv(interaction_file, names=['user', 'item', 'rating', 'timestamp'])
        self.users = self.data['user'].values
        self.pos_items = self.data['item'].values
        self.ratings = self.data['rating'].values
        self.num_items = num_items
        self.num_neg = num_neg

        # 构建用户历史交互字典，用于负采样过滤
        print("构建用户历史交互字典...")
        self.user_history = self.data.groupby('user')['item'].apply(set).to_dict()

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        pos_item = self.pos_items[idx]
        rating = self.ratings[idx]

        # 采样 num_neg 个负样本（排除用户历史）
        neg_items = []
        for _ in range(self.num_neg):
            neg = np.random.randint(0, self.num_items)
            while neg in self.user_history[user]:
                neg = np.random.randint(0, self.num_items)
            neg_items.append(neg)
        return user, pos_item, neg_items, rating


# ========== 2. 多模态推荐模块（使用多个负样本）==========
class MultimodalRecommender(nn.Module):
    def __init__(self, num_users, item_output_dim):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, item_output_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)

    def forward(self, users, pos_items, neg_items_list, all_item_embs):
        """
        :param users: (batch_size,)
        :param pos_items: (batch_size,)
        :param neg_items_list: (batch_size, num_neg)
        :param all_item_embs: (N, item_output_dim)
        """
        u_emb = self.user_embedding(users)                     # (B, D)
        pos_emb = all_item_embs[pos_items]                     # (B, D)
        # 负样本嵌入: (B, num_neg, D)
        neg_emb = all_item_embs[neg_items_list]                # (B, num_neg, D)

        pos_scores = (u_emb * pos_emb).sum(dim=1)              # (B,)
        # 负样本得分: (B, num_neg)
        neg_scores = (u_emb.unsqueeze(1) * neg_emb).sum(dim=2) # (B, num_neg)

        return pos_scores, neg_scores


# ========== 3. 损失函数（BPR + 多负样本）==========
def bpr_loss_with_multi_neg(pos_scores, neg_scores):
    """
    pos_scores: (batch_size,)
    neg_scores: (batch_size, num_neg)
    返回标量损失
    """
    # 对于每个正样本，计算其与每个负样本的 log-sigmoid 差值，然后取平均
    # loss = -1/num_neg * sum_{neg} log(sigmoid(pos - neg))
    # 广播: pos_scores (B,1) - neg_scores (B, num_neg)
    diff = pos_scores.unsqueeze(1) - neg_scores   # (B, num_neg)
    loss = -torch.log(torch.sigmoid(diff) + 1e-8).mean()
    return loss


# ========== 4. 主训练函数 ==========
def train():
    # --- 配置参数 ---
    INTERACTION_FILE = '01_elec_5core_interactions.csv'
    IMAGE_FEAT_PATH = 'image_feat_aligned.npy'
    TEXT_FEAT_PATH = 'text_feat_aligned.npy'
    EDGES_PATH = 'joint_knn_edges.npz'

    BATCH_SIZE = 1024          # 子图采样的 batch 大小（种子节点数）
    NUM_NEG = 5                # 每个正样本的负样本数量
    EPOCHS = 30
    LR = 1e-3
    WEIGHT_DECAY = 1e-5
    HIDDEN_DIM = 256
    OUTPUT_DIM = 128
    NUM_LAYERS = 2             # 与 GraphSAGE 层数一致
    NUM_NEIGHBORS = [20, 10]   # 每层采样的邻居数，需与层数长度一致

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"训练设备: {device}")

    # --- 1. 加载节点多模态特征并拼接 ---
    print("加载对齐特征...")
    img_feat = np.load(IMAGE_FEAT_PATH).astype(np.float32)
    txt_feat = np.load(TEXT_FEAT_PATH).astype(np.float32)
    x_np = np.concatenate([img_feat, txt_feat], axis=1)   # (N, 512)
    node_features = torch.tensor(x_np, device=device)
    num_items = node_features.shape[0]
    print(f"物品数量: {num_items}, 特征维度: {node_features.shape[1]}")

    # --- 2. 加载图拓扑结构（KNN 边）并转换为 PyG 格式 ---
    print("加载图边结构...")
    edges_data = np.load(EDGES_PATH)
    edge_index_np = edges_data[edges_data.files[0]]

    # 鲁棒修复逻辑：确保 edge_index_np 形状为 (2, num_edges)
    if len(edge_index_np.shape) == 2 and edge_index_np.shape[1] == 2:
        edge_index_np = edge_index_np.T
    elif len(edge_index_np.shape) == 1:
        edge_index_np = edge_index_np.reshape(2, -1)
    elif len(edge_index_np.shape) == 2 and edge_index_np.shape[0] == num_items:
        # 假设是 (N, K) 邻居矩阵，转换为边列表
        N, K = edge_index_np.shape
        src = np.repeat(np.arange(N), K)
        dst = edge_index_np.flatten()
        edge_index_np = np.vstack([src, dst])
    else:
        raise ValueError(f"无法识别的边矩阵形状: {edge_index_np.shape}")

    edge_index = torch.tensor(edge_index_np, dtype=torch.long, device=device)
    print(f"图边数量: {edge_index.shape[1]}")

    # 创建 PyG Data 对象
    data = Data(x=node_features, edge_index=edge_index)

    # --- 3. 准备 BPR 数据集和普通 DataLoader（用于获取用户、物品ID）---
    if not os.path.exists(INTERACTION_FILE):
        raise FileNotFoundError(f"❌ 找不到交互文件 {INTERACTION_FILE}")
    dataset = BPRMultiNegDataset(INTERACTION_FILE, num_items, num_neg=NUM_NEG)
    num_users = len(dataset.user_history)
    print(f"用户数: {num_users}, 交互数: {len(dataset)}")

    # 注意：这里使用普通的 DataLoader 获取 (user, pos, neg_items, rating)
    # 但不直接用于模型前向，而是提供种子节点 ID
    bpr_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # --- 4. 初始化模型 ---
    feature_dim = node_features.shape[1]
    gnn_model = InductiveGraphSAGE(feature_dim, HIDDEN_DIM, OUTPUT_DIM).to(device)
    rec_model = MultimodalRecommender(num_users, OUTPUT_DIM).to(device)

    optimizer = optim.Adam(
        list(gnn_model.parameters()) + list(rec_model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # --- 5. 训练循环（使用 NeighborLoader 子图采样）---
    for epoch in range(1, EPOCHS + 1):
        gnn_model.train()
        rec_model.train()
        total_loss = 0.0
        num_batches = 0

        # 每个 epoch 重新创建 NeighborLoader，保证采样随机性
        # 种子节点是当前 batch 中出现的所有物品 ID（包括正负样本）
        # 我们需要提前从 bpr_loader 中获取当前 batch 的物品 ID 集合，然后作为种子传给 NeighborLoader
        # 但 NeighborLoader 期望的 input_nodes 是一个节点索引张量，且需要与 batch 顺序对应。
        # 实现方法：在迭代 bpr_loader 的同时，对每个 batch 单独构建一个子图采样器。
        # 这样每个 batch 独立采样，开销稍大但可行。

        pbar = tqdm(bpr_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for batch_idx, (users, pos_items, neg_items_list, ratings) in enumerate(pbar):
            users = users.to(device)
            pos_items = pos_items.to(device)
            # neg_items_list 是 list of lists，转换为 tensor (batch_size, num_neg)
            neg_items = torch.tensor(neg_items_list, dtype=torch.long, device=device)  # (B, num_neg)
            ratings = ratings.to(device)

            # 收集当前 batch 中涉及的所有物品 ID（正样本 + 所有负样本）
            # 注意：负样本是 (B, num_neg)，需要展平
            all_item_ids = torch.cat([pos_items, neg_items.view(-1)]).unique()
            # 使用 NeighborLoader 采样这些种子节点的子图
            # 注意：NeighborLoader 要求 input_nodes 是 1D tensor，且必须在图节点范围内
            loader = NeighborLoader(
                data,
                num_neighbors=NUM_NEIGHBORS,
                batch_size=len(all_item_ids),
                input_nodes=all_item_ids,
                shuffle=False,   # 不需要打乱，因为种子节点顺序不重要
                num_workers=0,
            )
            # 由于 input_nodes 可能很多，但 NeighborLoader 会返回一个 batch 对象
            # 注意：NeighborLoader 可能将 input_nodes 拆分成多个子 batch（如果超过 batch_size），但我们设置的 batch_size 等于种子节点数，所以通常只有一个子图
            # 为了简单，我们直接取第一个（也是唯一一个）子图
            for subgraph in loader:
                sub_x = subgraph.x
                sub_edge_index = subgraph.edge_index
                # 子图中节点的顺序：前 len(all_item_ids) 个节点就是种子节点，顺序与 all_item_ids 一致
                seed_emb = gnn_model(sub_x, sub_edge_index)[:len(all_item_ids)]
                # 建立映射：从 all_item_ids 到其嵌入的索引
                id_to_emb = {int(all_item_ids[i]): seed_emb[i] for i in range(len(all_item_ids))}
                # 提取 pos_items 和 neg_items 对应的嵌入
                pos_emb = torch.stack([id_to_emb[int(pid)] for pid in pos_items])   # (B, out_dim)
                neg_emb = torch.stack([torch.stack([id_to_emb[int(nid)] for nid in row]) for row in neg_items])  # (B, num_neg, out_dim)
                # 注意：neg_items 是 (B, num_neg)
                # 重构 all_item_embs 为 (B, out_dim) 和 (B, num_neg, out_dim) 后，可以送入推荐模型
                # 但推荐模型需要全量的 all_item_embs？不，我们只需要当前 batch 的正负物品嵌入。
                # 因此我们直接计算 BPR 损失，不经过 rec_model 的 user embedding 部分（user embedding 仍然需要）
                # 我们需要获取用户嵌入
                u_emb = rec_model.user_embedding(users)   # (B, out_dim)
                pos_scores = (u_emb * pos_emb).sum(dim=1)
                # neg_scores: (B, num_neg)
                neg_scores = (u_emb.unsqueeze(1) * neg_emb).sum(dim=2)
                loss = bpr_loss_with_multi_neg(pos_scores, neg_scores)
                break   # 只有一个子图，跳出循环

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(gnn_model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(rec_model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch} 平均损失: {avg_loss:.4f}")
        scheduler.step(avg_loss)

    # --- 6. 保存最终模型 ---
    torch.save({
        'gnn_state_dict': gnn_model.state_dict(),
        'rec_state_dict': rec_model.state_dict(),
    }, 'multimodal_recommender_optimized_mhm.pth')
    print("✅ 训练完成，模型已保存为 multimodal_recommender_optimized_mhm.pth")

    # 可选：生成最终的全量物品嵌入（用于评估）
    print("生成全量物品嵌入...")
    gnn_model.eval()
    with torch.no_grad():
        full_item_emb = gnn_model(node_features, edge_index).cpu().numpy()
    np.save('item_embeddings_final_mhm.npy', full_item_emb)
    print("物品嵌入已保存至 item_embeddings_final_mhm.npy")


if __name__ == "__main__":
    train()