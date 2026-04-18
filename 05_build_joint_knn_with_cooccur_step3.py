import os
from collections import defaultdict
from itertools import combinations

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm

# ========== 配置 ==========
IMAGE_ALIGNED_FILE = '04_image_feat_aligned_item_coldstart.npy'
TEXT_ALIGNED_FILE = '04_text_feat_aligned_item_coldstart.npy'
INTERACTION_FILE = '01_elec_5core_interactions.csv'

OUTPUT_NEIGHBORS = '05_joint_knn_neighbors_item_coldstart.npy'
OUTPUT_SCORES = '05_joint_knn_scores_item_coldstart.npy'
OUTPUT_EDGES = '05_joint_knn_edges_item_coldstart.npz'

K_CONTENT = 40          # 内容图每个节点的KNN邻居数（不含自身）
K_COOC = 20             # 共现图每个节点最多保留多少个邻居
IMAGE_WEIGHT = 0.30
TEXT_WEIGHT = 0.70
CHUNK_SIZE = 10000


# ========== 1. 加载对齐特征 ==========
print('加载对齐后的图像特征...')
if not os.path.exists(IMAGE_ALIGNED_FILE):
    raise FileNotFoundError(f'未找到 {IMAGE_ALIGNED_FILE}，请先运行 04。')
img_feat = np.load(IMAGE_ALIGNED_FILE).astype(np.float32)

print('加载对齐后的文本特征...')
if not os.path.exists(TEXT_ALIGNED_FILE):
    raise FileNotFoundError(f'未找到 {TEXT_ALIGNED_FILE}，请先运行 04。')
txt_feat = np.load(TEXT_ALIGNED_FILE).astype(np.float32)

if img_feat.shape[0] != txt_feat.shape[0]:
    raise ValueError(f'图像和文本特征行数不一致: {img_feat.shape[0]} vs {txt_feat.shape[0]}')

N = img_feat.shape[0]
print(f'节点总数: {N}')
print(f'图像特征维度: {img_feat.shape[1]}, 文本特征维度: {txt_feat.shape[1]}')


# ========== 2. 构建内容相似图 ==========
print('分别归一化图像/文本特征...')
faiss.normalize_L2(img_feat)
faiss.normalize_L2(txt_feat)

print('按权重拼接多模态特征...')
joint_feat = np.concatenate([IMAGE_WEIGHT * img_feat, TEXT_WEIGHT * txt_feat], axis=1)
faiss.normalize_L2(joint_feat)
print(f'联合特征维度: {joint_feat.shape[1]} | image_weight={IMAGE_WEIGHT}, text_weight={TEXT_WEIGHT}')

print('构建 FAISS 内积索引...')
dim = joint_feat.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(joint_feat)

print(f'检索每个节点的 {K_CONTENT + 1} 个最近邻...')
neighbors_list = []
scores_list = []
with tqdm(total=N, desc='FAISS 搜索') as pbar:
    for start in range(0, N, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, N)
        chunk_vecs = joint_feat[start:end]
        scores_chunk, neighbors_chunk = index.search(chunk_vecs, K_CONTENT + 1)
        neighbors_list.append(neighbors_chunk)
        scores_list.append(scores_chunk)
        pbar.update(end - start)

neighbors = np.concatenate(neighbors_list, axis=0)[:, 1:]
scores = np.concatenate(scores_list, axis=0)[:, 1:]

np.save(OUTPUT_NEIGHBORS, neighbors)
np.save(OUTPUT_SCORES, scores)
print(f'已保存内容图邻居矩阵: {OUTPUT_NEIGHBORS} {neighbors.shape}')
print(f'已保存内容图分数矩阵: {OUTPUT_SCORES} {scores.shape}')


# ========== 3. 构建 item-item 共现图 ==========
def build_cooccurrence_edges(interaction_file: str, num_items: int, topk: int):
    print('从交互数据构建 item 共现图...')
    if not os.path.exists(interaction_file):
        raise FileNotFoundError(f'未找到 {interaction_file}')

    df = pd.read_csv(interaction_file, usecols=['user_id', 'item_id'])
    df['item_id'] = df['item_id'].astype(int)

    co_counts = defaultdict(lambda: defaultdict(int))
    user_groups = df.groupby('user_id')['item_id'].apply(list)

    for items in tqdm(user_groups, desc='统计共现'):
        uniq = sorted(set(int(x) for x in items if 0 <= int(x) < num_items))
        if len(uniq) < 2:
            continue
        for a, b in combinations(uniq, 2):
            co_counts[a][b] += 1
            co_counts[b][a] += 1

    rows = []
    cols = []
    for i in tqdm(range(num_items), desc='截断每个点的共现邻居'):
        if i not in co_counts:
            continue
        nbrs = sorted(co_counts[i].items(), key=lambda x: (-x[1], x[0]))[:topk]
        for j, _ in nbrs:
            rows.append(i)
            cols.append(int(j))

    return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)


co_rows, co_cols = build_cooccurrence_edges(INTERACTION_FILE, N, K_COOC)
print(f'共现图边数(有向): {len(co_rows):,}')


# ========== 4. 融合内容图与共现图 ==========
print('融合内容图与共现图...')
content_rows = np.repeat(np.arange(N, dtype=np.int64), neighbors.shape[1])
content_cols = neighbors.reshape(-1).astype(np.int64)

all_rows = np.concatenate([content_rows, co_rows])
all_cols = np.concatenate([content_cols, co_cols])

valid = (all_rows >= 0) & (all_rows < N) & (all_cols >= 0) & (all_cols < N) & (all_rows != all_cols)
all_rows = all_rows[valid]
all_cols = all_cols[valid]

# 对称化 + 去重
rows_sym = np.concatenate([all_rows, all_cols])
cols_sym = np.concatenate([all_cols, all_rows])
edge_pairs = np.stack([rows_sym, cols_sym], axis=1)
edge_pairs = np.unique(edge_pairs, axis=0)
rows_sym = edge_pairs[:, 0]
cols_sym = edge_pairs[:, 1]

np.savez(OUTPUT_EDGES, row=rows_sym, col=cols_sym)

print('=' * 60)
print('融合图构建完成')
print(f'内容图 K: {K_CONTENT}')
print(f'共现图 topK: {K_COOC}')
print(f'最终边文件: {OUTPUT_EDGES}')
print(f'最终边数(有向, 去重+对称化后): {len(rows_sym):,}')
print('=' * 60)
