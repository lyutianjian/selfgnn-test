"""
MovieLens-1M 数据集处理脚本
将原始数据转换为CTR和Top-k推荐任务的格式
"""

import numpy as np
import pandas as pd
import os
import json
import urllib.request
import zipfile
from datetime import datetime
from tqdm import tqdm

# ==================== 配置 ====================
DATASET = 'ml-1m'
RAW_PATH = os.path.join('.', DATASET)
CTR_PATH = './ML_1MCTR/'
TOPK_PATH = './ML_1MTOPK/'
RANDOM_SEED = 0
NEG_ITEMS = 99

# ==================== 下载数据 ====================
def download_data():
    """下载并解压MovieLens-1M数据集"""
    if not os.path.exists(RAW_PATH):
        os.makedirs(RAW_PATH, exist_ok=True)
    
    zip_path = os.path.join(RAW_PATH, DATASET + '.zip')
    if not os.path.exists(zip_path):
        print('正在下载数据到 ' + RAW_PATH)
        url = f'http://files.grouplens.org/datasets/movielens/{DATASET}.zip'
        try:
            urllib.request.urlretrieve(url, zip_path)
            print('下载完成，正在解压...')
        except Exception as e:
            print(f'下载失败: {e}')
            print('请手动下载数据放到 ' + RAW_PATH)
            return False
    
    if os.path.exists(zip_path):
        try:
            with zipfile.ZipFile(zip_path, 'r') as f:
                f.extractall(RAW_PATH)
            print('解压完成')
            return True
        except Exception as e:
            print(f'解压失败: {e}')
            return False
    
    return False

# ==================== 加载并预处理交互数据 ====================
def load_interactions():
    """加载交互数据"""
    interactions = []
    user_freq, item_freq = dict(), dict()
    
    file = os.path.join(RAW_PATH, DATASET, "ratings.dat")
    if not os.path.exists(file):
        print(f'错误: 找不到 {file}')
        return None, None, None
    
    print('正在加载交互数据...')
    with open(file, encoding='latin-1') as F:
        for line in tqdm(F):
            line = line.strip().split("::")
            uid, iid, rating, time = line[0], line[1], float(line[2]), float(line[3])
            label = 1 if rating >= 4 else 0
            interactions.append([uid, time, iid, label])
            
            if int(label) == 1:
                user_freq[uid] = user_freq.get(uid, 0) + 1
                item_freq[iid] = item_freq.get(iid, 0) + 1
    
    return interactions, user_freq, item_freq

# ==================== 5-core 过滤 ====================
def filter_5core(interactions, user_freq, item_freq):
    """5-core过滤：每个用户和项目至少5次交互"""
    print('正在进行5-core过滤...')
    
    select_uid, select_iid = [], []
    while len(select_uid) < len(user_freq) or len(select_iid) < len(item_freq):
        select_uid, select_iid = [], []
        for u in user_freq:
            if user_freq[u] >= 5:
                select_uid.append(u)
        for i in item_freq:
            if item_freq[i] >= 5:
                select_iid.append(i)
        print(f"用户: {len(select_uid)}/{len(user_freq)}, 项目: {len(select_iid)}/{len(item_freq)}")

        select_uid = set(select_uid)
        select_iid = set(select_iid)
        user_freq, item_freq = dict(), dict()
        interactions_5core = []
        
        for line in tqdm(interactions):
            uid, iid, label = line[0], line[2], line[-1]
            if uid in select_uid and iid in select_iid:
                interactions_5core.append(line)
                if int(label) == 1:
                    user_freq[uid] = user_freq.get(uid, 0) + 1
                    item_freq[iid] = item_freq.get(iid, 0) + 1
        interactions = interactions_5core
    
    print(f"过滤后: 交互 {len(interactions)}, 用户 {len(select_uid)}, 项目 {len(select_iid)}")
    return interactions, select_uid, select_iid

# ==================== 创建DataFrame ====================
def create_interaction_df(interactions):
    """创建交互数据DataFrame"""
    print('正在创建数据框...')
    
    ts = []
    for i in tqdm(range(len(interactions))):
        ts.append(datetime.fromtimestamp(interactions[i][1]))
    
    interaction_df = pd.DataFrame(interactions, columns=["user_id", "time", "news_id", "label"])
    interaction_df['timestamp'] = ts
    interaction_df['hour'] = interaction_df['timestamp'].apply(lambda x: x.hour)
    interaction_df['weekday'] = interaction_df['timestamp'].apply(lambda x: x.weekday())
    interaction_df['date'] = interaction_df['timestamp'].apply(lambda x: x.date())
    
    # 定义时间段
    def get_time_range(hour):
        if hour >= 5 and hour <= 8:
            return 0
        if hour > 8 and hour < 11:
            return 1
        if hour >= 11 and hour <= 12:
            return 2
        if hour > 12 and hour <= 15:
            return 3
        if hour > 15 and hour <= 17:
            return 4
        if hour >= 18 and hour <= 19:
            return 5
        if hour > 19 and hour <= 21:
            return 6
        if hour > 21:
            return 7
        return 8
    
    interaction_df['period'] = interaction_df.hour.apply(lambda x: get_time_range(x))
    min_date = interaction_df.date.min()
    interaction_df['day'] = (interaction_df.date - min_date).apply(lambda x: x.days)
    
    return interaction_df

# ==================== CTR任务数据准备 ====================
def prepare_ctr_data(interaction_df):
    """准备CTR任务数据"""
    print('\n========== 准备CTR任务数据 ==========')
    os.makedirs(CTR_PATH, exist_ok=True)
    
    interaction_ctr = interaction_df.copy()
    interaction_ctr.rename(columns={
        'hour': 'c_hour_c',
        'weekday': 'c_weekday_c',
        'period': 'c_period_c',
        'day': 'c_day_f',
        'user_id': 'original_user_id'
    }, inplace=True)
    
    # 重新编号用户和项目
    user2newid_ctr = dict(zip(sorted(interaction_ctr.original_user_id.unique()),
                              range(1, interaction_ctr.original_user_id.nunique() + 1)))
    interaction_ctr['user_id'] = interaction_ctr.original_user_id.apply(lambda x: user2newid_ctr[x])

    item2newid_ctr = dict(zip(sorted(interaction_ctr.news_id.unique()),
                              range(1, interaction_ctr.news_id.nunique() + 1)))
    interaction_ctr['item_id'] = interaction_ctr['news_id'].apply(lambda x: item2newid_ctr[x])
    interaction_ctr.sort_values(by=['user_id', 'time'], inplace=True)
    interaction_ctr = interaction_ctr.reset_index(drop=True)

    # 保存映射
    nu2nid = {int(k): v for k, v in user2newid_ctr.items()}
    ni2nid = {int(k): v for k, v in item2newid_ctr.items()}
    json.dump(nu2nid, open(os.path.join(CTR_PATH, "user2newid.json"), 'w'))
    json.dump(ni2nid, open(os.path.join(CTR_PATH, "item2newid.json"), 'w'))
    
    # 分割训练、验证、测试集
    split_time1 = interaction_ctr.c_day_f.max() * 0.8
    train = interaction_ctr.loc[interaction_ctr.c_day_f <= split_time1].copy()
    val_test = interaction_ctr.loc[(interaction_ctr.c_day_f > split_time1)].copy()
    split_time2 = interaction_ctr.c_day_f.max() * 0.9
    val = val_test.loc[val_test.c_day_f <= split_time2].copy()
    test = val_test.loc[val_test.c_day_f > split_time2].copy()

    # 删除验证/测试集中不在训练集的用户和项目
    train_u, train_i = set(train.user_id.unique()), set(train.item_id.unique())
    val_sel = val.loc[(val.user_id.isin(train_u)) & (val.item_id.isin(train_i))].copy()
    test_sel = test.loc[(test.user_id.isin(train_u)) & (test.item_id.isin(train_i))].copy()
    
    print(f"训练集 - 用户: {len(train_u)}, 项目: {len(train_i)}")
    print(f"验证集 - 用户: {val_sel.user_id.nunique()}, 项目: {val_sel.item_id.nunique()}")
    print(f"测试集 - 用户: {test_sel.user_id.nunique()}, 项目: {test_sel.item_id.nunique()}")
    
    # 分配impression IDs
    print('正在分配impression IDs...')
    max_imp_len = 20
    for interaction_partial in [train, val_sel, test_sel]:
        interaction_partial['last_user_id'] = interaction_partial['user_id'].shift(1)
        impression_ids = []
        impression_len = 0
        current_impid = 0
        
        for uid, last_uid in tqdm(interaction_partial[['user_id', 'last_user_id']].to_numpy()):
            if uid == last_uid:
                if impression_len >= max_imp_len:
                    current_impid += 1
                    impression_len = 1
                else:
                    impression_len += 1
                impression_ids.append(current_impid)
            else:
                current_impid += 1
                impression_len = 1
                impression_ids.append(current_impid)
        interaction_partial.loc[:, 'impression_id'] = impression_ids
    
    # 保存数据
    select_columns = ['user_id', 'item_id', 'time', 'label', 'c_hour_c', 'c_weekday_c', 'c_period_c', 'c_day_f', 'impression_id']
    train[select_columns].to_csv(os.path.join(CTR_PATH, 'train.csv'), sep="\t", index=False)
    val_sel[select_columns].to_csv(os.path.join(CTR_PATH, 'dev.csv'), sep="\t", index=False)
    test_sel[select_columns].to_csv(os.path.join(CTR_PATH, 'test.csv'), sep="\t", index=False)
    
    # 保存项目元数据
    item_meta = pd.read_csv(os.path.join(RAW_PATH, DATASET, "movies.dat"),
                sep='::', names=['movieId', 'title', 'genres'],
                encoding='latin-1', engine='python')
    item_select = item_meta.loc[item_meta.movieId.isin(interaction_ctr.news_id.unique())].copy()
    item_select['item_id'] = item_select.movieId.apply(lambda x: item2newid_ctr[str(x)])
    genres2id = dict(zip(sorted(item_select.genres.unique()),
                        range(1, item_select.genres.nunique() + 1)))
    item_select['i_genre_c'] = item_select['genres'].apply(lambda x: genres2id[x])
    title2id = dict(zip(sorted(item_select.title.unique()),
                       range(1, item_select.title.nunique() + 1)))
    item_select['i_title_c'] = item_select['title'].apply(lambda x: title2id[x])
    item_select[['item_id', 'i_genre_c', 'i_title_c']].to_csv(
        os.path.join(CTR_PATH, 'item_meta.csv'), sep="\t", index=False)
    
    print(f'CTR数据已保存到 {CTR_PATH}')

# ==================== Top-k任务数据准备 ====================
def prepare_topk_data(interaction_df):
    """准备Top-k推荐任务数据"""
    print('\n========== 准备Top-k推荐任务数据 ==========')
    os.makedirs(TOPK_PATH, exist_ok=True)
    
    # 仅保留正样本
    interaction_pos = interaction_df.loc[interaction_df.label == 1].copy()
    interaction_pos.rename(columns={
        'hour': 'c_hour_c',
        'weekday': 'c_weekday_c',
        'period': 'c_period_c',
        'day': 'c_day_f',
        'user_id': 'original_user_id'
    }, inplace=True)
    
    # 分割训练、验证、测试集
    split_time1 = int(interaction_pos.c_day_f.max() * 0.8)
    train = interaction_pos.loc[interaction_pos.c_day_f <= split_time1].copy()
    val_test = interaction_pos.loc[(interaction_pos.c_day_f > split_time1)].copy()
    val_test.sort_values(by='time', inplace=True)
    split_time2 = int(interaction_pos.c_day_f.max() * 0.9)
    val = val_test.loc[val_test.c_day_f <= split_time2].copy()
    test = val_test.loc[val_test.c_day_f > split_time2].copy()

    # 删除验证/测试集中不在训练集的用户和项目
    train_u, train_i = set(train.original_user_id.unique()), set(train.news_id.unique())
    val_sel = val.loc[(val.original_user_id.isin(train_u)) & (val.news_id.isin(train_i))].copy()
    test_sel = test.loc[(test.original_user_id.isin(train_u)) & (test.news_id.isin(train_i))].copy()
    
    print(f"训练集 - 用户: {len(train_u)}, 项目: {len(train_i)}")
    print(f"验证集 - 用户: {val_sel.original_user_id.nunique()}, 项目: {val_sel.news_id.nunique()}")
    print(f"测试集 - 用户: {test_sel.original_user_id.nunique()}, 项目: {test_sel.news_id.nunique()}")
    
    # 重新编号
    all_df = pd.concat([train, val_sel, test_sel], axis=0)
    user2newid_topk = dict(zip(sorted(all_df.original_user_id.unique()),
                              range(1, all_df.original_user_id.nunique() + 1)))
    for df in [train, val_sel, test_sel]:
        df['user_id'] = df.original_user_id.apply(lambda x: user2newid_topk[x])

    item2newid_topk = dict(zip(sorted(all_df.news_id.unique()),
                              range(1, all_df.news_id.nunique() + 1)))
    for df in [train, val_sel, test_sel]:
        df['item_id'] = df['news_id'].apply(lambda x: item2newid_topk[x])

    all_df['user_id'] = all_df.original_user_id.apply(lambda x: user2newid_topk[x])
    all_df['item_id'] = all_df['news_id'].apply(lambda x: item2newid_topk[x])
    
    # 保存映射
    nu2nid = {int(k): v for k, v in user2newid_topk.items()}
    ni2nid = {int(k): v for k, v in item2newid_topk.items()}
    json.dump(nu2nid, open(os.path.join(TOPK_PATH, "user2newid.json"), 'w'))
    json.dump(ni2nid, open(os.path.join(TOPK_PATH, "item2newid.json"), 'w'))
    
    # 生成负样本
    print('正在生成负样本...')
    def generate_negative(data_df, all_items, clicked_item_set, random_seed, neg_item_num=99):
        np.random.seed(random_seed)
        neg_items = np.random.choice(all_items, (len(data_df), neg_item_num))
        for i, uid in tqdm(enumerate(data_df['user_id'].values)):
            user_clicked = clicked_item_set[uid]
            for j in range(len(neg_items[i])):
                while neg_items[i][j] in user_clicked | set(neg_items[i][:j]):
                    neg_items[i][j] = np.random.choice(all_items, 1)
        return neg_items.tolist()

    clicked_item_set = dict()
    for user_id, seq_df in all_df.groupby('user_id'):
        clicked_item_set[user_id] = set(seq_df['item_id'].values.tolist())
    all_items = all_df.item_id.unique()
    val_sel['neg_items'] = generate_negative(val_sel, all_items, clicked_item_set, random_seed=1)
    test_sel['neg_items'] = generate_negative(test_sel, all_items, clicked_item_set, random_seed=2)
    
    # 保存数据
    select_columns = ['user_id', 'item_id', 'time', 'c_hour_c', 'c_weekday_c', 'c_period_c', 'c_day_f']
    train[select_columns].to_csv(os.path.join(TOPK_PATH, 'train.csv'), sep="\t", index=False)
    val_sel[select_columns + ['neg_items']].to_csv(os.path.join(TOPK_PATH, 'dev.csv'), sep="\t", index=False)
    test_sel[select_columns + ['neg_items']].to_csv(os.path.join(TOPK_PATH, 'test.csv'), sep="\t", index=False)
    
    # 保存项目元数据
    item_meta = pd.read_csv(os.path.join(RAW_PATH, DATASET, "movies.dat"),
                sep='::', names=['movieId', 'title', 'genres'],
                encoding='latin-1', engine='python')
    item_select = item_meta.loc[item_meta.movieId.isin(interaction_pos.news_id.unique())].copy()
    item_select['item_id'] = item_select.movieId.apply(lambda x: item2newid_topk[x])
    genres2id = dict(zip(sorted(item_select.genres.unique()),
                        range(1, item_select.genres.nunique() + 1)))
    item_select['i_genre_c'] = item_select['genres'].apply(lambda x: genres2id[x])
    title2id = dict(zip(sorted(item_select.title.unique()),
                       range(1, item_select.title.nunique() + 1)))
    item_select['i_title_c'] = item_select['title'].apply(lambda x: title2id[x])
    item_select[['item_id', 'i_genre_c', 'i_title_c']].to_csv(
        os.path.join(TOPK_PATH, 'item_meta.csv'), sep="\t", index=False)
    
    print(f'Top-k数据已保存到 {TOPK_PATH}')

# ==================== 主程序 ====================
def main():
    print('='*50)
    print('MovieLens-1M 数据集处理')
    print('='*50)
    
    # 下载数据
    if not download_data():
        print('请先下载数据集: http://files.grouplens.org/datasets/movielens/ml-1m.zip')
        print('并解压到 ' + RAW_PATH)
        return
    
    # 加载交互数据
    interactions, user_freq, item_freq = load_interactions()
    if interactions is None:
        return
    
    # 5-core过滤
    interactions, select_uid, select_iid = filter_5core(interactions, user_freq, item_freq)
    
    # 创建DataFrame
    interaction_df = create_interaction_df(interactions)
    
    # 准备CTR任务数据
    prepare_ctr_data(interaction_df)
    
    # 准备Top-k任务数据
    prepare_topk_data(interaction_df)
    
    print('\n'+'='*50)
    print('处理完成！')
    print('='*50)

if __name__ == '__main__':
    main()
