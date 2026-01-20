# import pandas as pd
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler
# import akshare as ak  # ✅ 引入 akshare


# # --- StockDataset 类保持完全不变 ---
# class StockDataset(Dataset):
#     def __init__(self, data_frames, window_size=30):
#         self.samples = []
#         self.labels_id = []    # Task 1: Stock ID
#         self.labels_price = [] # Task 2: Up/Down
        
#         feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        
#         for df in data_frames:
#             vals = df[feature_cols].values
#             ids = df['StockID'].values
#             prices = df['Target'].values
            
#             # Sliding window creation
#             for i in range(len(df) - window_size):
#                 window = vals[i : i+window_size] 
#                 stock_id = ids[i]
#                 price_target = prices[i + window_size - 1] 
                
#                 self.samples.append(window)
#                 self.labels_id.append(stock_id)
#                 self.labels_price.append(price_target)
                
#     def __len__(self):
#         return len(self.samples)
    
#     def __getitem__(self, idx):
#         return (torch.tensor(self.samples[idx], dtype=torch.float32),
#                 torch.tensor(self.labels_id[idx], dtype=torch.long),
#                 torch.tensor(self.labels_price[idx], dtype=torch.float32))

# # # --- 修改后的 get_dataloaders ---
# # def get_dataloaders(stock_list, window_size=30, batch_size=32):
# #     print(">>> Using AkShare (China-Friendly) to download Stock Data...")
    
# #     # 定义时间范围（AkShare下载的是全量历史，我们需要手动筛选）
# #     START_DATE = pd.to_datetime("2021-01-01")
# #     END_DATE = pd.to_datetime("2024-01-01")
    
# #     data_frames = []
    
# #     for stock_id, ticker in enumerate(stock_list):
# #         print(f"Downloading {ticker}...", end=" ")
# #         try:
# #             # ✅ 1. 使用 AkShare 下载美股日线数据
# #             # adjust="qfq" 表示前复权，相当于 yfinance 的 Adj Close 处理逻辑
# #             df = ak.stock_us_daily(symbol=ticker, adjust="qfq")
            
# #             # ✅ 2. 重命名列名
# #             # AkShare 返回的列名是小写的：date, open, high, low, close, volume
# #             # 我们需要改成大写首字母以匹配你的 Dataset 代码
# #             df = df.rename(columns={
# #                 'date': 'Date',
# #                 'open': 'Open',
# #                 'high': 'High',
# #                 'low': 'Low',
# #                 'close': 'Close',
# #                 'volume': 'Volume'
# #             })
            
# #             # ✅ 3. 日期筛选
# #             df['Date'] = pd.to_datetime(df['Date'])
# #             df = df[(df['Date'] >= START_DATE) & (df['Date'] <= END_DATE)]
            
# #             # 设置索引并排序，确保时间顺序正确
# #             df = df.set_index('Date').sort_index()
            
# #             # 确保数据不为空
# #             if len(df) == 0:
# #                 print("❌ No data found in range.")
# #                 continue
            
# #             # 只保留需要的列
# #             df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
            
# #             # --- 以下逻辑和原来保持一致 ---
            
# #             # Label 1: Tomorrow's Close > Today's Close ?
# #             df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
# #             # Label 2: Stock ID
# #             df['StockID'] = stock_id
            
# #             df = df.dropna()
            
# #             # Z-Score Normalization
# #             scaler = StandardScaler()
# #             cols = ['Open', 'High', 'Low', 'Close', 'Volume']
# #             df[cols] = scaler.fit_transform(df[cols])
            
# #             data_frames.append(df)
# #             print(f"✅ Success (Rows: {len(df)})")
            
# #         except Exception as e:
# #             print(f"❌ Failed: {e}")
# #             continue
        
# #     if not data_frames:
# #         raise ValueError("没有下载到任何数据！请检查网络或股票代码。")

# #     full_dataset = StockDataset(data_frames, window_size)
    
# #     # Split Train/Test
# #     train_size = int(0.8 * len(full_dataset))
# #     test_size = len(full_dataset) - train_size
# #     train_ds, test_ds = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
# #     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
# #     test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
# #     print(f">>> Data Ready: Train[{len(train_ds)}] / Test[{len(test_ds)}]")
# #     return train_loader, test_loader


# def get_dataloaders(stock_list, window_size=30, batch_size=32):
#     print(">>> Using AkShare (China-Friendly) to download Stock Data...")
    
#     # 定义全局时间范围
#     START_DATE = pd.to_datetime("2015-01-01") # 建议拉长一点时间，数据多一点
#     END_DATE = pd.to_datetime("2024-01-01")
    
#     # 用来暂存切分后的数据
#     train_dfs = []
#     test_dfs = []
    
#     for stock_id, ticker in enumerate(stock_list):
#         print(f"Processing {ticker}...", end=" ")
#         try:
#             # 1. 下载
#             df = ak.stock_us_daily(symbol=ticker, adjust="qfq")
            
#             # 2. 重命名
#             df = df.rename(columns={
#                 'date': 'Date', 'open': 'Open', 'high': 'High',
#                 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
#             })
            
#             # 3. 日期处理
#             df['Date'] = pd.to_datetime(df['Date'])
#             df = df.sort_values('Date') # 确保绝对按时间排序
#             df = df[(df['Date'] >= START_DATE) & (df['Date'] <= END_DATE)]
#             df = df.set_index('Date')
            
#             if len(df) < 200: # 如果数据太少就跳过
#                 print(f"❌ Too few data ({len(df)} rows)")
#                 continue

#             # 4. 构造 Label 和 StockID
#             df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
#             df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
#             df['StockID'] = stock_id
#             df = df.dropna()
            
#             # =========== 关键修改：按时间切分 ===========
#             split_idx = int(len(df) * 0.8) # 80% 时间点
            
#             train_df = df.iloc[:split_idx].copy()
#             test_df = df.iloc[split_idx:].copy()
            
#             # =========== 关键修改：标准化不泄露 ===========
#             cols = ['Open', 'High', 'Low', 'Close', 'Volume']
#             scaler = StandardScaler()
            
#             # 仅在训练集上 fit (学习均值方差)
#             train_df[cols] = scaler.fit_transform(train_df[cols])
            
#             # 在测试集上仅 transform (使用训练集的标准)
#             test_df[cols] = scaler.transform(test_df[cols])
            
#             train_dfs.append(train_df)
#             test_dfs.append(test_df)
            
#             print(f"✅ (Train: {len(train_df)}, Test: {len(test_df)})")
            
#         except Exception as e:
#             print(f"❌ Error: {e}")
#             continue
    
#     if not train_dfs:
#         raise ValueError("No data loaded!")

#     # 5. 构造 Dataset
#     # 这里我们不再在这个层面上 split，而是直接用切好的 DF 创建两个 Dataset
#     train_dataset = StockDataset(train_dfs, window_size)
#     test_dataset = StockDataset(test_dfs, window_size)
    
#     # 6. DataLoader
#     # 训练集可以打乱 (shuffle=True)，让模型学得更泛化
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
#     # 测试集不要打乱 (shuffle=False)，方便观察时间轴表现（虽然对准确率计算没影响）
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
#     print(f">>> Data Ready: Train Samples [{len(train_dataset)}] / Test Samples [{len(test_dataset)}]")
#     return train_loader, test_loader
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import akshare as ak

# --- StockDataset 类 (保持不变) ---
class StockDataset(Dataset):
    def __init__(self, data_frames, window_size=30):
        self.samples = []
        self.labels_id = []    
        self.labels_price = [] 
        
        feature_cols = ['Open_pct', 'High_pct', 'Low_pct', 'Close_pct', 'Volume_log']
        
        for df in data_frames:
            if not all(col in df.columns for col in feature_cols):
                continue
            
            # 转换为 float32 节省显存
            vals = df[feature_cols].values.astype(np.float32)
            ids = df['StockID'].values
            prices = df['Target'].values
            
            for i in range(len(df) - window_size):
                window = vals[i : i+window_size] 
                stock_id = ids[i + window_size - 1] 
                price_target = prices[i + window_size - 1] 
                
                self.samples.append(window)
                self.labels_id.append(stock_id)
                self.labels_price.append(price_target)
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return (torch.tensor(self.samples[idx], dtype=torch.float32),
                torch.tensor(self.labels_id[idx], dtype=torch.long),
                torch.tensor(self.labels_price[idx], dtype=torch.float32))

def get_dataloaders(stock_list, window_size=30, batch_size=32):
    print(f">>> [Data] Initializing download for {len(stock_list)} stocks...")
    print(">>> [Data] Feature Engineering: Percentage Change + Log Volume")
    
    START_DATE = pd.to_datetime("2016-01-01")
    END_DATE = pd.to_datetime("2024-01-01")
    
    train_dfs = []
    test_dfs = []
    
    # 进度计数
    success_count = 0
    
    for stock_id, ticker in enumerate(stock_list):
        try:
            # 使用 AkShare 下载美股
            df = ak.stock_us_daily(symbol=ticker, adjust="qfq")
            df = df.rename(columns={
                'date': 'Date', 'open': 'Open', 'high': 'High',
                'low': 'Low', 'close': 'Close', 'volume': 'Volume'
            })
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            df = df[(df['Date'] >= START_DATE) & (df['Date'] <= END_DATE)]
            df = df.set_index('Date')
            
            # 数据太少则跳过
            if len(df) < 300: continue

            # === 特征工程 ===
            # 1. 收益率 (解决不平稳性)
            df['Close_pct'] = df['Close'].pct_change()
            df['Open_pct'] = df['Open'] / df['Close'].shift(1) - 1
            df['High_pct'] = df['High'] / df['Close'].shift(1) - 1
            df['Low_pct'] = df['Low'] / df['Close'].shift(1) - 1
            # 2. 对数成交量
            df['Volume_log'] = np.log1p(df['Volume'])
            
            # 3. 标签
            df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
            df['StockID'] = stock_id
            
            df = df.dropna()
            
            # === 时间切分 ===
            split_idx = int(len(df) * 0.8)
            train_df = df.iloc[:split_idx].copy()
            test_df = df.iloc[split_idx:].copy()
            
            # === 防泄露标准化 ===
            cols = ['Open_pct', 'High_pct', 'Low_pct', 'Close_pct', 'Volume_log']
            scaler = StandardScaler()
            
            if len(train_df) > 50 and len(test_df) > 50:
                train_df[cols] = scaler.fit_transform(train_df[cols])
                test_df[cols] = scaler.transform(test_df[cols])
                
                train_dfs.append(train_df)
                test_dfs.append(test_df)
                success_count += 1
                
            # 简单打印进度，不刷屏
            if success_count % 10 == 0:
                print(f"   ... Processed {success_count} stocks")
            
        except Exception as e:
            # 忽略下载失败的个股，保证程序不崩
            continue
    
    print(f">>> [Data] Successfully loaded {success_count}/{len(stock_list)} stocks.")
    
    if not train_dfs: 
        raise ValueError("No data loaded! Check your network connection.")

    train_dataset = StockDataset(train_dfs, window_size)
    test_dataset = StockDataset(test_dfs, window_size)
    
    # 显存优化：num_workers=0 (Windows兼容性), pin_memory=True
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, pin_memory=True)
    
    return train_loader, test_loader