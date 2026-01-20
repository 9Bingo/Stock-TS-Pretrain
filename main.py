# import torch
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt
# from tqdm import tqdm

# # Import our modules
# from model import TransformerStockPrediction
# from data_loader import get_dataloaders

# # --- Configuration ---
# # STOCKS = ['AAPL', 'GOOG', 'MSFT', 'AMZN', 'TSLA', 'AMD', 'NVDA', 'INTC'] # More stocks = harder ID task
# STOCKS = [
#     'AAPL', 'MSFT', 'GOOG', 'AMZN', 'META',
#     'NVDA', 'AMD', 'INTC', 'QCOM', 'TSM',
#     'TSLA', 'F', 'GM', 'NFLX', 'ADBE',
#     'ORCL', 'IBM', 'CSCO', 'CRM', 'UBER',
#     'JPM', 'BAC', 'GS', 'MS', 'C',
#     'WMT', 'COST', 'TGT', 'HD', 'LOW'
# ]
# WINDOW_SIZE = 30
# BATCH_SIZE = 64
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def train_epoch(model, loader, optimizer, criterion, task_name):
#     model.train()
#     total_loss, correct, total = 0, 0, 0
    
#     # Set the model's task pointer (Matches the repo's logic)
#     model.pretrain_task = task_name 
    
#     for inputs, stock_ids, price_targets in loader:
#         inputs = inputs.to(DEVICE)
#         stock_ids = stock_ids.to(DEVICE)
#         price_targets = price_targets.to(DEVICE).unsqueeze(1)
        
#         optimizer.zero_grad()
        
#         if task_name == 'stock_id':
#             # Pre-training: Predict Stock ID
#             outputs = model(inputs) 
#             loss = criterion(outputs, stock_ids)
#             _, predicted = torch.max(outputs.data, 1)
#             correct += (predicted == stock_ids).sum().item()
#             total += stock_ids.size(0)
            
#         else: # task_name == ''
#             # Fine-tuning: Predict Price Direction
#             outputs = model(inputs)
#             loss = criterion(outputs, price_targets)
#             predicted = (torch.sigmoid(outputs) > 0.5).float()
#             correct += (predicted == price_targets).sum().item()
#             total += price_targets.size(0)
            
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
        
#     return total_loss / len(loader), correct / total

# def evaluate(model, loader, task_name):
#     model.eval()
#     correct, total = 0, 0
#     model.pretrain_task = task_name
    
#     with torch.no_grad():
#         for inputs, stock_ids, price_targets in loader:
#             inputs = inputs.to(DEVICE)
#             stock_ids = stock_ids.to(DEVICE)
#             price_targets = price_targets.to(DEVICE).unsqueeze(1)
            
#             if task_name == 'stock_id':
#                 outputs = model(inputs)
#                 _, predicted = torch.max(outputs.data, 1)
#                 correct += (predicted == stock_ids).sum().item()
#             else:
#                 outputs = model(inputs)
#                 predicted = (torch.sigmoid(outputs) > 0.5).float()
#                 correct += (predicted == price_targets).sum().item()
#             total += stock_ids.size(0) # or price_targets.size(0)
            
#     return correct / total

# def main():
#     print(f"Using Device: {DEVICE}")
    
#     # 1. Prepare Data
#     train_loader, test_loader = get_dataloaders(STOCKS, WINDOW_SIZE, BATCH_SIZE)
    
#     # 2. Initialize Model (Using the custom architecture)
#     # Note: 'num_class' here is for the default fine-tuning task (Price) -> 1 output
#     model = TransformerStockPrediction(
#         input_size=5, 
#         num_class=1,          # Output for price prediction (1 value)
#         hidden_size=64, 
#         num_feat_att_layers=1, # Layers in FeatExtractor
#         num_pre_att_layers=1,  # Layers in Upper Transformer
#         num_heads=4, 
#         days=WINDOW_SIZE,
#         dropout=0.1
#     ).to(DEVICE)
    
#     # 3. Add Pre-training Head Dynamically
#     # This adds a layer named 'stock_id' to pretrain_outlayers
#     model.add_outlayer('stock_id', num_class=len(STOCKS), device=DEVICE)
    
#     # --- Experiment A: Baseline (No Pre-training) ---
#     # We will clone the model state later for Experiment B, 
#     # but for A, we just train the default head immediately.
#     # To keep it fair, let's just instantiate two models or reset weights. 
#     # Here simpler: let's run Experiment B first (Pretrain -> Finetune), 
#     # then reset and run A.
    
#     # ==========================================
#     # Phase 1: Pre-training (Stock ID Classification)
#     # ==========================================
#     print("\n>>> Phase 1: Pre-training (Learning Stock Identity)...")
#     optimizer = optim.Adam(model.parameters(), lr=1e-3)
#     criterion = nn.CrossEntropyLoss()
    
#     pretrain_accs = []
#     for epoch in range(10):
#         loss, acc = train_epoch(model, train_loader, optimizer, criterion, 'stock_id')
#         print(f"Pretrain Epoch {epoch+1} | Loss: {loss:.4f} | ID Acc: {acc:.4f}")
#         pretrain_accs.append(acc)
        
#     print("Pre-training Done! Model has learned stock features.")
    
#     # Save pretrained weights to use in Experiment B comparison
#     torch.save(model.state_dict(), "pretrained_model.pth")
    
#     # ==========================================
#     # Phase 2: Fine-tuning (Price Prediction) - WITH Pre-training
#     # ==========================================
#     print("\n>>> Phase 2: Fine-tuning (With Pre-training)...")
    
#     # Optional: Freeze the bottom layers to show "transfer learning" effect
#     # model.change_finetune_mode(True, freezing='embedding') 
    
#     optimizer = optim.Adam(model.parameters(), lr=1e-3)
#     criterion = nn.BCEWithLogitsLoss() # More stable than Sigmoid + BCELoss
    
#     history_with_pretrain = []
#     for epoch in range(15):
#         # task_name='' means use default head (Price Prediction)
#         loss, acc = train_epoch(model, train_loader, optimizer, criterion, '') 
#         test_acc = evaluate(model, test_loader, '')
#         history_with_pretrain.append(test_acc)
#         print(f"Finetune Epoch {epoch+1} | Loss: {loss:.4f} | Test Acc: {test_acc:.4f}")
        
#     # ==========================================
#     # Phase 3: Baseline (Train from Scratch) - NO Pre-training
#     # ==========================================
#     print("\n>>> Phase 3: Baseline (No Pre-training)...")
#     # Re-initialize a fresh model
#     model_base = TransformerStockPrediction(
#         input_size=5, num_class=1, hidden_size=64, 
#         num_feat_att_layers=1, num_pre_att_layers=1, 
#         num_heads=4, days=WINDOW_SIZE
#     ).to(DEVICE)
    
#     optimizer = optim.Adam(model_base.parameters(), lr=1e-3)
#     criterion = nn.BCEWithLogitsLoss()
    
#     history_baseline = []
#     for epoch in range(15):
#         loss, acc = train_epoch(model_base, train_loader, optimizer, criterion, '')
#         test_acc = evaluate(model_base, test_loader, '')
#         history_baseline.append(test_acc)
#         print(f"Baseline Epoch {epoch+1} | Loss: {loss:.4f} | Test Acc: {test_acc:.4f}")

#     # ==========================================
#     # Plot Results
#     # ==========================================
#     plt.figure(figsize=(10, 5))
#     plt.plot(history_baseline, label='Baseline (From Scratch)', linestyle='--')
#     plt.plot(history_with_pretrain, label='Ours (After ID Pre-training)', marker='o')
#     plt.title('Impact of Stock ID Pre-training on Price Prediction')
#     plt.xlabel('Fine-tuning Epochs')
#     plt.ylabel('Test Accuracy')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('result_comparison.png')
#     plt.show()

# if __name__ == "__main__":
#     main()
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score
from copy import deepcopy

from model import TransformerStockPrediction
from data_loader import get_dataloaders

# --- 1. 大规模股票列表 (覆盖科技、消费、金融、医疗、能源) ---
# 这会证明模型具有普适性
BIG_STOCK_LIST = [
    # Tech Giants
    'AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX',
    # Semi-conductors
    'AMD', 'INTC', 'QCOM', 'TXN', 'AVGO', 'MU', 'LRCX', 'ADI',
    # Financials
    'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'AXP', 'V', 'MA',
    # Healthcare
    'JNJ', 'PFE', 'UNH', 'ABT', 'MRK', 'TMO', 'LLY', 'AMGN',
    # Consumer
    'WMT', 'PG', 'KO', 'PEP', 'COST', 'MCD', 'NKE', 'SBUX', 'TGT',
    # Industrial & Energy
    'XOM', 'CVX', 'GE', 'CAT', 'BA', 'UPS', 'MMM', 'HON',
    # Software & Cloud
    'CRM', 'ADBE', 'ORCL', 'IBM', 'NOW', 'INTU', 'SAP',
    # Others
    'DIS', 'CMCSA', 'VZ', 'T', 'TMUS', 'NKE', 'F', 'GM'
]

# Config
WINDOW_SIZE = 30
BATCH_SIZE = 128  # 增加Batch Size加快训练
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# --- 早停机制 ---
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# --- 全面的指标计算函数 ---
def calculate_metrics(targets, preds_prob, task_name):
    targets = np.array(targets)
    preds_prob = np.array(preds_prob)
    
    if task_name == 'stock_id':
        # 多分类任务
        preds_cls = np.argmax(preds_prob, axis=1)
        acc = accuracy_score(targets, preds_cls)
        f1 = f1_score(targets, preds_cls, average='macro')
        return {'Acc': acc, 'F1': f1}
    else:
        # 二分类任务 (Price Prediction)
        preds_cls = (preds_prob > 0.5).astype(int)
        
        acc = accuracy_score(targets, preds_cls)
        prec = precision_score(targets, preds_cls, zero_division=0)
        rec = recall_score(targets, preds_cls, zero_division=0)
        f1 = f1_score(targets, preds_cls, zero_division=0)
        
        # MCC: 金融分类最重要的指标 (加分项)
        mcc = matthews_corrcoef(targets, preds_cls)
        
        # AUC: 衡量排序能力
        try:
            auc = roc_auc_score(targets, preds_prob)
        except:
            auc = 0.5 # 异常处理
            
        return {
            'Acc': acc, 
            'Prec': prec, 
            'Rec': rec, 
            'F1': f1, 
            'MCC': mcc, 
            'AUC': auc
        }

def run_epoch(model, loader, optimizer, criterion, task_name, is_train=True):
    if is_train:
        model.train()
    else:
        model.eval()
        
    total_loss = 0
    all_targets = []
    all_probs = [] # 存储概率用于计算AUC
    
    model.pretrain_task = task_name
    
    with torch.set_grad_enabled(is_train):
        for inputs, stock_ids, price_targets in loader:
            inputs = inputs.to(DEVICE)
            stock_ids = stock_ids.to(DEVICE)
            price_targets = price_targets.to(DEVICE)
            
            outputs = model(inputs)
            
            if task_name == 'stock_id':
                loss = criterion(outputs, stock_ids)
                # Softmax概率
                probs = torch.softmax(outputs, dim=1)
                targets = stock_ids
            else: 
                loss = criterion(outputs, price_targets.unsqueeze(1))
                # Sigmoid概率
                probs = torch.sigmoid(outputs).squeeze()
                targets = price_targets
            
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
            total_loss += loss.item()
            
            # 收集数据到CPU
            all_targets.extend(targets.cpu().numpy())
            if task_name == 'stock_id':
                all_probs.extend(probs.detach().cpu().numpy())
            else:
                all_probs.extend(probs.detach().cpu().numpy())
            
    avg_loss = total_loss / len(loader)
    
    # 计算所有指标
    metrics = calculate_metrics(all_targets, all_probs, task_name)
    metrics['Loss'] = avg_loss
    
    return metrics

def main():
    print(f"Using Device: {DEVICE}")
    
    # 1. 准备数据 (大数据集)
    train_loader, test_loader = get_dataloaders(BIG_STOCK_LIST, WINDOW_SIZE, BATCH_SIZE)
    
    num_stocks = len(BIG_STOCK_LIST) # 注意：实际上可能因为下载失败少几个，这里只是给head用的维度
    # 稍微给大一点的维度给Head防止报错，或者在DataLoader里重新映射ID（为了简单，这里假设维度够大）
    # 严谨做法是 data_loader 返回真实的 unique stock count，这里简化处理直接用列表长度
    
    model_config = {
        'input_size': 5,
        'num_class': 1,
        'hidden_size': 64, 
        'num_feat_att_layers': 2, 
        'num_pre_att_layers': 2, 
        'num_heads': 4, 
        'days': WINDOW_SIZE,
        'dropout': 0.2
    }
    
    # ==========================================
    # 实验 A: 预训练 (Pre-train)
    # ==========================================
    print("\n>>> [Phase 1] Pre-training (Stock ID Classification)...")
    model = TransformerStockPrediction(**model_config).to(DEVICE)
    # 注意：这里的类别数给大一点，防止越界，或者在data_loader里做ID映射
    model.add_outlayer('stock_id', num_class=len(BIG_STOCK_LIST) + 5, device=DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    early_stop = EarlyStopping(patience=3) # 预训练不用太久
    
    for epoch in range(10): # 减少预训练轮数，节约时间
        train_m = run_epoch(model, train_loader, optimizer, criterion, 'stock_id', True)
        val_m = run_epoch(model, test_loader, None, criterion, 'stock_id', False)
        
        print(f"Pretrain Ep {epoch+1} | Loss: {train_m['Loss']:.4f} | Val Acc: {val_m['Acc']:.4f} F1: {val_m['F1']:.4f}")
        
        early_stop(val_m['Loss'])
        if early_stop.early_stop:
            print("Early stopping in Pre-training.")
            break
            
    pretrained_weights = deepcopy(model.state_dict())
    
    # ==========================================
    # 实验 A: 微调 (Fine-tune)
    # ==========================================
    print("\n>>> [Phase 2] Fine-tuning (Price Prediction) - Ours...")
    model.load_state_dict(pretrained_weights)
    model.change_finetune_mode(True, freezing='embedding')
    
    # optimizer = optim.AdamW(model.parameters(), lr=0.0005)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005)
    # 每 5 个 epoch，学习率减半
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)   

    criterion = nn.BCEWithLogitsLoss()
    
    history_ours = {'Acc': [], 'F1': [], 'MCC': [], 'AUC': []}
    
    for epoch in range(15):
        train_m = run_epoch(model, train_loader, optimizer, criterion, '', True)
        scheduler.step()
        val_m = run_epoch(model, test_loader, None, criterion, '', False)
        
        for k in history_ours: history_ours[k].append(val_m[k])
        
        print(f"Finetune Ep {epoch+1} | Val Acc: {val_m['Acc']:.4f} | F1: {val_m['F1']:.4f} | MCC: {val_m['MCC']:.4f} | AUC: {val_m['AUC']:.4f}")

    # ==========================================
    # 实验 B: 基线 (Baseline)
    # ==========================================
    print("\n>>> [Phase 3] Baseline (No Pre-training)...")
    set_seed(42)
    model_base = TransformerStockPrediction(**model_config).to(DEVICE)
    
    optimizer = optim.AdamW(model_base.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    history_base = {'Acc': [], 'F1': [], 'MCC': [], 'AUC': []}
    
    for epoch in range(15):
        train_m = run_epoch(model_base, train_loader, optimizer, criterion, '', True)
        val_m = run_epoch(model_base, test_loader, None, criterion, '', False)
        
        for k in history_base: history_base[k].append(val_m[k])
        
        print(f"Base Ep {epoch+1} | Val Acc: {val_m['Acc']:.4f} | F1: {val_m['F1']:.4f} | MCC: {val_m['MCC']:.4f} | AUC: {val_m['AUC']:.4f}")

    # ==========================================
    # 绘图 (多指标对比)
    # ==========================================
    epochs = range(1, 16)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics_to_plot = ['Acc', 'F1', 'MCC', 'AUC']
    
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i//2, i%2]
        ax.plot(epochs, history_base[metric], label='Baseline', linestyle='--', marker='x', alpha=0.6)
        ax.plot(epochs, history_ours[metric], label='Ours (Pre-trained)', marker='o', linewidth=2)
        ax.set_title(f'Comparison: {metric}')
        ax.set_xlabel('Epochs')
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True)
        
    plt.tight_layout()
    plt.savefig('comprehensive_analysis.png')
    print("\n>>> Analysis Complete. Saved plot to 'comprehensive_analysis.png'")
    print(f"Final MCC (Ours): {history_ours['MCC'][-1]:.4f} vs (Base): {history_base['MCC'][-1]:.4f}")

if __name__ == "__main__":
    main()