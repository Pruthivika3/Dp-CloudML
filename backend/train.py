import os
import time
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from opacus import PrivacyEngine
import psutil

SEED = 42
MAX_ROWS = 50000
MAX_FEATURES = 512
DEVICE = torch.device("cpu")
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)


def load_dataframe(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError("Only CSV and Excel (.xlsx/.xls) datasets are supported")
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=SEED)
    if df.empty:
        raise ValueError("The uploaded dataset is empty")
    return df


def prepare_data(path, label_column, batch_size=256):
    df = load_dataframe(path)
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' was not found")
    df = df.dropna(subset=[label_column]).copy()
    y_raw = df.pop(label_column)
    if y_raw.nunique(dropna=True) < 2:
        raise ValueError("The label column must contain at least 2 classes")
    if y_raw.nunique() > 100:
        raise ValueError("This dashboard currently supports classification targets with at most 100 classes")

    # Remove obviously non-feature identifier columns only when they are unique strings.
    for col in list(df.columns):
        if df[col].dtype == "object" and df[col].nunique(dropna=False) >= 0.98 * len(df):
            df = df.drop(columns=[col])
    if df.shape[1] == 0:
        raise ValueError("No usable feature columns remain after preprocessing")

    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in df.columns if c not in numeric]
    parts = []
    if numeric:
        num = df[numeric].apply(pd.to_numeric, errors="coerce")
        num = num.fillna(num.median(numeric_only=True)).fillna(0)
        parts.append(num.to_numpy(dtype=np.float32))
    if categorical:
        cat = df[categorical].fillna("<MISSING>").astype(str)
        encoded = pd.get_dummies(cat, dummy_na=False, dtype=np.float32)
        parts.append(encoded.to_numpy(dtype=np.float32))
    X = np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
    if X.shape[1] > MAX_FEATURES:
        raise ValueError(f"Preprocessed feature count {X.shape[1]} exceeds the {MAX_FEATURES} feature limit")

    le = LabelEncoder()
    y = le.fit_transform(y_raw.astype(str)).astype(np.int64)
    stratify = y if np.min(np.bincount(y)) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=stratify
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    train_loader = DataLoader(train_ds, batch_size=min(batch_size, len(train_ds)), shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=min(batch_size, len(test_ds)), shuffle=False)
    return train_loader, test_loader, len(train_ds), X.shape[1], len(le.classes_), list(le.classes_)


class TabularMLP(nn.Module):
    def __init__(self, input_dim, classes):
        super().__init__()
        hidden1 = min(256, max(64, input_dim * 2))
        hidden2 = min(128, max(32, input_dim))
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(hidden1, hidden2), nn.ReLU(),
            nn.Linear(hidden2, classes),
        )
    def forward(self, x):
        return self.net(x)


def evaluate(model, loader):
    model.eval(); y_true=[]; y_pred=[]; losses=[]; loss_fn=nn.CrossEntropyLoss()
    with torch.no_grad():
        for x,y in loader:
            logits=model(x.to(DEVICE)); loss=loss_fn(logits,y.to(DEVICE)); losses.append(float(loss))
            y_true.extend(y.numpy().tolist()); y_pred.extend(torch.argmax(logits,1).cpu().numpy().tolist())
    accuracy=accuracy_score(y_true,y_pred)
    precision,recall,f1,_=precision_recall_fscore_support(y_true,y_pred,average="weighted",zero_division=0)
    return {"accuracy":round(float(accuracy),6),"precision":round(float(precision),6),"recall":round(float(recall),6),"f1_score":round(float(f1),6),"test_loss":round(float(np.mean(losses)),6)}


def _resource_snapshot():
    p=psutil.Process(os.getpid())
    return {"cpu_percent":float(p.cpu_percent(interval=0.02)),"memory_mb":round(p.memory_info().rss/(1024*1024),2)}


def train_model(path, label_column, dataset_name, mode="baseline", epochs=3, batch_size=256, lr=0.01,
                noise_multiplier=1.0, max_grad_norm=1.0, delta=1e-5):
    if mode not in {"baseline","dp"}: raise ValueError("mode must be 'baseline' or 'dp'")
    if epochs < 1 or epochs > 20: raise ValueError("epochs must be between 1 and 20")
    if batch_size < 8 or batch_size > 1024: raise ValueError("batch_size must be between 8 and 1024")
    if mode == "dp" and noise_multiplier <= 0: raise ValueError("noise_multiplier must be > 0 for DP")
    if max_grad_norm <= 0: raise ValueError("max_grad_norm must be > 0")
    if not 0 < delta < 1: raise ValueError("delta must be between 0 and 1")

    train_loader,test_loader,dataset_size,input_dim,classes,class_names=prepare_data(path,label_column,batch_size)
    model=TabularMLP(input_dim,classes).to(DEVICE)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    loss_fn=nn.CrossEntropyLoss()
    privacy_engine=None
    if mode == "dp":
        privacy_engine=PrivacyEngine()
        model,optimizer,train_loader=privacy_engine.make_private(
            module=model,optimizer=optimizer,data_loader=train_loader,
            noise_multiplier=float(noise_multiplier),max_grad_norm=float(max_grad_norm))

    start=time.perf_counter(); process_start=_resource_snapshot(); epoch_losses=[]
    for _ in range(epochs):
        model.train(); running=0.0; count=0
        for x,y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss=loss_fn(model(x.to(DEVICE)),y.to(DEVICE)); loss.backward(); optimizer.step()
            running += float(loss); count += 1
        epoch_losses.append(round(running/max(count,1),6))
    training_time=time.perf_counter()-start; process_end=_resource_snapshot()
    metrics=evaluate(model,test_loader)
    epsilon=None
    if privacy_engine is not None:
        epsilon=float(privacy_engine.accountant.get_epsilon(delta=float(delta)))
    return {
        "mode":mode,"dataset":dataset_name,"label_column":label_column,"dataset_size":dataset_size,
        "feature_count":input_dim,"class_count":classes,"classes":class_names[:100],"model":"Adaptive Tabular MLP",
        "epochs":epochs,"batch_size":batch_size,"learning_rate":lr,
        "noise_multiplier":0.0 if mode=="baseline" else float(noise_multiplier),
        "max_grad_norm":None if mode=="baseline" else float(max_grad_norm),
        "delta":None if mode=="baseline" else float(delta),
        "epsilon":None if epsilon is None else round(epsilon,6),
        "training_time_seconds":round(float(training_time),4),
        "cpu_percent_sample":round(max(process_start["cpu_percent"],process_end["cpu_percent"]),2),
        "memory_mb":round(max(process_start["memory_mb"],process_end["memory_mb"]),2),
        "final_train_loss":epoch_losses[-1] if epoch_losses else None,"epoch_losses":epoch_losses,**metrics
    }
