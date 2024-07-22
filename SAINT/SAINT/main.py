import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
import argparse
from augmentations import embed_data_mask
from pretraining import SAINT_pretrain
from pretrainmodel import SAINT
import wandb
from aco import AntColonyOptimizer
import time

# Define global variables
cat_dims = [10]  # Define according to your categorical dimensions
con_idxs = [0, 1, 3, 5]  # Define according to your continuous features
vision_dset = False  # This is a placeholder; set it based on your dataset

def calculate_metrics(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    PD = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
    PF = cm[0, 1] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    balance = 1 - np.sqrt((1 - PD)**2 + PF**2) / np.sqrt(2)
    FI = (cm[1, 1] + cm[0, 1]) / (cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1]) if (cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1]) > 0 else 0
    FIR = (PD - FI) / PD if PD > 0 else 0
    return PD, PF, balance, FIR

def classifier_eval(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    PD, PF, balance, FIR = calculate_metrics(y_test, y_pred)
    print('Confusion Matrix:', cm)
    print('Length of y_test:', len(y_test))
    print('Length of y_pred:', len(y_pred))
    print(f'PD: {PD}, PF: {PF}, Balance: {balance}, FIR: {FIR}')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vision_dset', action='store_true')
    parser.add_argument('--cont_embeddings', default='MLP', type=str, choices=['MLP', 'Noemb', 'pos_singleMLP'])
    parser.add_argument('--embedding_size', default=32, type=int)
    parser.add_argument('--transformer_depth', default=6, type=int)
    parser.add_argument('--attention_heads', default=8, type=int)
    parser.add_argument('--attention_dropout', default=0.1, type=float)
    parser.add_argument('--ff_dropout', default=0.1, type=float)
    parser.add_argument('--attentiontype', default='colrow', type=str, choices=['col', 'colrow', 'row', 'justmlp', 'attn', 'attnmlp'])
    parser.add_argument('--optimizer', default='AdamW', type=str, choices=['AdamW', 'Adam', 'SGD'])
    parser.add_argument('--scheduler', default='cosine', type=str, choices=['cosine', 'linear'])
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--epochs', default=5, type=int)  # Reduced from 50 to 5
    parser.add_argument('--batchsize', default=64, type=int)  # Increased batch size
    parser.add_argument('--savemodelroot', default='./bestmodels', type=str)
    parser.add_argument('--run_name', default='testrun', type=str)
    parser.add_argument('--set_seed', default=1, type=int)
    parser.add_argument('--dset_seed', default=5, type=int)
    parser.add_argument('--active_log', action='store_true')
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--pretrain_epochs', default=1, type=int)
    parser.add_argument('--pt_tasks', default=['contrastive', 'denoising'], type=str, nargs='*', choices=['contrastive', 'contrastive_sim', 'denoising'])
    parser.add_argument('--pt_aug', default=[], type=str, nargs='*', choices=['mixup', 'cutmix'])
    parser.add_argument('--pt_aug_lam', default=0.1, type=float)
    parser.add_argument('--mixup_lam', default=0.3, type=float)
    parser.add_argument('--train_mask_prob', default=0, type=float)
    parser.add_argument('--mask_prob', default=0, type=float)
    parser.add_argument('--ssl_avail_y', default=0, type=int)
    parser.add_argument('--pt_projhead_style', default='diff', type=str, choices=['diff', 'same', 'nohead'])
    parser.add_argument('--nce_temp', default=0.7, type=float)
    parser.add_argument('--lam0', default=0.5, type=float)
    parser.add_argument('--lam1', default=10, type=float)
    parser.add_argument('--lam2', default=1, type=float)
    parser.add_argument('--lam3', default=10, type=float)
    parser.add_argument('--final_mlp_style', default='sep', type=str, choices=['common', 'sep'])
    return parser.parse_args()

def objective_function(params):
    global vision_dset
    vision_dset = opt.vision_dset  
    # Update global options with the given parameters
    opt.transformer_depth = params['transformer_depth']
    opt.attention_heads = params['attention_heads']
    opt.lr = params['lr']
    opt.batchsize = params['batchsize']

    # Initialize model with given parameters
    model = SAINT(
        num_continuous=len(con_idxs),
        categories=[10],
        dim=8,
        depth=int(opt.transformer_depth),
        heads=int(opt.attention_heads),
        attn_dropout=opt.attention_dropout,
        ff_dropout=opt.ff_dropout,
        mlp_hidden_mults=(4, 2),
        cont_embeddings=opt.cont_embeddings,
        scalingfactor=10,
        attentiontype=opt.attentiontype,
        final_mlp_style=opt.final_mlp_style,
        y_dim=2
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=opt.lr)

    # Train the model
    model.train()
    for epoch in range(opt.epochs):
        start_time = time.time()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            optimizer.zero_grad()
            x_cont = data[0].to(device).float()
            y_gts = data[1].to(device)
            x_cont_enc = embed_data_mask(x_cont, model, vision_dset)
            reps = model.transformer(x_cont_enc)
            y_reps = reps[:, 0, :]
            y_outs = model.mlpfory(y_reps)
            y_probs = torch.nn.functional.softmax(y_outs, dim=1)
            loss = torch.nn.functional.cross_entropy(y_outs, y_gts)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_time = time.time() - start_time
        if epoch % 1 == 0:  # Print every epoch (reduced from 10 to 1)
            print(f"Epoch [{epoch}/{opt.epochs}], Loss: {running_loss:.4f}, Time: {epoch_time:.2f}s")

    # Evaluate the model
    model.eval()
    y_preds = []
    y_gts = []
    with torch.no_grad():
        for i, data in enumerate(valloader, 0):
            x_cont = data[0].to(device).float()
            y_gts.extend(data[1].tolist())
            x_cont_enc = embed_data_mask(x_cont, model, vision_dset)
            reps = model.transformer(x_cont_enc)
            y_reps = reps[:, 0, :]
            y_outs = model.mlpfory(y_reps)
            y_probs = torch.nn.functional.softmax(y_outs, dim=1)
            y_preds.extend(torch.argmax(y_probs, dim=1).tolist())

    y_test = torch.tensor(y_gts)
    y_pred = torch.tensor(y_preds)
    _, _, balance, _ = calculate_metrics(y_test, y_pred)
    return balance

if __name__ == '__main__':
    opt = get_args()
    # Load and preprocess data
    data = pd.read_csv("EQ.csv")

    # Print column names to verify
    print(data.columns)

    # Replace 'label' with 'class'
    if 'class' not in data.columns:
        print("The column 'class' is not found in the dataset. Please check the column names.")
    else:
        X = data.drop(['class'], axis=1)
        y = data['class']
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

    scaler = MinMaxScaler()
    X_test_normalized = scaler.fit_transform(X_test)

    kf = KFold(n_splits=3, shuffle=True, random_state=42)  # Reduced from 5 to 3 splits

    pd_list = []
    pf_list = []
    bal_list = []
    fir_list = []

    def setup_fold_data(train_index, val_index):
        X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]
        X_fold_train_normalized = scaler.fit_transform(X_fold_train)
        X_fold_val_normalized = scaler.transform(X_fold_val)
        smote = SMOTE(random_state=42)
        X_fold_train_resampled, y_fold_train_resampled = smote.fit_resample(X_fold_train_normalized, y_fold_train)
        X_train_tensor = torch.tensor(X_fold_train_resampled).float()
        y_train_tensor = torch.tensor(y_fold_train_resampled).long()
        train_ds = TensorDataset(X_train_tensor, y_train_tensor)
        trainloader = DataLoader(train_ds, batch_size=opt.batchsize, shuffle=True, num_workers=4)
        X_valid_tensor = torch.tensor(X_fold_val_normalized).float()
        y_valid_tensor = torch.tensor(y_fold_val.to_numpy()).long()
        val_ds = TensorDataset(X_valid_tensor, y_valid_tensor)
        valloader = DataLoader(val_ds, batch_size=opt.batchsize, shuffle=False, num_workers=4)
        return trainloader, valloader

    for train_index, val_index in kf.split(X_train):
        trainloader, valloader = setup_fold_data(train_index, val_index)
        search_space = {
            'transformer_depth': [4, 6, 8],
            'attention_heads': [4, 8, 12],
            'lr': [0.0001, 0.001, 0.01],
            'batchsize': [16, 32, 64]
        }

        aco = AntColonyOptimizer(
            objective_function=objective_function,
            search_space=search_space,
            n_ants=5,
            n_best=3,
            n_iterations=5,
            decay=0.95,
            alpha=1,
            beta=1
        )

        best_params, best_score = aco.optimize()
        print("Best Parameters:", best_params)
        print("Best Score:", best_score)

    avg_PD = np.mean(pd_list)
    avg_PF = np.mean(pf_list)
    avg_balance = np.mean(bal_list)
    avg_FIR = np.mean(fir_list)

    print('Average PD:', avg_PD)
    print('Average PF:', avg_PF)
    print('Average balance:', avg_balance)
    print('Average FIR:', avg_FIR)
