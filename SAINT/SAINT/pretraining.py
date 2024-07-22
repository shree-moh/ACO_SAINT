import torch
from torch import nn

from torch.utils.data import DataLoader,TensorDataset
import torch.optim as optim
from augmentations import embed_data_mask


def SAINT_pretrain(model, X_train, y_train, opt, device):

    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train)

    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    trainloader = DataLoader(train_ds, batch_size=opt.batchsize , shuffle=True, num_workers=0, pin_memory=False)
    vision_dset = opt.vision_dset
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    pt_aug_dict = {
        'noise_type': opt.pt_aug,
        'lambda': opt.pt_aug_lam
    }
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()

    print("Pretraining begins!")
    for epoch in range(opt.pretrain_epochs):
        model.train()

        running_loss = 0.0
        for i, data in enumerate(trainloader):
            optimizer.zero_grad()

            x_input, y_label = data  # 수정: 데이터 튜플에서 입력과 레이블 추출

            x_cont = x_input.to(device)  # 모든 입력 데이터를 연속형 변수로 간주하여 하나의 텐서로 처리합니다.
            x_cont = x_cont.float()  # 데이터 유형을 float32로 변환

            cat_mask, con_mask = None, None  # cat_mask와 con_mask가 없는 경우

            # embed_data_mask 함수는 연속형 데이터만을 다룹니다.
            if 'cutmix' in opt.pt_aug:
                from augmentations import add_noise
                x_cont_corr = add_noise(x_cont, noise_params=pt_aug_dict)
                _, _, x_cont_enc_2 = embed_data_mask(x_cont_corr, con_mask, model, vision_dset)
            else:
                x_cont_enc_2 = embed_data_mask(x_cont, model, vision_dset)

            if 'mixup' in opt.pt_aug:
                from augmentations import mixup_data
                x_cont_enc_2 = mixup_data(x_cont_enc_2, lam=opt.mixup_lam)  # 연속형 변수에 대한 mixup을 수행합니다.

            loss = 0  # 연속형 변수만 다루므로 loss 초기화만 필요합니다.

            if 'contrastive' in opt.pt_tasks:
                aug_features_1 = model.transformer(x_cont_enc_2)
                aug_features_2 = model.transformer(x_cont_enc_2)
                aug_features_1 = (aug_features_1 / aug_features_1.norm(dim=-1, keepdim=True)).flatten(1, 2)
                aug_features_2 = (aug_features_2 / aug_features_2.norm(dim=-1, keepdim=True)).flatten(1, 2)
                if opt.pt_projhead_style == 'diff':
                    aug_features_1 = model.pt_mlp(aug_features_1)
                    aug_features_2 = model.pt_mlp2(aug_features_2)
                elif opt.pt_projhead_style == 'same':
                    aug_features_1 = model.pt_mlp(aug_features_1)
                    aug_features_2 = model.pt_mlp(aug_features_2)
                else:
                    print('Not using projection head')
                logits_per_aug1 = aug_features_1 @ aug_features_2.t() / opt.nce_temp
                logits_per_aug2 = aug_features_2 @ aug_features_1.t() / opt.nce_temp
                targets = torch.arange(logits_per_aug1.size(0)).to(logits_per_aug1.device)
                loss_1 = criterion1(logits_per_aug1, targets)
                loss_2 = criterion1(logits_per_aug2, targets)
                loss = opt.lam0 * (loss_1 + loss_2) / 2
            elif 'contrastive_sim' in opt.pt_tasks:
                aug_features_1 = model.transformer(x_categ_enc, x_cont_enc)
                aug_features_2 = model.transformer(x_categ_enc_2, x_cont_enc_2)
                aug_features_1 = (aug_features_1 / aug_features_1.norm(dim=-1, keepdim=True)).flatten(1, 2)
                aug_features_2 = (aug_features_2 / aug_features_2.norm(dim=-1, keepdim=True)).flatten(1, 2)
                aug_features_1 = model.pt_mlp(aug_features_1)
                aug_features_2 = model.pt_mlp2(aug_features_2)
                c1 = aug_features_1 @ aug_features_2.t()
                loss += opt.lam1 * torch.diagonal(-1 * c1).add_(1).pow_(2).sum()

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch: {epoch}, Running Loss: {running_loss}')

    print('END OF PRETRAINING!')
    return model
    # if opt.active_log:
    #     wandb.log({'pt_epoch': epoch ,'pretrain_epoch_loss': running_loss
    #     })
