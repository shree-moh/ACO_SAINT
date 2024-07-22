import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class simple_MLP(nn.Module):
    def __init__(self, layers):
        super(simple_MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                self.layers.append(nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class sep_MLP(nn.Module):
    def __init__(self, dim, len_feats, categories):
        super(sep_MLP, self).__init__()
        self.len_feats = len_feats
        self.layers = nn.ModuleList([simple_MLP([dim, 5 * dim, categories[i]]) for i in range(len_feats)])

    def forward(self, x):
        y_pred = []
        for i in range(self.len_feats):
            x_i = x[:, i, :]
            pred = self.layers[i](x_i)
            y_pred.append(pred)
        return y_pred

class SAINT(nn.Module):
    def __init__(
            self,
            *,
            categories,
            num_continuous,
            dim,
            depth,
            heads,
            dim_head=16,
            dim_out=1,
            mlp_hidden_mults=(4, 2),
            mlp_act=None,
            num_special_tokens=0,
            attn_dropout=0.,
            ff_dropout=0.,
            cont_embeddings='MLP',
            scalingfactor=10,
            attentiontype='col',
            final_mlp_style='common',
            y_dim=2
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        self.num_special_tokens = num_special_tokens
        self.total_tokens = self.num_unique_categories + self.num_special_tokens

        categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value=num_special_tokens)
        categories_offset = categories_offset.cumsum(dim=-1)[:-1]
        self.register_buffer('categories_offset', categories_offset)

        self.num_continuous = num_continuous
        self.dim = dim
        self.cont_embeddings = cont_embeddings
        self.final_mlp_style = final_mlp_style

        if self.cont_embeddings == 'MLP':
            self.simple_MLP = nn.ModuleList([simple_MLP([1, 100, self.dim]) for _ in range(self.num_continuous)])
            input_size = (dim * self.num_categories) + (dim * num_continuous)
            nfeats = self.num_categories + num_continuous
        elif self.cont_embeddings == 'pos_singleMLP':
            self.simple_MLP = nn.ModuleList([simple_MLP([1, 100, self.dim]) for _ in range(1)])
            input_size = (dim * self.num_categories) + (dim * num_continuous)
            nfeats = self.num_categories + num_continuous
        else:
            print('Continuous features are not passed through attention')
            input_size = (dim * self.num_categories) + num_continuous
            nfeats = self.num_categories

        # Use simple embedding and linear layers instead of transformer
        self.embedding = nn.Embedding(self.total_tokens, self.dim)
        self.fc = nn.Linear(input_size, dim)

        l = input_size // 8
        hidden_dimensions = list(map(lambda t: l * t, mlp_hidden_mults))
        all_dimensions = [dim, *hidden_dimensions, dim_out]
        self.mlp = simple_MLP(all_dimensions)

        self.embeds = nn.Embedding(self.total_tokens, self.dim)

        cat_mask_offset = F.pad(torch.Tensor(self.num_categories).fill_(2).type(torch.int8), (1, 0), value=0)
        cat_mask_offset = cat_mask_offset.cumsum(dim=-1)[:-1]
        self.register_buffer('cat_mask_offset', cat_mask_offset)

        con_mask_offset = F.pad(torch.Tensor(self.num_continuous).fill_(2).type(torch.int8), (1, 0), value=0)
        con_mask_offset = con_mask_offset.cumsum(dim=-1)[:-1]
        self.register_buffer('con_mask_offset', con_mask_offset)

        print(f"cat_mask_offset shape: {self.cat_mask_offset.shape}")
        print(f"con_mask_offset shape: {self.con_mask_offset.shape}")

        self.mask_embeds_cat = nn.Embedding(self.num_categories * 2, self.dim)
        self.mask_embeds_cont = nn.Embedding(self.num_continuous * 2, self.dim)
        self.single_mask = nn.Embedding(2, self.dim)
        self.pos_encodings = nn.Embedding(self.num_categories + self.num_continuous, self.dim)

        if self.final_mlp_style == 'common':
            self.mlp1 = simple_MLP([dim, (self.total_tokens) * 2, self.total_tokens])
            self.mlp2 = simple_MLP([dim, (self.num_continuous), 1])
        else:
            self.mlp1 = sep_MLP(dim, self.num_categories, categories)
            self.mlp2 = sep_MLP(dim, self.num_continuous, np.ones(self.num_continuous).astype(int))

        self.mlpfory = simple_MLP([dim, 1000, y_dim])
        self.pt_mlp = simple_MLP([dim * (self.num_continuous + self.num_categories),
                                  6 * dim * (self.num_continuous + self.num_categories) // 5,
                                  dim * (self.num_continuous + self.num_categories) // 2])
        self.pt_mlp2 = simple_MLP([dim * (self.num_continuous + self.num_categories),
                                   6 * dim * (self.num_continuous + self.num_categories) // 5,
                                   dim * (self.num_continuous + self.num_categories) // 2])

    def forward(self, x_categ, x_cont):
        x_categ_emb = self.embedding(x_categ)
        x_cont_transformed = torch.cat([self.simple_MLP[i](x_cont[:, i:i+1]) for i in range(self.num_continuous)], dim=-1)
        x_combined = torch.cat([x_categ_emb, x_cont_transformed], dim=-1)
        x_combined = self.fc(x_combined)  # Make sure this dimension matches the input to self.mlp1 and self.mlp2
        cat_outs = self.mlp1(x_combined[:, :self.num_categories])
        con_outs = self.mlp2(x_combined[:, self.num_categories:])
        return cat_outs, con_outs
