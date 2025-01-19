import torch
from einops import rearrange
from einops.layers.torch import Rearrange

from torch import nn, einsum
from torch.nn import AdaptiveAvgPool1d

from models.flamingo import GatedCrossAttentionBlock
from typing import Sequence


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


# pre-layernorm
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# aft-layernorm
class AftNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.norm(self.fn(x, **kwargs))


# feedforward
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# tabular-assisted feedforward
class T_Assisted_FeedForward(nn.Module):
    def __init__(self, img_dim, tabular_dim, hidden_dim):
        super().__init__()
        self.aux = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(img_dim + tabular_dim, hidden_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(hidden_dim, 2 * img_dim))
        )
        self.scale_activation = nn.Tanh()
        self.global_pool = nn.Sequential(Rearrange('b n d -> b d n'),
                                         AdaptiveAvgPool1d(1),
                                         Rearrange('b d n -> b (d n)'))
        self.img_dim = img_dim

    def forward(self, img, context=None):
        squeeze = self.global_pool(img)  # b, dim
        squeeze = torch.cat((squeeze, context), dim=1)  # b, d + d_t
        attention = self.aux(squeeze)  # b, 2d
        v_scale, v_shift = torch.split(attention, self.img_dim, dim=1)
        v_scale = v_scale.view(v_scale.size()[0], 1, v_scale.size()[1]).expand_as(img)
        # # activate to [-1,1]
        v_scale = self.scale_activation(v_scale)
        v_shift = v_shift.view(v_shift.size()[0], 1, v_shift.size()[1]).expand_as(img)
        return (v_scale * img) + v_shift


class IT_FeedForward(nn.Module):
    def __init__(self, img_dim, tabular_dim, hidden_dim, num_vectors):
        super().__init__()
        self.aux_tabular = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(tabular_dim, hidden_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(hidden_dim, img_dim))
        )
        self.aux = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(2 * img_dim, hidden_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(hidden_dim, 2 * num_vectors))
        )
        self.scale_activation = nn.Sigmoid()
        self.global_pool = nn.Sequential(Rearrange('b n d -> b d n'),
                                         AdaptiveAvgPool1d(1),
                                         Rearrange('b d n -> b (d n)'))
        self.img_dim = img_dim
        self.num_vectors = num_vectors

    def forward(self, img, context=None):
        # global average pooling for image features
        squeeze = self.global_pool(img)  # b, dim
        # feature transformation for tabular flattened features
        tabular_feat = self.aux_tabular(context) # context b, num_tabular * 4
        # get transformation parameters
        squeeze = torch.cat((squeeze, tabular_feat), dim=1)  # b, 2d
        attention = self.aux(squeeze)  # b, 2n
        v_scale, v_shift = torch.split(attention, self.num_vectors, dim=1) # b, n
        # expand to original img shape
        v_scale = v_scale.view(v_scale.size()[0], v_scale.size()[1], 1).expand_as(img)
        v_shift = v_shift.view(v_shift.size()[0], v_shift.size()[1], 1).expand_as(img)
        tabular_feat = tabular_feat.view(tabular_feat.size()[0], 1, tabular_feat.size()[1]).expand_as(img)
        # activate to [-1,1]
        v_scale = self.scale_activation(v_scale)
        # transform feature maps
        out = (v_scale * img) + ((1 - v_scale) * tabular_feat) + v_shift
        return out


class IT_FeedForward_Updated(nn.Module):
    def __init__(self, img_dim, tabular_dim, hidden_dim, num_vectors):
        super().__init__()
        self.aux_tabular = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(tabular_dim, hidden_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(hidden_dim, img_dim))
        )
        self.aux = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(9 * img_dim, hidden_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(hidden_dim, 2 * num_vectors))
        )
        self.scale_activation = nn.Sigmoid()
        self.reducion = nn.Sequential(nn.Conv1d(150, 8, 1),
                                      Rearrange('b n d -> b (d n)'))
        self.img_dim = img_dim
        self.num_vectors = num_vectors

    def forward(self, img, context):
        # global average pooling for image features
        squeeze = self.reducion(img)  # b, 16 * dim
        # feature transformation for tabular flattened features
        tabular_feat = self.aux_tabular(context) # context -> (b, num_tabular * 4)
        # get transformation parameters
        squeeze = torch.cat((squeeze, tabular_feat), dim=1)  # b, 2d
        attention = self.aux(squeeze)  # b, 2n
        v_scale, v_shift = torch.split(attention, self.num_vectors, dim=1) # b, n
        # expand to original img shape
        v_scale = v_scale.view(v_scale.size()[0], v_scale.size()[1], 1).expand_as(img)
        v_shift = v_shift.view(v_shift.size()[0], v_shift.size()[1], 1).expand_as(img)
        tabular_feat = tabular_feat.view(tabular_feat.size()[0], 1, tabular_feat.size()[1]).expand_as(img)
        # activate to [-1,1]
        v_scale = self.scale_activation(v_scale)
        # transform feature maps
        out = (v_scale * img) + ((1 - v_scale) * tabular_feat) + v_shift
        return out


# attention
class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, kv_include_self=False):
        b, n, _ = x.shape
        h = self.heads
        context = default(context, x)

        if kv_include_self:
            # cross attention requires CLS token includes itself as key / value
            context = torch.cat((x, context), dim=1)

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                AftNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                AftNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, context=None):
        for attn, ff in self.layers:
            x = attn(x, context=context) + x
            x = ff(x) + x
        return self.norm(x)


class Transformer_T_Assited(nn.Module):
    def __init__(self, dim, tabular_dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                AftNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                AftNorm(dim, T_Assisted_FeedForward(dim, tabular_dim, mlp_dim))
            ]))

    def forward(self, x, tabular, context=None):
        for attn, ff in self.layers:
            x = attn(x, context=context) + x
            x = ff(x, context=tabular) + x
        return self.norm(x)


class Transformer_T_Assisted_Flamingo(nn.Module):
    def __init__(self, dim, tabular_dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for ind in range(depth):
            self.layers.append(nn.ModuleList([
                Transformer_T_Assited(dim, tabular_dim, 1, heads, dim_head, mlp_dim, dropout=0.),
                GatedCrossAttentionBlock(dim=dim, dim_head=dim_head, heads=heads, ff_mult=mlp_dim // dim)
            ]))

    def forward(self, x, tabular, context=None):
        for attn, ff in self.layers:
            x = attn(x, context=context) + x
            x = ff(x, context=tabular) + x
        return self.norm(x)


class Transformer_IT(nn.Module):
    def __init__(self, dim, tabular_dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                AftNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                AftNorm(dim, IT_FeedForward_Updated(dim, tabular_dim, mlp_dim, num_vectors=150))
            ]))

    def forward(self, x, tabular, context=None):
        for attn, ff in self.layers:
            x = attn(x, context=context) + x
            x = ff(x, context=tabular) + x
        return self.norm(x)


class CrossTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Transformer(dim, 1, heads, dim_head, mlp_dim, dropout=dropout),
                Transformer(dim, 1, heads, dim_head, mlp_dim, dropout=dropout)
            ]))

    def forward(self, mri_tokens, pet_tokens):
        for mri_enc, pet_enc in self.layers:
            mri_tokens = mri_enc(mri_tokens, context=pet_tokens)
            pet_tokens = pet_enc(pet_tokens, context=mri_tokens)
        return mri_tokens, pet_tokens


class CrossTransformer_T(nn.Module):
    def __init__(self, dim, tabular_dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                Transformer_T_Assited(dim, tabular_dim, 1, heads, dim_head, mlp_dim, dropout=dropout)
            )

    def forward(self, mri_tokens, pet_tokens, tabular):
        for enc in self.layers:
            pet_tokens = enc(pet_tokens, tabular, context=mri_tokens)
        return pet_tokens


class CrossTransformer_IT(nn.Module):
    def __init__(self, dim, tabular_dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                Transformer_IT(dim, tabular_dim, 1, heads, dim_head, mlp_dim, dropout=dropout)
            )

    def forward(self, mri_tokens, pet_tokens, tabular):
        for enc in self.layers:
            pet_tokens = enc(pet_tokens, tabular, context=mri_tokens)
        return pet_tokens


# Adopted from https://github.com/ai-med/PANIC/blob/main/torchpanic/models/nam.py
class FeatureNet(nn.Module):
    """A neural network for a single feature"""

    def __init__(
            self,
            out_features: int,
            hidden_units: Sequence[int],
            dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()
        in_features = hidden_units[0]
        layers = {
            "in": nn.Sequential(
                nn.utils.weight_norm(nn.Linear(1, in_features)),
                nn.ReLU()),
        }
        for i, units in enumerate(hidden_units[1:]):
            layers[f"dense_{i}"] = nn.Sequential(
                nn.utils.weight_norm(nn.Linear(in_features, units)),
                nn.Dropout(p=dropout_rate),
                nn.ReLU(),
            )
            in_features = units
        layers["dense_out"] = nn.utils.weight_norm(nn.Linear(in_features, out_features, bias=False))
        self.hidden_layers = nn.ModuleDict(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.hidden_layers.values():
            out = layer(out)
        return out


# Adopted from https://github.com/ai-med/PANIC/blob/main/torchpanic/models/nam.py
class TabularEmbedding(nn.Module):
    def __init__(self,
                 idx_real_features: Sequence[int],
                 idx_cat_features: Sequence[int],
                 out_features: int,
                 hidden_units: Sequence[int],
                 dropout_rate: float = 0.5):
        super().__init__()
        self._idx_real_features = idx_real_features
        self._idx_cat_features = idx_cat_features

        # init missing value embeddings
        n_missing = len(idx_cat_features) + len(idx_real_features)
        self.tab_missing_embeddings = nn.Parameter(
            torch.empty((n_missing, out_features), dtype=torch.float32), requires_grad=True)
        nn.init.xavier_uniform_(self.tab_missing_embeddings)

        # init linear layer used to transform categorical features
        self.cat_linear = nn.Parameter(
            torch.empty((len(idx_cat_features), out_features), dtype=torch.float32, requires_grad=True))
        nn.init.xavier_uniform_(self.cat_linear)

        # init linear layer used to transform real features
        layers = {}
        for i in range(len(idx_real_features)):
            layers[f"fnet_{i}"] = nn.Sequential(
                FeatureNet(
                    out_features=out_features,
                    hidden_units=hidden_units,
                    dropout_rate=dropout_rate,
                ),
            )
        self.feature_nns = nn.ModuleDict(layers)

    def forward_real(self, x):
        features = torch.split(x, 1, dim=-1)
        outputs = []
        for x_i, layer in zip(features, self.feature_nns.values()):
            outputs.append(layer(x_i))

        outputs = torch.stack(outputs, dim=1)
        return outputs

    def forward(self, values, is_missing) -> torch.Tensor:
        val_real = values[:, self._idx_real_features]
        miss_real = is_missing[:, self._idx_real_features]
        has_miss_real = miss_real

        val_categ = values[:, self._idx_cat_features]
        miss_categ = is_missing[:, self._idx_cat_features]
        has_miss_categ = miss_categ

        features_real = self.forward_real(val_real) # [b, n_real, feat_dim]
        features_categ = self.cat_linear.unsqueeze(0) * val_categ.unsqueeze(-1) # [1, n_cat, feat_dim] * [b, n_cat, 1]

        # set features to zero where they are mising
        features_real = features_real * miss_real.unsqueeze(-1) # [b, n_real, feat_dim] * [b, n_real, 1]
        features_categ = features_categ * miss_categ.unsqueeze(-1) # [b, n_cat, feat_dim] * [b, n_cat, 1]

        # fill features with missing indicator where they are missing
        filler_real = self.tab_missing_embeddings[len(self._idx_cat_features):].unsqueeze(0) * \
                      (1.0 - has_miss_real.unsqueeze(-1)) # [1, n_real, feat_dim] * [b, n_real, 1]
        filler_categ = self.tab_missing_embeddings[:len(self._idx_cat_features)].unsqueeze(0) * \
                      (1.0 - has_miss_categ.unsqueeze(-1)) # [1, n_cat, feat_dim] * [b, n_cat, 1]
        features_real = features_real + filler_real
        features_categ = features_categ + filler_categ

        return torch.cat((features_real, features_categ), dim=1)


class CLS_token(torch.nn.Module):
    def __init__(self, dim):
        super(CLS_token, self).__init__()
        self.token = torch.nn.Parameter(torch.randn(1, 1, dim))

    def forward(self):
        return self.token
