from models.networks import *
from models.backbones import *
from models.daft import DAFT, InteractiveHNN
from models.resnet import generate_model
from munch import Munch
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier


def build_models(args, model):
    # Get Class Number
    if args.task == 'MCIC' or args.task == 'ADD' or args.task == 'MCINC':
        class_num = 2
    else:
        class_num = 3
    # Build Model
    if model == 'CNN_Single':
        CNN = SFCN(channel_number=[32, 64, 128, 256, 256, args.dim])
        CLS = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(args.dim, class_num)
        )
        nets = Munch(CNN=CNN, CLS=CLS)
    elif model == 'Transformer':
        CNN = sNet(args.dim)
        Trans = Transformer(dim=args.dim, depth=args.trans_enc_depth, heads=args.heads_num,
                            dim_head=args.dim // args.heads_num, mlp_dim=args.dim * 4, dropout=args.dropout)
        CLS = nn.Linear(args.dim * 2, class_num)
        nets = Munch(CNN=CNN, Trans=Trans, CLS=CLS)
    elif model == 'Transformer_T':
        CNN = sNet(args.dim)
        T_Embedding = TabularEmbedding(idx_real_features=args.tabular_continues_idx,
                                       idx_cat_features=args.tabular_categorical_idx,
                                       out_features=4, dropout_rate=0.5, hidden_units=[args.dim // 2, args.dim // 2])
        Trans = Transformer_T_Assited(dim=args.dim, tabular_dim=len(args.tabular) * 4, depth=args.trans_enc_depth,
                                      heads=args.heads_num, dim_head=args.dim // args.heads_num,
                                      mlp_dim=args.dim * 4, dropout=args.dropout)
        CLS_Head = nn.Linear(args.dim * 2, class_num)
        nets = Munch(CNN=CNN, Trans=Trans, CLS=CLS_Head, T_Embedding=T_Embedding)
    elif model == 'Transformer_IT':
        CNN = SFCN(channel_number=[32, 64, 128, 256, 256, args.dim])
        T_Embedding = TabularEmbedding(idx_real_features=args.tabular_continues_idx,
                                       idx_cat_features=args.tabular_categorical_idx,
                                       out_features=4, dropout_rate=0.5, hidden_units=[args.dim // 2, args.dim // 2])
        Trans = Transformer_IT(dim=args.dim, tabular_dim=len(args.tabular) * 4, depth=args.trans_enc_depth,
                               heads=args.heads_num, dim_head=args.dim // args.heads_num,
                               mlp_dim=args.dim * 4, dropout=args.dropout)
        CLS_Head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(args.dim * 2, class_num)
        )
        nets = Munch(CNN=CNN, Trans=Trans, CLS=CLS_Head, T_Embedding=T_Embedding)
    elif model == 'CrossTransformer_IT':
        # networks
        T_Embedding = TabularEmbedding(idx_real_features=args.tabular_continues_idx,
                                       idx_cat_features=args.tabular_categorical_idx,
                                       out_features=4, dropout_rate=0.5, hidden_units=[args.dim // 2, args.dim // 2])
        MRI_CNN = SFCN(channel_number=[32, 64, 128, 256, 256, args.dim])
        PET_CNN = SFCN(channel_number=[16, 32, 64, 128, 128, args.dim])
        MRI_Trans = Transformer_IT(dim=args.dim, tabular_dim=len(args.tabular) * 4, depth=args.trans_enc_depth,
                                   heads=args.heads_num, dim_head=args.dim // args.heads_num,
                                   mlp_dim=args.dim * 4, dropout=args.dropout)
        PET_Trans = CrossTransformer_IT(dim=args.dim, tabular_dim=len(args.tabular) * 4, depth=args.trans_enc_depth,
                                        heads=args.heads_num, dim_head=args.dim // args.heads_num,
                                        mlp_dim=args.dim * 4, dropout=args.dropout)
        Tabular_CLS_Head = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(len(args.tabular) * 4, args.dim * 4)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(args.dim * 4, class_num))
        )
        MRI_CLS_Head = nn.Linear(args.dim * 2, class_num)
        PET_CLS_Head = nn.Linear(args.dim * 2, class_num)
        nets = Munch(MRI=MRI_CNN, PET=PET_CNN, T_Embedding=T_Embedding,
                     MRI_Trans=MRI_Trans, PET_Trans=PET_Trans,
                     Tabular_CLS=Tabular_CLS_Head, MRI_CLS=MRI_CLS_Head, PET_CLS=PET_CLS_Head)
    elif model == 'mlp':
        nets = MLPClassifier(random_state=20230329, max_iter=600, warm_start=True)
    elif model == 'catboost':
        if class_num == 2:
            nets = CatBoostClassifier()
        else:
            nets = CatBoostClassifier(loss_function='MultiClass')
    elif model == 'tabular':
        T_Embedding = TabularEmbedding(idx_real_features=args.tabular_continues_idx,
                                       idx_cat_features=args.tabular_categorical_idx,
                                       out_features=4, dropout_rate=0.5, hidden_units=[args.dim // 2, args.dim // 2])
        MLPs = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(len(args.tabular) * 4, args.dim * 4)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(args.dim * 4, class_num))
        )
        nets = Munch(T_Embedding=T_Embedding, CLS=MLPs)
    # comparison methods
    elif model == '3MT':
        Q = CLS_token(args.dim)
        T_Embedding = TabularEmbedding(idx_real_features=args.tabular_continues_idx,
                                       idx_cat_features=args.tabular_categorical_idx,
                                       out_features=64, dropout_rate=0.5, hidden_units=[args.dim // 2, args.dim // 2])
        T_FUSION = Attention(dim=args.dim, heads=args.heads_num, dim_head=args.dim // args.heads_num)
        T_CLS = nn.Linear(args.dim, class_num)
        MRI_CNN = SFCN(channel_number=[32, 64, 128, 256, 256, args.dim])
        MRI_FUSION = Attention(dim=args.dim, heads=args.heads_num, dim_head=args.dim // args.heads_num)
        MRI_CLS = nn.Linear(args.dim, class_num)
        PET_CNN = SFCN(channel_number=[16, 32, 64, 128, 128, args.dim])
        PET_FUSION = Attention(dim=args.dim, heads=args.heads_num, dim_head=args.dim // args.heads_num)
        PET_CLS = nn.Linear(args.dim, class_num)
        nets = Munch(CLS_token=Q, T_Embedding=T_Embedding, T_FUSION=T_FUSION, T_CLS=T_CLS,
                     MRI=MRI_CNN, MRI_FUSION=MRI_FUSION, MRI_CLS=MRI_CLS,
                     PET=PET_CNN, PET_FUSION=PET_FUSION, PET_CLS=PET_CLS)
    elif model == 'DAFT':
        nets = Munch(DAFT=DAFT(in_channels=1, n_outputs=class_num))
    elif model == 'HNN':
        nets = Munch(DAFT=InteractiveHNN(in_channels=1, n_outputs=class_num))
    elif model == 'ResNet':
        CLS = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, class_num)
        )
        nets = Munch(CNN=generate_model(10), CLS=CLS)
    else:
        nets = None
    return nets