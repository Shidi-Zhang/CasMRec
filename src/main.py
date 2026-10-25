import os
import random
from pathlib import Path

ROOT = Path(os.path.abspath(__file__)).parents[1]
import torch
import numpy as np
import dill
import argparse
from torch.optim import Adam
from modules.HemoMole import HemoMoleModel
from modules.gnn import graph_batch_from_smile
from util import buildPrjSmiles
from training import Test, Train
from modules.Causal_discovery import CausalGraph


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_model_name(args):
    model_name = [
        f'dim_{args.dim}',  f'lr_{args.lr}', f'coef_{args.coef}',
        f'dp_{args.dp}', f'ddi_{args.target_ddi}', f'{args.dataset}',
        f'seed_{args.seed}', 'history' if args.history_query else 'diag-proc'
    ]
    if args.embedding:
        model_name.append('embedding')
    return '-'.join(model_name)

def parse_args():
    parser = argparse.ArgumentParser('Experiment For DrugRec')
    parser.add_argument('--Test', action='store_true', help="evaluating mode")
    parser.add_argument('--train_limit', type=int, default=0)
    parser.add_argument('--eval_limit', type=int, default=0)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--smoke_test', action='store_true', help='Train and validate on two patients only; not a performance experiment')
    parser.add_argument('--seed', default=1203, type=int)
    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--history_query', action='store_true', help='Include previous medications in the prediction query; retrain when enabled')
    parser.add_argument('--dim', default=64, type=int, help='model dimension')
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
    parser.add_argument('--dp', default=0.7, type=float, help='dropout ratio')
    parser.add_argument(
        '--model_name', type=str,
        help="the model name for training, if it's left blank,"
        " the code will generate a new model name on it own"
    )
    parser.add_argument(
        '--resume_path', type=str, default=None,
        help='path of well trained model, only for evaluating the model'
    )
    parser.add_argument(
        '--device', type=int, default=0,
        help='gpu id to run on, negative for cpu'
    )
    parser.add_argument(
        '--target_ddi', type=float, default=0.06,
        help='expected ddi for training'
    )
    parser.add_argument(
        '--coef', default=2.5, type=float,
        help='coefficient for DDI Loss Weight Annealing'
    )
    parser.add_argument('--dataset', choices=['mimic3'], default='mimic3')
    parser.add_argument(
        '--epochs', default=30, type=int,
        help='the epochs for training'
    )

    args = parser.parse_args()
    args.embedding = True
    if not 0 < args.threshold < 1:
        parser.error('--threshold must be between 0 and 1')
    if args.target_ddi <= 0:
        parser.error('--target_ddi must be positive')
    if args.Test and args.resume_path is None:
        parser.error('--Test requires --resume_path pointing to a trained checkpoint')
    if args.smoke_test:
        args.epochs = 1
        args.model_name = (args.model_name or 'smoke') + '-smoke'
    if args.model_name is None:
        args.model_name = get_model_name(args)

    return args


if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    torch.set_num_threads(args.threads)
    print(args)
    if not torch.cuda.is_available() or args.device < 0:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.device}')

    data_path = f'{ROOT}/data/{args.dataset}/output/records_final.pkl'
    voc_path = f'{ROOT}/data/{args.dataset}/output/voc_final.pkl'
    ddi_adj_path = f'{ROOT}/data/{args.dataset}/output/ddi_A_final.pkl'
    ddi_mask_path = f'{ROOT}/data/{args.dataset}/output/ddi_mask_H.pkl'
    molecule_path = f'{ROOT}/data/{args.dataset}/input/idx2drug.pkl'
    relevance_diag_med_path = f'{ROOT}/data/{args.dataset}/graphs/Diag_Med_causal_effect.pkl'
    relevance_proc_med_path = f'{ROOT}/data/{args.dataset}/graphs/Proc_Med_causal_effect.pkl'
    with open(ddi_adj_path, 'rb') as Fin:
        ddi_adj = torch.from_numpy(dill.load(Fin)).to(device)
    with open(ddi_mask_path, 'rb') as Fin:
        ddi_mask_H = torch.from_numpy(dill.load(Fin)).to(device)
    with open(data_path, 'rb') as Fin:
        data = dill.load(Fin)
        adm_id = 0
        for patient in data:
            for adm in patient:
                adm.append(adm_id)
                adm_id += 1
    with open(molecule_path, 'rb') as Fin:
        molecule = dill.load(Fin)
    with open(voc_path, 'rb') as Fin:
        voc = dill.load(Fin)
    with open(relevance_proc_med_path, 'rb') as Fin:
        relevance_proc_med = dill.load(Fin)
    with open(relevance_diag_med_path, 'rb') as Fin:
        relevance_diag_med = dill.load(Fin)

    diag_voc, pro_voc, med_voc = \
        voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    voc_size = [
        len(diag_voc.idx2word),
        len(pro_voc.idx2word),
        len(med_voc.idx2word)
    ]

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    print(len(data_train))
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point + eval_len:]

    binary_projection,average_projection, smiles_list = \
        buildPrjSmiles(molecule, med_voc.idx2word)

    average_projection = torch.from_numpy(average_projection).to(device).float()

    molecule_graphs = graph_batch_from_smile(smiles_list)
    molecule_forward = {'batched_data': molecule_graphs.to(device)}
    molecule_para = {
        'num_layer': 4, 'emb_dim': args.dim, 'graph_pooling': 'mean',
        'drop_ratio': args.dp, 'gnn_type': 'gin', 'virtual_node': False
    }   # 定义一个字典，包含描述分子的参数信息

    relevance_diag_mole = np.dot(relevance_diag_med.to_numpy(),binary_projection)
    relevance_proc_mole = np.dot(relevance_proc_med.to_numpy(),binary_projection)
    relerance_med_mole = average_projection
    mole_relevance = [relevance_diag_mole, relevance_proc_mole, relerance_med_mole, binary_projection]
    voc_size.append(relerance_med_mole.shape[1])

    causal_graph = CausalGraph(data, data_train, voc_size[0], voc_size[1], voc_size[2], args.dataset)

    substruct_para, substruct_forward = None, None

    model = HemoMoleModel(
        global_para=molecule_para, substruct_para=substruct_para,
        causal_graph=causal_graph,
        emb_dim=args.dim, global_dim=args.dim, substruct_dim=args.dim,
        substruct_num=ddi_mask_H.shape[1], voc_size=voc_size,
        mole_relevance=mole_relevance,
        use_embedding=args.embedding, device=device, dropout=args.dp,
        history_query=args.history_query
    ).to(device)

    drug_data = {
        'substruct_data': substruct_forward,
        'mol_data': molecule_forward,
        'ddi_mask_H': ddi_mask_H,
        'tensor_ddi_adj': ddi_adj,
        'average_projection': average_projection
    }

    subset_rng = np.random.RandomState(args.seed)
    if args.train_limit:
        indices = subset_rng.permutation(len(data_train))[:args.train_limit]
        data_train = [data_train[i] for i in indices]
    if args.eval_limit:
        indices = subset_rng.permutation(len(data_eval))[:args.eval_limit]
        data_eval = [data_eval[i] for i in indices]
    print('Selected patients:', len(data_train), len(data_eval), flush=True)
    if args.smoke_test:
        data_train = data_train[:2]
        data_eval = data_eval[:2]
        data_test = data_test[:2]

    if args.Test:
        Test(model, args.resume_path, device, data_test, voc_size, drug_data, threshold=args.threshold)
    else:
        log_dir = os.path.join(ROOT, 'runs', args.model_name)
        if os.path.isdir(log_dir) and os.listdir(log_dir):
            raise FileExistsError('Output directory is not empty; choose a new --model_name: ' + log_dir)
        os.makedirs(log_dir, exist_ok=True)
        import json
        with open(os.path.join(log_dir, 'config.json'), 'w') as fout:
            json.dump(vars(args), fout, indent=2)
        optimizer = Adam(model.parameters(), lr=args.lr)
        Train(
            model, device, data_train, data_eval, voc_size, drug_data,
            optimizer, log_dir, args.coef, args.target_ddi, ddi_adj=ddi_adj,EPOCH=args.epochs, threshold=args.threshold
        )
