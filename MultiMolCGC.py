import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import GATConv, global_add_pool, TransformerConv

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr, kendalltau

import matplotlib.pyplot as plt
from tqdm import tqdm

import argparse
import pickle
import seaborn as sns

parser = argparse.ArgumentParser(description="Prediction Pipeline for MultiMolCGC")
parser.add_argument('--data_csv', type=str, required=True,
                        help='Path to the clustered dataset CSV file for prediction.')
parser.add_argument('--output_path', type=str, required=True,
                        help='CSV path to save the prediction results.')

args = parser.parse_args()

loss_metric = 'mae' # mse, mae, huber
delta = 1  # for huber loss

pro_embed_model = 'esmc'
embed_dim = {'esm2': 1280, 'esmc': 1152}


# SMILES TO GRAPH

def one_hot(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def bond_to_features(bond):
    bond_type = bond.GetBondType()
    bond_stereo = bond.GetStereo()
    bond_conjugation = bond.GetIsConjugated()
    bond_is_in_ring = bond.IsInRing()

    bond_type_one_hot = one_hot(bond_type, [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ])
    bond_stereo_one_hot = one_hot(bond_stereo, [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS
    ])

    return torch.tensor(bond_type_one_hot + bond_stereo_one_hot + [bond_conjugation, bond_is_in_ring], dtype=torch.float)

def smiles_to_graph(smiles, fps):
    mol = Chem.MolFromSmiles(smiles)
    atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    atom_features_raw = torch.tensor(atoms, dtype=torch.float).view(-1, 1)

    ring = mol.GetRingInfo()

    fps_torch = torch.tensor(fps, dtype=torch.float).view(1, -1)

    # node fea
    node_features_list = []
    for atom_idx, atom in enumerate(mol.GetAtoms()):
        o = []
        o += one_hot(atom.GetSymbol(), ['C', 'H', 'O', 'N', 'S', 'Cl', 'F', 'Br', 'P', 'I'])
        o += [atom.GetDegree()]
        o += one_hot(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP,
                                               Chem.rdchem.HybridizationType.SP2,
                                               Chem.rdchem.HybridizationType.SP3,
                                               Chem.rdchem.HybridizationType.SP3D,
                                               Chem.rdchem.HybridizationType.SP3D2])
        o += [atom.GetImplicitValence()]
        o += [atom.GetIsAromatic()]
        o += [ring.IsAtomInRingOfSize(atom_idx, rsize) for rsize in [3, 4, 5, 6, 7, 8]]
        o += [atom.GetFormalCharge()]

        o_torch = torch.tensor(o, dtype=torch.float).view(1, -1)
        merged_feat = torch.cat([atom_features_raw[atom_idx].view(1, -1), o_torch, fps_torch], dim=1)
        node_features_list.append(merged_feat.squeeze(0))

    node_features = torch.stack(node_features_list, dim=0)

    # edge fea
    edges = []
    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.append([i, j])
        edges.append([j, i])

        b_feat = bond_to_features(bond)
        edge_features.append(b_feat)
        edge_features.append(b_feat)

    if len(edges) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 11), dtype=torch.float)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_features, dim=0)

    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
    return data

# READ CSV AND LOAD GRAPH DATASET AS A LIST

def load_graph_dataset(csv_path, start=0, finish=0, pct=1, shuffle=True):
    df = pd.read_csv(csv_path)
    data_list = []

    if shuffle:
        if pct == 1:
            pass
        else:
            if start ==  0 and  finish == 0:
                df = df.sample(frac=1, random_state=42).head(int(len(df) * pct))
            else:
                df = df.sample(frac=1, random_state=42)[start:finish]
    else:
        if pct == 1:
            pass
        else:
            if start ==  0 and  finish == 0:
                    df = df.head(int(len(df) * pct))
            else:
                df = df[start:finish]

    for idx, row in df.iterrows():
        smiles = row['SMILES']
        # if 'error' in row['score']:
        #     continue
        sars  = np.float32(row['sars'])
        mers  = np.float32(row['mers'])

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        # 1024-bit Morgan Fingerprint (radius=2)
        fp_morgan = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        fp_morgan_bits = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp_morgan, fp_morgan_bits)

        # MACCS Fingerprint (166 bits)
        fp_maccs = MACCSkeys.GenMACCSKeys(mol)
        fp_maccs_bits = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp_maccs, fp_maccs_bits)

        # physiochemical features
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        rtb = rdMolDescriptors.CalcNumRotatableBonds(mol)
        psa = rdMolDescriptors.CalcTPSA(mol)
        stereo_count = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
        c_logp, mr = rdMolDescriptors.CalcCrippenDescriptors(mol)
        csp3 = rdMolDescriptors.CalcFractionCSP3(mol)
        nrings = rdMolDescriptors.CalcNumRings(mol)
        nrings_h = rdMolDescriptors.CalcNumHeterocycles(mol)
        nrings_ar = rdMolDescriptors.CalcNumAromaticRings(mol)
        nrings_ar_h = rdMolDescriptors.CalcNumAromaticHeterocycles(mol)
        spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
        mw = rdMolDescriptors._CalcMolWt(mol)
        atm_hetero = rdMolDescriptors.CalcNumHeteroatoms(mol)
        atm_heavy = mol.GetNumHeavyAtoms()
        atm_all = mol.GetNumAtoms()

        # concat features
        fps_array = np.concatenate((
            np.array([hbd, hba, rtb, psa, stereo_count, c_logp, mr, csp3, nrings, nrings_h,
                      nrings_ar, nrings_ar_h, spiro, mw, atm_hetero, atm_heavy, atm_all]),
            fp_morgan_bits,
            fp_maccs_bits
        ), axis=0)

        graph_data = smiles_to_graph(smiles, fps_array)

        y_values = [sars, mers]
        graph_data.y = torch.tensor(y_values, dtype=torch.float).unsqueeze(0)
        graph_data.cluster = int(row['cluster'])
        graph_data.smiles = row['SMILES']

        data_list.append(graph_data)

    return data_list


def masked_loss(pred, target):
    mask = ~torch.isnan(target)
    pred = pred[mask]
    target = target[mask]

    if pred.numel() == 0:
        return torch.tensor(0.0, device=pred.device)

    if loss_metric == 'mse':
        return F.mse_loss(pred, target, reduction='mean')
    elif loss_metric == 'mae':
        return F.l1_loss(pred, target, reduction='mean')
    elif loss_metric == 'huber':
        return F.huber_loss(pred, target, delta=delta, reduction='mean')
    else:
        raise ValueError(f"Invalid loss_metric: {loss_metric}. Choose from ['mse', 'mae', 'huber']")

class MaskedStandardScaler(StandardScaler):
    def __init__(self):
        self.avgs = None
        self.vars = None
        self.mask = None

    def fit(self, X, y=None):
        self.avgs = [0.0]*X.shape[1]
        self.vars = [0.0]*X.shape[1]
        self.mask = ~np.isnan(X)
        for prop in range(X.shape[1]):
            mask = ~np.isnan(X[:,prop])
            self.avgs[prop] = np.mean(X[:,prop][mask])
            self.vars[prop] = np.sqrt(np.var(X[:,prop][mask]))
        return self

    def transform(self, X):
        for prop in range(X.shape[1]):
            for i in range(X.shape[0]):
                X[i, prop] = (X[i, prop] - self.avgs[prop]) / self.vars[prop]

        return X

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def inverse_transform(self, X):
        for prop in range(X.shape[1]):
            for i in range(X.shape[0]):
                X[i, prop] = X[i, prop]  * self.vars[prop] + self.avgs[prop]
        #X[~self.mask] = np.nan
        return X


# MODEL DEFINATION

TASKS = ['sars','mers']
TASK_OUTPUT_DIMS = {t: 1 for t in TASKS}


class CrossModalAttention(nn.Module):
    def __init__(self, ligand_dim, protein_dim, out_dim=None):
        super().__init__()
        self.ligand_dim = ligand_dim
        self.protein_dim = protein_dim
        if out_dim is None:
            out_dim = ligand_dim

        self.d_k = out_dim
        self.query_proj = nn.Linear(ligand_dim, self.d_k, bias=False)
        self.key_proj   = nn.Linear(protein_dim, self.d_k, bias=False)
        self.value_proj = nn.Linear(protein_dim, self.d_k, bias=False)

    def forward(self, ligand_rep, protein_embed):
        batch_size = ligand_rep.size(0)

        Q = self.query_proj(ligand_rep)
        K = self.key_proj(protein_embed)
        V = self.value_proj(protein_embed)

        Q = Q.unsqueeze(1)
        K = K.unsqueeze(0)
        K = K.expand(batch_size, -1, -1)
        
        V = V.unsqueeze(0)
        V = V.expand(batch_size, -1, -1)

        attn_logits = torch.bmm(Q, K.transpose(1,2)) / np.sqrt(self.d_k)
        attn_weights = F.softmax(attn_logits, dim=-1)

        context = torch.bmm(attn_weights, V)
        context = context.squeeze(1)

        fused = torch.cat([ligand_rep, context], dim=1)
        return fused


class MTLModelCGC_graph_protein(nn.Module):
    def __init__(self,
                 input_dim, edge_dim,
                 shared_hidden_dim=128,
                 individual_hidden_dim=128,
                 num_experts=2,
                 num_heads=1,
                 dp=0.2,
                 protein_embed_dim=embed_dim[pro_embed_model]  # ESM embedding dim
                 ):
        super(MTLModelCGC_graph_protein, self).__init__()

        self.tasks = ['sars', 'mers']
        self.num_tasks = len(self.tasks)
        self.edge_dim = edge_dim

        import pickle
        with open(f'sars_embed_{pro_embed_model}.pkl','rb') as f:
            sars_embed_np = pickle.load(f)  # shape: [res_num_sars, embedding_dim]
        with open(f'mers_embed_{pro_embed_model}.pkl','rb') as f:
            mers_embed_np = pickle.load(f)  # shape: [res_num_mers, embedding_dim]

        self.protein_dict = {
            'sars': torch.tensor(sars_embed_np, dtype=torch.float),
            'mers': torch.tensor(mers_embed_np, dtype=torch.float)
        }

        # shared experts
        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                GATConv(input_dim, shared_hidden_dim // num_heads, heads=num_heads, edge_dim=edge_dim),
                nn.BatchNorm1d(shared_hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(dp),
                GATConv(shared_hidden_dim, shared_hidden_dim // num_heads, heads=num_heads, edge_dim=edge_dim),
                nn.BatchNorm1d(shared_hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(dp)
            ) for _ in range(num_experts)
        ])

        # task-specific experts
        self.task_specific_experts = nn.ModuleDict({
            task: nn.ModuleList([
                nn.Sequential(
                    GATConv(input_dim, shared_hidden_dim // num_heads, heads=num_heads, edge_dim=edge_dim),
                    nn.BatchNorm1d(shared_hidden_dim),
                    nn.LeakyReLU(),
                    nn.Dropout(dp),
                    GATConv(shared_hidden_dim, shared_hidden_dim // num_heads, heads=num_heads, edge_dim=edge_dim),
                    nn.BatchNorm1d(shared_hidden_dim),
                    nn.LeakyReLU(),
                    nn.Dropout(dp)
                ) for _ in range(num_experts)
            ]) for task in self.tasks
        })

        # gating
        self.gates = nn.ModuleDict({
            task: nn.Sequential(
                GATConv(input_dim, num_experts * 2 // num_heads, heads=num_heads, edge_dim=edge_dim),
                nn.Softmax(dim=1)
            ) for task in self.tasks
        })

        # cross attention
        self.cross_attn_modules = nn.ModuleDict({
            task: CrossModalAttention(ligand_dim=shared_hidden_dim,
                                      protein_dim=protein_embed_dim,
                                      out_dim=shared_hidden_dim)
            for task in self.tasks
        })

        # task heads
        final_dim = shared_hidden_dim * 2
        self.task_heads = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(final_dim, individual_hidden_dim),
                nn.BatchNorm1d(individual_hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(dp),
                nn.Linear(individual_hidden_dim, 1)
            ) for task in self.tasks
        })

    def _apply_expert(self, expert_block, x, edge_index, edge_attr):
        out = x
        for layer in expert_block:
            if isinstance(layer, GATConv):
                out = layer(out, edge_index, edge_attr)
            else:
                out = layer(out)
        return out

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        shared_rep = torch.stack(
            [self._apply_expert(expert, x, edge_index, edge_attr)
             for expert in self.shared_experts], dim=1
        ) 

        task_outputs = {}
        task_reps = []
        for task, experts in self.task_specific_experts.items():
            task_rep = torch.stack(
                [self._apply_expert(expert, x, edge_index, edge_attr)
                 for expert in experts], dim=1
            )

            merged_rep = torch.cat([shared_rep, task_rep], dim=1)

            gate_logits = self._apply_expert(self.gates[task], x, edge_index, edge_attr)
            node_rep = torch.einsum('beh,be->bh', merged_rep, gate_logits)

            node_rep_pooled = global_add_pool(node_rep, batch)

            protein_embed = self.protein_dict[task].to(node_rep_pooled.device)  
            fused_rep = self.cross_attn_modules[task](node_rep_pooled, protein_embed)  
            task_reps.append(fused_rep)

            out = self.task_heads[task](fused_rep)
            task_outputs[task] = out

        preds = []
        for t in self.tasks:
            preds.append(task_outputs[t])
        preds = torch.cat(preds, dim=1)

        return preds, task_reps


# TRAINING
def compute_quality_metrics(ref, pred):
    rmse = np.sqrt(np.mean((ref - pred)**2))
    mae = np.mean(np.abs(ref - pred))
    corr_pearson = pearsonr(ref, pred)[0] if len(ref) > 1 else np.nan
    corr_kendall = kendalltau(ref, pred)[0] if len(ref) > 1 else np.nan
    return rmse, mae, corr_pearson, corr_kendall


def cluster_train_test_split(valid_data_list, test_cluster=1):
    train, test = [], []
    for data in valid_data_list:
        if data.cluster == test_cluster:
            test.append(data)
        else:
            train.append(data)
    return train, test


if __name__ == '__main__':
    #csv_path = 'potency_train_val_clustered.csv'
    csv_path = args.data_csv
    full_data_list = load_graph_dataset(csv_path, start=0, finish=100 , pct=1, shuffle=True)
    print(f"num of graphs in the dataset: {len(full_data_list)}")

    valid_data_list = []
    for d in full_data_list:
        if torch.isnan(d.y).all():
            continue
        valid_data_list.append(d)

    sample_data = valid_data_list[0]
    input_dim = sample_data.x.size(1)
    edge_dim = sample_data.edge_attr.size(1) if sample_data.edge_attr is not None and sample_data.edge_attr.size(0) > 0 else 11
    print("Graph node feature dim:", input_dim)
    print("Graph edge feature dim:", edge_dim)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_clusters = 5
    train_epoch = 80
    batch_size = 256

    hidden_dim = 256
    dp = 0.1
    lr = 5e-4
    wd = 0
    multistep_lr = [7*i for i in range(1, 993)]

    norm = True
    if norm:
        all_y = []
        for d in valid_data_list:
            all_y.append(d.y.numpy().reshape(1, -1))
        all_y = np.concatenate(all_y, axis=0)

        scaler_y = MaskedStandardScaler()
        all_y_scaled = scaler_y.fit_transform(all_y)
        idx_ = 0
        for d in valid_data_list:
            nparr = all_y_scaled[idx_]
            idx_ += 1
            d.y = torch.tensor(nparr, dtype=torch.float).unsqueeze(0)


    results = []
    metrics = np.zeros((num_clusters, len(TASKS), 4))
    all_smiles = []
    all_true_sars, all_true_mers = [], []
    all_pred_sars, all_pred_mers = [], []
    all_cluster = []
    all_task_rep_sars, all_task_rep_mers = [], []

    for cluster in range(num_clusters):
        train_dataset, test_dataset = cluster_train_test_split(valid_data_list, cluster)
        print(f'Test Cluster: {cluster}, Train size: {len(train_dataset)}, Test Size: {len(test_dataset)}')

        train_loader = GeoDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = GeoDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # model
        num_PLE_layers = 2
        model = MTLModelCGC_graph_protein(input_dim=input_dim,
        edge_dim=edge_dim,
        shared_hidden_dim=256,
        individual_hidden_dim=256,
        num_experts=2,
        num_heads=1,
        dp=0.2,).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=multistep_lr, gamma=0.8)

        # training
        train_loss_curve = []
        test_loss_curve = []

        with tqdm(range(train_epoch)) as epochs:
            for epoch in epochs:
                model.train()
                total_train_loss = 0
                for batch_data in train_loader:
                    batch_data = batch_data.to(device)
                    pred, _ = model(batch_data)
                    #print(pred.shape, batch_data.y.shape)
                    loss = masked_loss(pred, batch_data.y)
                    # print(loss)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_train_loss += loss.item()

                    if np.isnan(loss.item()):
                        exit(1)
                scheduler.step()

                ave_train_loss = total_train_loss / len(train_loader)
                train_loss_curve.append(np.sqrt(ave_train_loss))

                # eval
                model.eval()
                total_test_loss = 0
                with torch.no_grad():
                    for batch_data in test_loader:
                        batch_data = batch_data.to(device)
                        pred, _ = model(batch_data)
                        loss = masked_loss(pred, batch_data.y)
                        total_test_loss += loss.item()
                ave_test_loss = total_test_loss / len(test_loader)
                test_loss_curve.append(np.sqrt(ave_test_loss))

                # if (epoch+1) % 5 == 0:
                #     print(f"[Seed={42+seed_offset} Epoch={epoch+1}] "
                #         f"Train RMSE={train_loss_curve[-1]:.4f}, Test RMSE={test_loss_curve[-1]:.4f}")
                
                infos = {
                    'Epoch': epoch,
                    'TrainLoss': f'{np.sqrt(ave_train_loss):.3f}',
                    'TestLoss': f'{np.sqrt(ave_test_loss):.3f}',
                }
                epochs.set_postfix(infos)

        # eval on final epoch
        model.eval()
        y_true_all = []
        y_pred_all = []
        y_seq = []
        task_reps_sars = []
        task_reps_mers = []
        with torch.no_grad():
            for batch_data in test_loader:
                batch_data = batch_data.to(device)
                pred, task_reps = model(batch_data)
                y_true_all.append(batch_data.y.cpu().numpy())
                y_pred_all.append(pred.cpu().numpy())
                y_seq.append(batch_data.smiles)
                task_reps_sars.append(task_reps[0].cpu().numpy())
                task_reps_mers.append(task_reps[1].cpu().numpy())


        y_true_all = np.concatenate(y_true_all, axis=0)
        y_pred_all = np.concatenate(y_pred_all, axis=0)
        y_seq = np.concatenate(y_seq, axis=0)
        task_reps_sars_all = np.concatenate(task_reps_sars, axis=0)  # [N, hiddendim]
        task_reps_mers_all = np.concatenate(task_reps_mers, axis=0)  # [N, hiddendim]
        all_task_rep_sars.append(task_reps_sars_all)
        all_task_rep_mers.append(task_reps_mers_all)

        if norm:
            y_true_all = scaler_y.inverse_transform(y_true_all)
            y_pred_all = scaler_y.inverse_transform(y_pred_all)
        
        all_smiles.extend(y_seq)
        all_pred_sars.extend(list(y_pred_all[:,0])); all_pred_mers.extend(list(y_pred_all[:,1]))
        all_true_sars.extend(list(y_true_all[:,0])); all_true_mers.extend(list(y_true_all[:,1]))
        all_cluster.extend([cluster]*len(y_seq))

        fig, axs = plt.subplots(2, 3, figsize=(15, 8))
        axs = axs.ravel()

        y_true_all_f = y_true_all.flatten()
        y_pred_all_f = y_pred_all.flatten()

        for i, task_name in enumerate(TASKS):
            ax = axs[i]
            mask = ~np.isnan(y_true_all[:, i])
            true_ = y_true_all[mask, i]
            pred_ = y_pred_all[mask, i]

            ax.set_title(task_name, fontsize=14)

            if len(true_) == 0:
                ax.text(0.5, 0.5, 'No valid data', ha='center', va='center')
                continue

            min_ = min(true_.min(), pred_.min())
            max_ = max(true_.max(), pred_.max())
            ax.plot([min_, max_], [min_, max_], color='r', linestyle='--', label='y=x')

            ax.scatter(true_, pred_, alpha=0.6)
            ax.set_xlabel('Reference')
            ax.set_ylabel('Prediction')

            rmse, mae, r_p, r_k = compute_quality_metrics(true_, pred_)
            ax.legend(title=f'RMSE={rmse:.3f}\nMAE={mae:.3f}\nPearson={r_p:.3f}\nKendall={r_k:.3f}')
            metrics[cluster, i] = [rmse, mae, r_p, r_k]

        plt.tight_layout()
        os.makedirs('epoch_plots', exist_ok=True)

        results.append([train_loss_curve[-1], test_loss_curve[-1]])

        # fig2, axes = plt.subplots(1, 1, figsize=(8, 8))
        # axs = axs.ravel()
        # ax=axs[0]
        plt.plot(range(len(train_loss_curve)), train_loss_curve, c='red')
        plt.plot(range(len(train_loss_curve)), test_loss_curve, c='blue')
        plt.savefig(f'epoch_plots/Cluster_{cluster}_curve_CGC.png')
    
    pred_df = {
        'SMILES': all_smiles, 
        'sars': all_true_sars, 
        'sars_pred': all_pred_sars,
        'mers': all_true_mers, 
        'mers_pred': all_pred_mers, 
        'cluster': all_cluster
    }
    pred_df = pd.DataFrame(pred_df)
    pred_df.to_csv(args.output_path, index=False)
    # print(np.array2string(metrics, formatter={'float_kind': lambda x: "%.3f" % x}))
    for cluster in range(num_clusters):
        print(f'cluster {cluster} MAE: sars - {metrics[cluster, 0, 1]}, mers - {metrics[cluster, 1, 1]}')
    print(f'=========total=========')
    all_true_sars = np.array(all_true_sars); all_true_mers = np.array(all_true_mers); all_pred_sars = np.array(all_pred_sars); all_pred_mers = np.array(all_pred_mers)
    mask_sars = ~np.isnan(all_true_sars); mask_mers = ~np.isnan(all_true_mers)
    all_true_sars = all_true_sars[mask_sars]; all_true_mers = all_true_mers[mask_mers]; all_pred_sars = all_pred_sars[mask_sars]; all_pred_mers = all_pred_mers[mask_mers]
    sars_mae = np.mean(np.abs(all_true_sars - all_pred_sars))
    mers_mae = np.mean(np.abs(all_true_mers - all_pred_mers))
    print(f'MAE: sars - {sars_mae}, mers - {mers_mae}')
