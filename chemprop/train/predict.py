from typing import List, Tuple, Union

import torch
import torch.nn as nn
from rdkit.Chem.Draw import rdMolDraw2D
from tqdm import trange
import numpy as np
from rdkit import Chem
import pandas as pd
from chemprop.data import MoleculeDataset, StandardScaler


def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()


def predict(model: nn.Module,
            datasets: MoleculeDataset,
            scaler: StandardScaler = None,
            analyze_dir=None):
    model.eval()
    model.apply(apply_dropout)
    preds = []
    uq_sum = []
    labels = []
    sum_atoms = 0
    all_atoms_uq_list = []
    atoms = 0
    j = 0
    for batch, features_batch, targets in datasets:

        with torch.no_grad():
            batch_preds, uq, n_atoms, uq_out, a_scope, b_scope = model(batch, features_batch)
            sum_atoms += n_atoms
            if analyze_dir is not None:
                all_uq = torch.stack(uq).sum(-1)
                for i, (a_start, a_size) in enumerate(a_scope):
                    atoms += 1
                    smiles = [atom.GetSymbol() for atom in Chem.MolFromSmiles(batch[i]).GetAtoms()]
                    iter_atom_uq = all_uq.narrow(1, a_start, a_size).cpu().numpy()
                    df = pd.DataFrame(data=iter_atom_uq,
                                      columns=smiles,
                                      index=range(10))
                    df.insert(0, batch[i], '', allow_duplicates=False)
                    df.to_csv(f'{analyze_dir}/测试集所有原子的迭代UQ/mol_{atoms}.csv')

            # # Draw mol
            # uq = torch.stack(uq)
            # uq = uq.sum(-1)
            # uq = uq.mean(0)
            #
            # for n in range(len(batch)):
            #     j+=1
            #     iter_atom_uq = uq.narrow(-1, a_scope[n][0], a_scope[n][1])
            #     mol_uq = list(iter_atom_uq.detach().cpu().numpy())
            #
            #     smiles = batch[n]
            #     mol = Chem.MolFromSmiles(smiles)
            #     num_atoms = mol.GetNumAtoms()
            #
            #     d = rdMolDraw2D.MolDraw2DCairo(500, 500)
            #     for i in range(num_atoms):
            #         mol.GetAtomWithIdx(i).SetProp('atomNote', '\n' + str('%.5f' % mol_uq[i]))
            #     d.drawOptions().addStereoAnnotation = True
            #     d.drawOptions().addAtomIndices = True
            #     d.DrawMolecule(mol,legend=smiles)
            #
            #     d.FinishDrawing()
            #     d.WriteDrawingText(f'mol_{j}.png')
        batch_preds = batch_preds.data.cpu().numpy()
        batch_ale_unc = torch.stack(uq).cpu().numpy().mean(0).sum()
        targets = targets.data.cpu().numpy()

        # # Inverse scale if regression
        if scaler is not None:
            batch_preds = scaler.inverse_transform(batch_preds)
        # Collect vectors
        preds.append(batch_preds)
        uq_sum.append(batch_ale_unc)
        labels.append(targets)

    preds = np.array(preds)
    uq_sum = np.array(uq_sum)
    labels = np.array(labels)

    return preds.reshape(-1, preds.shape[-1]), uq_sum, labels.reshape(-1, labels.shape[
        -1]), sum_atoms, uq, a_scope, batch, all_atoms_uq_list
