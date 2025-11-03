from argparse import Namespace
from typing import List, Union

import torch
import torch.nn as nn
import numpy as np

from chemprop.features import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph
from chemprop.nn_utils import index_select_ND, get_activation_function, get_cc_dropout_hyper
from chemprop.models.concrete_dropout import ConcreteDropout

import time

from torch.nn.utils.rnn import pad_sequence

torch.set_printoptions(edgeitems=7)


class MPNEncoder(nn.Module):  # for atomic_vecs_d2, atomic_vecs_final, mol_vecs
    """A message passing neural network for encoding a molecule."""

    def __init__(self, args: Namespace, atom_fdim: int, bond_fdim: int):

        super(MPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.depth = args.depth
        self.dropout = args.dropout
        self.layers_per_message = 1
        self.undirected = args.undirected
        self.atom_messages = args.atom_messages
        self.use_input_features = args.use_input_features
        self.max_atom_size = args.max_atom_size
        self.epistemic = args.epistemic
        self.mc_dropout = self.epistemic == 'mc_dropout'
        self.aggregation = args.aggregation
        self.aggregation_norm = args.aggregation_norm
        self.fp_method = args.fp_method
        self.corr_similarity_function = args.corr_similarity_function
        self.args = args
        self.gradients = None
        self.tensorhook = []
        self.layerhook = []
        self.selected_out = None
        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function(args.activation)

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        # Concrete Dropout for Bayesian NN
        wd, dd = get_cc_dropout_hyper(args.train_data_size, args.regularization_scale)

        # Input
        input_dim = self.atom_fdim if self.atom_messages else self.bond_fdim  # self.bond_fdim=145

        # cosine similarity
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-8)

        if self.mc_dropout:
            self.W_i = ConcreteDropout(layer=nn.Linear(input_dim, self.hidden_size, bias=self.bias),
                                       reg_acc=args.reg_acc, weight_regularizer=wd, dropout_regularizer=dd)

        else:
            self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)
            self.W_b = nn.Linear(300, 128, bias=self.bias)

        if self.atom_messages:
            w_h_input_size = self.hidden_size + self.bond_fdim
        else:  # hidden_size
            w_h_input_size = self.hidden_size

        # Shared weight matrix across depths (default)
        if self.mc_dropout:
            self.W_h = ConcreteDropout(layer=nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias),
                                       reg_acc=args.reg_acc, weight_regularizer=wd, dropout_regularizer=dd
                                       )
            self.W_o = ConcreteDropout(layer=nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size),
                                       reg_acc=args.reg_acc, weight_regularizer=wd, dropout_regularizer=dd)
        else:
            self.W_h = nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)
            self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)



    def iter_mod(self, input, message, f_atoms, f_bonds, a2b, b2a, b2revb,a2a=None):

        # Message passing
        for depth in range(self.depth - 1):
            if self.undirected:
                message = (message + message[b2revb]) / 2

            if self.atom_messages:  # False
                nei_a_message = index_select_ND(message, a2a)  # num_atoms x max_num_bonds x hidden
                nei_f_bonds = index_select_ND(f_bonds, a2b)  # num_atoms x max_num_bonds x bond_fdim
                nei_message = torch.cat((nei_a_message, nei_f_bonds),
                                        dim=2)  # num_atoms x max_num_bonds x hidden + bond_fdim
                message = nei_message.sum(dim=1)  # num_atoms x hidden + bond_fdim
            else:

                nei_a_message = index_select_ND(message, a2b)  # num_atoms x max_num_bonds x hidden
                a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
                rev_message = message[b2revb]  # num_bonds x hidden
                message = a_message[b2a] - rev_message  # num_bonds x hidden

            message = self.W_h(message)
            message = self.act_func(input + message)  # num_bonds x hidden_size
            message = self.dropout_layer(message)  # num_bonds x hidden

        a2x = a2a if self.atom_messages else a2b
        nei_a_message = index_select_ND(message, a2x)  # num_atoms x max_num_bonds x hidden
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        a_input = torch.cat([f_atoms, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)
        ############ without relu ##############
        atom_hiddens = self.W_o(a_input)  # num_atoms x hidden  # norelu!!, self.act_func(self.W_o(a_input))
        ############ without relu ##############
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden
        return atom_hiddens, message

    def forward(self,
                mol_graph: BatchMolGraph,
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:

        if self.use_input_features:
            features_batch = torch.from_numpy(np.stack(features_batch)).float()

            if self.args.cuda:
                features_batch = features_batch.cuda()

            if self.features_only:
                return features_batch

        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, conv_bt = mol_graph.get_components()
        if self.atom_messages:
            a2a = mol_graph.get_a2a()

        if self.args.cuda or next(self.parameters()).is_cuda:
            f_atoms, f_bonds, a2b, b2a, b2revb, conv_bt = f_atoms.cuda(), f_bonds.cuda(), a2b.cuda(), b2a.cuda(), b2revb.cuda(), conv_bt.cuda()  # wei fix

            if self.atom_messages:
                a2a = a2a.cuda()

        messages_uq_list = []
        atom_uq_list = []
        if self.atom_messages:  # false
            input = self.W_i(f_atoms)  # num_atoms x hidden_size
            message = self.act_func(input)  # num_bonds x hidden_size
        else:
            input = self.W_i(f_bonds)  # num_bonds x hidden_size
            message = self.act_func(input)  # num_bonds x hidden_size
        for i in range(10):

            with torch.no_grad():
                iter_uq = []
                iter_atuq = []
                for j in range(self.args.dropouts):
                    atom_hiddens, messages = self.iter_mod(input, message, f_atoms, f_bonds, a2b, b2a, b2revb)
                    iter_uq.append(messages)
                    iter_atuq.append(atom_hiddens)
                messages_uq = torch.stack(iter_uq)
                messages_uq = torch.var(messages_uq, 0)

                atom_uq = torch.stack(iter_atuq)
                atom_uq = torch.var(atom_uq, 0)

                messages_uq_list.append(messages_uq)
                atom_uq_list.append(atom_uq)

                message = message + message * messages_uq * 0.001

        # Readout
        atom_hiddens, _ = self.iter_mod(input, message, f_atoms, f_bonds, a2b, b2a, b2revb)
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                mol_vecs.append(atom_hiddens.narrow(0, a_start, a_size))
                # padding


        return pad_sequence(mol_vecs,batch_first=True), atom_uq_list, a_scope,b_scope


class MPN(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(self,
                 args: Namespace,
                 atom_fdim: int = None,
                 bond_fdim: int = None,
                 graph_input: bool = False):

        super(MPN, self).__init__()  # equals to nn.Module.__init__()
        self.features_only = args.features_only
        self.args = args
        self.atom_fdim = atom_fdim or get_atom_fdim(args)
        self.bond_fdim = bond_fdim or get_bond_fdim(args) + (
            not args.atom_messages) * self.atom_fdim
        self.graph_input = graph_input
        self.encoder = MPNEncoder(self.args, self.atom_fdim, self.bond_fdim)

        if self.features_only:
            return

    def forward(self,
                batch: Union[List[str], BatchMolGraph],
                features_batch: List[np.ndarray] = None):

        if not self.graph_input:  # if features only, batch won't even be used
            batch = mol2graph(batch, self.args)

        hid_vec, atom_uq, a_scope,b_scope = self.encoder.forward(batch, features_batch)
        return hid_vec, batch.n_atoms, batch.uq_out, atom_uq, a_scope,b_scope
