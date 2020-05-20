import numpy as np
import torch
from torch import nn
from torch.nn.modules.loss import *

from data import EmotionDataset


class EMDLoss(torch.nn.Module):
    
    def __init__(self, args, label_type):
        """
        this loss is designed for the task type: "vad-from-categories"
        """
        super(EMDLoss, self).__init__()
        self.args = args

        assert self.args['task'] == 'vad-from-categories'
        assert label_type in ['single', 'multi']

        if label_type == 'single':
            self.activation = nn.Softmax(dim=1)
        else: # 'multi'
            self.activation = nn.Sigmoid()

        if args['task'] == 'vad-from-categories':
            self.category_label_vads = self.args['label_vads']
            self.category_label_names = self.args['label_names']
            self._sort_labels()

        self.eps = 1e-05


    def _sort_labels(self):
        v_scores = [self.category_label_vads[key][0] for key in self.category_label_names]
        self.v_sorted_idxs = np.argsort(v_scores).tolist()
        a_scores = [self.category_label_vads[key][1] for key in self.category_label_names]
        self.a_sorted_idxs = np.argsort(a_scores).tolist()
        d_scores = [self.category_label_vads[key][2] for key in self.category_label_names]
        self.d_sorted_idxs = np.argsort(d_scores).tolist()


    def _sort_labels_by_vad_coordinates(self, labels):
        v_labels = torch.index_select(labels, 1, torch.tensor(self.v_sorted_idxs))
        a_labels = torch.index_select(labels, 1, torch.tensor(self.a_sorted_idxs))
        d_labels = torch.index_select(labels, 1, torch.tensor(self.d_sorted_idxs))
        return v_labels, a_labels, d_labels


    def _intra_EMD_loss(self, input_probs, label_probs):
        intra_emd_loss = torch.sum(
            torch.square(input_probs - label_probs), dim=1)
        return intra_emd_loss


    def _inter_EMD_loss(self, input_probs, label_probs):
        normalized_input_probs = input_probs / (torch.sum(input_probs, keepdim=True, dim=1) + self.eps)
        normalized_label_probs = label_probs / (torch.sum(label_probs, keepdim=True, dim=1) + self.eps)   
        inter_emd_loss = torch.sum(
            torch.square(
                torch.cumsum(normalized_input_probs, dim=1) - 
                torch.cumsum(normalized_label_probs, dim=1),
            ), dim=1)
        return inter_emd_loss


    def forward(self, logits, labels):
        """
        logits : (batch_size, 3*n_labels) # 3 for each (v, a, d)
        labels : (batch_size, n_labels) # only categorical labels
        """
        loss = 0

        split_logits = torch.split(logits, len(self.category_label_names), dim=1) # logits for sorted (v, a, d)
        sorted_labels = self._sort_labels_by_vad_coordinates(labels)              # labels for sorted (v, a, d)

        for logit, label in zip(split_logits, sorted_labels):
            input_probs = self.activation(logit)
            inter_emd_loss = self._inter_EMD_loss(input_probs, labels)
            intra_emd_loss = self._intra_EMD_loss(input_probs, labels)
            emd_loss = inter_emd_loss + intra_emd_loss
            loss += emd_loss
        return loss


class Trainer():

    def __init__(self, args):
        self.args = args
        self._set_loss()


    def _set_loss(self):
        if self.args['task'] == 'category-classification':
            assert self.args['dataset'] in ['semeval', 'ssec', 'isear']
            if self.args['dataset'] == 'semeval': # multi-labeled
                self.loss = torch.nn.BCEWithLogitsLoss()
            elif self.args['dataset'] == 'ssec': # multi-labeled
                self.loss = torch.nn.BCEWithLogitsLoss()
            elif self.args['dataset'] == 'isear': # single-labeled
                self.loss = torch.nn.CrossEntropyLoss()

        elif self.args['task'] == 'vad-regression':
            assert self.args['dataset'] in ['emobank']
            self.loss = nn.MSELoss()

        elif self.args['task'] == 'vad-from-categories':
            assert self.args['dataset'] in ['semeval', 'ssec', 'isear']
            if self.args['dataset'] == 'semeval': # multi-labeled
                self.loss = EMDLoss(self.args, label_type='multi')
            elif self.args['dataset'] == 'ssec': # multi-labeled
                self.loss = EMDLoss(self.args, label_type='multi')
            elif self.args['dataset'] == 'isear': # single-labeled
                self.loss = EMDLoss(self.args, label_type='single')
    

    def _set_optimizer(self):
        self.optim = None


    def compute_loss(self, logits, labels):
        loss = self.loss(logits, labels)
        return loss

    def optimize(self):

        return