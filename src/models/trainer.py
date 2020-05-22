import numpy as np
from scipy.stats import pearsonr

import torch
from torch import nn
from torch.nn.modules.loss import *
import torch.nn.functional as F

from transformers import AdamW, get_linear_schedule_with_warmup

from data import EmotionDataset


class EMDLoss(torch.nn.Module):
    
    def __init__(self, args, label_type):
        """
        this loss is designed for the task type: "vad-from-categories"
        """
        super(EMDLoss, self).__init__()
        self.args = args
        self.label_type = label_type
        self._check_args()

        if label_type == 'single':
            self.activation = nn.Softmax(dim=1)
        else: # 'multi'
            self.activation = nn.Sigmoid()

        self.category_label_vads = self.args['label_vads']
        self.category_label_names = self.args['label_names']
        self._sort_labels()

        self.eps = 1e-05


    def _check_args(self):
        assert self.args['task'] == 'vad-from-categories'
        assert self.label_type in ['single', 'multi']
        assert self.args['label_vads'] is not None
        assert self.args['label_names'] is not None


    def _sort_labels(self):
        v_scores = [self.category_label_vads[key][0] for key in self.category_label_names]
        self.v_sorted_idxs = torch.tensor(np.argsort(v_scores).tolist()).to(self.args['device'])
        a_scores = [self.category_label_vads[key][1] for key in self.category_label_names]
        self.a_sorted_idxs = torch.tensor(np.argsort(a_scores).tolist()).to(self.args['device'])
        d_scores = [self.category_label_vads[key][2] for key in self.category_label_names]
        self.d_sorted_idxs = torch.tensor(np.argsort(d_scores).tolist()).to(self.args['device'])


    def _sort_labels_by_vad_coordinates(self, labels):
        v_labels = torch.index_select(labels, 1, self.v_sorted_idxs)
        a_labels = torch.index_select(labels, 1, self.a_sorted_idxs)
        d_labels = torch.index_select(labels, 1, self.d_sorted_idxs)
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
        self.args['device'] = self.set_device()
        self._set_eval_layers()
        self._set_loss()

    # https://mccormickml.com/2019/07/22/BERT-fine-tuning/
    def set_device(self):
        if torch.cuda.is_available():      
            device = "cuda"
            print('There are %d GPU(s) available.' % torch.cuda.device_count())
            print('We will use the GPU:', torch.cuda.get_device_name(0))
        else:
            device = "cpu"
            print('No GPU available, using the CPU instead.')
        return device


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


    def compute_loss(self, logits, labels):
        if self.args['task'] == 'vad-regression':
            logits = F.relu(logits)
        return self.loss(logits, labels)

    # https://huggingface.co/transformers/migration.html?highlight=forsequenceclassification
    def set_optimizer(self, params):
        optimizer = AdamW(
            params, 
            lr = self.args['learning_rate'],
            betas = (0.9, 0.98),
            eps = 1e-06,
            correct_bias=False
        )

        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=int(self.args['total_n_updates'] * self.args['warmup_proportion']), 
            num_training_steps=self.args['total_n_updates'])
            
        return optimizer, lr_scheduler
    

    def backward_step(self, it, n_updates, loss, accumulated_loss, optimizer, lr_scheduler):
        loss.backward() #Backpropagating the gradients

        if (it + 1) % self.args['update_freq'] == 0:
            optimizer.step()
            lr_scheduler.step()

            print('step:', it, 
                    "(updates:", n_updates ,")", 'loss:', accumulated_loss)

            accumulated_loss = 0
            optimizer.zero_grad()
            n_updates += 1

        return accumulated_loss, n_updates


    def _set_eval_layers(self):
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()


    def compute_eval_metric(self, predictions, labels):
        assert predictions.size() == labels.size()
        print(predictions.size())
        print(labels.size())

        predictions = predictions.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        metrics = {}

        if self.args['task'] == 'vad-regression': # corrleations between v, a, d
            for x, y, name in zip(predictions.T, labels.T, ["v_cor", 'a_cor', 'd_cor']):
                metrics[name] = pearsonr(x, y)
        elif self.args['task'] == 'vad-from-categories':
            #predictions =            # vad 
            pass
        elif self.args['task'] == 'category-classification':
            pass

        return metrics


    def predict(self, model, dataloader):
        model.eval()

        total_losses = []
        total_predictions = []
        total_labels = []

        with torch.no_grad():

            for it, batch in enumerate(dataloader):
                
                # 1. compute logits
                input_ids = batch[0].to(torch.device(self.args['device']))
                attention_masks = batch[1].to(torch.device(self.args['device']))
                labels = batch[2].to(torch.device(self.args['device']))
                total_labels.append(labels)

                logits = model(
                    input_ids,
                    attention_mask=attention_masks)

                # 2. compute loss (objective)
                eval_loss = self.compute_loss(logits, labels)
                total_losses.append(eval_loss)
                
                # 3. model predictions
                if self.args['task'] == 'vad-regression':
                    predictions = F.relu(logits) # vads
                elif self.args['task'] == 'vad-from-categories':
                    #predictions =            # vad 
                    pass
                elif self.args['task'] == 'category-classification':
                    assert self.args['dataset'] in ['semeval', 'ssec', 'isear']
                    if self.args['dataset'] == 'semeval': # multi-labeled
                        predictions = self.sigmoid(logits)
                    elif self.args['dataset'] == 'ssec': # multi-labeled
                        predictions = self.sigmoid(logits)
                    elif self.args['dataset'] == 'isear': # single-labeled
                        predictions = self.softmax(logits)
                total_predictions.append(predictions)

        total_predictions = torch.cat(total_predictions, 0)
        total_labels = torch.cat(total_labels, 0)
        total_losses = torch.stack(total_losses, 0)

        return total_predictions, total_labels, total_losses


    def evaluate(self, model, dataloader):
        predictions = self.predict(model, dataloader)
        total_predictions, total_labels, total_losses = predictions

        eval_loss = torch.mean(total_losses)
        eval_metrics = self.compute_eval_metric(total_predictions, total_labels)

        return eval_loss, eval_metrics
