import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import (
    classification_report,
    jaccard_score
)

import torch
from torch import nn
from torch.nn.modules.loss import *
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from transformers import AdamW, get_linear_schedule_with_warmup
from pytorch_pretrained_bert import BertAdam

from data import EmotionDataset
import pandas as pd



# figure 1-a, Computing Loss + Training
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
            self.ce_loss = torch.nn.CrossEntropyLoss()
        else: # 'multi'
            self.activation = nn.Sigmoid()
            self.ce_loss = torch.nn.BCEWithLogitsLoss()

        self.category_label_vads = self.args.label_vads
        self.category_label_names = self.args.label_names
        self._sort_labels()

        self.eps = 1e-05


    def _check_args(self):
        assert self.args.task == 'vad-from-categories'
        assert self.label_type in ['single', 'multi']
        assert self.args.label_vads is not None
        assert self.args.label_names is not None


    def _sort_labels(self):
        v_scores = [self.category_label_vads[key][0] for key in self.category_label_names]
        self.v_sorted_idxs = torch.tensor(np.argsort(v_scores).tolist()).to(self.args.device)
        a_scores = [self.category_label_vads[key][1] for key in self.category_label_names]
        self.a_sorted_idxs = torch.tensor(np.argsort(a_scores).tolist()).to(self.args.device)
        d_scores = [self.category_label_vads[key][2] for key in self.category_label_names]
        self.d_sorted_idxs = torch.tensor(np.argsort(d_scores).tolist()).to(self.args.device)
        self.v_sorted_values = torch.tensor(np.sort(v_scores).tolist()).to(self.args.device)
        self.a_sorted_values = torch.tensor(np.sort(a_scores).tolist()).to(self.args.device)
        self.d_sorted_values = torch.tensor(np.sort(d_scores).tolist()).to(self.args.device)

    def _sort_labels_by_vad_coordinates(self, labels):
        v_labels = torch.index_select(labels, 1, self.v_sorted_idxs)
        a_labels = torch.index_select(labels, 1, self.a_sorted_idxs)
        d_labels = torch.index_select(labels, 1, self.d_sorted_idxs)
        return v_labels, a_labels, d_labels

    def _set_vad_distance_matrix(self):
        v_distance_vector = torch.roll(self.v_sorted_values, -1, 0) - self.v_sorted_values 
        for idx, v_distance_element in enumerate(v_distance_vector):
            if v_distance_element == 0:
                assert idx != len(v_distance_vector)-1
                v_distance_vector[idx] = v_distance_vector[idx+1]     
        v_distance_vector[-1]=0
        a_distance_vector = torch.roll(self.a_sorted_values, -1, 0) - self.a_sorted_values
        for idx, a_distance_element in enumerate(a_distance_vector):
            if a_distance_element == 0:
                assert idx != len(a_distance_vector)-1
                a_distance_vector[idx] = a_distance_vector[idx+1] 
        a_distance_vector[-1]=0
        d_distance_vector = torch.roll(self.d_sorted_values, -1, 0) - self.d_sorted_values
        for idx, d_distance_element in enumerate(d_distance_vector):
            if d_distance_element == 0:
                assert idx != len(d_distance_vector)-1
                d_distance_vector[idx] = d_distance_vector[idx+1] 
        d_distance_vector[-1]=0
        return v_distance_vector, a_distance_vector, d_distance_vector

    def _intra_EMD_loss(self, input_probs, label_probs):
        intra_emd_loss = torch.div( torch.sum(
            torch.square(input_probs - label_probs), dim=1), len(self.category_label_names))
        return intra_emd_loss 


    def _inter_EMD_loss(self, input_probs, label_probs, distance):
        normalized_input_probs = input_probs / (torch.sum(input_probs, keepdim=True, dim=1) + self.eps)
        normalized_label_probs = label_probs / (torch.sum(label_probs, keepdim=True, dim=1) + self.eps)  

        # multiply vad distance weight to subtraction of cumsum
        inter_emd_loss = torch.matmul(distance, torch.transpose(torch.square(
                torch.cumsum(normalized_input_probs, dim=1) - torch.cumsum(normalized_label_probs, dim=1),
            ), 0, 1))
        return inter_emd_loss


    def forward(self, logits, labels, use_emd=True):
        """
        logits : (batch_size, 3*n_labels) # 3 for each (v, a, d)
        labels : (batch_size, n_labels) # only categorical labels
        """

        if self.label_type == 'single':
            label_one_hot = torch.eye(len(self.category_label_names)).to(self.args.device)
            labels = label_one_hot[labels]

        split_logits = torch.split(logits, len(self.category_label_names), dim=1) # logits for sorted (v, a, d)
        sorted_labels = self._sort_labels_by_vad_coordinates(labels)              # labels for sorted (v, a, d)
        distance_labels = self._set_vad_distance_matrix()

        if self.args.use_emd:
            losses = []
            for logit, sorted_label, distance_label in zip(split_logits, sorted_labels, distance_labels):
                input_probs = self.activation(logit)
                inter_emd_loss = self._inter_EMD_loss(input_probs, sorted_label, distance_label)
                intra_emd_loss = self._intra_EMD_loss(input_probs, sorted_label)
                emd_loss = inter_emd_loss + intra_emd_loss
                losses.append(emd_loss)
            loss = torch.mean(torch.stack(losses, dim=1), dim=1)

        else: # using ce loss
            losses = torch.tensor(0.0).to(self.args.device)
            for logit, label in zip(split_logits, sorted_labels):
                if self.label_type == 'single':
                    label = torch.max(label, 1)[1] # argmax along dim=1
                else:
                    label = label.type_as(logit)
                ce_loss = self.ce_loss(logit, label)
                losses += ce_loss
            loss = losses # (sum of 3 dim)

        return loss



# figure 1-b, c, Predicting VAD scores / class predictions from our method
class PredcitVADandClassfromLogit(torch.nn.Module):
    
    def __init__(self, args, label_type):
        """
        this loss is designed for the task type: "vad-from-categories"
        """
        super(PredcitVADandClassfromLogit, self).__init__()
        self.args = args
        self.label_type = label_type
        self._check_args()

        if label_type == 'single':
            self.activation = nn.Softmax(dim=1)
            self.log_activation = nn.LogSoftmax(dim=1)
        else: # 'multi'
            self.activation = nn.Sigmoid()
            self.log_activation = nn.LogSigmoid()

        self.category_label_vads = self.args.label_vads
        self.category_label_names = self.args.label_names
        self._sort_labels()


    def _check_args(self):
        assert self.args.task == 'vad-from-categories'
        assert self.label_type in ['single', 'multi']
        assert self.args.label_vads is not None
        assert self.args.label_names is not None


    def _sort_labels(self):
        v_scores = [self.category_label_vads[key][0] for key in self.category_label_names]
        self.v_sorted_idxs = torch.tensor(np.argsort(v_scores).tolist()).to(self.args.device)
        self.v_recover_idxs = torch.argsort(self.v_sorted_idxs)
        self.v_sorted_values = torch.tensor(np.sort(v_scores).tolist()).to(self.args.device)

        a_scores = [self.category_label_vads[key][1] for key in self.category_label_names]
        self.a_sorted_idxs = torch.tensor(np.argsort(a_scores).tolist()).to(self.args.device)
        self.a_recover_idxs = torch.argsort(self.a_sorted_idxs)
        self.a_sorted_values = torch.tensor(np.sort(a_scores).tolist()).to(self.args.device)

        d_scores = [self.category_label_vads[key][2] for key in self.category_label_names]
        self.d_sorted_idxs = torch.tensor(np.argsort(d_scores).tolist()).to(self.args.device)
        self.d_recover_idxs = torch.argsort(self.d_sorted_idxs)
        self.d_sorted_values = torch.tensor(np.sort(d_scores).tolist()).to(self.args.device)


    def forward(self, logits, predict):
        assert predict in ['vad', 'cat']
        """
        logits : (batch_size, 3*n_labels) # 3 for each (v, a, d)
        labels : (batch_size, n_labels) # only categorical labels
        """
        # 1. compute (sparse) p(v), p(a), p(d)
        v_logit, a_logit, d_logit = torch.split(logits, len(self.category_label_names), dim=1) # logits for sorted (v, a, d)
        v_probs = self.activation(v_logit)
        a_probs = self.activation(a_logit)
        d_probs = self.activation(d_logit)

        if predict == "vad": # [ compute (v, a, d) == expected values ]
            e_v = torch.sum(v_probs * self.v_sorted_values, dim=1)
            e_a = torch.sum(a_probs * self.a_sorted_values, dim=1)
            e_d = torch.sum(d_probs * self.d_sorted_values, dim=1)
            predictions = torch.stack([e_v, e_a, e_d], dim=1)

        else: #predict == 'cat': [ compute argmax(classes) ]
            v_logits_origin = torch.index_select(v_logit, 1, self.v_recover_idxs)
            a_logits_origin = torch.index_select(a_logit, 1, self.a_recover_idxs)
            d_logits_origin = torch.index_select(d_logit, 1, self.d_recover_idxs)
            class_logits_origin = v_logits_origin + a_logits_origin + d_logits_origin
            if self.label_type == 'multi':
                logprob = class_logits_origin - \
                    torch.log(torch.exp(v_logits_origin) + 1) - \
                    torch.log(torch.exp(a_logits_origin) + 1) - \
                    torch.log(torch.exp(d_logits_origin) + 1)                    
                predictions = torch.pow(torch.exp(logprob), 1/3) >= 0.5
                predictions = torch.squeeze(predictions.float()) 
            else: 
                predictions = torch.max(class_logits_origin, 1)[1] # argmax along dim=1

        return predictions



class Trainer():

    def __init__(self, args):
        self.args = args
        self.args.device = self.set_device()
        self._set_eval_layers()
        self._set_loss()
        if self.args.task == 'vad-from-categories':
            self._set_prediction()


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
        if self.args.task == 'category-classification':
            assert self.args.dataset in ['semeval', 'ssec', 'isear', 'goemotions', 'ekman']
            if self.args.dataset == 'semeval': # multi-labeled
                self.loss = torch.nn.BCEWithLogitsLoss()
            elif self.args.dataset == 'ssec': # multi-labeled
                self.loss = torch.nn.BCEWithLogitsLoss()
            elif self.args.dataset == 'isear': # single-labeled
                self.loss = torch.nn.CrossEntropyLoss()
            elif self.args.dataset == 'goemotions': # multi-labeled
                self.loss = torch.nn.BCEWithLogitsLoss()
            elif self.args.dataset == 'ekman': # multi-labeled
                self.loss = torch.nn.BCEWithLogitsLoss()
            

        elif self.args.task == 'vad-regression':
            assert self.args.dataset in ['emobank']
            self.loss = nn.MSELoss()

        elif self.args.task == 'vad-from-categories':
            assert self.args.dataset in ['semeval', 'ssec', 'isear', 'goemotions', 'ekman']
            if self.args.dataset == 'semeval': # multi-labeled
                self.loss = EMDLoss(self.args, label_type='multi')
            elif self.args.dataset == 'ssec': # multi-labeled
                self.loss = EMDLoss(self.args, label_type='multi')
            elif self.args.dataset == 'isear': # single-labeled
                self.loss = EMDLoss(self.args, label_type='single')
            elif self.args.dataset == 'goemotions': # multi-labeled
                self.loss = EMDLoss(self.args, label_type='multi')
            elif self.args.dataset == 'ekman': # multi-labeled
                self.loss = EMDLoss(self.args, label_type='multi')  


    def _set_prediction(self):
        assert self.args.dataset in ['semeval', 'ssec', 'isear', 'goemotions', 'ekman']
        if self.args.dataset == 'semeval': # multi-labeled
            self.convert_logits_to_predictions = PredcitVADandClassfromLogit(self.args, label_type='multi')
        elif self.args.dataset == 'ssec': # multi-labeled
            self.convert_logits_to_predictions = PredcitVADandClassfromLogit(self.args, label_type='multi')
        elif self.args.dataset == 'isear': # single-labeled
            self.convert_logits_to_predictions = PredcitVADandClassfromLogit(self.args, label_type='single')
        elif self.args.dataset == 'goemotions': # multi-labeled
            self.convert_logits_to_predictions = PredcitVADandClassfromLogit(self.args, label_type='multi')
        elif self.args.dataset == 'ekman': # multi-labeled
            self.convert_logits_to_predictions = PredcitVADandClassfromLogit(self.args, label_type='multi')
        

    def _set_eval_layers(self):
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def _anneal(self, param):
        x = np.linspace(param[0], param[1], num=15)
        return np.exp(x).tolist()

    def _roll_seq(self, x, dim=1, shift=1):
        length = x.size(dim) - shift

        seq = torch.cat([x.narrow(dim, shift, length),
                         torch.zeros_like(x[:, :1])], dim)

        return seq


    def compute_loss(self, inputs, lm_logits, logits, labels):
        if self.args.task == 'vad-regression':
            logits = F.relu(logits)
            overall_loss = self.loss(logits, labels)
        if self.args.task == 'category-classification':
            if self.args.dataset == 'semeval' or self.args.dataset == 'ssec' or self.args.dataset == 'goemotions' or self.args.dataset == 'ekman' : # multi-labeled
                labels = labels.type_as(logits)
                overall_loss = self.loss(logits, labels)
            else:
                overall_loss = self.loss(logits, labels)
        if self.args.task == 'vad-from-categories':
            emd_loss = self.loss(logits, labels)
            exp_decay=[-3.9, -4.6]
            self.lm_coef = exp_decay
            self.lm_coef = self._anneal(self.lm_coef)
            try:
                lm_coef = self.lm_coef[self.coef_step]
            except:
                lm_coef = self.lm_coef[-1]
            
            lm_criterion = nn.CrossEntropyLoss(ignore_index=0)
            lm_loss = lm_criterion(lm_logits.contiguous().view(-1, lm_logits.size(-1)),self._roll_seq(inputs).contiguous().view(-1))
            overall_loss = lm_loss * lm_coef + emd_loss
        return overall_loss


    # https://huggingface.co/transformers/migration.html?highlight=forsequenceclassification
    def set_optimizer(self, params):
        optimizer = AdamW(
            params, 
            lr = self.args.learning_rate,
            betas = (0.9, 0.98),
            eps = 1e-06,
            correct_bias=False
        )

        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=int(self.args.total_n_updates * self.args.warmup_proportion), 
            num_training_steps=self.args.total_n_updates)
            
        return optimizer, lr_scheduler
    

    def set_legacy_optimizer(self, params):
        optimizer = BertAdam(
            params, 
            lr = self.args.learning_rate, 
            schedule='warmup_linear', 
            warmup = self.args.warmup_proportion, 
            t_total = self.args.total_n_updates
        )
        return optimizer, None


    def backward_step(self, it, n_updates, model, loss, accumulated_loss, optimizer, lr_scheduler):
        loss.backward() #Backpropagating the gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip_grad)

        if (it + 1) % self.args.update_freq == 0:
            if self.args.optimizer_type == 'trans':
                lr_scheduler.step()
            optimizer.step()
            if self.args.log_updates:
                print('step:', it, 
                        "(updates:", n_updates ,")", 'loss:', accumulated_loss.item())
            accumulated_loss = 0
            optimizer.zero_grad()
            n_updates += 1

        return accumulated_loss, n_updates


    def _compute_vad_eval_metrics(self, metrics, predictions, labels):
        for x, y, name in zip(predictions.T, labels.T, ["v_cor", 'a_cor', 'd_cor']):
            metrics[name] = pearsonr(x, y)
        return metrics


    def _compute_classification_eval_metrics(self, metrics, predictions, labels, add_jaccard_score):
        report = classification_report(
            labels, 
            predictions, 
            digits=4,
            zero_division='warn', output_dict=True)
        metrics['classification'] = report
        if add_jaccard_score:
            metrics['jaccard_score'] = {}
            metrics['jaccard_score']['samples'] = jaccard_score(labels, predictions, average='samples')
            metrics['jaccard_score']['macro'] = jaccard_score(labels, predictions, average='macro')
            metrics['jaccard_score']['micro'] = jaccard_score(labels, predictions, average='micro')
        return metrics


    def compute_eval_metric(self, predictions, labels, eval_type=None):
        assert predictions.size() == labels.size()
        predictions = predictions.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        metrics = {}

        # 1. vad-regression ---------------------------------------------------------
        if self.args.task == 'vad-regression': # corrleations between v, a, d
            metrics = self._compute_vad_eval_metrics(metrics, predictions, labels)
        
        # 2. vad-from-categories ----------------------------------------------------
        elif self.args.task == 'vad-from-categories':
            assert eval_type in ['vad', 'cat']
            if eval_type == 'vad':
                metrics = self._compute_vad_eval_metrics(metrics, predictions, labels)
            else: # eval_type == 'cat':
                if self.args.dataset in ['semeval', 'ssec', 'goemotions', 'ekman']: # multi-labeled
                    add_jaccard_score = True
                elif self.args.dataset == 'isear': # single-labeled
                    add_jaccard_score = False
                metrics = self._compute_classification_eval_metrics(
                    metrics, 
                    predictions, 
                    labels,
                    add_jaccard_score=add_jaccard_score)             
        
        # 3. category-classification ------------------------------------------------
        elif self.args.task == 'category-classification':
            if self.args.dataset in ['semeval', 'ssec', 'goemotions', 'ekman']: # multi-labeled
                add_jaccard_score = True
            elif self.args.dataset == 'isear': # single-labeled
                add_jaccard_score = False
            metrics = self._compute_classification_eval_metrics(
                metrics, 
                predictions, 
                labels,
                add_jaccard_score=add_jaccard_score)             

        return metrics


    def predict(self, model, dataloader, prediction_type=None, compute_loss=True):
        model.eval()

        total_losses = torch.tensor(0).to(torch.device(self.args.device)).float()
        total_predictions = []
        total_labels = []

        with torch.no_grad():

            for it, batch in enumerate(dataloader):
                
                # 1. compute logits
                input_ids = batch[0].to(torch.device(self.args.device))
                attention_masks = batch[1].to(torch.device(self.args.device))
                labels = batch[2].to(torch.device(self.args.device))
                total_labels.append(labels)

                lm_logits, cls_logits = model(
                    input_ids,
                    attention_mask=attention_masks)

                # 2. compute loss (objective)
                if compute_loss:
                    eval_loss = self.compute_loss(input_ids, lm_logits, cls_logits, labels)
                    total_losses += torch.sum(eval_loss)
                
                # 3. model predictions
                if self.args.task == 'vad-regression':
                    predictions = F.relu(cls_logits) # vads

                elif self.args.task == 'vad-from-categories':
                    if prediction_type == 'vad':
                        predictions = self.convert_logits_to_predictions(cls_logits, predict='vad')         # vad 
                    else: # prediction_type == 'cat':
                        predictions = self.convert_logits_to_predictions(cls_logits, predict='cat')         # vad 

                elif self.args.task == 'category-classification':
                    assert self.args.dataset in ['semeval', 'ssec', 'isear', 'goemotions', 'ekman']
                    if self.args.dataset == 'semeval': # multi-labeled
                        predictions = self.sigmoid(cls_logits) >= 0.5
                        predictions = torch.squeeze(predictions.float()) 
                    elif self.args.dataset == 'ssec': # multi-labeled
                        predictions = self.sigmoid(cls_logits) >= 0.5
                        predictions = torch.squeeze(predictions.float()) 
                    elif self.args.dataset == 'isear': # single-labeled
                        predictions = self.softmax(cls_logits)
                        predictions = torch.max(predictions, 1)[1] # argmax along dim=1
                    elif self.args.dataset == 'goemotions': # multi-labeled
                        predictions = self.sigmoid(cls_logits) >= 0.5
                        predictions = torch.squeeze(predictions.float())
                    elif self.args.dataset == 'ekman': # multi-labeled
                        predictions = self.sigmoid(cls_logits) >= 0.5
                        predictions = torch.squeeze(predictions.float())

                total_predictions.append(predictions)

        total_predictions = torch.cat(total_predictions, 0)
        total_labels = torch.cat(total_labels, 0)
        total_losses = total_losses / total_predictions.size()[0] # mean losses

        return total_predictions, total_labels, total_losses


    def evaluate(self, model, dataloader, prediction_type=None, compute_loss=True):
        predictions = self.predict(model, dataloader, prediction_type, compute_loss)
        total_predictions, total_labels, total_losses = predictions

        eval_loss = torch.mean(total_losses)
        eval_metrics = self.compute_eval_metric(total_predictions, total_labels, eval_type=prediction_type)

        return eval_loss, eval_metrics, total_predictions.size()

