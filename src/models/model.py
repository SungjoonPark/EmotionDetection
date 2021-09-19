from torch import nn
import torch
from transformers import *
import torch.nn.functional as F
from data.loader import SemEvalLoader, ISEARLoader, GOEMOTIONSEkmanLoader
import numpy as np


class PretrainedLMModel(BertPreTrainedModel):
    
    def __init__(self, config, cache_path, model_name):
        super(PretrainedLMModel, self).__init__(config)
        self.config = config
        self.args = config.args

        # language models
        if self.args.load_pretrained_lm_weights:
            if self.args.model == 'bert':
                self.pre_trained_lm, loading_info = BertModel.from_pretrained(
                    model_name, 
                    cache_dir=cache_path+'/model/init/',
                    config=self.config,
                    output_loading_info=True)
            else: # 'roberta
                self.pre_trained_lm, loading_info = RobertaModel.from_pretrained(
                    model_name, 
                    cache_dir=cache_path+'/model/init/',
                    config=self.config,
                    output_loading_info=True)
            print("Model Loading Info:", loading_info)
        else:
            if self.args.model == 'bert':
                self.pre_trained_lm = BertModel(self.config)
            else: # 'roberta
                self.pre_trained_lm = RobertaModel(self.config)            

        # dropout
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.projection_lm = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.activation = nn.Sigmoid()

        # classification/regression head
        if self.args.task == "vad-regression":
            # baseline roberta case
            if self.args.load_ckeckpoint is False:
                self.label_num = 1
                self.loader = None
            else:
                if self.args.load_dataset=='semeval':
                    self.label_num = 11
                    self.loader = SemEvalLoader()
                elif self.args.load_dataset=='isear':
                    self.label_num = 7
                    self.loader = ISEARLoader()
                elif self.args.load_dataset=='ekman':
                    self.label_num = 7
                    self.loader = GOEMOTIONSEkmanLoader()

            self.head =nn.Linear(
                self.config.hidden_size,
                self.label_num * 3
            )

            # initialize head weight for each dimension (V,A,D)
            if self.loader is not None:
                self.category_label_names = self.loader.labels
                self.category_label_vads = self.loader.get_vad_coordinates_of_labels()
                v_scores = [self.category_label_vads[key][0] for key in self.category_label_names]
                self.v_sorted_idxs = torch.tensor(np.argsort(v_scores).tolist()).to(self.args.device)
                a_scores = [self.category_label_vads[key][1] for key in self.category_label_names]
                self.a_sorted_idxs = torch.tensor(np.argsort(a_scores).tolist()).to(self.args.device)
                d_scores = [self.category_label_vads[key][2] for key in self.category_label_names]
                self.d_sorted_idxs = torch.tensor(np.argsort(d_scores).tolist()).to(self.args.device)
                self.v_sorted_values = torch.tensor(np.sort(v_scores).tolist()).to(self.args.device)
                self.a_sorted_values = torch.tensor(np.sort(a_scores).tolist()).to(self.args.device)
                self.d_sorted_values = torch.tensor(np.sort(d_scores).tolist()).to(self.args.device)

                self.v_head = nn.Linear(self.label_num, 1 ,bias=False)
                self.v_head.weight = nn.Parameter(torch.unsqueeze(self.v_sorted_values, 0))
                
                self.a_head = nn.Linear(self.label_num, 1 ,bias=False)   
                self.a_head.weight = nn.Parameter(torch.unsqueeze(self.a_sorted_values, 0))         
                
                self.d_head = nn.Linear(self.label_num, 1 ,bias=False)
                self.d_head.weight = nn.Parameter(torch.unsqueeze(self.d_sorted_values, 0))   
                
        elif self.args.task == "vad-from-categories":
            self.head = nn.Linear(
                self.config.hidden_size, 
                len(self.args.label_names)*3 )
        else:
            self.head = nn.Linear(
                self.config.hidden_size,
                len(self.args.label_names)
            )


    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            n_epoch=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
        ):

        lm_outputs = self.pre_trained_lm(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            return_dict=False,
        )

        # hidden states: (batch_size, seq_len, embed_dim)
        # pooled_output: (batch_size, embed_dim)
        # loading pretrained weights
        hidden_states, pooled_output = lm_outputs

        lm_logits = self.projection_lm(hidden_states)

        # add head over [CLS] token
        # ramdomly initialized layers
        pooled_output = self.dropout(pooled_output)
        logits = self.head(pooled_output)

        # additional head for VAD dimension
        if self.args.task == "vad-regression" and self.args.load_ckeckpoint is True:
            
            v_logit, a_logit, d_logit = torch.split(logits, self.label_num, dim=1) # logits for sorted (v, a, d)
            v_probs = self.activation(v_logit)
            a_probs = self.activation(a_logit)
            d_probs = self.activation(d_logit)
            v_logits = self.v_head(v_probs)
            a_logits = self.a_head(a_probs)
            d_logits = self.d_head(d_probs)
            logits = torch.cat((v_logits,a_logits,d_logits), dim=1)

        return lm_logits, logits

