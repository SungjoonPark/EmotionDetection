from torch import nn
from transformers import *


class PretrainedLMModel(BertPreTrainedModel):
    
    def __init__(self, config, cache_path, model_name):
        super(PretrainedLMModel, self).__init__(config)
        self.config = config
        self.args = config.args

        # language models
        if self.args['load_pretrained_lm_weights']:
            if self.args['model'] == 'bert':
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
            if self.args['model'] == 'bert':
                self.pre_trained_lm = BertModel(self.config)
            else: # 'roberta
                self.pre_trained_lm = RobertaModel(self.config)            

        # dropout
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.projection_lm = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

        # classification/regression head
        if self.args['task'] != "vad-from-categories":
            self.head = nn.Linear(
                self.config.hidden_size, 
                len(self.args['label_names']) )
        else:
            self.head = nn.Linear(
                self.config.hidden_size, 
                len(self.args['label_names'])*3 )


    def forward(
            self,
            input_ids=None,
            attention_mask=None,
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

        return lm_logits, logits

