from torch import nn
from transformers import *


class PretrainedLMModel(BertPreTrainedModel):
    
    def __init__(self, config):
        super(PretrainedLMModel, self).__init__(config)
        self.config = config
        self.args = config.args

        # language models
        if self.args['model'] == 'bert':
            self.pre_trained_lm = BertModel(self.config)
        else: # 'roberta
            self.pre_trained_lm = RobertaModel(self.config)

        # dropout
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        # classification/regression head
        if self.args['task'] != "vad-from-categories":
            self.head = nn.Linear(
                self.config.hidden_size, 
                len(self.args['label_names']) )
        else:
            self.head = nn.Linear(
                self.config.hidden_size, 
                len(self.args['label_names'])*3 )

        # init weights
        self.init_weights()


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

        # add head over [CLS] token
        # ramdomly initialized layers
        pooled_output = self.dropout(pooled_output)
        logits = self.head(pooled_output)

        return logits

