from torch import nn
from transformers import *


class PretrainedBERTModel(BertPreTrainedModel):
    
    def __init__(self, config):
        super(PretrainedBERTModel, self).__init__(config)
        self.config = config
        self.args = config.args

        self.bert = BertModel(self.config)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.head = nn.Linear(self.config.hidden_size, len(self.args['labels']))

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

        bert_outputs = self.bert(
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
        hidden_states, pooled_output = bert_outputs

        # add head over [CLS] token
        # ramdomly initialized layers
        pooled_output = self.dropout(pooled_output)
        logits = self.head(pooled_output)

        return logits


class PretrainedRoBERTaModel(BertPreTrainedModel):

    def __init__(self, config):
        super(PretrainedRoBERTaModel, self).__init__(config)
        self.config = config
        self.args = config.args

        self.roberta = RobertaModel(self.config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.head = nn.Linear(config.hidden_size, len(self.args['labels']))

        self.init_weights()


    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
        ):

        roberta_outputs = self.roberta(
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
        hidden_states, pooled_output = roberta_outputs

        # add head over [CLS] token
        # ramdomly initialized layers
        pooled_output = self.dropout(pooled_output)
        logits = self.head(pooled_output)

        return logits
