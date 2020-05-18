import os
import torch
from transformers import (
    BertTokenizer, 
    RobertaTokenizer,
    BertModel, 
    RobertaModel, 
    BertConfig, 
    RobertaConfig
)

from data import (
    EmobankLoader, 
    SemEvalLoader, 
    ISEARLoader, 
    SSECLoader,
    EmotionDataset
)

from models import (
    PretrainedBERTModel,
    PretrainedRoBERTaModel,
    Trainer,
)


class SingleDatasetTrainer():

    def __init__(self, args):
        os.environ["CUDA_VISIBLE_DEVICES"]= args['CUDA_VISIBLE_DEVICES']

        self._check_args(args)
        self.args = args

        self.cache_dir = "./../ckpt/"
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        self.dataset = EmotionDataset(args)
        self.trainer = Trainer(args)


    def _check_args(self, args):
        assert args['task'] in ['category-classification', 'vad-regression', 'vad-from-categories']
        assert args['model'] in ['bert', 'roberta']
        assert args['dataset'] in ['semeval', 'emobank', 'isear', 'ssec']
        assert args['load_model'] in ['pretrained_lm', 'fine_tuned_lm']


    def _set_model_args(self):
        if self.args['model'] == 'bert':
            model_name = 'bert-large-cased-whole-word-masking'
        elif self.args['model'] == 'roberta':
            model_name = 'roberta-large'
        cache_path = self.cache_dir + model_name
        if not os.path.exists(cache_path): 
            os.makedirs(cache_path + '/vocab/')
            os.makedirs(cache_path + '/model/init/')
            os.makedirs(cache_path + '/model/config/')
        return model_name, cache_path


    def load_tokenizer(self):
        model_name, cache_path = self._set_model_args()
        if self.args['model'] == 'bert':
            Tokenizer = BertTokenizer
        elif self.args['model'] == 'roberta':
            Tokenizer = RobertaTokenizer
        tokenizer = Tokenizer.from_pretrained(
            model_name, 
            cache_dir=cache_path+'/vocab/')
        return tokenizer


    def load_model(self):
        model_name, cache_path = self._set_model_args()
        self.args['labels'] = self.dataset.labels

        if self.args['model'] == 'bert':
            config = BertConfig.from_pretrained(
                model_name, 
                cache_dir=cache_path+'/model/config/')
            config.args = self.args
            model = PretrainedBERTModel.from_pretrained(
                model_name, 
                cache_dir=cache_path+'/model/init/',
                config=config)

        elif self.args['model'] == 'roberta':
            config = RobertaConfig.from_pretrained(
                model_name, 
                cache_dir=cache_path+'/model/config/')
            config.args = self.args
            model = PretrainedRoBERTaModel.from_pretrained(
                model_name, 
                cache_dir=cache_path+'/model/init/',
                config=config)

        return model, config
    

    def compute_loss(self, batch_logit, batch_labels):
        loss = self.trainer.compute_loss(
            batch_logit, 
            batch_labels)
        return loss


    def train(self):
        # 1. build dataset for train/valid/test
        print("build dataset for train/valid/test")
        tokenizer = self.load_tokenizer()
        dataset = self.dataset.build_datasets(tokenizer)
        train_loader, valid_loader, test_loader = self.dataset.build_dataloaders(dataset)

        # 2. build/load models
        print("build/load models")
        model, config = self.load_model()
        print(model)
        print(config)
        print(config.args["labels"])

        for train_batch in train_loader:
            input_ids, attention_masks, label = train_batch
            print(input_ids.size())        #[batch_size, max_len]
            #print(attention_masks.size()) #[batch_size, max_len]
            #print(label.size())           #[batch_size, n_labels]
            break

        outputs = model(
            input_ids,
            attention_mask=attention_masks)

        logits = outputs                    #[batch_size, n_labels]
        print(input_ids[0])
        print(logits[0])

        loss = self.compute_loss(logit, labels)


def main():
    args = {
        'CUDA_VISIBLE_DEVICES': "0",

        'task': 'category-classification', # ['category-classification', 'vad-regression', 'vad-from-categories']
        'model': 'roberta', # ['bert', 'roberta']
        'dataset': 'semeval', # ['semeval', 'emobank', 'isear', 'ssec']
        'load_model': 'pretrained_lm', # else: fine_tuned_lm

        'max_seq_len': 128,
        'train_batch_size': 32,
        'eval_batch_size': 32,

    }

    trainer = SingleDatasetTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()