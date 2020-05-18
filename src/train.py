import os
import torch
from transformers import BertTokenizer, RobertaTokenizer
from transformers import BertModel, RobertaModel

from data import (
    EmobankLoader, 
    SemEvalLoader, 
    ISEARLoader, 
    SSECLoader,
    EmotionDataset
)

from models import (
    PretrainedBERTModel,
    PretrainedRoBERTaModel
)


class Trainer():

    def __init__(self, args):
        os.environ["CUDA_VISIBLE_DEVICES"]= args['CUDA_VISIBLE_DEVICES']

        self._check_args(args)
        self.args = args

        self.cache_dir = "./../ckpt/"
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        self.dataset = EmotionDataset(args)


    def _check_args(self, args):
        assert args['task'] in ['category-classification', 'vad-regression', 'vad-from-categories']
        assert args['model'] in ['bert', 'roberta']
        assert args['dataset'] in ['semeval', 'emobank', 'isear', 'ssec']
        assert args['load_model'] in ['pretrained_lm', 'fine_tuned_lm']


    #def load_model(self, ckpt_name=None):
    #    if self.args['load_model'] == 'pretrained_lm':
    #        if self.args['model'] == 'bert':
    #            model_name = 'bert-large-cased-whole-word-masking'
    #            Tokenizer = BertTokenizer
    #            Model = BertModel
    #        elif self.args['model'] == 'roberta':
    #            model_name = 'roberta-large'
    #            Tokenizer = RobertaTokenizer
    #            Model = RobertaModel
    #        cache_path = self.cache_dir + model_name
    #        if not os.path.exists(cache_path): 
    #            os.makedirs(cache_path + '/vocab/')
    #            os.makedirs(cache_path + '/model/init/')
    #        tokenizer = Tokenizer.from_pretrained(model_name, cache_dir=cache_path+'/vocab/')
    #        model = Model.from_pretrained(model_name, cache_dir=cache_path+'/model/init/')
    #        return model, tokenizer


    #    elif self.args['load_model'] == 'fine_tuned_lm':    
    #        assert ckpt_name is not None                      
    #        pass # to do

    def _set_model_args(self):
        if self.args['model'] == 'bert':
            model_name = 'bert-large-cased-whole-word-masking'
        elif self.args['model'] == 'roberta':
            model_name = 'roberta-large'
        cache_path = self.cache_dir + model_name
        if not os.path.exists(cache_path): 
            os.makedirs(cache_path + '/vocab/')
            os.makedirs(cache_path + '/model/init/')
        return model_name, cache_path


    def load_tokenizer(self):
        model_name, cache_path = self._set_model_args()
        if self.args['model'] == 'bert':
            Tokenizer = BertTokenizer
        elif self.args['model'] == 'roberta':
            Tokenizer = RobertaTokenizer
        tokenizer = Tokenizer.from_pretrained(model_name, cache_dir=cache_path+'/vocab/')
        return tokenizer


    def load_model(self):
        model_name, cache_path = self._set_model_args()
        if self.args['model'] == 'bert':
            model = PretrainedBERTModel.from_pretrained(model_name, cache_dir=cache_path+'/model/init/')
        elif self.args['model'] == 'roberta':
            model = PretrainedRoBERTaModel.from_pretrained(model_name, cache_dir=cache_path+'/model/init/')
        return model


    def train(self):
        # 1. build dataset for train/valid/test
        #tokenizer = self.load_tokenizer()
        #dataset = self.dataset.build_datasets(tokenizer)
        #train_loader, valid_loader, test_loader = self.dataset.build_dataloaders(dataset)

        # 2. build/load models
        model = self.load_model()


def main():
    args = {
        'CUDA_VISIBLE_DEVICES': "0",

        'task': 'category-classification', # ['category-classification', 'vad-regression', 'vad-from-categories']
        'model': 'bert', # ['bert', 'roberta']
        'dataset': 'semeval', # ['semeval', 'emobank', 'isear', 'ssec']
        'load_model': 'pretrained_lm', # else: fine_tuned_lm

        'max_seq_len': 128,
        'train_batch_size': 32,
        'eval_batch_size': 32,

    }

    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()