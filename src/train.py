import os
import torch
from transformers import BertTokenizer, RobertaTokenizer
from transformers import BertModel, RobertaModel

from data import EmobankLoader, SemEvalLoader, ISEARLoader, SSECLoader
from model import EmotionDataset, LM



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


    def load_model(self, ckpt_name=None):
        if self.args['load_model'] == 'pretrained_lm':
            if self.args['model'] == 'bert':
                model_name = 'bert-large-cased-whole-word-masking'
                Tokenizer = BertTokenizer
                Model = BertModel
            elif self.args['model'] == 'roberta':
                model_name = 'roberta-large'
                Tokenizer = RobertaTokenizer
                Model = RobertaModel
            cache_path = self.cache_dir + model_name
            if not os.path.exists(cache_path): 
                os.makedirs(cache_path + '/vocab/')
                os.makedirs(cache_path + '/model/init/')
            tokenizer = Tokenizer.from_pretrained(model_name, cache_dir=cache_path+'/vocab/')
            model = Model.from_pretrained(model_name, cache_dir=cache_path+'/model/init/')
            return model, tokenizer


        elif self.args['load_model'] == 'fine_tuned_lm':    
            assert ckpt_name is not None                      
            pass # to do


    def build_dataloaders(self, tokenizer):
        dataset = self.dataset.build_dataset(tokenizer)
        dataloaders = self.dataset.build_dataloaders(dataset)
        return dataloaders


    def train(self):
        model, tokenizer = self.load_model(ckpt_name=None)
        train_loader, valid_loader, test_loader = self.build_dataloaders(tokenizer)




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

    trainer = Trainer(args)
    trainer.train()

if __name__ == "__main__":
    main()