from transformers import BertTokenizer, RobertaTokenizer
from transformers import BertModel, RobertaModel


class Trainer():


    def __init__(self, args):
        assert args['task'] in ['category-classification', 'vad-regression', 'vad-from-catogories']
        assert args['model'] in ['bert', 'roberta']
        assert args['dataset'] in ['semeval', 'emobank', 'isear', 'ssec']
        self.args = args
 

    def init_model(self):
        if self.args['model'] == 'bert':
            model_name = 'bert-large-cased-whole-word-masking'
            Tokenizer = BertTokenizer.from_pretrained(model_name)
            model = BertModel.from_pretrained(model_name)
        elif self.args['model'] == 'roberta':
            model_name = 'roberta-large'
            Tokenizer = RobertaTokenizer.from_pretrained(model_name)
            model = RobertaModel.from_pretrained(model_name)
        return model, tokenizer


    def train(self):
        self.tokenizer, self.model = self.init_model()



def main():

    args = {
        'task': 'category-classification', # ['category-classification', 'vad-regression', 'vad-from-catogories']
        'model': 'roberta', # ['bert', 'roberta']
        'dataset': 'semeval' # ['semeval', 'emobank', 'isear', 'ssec']
    }

    trainer = Trainer(args)
    trainer.train()

if __name__ == "__main__":
    main()