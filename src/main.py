import os, pprint
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
    PretrainedLMModel,
    Trainer,
)

pp = pprint.PrettyPrinter(indent=1, width=90)


class SingleDatasetTrainer():

    def __init__(self, args):
        # args
        os.environ["CUDA_VISIBLE_DEVICES"]= args['CUDA_VISIBLE_DEVICES']
        self._check_args(args)
        self.args = args

        # cache path
        self.cache_dir = "./../ckpt/"
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        # dataset class
        self.dataset = EmotionDataset(args)
        if args['task'] == 'vad-from-categories':
            label_names, label_vads = self.dataset.load_label_names_and_vads()
            self.args['label_names'] = label_names
            self.args['label_vads'] = label_vads
        else:
            self.args['label_names'] = self.dataset.load_label_names()

        # trainer class
        self.trainer = Trainer(args)
        self.device = self.trainer.args['device']


    def _check_args(self, args):
        assert args['task'] in ['category-classification', 'vad-regression', 'vad-from-categories']
        assert args['model'] in ['bert', 'roberta']
        assert args['dataset'] in ['semeval', 'emobank', 'isear', 'ssec']
        assert args['load_model'] in ['pretrained_lm', 'fine_tuned_lm']
        assert args['label-type'] in ['categorical', 'dimensional']
        assert args['optimizer_type'] in ['legacy', 'trans']


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


    def set_train_vars(self):
        self.n_updates = 0
        self.accumulated_loss = 0
        self.n_epoch = 0


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

        if self.args['model'] == 'bert':
            config = BertConfig.from_pretrained(
                model_name, 
                cache_dir=cache_path+'/model/config/')
        elif self.args['model'] == 'roberta':
            config = RobertaConfig.from_pretrained(
                model_name, 
                cache_dir=cache_path+'/model/config/')

        config.args = self.args
        model = PretrainedLMModel(config, cache_path, model_name)

        model.to(torch.device(self.device))

        return model, config
    

    def compute_loss(self, batch_logit, batch_labels, reduce=True):
        loss = self.trainer.compute_loss(
            batch_logit, 
            batch_labels)
        if reduce:
            loss = torch.mean(loss)
        return loss


    def set_optimizer(self, model):
        if self.args['optimizer_type'] == 'trans':
            optim, lr_scheduler = self.trainer.set_optimizer(model.parameters())
        else : # self.args['optimizer_type'] == 'legacy':
            optim, lr_scheduler = self.trainer.set_legacy_optimizer(model.parameters())
        return optim, lr_scheduler


    def backward_step(self, it, n_updates, model, loss, accumulated_loss, optimizer, lr_scheduler):
        accumulated_loss = self.trainer.backward_step(
            it, 
            n_updates,
            model,
            loss, 
            accumulated_loss,
            optimizer,
            lr_scheduler
        )
        return accumulated_loss


    def evaluate(self, 
                 model, 
                 valid_loader, 
                 test_loader, 
                 prediction_type=None,
                 compute_loss=True):
        valid_loss, valid_metrics, valid_size = self.trainer.evaluate(
            model, 
            valid_loader, 
            prediction_type,
            compute_loss)
        test_loss, test_metrics, test_size = self.trainer.evaluate(
            model, 
            test_loader, 
            prediction_type,
            compute_loss)

        pp.pprint([
            "Evaluation:",
            self.n_updates,
            valid_loss.item(),
            valid_metrics,
            valid_size,
            test_loss.item(),
            test_metrics,
            test_size
            ])
        print("", flush=True)


    def save_model(self, model, optimizer):
        ckpt_name = "-".join([
            self.args['dataset'], 
            self.args['task'],
            str(self.n_updates),
            str(self.n_epoch)]) + ".ckpt"
        save_path = self.args['save_dir'] + ckpt_name
        print("Saving Model to:", save_path)
        save_state = {
            'n_updates': self.n_updates,
            'epoch': self.n_epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(save_state, save_path)
        print("Saving Model to:", save_path, "...Finished.")


    def load_model_from_ckeckpoint(self, n_updates, n_epoch, model, optimizer):
        print("Loading Model from:", save_path)
        ckpt_name = "-".join([
            self.args['dataset'], 
            self.args['task'],
            str(n_updates),
            str(n_epoch)]) + ".ckpt"
        save_path = self.args['save_dir'] + ckpt_name
        state = torch.load(save_path)

        model.load_state_dict(state['state_dict'])
        if self.args['load_optimizer']:
            optimizer.load_state_dict(state['optimizer'])

        self.n_updates = 0 # renewed
        self.n_epoch = 0 # renewed

        print("Loading Model from:", save_path, "...Finished.")
        return model, optimizer


    def train(self):
        # 1. build dataset for train/valid/test
        print("Build dataset for train/valid/test", flush=True)
        tokenizer = self.load_tokenizer()
        dataset = self.dataset.build_datasets(tokenizer)
        train_loader, valid_loader, test_loader = self.dataset.build_dataloaders(dataset)
        print("Dataset Loaded:", self.args['dataset'], "with labels:", self.dataset.load_label_names())
        
        if self.args['task'] == "vad-from-categories":
            print("Build emobank for valid/test", flush=True)
            args_ = self.args.copy()
            args_['dataset'] = 'emobank'
            d = EmotionDataset(args_)
            dataset = d.build_datasets(tokenizer)
            eb_train_loader, eb_valid_loader, eb_test_loader = d.build_dataloaders(dataset)
            print("Dataset Loaded:", args_['dataset'], "with labels:", d.load_label_names())

        # 2. build/load models
        print("build/load models", flush=True)
        model, config = self.load_model()
        #print(model, flush=True)
        print(config, flush=True)
        #print(config.args["labels"])
        optimizer, lr_scheduler = self.set_optimizer(model)

        self.set_train_vars()
        if self.args['load_ckeckpoint']:
            model, optimizer = self.load_model_from_ckeckpoint(
                self.args['load_n_it'], 
                self.args['load_n_epoch'], 
                model, 
                optimizer)
        optimizer.zero_grad()

        while self.n_updates != self.args['total_n_updates'] and self.n_epoch != self.args['max_epoch']: 

            for it, train_batch in enumerate(train_loader):
                model.train()

                input_ids = train_batch[0].to(torch.device(self.device))        #[batch_size, max_len]
                attention_masks = train_batch[1].to(torch.device(self.device))  #[batch_size, max_len]
                labels = train_batch[2].to(torch.device(self.device))           #[batch_size, n_labels]

                # forward
                logits = model(                                                 #[batch_size, n_labels]
                    input_ids,
                    attention_mask=attention_masks)

                # compute loss
                loss = self.compute_loss(logits, labels) # [1] if reduce=True
                self.accumulated_loss += loss

                # backward
                self.accumulated_loss, self.n_updates = self.backward_step(
                    it, 
                    self.n_updates,
                    model,
                    loss, 
                    self.accumulated_loss, 
                    optimizer, 
                    lr_scheduler)

                # evaluation
                #if it == 0  : # eval every epoch
                if it == 0 and self.n_updates != 0 : # eval (and save) every epoch
                    self.n_epoch += 1; print("Epoch:", self.n_epoch, flush=True)
                    if self.args['task'] != 'vad-from-categories':
                        self.evaluate(
                            model, 
                            valid_loader, 
                            test_loader)
                    else: # self.args['task'] == 'vad-from-categories
                        self.evaluate(
                            model, 
                            valid_loader, 
                            test_loader,
                            prediction_type='cat')
                        self.evaluate(
                            model, 
                            eb_valid_loader,
                            eb_test_loader,
                            prediction_type='vad',
                            compute_loss=False)
                    
                    if self.args['save_model']:
                        self.save_model(model, optimizer)

                if self.n_updates == self.args['total_n_updates']: break
                if self.n_epoch == self.args['max_epoch']: break
                

def main():

    args = {

        'CUDA_VISIBLE_DEVICES': "3",

        # task and models
        'task': 'vad-from-categories', # ['category-classification', 'vad-regression', 'vad-from-categories']
        'label-type': 'categorical', # ['categorical', 'dimensional']
        'model': 'roberta', # ['bert', 'roberta'],
        'load_pretrained_lm_weights': True, # if false, only using architecture, randomly init weights.
        'dataset': 'semeval', # ['semeval', 'emobank', 'isear', 'ssec']
        'load_model': 'pretrained_lm', # else: fine_tuned_lm
        'use_emd': False, # if False, use Cross-entropy loss

        # memory-args
        'max_seq_len': 256,
        'train_batch_size': 32,
        'eval_batch_size': 32,
        'update_freq': 2,

        # optim-args
        'optimizer_type' : 'legacy', # ['legacy', 'trans']
        'learning_rate': 2e-05,
        'total_n_updates': 10000,
        'max_epoch': 40,
        'warmup_proportion': 0.1,
        'clip_grad': 1.0,

        # save & load args
        'save_model': False,
        'load_ckeckpoint': False,
        'load_optimizer': False,
        'load_n_epoch': None,
        'load_n_it': None,
        'save_dir': "/data/private/Emotion/ckpt/trained/",

        # etc
        'log_updates': False,
    }

    sdt = SingleDatasetTrainer(args)
    sdt.train()


if __name__ == "__main__":
    main()