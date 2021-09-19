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
    EmotionDataset,
    GOEMOTIONSLoader,
    GOEMOTIONSEkmanLoader,
)

from models import (
    PretrainedLMModel,
    Trainer,
)
import argparse
from argparse import ArgumentParser
import json
import sys
import csv

import logging
logging.basicConfig(level=logging.ERROR)

pp = pprint.PrettyPrinter(indent=1, width=90)


class SingleDatasetTrainer():

    def __init__(self, args):
        # args
        os.environ["CUDA_VISIBLE_DEVICES"]= args.CUDA_VISIBLE_DEVICES
        self._check_args(args)
        self.args = args

        # output file
        self.csv_file_name = args.csv_file_name

        # cache path
        self.cache_dir = "./../ckpt/"
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        # dataset class
        self.dataset = EmotionDataset(args)
        if args.task == 'vad-from-categories':
            label_names, label_vads = self.dataset.load_label_names_and_vads()
            self.args.label_names = label_names
            self.args.label_vads = label_vads
        else:
            self.args.label_names = self.dataset.load_label_names()

        # trainer class
        self.trainer = Trainer(args)
        self.device = self.trainer.args.device
        self.coef_step =0

    def _check_args(self, args):
        assert args.task in ['category-classification', 'vad-regression', 'vad-from-categories']
        assert args.model in ['bert', 'roberta']
        assert args.dataset in ['semeval', 'emobank', 'isear', 'ssec', 'goemotions', 'ekman']
        assert args.load_model in ['pretrained_lm', 'fine_tuned_lm']
        assert args.label_type in ['categorical', 'dimensional']
        assert args.optimizer_type in ['legacy', 'trans']


    def _set_model_args(self):
        if self.args.model == 'bert':
            model_name = 'bert-large-cased-whole-word-masking'
        elif self.args.model == 'roberta':
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
        if self.args.model == 'bert':
            Tokenizer = BertTokenizer
        elif self.args.model == 'roberta':
            Tokenizer = RobertaTokenizer
        tokenizer = Tokenizer.from_pretrained(
            model_name, 
            cache_dir=cache_path+'/vocab/')
        return tokenizer


    def load_model(self):
        model_name, cache_path = self._set_model_args()

        if self.args.model == 'bert':
            config = BertConfig.from_pretrained(
                model_name, 
                cache_dir=cache_path+'/model/config/')
        elif self.args.model == 'roberta':
            config = RobertaConfig.from_pretrained(
                model_name, 
                cache_dir=cache_path+'/model/config/')

        config.args = self.args
        model = PretrainedLMModel(config, cache_path, model_name)

        model.to(torch.device(self.device))

        return model, config
    

    def compute_loss(self, inputs, lm_logit, batch_logit, batch_labels, reduce=True):
        loss = self.trainer.compute_loss(
            inputs,
            lm_logit,
            batch_logit, 
            batch_labels)
        if reduce:
            loss = torch.mean(loss)
        return loss


    def set_optimizer(self, model):
        if self.args.optimizer_type == 'trans':
            optim, lr_scheduler = self.trainer.set_optimizer(model.parameters())
        else : # self.args.optimizer_type == 'legacy':
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

    def evaluate_save_result(self,
                 epoch,
                 file_name,
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

        if prediction_type == 'vad':
            self._save_vad_results(epoch, file_name,
                            self.n_updates,
                            valid_loss.item(),
                            valid_metrics,
                            valid_size,
                            test_loss.item(),
                            test_metrics,
                            test_size)

        else:
            self._save_class_results(epoch, file_name,
                self.n_updates,
                valid_loss.item(),
                valid_metrics,
                valid_size,
                test_loss.item(),
                test_metrics,
                test_size, self.args.dataset)

        print("", flush=True)

    def _save_vad_results(self, epoch, file_name, n_updates, 
                    valid_loss, valid_metrics, valid_size,
                    test_loss, test_metrics, test_size):
        columns = ['epoch', 'file_name', 'n_updates', 'valid_size', 'test_size', 'valid_loss', 'test_loss',
                    'valid_v', 'valid_v_p', 'valid_a', 'valid_a_p', 'valid_d', 'valid_d_p',
                    'test_v', 'test_v_p', 'test_a', 'test_a_p', 'test_d', 'test_d_p']

        input_row = [epoch, file_name, n_updates,valid_size, test_size, valid_loss, test_loss]

        for vad_dim_value in valid_metrics:
            input_row.extend([valid_metrics[vad_dim_value][0],valid_metrics[vad_dim_value][1]])
        for vad_dim_value in test_metrics:
            input_row.extend([test_metrics[vad_dim_value][0],test_metrics[vad_dim_value][1]])
        self._write_items_in_metric(columns, file_name, input_row)


    def _save_class_results(self, epoch, file_name, n_updates, valid_loss, valid_metrics, valid_size, test_loss, test_metrics, test_size, dataset):  
        input_row = [epoch, file_name, n_updates,valid_size, test_size, valid_loss, test_loss]

        if dataset != 'isear':
            columns = ['epoch', 'file_name', 'n_updates', 'valid_size', 'test_size', 'valid_loss', 'test_loss',
                        'valid_microF1', 'valid_macroF1', 'valid_jaccard_micro', 'valid_jaccard_macro', 'valid_acc', 
                        'test_microF1','test_macroF1', 'test_jaccard_micro', 'test_jaccard_macro', 'test_acc']

            input_row.extend([valid_metrics['classification']['micro avg']['f1-score'],valid_metrics['classification']['macro avg']['f1-score'],
            valid_metrics['jaccard_score']['micro'], valid_metrics['jaccard_score']['macro'], valid_metrics['jaccard_score']['samples']])

            input_row.extend([test_metrics['classification']['micro avg']['f1-score'],test_metrics['classification']['macro avg']['f1-score'], 
            test_metrics['jaccard_score']['micro'], test_metrics['jaccard_score']['macro'], test_metrics['jaccard_score']['samples']])
        else:
            columns = ['epoch', 'file_name', 'n_updates', 'valid_size', 'test_size', 'valid_loss', 'test_loss',
                        'valid_macroF1', 'valid_weightF1', 'valid_acc', 
                        'test_macroF1', 'test_weightF1', 'test_acc']
            if valid_metrics.get('classification') is None:
                return
            
            input_row.extend([valid_metrics['classification']['macro avg']['f1-score'],
            valid_metrics['classification']['weighted avg']['f1-score'], valid_metrics['classification']['accuracy']])

            input_row.extend([test_metrics['classification']['macro avg']['f1-score'],
            test_metrics['classification']['weighted avg']['f1-score'], test_metrics['classification']['accuracy']])
        self._write_items_in_metric(columns, file_name, input_row)

    def _write_items_in_metric(self, columns, file_name, input_row):
        savefile = "csv_result/{}.csv".format(file_name)
        file_exists = os.path.isfile(savefile)
        with open(savefile, 'a') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(columns)
            writer.writerow(input_row)

    def save_model(self, model, optimizer):
        ckpt_name = "-".join([
            self.args.dataset, 
            self.args.task,
            str(self.n_updates),
            str(self.n_epoch)]) + ".ckpt"
        save_path = self.args.save_dir + ckpt_name
        print("Saving Model to:", save_path)
        save_state = {
            'n_updates': self.n_updates,
            'epoch': self.n_epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        if not os.path.isfile(save_path):
            f = open(save_path, "x")
        torch.save(save_state, save_path)
        print("Saving Model to:", save_path, "...Finished.")


    def load_model_from_ckeckpoint(self, n_updates, n_epoch, model, optimizer):
        # 1. set path and load states
        ckpt_name = "-".join([
            self.args.load_dataset, 
            self.args.load_task,
            str(n_updates),
            str(n_epoch)]) + ".ckpt"
        save_path = self.args.save_dir + ckpt_name
        print("Loading Model from:", save_path)
        state = torch.load(save_path)
        
        # 2. load (override) pre-trained model (without head)
        model_dict = model.state_dict()
        ckpt__dict = state['state_dict']
        ckpt__dict_head_removed = {k: v for k, v in ckpt__dict.items() if k not in ['head.bias', 'head.weight']}
        model_dict.update(ckpt__dict_head_removed) 
        model.load_state_dict(model_dict, strict=False)

        # 3. recover optim state
        if self.args.load_optimizer:
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
        print("Dataset Loaded:", self.args.dataset, "with labels:", self.dataset.load_label_names())
        
        if self.args.task == "vad-from-categories":
            print("Build emobank for valid/test", flush=True)
            args_ = argparse.Namespace(**vars(self.args))
            args_.dataset = 'emobank'
            d = EmotionDataset(args_)
            dataset = d.build_datasets(tokenizer)
            eb_train_loader, eb_valid_loader, eb_test_loader = d.build_dataloaders(dataset)
            print("Dataset Loaded:", args_.dataset, "with labels:", d.load_label_names())

        # 2. build/load models
        print("build/load models", flush=True)
        model, config = self.load_model()
        optimizer, lr_scheduler = self.set_optimizer(model)

        self.set_train_vars()
        if self.args.load_ckeckpoint:
            model, optimizer = self.load_model_from_ckeckpoint(
                self.args.load_n_it, 
                self.args.load_n_epoch, 
                model, 
                optimizer)

        if len(self.args.CUDA_VISIBLE_DEVICES.split(',')) > 1:
            device_list = self.args.CUDA_VISIBLE_DEVICES.split(',')
            print("Using %d GPUs now" % len(device_list))
            map_object = map(int, device_list)
            print("Using GPU list of", list(map_object))
            model = torch.nn.DataParallel(model)

        optimizer.zero_grad()
        lr_switch = 0

        while self.n_updates != self.args.total_n_updates and self.n_epoch != self.args.max_epoch:

            for it, train_batch in enumerate(train_loader):
                model.train() 
                if self.args.task == "vad-regression" and self.args.load_ckeckpoint is True:
                    if self.n_epoch < self.args.max_freeze_epoch:
                        for para in model.parameters():
                            para.requires_grad = False
                        model.v_head.weight.requires_grad = True
                        model.a_head.weight.requires_grad = True
                        model.d_head.weight.requires_grad = True
                    else: 
                        if self.n_epoch == self.args.max_freeze_epoch and lr_switch==0:
                            for g in optimizer.param_groups:
                                g['lr'] = self.args.learning_rate_unfreeze
                            lr_switch = 1
                        for para2 in model.parameters():
                            para2.requires_grad = True
                

                input_ids = train_batch[0].to(torch.device(self.device))        #[batch_size, max_len]
                attention_masks = train_batch[1].to(torch.device(self.device))  #[batch_size, max_len]
                labels = train_batch[2].to(torch.device(self.device))           #[batch_size, n_labels]

                # forward
                lm_logits, cls_logits = model(                                                 #[batch_size, n_labels]
                    input_ids,
                    attention_mask=attention_masks, n_epoch=self.n_epoch)

                # compute loss
                loss = self.compute_loss(input_ids, lm_logits, cls_logits, labels) # [1] if reduce=True
                self.accumulated_loss += loss

                # backward
                print('Epoch-{0} lr: {1}'.format(self.n_epoch, optimizer.param_groups[0]['lr']))
                self.accumulated_loss, self.n_updates = self.backward_step(
                    it, 
                    self.n_updates,
                    model,
                    loss, 
                    self.accumulated_loss, 
                    optimizer, 
                    lr_scheduler)
                
                self.coef_step +=1

                # evaluation
                if it == 0 and self.n_updates != 0 : # eval (and save) every epoch
                    self.n_epoch += 1; print("Epoch:", self.n_epoch, flush=True)
                    if self.args.task == 'vad-from-categories':
                        self.evaluate_save_result(
                            self.n_epoch,
                            self.csv_file_name,
                            model, 
                            valid_loader, 
                            test_loader,
                            prediction_type='cat')
                        self.evaluate_save_result(
                            self.n_epoch,
                            self.csv_file_name,
                            model, 
                            eb_valid_loader,
                            eb_test_loader,
                            prediction_type='vad',
                            compute_loss=False)
                    elif self.args.task == 'vad-regression':
                        self.evaluate_save_result(
                            self.n_epoch,
                            self.csv_file_name,
                            model, 
                            valid_loader, 
                            test_loader,
                            prediction_type='vad')
                    else:
                        self.evaluate_save_result(
                            self.n_epoch,
                            self.csv_file_name,
                            model, 
                            valid_loader, 
                            test_loader)
                    
                    if self.args.save_model:
                        self.save_model(model, optimizer)

                if self.n_updates == self.args.total_n_updates: break
                if self.n_epoch == self.args.max_epoch: break
                

def main():
    parser = ArgumentParser()
    parser.add_argument('--config', default=None, type=str)
    arg_ = parser.parse_args()
    if arg_.config == None:
        raise NameError("Include a config file in the argument please.")

    #Getting configurations
    with open(arg_.config) as config_file:
        args = json.load(config_file)
    args = argparse.Namespace(**args)

    sdt = SingleDatasetTrainer(args)
    sdt.train()


if __name__ == "__main__":
    main()