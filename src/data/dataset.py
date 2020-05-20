import torch
from torch.utils.data import (
    Dataset, 
    DataLoader,
    RandomSampler, 
    SequentialSampler, 
    TensorDataset
    )

from data.loader import EmobankLoader, SemEvalLoader, ISEARLoader, SSECLoader


class EmotionDataset():

    def __init__(self, args):
        self.args = args
        self._set_loader()

    def _set_loader(self):
        if self.args['dataset'] == 'emobank':
            self.loader = EmobankLoader()
        elif self.args['dataset'] == 'semeval':
            self.loader = SemEvalLoader()
        elif self.args['dataset'] == 'isear':
            self.loader = ISEARLoader()
        elif self.args['dataset'] == 'ssec':
            self.loader = SSECLoader()

    def load_label_names_and_vads(self):
        return self.loader.labels, self.loader.get_vad_coordinates_of_labels()

    def load_label_names(self):
        return self.loader.labels

    def _load_data(self):
        return self.loader.load_data()
    
    # https://mccormickml.com/2019/07/22/BERT-fine-tuning/#22-parse
    def _tokenize_split(self, split, tokenizer):
        input_ids = []
        attention_masks = []
        for sent in split['text']:
            # `encode_plus` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            #   (5) Pad or truncate the sentence to `max_length`
            #   (6) Create attention masks for [PAD] tokens.
            encoded_dict = tokenizer.encode_plus(
                                sent,                      # Sentence to encode.
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                max_length = self.args['max_seq_len'],           # Pad & truncate all sentences.
                                pad_to_max_length = True,
                                return_attention_mask = True,   # Construct attn. masks.
                                return_tensors = 'pt',     # Return pytorch tensors.
                        )
            
            # Add the encoded sentence to the list.    
            input_ids.append(encoded_dict['input_ids'])
            
            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoded_dict['attention_mask'])

        # Convert the lists into tensors.
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(split['label'])
        return input_ids, attention_masks, labels


    def tokenize_dataset(self, data_dict, tokenizer):
        for split_name in self.loader.split_names:
            input_ids, attention_masks, labels = self._tokenize_split(data_dict[split_name], tokenizer)
            data_dict[split_name]['input_ids'] = input_ids
            data_dict[split_name]['attention_masks'] = attention_masks
            data_dict[split_name]['label'] = labels
        return data_dict


    def _decode_ids(self, input_ids, tokenizer):
        return tokenizer.decode(input_ids)


    def build_datasets(self, tokenizer):
        data_dict = self._load_data()
        data_dict = self.tokenize_dataset(data_dict, tokenizer)
        #print(data_dict['train']['text'][0])
        #print(data_dict['train']['input_ids'][0])
        #print(self._decode_ids(data_dict['train']['input_ids'][0], tokenizer))
        #print(data_dict['train']['attention_masks'][0])
        #print(data_dict['train']['label'][0])
        return data_dict


    def build_dataloaders(self, data_dict):
        dataloaders = []
        for split_name in self.loader.split_names:
            dataset = TensorDataset(
                data_dict[split_name]['input_ids'], 
                data_dict[split_name]['attention_masks'], 
                data_dict[split_name]['label'])

            if split_name != 'train':
                sampler = SequentialSampler
                batch_size = self.args['eval_batch_size']
            else:
                sampler = RandomSampler
                batch_size = self.args['train_batch_size']

            dataloader = DataLoader(
                dataset,
                #sampler = sampler(dataset),
                batch_size = batch_size,
                shuffle = False,   
            )
            dataloaders.append(dataloader)

        return dataloaders