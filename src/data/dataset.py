import torch
from torch.utils.data import (
    Dataset, 
    DataLoader,
    RandomSampler, 
    SequentialSampler, 
    TensorDataset,
    SubsetRandomSampler,
    )

from data.loader import EmobankLoader, SemEvalLoader, ISEARLoader, SSECLoader, GOEMOTIONSLoader, GOEMOTIONSEkmanLoader, IEMOCAPVADLoader


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
        elif self.args['dataset'] == 'goemotions':
            self.loader = GOEMOTIONSLoader()
        elif self.args['dataset'] =='ekman':
            self.loader = GOEMOTIONSEkmanLoader()
        elif self.args['dataset'] =='iemocap':
            self.loader = IEMOCAPVADLoader()

    def load_label_names_and_vads(self):
        return self.loader.labels, self.loader.get_vad_coordinates_of_labels()

    def load_label_names(self):
        return self.loader.labels

    def _load_data(self):
        return self.loader.load_data()
    
    # https://mccormickml.com/2019/07/22/BERT-fine-tuning/#22-parse
    def _tokenize_split(self, split, tokenizer, split_name):
        input_ids = []
        attention_masks = []
        labels = []

        for sent, l in zip(split['text'], split['label']):
            # `encode_plus` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            #   (5) Pad or truncate the sentence to `max_length`
            #   (6) Create attention masks for [PAD] tokens.
            if sent == "": 
                print("An empty sentence example encountered. (after preprocessing): skipping... (", split_name, "set )")
                continue # if sentence is blank preprocessing, skip the example (emobank-train) 
            
            encoded_dict = tokenizer.encode_plus(
                                sent,                      # Sentence to encode.
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                max_length = self.args['max_seq_len'],           # Pad & truncate all sentences.
                                pad_to_max_length = True,
                                return_attention_mask = True,   # Construct attn. masks.
                                return_tensors = 'pt',     # Return pytorch tensors.
                                truncation=True
                        )
            
            # Add the encoded sentence to the list.    
            if len(encoded_dict['input_ids']) > self.args['max_seq_len']:
                input_id = encoded_dict['input_ids'][:self.args['max_seq_len']]
                print("An example has longer sequence > max_seq_len")
            else:
                input_id = encoded_dict['input_ids']
            input_ids.append(input_id)
            
            # And its attention mask (simply differentiates padding from non-padding).
            if len(encoded_dict['attention_mask']) > self.args['max_seq_len']:
                attention_mask = encoded_dict['attention_mask'][:self.args['max_seq_len']]
                print("An example has longer attention mask > max_seq_len")
            else:
                attention_mask = encoded_dict['attention_mask']
            attention_masks.append(attention_mask)

            # add labels
            labels.append(l)

        # Convert the lists into tensors.
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels)
        return input_ids, attention_masks, labels


    def tokenize_dataset(self, data_dict, tokenizer):
        for split_name in self.loader.split_names:
            input_ids, attention_masks, labels = self._tokenize_split(data_dict[split_name], tokenizer, split_name)
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
                sampler = SequentialSampler(dataset)
                batch_size = self.args['eval_batch_size']
                # sampler = SequentialSampler
                # batch_size = self.args['eval_batch_size']
            else:
                # sampler = RandomSampler
                # batch_size = self.args['train_batch_size']
                if self.args['few_shot_ratio'] == 1:
                    sampler = RandomSampler(dataset)
                    batch_size = self.args['train_batch_size']
                else:
                    print("dataset length is",len(dataset) )
                    subset = int(len(dataset) * self.args['few_shot_ratio'])
                    print("subset is ", subset)
                    # dataset = dataset[:subset]
                    # print("dataset subset", dataset[:subset])
                    # print("dataset length after is",len(dataset) )
                    # sampler = RandomSampler(dataset[:subset])
                    # indices = torch.randperm(len(dataset))[:subset]
                    indices = torch.arange(0,subset)
                    print("indices", indices)
                    sampler = SubsetRandomSampler(indices)
                    batch_size = self.args['train_batch_size']

            dataloader = DataLoader(
                dataset,
                sampler = sampler,
                batch_size = batch_size,
            #    shuffle = True, # sampler option is mutually exclusive with shuffle
            )
            dataloaders.append(dataloader)

        return dataloaders