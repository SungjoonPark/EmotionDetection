import os
import re
import string
import csv
import pandas as pd
import preprocessor as p
import emoji
import lxml.html
import lxml.html.clean
from sklearn.model_selection import train_test_split



class EmotionDatasetLoader():

    def __init__(self, emotion_labels, emotion_label_type):
        assert type(emotion_labels) == list
        assert emotion_label_type in ['cat', 'dim']
        self.rel_path = os.path.dirname(__file__)
        self.labels = emotion_labels
        self.label_type = emotion_label_type
        self.split_names = ['train', 'valid', 'test']
        self.data_types = ['text', 'label']


    def load_data(self):
        """
        load data: should return data:

        data = {
            'train':
                'text': [sent1, sent2, ..] (normalized text)
                'label: [(label1), (label2), ...] (int/floats)
            'valid':
                'text': [sent1, sent2, ..] (normalized text)
                'label: [(label1), (label2), ...] (int/floats)
            'test':
                'text': [sent1, sent2, ..] (normalized text)
                'label: [(label1), (label2), ...] (int/floats)             
        }
        """
        raise NotImplementedError


    def check_number_of_data(self):
        data = self.load_data()
        for s in data.keys():
            for t in data[s].keys():
                print(s, t, len(data[s][t]))


    def _get_emotion_label_VAD_scores(self):
        vad_score_dict = {}
        dir_path = os.path.join(self.rel_path, "./../../data/NRC-VAD/")
        vad_scores = pd.read_csv(dir_path + "NRC-VAD-Lexicon.txt", sep='\t', index_col='Word')
        for w, (v, a, d) in vad_scores.iterrows():
            vad_score_dict[w] = (round(v, 3), round(a, 3), round(d, 3))
        return vad_score_dict


    def get_vad_coordinates_of_labels(self):
        assert 'V' not in self.labels # for categorical labels
        total_label_vad_score_dict = self._get_emotion_label_VAD_scores()
        label_vad_score_dict = {}
        for e in self.labels:
            label_vad_score_dict[e] = total_label_vad_score_dict[e]
        return label_vad_score_dict



# https://github.com/JULIELab/EmoBank
class EmobankLoader(EmotionDatasetLoader):

    def __init__(self):
        emotion_labels = ['V', 'A', 'D']
        super(EmobankLoader, self).__init__(emotion_labels, 'dim')
        self.path = os.path.join(self.rel_path, "./../../data/Emobank/")


    def _load_data(self):
        eb = pd.read_csv(self.path + 'emobank.csv', index_col=0)
        meta = pd.read_csv(self.path + 'meta.tsv', sep='\t', index_col=0)
        return eb.join(meta, how='inner')


    def _split_data(self, eb, validate=True, save=False):
        tmp, test = train_test_split(eb.index, stratify=eb.category, random_state=42, test_size=1000)
        train, dev = train_test_split(tmp, stratify=eb.loc[tmp].category, random_state=42, test_size=1000)
        if validate or save:
            relfreqs = {}
            splits = {'train':train, 'dev': dev, 'test':test}
            for key, split in splits.items():
                relfreqs[key] = eb.loc[split].category.value_counts() / len(split)
            rf = pd.DataFrame(relfreqs).round(3)
            print(rf)
        if save:
            for key, split in splits.items():
                eb.loc[split, 'split'] = key
            eb = eb.drop(columns=['document', 'category', 'subcategory'])
            eb = eb[['split', 'V', 'A', 'D', 'text']]
            eb.to_csv(self.path + 'emobank_split.csv')
        train = eb.loc[train]
        dev = eb.loc[dev]
        test = eb.loc[test]
        return train, dev, test


    def _split_emobank_df(self, validate=True, save=True):
        eb = self._load_data()
        train, valid, test = self._split_data(eb, validate, save)
        return train, valid, test


    def validate_splits(self):
        splits = self._split_emobank_df()
        splits_ = self._load_split_dfs()
        for s, s_, s_name in zip(splits, splits_, self.split_names):
            validate = sorted(s.index) == sorted(s_.index)
            print(s_name, validate)


    def _load_split_dfs(self):
        data = []
        eb = pd.read_csv(self.path + 'emobank.csv', index_col=0)
        for split in ['train', 'dev', 'test']:
            s = eb[eb['split'] == split]
            data.append(s)
        return data


    def _preprocessing_text(self, text):
        """
        strip " and whitespace for every text
        """
        cleaned_text = []
        for t in text:
            t = t.strip('"').strip()
            t = re.sub(r'([{}])'.format(string.punctuation), r' \1 ', t)
            t = re.sub('\s{2,}', ' ', t) # pad punctuations for bpe
            cleaned_text.append(t)
        return cleaned_text


    def load_data(self, preprocessing=True):
        data = {}
        splits = self._load_split_dfs()
        for s_name, s_data in zip(self.split_names, splits):
            text = s_data['text'].to_list()
            if preprocessing:
                text = self._preprocessing_text(text)
            valence = s_data['V'].to_list()
            arousal = s_data['A'].to_list()
            dominance = s_data['D'].to_list()
            labels = [(v, a, d) for v, a, d in zip(valence, arousal, dominance)]
            data[s_name] = {}
            for name, d in zip(self.data_types, [text, labels]):
                data[s_name][name] = d
        return data



class SemEvalLoader(EmotionDatasetLoader):

    def __init__(self):
        emotion_labels = [
            "anger", "anticipation", "disgust", 
            "fear", "joy", "love", "optimism", 
            "pessimism", "sadness", "surprise", "trust"]
        super(SemEvalLoader, self).__init__(emotion_labels, 'cat')
        self.path = os.path.join(self.rel_path, './../../data/SEMEVAL2018/E-c/')
    

    def _load_split_files(self):
        splits = []
        file_prefix = "2018-E-c-En-"
        for s_name in self.split_names:
            if s_name == 'test': s_name = 'test-gold'
            if s_name == 'valid': s_name = 'dev'
            file_path = self.path + file_prefix + s_name + '.txt'
            split = pd.read_csv(file_path, index_col=0, sep='\t')
            splits.append(split)
        return splits


    def load_data(self, preprocessing=True):
        data = {}
        splits = self._load_split_files()
        for s_name, s_data in zip(self.split_names, splits):
            text = s_data['Tweet'].to_list()
            if preprocessing:
                text = self._preprocessing_text(text)
            
            emotions = []
            for e in self.labels:
                emotion = s_data[e].to_list()
                emotions.append(emotion)
            labels = [e for e in zip(*emotions)]

            data[s_name] = {}
            for name, d in zip(self.data_types, [text, labels]):
                data[s_name][name] = d

        return data


    def _preprocessing_text(self, tweet):
        """
        strip " and whitespace for every text
        """
        p.set_options(p.OPT.URL, p.OPT.MENTION) # remove url and metions
        html_cleaner = lxml.html.clean.Cleaner(style=True)
        cleaned_tweets = []

        for t in tweet:
            # 1. remove urls and mentions
            t = p.clean(t)

            # 2. convert html contents
            t = lxml.html.fromstring(t)
            t = html_cleaner.clean_html(t)
            t = t.text_content()

            # 3. clean some puncs and whitespaces
            t = t.replace("“", "\"").replace("”", "\"") # normalize “ ”
            t = re.sub(r'([{}])'.format(string.punctuation), r' \1 ', t)
            t = re.sub('\s{2,}', ' ', t) # pad punctuations for bpe
            t = re.sub(r"\\n", " ", t) # remove explicit \n
            t = re.sub(" +", " ", t) # remove duplicated whitespaces

            # 4. convert emojis to text (their names)
            t = emoji.demojize(t)
            t = re.sub('(:\S+:)', r" \1 ", t) # pad emojis for bpe
            t = t.replace("::", ": :") # pad emojis for bpe
            t = re.sub('\s{2,}', ' ', t) # pad emojis for bpe

            # strip whitespaces
            t = t.strip()

            cleaned_tweets.append(t)

        return cleaned_tweets



class ISEARLoader(EmotionDatasetLoader):

    def __init__(self):
        emotion_labels = [
            'fear', 'anger', 'guilt', 
            'joy', 'disgust', 'shame', 'sadness']
        super(ISEARLoader, self).__init__(emotion_labels, 'cat')
        self.path = os.path.join(self.rel_path, "./../../data/ISEAR/")
    

    def _preprocessing_text(self, text):
        text = text.replace("á\n", "\n")
        text = text.replace("\n", "")
        text = re.sub(" +", " ", text)
        text.strip(string.punctuation).strip()
        text = re.sub(r'([{}])'.format(string.punctuation), r' \1 ', text)
        text = re.sub('\s{2,}', ' ', text) # pad punctuations for bpe
        text = text.strip()
        return text


    def _load_xls(self):
        df = pd.read_excel(self.path + 'isear_databank.mdb.xls')
        df = df[['SIT', 'EMOT', 'Field1']]

        data = []
        # (3.0, 'anger'), (7.0, 'guilt'), (5.0, 'disgust'), (6.0, 'shame'), (4.0, 'sadness'), (1.0, 'joy'), (2.0, 'fear')
        for id_, (raw_text, emotion_id, emotion_text) in df.iterrows():
            text = self._preprocessing_text(raw_text)
            data.append((text, emotion_text))
            #print(text, emotion_text)
        data = pd.DataFrame(data, columns=self.data_types)
        return data


    def _split_data(self, data):
        train, _ = train_test_split(data.index, stratify=data.label, random_state=42, test_size=0.3)
        valid, test = train_test_split(_, stratify=data.loc[_].label, random_state=42, test_size=0.5)
        train = data.loc[train]
        valid = data.loc[valid]
        test = data.loc[test]
        return train, valid, test


    def load_data(self):
        df = self._load_xls()
        splits = self._split_data(df)

        data = {}
        for s_name, s_data in zip(self.split_names, splits):
            text = s_data[self.data_types[0]].to_list()
            labels = s_data[self.data_types[1]].to_list()
            data[s_name] = {}
            for name, d in zip(self.data_types, [text, labels]):
                data[s_name][name] = d
        return data



# http://www.romanklinger.de/ssec/
# Only use our best annotations train-combined-0.0.csv and test-combined-0.0.csv from this file. 
# There is a line-by-line correspondence.
class SSECLoader(EmotionDatasetLoader):

    def __init__(self):
        emotion_labels = [
            'anger', 'anticipation', 'disgust',
            'fear', 'joy', 'sadness', 
            'surprise', 'trust']
        super(SSECLoader, self).__init__(emotion_labels, 'cat')
        self.path = os.path.join(self.rel_path, "./../../data/SSEC/")
        # tweet processing from EmobankLoader()
        self._preprocessing_text = EmobankLoader()._preprocessing_text


    def _load_csv(self):
        train = pd.read_csv(self.path + 'train-combined-0.0.csv', sep='\t', header=None, skiprows=1828)
        # line 1828: XXXXXXXXXXXX EMPTY ANNOTATION
        test = pd.read_csv(self.path + 'test-combined-0.0.csv', sep='\t', header=None)
        train.columns = self.labels + ["text"]
        test.columns = self.labels + ["text"]
        return train, test


    def _preprocessing_df(self, df):
        # test processing
        tweets = df['text'].to_list()
        processed_tweets = self._preprocessing_text(tweets)
        df['text'] = processed_tweets
        # emotion label processing
        for emotion in self.labels:
            emotion_text = df[emotion].to_list()
            emotion_binary = [1 if et.lower() == emotion else 0 for et in emotion_text]
            df[emotion] = emotion_binary
        return df


    def _split_train(self, train):
        train_idx, valid_idx = train_test_split(train.index, random_state=42, test_size=0.2)
        train_set = train.loc[train_idx]
        valid_set = train.loc[valid_idx]
        return train_set, valid_set


    def load_data(self):
        train, test = self._load_csv()
        train = self._preprocessing_df(train)
        test = self._preprocessing_df(test)
        train, valid = self._split_train(train)
        splits = [train, valid, test]

        data = {}
        for s_name, s_data in zip(self.split_names, splits):
            text = s_data['text'].to_list()
            emotions = []
            for e in self.labels:
                emotion = s_data[e].to_list()
                emotions.append(emotion)
            labels = [e for e in zip(*emotions)]
            data[s_name] = {}
            for name, d in zip(self.data_types, [text, labels]):
                data[s_name][name] = d
        return data



def main():

    print("---- EMOBANK ----")
    emobank = EmobankLoader()
    #data = emobank.load_data()
    #emobank.validate_splits()
    emobank.check_number_of_data()
    #print(data)

    print("---- SEMEVAL 2018 E-c ----")
    semeval = SemEvalLoader()
    #data = semeval.load_data()
    semeval.check_number_of_data()
    print(semeval.get_vad_coordinates_of_labels())
    #print(data)
    
    print("---- ISEAR ----")
    isear = ISEARLoader()
    #data = isear.load_data()
    isear.check_number_of_data()
    print(isear.get_vad_coordinates_of_labels())
    #print(data)

    print("---- SSEC ----")
    ssec = SSECLoader()
    #data = ssec.load_data()
    ssec.check_number_of_data()
    print(ssec.get_vad_coordinates_of_labels())
    #print(data)



if __name__ == "__main__":
    main()