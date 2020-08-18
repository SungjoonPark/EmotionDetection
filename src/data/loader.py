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
import numpy as np
import random


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
        vad_scores = pd.read_csv(
            dir_path + "NRC-VAD-Lexicon.txt", sep='\t', index_col='Word')
        for w, (v, a, d) in vad_scores.iterrows():
            vad_score_dict[w] = (round(v, 3), round(a, 3), round(d, 3))
        return vad_score_dict

    def get_vad_coordinates_of_labels(self):
        assert 'V' not in self.labels  # for categorical labels
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
        tmp, test = train_test_split(
            eb.index, stratify=eb.category, random_state=42, test_size=1000)
        train, dev = train_test_split(
            tmp, stratify=eb.loc[tmp].category, random_state=42, test_size=1000)
        if validate or save:
            relfreqs = {}
            splits = {'train': train, 'dev': dev, 'test': test}
            for key, split in splits.items():
                relfreqs[key] = eb.loc[split].category.value_counts() / \
                    len(split)
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
            t = t.strip('\"').strip('\'').strip()
            t = re.sub(r'([{}])'.format(string.punctuation), r' \1 ', t)
            t = re.sub('\s{2,}', ' ', t)  # pad punctuations for bpe
            t = t.strip()
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
            labels = [(v, a, d)
                      for v, a, d in zip(valence, arousal, dominance)]
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
        self.path = os.path.join(
            self.rel_path, './../../data/SEMEVAL2018/E-c/')

    def _load_split_files(self):
        splits = []
        file_prefix = "2018-E-c-En-"
        for s_name in self.split_names:
            if s_name == 'test':
                s_name = 'test-gold'
            if s_name == 'valid':
                s_name = 'dev'
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
        p.set_options(p.OPT.URL, p.OPT.MENTION)  # remove url and metions
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
            t = t.replace("“", "\"").replace("”", "\"")  # normalize “ ”
            t = t.strip('\"').strip('\'').strip()
            t = re.sub(r'([{}])'.format(string.punctuation), r' \1 ', t)
            t = re.sub('\s{2,}', ' ', t)  # pad punctuations for bpe
            t = re.sub(r"\\n", " ", t)  # remove explicit \n
            t = re.sub(" +", " ", t)  # remove duplicated whitespaces

            # 4. convert emojis to text (their names)
            t = emoji.demojize(t)
            t = re.sub('(:\S+:)', r" \1 ", t)  # pad emojis for bpe
            t = t.replace("::", ": :")  # pad emojis for bpe
            t = re.sub('\s{2,}', ' ', t)  # pad emojis for bpe

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
        text = text.strip('\"').strip('\'').strip()
        text = re.sub(" +", " ", text)
        text.strip(string.punctuation).strip()
        text = re.sub(r'([{}])'.format(string.punctuation), r' \1 ', text)
        text = re.sub('\s{2,}', ' ', text)  # pad punctuations for bpe
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
        train, _ = train_test_split(
            data.index, stratify=data.label, random_state=42, test_size=0.3)
        valid, test = train_test_split(
            _, stratify=data.loc[_].label, random_state=42, test_size=0.5)
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
            labels = [self.labels.index(l) for l in labels]
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
        train = pd.read_csv(self.path + 'train-combined-0.0.csv',
                            sep='\t', header=None, skiprows=1828)
        # line 1828: XXXXXXXXXXXX EMPTY ANNOTATION
        test = pd.read_csv(
            self.path + 'test-combined-0.0.csv', sep='\t', header=None)
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
            emotion_binary = [
                1 if et.lower() == emotion else 0 for et in emotion_text]
            df[emotion] = emotion_binary
        return df

    def _split_train(self, train):
        train_idx, valid_idx = train_test_split(
            train.index, random_state=42, test_size=0.2)
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

# https://github.com/Samarth-Tripathi/IEMOCAP-Emotion-Detection/blob/master/code/python_files/Experiments/Untitled.ipynb


class IEMOCAPCatLoader(EmotionDatasetLoader):

    def __init__(self):
        emotion_labels = [
            'anger', 'happy', 'sadness',
            'frustrate', 'excite', 'fear',
            'surprise', 'disgust', 'neutral']
        super(IEMOCAPCatLoader, self).__init__(emotion_labels, 'cat')
        self.path = os.path.join(self.rel_path, './../../data/IEMOCAP/')

    def _change_label_name_to_NRC_VAD_label(self, label):
        # Counter({'xxx': 2507, 'fru': 1849, 'neu': 1708, 'ang': 1103, 'sad': 1084, 'exc': 1041, 'hap': 595, 'sur': 107, 'fea': 40, 'oth': 3, 'dis': 2})
        # original_emotion_label = ['ang', 'hap', 'sad', 'fru', 'exc', 'fea', 'sur', 'neu', 'dis']#, 'oth',  'xxx']
        # original_label: {angry, happy, sad, neutral, frustrated, excited, fearful, surprised, disgusted, other}
        map_to_NRC_VAD_label_dict = {'ang': 'anger', 'hap': 'happy', 'sad': 'sadness', 'fru': 'frustrate',
                                     'exc': 'excite', 'fea': 'fear', 'sur': 'surprise', 'dis': 'disgust', 'neu': 'neutral'}
        return map_to_NRC_VAD_label_dict[label] if label in map_to_NRC_VAD_label_dict else label

    def _load_preprocess_raw_data(self):
        data = []

        sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
        for session in sessions:
            path_to_emotions = self.path + session + '/dialog/EmoEvaluation/'
            path_to_transcriptions = self.path + session + '/dialog/transcriptions/'

            files = os.listdir(path_to_emotions)
            files = [f[:-4] for f in files if f.endswith(".txt")]
            for f in files:
                transcriptions = self._get_transcriptions(
                    path_to_transcriptions, f + '.txt')
                emotions = self._get_emotions(path_to_emotions, f + '.txt')

                for ie, e in enumerate(emotions):
                    e['transcription'] = self._preprocessing_text(
                        transcriptions[e['id']])
                    e['emotion'] = self._change_label_name_to_NRC_VAD_label(
                        e['emotion'])
                    if e['emotion'] in self.labels:
                        data.append(e)
        sort_key = self._get_field(data, "id")
        preprocessed_data = np.array(data)[np.argsort(sort_key)]
        return preprocessed_data

    def _get_transcriptions(self, path_to_transcriptions, filename):
        f = open(path_to_transcriptions + filename, 'r').read()
        f = np.array(f.split('\n'))
        transcription = {}
        for i in range(len(f) - 1):
            g = f[i]
            i1 = g.find(': ')
            i0 = g.find(' [')
            ind_id = g[:i0]
            ind_ts = g[i1+2:]
            transcription[ind_id] = ind_ts
        return transcription

    def _get_emotions(self, path_to_emotions, filename):
        f = open(path_to_emotions + filename, 'r').read()
        f = np.array(f.split('\n'))
        idx = f == ''
        idx_n = np.arange(len(f))[idx]
        emotion = []
        for i in range(len(idx_n) - 2):
            g = f[idx_n[i]+1:idx_n[i+1]]
            head = g[0]
            actor_id = head[head.find(filename[:-4]) + len(filename[:-4]) + 1:
                            head.find(filename[:-4]) + len(filename[:-4]) + 5]
            emo = head[head.find('\t[') - 3:head.find('\t[')]
            vad = head[head.find('\t[') + 1:]

            v = float(vad[1:7])
            a = float(vad[9:15])
            d = float(vad[17:23])

            emotion.append({'id': filename[:-4] + '_' + actor_id,
                            'v': v,
                            'a': a,
                            'd': d,
                            'emotion': emo})
        return emotion

    def _get_field(self, data, key):
        return np.array([e[key] for e in data])

    def _preprocessing_text(self, text):
        t = re.sub(r'([{}])'.format(string.punctuation), r' \1 ', text)
        t = re.sub('\s{2,}', ' ', t)  # pad punctuations for bpe
        t = re.sub(r"\\n", " ", t)  # remove explicit \n
        t = re.sub(" +", " ", t)  # remove duplicated whitespaces
        clean_t = t.strip()
        return clean_t

    def _split_data(self, preprocessed_whole_data):
        num_dataset = list(range(len(preprocessed_whole_data)))
        random.seed(42)
        random.shuffle(num_dataset)
        train_len = int(len(num_dataset)*0.8)
        val_len = int(len(num_dataset)*0.1)
        train = preprocessed_whole_data[num_dataset[:train_len]]
        valid = preprocessed_whole_data[num_dataset[train_len:train_len+val_len]]
        test = preprocessed_whole_data[num_dataset[train_len+val_len:]]
        return train, valid, test

    def load_data(self):
        preprocessed_data = self._load_preprocess_raw_data()
        splits = self._split_data(preprocessed_data)

        data = {}
        for s_name, s_data in zip(self.split_names, splits):
            text = [s_row['transcription'] for s_row in s_data]
            labels = [s_row['emotion'] for s_row in s_data]
            data[s_name] = {}
            for name, d in zip(self.data_types, [text, labels]):
                data[s_name][name] = d
        return data


class IEMOCAPVADLoader(EmotionDatasetLoader):

    def __init__(self):
        emotion_labels = ['V', 'A', 'D']
        super(IEMOCAPVADLoader, self).__init__(emotion_labels, 'dim')
        self.path = os.path.join(self.rel_path, './../../data/IEMOCAP/')

    def _change_label_name_to_NRC_VAD_label(self, label):
        # Counter({'xxx': 2507, 'fru': 1849, 'neu': 1708, 'ang': 1103, 'sad': 1084, 'exc': 1041, 'hap': 595, 'sur': 107, 'fea': 40, 'oth': 3, 'dis': 2})
        # original_emotion_label = ['ang', 'hap', 'sad', 'fru', 'exc', 'fea', 'sur', 'neu', 'dis']#, 'oth',  'xxx']
        # original_label: {angry, happy, sad, neutral, frustrated, excited, fearful, surprised, disgusted, other}
        map_to_NRC_VAD_label_dict = {'ang': 'anger', 'hap': 'happy', 'sad': 'sadness', 'fru': 'frustrate',
                                     'exc': 'excite', 'fea': 'fear', 'sur': 'surprise', 'dis': 'disgust', 'neu': 'neutral'}
        return map_to_NRC_VAD_label_dict[label] if label in map_to_NRC_VAD_label_dict else label

    def _load_preprocess_raw_data(self):
        data = []
        emotion_labels = [
            'anger', 'happy', 'sadness',
            'frustrate', 'excite', 'fear',
            'surprise', 'disgust', 'neutral']
        sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
        for session in sessions:
            path_to_emotions = self.path + session + '/dialog/EmoEvaluation/'
            path_to_transcriptions = self.path + session + '/dialog/transcriptions/'

            files = os.listdir(path_to_emotions)
            files = [f[:-4] for f in files if f.endswith(".txt")]
            for f in files:
                transcriptions = self._get_transcriptions(
                    path_to_transcriptions, f + '.txt')
                emotions = self._get_emotions(path_to_emotions, f + '.txt')

                for ie, e in enumerate(emotions):
                    e['transcription'] = self._preprocessing_text(
                        transcriptions[e['id']])
                    e['emotion'] = self._change_label_name_to_NRC_VAD_label(
                        e['emotion'])

                    if e['emotion'] in emotion_labels:
                        data.append(e)

        sort_key = self._get_field(data, "id")
        preprocessed_data = np.array(data)[np.argsort(sort_key)]
        return preprocessed_data

    def _get_transcriptions(self, path_to_transcriptions, filename):
        f = open(path_to_transcriptions + filename, 'r').read()
        f = np.array(f.split('\n'))
        transcription = {}
        for i in range(len(f) - 1):
            g = f[i]
            i1 = g.find(': ')
            i0 = g.find(' [')
            ind_id = g[:i0]
            ind_ts = g[i1+2:]
            transcription[ind_id] = ind_ts
        return transcription

    def _get_emotions(self, path_to_emotions, filename):
        f = open(path_to_emotions + filename, 'r').read()
        f = np.array(f.split('\n'))
        idx = f == ''
        idx_n = np.arange(len(f))[idx]
        emotion = []
        for i in range(len(idx_n) - 2):
            g = f[idx_n[i]+1:idx_n[i+1]]
            head = g[0]
            actor_id = head[head.find(filename[:-4]) + len(filename[:-4]) + 1:
                            head.find(filename[:-4]) + len(filename[:-4]) + 5]
            emo = head[head.find('\t[') - 3:head.find('\t[')]
            vad = head[head.find('\t[') + 1:]

            v = float(vad[1:7])
            a = float(vad[9:15])
            d = float(vad[17:23])

            emotion.append({'id': filename[:-4] + '_' + actor_id,
                            'v': v,
                            'a': a,
                            'd': d,
                            'emotion': emo})
        return emotion

    def _get_field(self, data, key):
        return np.array([e[key] for e in data])

    def _preprocessing_text(self, text):
        t = re.sub(r'([{}])'.format(string.punctuation), r' \1 ', text)
        t = re.sub('\s{2,}', ' ', t)  # pad punctuations for bpe
        t = re.sub(r"\\n", " ", t)  # remove explicit \n
        t = re.sub(" +", " ", t)  # remove duplicated whitespaces
        clean_t = t.strip()
        return clean_t

    def _split_data(self, preprocessed_whole_data):
        num_dataset = list(range(len(preprocessed_whole_data)))
        random.seed(42)
        random.shuffle(num_dataset)
        train_len = int(len(num_dataset)*0.8)
        val_len = int(len(num_dataset)*0.1)
        train = preprocessed_whole_data[num_dataset[:train_len]]
        valid = preprocessed_whole_data[num_dataset[train_len:train_len+val_len]]
        test = preprocessed_whole_data[num_dataset[train_len+val_len:]]
        return train, valid, test

    def load_data(self):
        preprocessed_data = self._load_preprocess_raw_data()
        splits = self._split_data(preprocessed_data)

        data = {}
        for s_name, s_data in zip(self.split_names, splits):
            text = [s_row['transcription'] for s_row in s_data]
            labels = [(s_row['v'], s_row['a'], s_row['d']) for s_row in s_data]
            data[s_name] = {}
            for name, d in zip(self.data_types, [text, labels]):
                data[s_name][name] = d
        return data

class GOEMOTIONSLoader(EmotionDatasetLoader):

    def __init__(self):
        emotion_labels = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
                    'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
                    'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']
        super(GOEMOTIONSLoader, self).__init__(emotion_labels, 'cat')
        self.path = os.path.join(self.rel_path, "./../../data/original/")

    def _load_split_files(self):
        splits = []
        for s_name in self.split_names:
            if s_name == 'test':
                s_name = 'test'
            if s_name == 'valid':
                s_name = 'dev'
            file_path = self.path + s_name + '.tsv'
            split = pd.read_csv(file_path, sep='\t', names=['Text','Label','Alpha'])
            splits.append(split)
        return splits

    def _convert_to_one_hot_label(self,label,label_list_len):
        one_hot_label = [0] * label_list_len
        for l in label:
            l_int = int(l)
            one_hot_label[l_int] = 1
        return tuple(one_hot_label)

    def _preprocessing_text(self, text):
        """
        strip " and whitespace for every text
        """
        cleaned_text = []
        for t in text:
            t = t.strip('\"').strip('\'').strip()
            t = re.sub(r'([{}])'.format(string.punctuation), r' \1 ', t)
            t = re.sub('\s{2,}', ' ', t)  # pad punctuations for bpe
            t = t.strip()
            cleaned_text.append(t)
        return cleaned_text

    def load_data(self, preprocessing=True):
        data = {}
        splits = self._load_split_files()
        for s_name, s_data in zip(self.split_names, splits):
            text = s_data['Text'].to_list()
            if preprocessing:
                text = self._preprocessing_text(text)

            emotions = []
            for e in s_data['Label']:
                label_list = [int(s) for s in e.split(',')]
                emotion = self._convert_to_one_hot_label(label_list, len(self.labels))
                emotions.append(emotion)

            data[s_name] = {}
            for name, d in zip(self.data_types, [text, emotions]):
                data[s_name][name] = d
        return data


def main():

    # print("---- EMOBANK ----")
    # emobank = EmobankLoader()
    # data = emobank.load_data()
    # # emobank.validate_splits()
    # emobank.check_number_of_data()
    # # print(data)

    print("---- SEMEVAL 2018 E-c ----")
    semeval = SemEvalLoader()
    data = semeval.load_data()
    semeval.check_number_of_data()
    print(semeval.get_vad_coordinates_of_labels())
    # print(data['test']['label'])
    # print(data)

    # print("---- ISEAR ----")
    # isear = ISEARLoader()
    # data = isear.load_data()
    # isear.check_number_of_data()
    # print(isear.get_vad_coordinates_of_labels())
    # # print(data)

    # print("---- SSEC ----")
    # ssec = SSECLoader()
    # data = ssec.load_data()
    # ssec.check_number_of_data()
    # print(ssec.get_vad_coordinates_of_labels())
    # #print(data)

    print("---- IEMOCAPCAT ----")
    iemocapcat = IEMOCAPCatLoader()
    data = iemocapcat.load_data()
    iemocapcat.check_number_of_data()
    print(iemocapcat.get_vad_coordinates_of_labels())
    # # print(data)

    # print("---- IEMOCAPVAD ----")
    # iemocapvad = IEMOCAPVADLoader()
    # data = iemocapvad.load_data()
    # iemocapvad.check_number_of_data()
    # # print(data)

    print("---- GOEMOTIONS ----")
    goemotions = GOEMOTIONSLoader()
    data = goemotions.load_data()
    goemotions.check_number_of_data()
    # print(len(data['test']['label']))


if __name__ == "__main__":
    main()
