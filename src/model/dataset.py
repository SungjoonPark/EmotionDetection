from torch.utils.data import Dataset, DataLoader
from data.loader import EmobankLoader, SemEvalLoader, ISEARLoader, SSECLoader



class EmotionDataset(Trainer):

    def __init__(self, args):
        self.args = args
        self.raw_data = self.load_data()
    

    def _load_data(self):
        if self.args['dataset'] == 'emobank':
            self.loader = EmobankLoader()
        elif self.args['dataset'] == 'semeval':
            self.loader = SemEvalLoader()
        elif self.args['dataset'] == 'isear':
            self.loader = ISEARLoader()
        elif self.args['dataset'] == 'ssec':
            self.loader = SSECLoader()
        return self.loader.load_data()
    