from modules.data_utils import *


class Dataset:
    def __init__(self, name:str, max_seq_length, pad_token:str='PAD', batch_size:int=32, encoding:str='utf-8') -> None:
        self.name = name
        self.train_path = os.path.join('./data/',name, 'train')
        self.test_path = os.path.join('./data/',name, 'test')
        self.dev_path = os.path.join('./data/',name, 'dev')
        
        self.train = load_data(self.train_path, encoding)
        self.test = load_data(self.test_path, encoding)
        self.dev = load_data(self.dev_path, encoding)

        self.voc2indx = map_index(['UNK'] + list(set(sum(self.train['text'], []))))
        self.slot2indx = map_index(['PAD'] + list(set(sum(self.train['slot'] + self.test['slot'] + self.dev['slot'], []))))
        self.intent2indx = map_index(list(set(self.train['intent'] + self.test['intent'] + self.dev['intent'])))

        self.indx2vocab = {j: i for i, j in self.voc2indx.items()}
        self.indx2slot = {j: i for i, j in self.slot2indx.items()}
        self.indx2intent = {j: i for i, j in self.intent2indx.items()}

        self.intent_size = len(self.intent2indx.keys())
        self.slot_size = len(self.slot2indx.keys())
        self.vocab_size = len(self.voc2indx.keys())

        if type(max_seq_length) is str:
            if max_seq_length == 'max':
                self.max_seq_length = max(list(map(len, self.train['text'])))
            elif max_seq_length == 'mean':
                self.max_seq_length = int(sum(list(map(len, self.train['text']))) / len(self.train['text'])) + 1
        elif type(max_seq_length) is int:
            self.max_seq_length = max_seq_length
        else:
            raise Exception('The value for max_seq_length is not supported. Please make sure if you set an int value to it or use max or mean as str.')

        self.train_dataset = (
            encoder(self.train['text'], self.voc2indx, self.max_seq_length, pad_token),
            encoder(self.train['slot'], self.slot2indx, self.max_seq_length, pad_token),
            encoder(self.train['intent'], self.intent2indx)
        )
        self.train_dataset = to_tf_format(self.train_dataset, len(self.train['text']), batch_size)

        self.test_dataset = (
            encoder(self.test['text'], self.voc2indx, self.max_seq_length, pad_token),
            encoder(self.test['slot'], self.slot2indx, self.max_seq_length, pad_token),
            encoder(self.test['intent'], self.intent2indx)
        )

        self.dev_dataset = (
            encoder(self.dev['text'], self.voc2indx, self.max_seq_length, pad_token),
            encoder(self.dev['slot'], self.slot2indx, self.max_seq_length, pad_token),
            encoder(self.dev['intent'], self.intent2indx)
        )