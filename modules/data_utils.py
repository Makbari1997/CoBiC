import os
import numpy as np
from gensim.models import KeyedVectors
from tensorflow import data


def load_data(path:str, encoding:str) -> dict:
    data = {
        'intent': [], 'slot': [], 'text': []
    }
    with open(os.path.join(path, 'label'), 'r', encoding=encoding) as f:
        lines = f.readlines()
        for line in lines:
            data['intent'].append(line.split()[0])
    with open(os.path.join(path, 'seq.in'), 'r', encoding=encoding) as f:
        lines = f.readlines()
        for line in lines:
            data['text'].append(line.split())
    with open(os.path.join(path, 'seq.out'), 'r', encoding=encoding) as f:
        lines = f.readlines()
        for line in lines:
            data['slot'].append(line.split())
        
    return data

def map_index(data:list) -> dict:
    return {i:j for j, i in enumerate(data)}

def get_embeddings(path:str, vocab2indx:dict, dim:int, vocab_size:int) -> np.ndarray:
    if path is None:
        return None
    print('This may take several minutes...')
    model = KeyedVectors.load_word2vec_format(fname=path, binary=False)
    embedding_matrix = np.zeros((vocab_size, dim))
    missed = 0

    for word, index in vocab2indx.items():
        try:
            vector = model.get_vector(word)
            embedding_matrix[index] = vector
        except:
            missed += 1

    print('{} word(s) missed!'.format(missed))

    return embedding_matrix


def padding(seq:list, max_seq_length:int, pad_token:str) -> list:
    if len(seq) == max_seq_length:
        return seq
    elif len(seq) > max_seq_length:
        return seq[:max_seq_length]
    else:
        return seq + [pad_token] * (max_seq_length - len(seq))

def indx_replacer(seq:list, mapping:dict) -> list:
    tmp = []
    for s in seq:
        try:
            tmp.append(mapping[s])
        except:
            tmp.append(mapping['UNK'])
    return tmp

def encoder(data:list, indx:dict, max_seq_length:int=None, pad_token:str='PAD'):
    list_of_lists = True if isinstance(data[0], list) == True else False

    if list_of_lists == True:
        
        padded = list(map(padding, data, [max_seq_length] * len(data), [pad_token] * len(data)))
        encoded = list(map(lambda x : indx_replacer(x, indx), padded))
        return np.array(encoded)

    else:
        return np.array(indx_replacer(data, indx))
        
def to_tf_format(dataset:tuple, buffer_size:int, batch_size:int=32):
  '''
  converts given data to batched tensorflow dataset\n
  Parameters:\n
  -------------
  dataset: tuple of iterable objects like list or numpy array\n
  buffer_size: int\n
  batch_size: int\n
  -------------
  Return: tf.Data.Dataset 
  '''
  tf_dataset = data.Dataset.from_tensor_slices(dataset)
  tf_dataset = tf_dataset.shuffle(buffer_size=buffer_size).batch(batch_size)
  return tf_dataset