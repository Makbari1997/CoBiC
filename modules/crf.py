import tensorflow as tf
from tensorflow import int32
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
from tensorflow import Variable, random, math
from tensorflow import convert_to_tensor, cast, argmax, not_equal

import tensorflow_addons as tfa


class CRF(Layer):
    '''
    # Custom CRF layer
    computes log likelihood during training
    and uses Viterbi decoding for prediction

    ## Params:
    -----------
    label_size: int, number of labels
    mask_id: int, used to recover original sequence lengths, default is 0
    trans_params: np.array, a 2d array with size of label_size*label_size
    '''
    def __init__(self, label_size, mask_id=0, trans_params=None, name='crf', **kwargs):
        super(CRF, self).__init__(name=name, **kwargs)
        self.label_size = label_size
        self.mask_id = mask_id

        if trans_params == None: # there is no pre-computed transition matrix, so we randomly initiate it
            self.trans_params = Variable(
                random.uniform(
                    shape=(label_size, label_size)
                ),
                trainable=False
            )
        else:
            self.trans_params = trans_params
    
    def pad_viterbi(self, viterbi, max_seq_len):
        if len(viterbi) < max_seq_len:
            viterbi = viterbi + [self.mask_id] * (max_seq_len - len(viterbi))
        return viterbi

    def call(self, inputs, seq_lengths, training=None):

        if training == None: # check the phase of running the model using keras backend
            training = K.learning_phase()
        
        if training: # CRF only returns inputs during training phase
            return inputs
        
        # implementing viterbi decoding to predict slot tags during inference
        _, max_seq_len, _ = inputs.shape
        paths = []
        for logit, seq_len in zip(inputs, seq_lengths):
            viterbi_path, _ = tfa.text.viterbi_decode(
                logit[:seq_len],
                self.trans_params
            )
            paths.append(self.pad_viterbi(
                viterbi_path, max_seq_len
            ))
        
        return convert_to_tensor(paths)

  
    def get_proper_labels(self, y_true):
        shape = y_true.shape
        if len(shape) > 2:
            return argmax(y_true, -1, output_type=int32)
        return y_true

    def get_seq_lengths(self, matrix):
        # matrix is of shape (batch_size, max_seq_len)
        mask = not_equal(matrix, self.mask_id)
        seq_lengths = math.reduce_sum(
            cast(mask, dtype=int32),
            axis=-1
        )
        return seq_lengths

    def loss(self, y_true, y_pred):
        y_pred = convert_to_tensor(y_pred)
        y_true = cast(
            self.get_proper_labels(y_true), y_pred.dtype
        )

        seq_lengths = self.get_seq_lengths(y_true)
        log_likelihood, trans_params = tfa.text.crf_log_likelihood(
            y_pred, y_true, seq_lengths, self.trans_params
        )

        self.trans_params = Variable(trans_params, trainable=False)

        loss = - log_likelihood
        return loss
    
    def get_config(self):
        config = super().get_config().copy()

        config.update({
            'label_size': self.label_size,
            'mask_id': self.mask_id,
            'trans_params': self.trans_params,
            'name': self.name,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)