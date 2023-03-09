import numpy as np
import tensorflow as tf

from modules.crf import CRF

# tf.config.run_functions_eagerly(True)


class CoBiC(tf.keras.Model):
    '''
    # Implementation of CoBiC model
    '''
    def __init__(
        self, intent_size, slot_size, vocab_size, embedding_dim, max_seq_length, embedding_matrix=None,
        cnn_activation='relu', cnn_padding='same',
        cnn_filters=50, cnn_window_length=4, lstm_layers=2, lstm_units=200, name='CoBiC', **kwargs
    ):
        super(CoBiC, self).__init__(name=name, **kwargs)
        self.intent_size = intent_size
        self.slot_size = slot_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding_matrix = embedding_matrix
        self.max_seq_length = max_seq_length

        self.cnn_filters = cnn_filters
        self.cnn_window_length = cnn_window_length
        self.cnn_activation = cnn_activation
        self.cnn_padding = cnn_padding

        self.lstm_layers = lstm_layers
        self.lstm_units = lstm_units

        #definition of layers of the CoBiC model
        if embedding_matrix is None: # check if there is a pretrained embedding
            self.embedding = tf.keras.layers.Embedding(
                self.vocab_size, self.embedding_dim, weights=[np.random.uniform(-1, 1, (self.vocab_size, self.embedding_dim))], mask_zero=True, name="Embedding", trainable=True
            )
        else:
            self.embedding = tf.keras.layers.Embedding(
                self.vocab_size, self.embedding_dim, weights=[self.embedding_matrix], input_length=self.max_seq_length, trainable=False
            )
        
        self.cnn = tf.keras.layers.Conv1D(
            self.cnn_filters, self.cnn_window_length, padding=self.cnn_padding, activation=self.cnn_activation, name='CNN'
        )

        self.first_bilstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                units=self.lstm_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2
            ), name='1st_BiLSTM'
        )
        self.second_bilstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                units=self.lstm_units, return_sequences=True, return_state=True, dropout=0.2, recurrent_dropout=0.2
            ), name='2nd_BiLSTM'
        )
        
        self.concatenate = tf.keras.layers.Concatenate()
        self.classifier = tf.keras.layers.Dense(self.intent_size, activation='softmax', name='intent_classifier')
        self.time_distributed = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(
                slot_size, name='time_distributed'
            )
        )

        self.crf = CRF(self.slot_size, name='CRF')

    def call(self, inputs, training=None):
        seq_lengths = tf.math.reduce_sum(
            tf.cast(tf.math.not_equal(inputs, 0), dtype=tf.int32), axis=-1
        )

        if training == None:
            training = tf.keras.backend.learning_phase()
        
        embeddings = self.embedding(inputs)
        cnn = self.cnn(embeddings)
        first_bilstm = self.first_bilstm(cnn)
        second_bilstm, frw_h, _, bkw_h, _  = self.second_bilstm(first_bilstm)
        concatenate = self.concatenate([frw_h, bkw_h])
        classifier = self.classifier(concatenate)
        time_distributed = self.time_distributed(second_bilstm)
        crf = self.crf(time_distributed, seq_lengths, training)

        return crf, classifier
    
    def get_config(self):
        config = super().get_config().copy()

        config.update({
            'intent_size': self.intent_size,
            'slot_size': self.slot_size,
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'embedding_matrix': self.embedding_matrix,
            'max_seq_length': self.max_seq_length,
            'cnn_filters': self.cnn_filters,
            'cnn_window_length': self.cnn_window_length,
            'cnn_activation': self.cnn_activation,
            'cnn_padding': self.cnn_padding,
            'lstm_layers': self.lstm_layers,
            'lstm_units': self.lstm_units,
            'crf': self.crf,
            'name': self.name,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
