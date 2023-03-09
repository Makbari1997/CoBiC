import json
from modules.train import train, evaluate
from modules.cobic import CoBiC
from modules.dataset import Dataset
from modules.data_utils import get_embeddings
from tensorflow import saved_model
from tensorflow.keras import optimizers, models
import tensorflow as tf

# tf.config.run_functions_eagerly(True)


if __name__ == '__main__':
    config_file = open('./config.json')
    config = json.load(config_file)

    print('=========================================================')
    print('Loading dataset...')
    data = Dataset(
        name=config['dataset'],
        max_seq_length=config['max_seq_length'],
        pad_token=config['pad_token'],
        batch_size=config['batch_size']
    )
    print(data.test_dataset)
    print('Loading done...')
    print('=========================================================')

    print('loading embeddings...')
    embedding_matrix = get_embeddings(
        path=config['embedding_path'],
        vocab2indx=data.voc2indx,
        dim=config['embedding_dim'],
        vocab_size=data.vocab_size
    )
    print('Loading done...')
    print('=========================================================')

    model = CoBiC(
        intent_size=data.intent_size,
        slot_size=data.slot_size,
        vocab_size=data.vocab_size,
        embedding_dim=config['embedding_dim'],
        max_seq_length=data.max_seq_length,
        embedding_matrix=embedding_matrix,
        cnn_activation=config['cnn_activation'],
        cnn_padding=config['cnn_padding'],
        cnn_filters=config['cnn_filters'],
        cnn_window_length=config['cnn_window_length'],
        lstm_layers=config['lstm_layers'],
        lstm_units=config['lstm_units']
    )


    print('Start of training...')
    train(
        model=model,
        data=data,
        epochs=config['epochs'],
        optimizer=optimizers.Adam(learning_rate=config['learning_rate']),
        save_path=config['model_path'],
        validate=True
    )

    print('=========================================================')

    model.load_weights(config['model_path'])

    if config['evaluate'] == True:
        print('Start of evaluation...')
        evaluate(
            model=model,
            data=data
        )
        print('=========================================================')