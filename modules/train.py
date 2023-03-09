import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report as sk_classificaiton_report

from seqeval.metrics import classification_report as seq_classificaiton_report
from seqeval.metrics import f1_score


def train_step(model, optimizer, text, slot_labels, intent_labels):
    with tf.GradientTape() as tape:
        slot_logits, intent_probs = model(text, training=True)
        slot_loss = model.crf.loss(slot_labels, slot_logits)
        intent_loss = tf.keras.losses.sparse_categorical_crossentropy(intent_labels, intent_probs)

        total_loss = tf.math.add(slot_loss, intent_loss)
        grads = tape.gradient(total_loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return total_loss

def predict(model, x):
    return model(x, training=False)
    
def validation(model, data, val_loss_metric, min_val_loss, save_path):
    val_slot_logits, val_intent_probs = predict(model, data.dev_dataset[0])
    val_intent = np.argmax(val_intent_probs, axis=-1)

    val_slot_loss = model.crf.loss(
        tf.convert_to_tensor(data.dev_dataset[1].reshape(data.dev_dataset[1].shape[0], 1, data.dev_dataset[1].shape[1]), dtype=tf.float32),
        tf.cast(tf.reshape(val_slot_logits, [data.dev_dataset[1].shape[0], 1, data.dev_dataset[1].shape[1]]), dtype=tf.float32)
    )
    val_intent_loss = tf.keras.losses.sparse_categorical_crossentropy(data.dev_dataset[2], val_intent_probs)
    val_total_loss = tf.math.add(val_slot_loss, val_intent_loss)
    val_loss_metric.update_state(val_total_loss)

    slot_f1 = f1_score(
        [[data.indx2slot[i] for i in j] for j in data.dev_dataset[1]],
        [[data.indx2slot[i] for i in j] for j in val_slot_logits.numpy()]
    )

    intent_acc = accuracy_score(data.dev_dataset[2], val_intent)

    print('*****Validation Results*****')
    print('slot-filling f1-score : {}'.format(np.round(slot_f1, 2)))
    print('intent detection accuracy : {}'.format(np.round(intent_acc, 2)))

    if val_loss_metric.result().numpy() < min_val_loss:
        print('validation loss has improved from {} to {}'.format(min_val_loss, val_loss_metric.result().numpy()))
        print('saving model to {} ...'.format(save_path))
        min_val_loss = val_loss_metric.result().numpy()
        model.save_weights(filepath=save_path, overwrite=True, save_format='h5')
    else:
        print('validation loss has not improved from {}'.format(min_val_loss))
    val_loss_metric.reset_states()

    return val_loss_metric, min_val_loss

def train(model, data:object, epochs:int, optimizer, save_path:str, validate:bool=True):
    train_loss_metric = tf.keras.metrics.Mean()
    val_loss_metric = tf.keras.metrics.Mean()
    min_val_loss = np.inf

    for epoch in range(epochs):
        print('Start of epoch {}:'.format(epoch))
        for step, (text, slot_labels, intent_labels) in enumerate(data.train_dataset):
            total_loss = train_step(model, optimizer, text, slot_labels, intent_labels)
            train_loss_metric.update_state(total_loss)

            if step % 50 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(train_loss_metric.result().numpy()))
                )
        print("Training loss over epoch: %.4f" % (float(train_loss_metric.result().numpy()),))
        train_loss_metric.reset_states()

        if validate:
            val_loss_metric, min_val_loss = validation(model, data, val_loss_metric, min_val_loss, save_path)
        print('=========================================================')

def evaluate(model, data):
    slot_logits, intent_probs = predict(model, data.test_dataset[0])
    val_intent = np.argmax(intent_probs, axis=-1)

    pd.DataFrame(
        sk_classificaiton_report(
            data.test_dataset[2], val_intent, output_dict=True
        )
    ).transpose().to_csv('./results/{}_joint_slot_report.csv'.format(data.name), index=True)

    pd.DataFrame(
        seq_classificaiton_report(
            [[data.indx2slot[i] for i in j] for j in data.test_dataset[1]],
            [[data.indx2slot[i] for i in j] for j in slot_logits.numpy()], output_dict=True
        )
    ).transpose().to_csv('./results/{}_joint_intent_report.csv'.format(data.name), index=True)