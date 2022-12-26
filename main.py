import tensorflow as tf
import numpy as np
import preprocessing
from preprocessing import pipeline, get_num_words
import os
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow_hub as hub
import matplotlib.pyplot as plt


def create_model_checkpoint(model_name, save_path="model_experiments"):
    return ModelCheckpoint(filepath=os.path.join(save_path, model_name),
                                              monitor='val_loss',
                                              mode='min',
                                              save_best_only=True)  # save only the best model to file


callbacks = [
    EarlyStopping(patience=200, monitor='val_loss', restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', min_lr=1e-10, patience=20, mode='min', factor=0.1),
    create_model_checkpoint(model_name='model')
    ]

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(get_num_words(), 10),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True, recurrent_dropout=0.2)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False, recurrent_dropout=0.2)),
        tf.keras.layers.Dense(32,activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def compile(model):
    model.compile(loss='binary_crossentropy',
                  metrics=['accuracy'],
                  optimizer=tf.keras.optimizers.Adam(1e-2))
def train(model,train_sequences, train_label):
    history = model.fit(train_sequences,
                        train_label,
                        epochs=100,
                        validation_split=0.1,
                        batch_size=32,
                        callbacks=[callbacks])
    return history

train_sequences, train_label, test_sequences, test_label = pipeline()
model = build_model()
compile(model)
history = train(model,train_sequences, train_label)
tf.keras.models.load_model(
    r'C:\Users\mturri\Repo\NLP_food_customer_reviews\model_experiments\model',
    custom_objects=None, compile=True, options=None
)
print(model.evaluate(test_sequences, test_label))
print(model.evaluate(train_sequences, train_label))



# Transfer Learning

hub_layer = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1", output_shape=[20],
                           input_shape=[], dtype=tf.string, trainable=False)

model = tf.keras.Sequential([
    hub_layer,
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
train_text, train_label, test_text, test_label = preprocessing.getting_sentences()
history = model.fit(np.array(train_text), np.array(train_label),
                      epochs=1000,
                      batch_size=32,
                      callbacks=callbacks,
                      validation_split=0.1)
loss, accuracy = model.evaluate(test_text, test_label)
print("Loss:", loss)
print("Accuracy:", accuracy)

loss = history.history['loss']
val_loss = history.history['val_loss']
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot the loss and accuracy on the left subplot
ax1.plot(loss, label='Loss')
ax1.plot(val_loss, label='Validation Loss')
ax1.set_title('Loss and Validation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()

# Plot the accuracy and validation accuracy on the right subplot
ax2.plot(accuracy, label='Accuracy')
ax2.plot(val_accuracy, label='Validation Accuracy')
ax2.set_title('Accuracy and Validation Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()

plt.show()


