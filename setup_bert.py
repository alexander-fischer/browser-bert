import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from transformers import TFBertModel
from transformers import BertTokenizerFast
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def create_model(max_len, classifier_layer=True):
    # Load tiny BERT model
    encoder = TFBertModel.from_pretrained(
        "google/bert_uncased_L-2_H-128_A-2", from_pt=True)

    # Setup input layer
    input_ids = layers.Input(
        shape=(max_len,), dtype=tf.int32, name="input_ids")
    token_type_ids = layers.Input(
        shape=(max_len,), dtype=tf.int32, name="token_type_ids")
    attention_mask = layers.Input(
        shape=(max_len,), dtype=tf.int32, name="attention_mask")
    bert = encoder(
        input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
    )[0]

    # Make sure BERT weights stay the same during training
    bert.trainable = False

    # For python training we add a classification layer
    if classifier_layer:
        bert = layers.Dense(1, activation="sigmoid")(bert)

    # For TFJS we just add a layer to flatten the output
    else:
        bert = layers.Flatten()(bert)

    # Put model together
    model = keras.Model(
        inputs=[input_ids, token_type_ids, attention_mask],
        outputs=[bert],
    )
    loss = keras.losses.BinaryCrossentropy(from_logits=False)
    optimizer = keras.optimizers.Adam(lr=0.0001)
    model.compile(optimizer=optimizer, loss=[loss], metrics=["accuracy"])

    return model


# Model takes 128 tokens as input
MAX_LEN = 128

# Save model for TFJS
model_to_save = create_model(MAX_LEN, False)
model_to_save.save("./model")

# Setup model and tokenizer for python training
model = create_model(MAX_LEN)
vocab_file = "./vocab.txt"
tokenizer = BertTokenizerFast(vocab_file)


def tokenize(text):
    # Use encoding functionality from transformers lib
    example = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        return_attention_mask=True,
        padding="max_length",
        truncation=True,
    )

    input_ids = np.array(example["input_ids"], dtype=np.int32)
    token_type_ids = np.array(example["token_type_ids"], dtype=np.int32)
    attention_masks = np.array(example["attention_mask"], dtype=np.int32)
    return input_ids, token_type_ids, attention_masks


def prepare_spam_dataset(df):
    # Get features and labels from spam data
    features = df["Message"].values
    labels = df["Type"].values

    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    label_list = []
    for i in range(len(features)):
        feature = features[i]

        # Encode example text
        input_ids, token_type_ids, attention_masks = tokenize(feature)
        input_ids_list.append(input_ids)
        token_type_ids_list.append(token_type_ids)
        attention_mask_list.append(attention_masks)

        # Set label to 1 if example is spam and to 0 if it's ham
        label = 1 if labels[i] == "spam" else 0
        label_list.append(label)

    return np.array(input_ids_list), np.array(token_type_ids_list), np.array(attention_mask_list), np.array(label_list).reshape(-1, 1)


# Load and split dataset
spam_file = "./spam.csv"
spam_df = pd.read_csv(spam_file, sep="\t")
train_df, test_df = train_test_split(spam_df)

# Setup training data
train_input_ids, train_token_type_ids, train_attention_mask, y_train = prepare_spam_dataset(
    train_df)

# Train model
bert_history = model.fit({
    "input_ids": train_input_ids,
    "token_type_ids": train_token_type_ids,
    "attention_mask": train_attention_mask},
    y_train,
    epochs=10
)

# Setup test data
test_input_ids, test_token_type_ids, test_attention_mask, y_test = prepare_spam_dataset(
    test_df)

# Evaluate model with test data
model.evaluate({
    "input_ids": test_input_ids,
    "token_type_ids": test_token_type_ids,
    "attention_mask": test_attention_mask},
    y_test
)
