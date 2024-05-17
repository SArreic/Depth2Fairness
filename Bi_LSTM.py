import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import model_selection
from sklearn.metrics import roc_auc_score
from tensorflow.keras.layers import Input, Dense, Dropout, Bidirectional, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
from transformers import TFAutoModel, AutoTokenizer

# Suppress deprecated warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Constants
MAX_NUM_WORDS = 10000
MAX_SEQUENCE_LENGTH = 250
EMBEDDINGS_PATH = 'Embedding_file/glove.6B.100d.txt'
EMBEDDINGS_DIMENSION = 100
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.00005
NUM_EPOCHS = 1
BATCH_SIZE = 128
TOXICITY_COLUMN = 'target'
TEXT_COLUMN = 'comment_text'
BERT_MODEL_NAME = 'bert-base-multilingual-cased'

identity_columns = [
    'male', 'female', 'bisexual', 'homosexual_gay_or_lesbian', 'heterosexual',
    'transgender', 'other_gender', 'other_sexual_orientation', 'atheist',
    'buddhist', 'christian', 'jewish', 'muslim', 'other_religion', 'black',
    'white', 'asian', 'latino', 'hindu', 'other_race_or_ethnicity',
    'physical_disability', 'psychiatric_or_mental_illness',
    'intellectual_or_learning_disability', 'other_disability']

# Load BERT tokenizer
bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)


def pad_text(texts, tokenizer):
    # Pads sequences to the same length
    return pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=MAX_SEQUENCE_LENGTH)


def split_data(data):
    # Split the data into training and validation sets
    train_df, validate_df = model_selection.train_test_split(data, test_size=0.2)
    print(f'{len(train_df)} train comments, {len(validate_df)} validate comments')
    return train_df, validate_df


def power_mean(series, p):
    """Calculate the power mean of a series"""
    total = sum(np.power(series, p))
    return np.power(total / len(series), 1 / p)


def calculate_final_score(overall_auc, bias_auc_dict, POWER=-5, OVERALL_MODEL_WEIGHT=0.25):
    """Calculate the final score based on overall AUC and bias AUCs"""
    bias_auc_list = list(bias_auc_dict.values())
    bias_power_mean = power_mean(bias_auc_list, POWER)
    final_score = (OVERALL_MODEL_WEIGHT * overall_auc) + ((1 - OVERALL_MODEL_WEIGHT) * bias_power_mean)
    return final_score


# Define model structure
def create_model(word_index):
    # Load BERT model
    bert_model = TFAutoModel.from_pretrained(BERT_MODEL_NAME)

    # BERT input
    input_ids = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name="input_ids")
    attention_mask = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name="attention_mask")

    # BERT embedding
    bert_output = bert_model(input_ids, attention_mask=attention_mask)
    sequence_output = bert_output.last_hidden_state

    # Bi-LSTM layer
    x = Bidirectional(LSTM(64))(sequence_output)
    x = Dropout(DROPOUT_RATE)(x)

    # Prediction layer
    preds = Dense(2, activation='softmax')(x)

    # Compile model
    model = Model(inputs=[input_ids, attention_mask], outputs=preds)
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(learning_rate=LEARNING_RATE),
                  metrics=['acc'])

    return model


# Create and train the model
def create_and_train_model(train_df, validate_df, tokenizer):
    # Prepare the data
    train_text = tokenizer(train_df[TEXT_COLUMN].tolist(), max_length=MAX_SEQUENCE_LENGTH, padding='max_length',
                           truncation=True, return_tensors="tf")
    train_labels = to_categorical(train_df[TOXICITY_COLUMN])

    validate_text = tokenizer(validate_df[TEXT_COLUMN].tolist(), max_length=MAX_SEQUENCE_LENGTH, padding='max_length',
                              truncation=True, return_tensors="tf")
    validate_labels = to_categorical(validate_df[TOXICITY_COLUMN])

    # Create the model
    model = create_model(tokenizer.vocab_size)

    # Train the model
    history = model.fit(
        {'input_ids': train_text['input_ids'], 'attention_mask': train_text['attention_mask']},
        train_labels,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        validation_data=(
            {'input_ids': validate_text['input_ids'], 'attention_mask': validate_text['attention_mask']},
            validate_labels
        ),
        verbose=1
    )

    return model, history


def calculate_bias_metrics_for_model(dataset, model, tokenizer, identity_columns, TOXICITY_COLUMN):
    """Calculate and return a dictionary of bias metrics for the model"""
    bias_metrics = {}

    for col in identity_columns:
        # Create a mask for the subgroup
        subgroup_mask = dataset[col].astype(bool)
        subgroup_examples = dataset[subgroup_mask]
        subgroup_text = subgroup_examples[TEXT_COLUMN].tolist()
        # Prepare the data using BERT Tokenizer
        inputs = tokenizer(subgroup_text, max_length=MAX_SEQUENCE_LENGTH,
                           padding='max_length', truncation=True, return_tensors="tf")
        subgroup_labels = subgroup_examples[TOXICITY_COLUMN].values
        # Get model predictions
        predictions = model.predict(inputs)[0][:, 1]
        # Calculate ROC AUC
        bias_metrics[col] = roc_auc_score(subgroup_labels, predictions)

    return bias_metrics


# Main function
def main():
    # Load the preprocessed data
    preprocessed_data_filepath = 'Data/preprocessed_train.csv'
    data = pd.read_csv(preprocessed_data_filepath, dtype={TEXT_COLUMN: str}, na_filter=False)

    # Split the dataset
    train_df, validate_df = split_data(data)

    # Clean the text data to ensure it is all strings
    train_df[TEXT_COLUMN] = train_df[TEXT_COLUMN].astype(str)
    validate_df[TEXT_COLUMN] = validate_df[TEXT_COLUMN].astype(str)

    # Create a tokenizer for text
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(tqdm(train_df[TEXT_COLUMN], desc="Tokenizing text"))

    # Create and train the model
    model, history = create_and_train_model(train_df, validate_df, bert_tokenizer)

    # Evaluate the model
    validate_text = bert_tokenizer(validate_df[TEXT_COLUMN].tolist(), max_length=MAX_SEQUENCE_LENGTH,
                                   padding='max_length', truncation=True, return_tensors="tf")
    validate_labels = validate_df[TOXICITY_COLUMN].values
    predictions = model.predict(
        {'input_ids': validate_text['input_ids'], 'attention_mask': validate_text['attention_mask']})[:, 1]
    overall_auc = roc_auc_score(validate_labels, predictions)
    print(f'Validation ROC AUC: {overall_auc}')

    bias_metrics = calculate_bias_metrics_for_model(validate_df, model, tokenizer, identity_columns, TOXICITY_COLUMN)

    # Calculate the final score
    final_score = calculate_final_score(overall_auc, bias_metrics)
    print(f'Final Score: {final_score}')

    # Predict and submit
    test = pd.read_csv('Data/test.csv')
    submission = pd.read_csv('Data/sample_submission.csv', index_col='id')
    test_text = bert_tokenizer(test[TEXT_COLUMN].tolist(), max_length=MAX_SEQUENCE_LENGTH, padding='max_length',
                               truncation=True, return_tensors="tf")
    submission['prediction'] = model.predict(
        {'input_ids': test_text['input_ids'], 'attention_mask': test_text['attention_mask']})[:, 1]
    submission.to_csv('final_submission.csv')

    # save the model
    model.save('Models/Bi_LSTM.h5')

    # load the model
    # reconstructed_model = tf.keras.models.load_model('path_to_my_model.h5')


if __name__ == '__main__':
    main()
