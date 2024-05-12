import re
import warnings

import numpy as np
import pandas as pd
from keras.src.layers import Bidirectional
from nltk import WordNetLemmatizer, word_tokenize
from nltk.corpus import stopwords
from sklearn import model_selection
from sklearn.metrics import roc_auc_score
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Embedding, Input, LSTM, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
import nlpaug.augmenter.word as naw
import nltk

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

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

# 初始化词形还原器和停用词列表
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def clean_text(text):
    # 去除标点和数字
    text = re.sub(r'[\d\W]+', ' ', text)
    # 分词
    words = word_tokenize(text)
    # 去除停用词并进行词形还原
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)


# 初始化同义词替换增强器
synonym_aug = naw.SynonymAug(aug_src='wordnet')


def augment_text(texts, augmenter, num_augmented=1):
    augmented_texts = []
    for text in texts:
        augmented = augmenter.augment(text, n=num_augmented)
        augmented_texts.extend(augmented if isinstance(augmented, list) else [augmented])
    return augmented_texts


def pad_text(texts, tokenizer):
    # Pads sequences to the same length
    return pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=MAX_SEQUENCE_LENGTH)


# Data loading and preprocessing
def load_and_preprocess_data(filepath):
    # Load the data
    data = pd.read_csv(filepath)
    print(f'Loaded {len(data)} records')

    # Ensure all comment_text values are strings
    data[TEXT_COLUMN] = data[TEXT_COLUMN].astype(str)

    # 文本清洗和数据增强
    data[TEXT_COLUMN] = data[TEXT_COLUMN].apply(clean_text)
    # 假设我们只对训练集进行数据增强
    augmented_texts = augment_text(data[TEXT_COLUMN], synonym_aug)
    augmented_data = pd.DataFrame({TEXT_COLUMN: augmented_texts, TOXICITY_COLUMN: 0})  # 增强的文本暂时标记为非有害
    data = pd.concat([data, augmented_data], ignore_index=True)

    # Convert target column and identity columns to boolean
    identity_columns = [
        'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
        'muslim', 'black', 'white', 'psychiatric_or_mental_illness']

    def convert_to_bool(df, col_name):
        df[col_name] = np.where(df[col_name] >= 0.5, True, False)

    for col in [TOXICITY_COLUMN] + identity_columns:
        convert_to_bool(data, col)

    return data


# Split the dataset
def split_data(data):
    # Split the data into training and validation sets
    train_df, validate_df = model_selection.train_test_split(data, test_size=0.2)
    print(f'{len(train_df)} train comments, {len(validate_df)} validate comments')
    return train_df, validate_df


# Load embedding layer
def load_embedding_matrix(word_index, embedding_path=EMBEDDINGS_PATH):
    # Load pre-trained word vectors
    embeddings_index = {}
    with open(embedding_path, 'r', encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    # Prepare the embedding matrix
    num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, EMBEDDINGS_DIMENSION))
    for word, i in word_index.items():
        if i >= MAX_NUM_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in the embedding index will be all zeros
            embedding_matrix[i] = embedding_vector

    # Load pre-trained word vectors as an embedding layer, set trainable=False to keep the embeddings fixed
    embedding_layer = Embedding(
        num_words,
        EMBEDDINGS_DIMENSION,
        embeddings_initializer=Constant(embedding_matrix),
        trainable=True
    )
    return embedding_layer


# Define model structure
def create_model(word_index):
    embedding_layer = load_embedding_matrix(word_index)
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    # Replace the original LSTM layer with a Bidirectional LSTM layer
    x = Bidirectional(LSTM(64))(embedded_sequences)  # Adjust the number of LSTM units as needed
    x = Dropout(DROPOUT_RATE)(x)
    preds = Dense(2, activation='softmax')(x)

    model = Model(sequence_input, preds)
    return model


# Compile and train the model
def compile_and_train_model(model, train_text, train_labels, validate_text, validate_labels):
    # Compile the model with binary crossentropy loss and RMSprop optimizer
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(learning_rate=LEARNING_RATE),
                  metrics=['acc'])

    # Train the model
    history = model.fit(train_text, train_labels,
                        batch_size=BATCH_SIZE,
                        epochs=NUM_EPOCHS,
                        validation_data=(validate_text, validate_labels),
                        verbose=1)
    return history


# Create and train the model
def create_and_train_model(train_df, validate_df, tokenizer):
    # Prepare the data
    train_text = pad_text(train_df[TEXT_COLUMN], tokenizer)
    train_labels = to_categorical(train_df[TOXICITY_COLUMN])
    validate_text = pad_text(validate_df[TEXT_COLUMN], tokenizer)
    validate_labels = to_categorical(validate_df[TOXICITY_COLUMN])

    # Create the model
    model = create_model(tokenizer.word_index)

    # Compile and train the model
    history = compile_and_train_model(model, train_text, train_labels, validate_text, validate_labels)

    return model, history


def calculate_bias_metrics_for_model(dataset, model, tokenizer, identity_columns, TOXICITY_COLUMN):
    """Calculate and return a dictionary of bias metrics for the model"""
    bias_metrics = {}

    for col in identity_columns:
        # Create a mask for the subgroup
        subgroup_mask = dataset[col]
        subgroup_examples = dataset[subgroup_mask]
        subgroup_text = pad_text(subgroup_examples[TEXT_COLUMN], tokenizer)
        subgroup_labels = subgroup_examples[TOXICITY_COLUMN].values
        subgroup_scores = model.predict(subgroup_text)[:, 1]
        bias_metrics[col] = roc_auc_score(subgroup_labels, subgroup_scores)

    return bias_metrics


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


# Main function
def main():
    # Load and preprocess data
    train_data_filepath = 'Data/train.csv'
    data = load_and_preprocess_data(train_data_filepath)

    # Split the dataset
    train_df, validate_df = split_data(data)

    # Create a tokenizer for text
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(train_df[TEXT_COLUMN])

    # Create and train the model
    model, history = create_and_train_model(train_df, validate_df, tokenizer)

    # Evaluate the model
    validate_text = pad_sequences(tokenizer.texts_to_sequences(validate_df[TEXT_COLUMN]), maxlen=MAX_SEQUENCE_LENGTH)
    validate_labels = validate_df[TOXICITY_COLUMN].values
    predictions = model.predict(validate_text)[:, 1]
    overall_auc = roc_auc_score(validate_labels, predictions)
    print(f'Validation ROC AUC: {overall_auc}')

    # Calculate bias metrics
    identity_columns = [
        'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
        'muslim', 'black', 'white', 'psychiatric_or_mental_illness'
    ]
    bias_metrics = calculate_bias_metrics_for_model(validate_df, model, tokenizer, identity_columns, TOXICITY_COLUMN)

    # Calculate the final score
    final_score = calculate_final_score(overall_auc, bias_metrics)
    print(f'Final Score: {final_score}')

    # Predict and submit
    test = pd.read_csv('Data/test.csv')
    submission = pd.read_csv('Data/sample_submission.csv', index_col='id')
    submission['prediction'] = model.predict(pad_text(test[TEXT_COLUMN], tokenizer))[:, 1]
    submission.to_csv('final_submission.csv')


if __name__ == '__main__':
    main()
