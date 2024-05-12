import re
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk import WordNetLemmatizer, word_tokenize
from nltk.corpus import stopwords
import nlpaug.augmenter.word as naw
import nltk
from tqdm import tqdm

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Constants
MAX_NUM_WORDS = 10000
MAX_SEQUENCE_LENGTH = 250
TOXICITY_COLUMN = 'target'
TEXT_COLUMN = 'comment_text'

# Initialize lemmatizer and stop words list
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def clean_text(text):
    """Remove punctuation and numbers, tokenize, remove stop words, and lemmatize"""
    text = re.sub(r'[\d\W]+', ' ', text)
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)


# Initialize synonym augmenter
synonym_aug = naw.SynonymAug(aug_src='wordnet')


def augment_text(texts, augmenter, num_augmented=1):
    """Augment texts by replacing words with their synonyms"""
    augmented_texts = []
    for text in tqdm(texts, desc="Augmenting text"):
        augmented = augmenter.augment(text, n=num_augmented)
        augmented_texts.extend(augmented if isinstance(augmented, list) else [augmented])
    return augmented_texts


def pad_text(texts, tokenizer):
    """Pad sequences to the same length"""
    return pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=MAX_SEQUENCE_LENGTH)


# Data loading and preprocessing
def load_and_preprocess_data(filepath):
    """Load data, clean text, augment text, and convert columns to boolean"""
    data = pd.read_csv(filepath)
    print(f'Loaded {len(data)} records')

    data[TEXT_COLUMN] = data[TEXT_COLUMN].astype(str)
    tqdm.pandas(desc="Cleaning text")
    data[TEXT_COLUMN] = data[TEXT_COLUMN].progress_apply(clean_text)

    augmented_texts = augment_text(data[TEXT_COLUMN], synonym_aug)
    augmented_data = pd.DataFrame({TEXT_COLUMN: augmented_texts, TOXICITY_COLUMN: 0})
    data = pd.concat([data, augmented_data], ignore_index=True)

    identity_columns = [
        'male', 'female', 'bisexual', 'homosexual_gay_or_lesbian', 'heterosexual',
        'transgender', 'other_gender', 'other_sexual_orientation', 'atheist',
        'buddhist', 'christian', 'jewish', 'muslim', 'other_religion', 'black',
        'white', 'asian', 'latino', 'hindu', 'other_race_or_ethnicity',
        'physical_disability', 'psychiatric_or_mental_illness',
        'intellectual_or_learning_disability', 'other_disability']

    def convert_to_bool(df, col_name):
        df[col_name] = np.where(df[col_name] >= 0.5, True, False)

    for col in [TOXICITY_COLUMN] + identity_columns:
        convert_to_bool(data, col)

    return data


# Main function
def main():
    """Main function to load and preprocess data, then save it to a new CSV file"""
    train_data_filepath = 'Data/train.csv'
    data = load_and_preprocess_data(train_data_filepath)

    preprocessed_data_filepath = 'Data/preprocessed_train.csv'
    data.to_csv(preprocessed_data_filepath, index=False)
    print(f'Preprocessed data saved to {preprocessed_data_filepath}')


if __name__ == '__main__':
    main()
