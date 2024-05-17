import re
import numpy as np
import pandas as pd
import spacy
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

nlp = spacy.load('en_core_web_sm')

misspell_dict = {"aren't": "are not", "can't": "cannot", "couldn't": "could not",
                 "didn't": "did not", "doesn't": "does not", "don't": "do not",
                 "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                 "he'd": "he would", "he'll": "he will", "he's": "he is",
                 "i'd": "I had", "i'll": "I will", "i'm": "I am", "isn't": "is not",
                 "it's": "it is", "it'll": "it will", "i've": "I have", "let's": "let us",
                 "mightn't": "might not", "mustn't": "must not", "shan't": "shall not",
                 "she'd": "she would", "she'll": "she will", "she's": "she is",
                 "shouldn't": "should not", "that's": "that is", "there's": "there is",
                 "they'd": "they would", "they'll": "they will", "they're": "they are",
                 "they've": "they have", "we'd": "we would", "we're": "we are",
                 "weren't": "were not", "we've": "we have", "what'll": "what will",
                 "what're": "what are", "what's": "what is", "what've": "what have",
                 "where's": "where is", "who'd": "who would", "who'll": "who will",
                 "who're": "who are", "who's": "who is", "who've": "who have",
                 "won't": "will not", "wouldn't": "would not", "you'd": "you would",
                 "you'll": "you will", "you're": "you are", "you've": "you have",
                 "'re": " are", "wasn't": "was not", "we'll": " will", "tryin'": "trying"}


def replace_typical_misspell(text):
    misspell_re = re.compile('(%s)' % '|'.join(misspell_dict.keys()))

    def replace(match):
        return misspell_dict[match.group(0)]

    return misspell_re.sub(replace, text)


def clean_text(text):
    """Clean text by removing punctuation, numbers, and misspellings, and then lemmatizing using spaCy."""
    text = text.lower()  # Convert to lowercase
    text = replace_typical_misspell(text)  # Correct misspellings
    text = re.sub(r'\d+', ' ', text)  # Remove numbers (keep words with apostrophes)
    doc = nlp(text)  # Process text with spaCy
    lemmatized = [token.lemma_.lower().strip() for token in doc if token.lemma_ != "-PRON-" and not token.is_stop and
                  not token.is_punct and not token.like_num]
    return ' '.join(lemmatized)


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
    """Load data, clean text, and convert columns to boolean"""
    data = pd.read_csv(filepath)
    print(f'Loaded {len(data)} records')

    # Clean and preprocess text
    texts = data[TEXT_COLUMN].astype(str).tolist()
    cleaned_texts = [clean_text(text) for text in tqdm(texts, total=len(texts), desc="Cleaning text")]
    data[TEXT_COLUMN] = cleaned_texts

    # Handle missing values
    data[TEXT_COLUMN].replace('', np.nan, inplace=True)
    data[TEXT_COLUMN].fillna('_#_', inplace=True)

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
