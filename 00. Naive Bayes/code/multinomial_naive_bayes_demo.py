
import pandas as pd
import numpy as np
from multinomial_naive_bayes import MultinomialNaiveBayesClassifier, confusion_matrix

# Read the data from a gz CSV file
reviews_df = pd.read_csv('00. Naive Bayes/data/IMDB_Dataset.csv.gz', compression='gzip')
reviews_df.head()

# Preprocess the dataframe (rename columns, convert labels to 0 and 1, etc.)
reviews_df.rename(columns={'review':'text', 'sentiment':'label'}, inplace = True)
reviews_df['label'] = reviews_df['label'].map({'positive':1, 'negative':0}).astype('uint8') # Convert the labels to 0 and 1
reviews_df.head()

# Clean the text from tags, punctuation, etc.
def clean_text(text: str) -> str:
    text = text.lower()                             # convert to lowercase
    text = text.replace('<br />', ' ')              # remove html tags
    # Remove meaningless suffixes
    text = text.replace('\'s', '')
    text = text.replace('\'m', '')
    text = text.replace('\'re', '')
    text = text.replace('\'ve', '')
    text = text.replace('\'d', '')
    text = text.replace('\'ll', '')
    text = text.replace('\'t', '')
    # Remove punctuation
    text = text.replace('\\', ' ')
    text = text.replace('/', ' ')
    text = text.replace('\'', ' ')
    text = text.replace('"', ' ')
    text = text.replace('(', ' ')
    text = text.replace(')', ' ')
    text = text.replace(',', ' ')
    text = text.replace('.', ' ')
    text = text.replace('!', ' ')
    text = text.replace('?', ' ')
    text = text.replace(':', '')
    text = text.replace(';', ' ')
    text = text.replace('-', ' ')
    text = text.replace('_', ' ')
    
    text = text.replace('   ', ' ')                 # remove extra spaces
    text = text.replace('  ', ' ')                  # remove extra spaces
    text = text.strip()                             # remove leading and trailing spaces
    return text

reviews_df['text'] = reviews_df['text'].apply(clean_text)

# Split into words
reviews_df['words'] = reviews_df['text'].str.split()

# Remove 1-letter words
reviews_df['words'] = reviews_df['words'].apply(lambda x: [word for word in x if len(word) > 1])

# Remove stop words
stop_words = set(['and', 'the', 'this', 'that', 'is', 'was', 'were', 'has', 'have', 'had', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 
                  'me', 'him', 'her', 'us', 'them', 'are', 'his', 'her', 'its', 'who', 'for', 'with', 'by', 'on', 'of', 'at', 'to', 
                  'from', 'in', 'out', 'further', 'then', 'once', 'here', 'there', 'when', 'what', 'where', 'some', 'as', 'be', 'if', 
                  'an', 'so', 'or', 'if'])
reviews_df['words'] = reviews_df['words'].apply(lambda x: [word for word in x if word not in stop_words])


# Build word-to-token_id map
WORD_TO_NUMBER_MAP = {}
# Sort all words by frequency
words_by_frequency = reviews_df['words'].explode().value_counts()
# Build the map by using word's frequency (the most frequent word gets 0, the second most frequent gets 1, etc.)
for word, _ in words_by_frequency.items():
    WORD_TO_NUMBER_MAP[word] = len(WORD_TO_NUMBER_MAP)

# Convert words to numbers
def convert_words_to_numbers(words: list[str]) -> list[int]:
    return [WORD_TO_NUMBER_MAP[word] for word in words]
reviews_df['word_ids'] = reviews_df['words'].apply(convert_words_to_numbers)



# Vectorize the words - converts a list of word IDs to a vector of 0s and 1s
def vectorize_word_ids(word_ids: list[int]) -> list[int]:
    # Create a vector of zeros (the size of the map)
    vector = np.zeros(len(WORD_TO_NUMBER_MAP), dtype=np.uint8)
    # Loop through each word in the review
    for word_id in word_ids:
        vector[word_id] = 1
    return vector
# Add new column `vector` to the dataframe with 1 vector for each review
reviews_df['vector'] = reviews_df['word_ids'].apply(vectorize_word_ids)


# Split the dataset into training and testing sets
def train_test_split(df: pd.DataFrame, test_size: float = 0.1, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into training and testing sets.
    """
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    # Calculate the number of rows in the test set
    n_test_rows = int(len(df) * test_size)
    # Split the dataset
    train_df = df.iloc[:-n_test_rows]
    test_df = df.iloc[-n_test_rows:]
    return train_df, test_df

train_df, test_df = train_test_split(reviews_df)
X_train = np.array(train_df['vector'].tolist())
y_train = np.array(train_df['label'].tolist())
X_test = np.array(test_df['vector'].tolist())
y_test = np.array(test_df['label'].tolist())



# Fit the model
model = MultinomialNaiveBayesClassifier()
print("Fitting the model...")
model.fit(X_train, y_train)



# Print the confusion matrix
print("Testing the model...")
predictions = model.predict(X_test)
confusion_matrix(y_test, predictions)