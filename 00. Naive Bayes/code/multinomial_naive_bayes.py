import numpy as np

class MultinomialNaiveBayesClassifier:
    """
    Naive Bayes Classifier. It uses the Naive Bayes theorem and Laplace smoothing to classify text data.
    """

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the model to the text data.

        Args:
            X: numpy array of shape (n_samples, n_words)
            y: numpy array of shape (n_samples,) - target labels
        """
        n_samples, n_words = X.shape
        self._vocab_size = n_words      # The number of unique words in the vocabulary
        self._classes = np.unique(y)    # The output classes (in our case - 0 and 1)
        n_classes = len(self._classes)  # The number of unique classes (in our case - 2)

        # Pre-calculate word counts and class (prior) probabilities for each class. We will use them later to calculate the posterior probabilities.
        self._total_words_count = np.zeros(n_classes, dtype=np.int32)
        self._word_freqs = np.zeros((n_classes, n_words), dtype=np.int32)
        self._class_probabilities = np.zeros(n_classes, dtype=np.float32)

        # For each class (in our case - 0 and 1)
        for idx, c in enumerate(self._classes):
            # Filter the samples for this class
            X_c = X[y == c]

            # Calculate the total number of words in all samples for this class
            self._total_words_count[idx] = np.sum(X_c)

            # Calculate the number of occurrences for each word in all samples for this class
            # NB: We use the np.sum method along the second axis (axis=1) to sum the values along the rows (axis=0).
            self._word_freqs[idx, :] = np.sum(X_c, axis=0)
                
            # Calculate the class (prior) probability for this class (num of class occurrences / total num of samples)
            self._class_probabilities[idx] = X_c.shape[0] / float(n_samples)

    def print(self):
        """Print the model parameters."""
        for idx, c in enumerate(self._classes):
            print(f'Class {self._classes[idx]}')
            print(f'\t Vocab size: {self._vocab_size}')
            print(f'\t Class probability: {self._class_probabilities[idx]:.2f}')
            print(f'\t Total words count: {self._total_words_count[idx]:,}')
            print(f'\t Word frequencies (max value): {np.max(self._word_freqs[idx]):,}')
            print(f'\t Word frequencies (min value): {np.min(self._word_freqs[idx]):,}')

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the class (0 or 1) for each sample in the input data."""
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred, dtype=np.uint8)

    def _predict (self, x: np.ndarray) -> int:
        """
        Calculate the posterior probability for each class.
          P(c|X) = P(C) * P(x_1 | C) * P(x_2 | C) * P(x_3 | C) * P(x_4 | C) *  ...

        Args:
            x: numpy array of shape (n_words,) - input features (e.g. [0, 1, 0, 1, 0])

        Returns:
            int: the class with the highest posterior probability (e.g. 0 or 1)
        """
        posteriors = []

        for idx, c in enumerate(self._classes):
            # P(C) - in our case it is 0.5
            class_probability = self._class_probabilities[idx]
            # P(x_1 | C), where x_1 is the token_id of the first word, x_2 is the token_id of the second word, etc.
            word_probabilities = self._laplace_smoothing(idx, x)
            # We will use logarithm to increase the probability values and avoid UNDERFLOW issues.
            # EXAMPLE: log(0.003) = -2.52
            word_probability_logs = np.log(word_probabilities)
            #  P(c|X) = P(C) * P(x_1 | C) * P(x_2 | C) *
            posterior_probability = class_probability * np.dot(x.T, word_probability_logs)
            posteriors.append(posterior_probability)

        # Return the class with the highest posterior probability (e.g. 0 or 1)
        return self._classes[np.argmax(posteriors)]
    
    def _laplace_smoothing(self, class_idx: int, x: np.ndarray):
        """
        Calculate the posterior probability for each word in the class. We use Laplace smoothing to avoid zero probabilities.

        Laplace formula:
            P(x_i | C) = (N_i + 1) / (N + V)
        where:
            N_i - the number of occurrences of the word (within the class)
            N   - the total number of words (within the class)
            V   - the number of unique words in the vocabulary

        Args:
            class_idx: int - the index of the class
            x: numpy array of shape (n_words,) - input features (e.g. [0, 1, 0, 1, 0])

        Returns:
            numpy array of shape (n_words,) - posterior probabilities for each word
        """
        # For all words in the input vector the denominator is the same: (TOTAL_WORD_COUNT + VOCAB_SIZE)
        V = self._vocab_size
        N = self._total_words_count[class_idx]

        # For each word in the input vector we calculate the posterior probability of that word (P(x_i | C))
        # CASE 1: The word is present in the text sample
        #         N_i = self._word_counts[class_idx][i] + 1
        # CASE 2: The word is not present in the text sample
        #         For missing words return a probability of 10 (log(10) = 1). This way it won't affect the final probability product.
        word_posterior_probabilities = np.where(x == 1, (self._word_freqs[class_idx] + 1) / (N + V), 10)
        # Return a vector of posterior probabilities for each word [0.0003, 0.00027, 10, 0.0004, 0.0005]
        return word_posterior_probabilities
    
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Calculate and print the confusion matrix
    """
    # Calculate the confusion matrix
    true_positives = sum(1 for p, y in zip(y_pred, y_true) if p == 1 and y == 1)
    true_negatives = sum(1 for p, y in zip(y_pred, y_true) if p == 0 and y == 0)
    false_positives = sum(1 for p, y in zip(y_pred, y_true) if p == 1 and y == 0)
    false_negatives = sum(1 for p, y in zip(y_pred, y_true) if p == 0 and y == 1)
    # Print the confusion matrix in a matrix format
    print(f'Confusion Matrix:')
    print(f'---------------------')
    print(f' |  0\t|  1')
    print(f'---------------------')
    print(f'0| {true_positives}\t| {false_positives}')
    print(f'1| {false_negatives}\t| {true_negatives}')
    print(f'---------------------')
    accuracy = (true_positives + true_negatives) / len(y_true)
    print(f'Accuracy:  {accuracy:.2f}')
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)
    print(f'Precision: {precision:.2f}')
    print(f'Recall:    {recall:.2f}')
    print(f'F1-Score:  {f1_score:.2f}')