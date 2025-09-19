import os
import requests
import zipfile
import numpy as np

class DataLoader:
    """
    Load and preprocess the data.
    """

    @ staticmethod
    def load_and_preprocess_data():
        """
        Load and preprocess the data.
        """
        # Load the data.
        X_train, Y_train, X_test, Y_test = DataLoader.load_data()
        # Preprocess the data.
        print("5. Preprocessing the data...")
        X_train_processed, Y_train_processed = DataLoader.preprocess_data(X_train, Y_train, limit=500)
        X_test_processed, Y_test_processed = DataLoader.preprocess_data(X_test, Y_test, limit=500)
        print(f"\tX_train_processed: {X_train_processed.shape}, Y_train_processed: {Y_train_processed.shape}")
        print(f"\tX_test_processed: {X_test_processed.shape}, Y_test_processed: {Y_test_processed.shape}")
        return X_train_processed, Y_train_processed, X_test_processed, Y_test_processed

    @staticmethod
    def load_data():
        """
        Load the data.
        """
        # STEP 1: Create `data` folder, if it does not exist.
        if not os.path.isdir("data"):
            os.makedirs("data")

        # STEP 2: Load the data from the server, unless the file exists.
        if not os.path.isfile("data/mnist.zip"):
            print("1. Downloading the mnist.zip file...")
            response = requests.get("https://www.kaggle.com/api/v1/datasets/download/oddrationale/mnist-in-csv")
            with open("data/mnist.zip", "wb") as f:
                f.write(response.content)

        # STEP 3: Unpack the `mnist.zip` file, if it is not already unpacked.
        if not os.path.isfile("data/mnist/mnist_train.csv"):
            print("2. Unpacking the `mnist.zip` file...")
            with zipfile.ZipFile('data/mnist.zip', 'r') as zip_ref:
                zip_ref.extractall('data/mnist')

        # STEP 4: Read the csv files.
        print("3. Reading the csv files...")
        X_train = np.loadtxt("data/mnist/mnist_train.csv", delimiter=",", skiprows=1, dtype=np.uint8)
        X_test = np.loadtxt("data/mnist/mnist_test.csv", delimiter=",", skiprows=1, dtype=np.uint8)

        # STEP 5: Unpack the data.
        print("4. Unpacking the data...")
        # Extract the first column as the label.
        Y_train, Y_test = X_train[:, 0], X_test[:, 0]
        X_train, X_test = X_train[:, 1:], X_test[:, 1:]
        # Reshape from (60000, 784) to (60000, 28, 28)
        X_train = X_train.reshape(-1, 28, 28)
        X_test = X_test.reshape(-1, 28, 28)
        print(f"\tX_train: {X_train.shape}, X_test: {X_test.shape}, Y_train: {Y_train.shape}, Y_test: {Y_test.shape}")

        return X_train, Y_train, X_test, Y_test

    @staticmethod
    def one_hot_encode(Y: np.ndarray, num_classes: int):
        """
        Encode output (a number in [0,9]) into a vector of size 10 (e.g. 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0])

        Args:
            Y: np.ndarray (e.g. with shape (60000,))

        Returns:
            Y_encoded: np.ndarray (e.g. with shape (60000, 10))
        """
        # Initialize a matrix of zeros with shape (len(Y), num_classes - e.g. (1000, 2))
        Y_encoded = np.zeros((Y.shape[0], num_classes))
        # For each row, set the corresponding column (feature) to 1 (e.g. [0, 1] for 1 and [1, 0] for 0)
        for i in range(Y.shape[0]):
            Y_encoded[i, Y[i]] = 1
        # Return the encoded matrix
        return Y_encoded

    @staticmethod
    def preprocess_data(X: np.ndarray, Y: np.ndarray, limit: int):
        """
        Preprocess data for training a convolutional neural network.

        Args:
            X: np.ndarray (e.g. with shape (1000, 28, 28))
            Y: np.ndarray (e.g. with shape (1000,))
            limit: int

        Returns:
            X: np.ndarray (e.g. with shape (1000, 28 * 28, 1))
            Y: np.ndarray (e.g. with shape (1000, 10, 1))
        """
        # Limit only to 0 and 1 images.
        zero_index = np.where(Y == 0)[0][:limit]
        one_index = np.where(Y == 1)[0][:limit]
        all_indices = np.concatenate((zero_index, one_index))
        X, Y = X[all_indices], Y[all_indices]
        # Reshape from (1000, 28, 28) to (1000, 1, 28, 28) since the CNN expects the depth to be the first dimension
        X = X.reshape(len(X), 1, 28, 28)
        # Normalize [0, 255] to [0, 1]
        X = X.astype("float32") / 255
        # Encode output (a number in [0,9]) into a vector of size 10 (e.g. 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        Y = DataLoader.one_hot_encode(Y, 2)
        # Reshape from (1000, 2) to (1000, 2, 1) - i.e. to be a column vector, since this is what the dense layer expects as input
        Y = Y.reshape(len(Y), 2, 1)
        return X, Y