import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SimpleAutoencoder:
    '''
    Simple Autoencoder. It is a type of unsupervised learning algorithm used to find the 
    latent features of the data, by compressing it into a lower dimension.
    '''

    input_layer_size: int
    hidden_layer_size: int
    output_layer_size: int
    weights: list[np.ndarray]
    biases: list[np.ndarray]

    def __init__(self, input_layer_size: int, hidden_layer_size: int, output_layer_size: int):
        '''
        Initialize the SimpleAutoencoder class. It will initialize the weights and biases.
        
        Args:
            input_layer_size            : int
            hidden_layer_size           : int
            output_layer_size           : int
        '''
        # Set the layer sizes
        self.input_layer_size = input_layer_size    # It's the number of input features
        self.hidden_layer_size = hidden_layer_size  # It's the number of neurons in the hidden layer (compressed representation)
        self.output_layer_size = output_layer_size  # It's the number of output (decoded) features

        # Initialize weights and biases
        self.weights = [
            np.random.randn(input_layer_size, hidden_layer_size),
            np.random.randn(hidden_layer_size, output_layer_size)
        ]
        self.biases = [
            np.zeros(hidden_layer_size),
            np.zeros(output_layer_size)
        ]

    def sigmoid(self, X: np.ndarray) -> np.ndarray:
        '''Sigmoid function. It's a non-linear function that is used to compress the data within a range of 0 to 1.'''
        return 1 / (1 + np.exp(-X))

    def sigmoid_derivative(self, X: np.ndarray) -> np.ndarray:
        '''Sigmoid derivative function. It's the derivative of the sigmoid function. Needed for the backpropagation.'''
        return X * (1 - X)

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        '''Forward pass. It's the forward pass of the neural network.'''
        # Encoder (m, n) @ (n, hidden_size) = (m, hidden_size) + (hidden_size, 1) = (m, hidden_size)
        hidden = self.sigmoid(np.dot(X, self.weights[0]) + self.biases[0])
        # Decoder (m, hidden_size) @ (hidden_size, n) = (m, n) + (n, 1) = (m, n)    
        output = self.sigmoid(np.dot(hidden, self.weights[1]) + self.biases[1])
        return hidden, output

    def backward(self, X: np.ndarray, hidden: np.ndarray, output: np.ndarray, learn_rate: float):
        '''Backward pass. It's the backward pass of the neural network.'''
        # Calculate gradients
        output_error = output - X # (m, n) - (m, n) = (m, n)
        output_delta = output_error * self.sigmoid_derivative(output) # (m, n) * (m, n) = (m, n)

        hidden_error = np.dot(output_delta, self.weights[1].T) # (m, n) @ (n, hidden_size) = (m, hidden_size)
        hidden_delta = hidden_error * self.sigmoid_derivative(hidden) # (m, hidden_size) * (m, hidden_size) = (m, hidden_size)

        # Update weights and biases
        self.weights[1] -= learn_rate * np.dot(hidden.T, output_delta) # (hidden_size, m) @ (m, n) = (hidden_size, n)
        self.biases[1] -= learn_rate * np.sum(output_delta) # (n, 1)

        self.weights[0] -= learn_rate * np.dot(X.T, hidden_delta) # (n, m) @ (m, hidden_size) = (n, hidden_size)
        self.biases[0] -= learn_rate * np.sum(hidden_delta) # (hidden_size, 1)

    def fit(self, X: np.ndarray, epochs: int, learn_rate: float):
        '''Fit the model to the data.'''
        losses = []

        for epoch in range(epochs):
            # # Shuffle the data
            X = X[np.random.permutation(len(X))]

            # Forward pass
            hidden, output = self.forward(X) # (m, hidden_size), (m, n)

            # Calculate loss (MSE)
            loss = np.mean(np.square(output - X))
            losses.append(loss)
            if epoch % 5000 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

            # Backward pass
            self.backward(X, hidden, output, learn_rate)

        return losses

class Demo:
    def load_data(self) -> np.ndarray:
        '''Load the data from the csv file.'''
        # Print current working directory for PANDAS
        students = pd.read_csv("06. AutoEncoder/data/student_info.csv", index_col=0)

        # Convert categorical values to numeric indices
        cols = ['age', 'grade_level', 'gender', 'parent_education', 'internet_access', 'extra_activities', 'final_result']
        students[cols] = students[cols].apply(lambda x: x.astype('category').cat.codes)
        students.head()

        # Scale the big numbers (0-100) to be between 0 and 1
        cols = ['math_score', 'reading_score', 'writing_score', 'attendance_rate']
        students[cols] = students[cols].apply(lambda x: x/100)

        # Scale the medium numbers (0-5) to be between 0 and 1
        students[['gender', 'age']] = students[['gender', 'age']].apply(lambda x: x/2)
        students[['grade_level', 'parent_education']] = students[['grade_level', 'parent_education']].apply(lambda x: x/3)
        students['study_hours'] = students['study_hours'].apply(lambda x: x/5)

        return students.to_numpy()
    
    def run(self):
        '''Run the demo.'''
        X = self.load_data()
        autoencoder = SimpleAutoencoder(input_layer_size=12, hidden_layer_size=6, output_layer_size=12)
        epochs = 100000
        learn_rate = 0.0005
        losses = autoencoder.fit(X, epochs, learn_rate)

        # Plot the losses
        plt.plot(losses)
        plt.show()

if __name__ == "__main__":
    demo = Demo()
    demo.run()