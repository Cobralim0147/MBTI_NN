import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from collections import Counter



def visualise(mlp, input_label):

    # Print the input data
    print("Inputs fed to the network:")
    print(input_label)

    # Get the structure of the network
    n_neurons = [mlp.coefs_[0].shape[0]]  # Input layer neurons
    n_neurons += [layer.shape[1] for layer in mlp.coefs_]  # Hidden and output layers

    # Coordinates of neurons
    layer_positions = np.linspace(0, len(n_neurons) - 1, len(n_neurons))
    neuron_positions = [
        np.linspace(-1, 1, neurons) for neurons in n_neurons
    ]

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect('equal')
    ax.axis('off')

    # Draw connections
    for l, weights in enumerate(mlp.coefs_):
        for i, source_pos in enumerate(neuron_positions[l]):
            for j, target_pos in enumerate(neuron_positions[l + 1]):
                weight = weights[i, j]
                color = 'blue' if weight > 0 else 'red'
                lw = np.abs(weight) * 2  # Line width based on weight magnitude
                ax.plot([layer_positions[l], layer_positions[l + 1]],
                        [source_pos, target_pos], color=color, linewidth=lw)

    # Draw neurons
    for l, layer in enumerate(neuron_positions):
        ax.scatter([layer_positions[l]] * len(layer), layer, s=100, zorder=10, color='black')
        # Annotate input neurons with feature names
        if l == 0:  # Input layer
            for i, pos in enumerate(layer):
                ax.text(layer_positions[l] - 0.1, pos, input_label[i],
                        fontsize=10, ha='right', color='green')
    
    plt.show()


# Load the dataset
file_path = 'mbti_characters.csv'  # Replace with the path to your dataset
data = pd.read_csv(file_path)

# Select relevant columns for features and targets
features_columns = ['big_5_SLOAN', 'socionics', 'attitudinal_psyche', 'classic_jungian', 'enneagram']
target_columns = ['four_letter']

#drop data that are empty, for consistancy
data = data[features_columns + target_columns].dropna()

# Encode categorical variables
encoder = LabelEncoder()
for col in ['big_5_SLOAN', 'socionics', 'attitudinal_psyche', 'classic_jungian', 'enneagram', 'four_letter']:
    data[col] = encoder.fit_transform(data[col])

# Separate features and target variables
X = data[features_columns]
y = data[target_columns].values.ravel()

# Normalize features and targets
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = y

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.25, random_state=42)

# mlp = MLPClassifier(hidden_layer_sizes=(5), max_iter=50)
mlp = MLPClassifier(
    hidden_layer_sizes=(20, 20, 20), max_iter=200, activation='tanh', solver='lbfgs', random_state=42)

mlp.fit(X_train, y_train)
predictions = mlp.predict(X_test)

#to evaluate the matrix of the neuro network 
print(f'confusion matrix= \n{confusion_matrix(y_test, predictions)}')
print(f'classification report= \n{classification_report(y_test, predictions)}')
print("Final training loss:", mlp.loss_)
print("Actual labels:", Counter(y_test))
print("Predicted labels:", Counter(predictions))

#to visualize the chart
# visualise(mlp, features_columns)
