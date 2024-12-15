import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'mbti_characters.csv'  # Replace with the path to your dataset
data = pd.read_csv(file_path)

# Select relevant columns for features and targets
features_columns = ['big_5_SLOAN', 'socionics', 'attitudinal_psyche', 'classic_jungian', 'enneagram']
target_columns = ['four_letter']

data = data[features_columns + target_columns].dropna()

# Encode categorical variables
encoder = LabelEncoder()
for col in ['big_5_SLOAN', 'socionics', 'attitudinal_psyche', 'classic_jungian', 'enneagram', 'four_letter']:
    data[col] = encoder.fit_transform(data[col])

# Separate features and target variables
X = data[features_columns]
y = data[target_columns].values.ravel()

# Convert target to categorical
num_classes = len(np.unique(y))
y_categorical = to_categorical(y, num_classes=num_classes)

# Normalize features
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_categorical, test_size=0.25, random_state=42)

# Define the neural network architecture
# model = Sequential([
#     Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
#     Dropout(0.3),
#     Dense(128, activation='relu'),
#     Dropout(0.3),
#     Dense(64, activation='relu'),
#     Dense(num_classes, activation='softmax')
# ])
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model and capture history
history = model.fit(X_train, y_train, validation_split=0.1, epochs=200, batch_size=64, verbose=1)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Plot the loss curve
plt.figure(figsize=(12, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot the accuracy curve
plt.figure(figsize=(12, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
