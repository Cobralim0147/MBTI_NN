import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load and preprocess data
def load_and_preprocess_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Select relevant columns
    features_columns = ['category', 'subcategory', 'temperaments', 'big_5_SLOAN']
    target_column = ['four_letter']
    
    # Drop rows with missing values
    data = data[features_columns + target_column].dropna()
    
    # Create dummy variables for categorical features
    X = pd.get_dummies(data[features_columns])
    
    # Encode target variable
    le = LabelEncoder()
    y = le.fit_transform(data[target_column].values.ravel())
    # Convert to one-hot encoding
    y = to_categorical(y)
    
    return X, y, le

# Create neural network model
def create_model(input_dim, output_dim):
    model = Sequential([
        # Input layer
        Dense(512, input_dim=input_dim, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        # Hidden layers
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        # Output layer
        Dense(output_dim, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Main execution
X, y, label_encoder = load_and_preprocess_data('mbti_characters.csv')

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

# Create and train model
model = create_model(X_train.shape[1], y_train.shape[1])
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=300,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Evaluate model
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nTest Accuracy: {test_accuracy:.4f}")

# Make predictions
predictions = model.predict(X_test_scaled)
predicted_classes = np.argmax(predictions, axis=1)
actual_classes = np.argmax(y_test, axis=1)

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Display sample predictions
print("\nSample Predictions vs Actual Values:")
print("Predicted MBTI | Actual MBTI")
print("-" * 40)
for i in range(10):  # Show first 10 examples
    pred_mbti = label_encoder.inverse_transform([predicted_classes[i]])[0]
    actual_mbti = label_encoder.inverse_transform([actual_classes[i]])[0]
    print(f"Predicted: {pred_mbti} | Actual: {actual_mbti}")

# Calculate overall accuracy per personality type
unique_types = label_encoder.classes_
type_accuracy = {}

for idx, mbti_type in enumerate(unique_types):
    type_mask = actual_classes == idx
    if np.any(type_mask):
        correct = np.sum((predicted_classes == actual_classes) & type_mask)
        total = np.sum(type_mask)
        type_accuracy[mbti_type] = correct / total

print("\nAccuracy per MBTI type:")
for mbti_type, acc in type_accuracy.items():
    print(f"{mbti_type}: {acc:.4f}")