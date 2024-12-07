import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from sklearn.pipeline import Pipeline

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
    
    return X, y, le

# Create a function to visualize confusion matrix
def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(15, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Function to plot training history
def plot_learning_curve(mlp):
    plt.figure(figsize=(10, 5))
    plt.plot(mlp.loss_curve_, label='Training Loss')
    plt.title('Learning Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main execution
if __name__ == "__main__":
    # Load and preprocess data
    X, y, label_encoder = load_and_preprocess_data('mbti_characters.csv')
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, 
                                                        random_state=42, stratify=y)
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),  # Three hidden layers
            activation='relu',
            solver='adam',
            alpha=0.0001,  # L2 regularization
            batch_size='auto',
            learning_rate='adaptive',
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            verbose=True,
            random_state=42
        ))
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    predictions = pipeline.predict(X_test)
    
    # Print model performance
    print("\nModel Performance:")
    print("-" * 50)
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, 
                              target_names=label_encoder.classes_))
    
    # Print distribution of actual vs predicted
    print("\nClass Distribution:")
    print("-" * 50)
    print("Actual labels distribution:", Counter(label_encoder.inverse_transform(y_test)))
    print("Predicted labels distribution:", Counter(label_encoder.inverse_transform(predictions)))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, predictions, label_encoder.classes_)
    
    # Plot learning curve
    plot_learning_curve(pipeline.named_steps['mlp'])
    
    # Calculate and print top misclassifications
    print("\nTop Misclassifications:")
    print("-" * 50)
    misclassified = y_test != predictions
    if np.any(misclassified):
        actual_labels = label_encoder.inverse_transform(y_test[misclassified])
        predicted_labels = label_encoder.inverse_transform(predictions[misclassified])
        misclass_pairs = list(zip(actual_labels, predicted_labels))
        misclass_counts = Counter(misclass_pairs)
        
        print("\nMost common misclassifications (Actual -> Predicted):")
        for (actual, predicted), count in misclass_counts.most_common(5):
            print(f"{actual} -> {predicted}: {count} times")
    
    # Feature importance analysis (based on connection weights)
    feature_names = X.columns
    weights = np.abs(pipeline.named_steps['mlp'].coefs_[0])
    feature_importance = np.sum(weights, axis=1)
    feature_importance = feature_importance / np.sum(feature_importance)
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.barh(pos, feature_importance[sorted_idx])
    plt.yticks(pos, feature_names[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()

# Optional: Grid Search for hyperparameter tuning
def perform_grid_search(X_train, y_train):
    param_grid = {
        'hidden_layer_sizes': [(128, 64), (256, 128, 64), (512, 256, 128)],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.001, 0.01],
    }
    
    mlp = MLPClassifier(max_iter=1000, early_stopping=True, random_state=42)
    grid_search = GridSearchCV(mlp, param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    print("\nBest parameters:", grid_search.best_params_)
    return grid_search.best_estimator_