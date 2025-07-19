import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os
import joblib

def extract_features(url):
    features = {}

    if not isinstance(url, str) or not url:
        return features 

if __name__ == "__main__":
    try:
        print("Loading dataset...")
        df = pd.read_csv('/data/phishing_site_urls.csv')
        
        # Drop rows where 'URL' is NaN
        df.dropna(subset=['URL'], inplace=True)
        
        print("Extracting features...")
        features_df = df['URL'].apply(lambda url: pd.Series(extract_features(url)))

        # Prepare data for training
        X = features_df
        y = df['Label'].apply(lambda label: 1 if label == 'bad' else 0)

        # Split the dataset into training and testing sets
        print("Splitting dataset into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train the model
        print("Training the model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        # Evaluate the model
        print("Evaluating the model...")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Legitimate (good)', 'Phishing (bad)'])

        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)

        # Save the model with datetime in filename
        from datetime import datetime
        if not os.path.exists('/models'):
            os.makedirs('/models')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = f'/models/phishing_model_{timestamp}.joblib'
        joblib.dump(model, model_path)

        print(f"Model saved to {model_path}")

    except FileNotFoundError:
        print("File not found. Please ensure the path is correct.")
        print("Please download the dataset and place it in the /data directory.")
        exit()