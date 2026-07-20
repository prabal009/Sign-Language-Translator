import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

from utils.config import DATASET_PATH, LABELS

def train_pipeline():
    print("🚀 Starting AI Training Pipeline...")
    
    # 1. Load Dataset (skipping the header row)
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"❌ Dataset not found at {DATASET_PATH}. Please run data collection first!")
        
    df = pd.read_csv(DATASET_PATH, header=None, skiprows=1)
    print(f"📊 Dataset loaded successfully. Shape: {df.shape}")
    
    # Split features (X) and labels (y)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # 2. Train/Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"📉 Data split: {len(X_train)} training samples, {len(X_test)} testing samples.")
    
    # 3. Define Models to Evaluate
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True, kernel='rbf', C=1.0, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }
    
    best_accuracy = 0.0
    best_model = None
    best_model_name = ""
    
    # 4. Train and Cross-Evaluate with Single-Class Safety Net
    print("\n--- Training & Evaluation ---")
    for name, model in models.items():
        print(f"⏳ Training {name}...")
        try:
            model.fit(X_train, y_train)
            
            # Predict
            predictions = model.predict(X_test)
            acc = accuracy_score(y_test, predictions)
            print(f"✅ {name} Accuracy: {acc * 100:.2f}%")
            
            # Track the winner
            if acc > best_accuracy:
                best_accuracy = acc
                best_model = model
                best_model_name = name
        except Exception as e:
            print(f"⚠️ {name} skipped: {str(e)}")
            
    # 5. Save the Winner if at least one model trained successfully
    if best_model is not None:
        print("\n--- Final Evaluation Report ---")
        final_preds = best_model.predict(X_test)
        print(f"🏆 Winner: {best_model_name} with {best_accuracy * 100:.2f}% accuracy!")
        print("\n📝 Classification Report:")
        print(classification_report(y_test, final_preds, zero_division=0))
        
        os.makedirs("models", exist_ok=True)
        model_save_path = os.path.join("models", "best_sign_model.pkl")
        
        model_data = {
            "model": best_model,
            "model_name": best_model_name,
            "labels_mapping": LABELS
        }
        
        joblib.dump(model_data, model_save_path)
        print(f"💾 Successfully saved the winner to '{model_save_path}'!\n")
    else:
        print("\n❌ Error: No models were able to train successfully.")

if __name__ == "__main__":
    train_pipeline()