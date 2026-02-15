import pandas as pd
import pickle
import os

# Import all your model training functions
from model.logistic_regression import train_and_evaluate_logistic_regression
from model.decision_tree import train_and_evaluate_decision_tree
from model.knn import train_and_evaluate_knn
from model.naive_bayes import train_and_evaluate_naive_bayes
from model.random_forest import train_and_evaluate_random_forest
from model.xgboost import train_and_evaluate_xgboost

# Create saved_models directory if it doesn't exist
os.makedirs('saved_models', exist_ok=True)

# Load training data
print("Loading training data...")
df_full = pd.read_csv("data/bank_training_data.csv", sep=';')
print(f"✓ Loaded {len(df_full)} records\n")

# Dictionary of models to train
models_to_train = {
    'Logistic Regression': train_and_evaluate_logistic_regression,
    'Decision Tree': train_and_evaluate_decision_tree,
    'KNN': train_and_evaluate_knn,
    'Naive Bayes': train_and_evaluate_naive_bayes,
    'Random Forest': train_and_evaluate_random_forest,
    'XGBoost': train_and_evaluate_xgboost
}

# Train and save each model
print("=" * 60)
print("Starting model training and saving...")
print("=" * 60)

for model_name, train_function in models_to_train.items():
    print(f"\n🔄 Training {model_name}...")
    
    try:
        # Train the model
        model, metrics = train_function(df_full)
        
        # Create filename
        filename = f"saved_models/{model_name.lower().replace(' ', '_')}_model.pkl"
        
        # Save model using pickle
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"✅ {model_name} trained and saved to {filename}")
        print(f"   Metrics: Accuracy={metrics['Accuracy']:.2%}, AUC={metrics['AUC']:.2%}")
        
    except Exception as e:
        print(f"❌ Error training {model_name}: {str(e)}")

print("\n" + "=" * 60)
print("✅ All models trained and saved successfully!")
print("=" * 60)
print("\nSaved models:")
for filename in os.listdir('saved_models'):
    if filename.endswith('.pkl'):
        filepath = os.path.join('saved_models', filename)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  • {filename} ({size_mb:.2f} MB)")
