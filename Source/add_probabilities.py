import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def main():
    print("Loading Credit Set 3...")
    df = pd.read_csv("credit score optimization/credit set 3 exploration.csv")
    
    # Drop identifiers and target variables for training features
    features = df.drop(columns=['CUST_ID', 'CREDIT_SCORE', 'DEFAULT'])
    
    # Encode categorical columns
    categorical_cols = features.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        features[col] = le.fit_transform(features[col].astype(str))
        
    X = features
    y = df['DEFAULT']
    
    print("Training Random Forest to calculate default probabilities...")
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    # Predict probabilities (column 1 is the probability of DEFAULT=1)
    prob_default = model.predict_proba(X)[:, 1]
    
    # Append the probability column
    df['DEFAULT_PROB'] = prob_default
    
    # Save the new dataset
    output_path = "credit score optimization/credit_set_3_with_probs.csv"
    df.to_csv(output_path, index=False)
    print(f"Successfully saved new dataset with probabilities to {output_path}")

if __name__ == "__main__":
    main()
