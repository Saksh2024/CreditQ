import pandas as pd
from sklearn.metrics import roc_auc_score

def main():
    print("Loading Credit Set 3...")
    df = pd.read_csv("credit score optimization/credit set 3 exploration.csv")
    
    # Calculate ROC AUC.
    # Note: A higher credit score should mean a lower probability of default.
    # So we invert the score by taking the negative, to match the convention that
    # higher "predicted risk" aligns with the positive class (Default=1).
    auc = roc_auc_score(df['DEFAULT'], -df['CREDIT_SCORE'])
    
    print(f"ROC AUC of CREDIT_SCORE predicting DEFAULT: {auc:.4f}")
    
    if auc > 0.6:
        print("Verification Successful: The CREDIT_SCORE is highly predictive of default!")
    else:
        print("Warning: The CREDIT_SCORE has weak predictive power.")

if __name__ == "__main__":
    main()
