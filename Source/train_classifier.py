import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

def build_scoring_model():
    print("Loading enriched credit dataset...")
    df = pd.read_csv("credit score optimization/credit_set_5/Loan_Enriched.csv")
    
    # Target variable: Whether the customer defaults.
    # In Credit Set 5, 'RiskScore' or a custom threshold can determine default. 
    # Let's map any RiskScore > 50 or PreviousLoanDefaults == 1 as a default, or create an organic Default column
    np.random.seed(42)
    # Probability of default is highly correlated with CreditScore and RiskScore
    prob_default = 1 / (1 + np.exp((df['CreditScore'] - 600) / 50))
    df['Default'] = (np.random.rand(len(df)) < prob_default).astype(int)
    
    # Features including demographic features (Gender, EducationLevel, Age, MaritalStatus)
    categorical_cols = ['EmploymentStatus', 'EducationLevel', 'MaritalStatus', 'HomeOwnershipStatus', 'LoanPurpose', 'Gender']
    numerical_cols = ['Age', 'AnnualIncome', 'Experience', 'LoanAmount', 'LoanDuration', 
                      'NumberOfDependents', 'MonthlyDebtPayments', 'CreditCardUtilizationRate', 
                      'NumberOfOpenCreditLines', 'NumberOfCreditInquiries', 'DebtToIncomeRatio', 
                      'BankruptcyHistory', 'PreviousLoanDefaults', 'PaymentHistory', 
                      'LengthOfCreditHistory', 'SavingsAccountBalance', 'CheckingAccountBalance', 
                      'TotalAssets', 'TotalLiabilities', 'JobTenure', 'NetWorth']
    
    # Process categorical variables
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        
    X = df[categorical_cols + numerical_cols]
    y = df['Default']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training credit scoring model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    acc = model.score(X_test_scaled, y_test)
    print(f"Credit scoring model trained. Test Accuracy: {acc:.2%}")
    
    # Save objects
    joblib.dump(model, "Source/credit_classifier.pkl")
    joblib.dump(scaler, "Source/scaler.pkl")
    joblib.dump(encoders, "Source/encoders.pkl")
    
    # Save dataset with predicted probabilities to simulate scoring
    df['PD'] = model.predict_proba(scaler.transform(X))[:, 1]
    # Calibrate credit score: 300 to 850 based on PD
    df['CustomCreditScore'] = (300 + 550 * (1 - df['PD'])).astype(int)
    df.to_csv("credit score optimization/credit_set_5/Loan_Scored.csv", index=False)
    print("Saved scored data to 'credit score optimization/credit_set_5/Loan_Scored.csv'")

if __name__ == "__main__":
    build_scoring_model()
