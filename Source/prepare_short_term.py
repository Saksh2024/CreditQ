import os
import pandas as pd
import numpy as np

def prepare_short_term_dataset():
    input_path = os.path.join(os.path.dirname(__file__), "..", "credit score optimization", "credit set 6.csv")
    output_path = os.path.join(os.path.dirname(__file__), "short_term_loans_scored.csv")
    
    print(f"Reading records from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Initial raw rows read: {len(df):,}")
    
    # 1. Impute missing Credit_Score using Default_Risk group median
    df['Credit_Score'] = df['Credit_Score'].fillna(df.groupby('Default_Risk')['Credit_Score'].transform('median'))
    df['Income_USD'] = df['Income_USD'].fillna(df['Income_USD'].median())
    
    # 2. Calibrate Probability of Default (PD) for short-term micro-loans
    # Micro-loan industry benchmarks: Low Risk ~ 3% default, Medium Risk ~ 30%, High Risk ~ 80%
    base_pd = {'High': 0.80, 'Medium': 0.30, 'Low': 0.03}
    df['base_pd'] = df['Default_Risk'].map(base_pd)
    
    # Late payment history adjustment (+10% default risk for late payers)
    late_adj = df['Late_Payment_History'].map({'Yes': 0.10, 'No': -0.01})
    
    # Active BNPL debt overload (+5% default risk if >= 3 active micro-loans)
    loan_adj = np.where(df['Total_BNPL_Active_Loans'] >= 3, 0.05, 0.0)
    
    # Clipped continuous PD
    df['PD'] = np.clip(df['base_pd'] + late_adj + loan_adj, 0.01, 0.95).astype(np.float32)
    
    # 3. Realized Default simulation using fixed seed for reproducibility
    np.random.seed(42)
    df['Default'] = (np.random.rand(len(df)) < df['PD']).astype(np.int32)
    
    # 4. Standardized features matching CreditQ unified schema
    df['CreditScore'] = df['Credit_Score'].astype(np.float32)
    df['LoanAmount'] = df['Average_Transaction_Value_USD'].astype(np.float32)
    
    # Short-term micro-loan fee structure: 8% fee / margin per origination cycle
    df['InterestRate'] = np.full(len(df), 0.08, dtype=np.float32)
    
    clean_df = df[['CreditScore', 'LoanAmount', 'InterestRate', 'Default', 'PD']].copy()
    
    print(f"\nClean Short-Term Micro-Loan Dataset Summary:")
    print(f"Total Rows: {len(clean_df):,}")
    print(f"Credit Score Range: {clean_df['CreditScore'].min():.0f} - {clean_df['CreditScore'].max():.0f} (Mean: {clean_df['CreditScore'].mean():.1f})")
    print(f"Overall Default Rate: {clean_df['Default'].mean()*100:.2f}%")
    print(f"Mean Loan Ticket Size: ${clean_df['LoanAmount'].mean():,.2f} (Range: ${clean_df['LoanAmount'].min():.0f} - ${clean_df['LoanAmount'].max():.0f})")
    print(f"Fixed Short-Term Fee Rate: {clean_df['InterestRate'].mean()*100:.1f}%")
    
    clean_df.to_csv(output_path, index=False)
    print(f"\nSaved clean scored dataset to: {output_path}")
    return clean_df

if __name__ == "__main__":
    prepare_short_term_dataset()
