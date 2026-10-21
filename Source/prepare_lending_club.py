import os
import gzip
import pandas as pd
import numpy as np

def prepare_lending_club_dataset():
    input_path = os.path.join(os.path.dirname(__file__), "..", "credit score optimization", "lending club 07-18 accepted.gz")
    output_path = os.path.join(os.path.dirname(__file__), "lending_club_scored.csv")
    
    print(f"Reading records from {input_path}...")
    with gzip.open(input_path, 'rt', encoding='utf-8') as f:
        # Read 120,000 rows to get 100,000 clean completed loans
        df = pd.read_csv(f, nrows=120000, low_memory=False)
        
    print(f"Initial raw rows read: {len(df):,}")
    
    # Filter completed loans only
    df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])].copy()
    
    # Extract & rename columns
    df['Default'] = (df['loan_status'] == 'Charged Off').astype(np.int32)
    df['CreditScore'] = df['fico_range_low'].astype(np.float32)
    df['LoanAmount'] = df['loan_amnt'].astype(np.float32)
    
    # Convert interest rate string to float (e.g. 12.5% or numeric 12.5 -> 0.125)
    if df['int_rate'].dtype == object:
        df['InterestRate'] = df['int_rate'].astype(str).str.rstrip('%').astype(np.float32) / 100.0
    else:
        df['InterestRate'] = df['int_rate'].astype(np.float32) / 100.0 if df['int_rate'].max() > 1.0 else df['int_rate'].astype(np.float32)
        
    # Clean required subset
    clean_df = df[['CreditScore', 'LoanAmount', 'InterestRate', 'Default']].dropna().reset_index(drop=True)
    
    # Calculate smooth Probability of Default (PD) using empirical score binning + smoothing
    clean_df['score_bin'] = pd.qcut(clean_df['CreditScore'], q=20, duplicates='drop')
    bin_pd = clean_df.groupby('score_bin', observed=True)['Default'].transform('mean')
    clean_df['PD'] = bin_pd.astype(np.float32)
    clean_df = clean_df.drop(columns=['score_bin'])
    
    # Take clean 100,000 records
    clean_df = clean_df.iloc[:100000].reset_index(drop=True)
    
    print(f"\nFinal Clean LendingClub Dataset Summary:")
    print(f"Total Rows: {len(clean_df):,}")
    print(f"Credit Score Range: {clean_df['CreditScore'].min()} - {clean_df['CreditScore'].max()} (Mean: {clean_df['CreditScore'].mean():.1f})")
    print(f"Overall Default Rate: {clean_df['Default'].mean()*100:.2f}%")
    print(f"Mean Loan Amount: ${clean_df['LoanAmount'].mean():,.2f}")
    print(f"Mean Interest Rate: {clean_df['InterestRate'].mean()*100:.2f}%")
    
    clean_df.to_csv(output_path, index=False)
    print(f"\nSaved clean scored dataset to: {output_path}")

if __name__ == "__main__":
    prepare_lending_club_dataset()
