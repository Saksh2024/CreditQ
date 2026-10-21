import pandas as pd
import numpy as np

# Load Credit Set 5 (Loan.csv) which has 35+ numerical features
loan_df = pd.read_csv("credit score optimization/credit_set_5/Loan.csv")

# Load Credit Set 2 (UCI dataset) which has 'personal_status' (e.g. 'male single', 'female div/dep/mar')
german_df = pd.read_csv("credit score optimization/credit set 2 uncleaned.csv")

print("Merging gender/personal status variables from UCI into synthetic data...")
# Sample personal_status and assign it to our Loan.csv records to keep Gender feature
personal_statuses = german_df['personal_status'].values
np.random.seed(42)
loan_df['PersonalStatus'] = np.random.choice(personal_statuses, size=len(loan_df))

# Map PersonalStatus into Gender and MaritalStatus
def extract_gender(val):
    if str(val).startswith('male'):
        return 'Male'
    elif str(val).startswith('female'):
        return 'Female'
    else:
        return 'Unspecified'

loan_df['Gender'] = loan_df['PersonalStatus'].apply(extract_gender)

# Save the enriched dataset for our RL Environment
loan_df.to_csv("credit score optimization/credit_set_5/Loan_Enriched.csv", index=False)
print("Saved enriched dataset to 'credit score optimization/credit_set_5/Loan_Enriched.csv'")
