# --------------
# code starts here
mask1 = banks['Self_Employed']=='Yes'
mask2 = banks['Loan_Status']=='Y'
loan_approved_se = banks.loc[mask1 & mask2].shape[0]
#print (loan_approved_se)
mask3 = banks['Self_Employed']=='No'
loan_approved_nse = banks.loc[mask2 & mask3].shape[0]

percentage_se = loan_approved_se/614 * 100
percentage_nse = loan_approved_nse/614 * 100

# code ends here


# --------------
# code starts here

loan_term = banks['Loan_Amount_Term'].apply(lambda x: x/12)
#print(loan_term)

big_loan_term = banks.loc[loan_term>=25].shape[0]

# code ends here


# --------------
# Import packages
import numpy as np
import pandas as pd
from scipy.stats import mode 
 
bank =  pd.read_csv(path,sep=',')
categorical_var = bank.select_dtypes(include = 'object')
print (categorical_var)
# code starts here
numerical_var = bank.select_dtypes(include = 'number')
print(numerical_var)




# code ends here


# --------------
# code ends here
#print(banks.head())
loan_groupby = bank.groupby(['Loan_Status'],axis=0)['ApplicantIncome', 'Credit_History']
#print(loan_groupby)
mean_values = loan_groupby.agg([np.mean])
print(mean_values)

# code ends here


# --------------
# code starts here
banks = bank.drop(columns='Loan_ID')

print(banks.isnull().sum())

bank_mode = banks.mode()

banks.fillna(bank_mode.iloc[0],inplace=True)
#print(bank_mode)
#code ends here


# --------------
# Code starts here

avg_loan_amount =  banks.pivot_table(index=['Gender','Married','Self_Employed'], values='LoanAmount', aggfunc='mean')

# code ends here



