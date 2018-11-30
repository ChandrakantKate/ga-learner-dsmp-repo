# --------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# code starts here
df = pd.read_csv(path)
p_a = len(df[df['fico']>700])/len(df)
p_b = len(df[df['purpose']=='debt_consolidation'])/len(df)
print(p_b)
df1 = df[df['purpose']=='debt_consolation']
print(len(df1))
p_a_b = (len(df1[df1['fico']>700]/len(df)))/p_a

result = p_a_b==p_a
# code ends here


# --------------
# code starts here
prob_lp = len(df[df['paid.back.loan']=='Yes'])/len(df)

prob_cs = len(df[df['credit.policy']=='Yes'])/len(df)

new_df = df[df['paid.back.loan']=='Yes']

prob_pd_cs = len(new_df[new_df['credit.policy']=='Yes'])/len(new_df)

bayes = prob_pd_cs*prob_lp/prob_cs
# code ends here


# --------------
# code starts here
pb = df['purpose'].value_counts()

plt.bar(pb.index,pb.values)
print()

df1 = df[df['paid.back.loan']=='No']

pb1 = df1['purpose'].value_counts()
plt.bar(pb1.index,pb1.values)
#plt.bar(df1['purpose'])
plt.show()
# code ends here


# --------------
# code starts here
inst_median = np.median(df['installment'])

inst_mean = np.mean(df['installment'])

plt.hist(df['installment'],bins=20)
plt.show()
# code ends here


