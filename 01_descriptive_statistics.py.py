import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# Load dataset
df = pd.read_csv("Respiratory_AI_Database_v1.csv")

print("*** LOCKDOWN STATISTICS ***\n")

# 1. Filtering Transparency
initial_n = len(df)
clean_df = df[~df['Target_Condition'].astype(str).str.lower().isin(['no abstract'])]
final_n = len(clean_df)

print("1. Filtering Transparency:")
print(f"* Initial Mined Dataset: {initial_n} papers")
print(f"* Removed due to missing abstract/paywall: {initial_n - final_n} papers")
print(f"* Final Locked Dataset (N): {final_n} papers\n")

# 2. Reporting Bias
missing_disease = len(df[df['Target_Condition'].astype(str).str.lower() == 'unknown'])
missing_algo = len(df[df['Model_Architecture'].astype(str).str.lower() == 'not specified'])
df['Sample_Size'] = pd.to_numeric(df['Sample_Size'], errors='coerce')
missing_sample = df['Sample_Size'].isna().sum()

print("2. Literature Reporting Bias (Out of initial 479):")
print(f"* Missing explicit disease label: {missing_disease} ({(missing_disease/initial_n)*100:.1f}%)")
print(f"* Missing specific algorithm: {missing_algo} ({(missing_algo/initial_n)*100:.1f}%)")
print(f"* Missing cohort sample size: {missing_sample} ({(missing_sample/initial_n)*100:.1f}%)\n")

# 3. Time Bias
df['Numeric_Year'] = pd.to_numeric(df['Year'], errors='coerce')
pre_2020 = len(df[df['Numeric_Year'] < 2020])
post_2020 = len(df[df['Numeric_Year'] >= 2020])

print("3. Publication Time Bias:")
print(f"* Pre-2020 Papers: {pre_2020}")
print(f"* Post-2020 Papers: {post_2020}\n")

# 4. Statistical Correlation
stat_df = df[(df['Sample_Size'] > 0) & (pd.to_numeric(df['Best_Performance_Metric'], errors='coerce') > 0)].dropna(subset=['Sample_Size', 'Best_Performance_Metric'])
stat_df['Best_Performance_Metric'] = pd.to_numeric(stat_df['Best_Performance_Metric'])

r, p = spearmanr(stat_df['Sample_Size'], stat_df['Best_Performance_Metric'])
z = np.arctanh(r)
se = 1 / np.sqrt(len(stat_df) - 3)
ci_lower = np.tanh(z - 1.96 * se)
ci_upper = np.tanh(z + 1.96 * se)

print("4. Performance vs Sample Size:")
print(f"* Spearman r: {r:.3f}")
print(f"* 95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
print(f"* p-value: {p:.3f}")