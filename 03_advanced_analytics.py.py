import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
df = pd.read_csv("Respiratory_AI_Database_v1.csv")

# Clean numeric columns
df['Sample_Size'] = pd.to_numeric(df['Sample_Size'], errors='coerce')
df['Best_Performance_Metric'] = pd.to_numeric(df['Best_Performance_Metric'], errors='coerce')
stat_df = df[(df['Sample_Size'] > 0) & (df['Best_Performance_Metric'] > 0)].dropna(subset=['Sample_Size', 'Best_Performance_Metric'])

# 1. Regression Plot
plt.figure(figsize=(8, 6))
sns.regplot(data=stat_df, x='Sample_Size', y='Best_Performance_Metric', 
            scatter_kws={'alpha':0.5, 'color':'indigo'}, line_kws={'color':'red'})
plt.xscale('log')
plt.title('Regression: Model Performance vs. Cohort Size')
plt.xlabel('Cohort Size (Log Scale)')
plt.ylabel('Reported Performance')
plt.tight_layout()
plt.savefig("Figure_Regression.png", dpi=300)
plt.close()

# 2. Disease vs Algorithm Heatmap
plt.figure(figsize=(10, 8))
top_diseases = df['Target_Condition'].value_counts().nlargest(10).index
top_models = df['Model_Architecture'].value_counts().nlargest(8).index
heatmap_data = df[df['Target_Condition'].isin(top_diseases) & df['Model_Architecture'].isin(top_models)]
pivot_table = pd.crosstab(heatmap_data['Target_Condition'], heatmap_data['Model_Architecture'])

sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", fmt='g', linewidths=.5)
plt.title('Research Concentration: Disease vs. AI Architecture')
plt.xlabel('Model Architecture')
plt.ylabel('Target Condition')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("Figure_Heatmap.png", dpi=300)
plt.close()

# 3. Evidence Maturity Matrix
plt.figure(figsize=(10, 8))
maturity_df = stat_df.groupby('Target_Condition').agg(
    Paper_Count=('PMID', 'count'),
    Median_Sample=('Sample_Size', 'median')
).reset_index()

maturity_df = maturity_df[maturity_df['Paper_Count'] >= 3]

sns.scatterplot(data=maturity_df, x='Paper_Count', y='Median_Sample', s=100, color='teal')
plt.yscale('log')

for i in range(len(maturity_df)):
    if maturity_df['Paper_Count'].iloc[i] > 10 or maturity_df['Median_Sample'].iloc[i] > 500:
        plt.text(maturity_df['Paper_Count'].iloc[i] + 0.5, maturity_df['Median_Sample'].iloc[i], 
                 maturity_df['Target_Condition'].iloc[i], fontsize=9)

plt.axvline(maturity_df['Paper_Count'].median(), color='gray', linestyle='--')
plt.axhline(maturity_df['Median_Sample'].median(), color='gray', linestyle='--')
plt.title('Evidence Maturity Matrix')
plt.xlabel('Total Number of Publications')
plt.ylabel('Median Cohort Size (Log Scale)')
plt.tight_layout()
plt.savefig("Figure_Maturity_Matrix.png", dpi=300)
plt.close()

print("Advanced analytical figures generated and saved.")