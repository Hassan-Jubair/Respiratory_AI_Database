import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Respiratory_AI_Database_v1.csv")

# Clean the year column
clean_df = df[df['Year'].astype(str).str.lower() != 'unknown'].copy()
clean_df['Year'] = pd.to_numeric(clean_df['Year'], errors='coerce')
clean_df = clean_df.dropna(subset=['Year'])
clean_df['Year'] = clean_df['Year'].astype(int)

# Calculate counts per year
trend_data = clean_df['Year'].value_counts().sort_index().reset_index()
trend_data.columns = ['Year', 'Count']

# Generate plot
plt.figure(figsize=(8, 5))
sns.barplot(data=trend_data, x='Year', y='Count', color='skyblue')

for index, row in trend_data.iterrows():
    plt.text(index, row['Count'] + 0.5, str(row['Count']), ha='center', va='bottom', fontsize=10)

plt.title('Figure 1: Publication Trends in Respiratory AI', fontsize=14, pad=15)
plt.xlabel('Year of Publication', fontsize=12)
plt.ylabel('Number of Papers', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.savefig('Figure1_Publication_Trends.png', dpi=300)
print("Figure 1 generated and saved.")