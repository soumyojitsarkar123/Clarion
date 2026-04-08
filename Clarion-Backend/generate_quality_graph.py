import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Connect and Fetch Real Data
db_path = 'data/relation_dataset.db'
conn = sqlite3.connect(db_path)

# Extracting records from recent sessions
query = "SELECT llm_confidence, chunk_context FROM relation_dataset WHERE created_at LIKE '2026-04%'"
df = pd.read_sql_query(query, conn)
conn.close()

# Use raw token proxy (4 characters per token)
df['tokens'] = df['chunk_context'].apply(lambda x: int(len(x)/4))
df = df.sort_values(by='tokens')

#Quality (Confidence) vs Token Context load
plt.figure(figsize=(10, 6))
plt.scatter(df['tokens'], df['llm_confidence'], color='#27ae60', s=50, alpha=0.7, edgecolors='k', label='Model Confidence (Real)')

# Add trendline to show quality stability/degradation
z = np.polyfit(df['tokens'], df['llm_confidence'], 1)
p = np.poly1d(z)
plt.plot(df['tokens'], p(df['tokens']), color='black', linestyle='--', alpha=0.4, label='Quality Trend')

plt.xlabel('Input Token Context Size', fontsize=12)
plt.ylabel('Confidence Score (Accuracy Proxy)', fontsize=12)
plt.title('Impact of Token Context on Generation Quality (Optimized)', fontsize=14, fontweight='bold')
plt.grid(True, linestyle=':', alpha=0.8)
plt.ylim(0, 1.1)
plt.legend()

plt.tight_layout()
plt.savefig('optimized_quality_vs_tokens.png', dpi=300)
print("Graph saved: optimized_quality_vs_tokens.png")
