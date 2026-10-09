"""
Generates exploratory plot of self-reported LLM confidence versus chunk token context size.

SCIENTIFIC NOTE:
Self-reported LLM confidence scores are uncalibrated and reflect subjective model
certainty, not empirical extraction accuracy or ground-truth relation validity.
This visualization illustrates model confidence behavior across context sizes,
NOT accuracy or quality verification.
"""

import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Connect and Fetch Data
db_path = 'data/relation_dataset.db'
conn = sqlite3.connect(db_path)

# Extracting records from recent sessions
query = "SELECT llm_confidence, chunk_context FROM relation_dataset WHERE created_at LIKE '2026-04%'"
df = pd.read_sql_query(query, conn)
conn.close()

# Use raw token proxy (approx. 4 characters per token)
df['tokens'] = df['chunk_context'].apply(lambda x: int(len(x)/4) if x else 0)
df = df.sort_values(by='tokens')

# Uncalibrated Confidence vs Token Context load
plt.figure(figsize=(10, 6))
plt.scatter(
    df['tokens'],
    df['llm_confidence'],
    color='#27ae60',
    s=50,
    alpha=0.7,
    edgecolors='k',
    label='Self-Reported Model Confidence (Uncalibrated)',
)

# Add trendline to show confidence stability across context size
if len(df) > 1:
    z = np.polyfit(df['tokens'], df['llm_confidence'], 1)
    p = np.poly1d(z)
    plt.plot(df['tokens'], p(df['tokens']), color='black', linestyle='--', alpha=0.4, label='Confidence Trendline')

plt.xlabel('Input Token Context Size (Approx. Chars / 4)', fontsize=12)
plt.ylabel('Uncalibrated Model Confidence Score', fontsize=12)
plt.title('Impact of Token Context on Uncalibrated Model Confidence', fontsize=14, fontweight='bold')
plt.grid(True, linestyle=':', alpha=0.8)
plt.ylim(0, 1.1)
plt.legend()

plt.tight_layout()
plt.savefig('quality_vs_tokens.png', dpi=300)
print("Graph saved: quality_vs_tokens.png")
