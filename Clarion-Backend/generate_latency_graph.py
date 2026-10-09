"""
Generates exploratory plot of token context versus derived session latency estimate.

SCIENTIFIC NOTE:
The latency values depicted in this script are synthetically derived by proportionally
allocating an aggregate 92.5s batch processing duration across relations according
to token length. They do NOT represent individual per-relation wall-clock latency
measurements. True wall-clock latency instrumentation requires per-stage monotonic
timing (via time.perf_counter in services.pipeline_observability).
"""

import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Connect and Fetch Session Data
db_path = 'data/relation_dataset.db'
conn = sqlite3.connect(db_path)

# Extracting records from recent sessions
query = "SELECT chunk_context FROM relation_dataset WHERE created_at LIKE '2026-04%'"
df = pd.read_sql_query(query, conn)
conn.close()

# Use raw token proxy (approx. 4 characters per token)
df['tokens'] = df['chunk_context'].apply(lambda x: int(len(x)/4) if x else 0)
df = df.sort_values(by='tokens')

# Total session took 92.5 seconds for these processing runs.
# Derived latency: distributing aggregate session time proportionally across generated relations.
token_sum = df['tokens'].sum()
if token_sum > 0:
    df['allocated_time_sec'] = (df['tokens'] / token_sum) * 92.5
else:
    df['allocated_time_sec'] = 0.0

# Generate Graph : Derived Latency vs Token Context
plt.figure(figsize=(10, 6))
plt.scatter(
    df['tokens'],
    df['allocated_time_sec'],
    color='#16a085',
    s=50,
    alpha=0.7,
    edgecolors='k',
    label='Proportionally Allocated Session Latency (Derived Estimate)',
)

# Add trendline to show scaling factor
if len(df) > 1:
    z = np.polyfit(df['tokens'], df['allocated_time_sec'], 1)
    p = np.poly1d(z)
    plt.plot(df['tokens'], p(df['tokens']), color='black', linestyle='--', alpha=0.4, label='Allocation Scaling Trendline')

plt.xlabel('Input Token Context Size (Approx. Chars / 4)', fontsize=12)
plt.ylabel('Estimated Allocated Latency (Seconds)', fontsize=12)
plt.title('Impact of Token Context on Proportionally Allocated Latency (Estimated)', fontsize=14, fontweight='bold')
plt.grid(True, linestyle=':', alpha=0.8)
plt.legend()

plt.tight_layout()
plt.savefig('latency_vs_tokens.png', dpi=300)
print("Graph saved: latency_vs_tokens.png")
