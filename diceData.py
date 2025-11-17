import pandas as pd

# Load full dataset
df = pd.read_csv("waterQuality1.csv")

# How many rows you want in the sample
TARGET_N = 100  # change to e.g. 500, 200, etc.

n_rows = len(df)
target_n = min(TARGET_N, n_rows)

if "is_safe" in df.columns:
    frac = target_n / n_rows
    sampled = df.groupby("is_safe", group_keys=False).apply(
        lambda g: g.sample(frac=min(1.0, frac), random_state=42)
    )
    sampled = sampled.head(target_n)
else:
    sampled = df.sample(n=target_n, random_state=42)

sampled.to_csv("waterQuality1_sample.csv", index=False)
print("Saved sample to waterQuality1_sample.csv with shape:", sampled.shape)
print(sampled["is_safe"].value_counts())
