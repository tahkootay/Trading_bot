#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

print("Starting test...")

# Create simple synthetic data
np.random.seed(42)
n_samples = 100
X = np.random.randn(n_samples, 3)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

print(f"Created data: X shape {X.shape}, y shape {y.shape}")

# Train simple model
model = LogisticRegression()
model.fit(X, y)

accuracy = model.score(X, y)
print(f"Model accuracy: {accuracy:.4f}")

print("✅ Simple test completed successfully!")