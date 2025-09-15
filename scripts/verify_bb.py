#!/usr/bin/env python3
"""Verify Bollinger Bands calculation manually."""

import math

# 20 close prices leading up to 2024-08-01 19:00:00
prices = [150.0, 150.56, 148.66, 147.76, 146.65, 147.6, 148.3, 149.87, 
         148.22, 147.91, 146.03, 144.9, 144.92, 143.03, 141.82, 142.42, 
         142.6, 141.49, 141.84, 143.08]

print("Manual Bollinger Bands calculation:")
print(f"Prices: {prices}")
print(f"Count: {len(prices)}")

# Calculate SMA (middle band)
sma = sum(prices) / len(prices)
print(f"SMA (Middle Band): {sma}")

# Calculate population standard deviation (divide by n)
pop_variance = sum((price - sma) ** 2 for price in prices) / len(prices)
pop_std_dev = math.sqrt(pop_variance)
print(f"Population Standard Deviation: {pop_std_dev}")

# Calculate sample standard deviation (divide by n-1)
sample_variance = sum((price - sma) ** 2 for price in prices) / (len(prices) - 1)
sample_std_dev = math.sqrt(sample_variance)
print(f"Sample Standard Deviation: {sample_std_dev}")

# Calculate bands with population std dev
pop_upper = sma + (2 * pop_std_dev)
pop_lower = sma - (2 * pop_std_dev)

# Calculate bands with sample std dev
sample_upper = sma + (2 * sample_std_dev)
sample_lower = sma - (2 * sample_std_dev)

print(f"\nPopulation Std Dev Bands:")
print(f"Upper: {pop_upper}")
print(f"Lower: {pop_lower}")

print(f"\nSample Std Dev Bands:")
print(f"Upper: {sample_upper}")
print(f"Lower: {sample_lower}")

print("\nExpected values according to user:")
print("bb_upper: 171.740")
print("bb_middle: 168.477")
print("bb_lower: 165.213")

print("\nNote: User's values don't follow BB logic (lower > middle), suggesting different calculation or data source.")