#!/usr/bin/env python3
"""Test different BB parameters to match expected values."""

import math

# Our synthetic data prices
prices = [150.0, 150.56, 148.66, 147.76, 146.65, 147.6, 148.3, 149.87, 
         148.22, 147.91, 146.03, 144.9, 144.92, 143.03, 141.82, 142.42, 
         142.6, 141.49, 141.84, 143.08]

target_middle = 168.477
target_upper = 171.740
target_lower = 165.213

print(f"Target values: Upper={target_upper}, Middle={target_middle}, Lower={target_lower}")

# Calculate what SMA would be needed
our_sma = sum(prices) / len(prices)
print(f"Our SMA: {our_sma}")
print(f"Expected middle: {target_middle}")
print(f"Difference: {target_middle - our_sma}")

# If the middle (SMA) is different, it means different price data
print(f"\nTo get middle band = {target_middle}, we need different prices or period")

# Calculate what prices would give us the target SMA
needed_sum = target_middle * 20
our_sum = sum(prices)
print(f"Our sum: {our_sum}")
print(f"Needed sum for target SMA: {needed_sum}")

# Maybe it's a different period?
for period in [10, 15, 20, 25, 30]:
    if period <= len(prices):
        window = prices[-period:]
        sma = sum(window) / len(window)
        print(f"Period {period}: SMA = {sma}")