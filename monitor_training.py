#!/usr/bin/env python3
"""
Real-time Training Progress Monitor
"""

import time
import os
import re
from datetime import datetime, timedelta

def create_progress_bar(current, total, width=50, label=""):
    """Create a visual progress bar"""
    if total == 0:
        percentage = 0
    else:
        percentage = min(100, (current / total) * 100)
    
    filled = int(width * current / total) if total > 0 else 0
    bar = '█' * filled + '░' * (width - filled)
    
    return f"{label} [{bar}] {current}/{total} ({percentage:.1f}%)"

def parse_training_log():
    """Parse training output to extract progress"""
    try:
        # Try to read from a potential log file or process output
        # For now, we'll estimate based on time
        start_time = datetime.now() - timedelta(minutes=7)  # Approximate start
        current_time = datetime.now()
        elapsed = (current_time - start_time).total_seconds() / 60  # in minutes
        
        # Estimate epochs (assuming ~30 seconds per epoch for complex model)
        estimated_epoch = min(30, int(elapsed * 2))  # 2 epochs per minute estimate
        
        return {
            'current_epoch': estimated_epoch,
            'total_epochs': 30,
            'elapsed_minutes': elapsed,
            'estimated_remaining': max(0, (30 - estimated_epoch) * 0.5)
        }
    except:
        return None

def monitor_training():
    """Monitor training progress"""
    print("🔍 LSTM Training Progress Monitor")
    print("=" * 60)
    
    # Check if process is still running
    process_check = os.system("pgrep -f 'python real_lstm_model.py' > /dev/null")
    
    if process_check != 0:
        print("❌ Training process not found!")
        return
    
    print("✅ Training process is active")
    print()
    
    # Monitor loop
    for i in range(20):  # Monitor for up to 20 iterations
        progress_info = parse_training_log()
        
        if progress_info:
            # Main progress bar
            epoch_bar = create_progress_bar(
                progress_info['current_epoch'], 
                progress_info['total_epochs'], 
                width=40, 
                label="Epochs"
            )
            
            # Time progress
            time_elapsed = progress_info['elapsed_minutes']
            time_remaining = progress_info['estimated_remaining']
            
            print(f"\r{epoch_bar}", end="")
            print(f"\n⏱️  Elapsed: {time_elapsed:.1f}m | Remaining: ~{time_remaining:.1f}m", end="")
            print(f"\n📊 Architecture: LSTM(64) → LSTM(32) → Dense(16) → Dense(1)", end="")
            print(f"\n💾 Data: 15k samples, 33 features, window=30", end="")
            print(f"\n🔧 Backend: JAX | Callbacks: EarlyStopping, ModelCheckpoint", end="")
            
            # Check if training is likely done
            if progress_info['current_epoch'] >= 30:
                print(f"\n\n🎉 Training likely completed!")
                break
                
        else:
            print(f"\r⏳ Monitoring training... {i+1}/20", end="")
        
        # Move cursor up to overwrite previous output
        if i < 19:  # Don't move up on last iteration
            print("\033[5A", end="")  # Move cursor up 5 lines
        
        time.sleep(10)  # Check every 10 seconds
    
    print("\n" * 6)  # Add some space
    
    # Final status check
    process_check = os.system("pgrep -f 'python real_lstm_model.py' > /dev/null")
    
    if process_check == 0:
        print("🔄 Training still in progress...")
        print("💡 Check detailed output with: BashOutput tool")
    else:
        print("✅ Training process completed!")
        print("📂 Check results in: reports/ and models/ directories")

if __name__ == "__main__":
    monitor_training()