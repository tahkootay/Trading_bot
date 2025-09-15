#!/usr/bin/env python3
"""
Скрипт для сбора месячных данных по частям.
Собирает данные неделя за неделей, чтобы избежать лимитов.
"""

import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd


def collect_weekly_data(symbol: str, timeframe: str, weeks_back: int = 4):
    """Собирает данные по неделям за указанное количество недель назад."""
    
    print(f"🚀 Начинаем сбор месячных данных для {symbol}")
    print(f"📊 Таймфрейм: {timeframe}")
    print(f"📅 Недель назад: {weeks_back}")
    
    all_files = []
    end_date = datetime.now()
    
    for week in range(weeks_back):
        week_end = end_date - timedelta(weeks=week)
        week_start = week_end - timedelta(weeks=1)
        
        print(f"\n📈 Неделя {week+1}/{weeks_back}: {week_start.date()} - {week_end.date()}")
        
        # Формируем команду для сбора данных
        cmd = [
            "python", "-m", "modules.data_collector",
            "--symbol", symbol,
            "--timeframe", timeframe,
            "--start", week_start.strftime("%Y-%m-%d"),
            "--end", week_end.strftime("%Y-%m-%d")
        ]
        
        try:
            # Активируем virtual environment и запускаем команду
            full_cmd = ["bash", "-c", f"source venv/bin/activate && {' '.join(cmd)}"]
            result = subprocess.run(full_cmd, capture_output=True, text=True, cwd=".")
            
            if result.returncode == 0:
                print("   ✅ Данные собраны успешно")
                # Ищем созданный файл в выводе
                output_lines = result.stdout.split('\n')
                for line in output_lines:
                    if 'csv' in line and symbol in line:
                        # Извлекаем путь к файлу
                        if ':' in line:
                            file_path = line.split(':', 1)[1].strip()
                            if Path(file_path).exists():
                                all_files.append(file_path)
                                break
            else:
                print(f"   ❌ Ошибка: {result.stderr}")
                
        except Exception as e:
            print(f"   ❌ Исключение: {e}")
    
    return all_files


def merge_csv_files(files: list, symbol: str, timeframe: str) -> str:
    """Объединяет CSV файлы в один."""
    
    if not files:
        print("❌ Нет файлов для объединения")
        return ""
    
    print(f"\n🔄 Объединяем {len(files)} файлов...")
    
    all_data = []
    for file_path in files:
        try:
            df = pd.read_csv(file_path)
            all_data.append(df)
            print(f"   📊 {file_path}: {len(df)} свечей")
        except Exception as e:
            print(f"   ❌ Ошибка чтения {file_path}: {e}")
    
    if not all_data:
        print("❌ Нет данных для объединения")
        return ""
    
    # Объединяем и сортируем
    merged_df = pd.concat(all_data, ignore_index=True)
    merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'])
    merged_df = merged_df.sort_values('timestamp').drop_duplicates(subset=['timestamp'])
    
    # Создаем имя файла
    start_date = merged_df['timestamp'].min().strftime('%Y%m%d')
    end_date = merged_df['timestamp'].max().strftime('%Y%m%d')
    output_file = f"data/raw/{symbol}_{timeframe}_{start_date}_{end_date}_monthly.csv"
    
    # Сохраняем
    merged_df.to_csv(output_file, index=False)
    
    print(f"✅ Объединенные данные сохранены: {output_file}")
    print(f"📊 Общее количество свечей: {len(merged_df)}")
    print(f"📅 Период: {merged_df['timestamp'].min()} - {merged_df['timestamp'].max()}")
    
    return output_file


def main():
    """Основная функция."""
    
    if len(sys.argv) < 3:
        print("Использование: python collect_monthly_data.py SYMBOL TIMEFRAME [WEEKS]")
        print("Пример: python collect_monthly_data.py SOLUSDT 5m 4")
        sys.exit(1)
    
    symbol = sys.argv[1]
    timeframe = sys.argv[2]
    weeks_back = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    
    # Собираем данные по неделям
    files = collect_weekly_data(symbol, timeframe, weeks_back)
    
    if files:
        # Объединяем в один файл
        merged_file = merge_csv_files(files, symbol, timeframe)
        
        if merged_file:
            print(f"\n🎉 Готово! Месячные данные сохранены в: {merged_file}")
        else:
            print("\n❌ Не удалось объединить файлы")
    else:
        print("\n❌ Не удалось собрать данные")


if __name__ == "__main__":
    main()