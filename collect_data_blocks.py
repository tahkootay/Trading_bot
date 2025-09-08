#!/usr/bin/env python3
"""
Улучшенный сборщик данных по блокам в 1000 свечей.
Собирает данные последовательными блоками без пропусков.
"""

import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import json


def calculate_timeframe_minutes(timeframe: str) -> int:
    """Рассчитывает количество минут для таймфрейма."""
    timeframe_map = {
        '1m': 1, '5m': 5, '15m': 15, '30m': 30,
        '1h': 60, '2h': 120, '4h': 240, '6h': 360, '12h': 720,
        '1d': 1440
    }
    return timeframe_map.get(timeframe, 5)


def collect_data_blocks(symbol: str, timeframe: str, target_candles: int, end_time: datetime = None):
    """
    Собирает данные блоками по 1000 свечей до достижения целевого количества.
    
    Args:
        symbol: Торговая пара (SOLUSDT)
        timeframe: Таймфрейм (5m, 1h, etc.)
        target_candles: Целевое количество свечей
        end_time: Конечное время (по умолчанию - сейчас)
    """
    
    if end_time is None:
        end_time = datetime.now()
    
    print(f"🚀 Начинаем сбор данных блоками для {symbol}")
    print(f"📊 Таймфрейм: {timeframe}")
    print(f"🎯 Целевое количество свечей: {target_candles}")
    print(f"📅 Конечное время: {end_time}")
    
    # Рассчитываем временные параметры
    tf_minutes = calculate_timeframe_minutes(timeframe)
    block_size = 1000  # Максимум свечей за запрос
    
    all_files = []
    collected_candles = 0
    block_num = 1
    current_end = end_time
    
    while collected_candles < target_candles:
        # Оставшееся количество свечей
        remaining_candles = target_candles - collected_candles
        current_block_size = min(block_size, remaining_candles)
        
        # Рассчитываем время начала для текущего блока
        block_minutes = current_block_size * tf_minutes
        block_start = current_end - timedelta(minutes=block_minutes + 5)  # +5 мин для буфера
        
        print(f"\n📦 Блок {block_num}: {block_start.date()} - {current_end.date()}")
        print(f"   Ожидаем: до {current_block_size} свечей ({block_minutes} мин)")
        
        # Формируем команду для сбора данных
        cmd = [
            "python", "-m", "modules.data_collector",
            "--symbol", symbol,
            "--timeframe", timeframe,
            "--start", block_start.strftime("%Y-%m-%d"),
            "--end", current_end.strftime("%Y-%m-%d")
        ]
        
        try:
            # Запускаем команду в venv
            full_cmd = ["bash", "-c", f"source venv/bin/activate && {' '.join(cmd)}"]
            result = subprocess.run(full_cmd, capture_output=True, text=True, cwd=".")
            
            if result.returncode == 0:
                # Ищем созданный файл в выводе
                file_path = None
                output_lines = result.stdout.split('\n')
                
                for line in output_lines:
                    if '.csv' in line and symbol in line and ':' in line:
                        file_path = line.split(':', 1)[1].strip()
                        break
                
                if file_path and Path(file_path).exists():
                    # Проверяем количество свечей в файле
                    try:
                        df = pd.read_csv(file_path)
                        candles_in_block = len(df)
                        collected_candles += candles_in_block
                        all_files.append(file_path)
                        
                        print(f"   ✅ Получено: {candles_in_block} свечей")
                        print(f"   📊 Всего собрано: {collected_candles}/{target_candles}")
                        
                        # Обновляем время для следующего блока на основе реальных данных
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        earliest_time = df['timestamp'].min()
                        current_end = earliest_time - timedelta(minutes=tf_minutes)  # Сдвиг на 1 свечу назад
                        
                        print(f"   🕐 Следующий блок начнется от: {current_end}")
                        
                        # Если собрали достаточно свечей, завершаем
                        if collected_candles >= target_candles:
                            break
                            
                    except Exception as e:
                        print(f"   ❌ Ошибка чтения файла: {e}")
                        break
                else:
                    print(f"   ❌ Файл не найден в выводе")
                    break
            else:
                print(f"   ❌ Ошибка команды: {result.stderr}")
                break
        
        except Exception as e:
            print(f"   ❌ Исключение: {e}")
            break
        
        # Переходим к следующему блоку (current_end уже обновлен в цикле выше)
        block_num += 1
        
        # Защита от бесконечного цикла
        if block_num > 50:  # Максимум 50 блоков = 50k свечей
            print("⚠️ Достигнут лимит блоков (50)")
            break
    
    print(f"\n📊 Сбор данных завершен:")
    print(f"   Блоков собрано: {len(all_files)}")
    print(f"   Свечей собрано: {collected_candles}")
    
    return all_files


def merge_csv_files(files: list, symbol: str, timeframe: str) -> str:
    """Объединяет CSV файлы в один с правильной сортировкой."""
    
    if not files:
        print("❌ Нет файлов для объединения")
        return ""
    
    print(f"\n🔄 Объединяем {len(files)} файлов...")
    
    all_data = []
    total_candles = 0
    
    for file_path in files:
        try:
            df = pd.read_csv(file_path)
            all_data.append(df)
            candles = len(df)
            total_candles += candles
            print(f"   📊 {Path(file_path).name}: {candles} свечей")
        except Exception as e:
            print(f"   ❌ Ошибка чтения {file_path}: {e}")
    
    if not all_data:
        print("❌ Нет данных для объединения")
        return ""
    
    # Объединяем и сортируем по времени
    merged_df = pd.concat(all_data, ignore_index=True)
    merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'])
    merged_df = merged_df.sort_values('timestamp').drop_duplicates(subset=['timestamp'])
    
    # Создаем имя файла
    start_date = merged_df['timestamp'].min().strftime('%Y%m%d')
    end_date = merged_df['timestamp'].max().strftime('%Y%m%d')
    output_file = f"data/raw/{symbol}_{timeframe}_{start_date}_{end_date}_blocks.csv"
    
    # Сохраняем
    merged_df.to_csv(output_file, index=False)
    
    print(f"✅ Объединенные данные сохранены: {output_file}")
    print(f"📊 Итоговое количество свечей: {len(merged_df)}")
    print(f"📅 Период: {merged_df['timestamp'].min()} - {merged_df['timestamp'].max()}")
    
    # Статистика по времени
    time_diff = merged_df['timestamp'].max() - merged_df['timestamp'].min()
    days = time_diff.days
    print(f"⏱️ Период: {days} дней ({days/30:.1f} месяцев)")
    
    return output_file


def main():
    """Основная функция."""
    
    if len(sys.argv) < 4:
        print("Использование: python collect_data_blocks.py SYMBOL TIMEFRAME TARGET_CANDLES")
        print("Пример: python collect_data_blocks.py SOLUSDT 5m 10000")
        print()
        print("Для месячных данных на 5m: ~8640 свечей")
        print("Для 2 месяцев на 5m: ~17280 свечей") 
        sys.exit(1)
    
    symbol = sys.argv[1]
    timeframe = sys.argv[2]
    target_candles = int(sys.argv[3])
    
    print(f"🎯 Цель: {target_candles} свечей")
    
    # Собираем данные блоками
    files = collect_data_blocks(symbol, timeframe, target_candles)
    
    if files:
        # Объединяем в один файл
        merged_file = merge_csv_files(files, symbol, timeframe)
        
        if merged_file:
            print(f"\n🎉 Готово! Данные сохранены в: {merged_file}")
            
            # Показываем статистику
            if Path(merged_file).exists():
                df = pd.read_csv(merged_file)
                tf_minutes = calculate_timeframe_minutes(timeframe)
                theoretical_days = len(df) * tf_minutes / (24 * 60)
                print(f"📈 Теоретический период: {theoretical_days:.1f} дней")
        else:
            print("\n❌ Не удалось объединить файлы")
    else:
        print("\n❌ Не удалось собрать данные")


if __name__ == "__main__":
    main()