#!/usr/bin/env python3
"""
Тест функциональности сбора данных веб-интерфейса
"""

import requests
import json
import time

def test_data_collection_api():
    """Тестирует API сбора данных"""
    base_url = "http://localhost:8080"
    
    print("🔄 Тестирование Функциональности Сбора Данных")
    print("=" * 60)
    
    # Test 1: Basic API access
    print("\n📡 Тестирование базового доступа к API...")
    try:
        response = requests.get(f"{base_url}/api/dashboard", timeout=5)
        if response.status_code == 200:
            print("✅ API доступно")
        else:
            print(f"❌ API недоступно: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Ошибка соединения: {e}")
        return False
    
    # Test 2: Web interface with data collection
    print("\n🌐 Проверка веб-интерфейса с новой функциональностью...")
    try:
        response = requests.get(base_url, timeout=5)
        if response.status_code == 200 and 'Сбор Данных' in response.text:
            print("✅ Веб-интерфейс обновлен с функцией сбора данных")
        else:
            print("❌ Веб-интерфейс не содержит функцию сбора данных")
    except Exception as e:
        print(f"❌ Ошибка доступа к веб-интерфейсу: {e}")
    
    # Test 3: Data collection API endpoints
    print("\n🔍 Проверка эндпоинтов сбора данных...")
    
    # Test collection history endpoint (should work even if empty)
    try:
        response = requests.get(f"{base_url}/api/data-collection/history", timeout=5)
        if response.status_code == 200:
            history = response.json()
            print(f"✅ История сбора данных: {len(history)} записей")
        else:
            print(f"❌ Эндпоинт истории недоступен: {response.status_code}")
    except Exception as e:
        print(f"❌ Ошибка получения истории: {e}")
    
    # Test 4: Mock data collection start
    print("\n▶️ Тестирование запуска сбора данных...")
    
    collection_data = {
        "symbol": "SOLUSDT",
        "start_date": "2025-08-22T10:00",
        "end_date": "2025-08-22T11:00", 
        "timeframes": ["1m", "5m"],
        "dataset_name": "test_collection"
    }
    
    try:
        response = requests.post(
            f"{base_url}/api/data-collection/start",
            json=collection_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            process_id = result.get('process_id')
            print(f"✅ Сбор данных запущен: {process_id}")
            
            # Test status monitoring
            print("\n📊 Мониторинг прогресса сбора данных...")
            
            for i in range(5):  # Check status 5 times
                time.sleep(2)
                try:
                    status_response = requests.get(
                        f"{base_url}/api/data-collection/{process_id}/status",
                        timeout=5
                    )
                    
                    if status_response.status_code == 200:
                        status = status_response.json()
                        progress = status.get('progress', 0)
                        message = status.get('message', 'N/A')
                        print(f"   Прогресс: {progress:.1f}% - {message}")
                        
                        if status.get('status') == 'completed':
                            print("✅ Сбор данных завершен успешно!")
                            
                            # Show results
                            if status.get('results'):
                                results = status['results']
                                print(f"   📈 Символ: {results.get('symbol', 'N/A')}")
                                print(f"   📅 Период: {results.get('period', 'N/A')}")
                                print(f"   📊 Таймфреймы: {', '.join(results.get('timeframes', []))}")
                                print(f"   🕯️ Свечей: {results.get('candles_collected', 0)}")
                                if 'collection_time' in results:
                                    print(f"   ⏱️ Время: {results['collection_time']:.1f}s")
                            break
                        elif status.get('status') == 'failed':
                            print(f"❌ Сбор данных завершился с ошибкой: {status.get('error', 'Unknown')}")
                            break
                    else:
                        print(f"❌ Ошибка получения статуса: {status_response.status_code}")
                        
                except Exception as e:
                    print(f"❌ Ошибка мониторинга: {e}")
        else:
            print(f"❌ Ошибка запуска сбора: {response.status_code}")
            if response.text:
                print(f"   Детали: {response.text}")
                
    except Exception as e:
        print(f"❌ Ошибка запуска сбора данных: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Тестирование функциональности сбора данных завершено!")
    print("\n📋 Для использования:")
    print("1. Откройте http://localhost:8080 в браузере")
    print("2. Перейдите в раздел 'Сбор Данных'")
    print("3. Настройте период и таймфреймы")
    print("4. Нажмите 'Начать Сбор Данных'")
    print("5. Отслеживайте прогресс в реальном времени")
    
    return True

if __name__ == "__main__":
    test_data_collection_api()