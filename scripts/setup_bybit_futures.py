#!/usr/bin/env python3
"""
Настройка и тест подключения к Bybit Futures API
"""

import asyncio
import sys
from pathlib import Path
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Add src to path
src_path = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, src_path)

from src.data_collector.bybit_client import BybitHTTPClient
from src.utils.types import TimeFrame

async def test_bybit_futures():
    print("🚀 Тест подключения к Bybit Futures API")
    print("=" * 50)
    
    # Load environment
    load_dotenv()
    
    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")
    
    if not api_key or not api_secret:
        print("❌ API ключи не найдены в .env файле")
        print("Установите BYBIT_API_KEY и BYBIT_API_SECRET")
        return False
    
    print(f"🔑 API Key: {api_key[:8]}...")
    
    # Test both testnet and mainnet
    for testnet in [True, False]:
        env_name = "TESTNET" if testnet else "MAINNET"
        print(f"\n🌐 Тестирование {env_name}:")
        print("-" * 30)
        
        try:
            client = BybitHTTPClient(
                api_key=api_key,
                api_secret=api_secret,
                testnet=testnet,
                rate_limit=1  # Медленнее для тестов
            )
            
            async with client:
                # 1. Test public data (не требует авторизации)
                print("📊 Тест публичных данных:")
                
                try:
                    ticker = await client.get_ticker("SOLUSDT")
                    if ticker:
                        print(f"  ✅ Ticker: ${ticker.price:.2f} (bid: ${ticker.bid:.2f}, ask: ${ticker.ask:.2f})")
                    else:
                        print("  ❌ Ticker не получен")
                except Exception as e:
                    print(f"  ❌ Ticker ошибка: {e}")
                
                # 2. Test klines (исторические данные)
                print("\n📈 Тест исторических данных:")
                
                try:
                    # Попробуем получить последние 10 часовых свечей
                    candles = await client.get_klines(
                        symbol="SOLUSDT",
                        interval=TimeFrame.H1,
                        limit=10
                    )
                    
                    if candles:
                        print(f"  ✅ Получено {len(candles)} свечей")
                        latest = candles[0]
                        print(f"  📅 Последняя свеча: {latest.timestamp}")
                        print(f"  💰 OHLC: O=${latest.open:.2f} H=${latest.high:.2f} L=${latest.low:.2f} C=${latest.close:.2f}")
                        print(f"  📊 Volume: {latest.volume:,.0f}")
                    else:
                        print("  ❌ Данные не получены")
                        
                except Exception as e:
                    print(f"  ❌ Klines ошибка: {e}")
                
                # 3. Test с временными параметрами
                print("\n⏰ Тест с временными параметрами:")
                
                try:
                    end_time = datetime.now()
                    start_time = end_time - timedelta(hours=24)  # Последние 24 часа
                    
                    candles_time = await client.get_klines(
                        symbol="SOLUSDT",
                        interval=TimeFrame.H1,
                        limit=24,
                        start_time=int(start_time.timestamp() * 1000),
                        end_time=int(end_time.timestamp() * 1000)
                    )
                    
                    if candles_time:
                        print(f"  ✅ Получено {len(candles_time)} свечей за 24ч")
                        period_start = candles_time[-1].timestamp
                        period_end = candles_time[0].timestamp
                        print(f"  📅 Период: {period_start} → {period_end}")
                    else:
                        print("  ❌ Данные с временными параметрами не получены")
                        
                except Exception as e:
                    print(f"  ❌ Временные параметры ошибка: {e}")
                
                # 4. Test account data (требует авторизации)
                print("\n🔐 Тест авторизованных данных:")
                
                try:
                    account = await client.get_account_info()
                    if account:
                        print(f"  ✅ Account: Balance=${account.available_balance:.2f}")
                        print(f"  📊 Total: ${account.total_balance:.2f}")
                        return True  # Успех!
                    else:
                        print("  ❌ Account info недоступна")
                except Exception as e:
                    print(f"  ❌ Account ошибка: {e}")
                    if "expired" in str(e).lower() or "33004" in str(e):
                        print("  🔄 API ключ истек - нужно обновить")
                    elif "permissions" in str(e).lower():
                        print("  🔑 Недостаточно прав - проверьте настройки API")
        
        except Exception as e:
            print(f"❌ Общая ошибка {env_name}: {e}")
    
    return False

def create_env_template():
    """Создать шаблон .env файла"""
    template = """# Bybit API Configuration
BYBIT_API_KEY=your_api_key_here
BYBIT_API_SECRET=your_api_secret_here

# Trading Configuration  
SYMBOL=SOLUSDT
INITIAL_BALANCE=10000
COMMISSION_RATE=0.001

# Telegram Notifications (optional)
TELEGRAM_BOT_TOKEN=your_telegram_token
TELEGRAM_CHAT_ID=your_chat_id
"""
    
    env_path = Path(".env.example")
    if not env_path.exists():
        with open(env_path, "w") as f:
            f.write(template)
        print(f"📝 Создан шаблон конфигурации: {env_path}")
        print("Скопируйте его в .env и заполните своими данными")

async def main():
    success = await test_bybit_futures()
    
    if success:
        print(f"\n🎉 Подключение к Bybit Futures успешно настроено!")
        print(f"✅ Можно запускать сбор данных и торговлю")
    else:
        print(f"\n⚠️  Проблемы с подключением к Bybit")
        print(f"🔧 Возможные решения:")
        print(f"  1. Обновите API ключи в .env файле")
        print(f"  2. Проверьте права API ключей (Futures Trading, Read)")
        print(f"  3. Убедитесь что IP добавлен в whitelist")
        
        create_env_template()
    
    return success

if __name__ == "__main__":
    asyncio.run(main())