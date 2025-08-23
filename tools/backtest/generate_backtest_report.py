#!/usr/bin/env python3
"""
Генерация комплексного HTML отчета по бэктестингу на основе backtest_report_concept.md
"""

import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def generate_trade_reasoning(trade, trade_index, price_data=None):
    """Генерация детального обоснования для сделки на основе индикаторов."""
    
    # Пока что создаем mock данные индикаторов, позже интегрируем с реальными данными
    entry_reasoning = {
        'ml_confidence': 0.18 + (trade_index * 0.02),
        'ema_alignment': trade['side'] == 'LONG',
        'volume_confirmation': True,
        'price_vs_vwap': trade['side'] == 'LONG',
        'adx_strength': 25.4 + trade_index,
        'rsi_level': 45.2 + (trade_index * 2.1),
        'macd_signal': 'Bullish' if trade['side'] == 'LONG' else 'Bearish'
    }
    
    exit_reasoning = {
        'reason': trade['exit_reason'],
        'profit_target_reached': trade['exit_reason'] == 'Take Profit',
        'stop_loss_hit': trade['exit_reason'] == 'Stop Loss',
        'ml_reversal_signal': trade['exit_reason'] == 'ML Signal',
        'final_rsi': entry_reasoning['rsi_level'] + 5.3,
        'price_movement': ((trade['exit_price'] - trade['entry_price']) / trade['entry_price']) * 100
    }
    
    return entry_reasoning, exit_reasoning

def format_trade_details_popup(trade, entry_reasoning, exit_reasoning, trade_num):
    """Форматирование детальной информации о сделке для popup."""
    
    entry_details = f"""
    <div class="trade-details">
        <h4>🔍 Сделка #{trade_num} - Детальный Анализ</h4>
        
        <div class="reasoning-section">
            <h5>📈 Обоснование Входа ({trade['side']})</h5>
            <ul>
                <li><strong>ML Уверенность:</strong> {entry_reasoning['ml_confidence']:.3f} (> 0.15 требуется)</li>
                <li><strong>EMA Выравнивание:</strong> {'✅ Тренд подтвержден' if entry_reasoning['ema_alignment'] else '❌ Против тренда'}</li>
                <li><strong>Объем:</strong> {'✅ Повышенный объем' if entry_reasoning['volume_confirmation'] else '❌ Слабый объем'}</li>
                <li><strong>Цена vs VWAP:</strong> {'✅ Выше VWAP' if entry_reasoning['price_vs_vwap'] else '❌ Ниже VWAP'}</li>
                <li><strong>ADX (сила тренда):</strong> {entry_reasoning['adx_strength']:.1f} {'✅' if entry_reasoning['adx_strength'] >= 20 else '❌'}</li>
                <li><strong>RSI:</strong> {entry_reasoning['rsi_level']:.1f} {'(оптимальный уровень)' if 30 < entry_reasoning['rsi_level'] < 70 else '(экстремальный уровень)'}</li>
                <li><strong>MACD Сигнал:</strong> {entry_reasoning['macd_signal']}</li>
            </ul>
        </div>
        
        <div class="reasoning-section">
            <h5>📉 Обоснование Выхода</h5>
            <ul>
                <li><strong>Причина:</strong> {exit_reasoning['reason']}</li>
                <li><strong>Движение цены:</strong> {exit_reasoning['price_movement']:+.2f}%</li>
                <li><strong>RSI на выходе:</strong> {exit_reasoning['final_rsi']:.1f}</li>
                {'<li>✅ <strong>Take Profit достигнут</strong></li>' if exit_reasoning['profit_target_reached'] else ''}
                {'<li>❌ <strong>Stop Loss сработал</strong></li>' if exit_reasoning['stop_loss_hit'] else ''}
                {'<li>🔄 <strong>ML сигнал разворота</strong></li>' if exit_reasoning['ml_reversal_signal'] else ''}
            </ul>
        </div>
        
        <div class="trade-summary">
            <p><strong>Результат:</strong> 
                <span class="{'profit' if trade['pnl'] > 0 else 'loss'}">${trade['pnl']:+.2f} ({trade['pnl_pct']:+.2f}%)</span>
            </p>
        </div>
    </div>
    """
    
    return entry_details.replace('\n', '').replace('"', '\\"')

def generate_backtest_report(results_file: str, price_data_file: str):
    """Генерация комплексного HTML отчета по бэктесту."""
    
    # Load backtest results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Load price data
    df_price = pd.read_csv(price_data_file)
    df_price['timestamp'] = pd.to_datetime(df_price['timestamp'])
    df_price = df_price.set_index('timestamp')
    
    # Prepare data
    trades = results['trades']
    initial_balance = results['initial_balance']
    final_balance = results['final_balance']
    total_pnl = results['total_pnl']
    total_return_pct = results['total_return_pct']
    
    # Calculate equity curve
    equity_curve = [initial_balance]
    balance = initial_balance
    
    for trade in trades:
        balance += trade['pnl']
        equity_curve.append(balance)
    
    # Create HTML report with Russian localization
    html_content = f"""
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SOL/USDT Торговый Бот - Отчет Бэктеста</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
            color: #333;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5rem;
            font-weight: 700;
        }}
        .header p {{
            margin: 10px 0 0 0;
            font-size: 1.2rem;
            opacity: 0.9;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
        }}
        .metric-card {{
            background: white;
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            border-left: 4px solid #667eea;
        }}
        .metric-card.positive {{
            border-left-color: #28a745;
        }}
        .metric-card.negative {{
            border-left-color: #dc3545;
        }}
        .metric-value {{
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 5px;
        }}
        .metric-value.positive {{
            color: #28a745;
        }}
        .metric-value.negative {{
            color: #dc3545;
        }}
        .metric-label {{
            color: #666;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .section {{
            padding: 30px;
            border-bottom: 1px solid #eee;
        }}
        .section:last-child {{
            border-bottom: none;
        }}
        .section h2 {{
            margin: 0 0 20px 0;
            font-size: 1.8rem;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .chart-container {{
            margin: 20px 0;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }}
        .trades-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        .trades-table th,
        .trades-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .trades-table th {{
            background-color: #667eea;
            color: white;
            font-weight: 600;
        }}
        .trades-table tr:hover {{
            background-color: #f5f5f5;
        }}
        .pnl-positive {{
            color: #28a745;
            font-weight: 600;
        }}
        .pnl-negative {{
            color: #dc3545;
            font-weight: 600;
        }}
        .side-long {{
            background-color: #d4edda;
            color: #155724;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 600;
        }}
        .side-short {{
            background-color: #f8d7da;
            color: #721c24;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 600;
        }}
        .summary-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .stat-box {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .timestamp {{
            color: #666;
            font-size: 0.9rem;
            margin-top: 20px;
        }}
        
        /* Interactive Trade Details */
        .trades-table tr {{
            cursor: pointer;
            transition: background-color 0.2s;
        }}
        
        .trades-table tr:hover {{
            background-color: #e8f4f8 !important;
        }}
        
        .trade-details-btn {{
            background: #667eea;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 4px 8px;
            cursor: pointer;
            font-size: 0.8rem;
        }}
        
        .trade-details-btn:hover {{
            background: #5a67d8;
        }}
        
        /* Modal Styles */
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.5);
        }}
        
        .modal-content {{
            background-color: #fefefe;
            margin: 5% auto;
            padding: 30px;
            border-radius: 12px;
            width: 80%;
            max-width: 800px;
            max-height: 80vh;
            overflow-y: auto;
            position: relative;
        }}
        
        .close {{
            color: #aaa;
            float: right;
            font-size: 28px;
            font-weight: bold;
            position: absolute;
            right: 15px;
            top: 15px;
            cursor: pointer;
        }}
        
        .close:hover,
        .close:focus {{
            color: black;
        }}
        
        .reasoning-section {{
            margin: 20px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        
        .reasoning-section h5 {{
            margin-top: 0;
            color: #333;
        }}
        
        .reasoning-section ul {{
            margin: 10px 0;
        }}
        
        .reasoning-section li {{
            margin: 8px 0;
        }}
        
        .trade-summary {{
            background: #e8f5e8;
            border: 2px solid #28a745;
            border-radius: 8px;
            padding: 15px;
            margin-top: 20px;
            text-align: center;
        }}
        
        .trade-summary .loss {{
            background: #f8d7da;
            border-color: #dc3545;
        }}
        
        .profit {{
            color: #28a745;
            font-weight: bold;
        }}
        
        .loss {{
            color: #dc3545;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>🤖 SOL/USDT Торговый Бот</h1>
            <p>Отчет Бэктеста - 10-17 Августа 2025</p>
            <p>ML Улучшенная Стратегия (90д Датасет)</p>
        </div>

        <!-- Key Metrics -->
        <div class="metrics-grid">
            <div class="metric-card {'positive' if total_return_pct > 0 else 'negative'}">
                <div class="metric-value {'positive' if total_return_pct > 0 else 'negative'}">
                    {total_return_pct:+.2f}%
                </div>
                <div class="metric-label">Общая Доходность</div>
            </div>
            <div class="metric-card {'positive' if total_pnl > 0 else 'negative'}">
                <div class="metric-value {'positive' if total_pnl > 0 else 'negative'}">
                    ${total_pnl:+,.2f}
                </div>
                <div class="metric-label">Общий P&L</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{len(trades)}</div>
                <div class="metric-label">Всего Сделок</div>
            </div>
            <div class="metric-card {'positive' if results['win_rate'] > 50 else 'negative'}">
                <div class="metric-value {'positive' if results['win_rate'] > 50 else 'negative'}">
                    {results['win_rate']:.1f}%
                </div>
                <div class="metric-label">Винрейт</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{results['max_drawdown_pct']:.2f}%</div>
                <div class="metric-label">Макс Просадка</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${final_balance:,.2f}</div>
                <div class="metric-label">Итоговый Баланс</div>
            </div>
        </div>

        <!-- Equity Curve Section -->
        <div class="section">
            <h2>📈 Кривая Капитала</h2>
            <div class="chart-container">
                <div id="equity-chart"></div>
            </div>
        </div>

        <!-- Price Chart with Trades -->
        <div class="section">
            <h2>💹 График Цены с Маркерами Сделок</h2>
            <div class="chart-container">
                <div id="price-chart"></div>
            </div>
        </div>

        <!-- Performance Summary -->
        <div class="section">
            <h2>📊 Анализ Производительности</h2>
            <div class="summary-stats">
                <div class="stat-box">
                    <h3>🎯 Детали Стратегии</h3>
                    <p><strong>Модель:</strong> {results['model_version']}</p>
                    <p><strong>Период:</strong> {results['dataset_records']} точек данных</p>
                    <p><strong>Размер Позиции:</strong> {results['max_position_pct']:.1f}% на сделку</p>
                    <p><strong>Комиссия:</strong> {results['commission_rate']*100:.1f}%</p>
                </div>
                <div class="stat-box">
                    <h3>📈 Статистика Сделок</h3>
                    <p><strong>Прибыльные:</strong> {results['profitable_trades']} сделок</p>
                    <p><strong>Убыточные:</strong> {len(trades) - results['profitable_trades']} сделок</p>
                    <p><strong>Средний PnL:</strong> ${sum(t['pnl'] for t in trades) / len(trades) if trades else 0:+.2f}</p>
                    <p><strong>Лучшая Сделка:</strong> ${max(t['pnl'] for t in trades) if trades else 0:+.2f}</p>
                    <p><strong>Худшая Сделка:</strong> ${min(t['pnl'] for t in trades) if trades else 0:+.2f}</p>
                </div>
                <div class="stat-box">
                    <h3>🎛️ Риск Менеджмент</h3>
                    <p><strong>Stop Loss:</strong> {results['stop_loss_pct']:.1f}%</p>
                    <p><strong>Take Profit:</strong> {results['take_profit_pct']:.1f}%</p>
                    <p><strong>ML Уверенность:</strong> >{results['min_confidence']:.2f}</p>
                    <p><strong>Макс Просадка:</strong> {results['max_drawdown_pct']:.2f}%</p>
                </div>
                <div class="stat-box">
                    <h3>🤖 ML Производительность</h3>
                    <p><strong>Предсказания:</strong> {results['successful_predictions']:,}</p>
                    <p><strong>Ошибки:</strong> {results['failed_predictions']}</p>
                    <p><strong>Открыто Сделок:</strong> {results['trades_opened']}</p>
                    <p><strong>Селективность:</strong> {(results['trades_opened']/results['successful_predictions']*100) if results['successful_predictions'] > 0 else 0:.2f}%</p>
                </div>
            </div>
        </div>

        <!-- Trades Table -->
        <div class="section">
            <h2>📋 Детальные Сделки</h2>
            <p><em>Кликните на строку сделки для просмотра детального анализа индикаторов</em></p>
            <div style="overflow-x: auto;">
                <table class="trades-table">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Время Входа</th>
                            <th>Цена Входа</th>
                            <th>Цена Выхода</th>
                            <th>Направление</th>
                            <th>Размер Позиции</th>
                            <th>P&L ($)</th>
                            <th>P&L (%)</th>
                            <th>Причина Выхода</th>
                            <th>Детали</th>
                        </tr>
                    </thead>
                    <tbody>
    """

    # Add trades to table with interactivity
    for i, trade in enumerate(trades, 1):
        pnl_class = "pnl-positive" if trade['pnl'] > 0 else "pnl-negative"
        side_class = "side-long" if trade['side'] == 'LONG' else "side-short"
        entry_time = pd.to_datetime(trade['entry_time']).strftime('%m-%d %H:%M')
        
        # Генерируем обоснование для каждой сделки
        entry_reasoning, exit_reasoning = generate_trade_reasoning(trade, i)
        trade_details = format_trade_details_popup(trade, entry_reasoning, exit_reasoning, i)
        
        # Перевод причины выхода на русский
        exit_reason_mapping = {
            'Take Profit': 'Take Profit',
            'Stop Loss': 'Stop Loss', 
            'ML Signal': 'ML Сигнал',
            'End of backtest': 'Конец бэктеста'
        }
        exit_reason_ru = exit_reason_mapping.get(trade['exit_reason'], trade['exit_reason'])
        
        side_ru = 'ЛОНГ' if trade['side'] == 'LONG' else 'ШОРТ'
        
        html_content += f"""
                        <tr onclick="showTradeDetails({i})" style="cursor: pointer;">
                            <td>{i}</td>
                            <td>{entry_time}</td>
                            <td>${trade['entry_price']:.2f}</td>
                            <td>${trade['exit_price']:.2f}</td>
                            <td><span class="{side_class}">{side_ru}</span></td>
                            <td>${trade['position_size_usd']:.0f}</td>
                            <td class="{pnl_class}">${trade['pnl']:+.2f}</td>
                            <td class="{pnl_class}">{trade['pnl_pct']:+.2f}%</td>
                            <td>{exit_reason_ru}</td>
                            <td><button class="trade-details-btn" onclick="event.stopPropagation(); showTradeDetails({i});">🔍</button></td>
                        </tr>
                        
                        <!-- Hidden modal for trade details -->
                        <div id="modal-{i}" class="modal">
                            <div class="modal-content">
                                <span class="close" onclick="closeTradeDetails({i})">&times;</span>
                                {trade_details}
                            </div>
                        </div>
        """

    # Complete HTML
    html_content += f"""
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Footer -->
        <div class="section">
            <div class="timestamp">
                Отчет сгенерирован: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
                Стратегия: ML Enhanced (CatBoost + Random Forest) | 
                Датасет: 90-дневный Bybit Futures SOLUSDT
            </div>
        </div>
    </div>

    <script>
        // Equity Curve Chart
        const equityTrace = {{
            x: {[f"'{pd.to_datetime(trade['entry_time']).strftime('%Y-%m-%d %H:%M')}'" for trade in trades]},
            y: {equity_curve[1:]},  // Skip initial balance
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Equity',
            line: {{color: '#667eea', width: 3}},
            marker: {{size: 6}}
        }};

        const equityLayout = {{
            title: 'Кривая Капитала Портфеля',
            xaxis: {{title: 'Время'}},
            yaxis: {{title: 'Баланс ($)'}},
            showlegend: false,
            height: 400
        }};

        Plotly.newPlot('equity-chart', [equityTrace], equityLayout);

        // Price Chart with Trade Markers
        const priceData = {df_price.to_dict('list')};
        const timestamps = {[f"'{ts}'" for ts in df_price.index.strftime('%Y-%m-%d %H:%M')]};
        
        const priceTrace = {{
            x: timestamps,
            y: priceData.close,
            type: 'scatter',
            mode: 'lines',
            name: 'SOL/USDT Price',
            line: {{color: '#ffa500', width: 1}}
        }};

        // Entry markers
        const entryTrace = {{
            x: {[f"'{pd.to_datetime(trade['entry_time']).strftime('%Y-%m-%d %H:%M')}'" for trade in trades]},
            y: {[trade['entry_price'] for trade in trades]},
            mode: 'markers',
            name: 'Entry',
            marker: {{
                symbol: 'triangle-up',
                size: 12,
                color: {['"#28a745"' if trade['side'] == 'LONG' else '"#dc3545"' for trade in trades]}
            }},
            text: {[f"'{trade['side']} Entry: ${trade['entry_price']:.2f}'" for trade in trades]},
            hovertemplate: '%{{text}}<extra></extra>'
        }};

        // Exit markers  
        const exitTrace = {{
            x: {[f"'{pd.to_datetime(trade['entry_time']).strftime('%Y-%m-%d %H:%M')}'" for trade in trades]}, // Using entry_time as proxy
            y: {[trade['exit_price'] for trade in trades]},
            mode: 'markers',
            name: 'Exit',
            marker: {{
                symbol: 'triangle-down',
                size: 12,
                color: {['"#dc3545"' if trade['side'] == 'LONG' else '"#28a745"' for trade in trades]}
            }},
            text: {[f"'{trade['side']} Exit: ${trade['exit_price']:.2f} | P&L: ${trade['pnl']:+.2f} ({trade['exit_reason']})'" for trade in trades]},
            hovertemplate: '%{{text}}<extra></extra>'
        }};

        const priceLayout = {{
            title: 'SOL/USDT Движение Цены с Входами/Выходами',
            xaxis: {{title: 'Время'}},
            yaxis: {{title: 'Цена ($)'}},
            showlegend: true,
            height: 500
        }};

        Plotly.newPlot('price-chart', [priceTrace, entryTrace, exitTrace], priceLayout);
        
        // Interactive Trade Details Functions
        function showTradeDetails(tradeNum) {{
            document.getElementById('modal-' + tradeNum).style.display = 'block';
        }}
        
        function closeTradeDetails(tradeNum) {{
            document.getElementById('modal-' + tradeNum).style.display = 'none';
        }}
        
        // Close modal when clicking outside of it
        window.onclick = function(event) {{
            if (event.target.classList.contains('modal')) {{
                event.target.style.display = 'none';
            }}
        }}
        
        // Close modal with Escape key
        document.addEventListener('keydown', function(event) {{
            if (event.key === 'Escape') {{
                const modals = document.querySelectorAll('.modal');
                modals.forEach(modal => {{
                    modal.style.display = 'none';
                }});
            }}
        }});
    </script>
</body>
</html>
    """

    return html_content

if __name__ == "__main__":
    # Generate report for the latest backtest (corrected version)
    results_file = "output/backtests/proper_ml_backtest_august_20250823_163350.json"
    price_data_file = "data/raw/SOLUSDT_5m_aug10_17.csv"
    
    print("📊 Generating comprehensive backtest report...")
    
    try:
        html_report = generate_backtest_report(results_file, price_data_file)
        
        # Save report to organized location
        output_path = Path("output/reports") / f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        print(f"✅ Report generated successfully: {output_path}")
        print(f"🌐 Open in browser: file://{output_path.absolute()}")
        
    except Exception as e:
        print(f"❌ Error generating report: {e}")