#!/usr/bin/env python3
"""
Web API для управления торговым ботом
Предоставляет REST API для веб-интерфейса
"""

import os
import sys
import json
import asyncio
import subprocess
import threading
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import logging

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Глобальные переменные для отслеживания процессов
running_processes: Dict[str, Dict] = {}
process_lock = threading.Lock()

@dataclass
class ProcessStatus:
    id: str
    type: str  # 'training' or 'backtest'
    status: str  # 'running', 'completed', 'failed'
    progress: float
    message: str
    start_time: datetime
    end_time: Optional[datetime] = None
    results: Optional[Dict] = None
    error: Optional[str] = None

class ProcessManager:
    """Менеджер для отслеживания запущенных процессов"""
    
    def __init__(self):
        self.processes: Dict[str, ProcessStatus] = {}
        self.lock = threading.Lock()
    
    def create_process(self, process_type: str) -> str:
        """Создать новый процесс"""
        process_id = str(uuid.uuid4())
        
        with self.lock:
            self.processes[process_id] = ProcessStatus(
                id=process_id,
                type=process_type,
                status='running',
                progress=0.0,
                message='Инициализация...',
                start_time=datetime.now()
            )
        
        return process_id
    
    def update_process(self, process_id: str, **kwargs):
        """Обновить статус процесса"""
        with self.lock:
            if process_id in self.processes:
                process = self.processes[process_id]
                for key, value in kwargs.items():
                    if hasattr(process, key):
                        setattr(process, key, value)
    
    def get_process(self, process_id: str) -> Optional[ProcessStatus]:
        """Получить информацию о процессе"""
        with self.lock:
            return self.processes.get(process_id)
    
    def complete_process(self, process_id: str, results: Dict = None, error: str = None):
        """Завершить процесс"""
        with self.lock:
            if process_id in self.processes:
                process = self.processes[process_id]
                process.status = 'completed' if error is None else 'failed'
                process.progress = 100.0
                process.end_time = datetime.now()
                process.results = results
                process.error = error
                
                if error is None:
                    process.message = 'Завершено успешно'
                else:
                    process.message = f'Ошибка: {error}'

# Глобальный менеджер процессов
process_manager = ProcessManager()

# Статические файлы веб-интерфейса
@app.route('/')
def serve_index():
    """Главная страница веб-интерфейса"""
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Статические файлы"""
    return send_from_directory('.', filename)

# API эндпоинты
@app.route('/api/dashboard')
def get_dashboard_data():
    """Получить данные для дашборда"""
    try:
        # Получаем информацию о моделях
        models_dir = Path('../models')
        active_models = 0
        latest_accuracy = 0.0
        last_training = 'Никогда'
        
        if models_dir.exists():
            model_dirs = [d for d in models_dir.iterdir() if d.is_dir() and d.name != 'latest']
            active_models = len(model_dirs)
            
            # Ищем последнюю модель
            if model_dirs:
                latest_model = sorted(model_dirs)[-1]
                try:
                    metadata_file = latest_model / 'metadata.json'
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            if 'model_scores' in metadata:
                                scores = list(metadata['model_scores'].values())
                                latest_accuracy = sum(scores) / len(scores) * 100 if scores else 0
                            if 'training_date' in metadata:
                                training_date = datetime.fromisoformat(metadata['training_date'].replace('Z', '+00:00'))
                                last_training = training_date.strftime('%d.%m.%Y %H:%M')
                except Exception as e:
                    logger.error(f"Error reading model metadata: {e}")
        
        # Получаем последний P&L из результатов
        last_pnl = 0.0
        output_dir = Path('../output')
        if output_dir.exists():
            json_files = list(output_dir.glob('*strategy*.json'))
            if json_files:
                latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
                try:
                    with open(latest_file, 'r') as f:
                        data = json.load(f)
                        if 'final_balance' in data and 'initial_balance' in data:
                            last_pnl = data['final_balance'] - data['initial_balance']
                except Exception as e:
                    logger.error(f"Error reading backtest results: {e}")
        
        return jsonify({
            'activeModels': active_models,
            'mlAccuracy': latest_accuracy,
            'lastPnl': last_pnl,
            'lastTraining': last_training,
            'systemStatus': 'online',
            'dataPoints': get_available_data_points()
        })
    
    except Exception as e:
        logger.error(f"Dashboard data error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/datasets')
def get_datasets():
    """Получить доступные датасеты"""
    try:
        datasets = []
        
        # Проверяем основную папку data
        data_dir = Path('../data')
        if data_dir.exists():
            for file in data_dir.glob('*.csv'):
                if 'SOLUSDT' in file.name:
                    try:
                        df = pd.read_csv(file)
                        datasets.append({
                            'id': file.stem,
                            'name': file.stem.replace('_', ' ').title(),
                            'path': str(file),
                            'size': f"{file.stat().st_size / 1024:.0f}KB",
                            'rows': len(df),
                            'timeframe': extract_timeframe_from_filename(file.name),
                            'date_range': get_date_range_from_df(df)
                        })
                    except Exception as e:
                        logger.error(f"Error reading dataset {file}: {e}")
        
        # Проверяем блоки данных
        blocks_dir = Path('../data/blocks/data')
        if blocks_dir.exists():
            for block_dir in blocks_dir.iterdir():
                if block_dir.is_dir():
                    csv_files = list(block_dir.glob('*.csv'))
                    if csv_files:
                        # Берем 1m файл как основной
                        main_file = next((f for f in csv_files if '1m' in f.name), csv_files[0])
                        try:
                            df = pd.read_csv(main_file)
                            total_size = sum(f.stat().st_size for f in csv_files)
                            
                            datasets.append({
                                'id': block_dir.name,
                                'name': format_block_name(block_dir.name),
                                'path': str(block_dir),
                                'size': f"{total_size / 1024:.0f}KB",
                                'rows': len(df),
                                'timeframes': [extract_timeframe_from_filename(f.name) for f in csv_files],
                                'date_range': get_date_range_from_df(df),
                                'files_count': len(csv_files)
                            })
                        except Exception as e:
                            logger.error(f"Error reading block dataset {block_dir}: {e}")
        
        return jsonify(datasets)
    
    except Exception as e:
        logger.error(f"Datasets error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models')
def get_models():
    """Получить доступные ML модели"""
    try:
        models = []
        models_dir = Path('../models')
        
        if models_dir.exists():
            # Получаем активную модель
            latest_link = models_dir / 'latest'
            active_model = None
            if latest_link.exists() and latest_link.is_symlink():
                active_model = latest_link.resolve().name
            
            # Сканируем все модели
            for model_dir in models_dir.iterdir():
                if model_dir.is_dir() and model_dir.name != 'latest':
                    try:
                        metadata_file = model_dir / 'metadata.json'
                        model_info = {
                            'id': model_dir.name,
                            'name': model_dir.name,
                            'path': str(model_dir),
                            'active': model_dir.name == active_model,
                            'created': datetime.fromtimestamp(model_dir.stat().st_ctime).isoformat()
                        }
                        
                        # Читаем метаданные
                        if metadata_file.exists():
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                                model_info.update({
                                    'accuracy': metadata.get('model_scores', {}),
                                    'features': metadata.get('features_count', 0),
                                    'training_samples': metadata.get('samples_trained', 0),
                                    'test_samples': metadata.get('samples_tested', 0),
                                    'ensemble_weights': metadata.get('ensemble_weights', {}),
                                    'training_date': metadata.get('training_date')
                                })
                        
                        # Проверяем файлы моделей
                        model_files = list(model_dir.glob('*.joblib'))
                        model_info['model_files'] = [f.stem for f in model_files if f.stem not in ['scaler', 'feature_names', 'performance_metrics']]
                        
                        models.append(model_info)
                    
                    except Exception as e:
                        logger.error(f"Error reading model {model_dir}: {e}")
        
        # Сортируем по дате создания (новые первые)
        models.sort(key=lambda x: x['created'], reverse=True)
        
        return jsonify(models)
    
    except Exception as e:
        logger.error(f"Models error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/training/start', methods=['POST'])
def start_training():
    """Запустить обучение ML моделей"""
    try:
        data = request.get_json()
        
        # Валидация данных
        if not data.get('dataset'):
            return jsonify({'error': 'Dataset is required'}), 400
        
        if not data.get('models'):
            return jsonify({'error': 'At least one model type is required'}), 400
        
        # Создаем процесс
        process_id = process_manager.create_process('training')
        
        # Запускаем обучение в отдельном потоке
        def run_training():
            try:
                # Подготавливаем команду
                cmd = [
                    'python', '../scripts/train_ml_models.py',
                    '--data-dir', '../data',
                    '--models-dir', '../models',
                    '--forward-periods', str(data.get('forward_periods', 30)),
                    '--min-samples', str(data.get('min_samples', 1000))
                ]
                
                # Указываем конкретный датасет, если это блок данных
                dataset = data['dataset']
                if 'august' in dataset:
                    cmd.extend(['--dataset', dataset])
                
                # Обновляем статус
                process_manager.update_process(process_id, 
                    progress=10.0, 
                    message='Подготовка команды обучения...'
                )
                
                # Запускаем процесс
                logger.info(f"Starting training with command: {' '.join(cmd)}")
                
                # Запускаем с PYTHONPATH
                env = os.environ.copy()
                env['PYTHONPATH'] = str(Path('../').resolve())
                
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                    cwd=Path('../').resolve()
                )
                
                # Мониторим прогресс
                progress_stages = [
                    (20, 'Загрузка данных...'),
                    (40, 'Расчет признаков...'),
                    (60, 'Обучение моделей...'),
                    (80, 'Валидация моделей...'),
                    (95, 'Сохранение результатов...')
                ]
                
                stage_index = 0
                output_lines = []
                
                while process.poll() is None:
                    line = process.stdout.readline()
                    if line:
                        output_lines.append(line.strip())
                        logger.info(f"Training output: {line.strip()}")
                        
                        # Обновляем прогресс на основе вывода
                        if stage_index < len(progress_stages):
                            progress, message = progress_stages[stage_index]
                            process_manager.update_process(process_id, 
                                progress=progress, 
                                message=message
                            )
                            stage_index += 1
                
                # Получаем финальный результат
                return_code = process.poll()
                final_output = '\n'.join(output_lines)
                
                if return_code == 0:
                    # Успешное завершение
                    results = {
                        'status': 'success',
                        'models_trained': data['models'],
                        'dataset_used': dataset,
                        'output': final_output,
                        'training_time': (datetime.now() - process_manager.get_process(process_id).start_time).total_seconds()
                    }
                    
                    # Пытаемся прочитать метаданные новой модели
                    try:
                        models_dir = Path('../models')
                        latest_model = max([d for d in models_dir.iterdir() if d.is_dir() and d.name != 'latest'], 
                                         key=lambda x: x.stat().st_mtime, default=None)
                        
                        if latest_model:
                            metadata_file = latest_model / 'metadata.json'
                            if metadata_file.exists():
                                with open(metadata_file, 'r') as f:
                                    metadata = json.load(f)
                                    results['model_metadata'] = metadata
                    except Exception as e:
                        logger.error(f"Error reading new model metadata: {e}")
                    
                    process_manager.complete_process(process_id, results)
                    
                else:
                    # Ошибка обучения
                    error_msg = f"Training failed with return code {return_code}\n{final_output}"
                    process_manager.complete_process(process_id, error=error_msg)
            
            except Exception as e:
                logger.error(f"Training thread error: {e}")
                process_manager.complete_process(process_id, error=str(e))
        
        # Запускаем в отдельном потоке
        training_thread = threading.Thread(target=run_training)
        training_thread.daemon = True
        training_thread.start()
        
        return jsonify({
            'process_id': process_id,
            'status': 'started',
            'message': 'Обучение запущено'
        })
    
    except Exception as e:
        logger.error(f"Start training error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/training/<process_id>/status')
def get_training_status(process_id):
    """Получить статус обучения"""
    try:
        process = process_manager.get_process(process_id)
        
        if not process:
            return jsonify({'error': 'Process not found'}), 404
        
        return jsonify({
            'process_id': process_id,
            'status': process.status,
            'progress': process.progress,
            'message': process.message,
            'start_time': process.start_time.isoformat(),
            'end_time': process.end_time.isoformat() if process.end_time else None,
            'results': process.results,
            'error': process.error
        })
    
    except Exception as e:
        logger.error(f"Training status error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/backtest/start', methods=['POST'])
def start_backtest():
    """Запустить бэктестирование"""
    try:
        data = request.get_json()
        
        # Валидация данных
        if not data.get('dataset'):
            return jsonify({'error': 'Dataset is required'}), 400
        
        # Создаем процесс
        process_id = process_manager.create_process('backtest')
        
        # Запускаем бэктест в отдельном потоке
        def run_backtest():
            try:
                # Подготавливаем команду
                dataset = data['dataset']
                balance = data.get('balance', 10000)
                commission = data.get('commission', 0.001)
                
                # Определяем количество дней на основе датасета
                days_map = {
                    'august_10_17_full': 8,
                    'august_12_single_day': 1,
                    'august_14_17_volatile': 4
                }
                days = days_map.get(dataset, 7)
                
                cmd = [
                    'python', '../scripts/enhanced_backtest.py',
                    '--symbol', 'SOLUSDT',
                    '--days', str(days),
                    '--balance', str(balance),
                    '--save-results'
                ]
                
                # Обновляем статус
                process_manager.update_process(process_id, 
                    progress=10.0, 
                    message='Подготовка бэктестирования...'
                )
                
                # Запускаем процесс
                logger.info(f"Starting backtest with command: {' '.join(cmd)}")
                
                env = os.environ.copy()
                env['PYTHONPATH'] = str(Path('../').resolve())
                
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                    cwd=Path('../').resolve()
                )
                
                # Мониторим прогресс
                output_lines = []
                progress_keywords = {
                    'Loading': 20,
                    'Running': 50,
                    'Analyzing': 80,
                    'Saving': 95
                }
                
                while process.poll() is None:
                    line = process.stdout.readline()
                    if line:
                        output_lines.append(line.strip())
                        logger.info(f"Backtest output: {line.strip()}")
                        
                        # Обновляем прогресс на основе ключевых слов
                        for keyword, progress in progress_keywords.items():
                            if keyword.lower() in line.lower():
                                process_manager.update_process(process_id, 
                                    progress=progress, 
                                    message=f'{keyword} данных...'
                                )
                                break
                
                # Получаем финальный результат
                return_code = process.poll()
                final_output = '\n'.join(output_lines)
                
                if return_code == 0:
                    # Успешное завершение - ищем результаты
                    results = find_latest_backtest_results()
                    
                    if results:
                        results['output'] = final_output
                        results['backtest_time'] = (datetime.now() - process_manager.get_process(process_id).start_time).total_seconds()
                        
                        process_manager.complete_process(process_id, results)
                    else:
                        process_manager.complete_process(process_id, error="No results found")
                        
                else:
                    # Ошибка бэктеста
                    error_msg = f"Backtest failed with return code {return_code}\n{final_output}"
                    process_manager.complete_process(process_id, error=error_msg)
            
            except Exception as e:
                logger.error(f"Backtest thread error: {e}")
                process_manager.complete_process(process_id, error=str(e))
        
        # Запускаем в отдельном потоке
        backtest_thread = threading.Thread(target=run_backtest)
        backtest_thread.daemon = True
        backtest_thread.start()
        
        return jsonify({
            'process_id': process_id,
            'status': 'started',
            'message': 'Бэктест запущен'
        })
    
    except Exception as e:
        logger.error(f"Start backtest error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/backtest/<process_id>/status')
def get_backtest_status(process_id):
    """Получить статус бэктестирования"""
    try:
        process = process_manager.get_process(process_id)
        
        if not process:
            return jsonify({'error': 'Process not found'}), 404
        
        return jsonify({
            'process_id': process_id,
            'status': process.status,
            'progress': process.progress,
            'message': process.message,
            'start_time': process.start_time.isoformat(),
            'end_time': process.end_time.isoformat() if process.end_time else None,
            'results': process.results,
            'error': process.error
        })
    
    except Exception as e:
        logger.error(f"Backtest status error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/trades')
def get_trades():
    """Получить список сделок"""
    try:
        # Ищем последние результаты бэктеста
        results = find_latest_backtest_results()
        
        if results and 'trades' in results:
            trades = results['trades']
            
            # Применяем фильтры если есть
            date_filter = request.args.get('date_filter', 'all')
            type_filter = request.args.get('type_filter', 'all')
            pnl_filter = request.args.get('pnl_filter', 'all')
            
            filtered_trades = apply_trade_filters(trades, date_filter, type_filter, pnl_filter)
            
            return jsonify({
                'trades': filtered_trades,
                'total_count': len(trades),
                'filtered_count': len(filtered_trades),
                'summary': calculate_trades_summary(filtered_trades)
            })
        else:
            return jsonify({
                'trades': [],
                'total_count': 0,
                'filtered_count': 0,
                'summary': {}
            })
    
    except Exception as e:
        logger.error(f"Trades error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/data-collection/start', methods=['POST'])
def start_data_collection():
    """Запустить сбор рыночных данных"""
    try:
        data = request.get_json()
        
        # Валидация данных
        if not data.get('symbol'):
            return jsonify({'error': 'Symbol is required'}), 400
        
        if not data.get('start_date') or not data.get('end_date'):
            return jsonify({'error': 'Start and end dates are required'}), 400
            
        if not data.get('timeframes'):
            return jsonify({'error': 'At least one timeframe is required'}), 400
        
        # Создаем процесс
        process_id = process_manager.create_process('data_collection')
        
        # Запускаем сбор данных в отдельном потоке
        def run_data_collection():
            try:
                symbol = data['symbol']
                start_date = data['start_date']
                end_date = data['end_date']
                timeframes = data['timeframes']
                dataset_name = data.get('dataset_name', '')
                
                # Рассчитываем количество дней
                from datetime import datetime
                start_dt = datetime.fromisoformat(start_date.replace('T', ' '))
                end_dt = datetime.fromisoformat(end_date.replace('T', ' '))
                days = (end_dt - start_dt).days + 1
                
                # Подготавливаем команду
                cmd = [
                    'python', 'scripts/collect_data.py',
                    '--symbol', symbol,
                    '--days', str(days)
                ]
                
                # Добавляем custom start date если указан
                if start_date:
                    cmd.extend(['--start-date', start_date])
                
                # Обновляем статус
                process_manager.update_process(process_id, 
                    progress=10.0, 
                    message=f'Начинаем сбор данных для {symbol}...'
                )
                
                # Запускаем процесс
                logger.info(f"Starting data collection with command: {' '.join(cmd)}")
                
                env = os.environ.copy()
                env['PYTHONPATH'] = str(Path(__file__).parent.parent.resolve())
                
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                    cwd=Path(__file__).parent.parent.resolve()
                )
                
                # Мониторим прогресс
                progress_stages = [
                    (25, 'Подключение к API...'),
                    (40, 'Сбор 1m данных...'),
                    (55, 'Сбор 5m данных...'),
                    (70, 'Сбор 15m данных...'),
                    (85, 'Сбор 1h данных...'),
                    (95, 'Сохранение файлов...')
                ]
                
                stage_index = 0
                output_lines = []
                candles_collected = 0
                
                while process.poll() is None:
                    line = process.stdout.readline()
                    if line:
                        output_lines.append(line.strip())
                        logger.info(f"Data collection output: {line.strip()}")
                        
                        # Парсим количество собранных свечей
                        if 'collected' in line.lower() and 'candles' in line.lower():
                            try:
                                import re
                                numbers = re.findall(r'\d+', line)
                                if numbers:
                                    candles_collected += int(numbers[-1])
                            except:
                                pass
                        
                        # Обновляем прогресс
                        if stage_index < len(progress_stages):
                            progress, message = progress_stages[stage_index]
                            process_manager.update_process(process_id, 
                                progress=progress, 
                                message=message
                            )
                            stage_index += 1
                
                # Получаем финальный результат
                return_code = process.poll()
                final_output = '\n'.join(output_lines)
                
                if return_code == 0:
                    # Успешное завершение
                    results = {
                        'status': 'success',
                        'symbol': symbol,
                        'period': f"{start_date} - {end_date}",
                        'timeframes': timeframes,
                        'candles_collected': candles_collected,
                        'days': days,
                        'output': final_output,
                        'collection_time': (datetime.now() - process_manager.get_process(process_id).start_time).total_seconds()
                    }
                    
                    # Проверяем созданные файлы
                    try:
                        data_dir = Path('../data')
                        csv_files = []
                        total_size = 0
                        
                        for tf in timeframes:
                            pattern = f"{symbol}*{tf}*.csv"
                            files = list(data_dir.glob(pattern))
                            if files:
                                latest_file = max(files, key=lambda x: x.stat().st_mtime)
                                csv_files.append(str(latest_file))
                                total_size += latest_file.stat().st_size
                        
                        results['files_created'] = csv_files
                        results['total_size'] = total_size
                    except Exception as e:
                        logger.error(f"Error checking created files: {e}")
                    
                    process_manager.complete_process(process_id, results)
                else:
                    # Ошибка сбора данных
                    error_msg = f"Data collection failed with return code {return_code}\n{final_output}"
                    process_manager.complete_process(process_id, error=error_msg)
            
            except Exception as e:
                logger.error(f"Data collection thread error: {e}")
                process_manager.complete_process(process_id, error=str(e))
        
        # Запускаем в отдельном потоке
        collection_thread = threading.Thread(target=run_data_collection)
        collection_thread.daemon = True
        collection_thread.start()
        
        return jsonify({
            'process_id': process_id,
            'status': 'started',
            'message': f'Сбор данных для {data["symbol"]} запущен'
        })
    
    except Exception as e:
        logger.error(f"Start data collection error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/data-collection/<process_id>/status')
def get_data_collection_status(process_id):
    """Получить статус сбора данных"""
    try:
        process = process_manager.get_process(process_id)
        
        if not process:
            return jsonify({'error': 'Process not found'}), 404
        
        return jsonify({
            'process_id': process_id,
            'status': process.status,
            'progress': process.progress,
            'message': process.message,
            'start_time': process.start_time.isoformat(),
            'end_time': process.end_time.isoformat() if process.end_time else None,
            'results': process.results,
            'error': process.error
        })
    
    except Exception as e:
        logger.error(f"Data collection status error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/data-collection/history')
def get_data_collection_history():
    """Получить историю сбора данных"""
    try:
        history = []
        
        # Ищем все процессы сбора данных
        for process_id, process in process_manager.processes.items():
            if process.type == 'data_collection' and process.status == 'completed':
                history_item = {
                    'id': process_id,
                    'date': process.start_time.isoformat(),
                    'status': process.status,
                    'duration': (process.end_time - process.start_time).total_seconds() if process.end_time else 0
                }
                
                if process.results:
                    history_item.update({
                        'symbol': process.results.get('symbol', 'N/A'),
                        'period': process.results.get('period', 'N/A'),
                        'timeframes': process.results.get('timeframes', []),
                        'candles_collected': process.results.get('candles_collected', 0),
                        'total_size': process.results.get('total_size', 0),
                        'files_created': process.results.get('files_created', [])
                    })
                
                history.append(history_item)
        
        # Сортируем по дате (новые первые)
        history.sort(key=lambda x: x['date'], reverse=True)
        
        return jsonify(history)
    
    except Exception as e:
        logger.error(f"Data collection history error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/system/status')
def get_system_status():
    """Получить статус системы"""
    try:
        # Проверяем статус компонентов
        status = {
            'system': 'online',
            'timestamp': datetime.now().isoformat(),
            'components': {
                'data_collector': check_component_status('data'),
                'ml_models': check_component_status('models'),
                'backtest_engine': check_component_status('scripts'),
                'web_interface': 'online'
            },
            'running_processes': len(process_manager.processes),
            'memory_usage': get_memory_usage(),
            'disk_space': get_disk_space()
        }
        
        return jsonify(status)
    
    except Exception as e:
        logger.error(f"System status error: {e}")
        return jsonify({'error': str(e)}), 500

# Вспомогательные функции
def extract_timeframe_from_filename(filename: str) -> str:
    """Извлечь таймфрейм из имени файла"""
    timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
    for tf in timeframes:
        if tf in filename:
            return tf
    return 'unknown'

def get_date_range_from_df(df: pd.DataFrame) -> Dict[str, str]:
    """Получить диапазон дат из DataFrame"""
    try:
        if 'timestamp' in df.columns:
            timestamps = pd.to_datetime(df['timestamp'])
            return {
                'start': timestamps.min().strftime('%Y-%m-%d'),
                'end': timestamps.max().strftime('%Y-%m-%d'),
                'days': (timestamps.max() - timestamps.min()).days + 1
            }
    except Exception:
        pass
    
    return {'start': 'unknown', 'end': 'unknown', 'days': 0}

def format_block_name(block_name: str) -> str:
    """Форматировать название блока данных"""
    name_map = {
        'august_10_17_full': '10-17 августа 2025 (Полный период)',
        'august_12_single_day': '12 августа 2025 (Один день)',
        'august_14_17_volatile': '14-17 августа 2025 (Волатильный период)',
        'august_10_13_trend': '10-13 августа 2025 (Трендовый период)'
    }
    
    return name_map.get(block_name, block_name.replace('_', ' ').title())

def get_available_data_points() -> int:
    """Подсчитать общее количество точек данных"""
    total = 0
    try:
        data_dir = Path('../data')
        if data_dir.exists():
            for csv_file in data_dir.glob('**/*.csv'):
                try:
                    df = pd.read_csv(csv_file)
                    total += len(df)
                except Exception:
                    continue
    except Exception:
        pass
    
    return total

def find_latest_backtest_results() -> Optional[Dict]:
    """Найти последние результаты бэктеста"""
    try:
        output_dir = Path('../output')
        if not output_dir.exists():
            return None
        
        # Ищем JSON файлы с результатами
        result_files = []
        for pattern in ['*strategy*.json', '*backtest*.json']:
            result_files.extend(output_dir.glob(pattern))
        
        if not result_files:
            return None
        
        # Берем самый новый файл
        latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
        
        with open(latest_file, 'r') as f:
            data = json.load(f)
            
        # Проверяем, что есть сделки
        if 'trades' in data and data['trades']:
            return data
            
    except Exception as e:
        logger.error(f"Error finding backtest results: {e}")
    
    return None

def apply_trade_filters(trades: List[Dict], date_filter: str, type_filter: str, pnl_filter: str) -> List[Dict]:
    """Применить фильтры к сделкам"""
    filtered = trades.copy()
    
    # Фильтр по типу
    if type_filter != 'all':
        filtered = [t for t in filtered if t.get('signal_type') == type_filter]
    
    # Фильтр по P&L
    if pnl_filter == 'profit':
        filtered = [t for t in filtered if t.get('net_pnl', 0) > 0]
    elif pnl_filter == 'loss':
        filtered = [t for t in filtered if t.get('net_pnl', 0) < 0]
    
    # Фильтр по дате (упрощенная реализация)
    if date_filter != 'all':
        # Для демонстрации - можно расширить
        pass
    
    return filtered

def calculate_trades_summary(trades: List[Dict]) -> Dict:
    """Вычислить сводку по сделкам"""
    if not trades:
        return {}
    
    winning_trades = [t for t in trades if t.get('net_pnl', 0) > 0]
    losing_trades = [t for t in trades if t.get('net_pnl', 0) < 0]
    
    total_pnl = sum(t.get('net_pnl', 0) for t in trades)
    total_commission = sum(t.get('commission', 0) for t in trades)
    
    return {
        'total_trades': len(trades),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'win_rate': len(winning_trades) / len(trades) * 100 if trades else 0,
        'total_pnl': total_pnl,
        'total_commission': total_commission,
        'avg_win': sum(t.get('net_pnl', 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0,
        'avg_loss': sum(t.get('net_pnl', 0) for t in losing_trades) / len(losing_trades) if losing_trades else 0
    }

def check_component_status(component_path: str) -> str:
    """Проверить статус компонента"""
    try:
        path = Path(f'../{component_path}')
        if path.exists():
            return 'online'
        else:
            return 'offline'
    except Exception:
        return 'error'

def get_memory_usage() -> Dict:
    """Получить информацию об использовании памяти"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        return {
            'total': memory.total,
            'used': memory.used,
            'percent': memory.percent
        }
    except ImportError:
        return {'error': 'psutil not available'}

def get_disk_space() -> Dict:
    """Получить информацию о дисковом пространстве"""
    try:
        import shutil
        total, used, free = shutil.disk_usage('.')
        return {
            'total': total,
            'used': used,
            'free': free,
            'percent': (used / total) * 100
        }
    except Exception:
        return {'error': 'Unable to get disk space'}

if __name__ == '__main__':
    # Проверяем зависимости
    try:
        import flask
        import flask_cors
        logger.info("Flask dependencies are available")
    except ImportError as e:
        logger.error(f"Missing Flask dependencies: {e}")
        print("Please install Flask dependencies:")
        print("pip install flask flask-cors")
        sys.exit(1)
    
    # Настраиваем приложение
    app.config['DEBUG'] = True
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching for development
    
    logger.info("Starting Trading Bot Web Interface API")
    logger.info("API will be available at: http://localhost:8080")
    logger.info("Web interface will be available at: http://localhost:8080")
    
    # Запускаем сервер
    app.run(host='0.0.0.0', port=8080, debug=True, threaded=True)