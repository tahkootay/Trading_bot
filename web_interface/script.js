// Trading Bot Web Interface JavaScript
class TradingBotInterface {
    constructor() {
        this.currentPanel = 'dashboard';
        this.autoRefresh = true;
        this.refreshInterval = 30000; // 30 seconds
        this.refreshTimer = null;
        this.processes = new Map(); // Track running processes
        
        this.init();
    }

    init() {
        this.bindNavigationEvents();
        this.bindFormEvents();
        this.bindButtonEvents();
        this.bindModalEvents();
        this.bindRangeSliders();
        this.loadInitialData();
        this.startAutoRefresh();
        
        console.log('Trading Bot Interface initialized');
    }

    // Navigation
    bindNavigationEvents() {
        const navLinks = document.querySelectorAll('.nav-link');
        navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const panelId = link.getAttribute('href').substring(1);
                this.showPanel(panelId);
                
                // Update active nav link
                navLinks.forEach(l => l.classList.remove('active'));
                link.classList.add('active');
                
                // Update page title
                const title = link.textContent.trim();
                document.getElementById('page-title').textContent = title;
            });
        });
    }

    showPanel(panelId) {
        // Hide all panels
        document.querySelectorAll('.panel').forEach(panel => {
            panel.classList.remove('active');
        });
        
        // Show selected panel
        const targetPanel = document.getElementById(panelId);
        if (targetPanel) {
            targetPanel.classList.add('active');
            this.currentPanel = panelId;
            
            // Load panel-specific data
            this.loadPanelData(panelId);
        }
    }

    // Data Loading
    loadInitialData() {
        this.loadDashboardData();
        this.loadAvailableDatasets();
        this.loadModels();
    }

    async loadDashboardData() {
        try {
            // Simulate API call to get dashboard data
            const data = await this.apiCall('GET', '/api/dashboard');
            this.updateDashboardStats(data);
        } catch (error) {
            console.error('Error loading dashboard data:', error);
            // Use fallback data
            this.updateDashboardStats({
                activeModels: 2,
                mlAccuracy: 98.90,
                lastPnl: 7.09,
                lastTraining: '22.08.2025 17:54'
            });
        }
    }

    updateDashboardStats(data) {
        if (data.activeModels !== undefined) {
            document.getElementById('activeModels').textContent = data.activeModels;
        }
        if (data.mlAccuracy !== undefined) {
            document.getElementById('mlAccuracy').textContent = `${data.mlAccuracy}%`;
        }
        if (data.lastPnl !== undefined) {
            const pnlElement = document.getElementById('lastPnl');
            pnlElement.textContent = `$${data.lastPnl > 0 ? '+' : ''}${data.lastPnl.toFixed(2)}`;
            pnlElement.className = `stat-value ${data.lastPnl >= 0 ? 'profit' : 'loss'}`;
        }
        if (data.lastTraining) {
            document.getElementById('lastTraining').textContent = data.lastTraining;
        }
    }

    async loadAvailableDatasets() {
        try {
            // Simulate API call to get available datasets
            const datasets = [
                {
                    id: 'august_10_17_full',
                    name: '10-17 августа 2025 (Полный)',
                    description: '8 дней торговых данных | 11,520 1m свечей',
                    size: '2.3MB',
                    path: 'data/blocks/data/august_10_17_full'
                },
                {
                    id: 'august_12_single_day',
                    name: '12 августа 2025 (Один день)',
                    description: '24 часа активной торговли | 1,440 1m свечей',
                    size: '288KB',
                    path: 'data/blocks/data/august_12_single_day'
                },
                {
                    id: 'august_14_17_volatile',
                    name: '14-17 августа 2025 (Волатильный)',
                    description: '4 дня высокой волатильности | 5,760 1m свечей',
                    size: '1.1MB',
                    path: 'data/blocks/data/august_14_17_volatile'
                }
            ];
            
            this.updateDatasetOptions(datasets);
        } catch (error) {
            console.error('Error loading datasets:', error);
        }
    }

    updateDatasetOptions(datasets) {
        // Update training dataset select
        const trainingSelect = document.getElementById('selectedDataset');
        const backtestSelect = document.getElementById('backtestDataset');
        
        [trainingSelect, backtestSelect].forEach(select => {
            if (select) {
                // Clear existing options except the first one
                while (select.children.length > 1) {
                    select.removeChild(select.lastChild);
                }
                
                datasets.forEach(dataset => {
                    const option = document.createElement('option');
                    option.value = dataset.id;
                    option.textContent = dataset.name;
                    select.appendChild(option);
                });
            }
        });
    }

    async loadModels() {
        try {
            // Simulate API call to get available models
            const models = await this.apiCall('GET', '/api/models');
            this.updateModelsDisplay(models);
        } catch (error) {
            console.error('Error loading models:', error);
        }
    }

    // Form Handlers
    bindFormEvents() {
        // Training form
        const trainingForm = document.getElementById('trainingForm');
        if (trainingForm) {
            trainingForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.handleTrainingSubmit(new FormData(trainingForm));
            });
        }

        // Backtest form
        const backtestForm = document.getElementById('backtestForm');
        if (backtestForm) {
            backtestForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.handleBacktestSubmit(new FormData(backtestForm));
            });
        }

        // Data collection form
        const dataCollectionForm = document.getElementById('dataCollectionForm');
        if (dataCollectionForm) {
            dataCollectionForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.handleDataCollectionSubmit(new FormData(dataCollectionForm));
            });
        }
    }

    async handleTrainingSubmit(formData) {
        const dataset = formData.get('dataset');
        const selectedModels = formData.getAll('models');
        const testSize = formData.get('test_size');
        const forwardPeriods = formData.get('forward_periods');

        if (!dataset) {
            this.showNotification('Выберите датасет для обучения', 'error');
            return;
        }

        if (selectedModels.length === 0) {
            this.showNotification('Выберите хотя бы одну модель', 'error');
            return;
        }

        // Show progress
        const progressContainer = document.getElementById('trainingProgress');
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');
        const progressPercent = document.getElementById('progressPercent');

        progressContainer.style.display = 'block';
        
        try {
            // Start training process
            const trainingId = await this.startTraining({
                dataset,
                models: selectedModels,
                test_size: parseFloat(testSize),
                forward_periods: parseInt(forwardPeriods)
            });

            this.processes.set(trainingId, {
                type: 'training',
                status: 'running',
                startTime: Date.now()
            });

            // Monitor training progress
            this.monitorTrainingProgress(trainingId, progressFill, progressText, progressPercent);
            
        } catch (error) {
            console.error('Training error:', error);
            this.showNotification('Ошибка запуска обучения', 'error');
            progressContainer.style.display = 'none';
        }
    }

    async handleBacktestSubmit(formData) {
        const dataset = formData.get('dataset');
        const balance = parseFloat(formData.get('balance'));
        const commission = parseFloat(formData.get('commission'));
        const confidence = parseFloat(formData.get('confidence'));
        const maxPosition = parseFloat(formData.get('max_position'));

        if (!dataset) {
            this.showNotification('Выберите датасет для бэктеста', 'error');
            return;
        }

        try {
            // Start backtest
            const backtestId = await this.startBacktest({
                dataset,
                balance,
                commission: commission / 100, // Convert to decimal
                confidence_threshold: confidence,
                max_position_size: maxPosition / 100 // Convert to decimal
            });

            this.processes.set(backtestId, {
                type: 'backtest',
                status: 'running',
                startTime: Date.now()
            });

            // Show loading state
            const resultsContainer = document.getElementById('backtestResults');
            resultsContainer.innerHTML = `
                <div class="loading-state">
                    <div class="loading-spinner"></div>
                    <p>Выполняется бэктестирование...</p>
                </div>
            `;

            // Monitor backtest progress
            this.monitorBacktestProgress(backtestId);
            
        } catch (error) {
            console.error('Backtest error:', error);
            this.showNotification('Ошибка запуска бэктестирования', 'error');
        }
    }

    async handleDataCollectionSubmit(formData) {
        const symbol = formData.get('symbol');
        const startDate = formData.get('start_date');
        const endDate = formData.get('end_date');
        const selectedTimeframes = formData.getAll('timeframes');
        const datasetName = formData.get('dataset_name');

        // Validation
        if (!symbol || !startDate || !endDate) {
            this.showNotification('Заполните все обязательные поля', 'error');
            return;
        }

        if (selectedTimeframes.length === 0) {
            this.showNotification('Выберите хотя бы один таймфрейм', 'error');
            return;
        }

        // Validate date range
        const start = new Date(startDate);
        const end = new Date(endDate);
        if (start >= end) {
            this.showNotification('Конечная дата должна быть позже начальной', 'error');
            return;
        }

        // Show progress
        const progressContainer = document.getElementById('collectionProgress');
        const progressFill = document.getElementById('collectionProgressFill');
        const progressText = document.getElementById('collectionProgressText');
        const progressPercent = document.getElementById('collectionProgressPercent');
        
        progressContainer.style.display = 'block';
        
        try {
            // Start data collection
            const collectionId = await this.startDataCollection({
                symbol,
                start_date: startDate,
                end_date: endDate,
                timeframes: selectedTimeframes,
                dataset_name: datasetName
            });

            this.processes.set(collectionId, {
                type: 'data_collection',
                status: 'running',
                startTime: Date.now()
            });

            // Monitor collection progress
            this.monitorDataCollectionProgress(collectionId, progressFill, progressText, progressPercent);
            
            this.showNotification(`Начат сбор данных для ${symbol}`, 'success');

        } catch (error) {
            console.error('Error starting data collection:', error);
            this.showNotification('Ошибка при запуске сбора данных', 'error');
            progressContainer.style.display = 'none';
        }
    }

    // Process Monitoring
    async monitorTrainingProgress(trainingId, progressFill, progressText, progressPercent) {
        const pollInterval = 2000; // 2 seconds
        
        const poll = async () => {
            try {
                const status = await this.apiCall('GET', `/api/training/${trainingId}/status`);
                
                if (status.progress !== undefined) {
                    progressFill.style.width = `${status.progress}%`;
                    progressPercent.textContent = `${Math.round(status.progress)}%`;
                }
                
                if (status.message) {
                    progressText.textContent = status.message;
                }
                
                if (status.status === 'completed') {
                    this.handleTrainingComplete(trainingId, status);
                    return;
                } else if (status.status === 'failed') {
                    this.handleTrainingFailed(trainingId, status);
                    return;
                }
                
                // Continue polling if still running
                setTimeout(poll, pollInterval);
                
            } catch (error) {
                console.error('Error polling training status:', error);
                setTimeout(poll, pollInterval * 2); // Retry with longer interval
            }
        };
        
        poll();
    }

    async monitorBacktestProgress(backtestId) {
        const pollInterval = 3000; // 3 seconds
        
        const poll = async () => {
            try {
                const status = await this.apiCall('GET', `/api/backtest/${backtestId}/status`);
                
                if (status.status === 'completed') {
                    this.handleBacktestComplete(backtestId, status);
                    return;
                } else if (status.status === 'failed') {
                    this.handleBacktestFailed(backtestId, status);
                    return;
                }
                
                // Continue polling if still running
                setTimeout(poll, pollInterval);
                
            } catch (error) {
                console.error('Error polling backtest status:', error);
                setTimeout(poll, pollInterval * 2);
            }
        };
        
        poll();
    }

    async monitorDataCollectionProgress(collectionId, progressFill, progressText, progressPercent) {
        const pollInterval = 2000; // 2 seconds
        const startTime = Date.now();
        
        const poll = async () => {
            try {
                const status = await this.apiCall('GET', `/api/data-collection/${collectionId}/status`);
                
                // Update progress bar
                if (status.progress !== undefined) {
                    progressFill.style.width = `${status.progress}%`;
                    progressPercent.textContent = `${Math.round(status.progress)}%`;
                }
                
                // Update status message
                if (status.message) {
                    progressText.textContent = status.message;
                }
                
                // Update collection stats
                if (status.results) {
                    const currentTimeframe = document.getElementById('currentTimeframe');
                    const candlesCollected = document.getElementById('candlesCollected');
                    const collectionTime = document.getElementById('collectionTime');
                    
                    if (status.results.timeframes && status.results.timeframes.length > 0) {
                        currentTimeframe.textContent = status.results.timeframes.join(', ');
                    }
                    
                    if (status.results.candles_collected) {
                        candlesCollected.textContent = status.results.candles_collected.toLocaleString();
                    }
                    
                    // Calculate elapsed time
                    const elapsed = Math.floor((Date.now() - startTime) / 1000);
                    const minutes = Math.floor(elapsed / 60);
                    const seconds = elapsed % 60;
                    collectionTime.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
                }
                
                // Check if completed
                if (status.status === 'completed') {
                    this.handleDataCollectionComplete(collectionId, status);
                    return;
                } else if (status.status === 'failed') {
                    this.handleDataCollectionFailed(collectionId, status);
                    return;
                }
                
                // Continue polling if still running
                setTimeout(poll, pollInterval);
                
            } catch (error) {
                console.error('Error polling data collection status:', error);
                setTimeout(poll, pollInterval * 2); // Retry with longer interval
            }
        };
        
        poll();
    }

    handleTrainingComplete(trainingId, status) {
        const progressText = document.getElementById('progressText');
        const progressFill = document.getElementById('progressFill');
        const progressPercent = document.getElementById('progressPercent');
        
        progressFill.style.width = '100%';
        progressPercent.textContent = '100%';
        progressText.textContent = 'Обучение завершено успешно!';
        
        this.showNotification('Обучение ML моделей завершено', 'success');
        
        // Update models list
        this.loadModels();
        
        // Clean up process
        this.processes.delete(trainingId);
        
        // Log training results
        if (status.results) {
            this.logTrainingResults(status.results);
        }
    }

    handleBacktestComplete(backtestId, status) {
        const resultsContainer = document.getElementById('backtestResults');
        
        if (status.results) {
            this.displayBacktestResults(status.results, resultsContainer);
            this.loadTradesData(status.results.trades || []);
        }
        
        this.showNotification('Бэктестирование завершено', 'success');
        this.processes.delete(backtestId);
    }

    handleDataCollectionComplete(collectionId, status) {
        const progressText = document.getElementById('collectionProgressText');
        const progressFill = document.getElementById('collectionProgressFill');
        const progressPercent = document.getElementById('collectionProgressPercent');
        const logs = document.getElementById('collectionLogs');
        
        // Update progress to 100%
        progressFill.style.width = '100%';
        progressPercent.textContent = '100%';
        progressText.textContent = 'Сбор данных завершен успешно!';
        
        // Update final statistics
        if (status.results) {
            const currentTimeframe = document.getElementById('currentTimeframe');
            const candlesCollected = document.getElementById('candlesCollected');
            
            if (status.results.timeframes) {
                currentTimeframe.textContent = status.results.timeframes.join(', ');
            }
            
            if (status.results.candles_collected) {
                candlesCollected.textContent = status.results.candles_collected.toLocaleString();
            }
        }
        
        // Show completion logs
        if (logs && status.results && status.results.output) {
            logs.innerHTML = `<pre>${status.results.output}</pre>`;
        }
        
        this.showNotification('Сбор данных завершен успешно', 'success');
        
        // Refresh datasets list to show new data
        this.loadAvailableDatasets();
        
        // Clean up process
        this.processes.delete(collectionId);
        
        // Show collection results summary
        if (status.results) {
            this.showDataCollectionSummary(status.results);
        }
    }

    handleDataCollectionFailed(collectionId, status) {
        const progressText = document.getElementById('collectionProgressText');
        const logs = document.getElementById('collectionLogs');
        
        progressText.textContent = 'Ошибка при сборе данных';
        
        if (logs && status.error) {
            logs.innerHTML = `<pre style="color: #e74c3c;">${status.error}</pre>`;
        }
        
        this.showNotification('Сбор данных завершился с ошибкой', 'error');
        this.processes.delete(collectionId);
    }

    showDataCollectionSummary(results) {
        const summary = `
            <div class="collection-summary">
                <h4>📊 Результаты Сбора Данных</h4>
                <div class="summary-stats">
                    <div class="stat">
                        <span class="label">Символ:</span>
                        <span class="value">${results.symbol}</span>
                    </div>
                    <div class="stat">
                        <span class="label">Период:</span>
                        <span class="value">${results.period}</span>
                    </div>
                    <div class="stat">
                        <span class="label">Таймфреймы:</span>
                        <span class="value">${results.timeframes.join(', ')}</span>
                    </div>
                    <div class="stat">
                        <span class="label">Свечей собрано:</span>
                        <span class="value">${results.candles_collected.toLocaleString()}</span>
                    </div>
                    <div class="stat">
                        <span class="label">Размер файлов:</span>
                        <span class="value">${this.formatFileSize(results.total_size)}</span>
                    </div>
                    <div class="stat">
                        <span class="label">Время сбора:</span>
                        <span class="value">${Math.round(results.collection_time)}s</span>
                    </div>
                </div>
            </div>
        `;
        
        // Show summary in a notification or modal
        this.showNotification(summary, 'success', 10000); // Show for 10 seconds
    }

    toggleCollectionHistory() {
        const historyContainer = document.getElementById('collectionHistory');
        const dataContainer = document.querySelector('.data-collection-container');
        
        if (historyContainer.style.display === 'none' || !historyContainer.style.display) {
            // Show history
            historyContainer.style.display = 'block';
            dataContainer.style.display = 'none';
            this.loadCollectionHistory();
            document.getElementById('collectionHistoryBtn').innerHTML = '<i class="fas fa-arrow-left"></i> Назад';
        } else {
            // Hide history
            historyContainer.style.display = 'none';
            dataContainer.style.display = 'grid';
            document.getElementById('collectionHistoryBtn').innerHTML = '<i class="fas fa-history"></i> История Сбора';
        }
    }

    async loadCollectionHistory() {
        try {
            const history = await this.apiCall('GET', '/api/data-collection/history');
            this.displayCollectionHistory(history);
        } catch (error) {
            console.error('Error loading collection history:', error);
            this.displayCollectionHistory([]); // Show empty state
        }
    }

    displayCollectionHistory(history) {
        const tableBody = document.getElementById('historyTableBody');
        
        if (!history || history.length === 0) {
            tableBody.innerHTML = `
                <tr>
                    <td colspan="8" class="no-data">
                        <i class="fas fa-history"></i>
                        <p>История сбора данных пуста</p>
                    </td>
                </tr>
            `;
            return;
        }

        tableBody.innerHTML = '';
        
        history.forEach(item => {
            const row = document.createElement('tr');
            const date = new Date(item.date).toLocaleString();
            const timeframes = Array.isArray(item.timeframes) ? item.timeframes.join(', ') : 'N/A';
            const fileSize = this.formatFileSize(item.total_size || 0);
            
            row.innerHTML = `
                <td>${date}</td>
                <td>${item.symbol}</td>
                <td>${item.period}</td>
                <td>${timeframes}</td>
                <td>${item.candles_collected ? item.candles_collected.toLocaleString() : '0'}</td>
                <td>${fileSize}</td>
                <td><span class="status-badge status-${item.status}">${item.status === 'completed' ? 'Завершен' : item.status}</span></td>
                <td>
                    <button class="btn btn-small btn-secondary" onclick="tradingBot.viewCollectionDetails('${item.id}')">
                        <i class="fas fa-eye"></i>
                    </button>
                </td>
            `;
            tableBody.appendChild(row);
        });
    }

    setQuickPeriod(period) {
        const startDateInput = document.getElementById('startDate');
        const endDateInput = document.getElementById('endDate');
        
        if (!startDateInput || !endDateInput) return;
        
        const now = new Date();
        let startDate, endDate;
        
        switch (period) {
            case '1':
                startDate = new Date(now.getTime() - 24 * 60 * 60 * 1000); // 1 day ago
                endDate = now;
                break;
            case '7':
                startDate = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000); // 7 days ago
                endDate = now;
                break;
            case '30':
                startDate = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000); // 30 days ago
                endDate = now;
                break;
            case 'custom':
                startDate = new Date(now.getTime() - 60 * 60 * 1000); // 1 hour ago
                endDate = now;
                break;
            default:
                return;
        }
        
        // Format dates for datetime-local input
        const formatDate = (date) => {
            const year = date.getFullYear();
            const month = String(date.getMonth() + 1).padStart(2, '0');
            const day = String(date.getDate()).padStart(2, '0');
            const hours = String(date.getHours()).padStart(2, '0');
            const minutes = String(date.getMinutes()).padStart(2, '0');
            return `${year}-${month}-${day}T${hours}:${minutes}`;
        };
        
        startDateInput.value = formatDate(startDate);
        endDateInput.value = formatDate(endDate);
        
        this.showNotification(`Период установлен: ${period === 'custom' ? 'текущий час' : period + (period === '1' ? ' день' : ' дней')}`, 'info');
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    }

    displayBacktestResults(results, container) {
        const profit = results.final_balance - results.initial_balance;
        const returnPercent = (profit / results.initial_balance * 100);
        const winRate = results.winning_trades / (results.winning_trades + results.losing_trades) * 100;
        
        container.innerHTML = `
            <div class="backtest-summary">
                <h4>Результаты Бэктестирования</h4>
                <div class="results-grid">
                    <div class="result-item">
                        <span class="label">Прибыль:</span>
                        <span class="value ${profit >= 0 ? 'profit' : 'loss'}">
                            $${profit > 0 ? '+' : ''}${profit.toFixed(2)}
                        </span>
                    </div>
                    <div class="result-item">
                        <span class="label">Доходность:</span>
                        <span class="value ${returnPercent >= 0 ? 'profit' : 'loss'}">
                            ${returnPercent > 0 ? '+' : ''}${returnPercent.toFixed(2)}%
                        </span>
                    </div>
                    <div class="result-item">
                        <span class="label">Win Rate:</span>
                        <span class="value">${winRate.toFixed(1)}%</span>
                    </div>
                    <div class="result-item">
                        <span class="label">Всего сделок:</span>
                        <span class="value">${results.winning_trades + results.losing_trades}</span>
                    </div>
                    <div class="result-item">
                        <span class="label">Прибыльных:</span>
                        <span class="value profit">${results.winning_trades}</span>
                    </div>
                    <div class="result-item">
                        <span class="label">Убыточных:</span>
                        <span class="value loss">${results.losing_trades}</span>
                    </div>
                </div>
                <div class="results-actions">
                    <button class="btn btn-primary" onclick="tradingBot.showPanel('trades')">
                        <i class="fas fa-chart-line"></i> Анализ Сделок
                    </button>
                    <button class="btn btn-secondary" onclick="tradingBot.exportResults('${backtestId}')">
                        <i class="fas fa-download"></i> Экспорт
                    </button>
                </div>
            </div>
        `;
    }

    loadTradesData(trades) {
        const tableBody = document.getElementById('tradesTableBody');
        
        if (!trades || trades.length === 0) {
            tableBody.innerHTML = `
                <tr>
                    <td colspan="10" class="no-data">
                        <i class="fas fa-chart-line"></i>
                        <p>Нет данных о сделках</p>
                    </td>
                </tr>
            `;
            return;
        }

        tableBody.innerHTML = '';
        
        trades.forEach((trade, index) => {
            const row = document.createElement('tr');
            const entryTime = new Date(trade.entry_time / 1000000).toLocaleString();
            const exitTime = new Date(trade.exit_time / 1000000).toLocaleString();
            const pnlClass = trade.net_pnl >= 0 ? 'profit' : 'loss';
            
            row.innerHTML = `
                <td>${index + 1}</td>
                <td>${entryTime}</td>
                <td>${exitTime}</td>
                <td><span class="trade-type trade-${trade.signal_type.toLowerCase()}">${trade.signal_type}</span></td>
                <td>$${trade.entry_price.toFixed(2)}</td>
                <td>$${trade.exit_price.toFixed(2)}</td>
                <td>${trade.quantity.toFixed(4)}</td>
                <td class="${pnlClass}">$${trade.net_pnl > 0 ? '+' : ''}${trade.net_pnl.toFixed(2)}</td>
                <td>${(trade.confidence * 100).toFixed(1)}%</td>
                <td>
                    <button class="btn btn-small btn-secondary" onclick="tradingBot.showTradeDetails(${index})">
                        <i class="fas fa-eye"></i> Детали
                    </button>
                </td>
            `;
            
            tableBody.appendChild(row);
        });
        
        // Store trades data for details modal
        this.currentTrades = trades;
    }

    // Button Events
    bindButtonEvents() {
        // Refresh buttons
        document.getElementById('refreshBtn')?.addEventListener('click', () => {
            this.refreshCurrentPanel();
        });
        
        document.getElementById('refreshDataBtn')?.addEventListener('click', () => {
            this.loadAvailableDatasets();
        });
        
        document.getElementById('refreshModelsBtn')?.addEventListener('click', () => {
            this.loadModels();
        });
        
        document.getElementById('refreshTradesBtn')?.addEventListener('click', () => {
            this.refreshTradesTable();
        });
        
        // Export buttons
        document.getElementById('exportTradesBtn')?.addEventListener('click', () => {
            this.exportTrades();
        });

        // Data collection buttons
        document.getElementById('collectionHistoryBtn')?.addEventListener('click', () => {
            this.toggleCollectionHistory();
        });

        // Quick period buttons for data collection
        document.querySelectorAll('[data-period]').forEach(button => {
            button.addEventListener('click', (e) => {
                e.preventDefault();
                this.setQuickPeriod(e.target.dataset.period);
            });
        });
    }

    // Range Sliders
    bindRangeSliders() {
        const testSizeSlider = document.getElementById('testSize');
        const testSizeValue = document.getElementById('testSizeValue');
        
        if (testSizeSlider && testSizeValue) {
            testSizeSlider.addEventListener('input', (e) => {
                testSizeValue.textContent = `${Math.round(e.target.value * 100)}%`;
            });
        }
        
        const confidenceSlider = document.getElementById('confidenceThreshold');
        const confidenceValue = document.getElementById('confidenceValue');
        
        if (confidenceSlider && confidenceValue) {
            confidenceSlider.addEventListener('input', (e) => {
                confidenceValue.textContent = `${Math.round(e.target.value * 100)}%`;
            });
        }
        
        const positionSlider = document.getElementById('maxPositionSize');
        const positionValue = document.getElementById('positionSizeValue');
        
        if (positionSlider && positionValue) {
            positionSlider.addEventListener('input', (e) => {
                positionValue.textContent = `${e.target.value}%`;
            });
        }
    }

    // Modal Events
    bindModalEvents() {
        const modal = document.getElementById('tradeDetailModal');
        const closeBtn = modal?.querySelector('.close');
        
        if (closeBtn) {
            closeBtn.addEventListener('click', () => {
                modal.style.display = 'none';
            });
        }
        
        window.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.style.display = 'none';
            }
        });
    }

    showTradeDetails(tradeIndex) {
        if (!this.currentTrades || !this.currentTrades[tradeIndex]) {
            return;
        }
        
        const trade = this.currentTrades[tradeIndex];
        const modal = document.getElementById('tradeDetailModal');
        const content = document.getElementById('tradeDetailContent');
        
        const entryTime = new Date(trade.entry_time / 1000000);
        const exitTime = new Date(trade.exit_time / 1000000);
        const duration = ((trade.exit_time - trade.entry_time) / 1000000000 / 60).toFixed(0); // minutes
        
        content.innerHTML = `
            <div class="trade-detail-grid">
                <div class="detail-section">
                    <h3><i class="fas fa-info-circle"></i> Основная Информация</h3>
                    <div class="detail-row">
                        <span class="label">ID Сделки:</span>
                        <span class="value">#${tradeIndex + 1}</span>
                    </div>
                    <div class="detail-row">
                        <span class="label">Тип:</span>
                        <span class="value">
                            <span class="trade-type trade-${trade.signal_type.toLowerCase()}">
                                ${trade.signal_type}
                            </span>
                        </span>
                    </div>
                    <div class="detail-row">
                        <span class="label">Время входа:</span>
                        <span class="value">${entryTime.toLocaleString()}</span>
                    </div>
                    <div class="detail-row">
                        <span class="label">Время выхода:</span>
                        <span class="value">${exitTime.toLocaleString()}</span>
                    </div>
                    <div class="detail-row">
                        <span class="label">Длительность:</span>
                        <span class="value">${duration} минут</span>
                    </div>
                </div>
                
                <div class="detail-section">
                    <h3><i class="fas fa-chart-line"></i> Цены и Объемы</h3>
                    <div class="detail-row">
                        <span class="label">Цена входа:</span>
                        <span class="value">$${trade.entry_price.toFixed(4)}</span>
                    </div>
                    <div class="detail-row">
                        <span class="label">Цена выхода:</span>
                        <span class="value">$${trade.exit_price.toFixed(4)}</span>
                    </div>
                    <div class="detail-row">
                        <span class="label">Количество:</span>
                        <span class="value">${trade.quantity.toFixed(6)}</span>
                    </div>
                    <div class="detail-row">
                        <span class="label">Изменение цены:</span>
                        <span class="value ${trade.entry_price < trade.exit_price ? 'profit' : 'loss'}">
                            ${((trade.exit_price - trade.entry_price) / trade.entry_price * 100).toFixed(2)}%
                        </span>
                    </div>
                </div>
                
                <div class="detail-section">
                    <h3><i class="fas fa-coins"></i> P&L Анализ</h3>
                    <div class="detail-row">
                        <span class="label">Валовая прибыль:</span>
                        <span class="value ${trade.pnl >= 0 ? 'profit' : 'loss'}">
                            $${trade.pnl > 0 ? '+' : ''}${trade.pnl.toFixed(4)}
                        </span>
                    </div>
                    <div class="detail-row">
                        <span class="label">Комиссия:</span>
                        <span class="value loss">-$${trade.commission.toFixed(4)}</span>
                    </div>
                    <div class="detail-row">
                        <span class="label">Чистая прибыль:</span>
                        <span class="value ${trade.net_pnl >= 0 ? 'profit' : 'loss'}">
                            $${trade.net_pnl > 0 ? '+' : ''}${trade.net_pnl.toFixed(4)}
                        </span>
                    </div>
                    <div class="detail-row">
                        <span class="label">ROI:</span>
                        <span class="value ${trade.net_pnl >= 0 ? 'profit' : 'loss'}">
                            ${((trade.net_pnl / (trade.quantity * trade.entry_price)) * 100).toFixed(2)}%
                        </span>
                    </div>
                </div>
                
                <div class="detail-section">
                    <h3><i class="fas fa-brain"></i> ML Анализ</h3>
                    <div class="detail-row">
                        <span class="label">Уверенность модели:</span>
                        <span class="value">${(trade.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div class="detail-row">
                        <span class="label">Причина выхода:</span>
                        <span class="value">${trade.exit_reason}</span>
                    </div>
                    <div class="detail-row">
                        <span class="label">Размер позиции:</span>
                        <span class="value">${trade.position_size_type}</span>
                    </div>
                    ${trade.reasoning ? `
                    <div class="detail-row">
                        <span class="label">Обоснование:</span>
                        <span class="value reasoning">${trade.reasoning}</span>
                    </div>
                    ` : ''}
                </div>
            </div>
        `;
        
        modal.style.display = 'block';
    }

    // API Calls (Mock Implementation)
    async apiCall(method, endpoint, data = null) {
        // This is a mock implementation
        // In real application, this would make actual HTTP requests
        
        console.log(`API Call: ${method} ${endpoint}`, data);
        
        // Simulate network delay
        await new Promise(resolve => setTimeout(resolve, 100));
        
        // Mock responses
        switch (endpoint) {
            case '/api/dashboard':
                return {
                    activeModels: 2,
                    mlAccuracy: 98.90,
                    lastPnl: 7.09,
                    lastTraining: '22.08.2025 17:54'
                };
            
            case '/api/models':
                return [
                    {
                        id: '20250822_175417',
                        name: '20250822_175417',
                        accuracy: { catboost: 98.90, random_forest: 98.90 },
                        features: 22,
                        created: '2025-08-22T17:54:17Z',
                        active: true
                    },
                    {
                        id: '20250822_141010',
                        name: '20250822_141010',
                        accuracy: { random_forest: 68.15, gradient_boosting: 76.17 },
                        features: 23,
                        created: '2025-08-22T14:10:10Z',
                        active: false
                    }
                ];
            
            default:
                if (endpoint.includes('/training/') && endpoint.includes('/status')) {
                    return this.mockTrainingStatus();
                } else if (endpoint.includes('/backtest/') && endpoint.includes('/status')) {
                    return this.mockBacktestStatus();
                } else if (endpoint.includes('/data-collection/') && endpoint.includes('/status')) {
                    return this.mockDataCollectionStatus();
                }
                
                throw new Error(`Unknown endpoint: ${endpoint}`);
        }
    }

    async startTraining(config) {
        console.log('Starting training with config:', config);
        const trainingId = 'training_' + Date.now();
        
        // In real implementation, this would call Python training script
        // For now, simulate starting the process
        
        return trainingId;
    }

    async startBacktest(config) {
        console.log('Starting backtest with config:', config);
        const backtestId = 'backtest_' + Date.now();
        
        // In real implementation, this would call Python backtest script
        
        return backtestId;
    }

    async startDataCollection(params) {
        try {
            const response = await this.apiCall('POST', '/api/data-collection/start', params);
            return response.process_id;
        } catch (error) {
            console.error('Error starting data collection:', error);
            throw error;
        }
    }

    mockTrainingStatus() {
        // Simulate training progress
        const now = Date.now();
        const elapsed = now - (this.mockTrainingStart || now);
        const totalTime = 60000; // 1 minute for mock
        const progress = Math.min((elapsed / totalTime) * 100, 100);
        
        if (!this.mockTrainingStart) {
            this.mockTrainingStart = now;
        }
        
        const messages = [
            'Подготовка данных...',
            'Расчет признаков...',
            'Обучение CatBoost...',
            'Обучение Random Forest...',
            'Валидация моделей...',
            'Сохранение результатов...'
        ];
        
        const messageIndex = Math.floor((progress / 100) * messages.length);
        const message = messages[Math.min(messageIndex, messages.length - 1)];
        
        if (progress >= 100) {
            delete this.mockTrainingStart;
            return {
                status: 'completed',
                progress: 100,
                message: 'Обучение завершено!',
                results: {
                    models: ['catboost', 'random_forest'],
                    accuracy: { catboost: 98.90, random_forest: 98.90 },
                    version: 'mock_' + Date.now()
                }
            };
        }
        
        return {
            status: 'running',
            progress: progress,
            message: message
        };
    }

    mockBacktestStatus() {
        // Simulate backtest completion after some time
        const now = Date.now();
        const elapsed = now - (this.mockBacktestStart || now);
        const totalTime = 30000; // 30 seconds for mock
        
        if (!this.mockBacktestStart) {
            this.mockBacktestStart = now;
        }
        
        if (elapsed >= totalTime) {
            delete this.mockBacktestStart;
            return {
                status: 'completed',
                results: this.generateMockBacktestResults()
            };
        }
        
        return {
            status: 'running',
            progress: (elapsed / totalTime) * 100
        };
    }

    generateMockBacktestResults() {
        // Generate realistic mock backtest results
        return {
            initial_balance: 10000,
            final_balance: 10075.50,
            winning_trades: 5,
            losing_trades: 7,
            trades: [
                {
                    signal_type: 'BUY',
                    entry_time: Date.now() * 1000000 - 3600000000000,
                    exit_time: Date.now() * 1000000 - 3000000000000,
                    entry_price: 185.50,
                    exit_price: 187.20,
                    quantity: 0.5405,
                    pnl: 0.92,
                    net_pnl: 0.72,
                    commission: 0.20,
                    exit_reason: 'take_profit',
                    confidence: 0.67,
                    reasoning: 'Strong BUY signal with high confidence',
                    position_size_type: 'medium'
                }
                // Add more mock trades...
            ]
        };
    }

    mockDataCollectionStatus() {
        // Simulate data collection progress
        const now = Date.now();
        const elapsed = now - (this.mockDataCollectionStart || now);
        const totalTime = 45000; // 45 seconds for mock
        const progress = Math.min((elapsed / totalTime) * 100, 100);
        
        if (!this.mockDataCollectionStart) {
            this.mockDataCollectionStart = now;
        }
        
        const stages = [
            { message: 'Подключение к API...', candles: 0 },
            { message: 'Сбор 1m данных...', candles: 1440 },
            { message: 'Сбор 5m данных...', candles: 1728 },
            { message: 'Сбор 15m данных...', candles: 1824 },
            { message: 'Сбор 1h данных...', candles: 1896 },
            { message: 'Сохранение файлов...', candles: 1920 }
        ];
        
        const stageIndex = Math.floor((progress / 100) * stages.length);
        const currentStage = stages[Math.min(stageIndex, stages.length - 1)];
        
        if (progress >= 100) {
            delete this.mockDataCollectionStart;
            return {
                status: 'completed',
                progress: 100,
                message: 'Сбор данных завершен!',
                results: {
                    symbol: 'SOLUSDT',
                    period: '2025-08-20 - 2025-08-22',
                    timeframes: ['1m', '5m', '15m', '1h'],
                    candles_collected: 1920,
                    total_size: 245760, // ~240KB
                    collection_time: 45,
                    files_created: [
                        'data/SOLUSDT_1m_20250820_20250822.csv',
                        'data/SOLUSDT_5m_20250820_20250822.csv',
                        'data/SOLUSDT_15m_20250820_20250822.csv',
                        'data/SOLUSDT_1h_20250820_20250822.csv'
                    ]
                }
            };
        }
        
        return {
            status: 'running',
            progress: progress,
            message: currentStage.message,
            results: {
                timeframes: ['1m', '5m', '15m', '1h'],
                candles_collected: Math.floor(currentStage.candles * (progress / 100))
            }
        };
    }

    // Utility Methods
    showNotification(message, type = 'info') {
        console.log(`Notification (${type}): ${message}`);
        
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <i class="fas fa-${this.getNotificationIcon(type)}"></i>
                <span>${message}</span>
            </div>
        `;
        
        // Add to page
        document.body.appendChild(notification);
        
        // Remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 5000);
    }

    getNotificationIcon(type) {
        const icons = {
            success: 'check-circle',
            error: 'exclamation-circle',
            warning: 'exclamation-triangle',
            info: 'info-circle'
        };
        return icons[type] || icons.info;
    }

    refreshCurrentPanel() {
        this.loadPanelData(this.currentPanel);
        this.showNotification('Данные обновлены', 'success');
    }

    loadPanelData(panelId) {
        switch (panelId) {
            case 'dashboard':
                this.loadDashboardData();
                break;
            case 'data-manager':
                this.loadAvailableDatasets();
                break;
            case 'models':
                this.loadModels();
                break;
            case 'trades':
                this.refreshTradesTable();
                break;
        }
    }

    refreshTradesTable() {
        // Reload current trades data
        if (this.currentTrades) {
            this.loadTradesData(this.currentTrades);
        }
    }

    exportTrades() {
        if (!this.currentTrades || this.currentTrades.length === 0) {
            this.showNotification('Нет данных для экспорта', 'warning');
            return;
        }
        
        // Convert trades to CSV
        const csvContent = this.tradesToCSV(this.currentTrades);
        
        // Create download link
        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `trades_${new Date().toISOString().split('T')[0]}.csv`;
        link.click();
        
        window.URL.revokeObjectURL(url);
        this.showNotification('Данные экспортированы', 'success');
    }

    tradesToCSV(trades) {
        const headers = [
            'ID', 'Entry Time', 'Exit Time', 'Type', 'Entry Price', 
            'Exit Price', 'Quantity', 'Gross PnL', 'Net PnL', 'Commission', 
            'Confidence', 'Exit Reason', 'Duration (min)'
        ];
        
        const csvRows = [headers.join(',')];
        
        trades.forEach((trade, index) => {
            const entryTime = new Date(trade.entry_time / 1000000).toISOString();
            const exitTime = new Date(trade.exit_time / 1000000).toISOString();
            const duration = ((trade.exit_time - trade.entry_time) / 1000000000 / 60).toFixed(0);
            
            const row = [
                index + 1,
                entryTime,
                exitTime,
                trade.signal_type,
                trade.entry_price.toFixed(4),
                trade.exit_price.toFixed(4),
                trade.quantity.toFixed(6),
                trade.pnl.toFixed(4),
                trade.net_pnl.toFixed(4),
                trade.commission.toFixed(4),
                (trade.confidence * 100).toFixed(1),
                trade.exit_reason,
                duration
            ].join(',');
            
            csvRows.push(row);
        });
        
        return csvRows.join('\n');
    }

    startAutoRefresh() {
        if (this.autoRefresh) {
            this.refreshTimer = setInterval(() => {
                this.refreshCurrentPanel();
            }, this.refreshInterval);
        }
    }

    stopAutoRefresh() {
        if (this.refreshTimer) {
            clearInterval(this.refreshTimer);
            this.refreshTimer = null;
        }
    }

    logTrainingResults(results) {
        console.log('Training Results:', results);
        // Add to training logs
        const logsContainer = document.getElementById('trainingLogs');
        if (logsContainer && !logsContainer.querySelector('.logs-placeholder')) {
            const logEntry = document.createElement('div');
            logEntry.className = 'log-entry';
            logEntry.innerHTML = `
                <span class="log-time">[${new Date().toLocaleTimeString()}]</span>
                <span class="log-message">Training completed successfully!</span>
            `;
            logsContainer.appendChild(logEntry);
        }
    }
}

// CSS for notifications (add to styles.css if needed)
const notificationStyles = `
.notification {
    position: fixed;
    top: 20px;
    right: 20px;
    background: white;
    border-radius: 8px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    z-index: 3000;
    min-width: 300px;
    animation: slideInRight 0.3s ease;
}

@keyframes slideInRight {
    from { transform: translateX(100%); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
}

.notification-content {
    display: flex;
    align-items: center;
    padding: 15px 20px;
}

.notification-content i {
    margin-right: 12px;
    font-size: 1.2em;
}

.notification-success { border-left: 4px solid #27ae60; }
.notification-success i { color: #27ae60; }

.notification-error { border-left: 4px solid #e74c3c; }
.notification-error i { color: #e74c3c; }

.notification-warning { border-left: 4px solid #f39c12; }
.notification-warning i { color: #f39c12; }

.notification-info { border-left: 4px solid #3498db; }
.notification-info i { color: #3498db; }

.trade-detail-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 30px;
}

.detail-section {
    background: #f8f9fa;
    padding: 20px;
    border-radius: 8px;
}

.detail-section h3 {
    margin-bottom: 15px;
    color: #2c3e50;
    display: flex;
    align-items: center;
}

.detail-section h3 i {
    margin-right: 10px;
    color: #3498db;
}

.detail-row {
    display: flex;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid #e9ecef;
}

.detail-row:last-child {
    border-bottom: none;
}

.detail-row .label {
    color: #7f8c8d;
    font-weight: 500;
}

.detail-row .value {
    font-weight: 600;
    color: #2c3e50;
}

.detail-row .value.profit { color: #27ae60; }
.detail-row .value.loss { color: #e74c3c; }

.detail-row .value.reasoning {
    font-style: italic;
    max-width: 200px;
    text-align: right;
}

.backtest-summary {
    padding: 20px;
}

.results-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 15px;
    margin: 20px 0;
}

.result-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 0;
    border-bottom: 1px solid #e9ecef;
}

.results-actions {
    margin-top: 20px;
    display: flex;
    gap: 10px;
}

.loading-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 40px;
    color: #7f8c8d;
}

.trade-buy { background: #27ae60; color: white; }
.trade-sell { background: #e74c3c; color: white; }
`;

// Add notification styles to head
const styleSheet = document.createElement('style');
styleSheet.textContent = notificationStyles;
document.head.appendChild(styleSheet);

// Initialize the interface when DOM is ready
let tradingBot;
document.addEventListener('DOMContentLoaded', () => {
    tradingBot = new TradingBotInterface();
    
    // Make it globally available for onclick handlers
    window.tradingBot = tradingBot;
});