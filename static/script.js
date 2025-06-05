let selectedModel = null;
let uploadedData = null;
let selectedColumns = new Set();
let modelData = {
    sercan: null,
    suheyla: null,
    atakan: null
};

window.onload = async function () {
    console.log('Sayfa yüklendi!');
    initializeEventListeners();

    try {
        const response = await axios.get('http://localhost:5000/last-data');
        if (response.data && response.data.length > 0) {
            uploadedData = response.data;
            displayDataPreview(uploadedData);
            console.log('✅ Önceki veri yeniden yüklendi.');
        }
    } catch (error) {
        console.warn('⛔ Son veri yüklenemedi:', error);
    }
};


function initializeEventListeners() {
    console.log('Event listeners başlatılıyor...');

    document.querySelectorAll('.model-card').forEach(card => {
        card.addEventListener('click', function (event) {
            event.preventDefault();
            event.stopPropagation();
            const modelType = this.dataset.model;
            console.log('Model kartına tıklandı:', modelType);
            selectModel(modelType);
        });
    });

    const fileInput = document.getElementById('fileInput');
    const uploadArea = document.getElementById('uploadArea');
    const fileSelectBtn = document.getElementById('fileSelectBtn');

    if (fileInput) {
        fileInput.addEventListener('change', function (event) {
            event.preventDefault();
            event.stopPropagation();
            console.log('Dosya seçildi:', event.target.files[0]);
            if (event.target.files[0]) {
                handleFileSelect(event);
            }
        });
    }

    if (fileSelectBtn) {
        fileSelectBtn.addEventListener('click', function (event) {
            event.preventDefault();
            event.stopPropagation();
            fileInput.click();
        });
    }

    if (uploadArea) {
        uploadArea.addEventListener('click', function (event) {
            if (event.target === this || event.target.classList.contains('upload-icon')) {
                fileInput.click();
            }
        });

        uploadArea.addEventListener('dragover', function (event) {
            event.preventDefault();
            event.stopPropagation();
            this.classList.add('dragover');
        });

        uploadArea.addEventListener('drop', function (event) {
            event.preventDefault();
            event.stopPropagation();
            this.classList.remove('dragover');
            const files = event.dataTransfer.files;
            if (files.length > 0) {
                console.log('Dosya sürüklendi:', files[0]);
                processFile(files[0]);
            }
        });

        uploadArea.addEventListener('dragleave', function (event) {
            this.classList.remove('dragover');
        });
    }

    const analyzeBtn = document.getElementById('analyzeBtn');
    if (analyzeBtn) {
        let isAnalyzing = false;

        analyzeBtn.addEventListener('click', async function (event) {
            event.preventDefault();
            event.stopPropagation();

            if (isAnalyzing) {
                console.log('Analiz zaten devam ediyor');
                return;
            }

            if (this.disabled) {
                console.log('Button disabled');
                return;
            }

            isAnalyzing = true;
            this.disabled = true;
            this.classList.add('loading');
            this.textContent = 'Analiz yapılıyor...';

            try {
                await startAnalysis();
            } finally {
                isAnalyzing = false;
                this.disabled = false;
                this.classList.remove('loading');
                checkReadyState(); 
            }
        });
    }
}

function selectModel(modelType) {
    console.log('Model seçildi:', modelType);
    selectedModel = modelType;

    document.querySelectorAll('.model-card').forEach(card => {
        card.classList.remove('selected');
    });

    const selectedCard = document.querySelector(`[data-model="${modelType}"]`);
    if (selectedCard) {
        selectedCard.classList.add('selected');
        console.log('Model kartı seçildi:', modelType);
    }

    checkReadyState();
}

function handleFileSelect(event) {
   
    event.preventDefault();
    event.stopPropagation();
    event.stopImmediatePropagation();

    console.log('handleFileSelect çağırıldı');
    const file = event.target.files[0];
    if (file) {
        console.log('Dosya bulundu:', file.name);
        processFile(file);
    } else {
        console.log('Dosya bulunamadı');
    }
    return false;
}
async function processFile(file) {
    console.log('Dosya işleniyor:', file.name);

    try {
        const fileExtension = file.name.split('.').pop().toLowerCase();
        let data;

        console.log('Dosya uzantısı:', fileExtension);

        if (fileExtension === 'csv' || fileExtension === 'txt') {
            console.log('CSV dosyası işleniyor...');
            data = await parseCSV(file);
        } else if (fileExtension === 'xlsx' || fileExtension === 'xls') {
            console.log('Excel dosyası işleniyor...');
            data = await parseExcel(file);
        }

        console.log('Veri başarıyla işlendi:', data.length, 'satır');
        uploadedData = data;
        displayDataPreview(data);
        checkReadyState();

    } catch (error) {
        console.error('Dosya işleme hatası:', error);
        showError('Dosya işlenirken hata oluştu: ' + error.message);
    }

    setTimeout(async () => {
        try {
            const formData = new FormData();
            formData.append('file', file);
            const response = await axios.post('http://localhost:5000/upload', formData);
            console.log('✅ Backend cevabı:', response.data);
        } catch (error) {
            console.error('❌ Backend hatası:', error);
        }
    }, 100); 
}


function parseCSV(file) {
    return new Promise((resolve, reject) => {
        Papa.parse(file, {
            header: true,
            dynamicTyping: true,
            skipEmptyLines: true,
            complete: function (results) {
                if (results.errors.length > 0) {
                    reject(new Error('CSV parsing hatası'));
                } else {
                    resolve(results.data);
                }
            },
            error: function (error) {
                reject(error);
            }
        });
    });
}

async function parseExcel(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = function (e) {
            try {
                const data = new Uint8Array(e.target.result);
                const workbook = XLSX.read(data, { type: 'array' });
                const firstSheetName = workbook.SheetNames[0];
                const worksheet = workbook.Sheets[firstSheetName];
                const jsonData = XLSX.utils.sheet_to_json(worksheet);
                resolve(jsonData);
            } catch (error) {
                reject(error);
            }
        };
        reader.onerror = function () {
            reject(new Error('Dosya okunamadı'));
        };
        reader.readAsArrayBuffer(file);
    });
}

function displayDataPreview(data) {
    const preview = document.getElementById('dataPreview');
    const info = document.getElementById('dataInfo');
    const table = document.getElementById('previewTable');
    const columnSelector = document.getElementById('columnSelector');

    preview.style.display = 'block';

    info.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
            <div>
                <p><strong>📈 Toplam Satır:</strong> ${data.length}</p>
                <p><strong>📊 Toplam Sütun:</strong> ${Object.keys(data[0] || {}).length}</p>
            </div>
            <button onclick="clearUpload()" style="background: #dc3545; color: white; border: none; padding: 8px 15px; border-radius: 5px; cursor: pointer; font-size: 14px;">
                🗑️ Dosyayı Sil
            </button>
        </div>
    `;

    columnSelector.innerHTML = '';

    const columns = Object.keys(data[0] || {});
    const previewData = data.slice(0, 10);

    let tableHTML = '<thead><tr>';
    columns.forEach(column => {
        tableHTML += `<th>${column}</th>`;
    });
    tableHTML += '</tr></thead><tbody>';

    previewData.forEach(row => {
        tableHTML += '<tr>';
        columns.forEach(column => {
            const cellValue = row[column] || '-';
            const displayValue = cellValue.toString().length > 20 ?
                cellValue.toString().substring(0, 20) + '...' : cellValue;
            tableHTML += `<td title="${cellValue}">${displayValue}</td>`;
        });
        tableHTML += '</tr>';
    });

    tableHTML += '</tbody>';
    table.innerHTML = tableHTML;

    checkReadyState();
}

function checkReadyState() {
    const analyzeBtn = document.getElementById('analyzeBtn');
    const isReady = selectedModel && uploadedData;

    analyzeBtn.disabled = !isReady;
    analyzeBtn.style.opacity = isReady ? '1' : '0.6';

    if (!isReady) {
        let missing = [];
        if (!selectedModel) missing.push('Model');
        if (!uploadedData) missing.push('Veri');

        analyzeBtn.textContent = `❌ Eksik: ${missing.join(', ')}`;
    } else {
        analyzeBtn.textContent = '🚀 Analizi Başlat';
    }
}

function clearUpload() {
    uploadedData = null;
    window.selectedTarget = null;
    document.getElementById('fileInput').value = '';
    document.getElementById('dataPreview').style.display = 'none';
    checkReadyState();
    console.log('Upload temizlendi');
}



async function startAnalysis() {
    const resultsSection = document.getElementById('analysisResults');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const resultsContent = document.getElementById('resultsContent');
    const progressFill = document.getElementById('progressFill');

    resultsSection.style.display = 'block';
    loadingIndicator.style.display = 'flex';
    resultsContent.innerHTML = '';

    try {
        for (let i = 0; i <= 100; i += 10) {
            progressFill.style.width = i + '%';
            await new Promise(resolve => setTimeout(resolve, 200));
        }

        const selectedData = prepareDataForAnalysis();

        let results;
        switch (selectedModel) {
            case 'sercan':
                results = await runSercanModel(selectedData);
                break;
            case 'suheyla':
                results = await runSuheylaModel(selectedData);
                break;
            case 'atakan':
                results = await runAtakanModel(selectedData);
                break;
            default:
                throw new Error('Bilinmeyen model tipi: ' + selectedModel);
        }

        displayResults(results);

    } catch (error) {
        showError('Analiz sırasında hata oluştu: ' + error.message);
    } finally {
        loadingIndicator.style.display = 'none';
    }
}

function prepareDataForAnalysis() {
    const targetColumn = window.selectedTarget;
    return uploadedData.map(row => {
        const filteredRow = {};
        Object.keys(row).forEach(col => {
            filteredRow[col] = row[col];
        });
        return filteredRow;
    });
}

async function runSercanModel(data) {
    console.log('Sercan modeli çalışıyor...');
    await new Promise(resolve => setTimeout(resolve, 1000));

    const targetColumn = window.selectedTarget;
    const uniqueTargets = [...new Set(data.map(row => row[targetColumn]))];

    console.log('Target sütunu:', targetColumn);
    console.log('Unique target değerleri:', uniqueTargets);

    const predictions = data.map(() => uniqueTargets[Math.floor(Math.random() * uniqueTargets.length)]);

    let correctPredictions = 0;
    data.forEach((row, index) => {
        if (Math.random() < 0.8) {
            predictions[index] = row[targetColumn];
            correctPredictions++;
        }
    });

    const accuracy = (correctPredictions / data.length);
    const classDistribution = {};

    predictions.forEach(pred => {
        classDistribution[pred] = (classDistribution[pred] || 0) + 1;
    });

    return {
        type: 'classification',
        modelName: 'Sercan Model',
        accuracy: accuracy,
        predictions: predictions,
        classDistribution: classDistribution,
        totalSamples: data.length,
        targetColumn: targetColumn,
        actualVsPredicted: data.map((row, index) => ({
            actual: row[targetColumn],
            predicted: predictions[index],
            correct: row[targetColumn] === predictions[index]
        }))
    };
}

async function runSuheylaModel(data) {
    console.log('Süheyla modeli çalışıyor...');
    await new Promise(resolve => setTimeout(resolve, 800));

    const targetColumn = window.selectedTarget;
    const uniqueTargets = [...new Set(data.map(row => row[targetColumn]))];
    const predictions = data.map(() => uniqueTargets[Math.floor(Math.random() * uniqueTargets.length)]);

    const accuracy = (Math.random() * 0.2 + 0.80);
    const classDistribution = {};

    predictions.forEach(pred => {
        classDistribution[pred] = (classDistribution[pred] || 0) + 1;
    });

    return {
        type: 'classification',
        modelName: 'Suheyla Model',
        accuracy: accuracy,
        predictions: predictions,
        classDistribution: classDistribution,
        totalSamples: data.length,
        targetColumn: targetColumn,
        actualVsPredicted: data.map((row, index) => ({
            actual: row[targetColumn],
            predicted: predictions[index],
            correct: row[targetColumn] === predictions[index]
        }))
    };
}

async function runAtakanModel(data) {
    console.log('Atakan modeli çalışıyor...');
    await new Promise(resolve => setTimeout(resolve, 1200));

    const targetColumn = window.selectedTarget;
    const uniqueTargets = [...new Set(data.map(row => row[targetColumn]))];
    const predictions = data.map(() => uniqueTargets[Math.floor(Math.random() * uniqueTargets.length)]);

    const accuracy = (Math.random() * 0.25 + 0.75);
    const classDistribution = {};

    predictions.forEach(pred => {
        classDistribution[pred] = (classDistribution[pred] || 0) + 1;
    });

    return {
        type: 'classification',
        modelName: 'Atakan Model',
        accuracy: accuracy,
        predictions: predictions,
        classDistribution: classDistribution,
        totalSamples: data.length,
        targetColumn: targetColumn,
        actualVsPredicted: data.map((row, index) => ({
            actual: row[targetColumn],
            predicted: predictions[index],
            correct: row[targetColumn] === predictions[index]
        }))
    };
}

function displayResults(results) {
    const resultsContent = document.getElementById('resultsContent');
    let html = '<div class="success-message">✅ Analiz başarıyla tamamlandı!</div>';

    switch (results.type) {
        case 'classification':
            html += `
                        <h3>🎯 ${results.modelName} Sonuçları</h3>
                        <div style="background: #e8f5e8; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                            <strong>Target Sütunu:</strong> ${results.targetColumn}<br>
                        </div>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0;">
                            <div style="background: white; padding: 20px; border-radius: 10px; text-align: center;">
                                <h4>Doğruluk Oranı</h4>
                                <div style="font-size: 2rem; color: #27ae60; font-weight: bold;">${(results.accuracy * 100).toFixed(1)}%</div>
                                <small style="color: #7f8c8d;">Doğru tahmin / Toplam tahmin</small>
                            </div>
                            <div style="background: white; padding: 20px; border-radius: 10px; text-align: center;">
                                <h4>Toplam Örnek</h4>
                                <div style="font-size: 2rem; color: #3498db; font-weight: bold;">${results.totalSamples}</div>
                            </div>
                            <div style="background: white; padding: 20px; border-radius: 10px; text-align: center;">
                                <h4>Doğru Tahmin</h4>
                                <div style="font-size: 2rem; color: #27ae60; font-weight: bold;">${Math.round(results.accuracy * results.totalSamples)}</div>
                            </div>
                            <div style="background: white; padding: 20px; border-radius: 10px; text-align: center;">
                                <h4>Yanlış Tahmin</h4>
                                <div style="font-size: 2rem; color: #e74c3c; font-weight: bold;">${results.totalSamples - Math.round(results.accuracy * results.totalSamples)}</div>
                            </div>
                        </div>
                        <h4>Tahmin Dağılımı:</h4>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                    `;

            Object.entries(results.classDistribution).forEach(([className, count]) => {
                html += `
                            <div style="background: white; padding: 15px; border-radius: 8px; border-left: 4px solid #3498db;">
                                <strong>${className}:</strong> ${count} tahmin (${((count / results.totalSamples) * 100).toFixed(1)}%)
                            </div>
                        `;
            });
            html += '</div>';
            console.log("İlk 3 tahmin karşılaştırması:", results.actualVsPredicted.slice(0, 3));
            if (results.actualVsPredicted) {
                html += `
                            <h4 style="margin-top: 30px;">🔍 İlk 10 Tahmin vs Gerçek:</h4>
                            <div class="table-container">
                                <table class="preview-table">
                                    <thead>
                                        <tr>
                                            <th>Sıra</th>
                                            <th>Gerçek Değer</th>
                                            <th>Tahmin</th>
                                            <th>Sonuç</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                        `;

                results.actualVsPredicted.slice(0, 10).forEach((item, index) => {
                    console.log(item);
                    const isCorrect = item.correct;
                    const bgColor = isCorrect ? '#d5f4e6' : '#fdeaea';
                    const icon = isCorrect ? '✅' : '❌';

                    html += `
                                <tr style="background: ${bgColor};">
                                    <td>${index + 1}</td>
                                    <td><strong>${item.actual}</strong></td>
                                    <td>${item.predicted}</td>
                                    <td>${icon}</td>
                                </tr>
                            `;
                });

                html += '</tbody></table></div>';
            }
            break;
    }

    html += `
                <div style="margin-top: 30px; text-align: center;">
                   
                    <button class="btn" onclick="resetAnalysis()" style="background: linear-gradient(45deg, #95a5a6, #7f8c8d);">
                        🔄 Yeni Analiz
                    </button>
                </div>
            `;

    resultsContent.innerHTML = html;
}

function downloadResults() {
    let csvContent = "data:text/csv;charset=utf-8,";
    csvContent += "KDD Karar Destek Sistemi - Analiz Raporu\n";
    csvContent += "Tarih: " + new Date().toLocaleString('tr-TR') + "\n";
    csvContent += "Model Tipi: " + selectedModel + "\n";
    csvContent += "Target Sütunu: " + (window.selectedTarget || 'Seçilmedi') + "\n";
    csvContent += "Toplam Veri: " + uploadedData.length + " satır\n\n";

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "kdd_analiz_raporu.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

function resetAnalysis() {
    selectedModel = null;
    uploadedData = null;
    selectedColumns.clear();
    window.selectedTarget = null;

    document.querySelectorAll('.model-card').forEach(card => {
        card.classList.remove('selected');
    });

    document.getElementById('fileInput').value = '';
    document.getElementById('dataPreview').style.display = 'none';
    document.getElementById('analysisResults').style.display = 'none';
    document.getElementById('progressFill').style.width = '0%';

    checkReadyState();
}

function showError(message) {
    const resultsSection = document.getElementById('analysisResults');
    const resultsContent = document.getElementById('resultsContent');

    resultsSection.style.display = 'block';
    document.getElementById('loadingIndicator').style.display = 'none';

    resultsContent.innerHTML = `
                <div class="error-message">
                    ❌ ${message}
                </div>
            `;
}

function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}
async function runSercanModel(data) {
    console.log('Sercan modeli seçildi, backend\'e bilgi gönderiliyor...');

    try {
        const response = await axios.post('http://localhost:5000/analyze', {
            selected_model: 'sercan'
        });

        console.log('Backend cevabı:', response.data);

        if (response.data && response.data.results) {
            return formatBackendResults(response.data.results, 'Sercan Model');
        }

    } catch (error) {
        console.error('Backend hatası:', error);
    }

    return null;
}

async function runSuheylaModel(data) {
    console.log('⚡ Süheyla modeli seçildi, backend\'e bilgi gönderiliyor...');

    try {
        const response = await axios.post('http://localhost:5000/analyze', {
            selected_model: 'suheyla'
        });

        console.log('Backend cevabı:', response.data);

        if (response.data && response.data.results) {
            return formatBackendResults(response.data.results, 'Süheyla Model');
        }

    } catch (error) {
        console.error('Backend hatası:', error);
    }

    return null;
}

async function runAtakanModel(data) {
    console.log('Atakan modeli seçildi, backend\'e bilgi gönderiliyor...');

    try {
        const response = await axios.post('http://localhost:5000/analyze', {
            selected_model: 'atakan'
        });

        console.log('Backend cevabı:', response.data);

        if (response.data && response.data.results) {
            return formatBackendResults(response.data.results, 'Atakan Model');
        }

    } catch (error) {
        console.error('Backend hatası:', error);
    }

    return null;
}

function formatBackendResults(backendResults, modelName) {
    return {
        type: 'classification',
        modelName: modelName + ' ',
        accuracy: backendResults.accuracy || 0.0,
        predictions: backendResults.predictions || [],
        classDistribution: backendResults.class_distribution || {},
        totalSamples: backendResults.total_samples || 0,
        specialFeature: backendResults.description || 'Model Analizi',
        targetColumn: backendResults.target_column || 'Target',
        actualVsPredicted: backendResults.actualVsPredicted || []
    };
}

