let currentColumns = [];

// 文件上传处理
document.getElementById('uploadForm').addEventListener('submit', function(e) {
    e.preventDefault();
    const formData = new FormData();
    const fileInput = document.getElementById('fileInput');
    formData.append('file', fileInput.files[0]);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
            return;
        }
        
        // 显示数据预览
        document.getElementById('previewSection').style.display = 'block';
        document.getElementById('dataPreview').innerHTML = data.preview;
        document.getElementById('dataDescription').innerHTML = data.description;
        
        // 保存列名并设置下拉菜单
        currentColumns = data.columns;
        setupModelControls(data.columns);
        
        // 显示模型部分
        document.getElementById('modelSection').style.display = 'block';
        
        // 进行数据分析
        analyzeData();
    })
    .catch(error => {
        alert('上传失败: ' + error);
    });
});

// 设置模型控制组件
function setupModelControls(columns) {
    // 设置目标变量下拉菜单
    const targetSelect = document.getElementById('targetSelect');
    targetSelect.innerHTML = columns.map(col => 
        `<option value="${col}">${col}</option>`
    ).join('');

    // 设置特征变量复选框
    const featureCheckboxes = document.getElementById('featureCheckboxes');
    featureCheckboxes.innerHTML = columns.map(col => 
        `<div class="form-check form-check-inline">
            <input class="form-check-input" type="checkbox" value="${col}" id="feature_${col}">
            <label class="form-check-label" for="feature_${col}">${col}</label>
        </div>`
    ).join('');
}

// 数据分析
function analyzeData() {
    fetch('/analyze', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            console.error('分析错误:', data.error);
            alert(data.error);
            return;
        }
        
        // 显示分析部分
        document.getElementById('analysisSection').style.display = 'block';
        
        // 显示图表
        if (data.hist_plot) {
            document.getElementById('histPlot').innerHTML = 
                `<img src="data:image/png;base64,${data.hist_plot}" alt="直方图" title="数据分布直方图">`;
        }
        if (data.scatter_plot) {
            document.getElementById('scatterPlot').innerHTML = 
                `<img src="data:image/png;base64,${data.scatter_plot}" alt="散点图" title="特征散点图矩阵">`;
        }
        if (data.heatmap) {
            document.getElementById('heatmap').innerHTML = 
                `<img src="data:image/png;base64,${data.heatmap}" alt="热力图" title="特征相关性热力图">`;
        }
    })
    .catch(error => {
        console.error('分析失败:', error);
        alert('分析失败: ' + error);
    });
}

// 训练模型
function trainModel() {
    const features = Array.from(document.querySelectorAll('#featureCheckboxes input:checked'))
        .map(cb => cb.value);
    const target = document.getElementById('targetSelect').value;
    
    if (features.length === 0) {
        alert('请选择至少一个特征变量');
        return;
    }
    
    fetch('/train', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            feature_columns: features,
            target_column: target
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
            return;
        }
        
        alert('模型训练完成');
        document.getElementById('predictionSection').style.display = 'block';
        setupPredictionInputs(features);
    })
    .catch(error => {
        alert('模型训练失败: ' + error);
    });
}

// 设置预测输入
function setupPredictionInputs(features) {
    const container = document.getElementById('predictionInputs');
    container.innerHTML = features.map(feature => `
        <div class="mb-3">
            <label class="form-label">${feature}</label>
            <input type="number" step="any" class="form-control" id="pred_${feature}" required>
        </div>
    `).join('');
}

// 预测
function predict() {
    const features = Array.from(document.querySelectorAll('#featureCheckboxes input:checked'))
        .map(cb => cb.value);
    
    const inputData = {};
    let isValid = true;
    features.forEach(feature => {
        const value = document.getElementById(`pred_${feature}`).value;
        if (!value) {
            alert(`请输入 ${feature} 的值`);
            isValid = false;
            return;
        }
        inputData[feature] = parseFloat(value);
    });
    
    if (!isValid) return;
    
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            input_data: inputData
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
            return;
        }
        
        document.getElementById('predictionResults').innerHTML = `
            <div class="alert alert-success">
                预测结果: ${data.predictions[0].toFixed(2)}
            </div>
        `;
    })
    .catch(error => {
        alert('预测失败: ' + error);
    });
}

// AI问答
function askAI() {
    const question = document.getElementById('questionInput').value.trim();
    if (!question) {
        alert('请输入问题');
        return;
    }
    
    fetch('/ask_ai', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            question: question
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            document.getElementById('aiResponse').innerHTML = `
                <div class="alert alert-danger">${data.error}</div>
            `;
        } else {
            document.getElementById('aiResponse').innerHTML = `
                <div class="alert alert-info">${data.response}</div>
            `;
        }
    })
    .catch(error => {
        alert('AI回答失败: ' + error);
    });
}

// 辅助函数：创建表格
function createTable(data) {
    let html = '<table class="table table-bordered table-sm">';
    
    // 表头
    html += '<thead><tr>';
    for (let col in data) {
        html += `<th>${col}</th>`;
    }
    html += '</tr></thead>';
    
    // 表体
    html += '<tbody>';
    const rowCount = Object.values(data)[0].length;
    for (let i = 0; i < rowCount; i++) {
        html += '<tr>';
        for (let col in data) {
            html += `<td>${data[col][i]}</td>`;
        }
        html += '</tr>';
    }
    html += '</tbody></table>';
    
    return html;
}

// 创建缺失值表格
function createMissingValuesTable(data) {
    let html = '<table class="table table-bordered table-sm">';
    html += '<thead><tr><th>列名</th><th>缺失值数量</th></tr></thead>';
    html += '<tbody>';
    
    for (let col in data) {
        html += `<tr><td>${col}</td><td>${data[col]}</td></tr>`;
    }
    
    html += '</tbody></table>';
    return html;
}