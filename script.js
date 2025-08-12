// 项目数据
const projectData = {
    'house-price': {
        name: '房价预测系统',
        description: '基于机器学习的房价预测分析系统',
        files: [
            { name: 'app.py', type: 'python', size: 156, description: 'Flask主应用文件' },
            { name: 'templates/index.html', type: 'html', size: 89, description: '前端模板' },
            { name: 'static/script.js', type: 'javascript', size: 234, description: '前端交互脚本' },
            { name: 'static/style.css', type: 'css', size: 67, description: '样式文件' },
            { name: 'requirements.txt', type: 'text', size: 12, description: '依赖包列表' },
            { name: 'Exp05Bboston_house_prices.csv', type: 'csv', size: 45, description: '训练数据集' },
            { name: 'model.pkl', type: 'binary', size: 2048, description: '训练好的模型' }
        ],
        techStack: ['Python', 'Flask', 'Scikit-learn', 'Pandas', 'Matplotlib'],
        complexity: 7
    },
    'underwater': {
        name: '水下图像增强',
        description: '使用GAN网络进行水下图像质量增强',
        files: [
            { name: 'A.ipynb', type: 'jupyter', size: 345, description: '主训练脚本' },
            { name: 'generator.py', type: 'python', size: 123, description: '生成器网络' },
            { name: 'discriminator.py', type: 'python', size: 89, description: '判别器网络' },
            { name: 'dataset.py', type: 'python', size: 67, description: '数据集处理' },
            { name: 'train_data/', type: 'folder', size: 0, description: '训练数据目录' },
            { name: 'test_data/', type: 'folder', size: 0, description: '测试数据目录' },
            { name: 'checkpoints/', type: 'folder', size: 0, description: '模型检查点' },
            { name: 'results/', type: 'folder', size: 0, description: '结果输出' }
        ],
        techStack: ['PyTorch', 'OpenCV', 'NumPy', 'PIL', 'Matplotlib'],
        complexity: 9
    },
    'realtime-enhancement': {
        name: '实时图像增强',
        description: '实时摄像头图像增强系统',
        files: [
            { name: 'image_enhancer.py', type: 'python', size: 456, description: '主增强器类' },
            { name: 'lut3d.py', type: 'python', size: 234, description: '3D LUT处理' },
            { name: '系统操作手册.ini', type: 'text', size: 23, description: '使用说明' }
        ],
        techStack: ['OpenCV', 'PyTorch', 'Tkinter', 'NumPy'],
        complexity: 6
    },
    'yolo': {
        name: 'YOLO目标检测',
        description: '基于YOLOv8的实时目标检测系统',
        files: [
            { name: 'yolo.ipynb', type: 'jupyter', size: 567, description: 'YOLO检测系统' }
        ],
        techStack: ['YOLOv8', 'OpenCV', 'Tkinter', 'Pandas'],
        complexity: 8
    }
};

// 数据流定义
const dataFlows = {
    'house-price': [
        { id: 1, name: '数据上传', x: 100, y: 100 },
        { id: 2, name: '数据预处理', x: 300, y: 100 },
        { id: 3, name: '特征工程', x: 500, y: 100 },
        { id: 4, name: '模型训练', x: 700, y: 100 },
        { id: 5, name: '预测结果', x: 900, y: 100 },
        { id: 6, name: '可视化展示', x: 500, y: 250 }
    ],
    'underwater': [
        { id: 1, name: '原始图像', x: 100, y: 150 },
        { id: 2, name: '预处理', x: 250, y: 150 },
        { id: 3, name: '生成器', x: 400, y: 100 },
        { id: 4, name: '判别器', x: 400, y: 200 },
        { id: 5, name: '损失计算', x: 550, y: 150 },
        { id: 6, name: '增强图像', x: 700, y: 150 }
    ],
    'yolo': [
        { id: 1, name: '摄像头输入', x: 100, y: 150 },
        { id: 2, name: '图像预处理', x: 250, y: 150 },
        { id: 3, name: 'YOLO检测', x: 400, y: 150 },
        { id: 4, name: '后处理', x: 550, y: 150 },
        { id: 5, name: '结果显示', x: 700, y: 150 }
    ]
};

// 初始化应用
document.addEventListener('DOMContentLoaded', function() {
    initializeNavigation();
    initializeProjectCards();
    initializeCharts();
    initializeStructureView();
    initializeDataFlowView();
    initializeTimelineView();
});

// 导航功能
function initializeNavigation() {
    const navButtons = document.querySelectorAll('.nav-btn');
    const views = document.querySelectorAll('.view');

    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetView = btn.dataset.view;
            
            // 更新按钮状态
            navButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // 切换视图
            views.forEach(view => {
                view.classList.remove('active');
                if (view.id === targetView) {
                    view.classList.add('active');
                }
            });
        });
    });
}

// 项目卡片功能
function initializeProjectCards() {
    const projectCards = document.querySelectorAll('.project-card');
    const modal = document.getElementById('projectModal');
    const closeBtn = document.querySelector('.close');

    projectCards.forEach(card => {
        card.addEventListener('click', () => {
            const projectKey = card.dataset.project;
            showProjectDetails(projectKey);
        });
    });

    closeBtn.addEventListener('click', () => {
        modal.style.display = 'none';
    });

    window.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.style.display = 'none';
        }
    });
}

// 显示项目详情
function showProjectDetails(projectKey) {
    const project = projectData[projectKey];
    const modal = document.getElementById('projectModal');
    const modalContent = document.getElementById('modalContent');

    let filesHtml = project.files.map(file => `
        <div class="file-item">
            <strong>${file.name}</strong> (${file.type})
            <br><small>${file.description}</small>
            <span class="file-size">${file.size > 0 ? file.size + ' lines' : ''}</span>
        </div>
    `).join('');

    let techStackHtml = project.techStack.map(tech => 
        `<span class="tech-badge">${tech}</span>`
    ).join('');

    modalContent.innerHTML = `
        <h2>${project.name}</h2>
        <p>${project.description}</p>
        
        <h3>技术栈</h3>
        <div class="tech-stack">${techStackHtml}</div>
        
        <h3>文件结构</h3>
        <div class="file-list">${filesHtml}</div>
        
        <h3>复杂度评分</h3>
        <div class="complexity-bar">
            <div class="complexity-fill" style="width: ${project.complexity * 10}%"></div>
            <span>${project.complexity}/10</span>
        </div>
        
        <style>
            .file-item {
                padding: 10px;
                border: 1px solid #eee;
                margin: 5px 0;
                border-radius: 5px;
                position: relative;
            }
            .file-size {
                position: absolute;
                right: 10px;
                top: 10px;
                color: #666;
                font-size: 0.8em;
            }
            .tech-badge {
                background: #667eea;
                color: white;
                padding: 3px 8px;
                border-radius: 12px;
                font-size: 0.8em;
                margin: 2px;
                display: inline-block;
            }
            .tech-stack {
                margin: 10px 0;
            }
            .complexity-bar {
                background: #eee;
                height: 20px;
                border-radius: 10px;
                position: relative;
                margin: 10px 0;
            }
            .complexity-fill {
                background: linear-gradient(45deg, #667eea, #764ba2);
                height: 100%;
                border-radius: 10px;
                transition: width 0.3s ease;
            }
            .complexity-bar span {
                position: absolute;
                right: 10px;
                top: 2px;
                font-size: 0.8em;
                color: #333;
            }
        </style>
    `;

    modal.style.display = 'block';
}

// 初始化图表
function initializeCharts() {
    // 项目复杂度分析
    const complexityCtx = document.getElementById('complexityChart').getContext('2d');
    new Chart(complexityCtx, {
        type: 'radar',
        data: {
            labels: ['房价预测', '水下增强', '实时增强', 'YOLO检测'],
            datasets: [{
                label: '复杂度评分',
                data: [7, 9, 6, 8],
                backgroundColor: 'rgba(102, 126, 234, 0.2)',
                borderColor: 'rgba(102, 126, 234, 1)',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 10
                }
            }
        }
    });

    // 技术栈分布
    const techStackCtx = document.getElementById('techStackChart').getContext('2d');
    const techCount = {};
    Object.values(projectData).forEach(project => {
        project.techStack.forEach(tech => {
            techCount[tech] = (techCount[tech] || 0) + 1;
        });
    });

    new Chart(techStackCtx, {
        type: 'doughnut',
        data: {
            labels: Object.keys(techCount),
            datasets: [{
                data: Object.values(techCount),
                backgroundColor: [
                    '#667eea', '#764ba2', '#f093fb', '#f5576c',
                    '#4facfe', '#00f2fe', '#43e97b', '#38f9d7'
                ]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false
        }
    });

    // 文件类型分布
    const fileTypeCtx = document.getElementById('fileTypeChart').getContext('2d');
    const fileTypeCount = {};
    Object.values(projectData).forEach(project => {
        project.files.forEach(file => {
            fileTypeCount[file.type] = (fileTypeCount[file.type] || 0) + 1;
        });
    });

    new Chart(fileTypeCtx, {
        type: 'bar',
        data: {
            labels: Object.keys(fileTypeCount),
            datasets: [{
                label: '文件数量',
                data: Object.values(fileTypeCount),
                backgroundColor: 'rgba(102, 126, 234, 0.8)'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });

    // 项目规模对比
    const projectSizeCtx = document.getElementById('projectSizeChart').getContext('2d');
    new Chart(projectSizeCtx, {
        type: 'line',
        data: {
            labels: Object.keys(projectData),
            datasets: [{
                label: '文件数量',
                data: Object.values(projectData).map(p => p.files.length),
                borderColor: 'rgba(102, 126, 234, 1)',
                backgroundColor: 'rgba(102, 126, 234, 0.1)',
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

// 初始化结构视图
function initializeStructureView() {
    const projectSelect = document.getElementById('projectSelect');
    
    projectSelect.addEventListener('change', (e) => {
        const selectedProject = e.target.value;
        drawProjectStructure(selectedProject);
    });

    // 默认显示第一个项目
    drawProjectStructure('house-price');
}

// 绘制项目结构树
function drawProjectStructure(projectKey) {
    const container = document.getElementById('tree-visualization');
    container.innerHTML = '';

    const project = projectData[projectKey];
    const width = container.clientWidth;
    const height = 500;

    const svg = d3.select('#tree-visualization')
        .append('svg')
        .attr('width', width)
        .attr('height', height);

    // 创建树形布局
    const treeData = {
        name: project.name,
        children: project.files.map(file => ({
            name: file.name,
            type: file.type,
            description: file.description
        }))
    };

    const root = d3.hierarchy(treeData);
    const treeLayout = d3.tree().size([width - 100, height - 100]);
    treeLayout(root);

    // 绘制连接线
    svg.selectAll('.link')
        .data(root.links())
        .enter()
        .append('path')
        .attr('class', 'link')
        .attr('d', d3.linkHorizontal()
            .x(d => d.y + 50)
            .y(d => d.x + 50));

    // 绘制节点
    const nodes = svg.selectAll('.node')
        .data(root.descendants())
        .enter()
        .append('g')
        .attr('class', 'node')
        .attr('transform', d => `translate(${d.y + 50}, ${d.x + 50})`);

    nodes.append('circle')
        .attr('r', d => d.depth === 0 ? 8 : 5)
        .style('fill', d => d.depth === 0 ? '#667eea' : '#764ba2');

    nodes.append('text')
        .attr('dx', d => d.depth === 0 ? -10 : 10)
        .attr('dy', 3)
        .style('text-anchor', d => d.depth === 0 ? 'end' : 'start')
        .text(d => d.data.name)
        .style('font-size', d => d.depth === 0 ? '14px' : '12px')
        .style('font-weight', d => d.depth === 0 ? 'bold' : 'normal');

    // 添加工具提示
    nodes.append('title')
        .text(d => d.data.description || d.data.name);
}

// 初始化数据流视图
function initializeDataFlowView() {
    const flowButtons = document.querySelectorAll('.flow-btn');
    
    flowButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            flowButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            const flowType = btn.dataset.flow;
            drawDataFlow(flowType);
        });
    });

    // 默认显示第一个流程
    drawDataFlow('house-price');
}

// 绘制数据流图
function drawDataFlow(flowType) {
    const container = document.getElementById('flow-diagram');
    container.innerHTML = '';

    const width = container.clientWidth;
    const height = 400;

    const svg = d3.select('#flow-diagram')
        .append('svg')
        .attr('width', width)
        .attr('height', height);

    // 添加箭头标记
    svg.append('defs').append('marker')
        .attr('id', 'arrowhead')
        .attr('viewBox', '-0 -5 10 10')
        .attr('refX', 13)
        .attr('refY', 0)
        .attr('orient', 'auto')
        .attr('markerWidth', 13)
        .attr('markerHeight', 13)
        .attr('xoverflow', 'visible')
        .append('svg:path')
        .attr('d', 'M 0,-5 L 10 ,0 L 0,5')
        .attr('fill', '#333')
        .style('stroke', 'none');

    const nodes = dataFlows[flowType];
    
    // 绘制连接线
    for (let i = 0; i < nodes.length - 1; i++) {
        if (flowType === 'house-price' && i === 3) {
            // 房价预测流程中从模型训练到可视化的连接
            svg.append('line')
                .attr('class', 'flow-arrow')
                .attr('x1', nodes[i].x + 50)
                .attr('y1', nodes[i].y + 15)
                .attr('x2', nodes[5].x + 50)
                .attr('y2', nodes[5].y - 15);
        } else if (flowType === 'underwater' && (i === 1 || i === 2)) {
            // 水下增强中的特殊连接
            if (i === 1) {
                // 预处理到生成器
                svg.append('line')
                    .attr('class', 'flow-arrow')
                    .attr('x1', nodes[i].x + 75)
                    .attr('y1', nodes[i].y)
                    .attr('x2', nodes[2].x - 25)
                    .attr('y2', nodes[2].y);
                // 预处理到判别器
                svg.append('line')
                    .attr('class', 'flow-arrow')
                    .attr('x1', nodes[i].x + 75)
                    .attr('y1', nodes[i].y)
                    .attr('x2', nodes[3].x - 25)
                    .attr('y2', nodes[3].y);
            } else if (i === 2) {
                // 生成器到损失计算
                svg.append('line')
                    .attr('class', 'flow-arrow')
                    .attr('x1', nodes[i].x + 75)
                    .attr('y1', nodes[i].y)
                    .attr('x2', nodes[4].x - 25)
                    .attr('y2', nodes[4].y);
            }
        } else if (i < nodes.length - 1 && !(flowType === 'house-price' && i === 4)) {
            svg.append('line')
                .attr('class', 'flow-arrow')
                .attr('x1', nodes[i].x + 75)
                .attr('y1', nodes[i].y)
                .attr('x2', nodes[i + 1].x - 25)
                .attr('y2', nodes[i + 1].y);
        }
    }

    // 特殊连接处理
    if (flowType === 'underwater') {
        // 判别器到损失计算
        svg.append('line')
            .attr('class', 'flow-arrow')
            .attr('x1', nodes[3].x + 75)
            .attr('y1', nodes[3].y)
            .attr('x2', nodes[4].x - 25)
            .attr('y2', nodes[4].y);
        // 损失计算到增强图像
        svg.append('line')
            .attr('class', 'flow-arrow')
            .attr('x1', nodes[4].x + 75)
            .attr('y1', nodes[4].y)
            .attr('x2', nodes[5].x - 25)
            .attr('y2', nodes[5].y);
    }

    // 绘制节点
    const nodeGroups = svg.selectAll('.flow-node-group')
        .data(nodes)
        .enter()
        .append('g')
        .attr('class', 'flow-node-group');

    nodeGroups.append('rect')
        .attr('class', 'flow-node')
        .attr('x', d => d.x - 25)
        .attr('y', d => d.y - 15)
        .attr('width', 100)
        .attr('height', 30);

    nodeGroups.append('text')
        .attr('class', 'flow-text')
        .attr('x', d => d.x + 25)
        .attr('y', d => d.y)
        .text(d => d.name);
}

// 初始化时间线视图
function initializeTimelineView() {
    const container = document.getElementById('timeline-visualization');
    
    // 模拟开发时间线数据
    const timelineData = [
        { project: '房价预测系统', start: '2024-01', end: '2024-03', phase: '开发完成' },
        { project: '台风预测模型', start: '2024-02', end: '2024-04', phase: '研究阶段' },
        { project: '水下图像增强', start: '2024-03', end: '2024-06', phase: '训练优化' },
        { project: '实时图像增强', start: '2024-05', end: '2024-07', phase: '功能完善' },
        { project: 'YOLO目标检测', start: '2024-06', end: '2024-08', phase: '集成测试' }
    ];

    let timelineHtml = '<div class="timeline">';
    timelineData.forEach((item, index) => {
        timelineHtml += `
            <div class="timeline-item" style="animation-delay: ${index * 0.2}s">
                <div class="timeline-marker"></div>
                <div class="timeline-content">
                    <h4>${item.project}</h4>
                    <p>开发周期: ${item.start} - ${item.end}</p>
                    <span class="timeline-phase">${item.phase}</span>
                </div>
            </div>
        `;
    });
    timelineHtml += '</div>';

    container.innerHTML = timelineHtml + `
        <style>
            .timeline {
                position: relative;
                padding: 20px 0;
            }
            .timeline::before {
                content: '';
                position: absolute;
                left: 30px;
                top: 0;
                bottom: 0;
                width: 2px;
                background: #667eea;
            }
            .timeline-item {
                position: relative;
                margin: 30px 0;
                padding-left: 80px;
                opacity: 0;
                animation: slideInLeft 0.6s ease-out forwards;
            }
            .timeline-marker {
                position: absolute;
                left: -39px;
                top: 5px;
                width: 20px;
                height: 20px;
                background: #667eea;
                border: 4px solid white;
                border-radius: 50%;
                box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            }
            .timeline-content {
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                border-left: 4px solid #667eea;
            }
            .timeline-content h4 {
                margin: 0 0 10px 0;
                color: #333;
            }
            .timeline-content p {
                margin: 5px 0;
                color: #666;
            }
            .timeline-phase {
                background: linear-gradient(45deg, #667eea, #764ba2);
                color: white;
                padding: 4px 12px;
                border-radius: 15px;
                font-size: 0.8em;
                font-weight: 500;
            }
            @keyframes slideInLeft {
                from {
                    opacity: 0;
                    transform: translateX(-50px);
                }
                to {
                    opacity: 1;
                    transform: translateX(0);
                }
            }
        </style>
    `;
}