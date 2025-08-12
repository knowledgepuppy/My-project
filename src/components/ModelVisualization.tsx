import React, { useState } from 'react';
import { Brain, Layers, TrendingUp, Zap, Settings, Play } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ScatterChart, Scatter, Cell } from 'recharts';

const ModelVisualization = () => {
  const [selectedModel, setSelectedModel] = useState('cnn');

  // 模拟训练数据
  const trainingData = [
    { epoch: 1, loss: 0.85, accuracy: 0.65, valLoss: 0.92, valAccuracy: 0.58 },
    { epoch: 5, loss: 0.45, accuracy: 0.78, valLoss: 0.52, valAccuracy: 0.72 },
    { epoch: 10, loss: 0.28, accuracy: 0.85, valLoss: 0.35, valAccuracy: 0.82 },
    { epoch: 15, loss: 0.18, accuracy: 0.89, valLoss: 0.25, valAccuracy: 0.87 },
    { epoch: 20, loss: 0.12, accuracy: 0.92, valLoss: 0.18, valAccuracy: 0.91 },
    { epoch: 25, loss: 0.08, accuracy: 0.94, valLoss: 0.15, valAccuracy: 0.93 },
  ];

  // 特征重要性数据
  const featureImportance = [
    { feature: '海表温度', importance: 0.25, x: 28.5, y: 0.25 },
    { feature: '气压', importance: 0.22, x: 1005, y: 0.22 },
    { feature: '风速', importance: 0.18, x: 45, y: 0.18 },
    { feature: '湿度', importance: 0.15, x: 75, y: 0.15 },
    { feature: '云量', importance: 0.12, x: 60, y: 0.12 },
    { feature: '海流速度', importance: 0.08, x: 2.3, y: 0.08 },
  ];

  const models = [
    { 
      id: 'cnn', 
      name: 'CNN', 
      description: '卷积神经网络，擅长处理空间数据',
      accuracy: 92.5,
      layers: ['Conv2D', 'MaxPool', 'Conv2D', 'Dense', 'Output']
    },
    { 
      id: 'lstm', 
      name: 'LSTM', 
      description: '长短期记忆网络，适合时序预测',
      accuracy: 89.3,
      layers: ['LSTM', 'Dropout', 'LSTM', 'Dense', 'Output']
    },
    { 
      id: 'transformer', 
      name: 'Transformer', 
      description: '注意力机制模型，处理复杂关系',
      accuracy: 94.1,
      layers: ['Attention', 'FFN', 'Attention', 'Dense', 'Output']
    },
  ];

  const currentModel = models.find(m => m.id === selectedModel);

  return (
    <div className="space-y-8">
      {/* 模型选择 */}
      <div className="card">
        <h2 className="text-xl font-bold text-gray-900 mb-6 flex items-center">
          <Brain className="w-6 h-6 mr-2 text-purple-600" />
          机器学习模型
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {models.map((model) => (
            <div
              key={model.id}
              onClick={() => setSelectedModel(model.id)}
              className={`p-4 rounded-lg border-2 cursor-pointer transition-all duration-200 ${
                selectedModel === model.id
                  ? 'border-purple-500 bg-purple-50'
                  : 'border-gray-200 hover:border-gray-300'
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <h3 className="font-semibold text-gray-900">{model.name}</h3>
                <div className="text-sm font-medium text-green-600">
                  {model.accuracy}%
                </div>
              </div>
              <p className="text-sm text-gray-600 mb-3">{model.description}</p>
              <div className="flex items-center space-x-2">
                <Layers className="w-4 h-4 text-gray-400" />
                <span className="text-xs text-gray-500">
                  {model.layers.length} 层网络
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* 模型架构可视化 */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
          <Layers className="w-5 h-5 mr-2 text-blue-600" />
          {currentModel?.name} 模型架构
        </h3>
        <div className="flex items-center justify-between p-6 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg">
          {currentModel?.layers.map((layer, index) => (
            <React.Fragment key={index}>
              <div className="flex flex-col items-center">
                <div className={`w-16 h-16 rounded-lg flex items-center justify-center text-white font-medium ${
                  index === 0 ? 'bg-green-500' :
                  index === currentModel.layers.length - 1 ? 'bg-red-500' :
                  'bg-blue-500'
                }`}>
                  {layer}
                </div>
                <span className="text-xs text-gray-600 mt-2">Layer {index + 1}</span>
              </div>
              {index < currentModel.layers.length - 1 && (
                <div className="flex-1 h-0.5 bg-gray-300 mx-4 relative">
                  <div className="absolute right-0 top-0 w-2 h-2 bg-gray-400 rounded-full transform -translate-y-0.5"></div>
                </div>
              )}
            </React.Fragment>
          ))}
        </div>
      </div>

      {/* 训练过程可视化 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <TrendingUp className="w-5 h-5 mr-2 text-green-600" />
            训练准确率
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={trainingData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="epoch" />
              <YAxis />
              <Tooltip />
              <Line 
                type="monotone" 
                dataKey="accuracy" 
                stroke="#10b981" 
                strokeWidth={3}
                name="训练准确率"
              />
              <Line 
                type="monotone" 
                dataKey="valAccuracy" 
                stroke="#3b82f6" 
                strokeWidth={3}
                strokeDasharray="5 5"
                name="验证准确率"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <TrendingUp className="w-5 h-5 mr-2 text-red-600" />
            损失函数
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={trainingData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="epoch" />
              <YAxis />
              <Tooltip />
              <Line 
                type="monotone" 
                dataKey="loss" 
                stroke="#ef4444" 
                strokeWidth={3}
                name="训练损失"
              />
              <Line 
                type="monotone" 
                dataKey="valLoss" 
                stroke="#f97316" 
                strokeWidth={3}
                strokeDasharray="5 5"
                name="验证损失"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* 特征重要性 */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
          <Zap className="w-5 h-5 mr-2 text-yellow-600" />
          特征重要性分析
        </h3>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div>
            <ResponsiveContainer width="100%" height={300}>
              <ScatterChart data={featureImportance}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="x" name="特征值" />
                <YAxis dataKey="y" name="重要性" />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                <Scatter dataKey="importance" fill="#8884d8">
                  {featureImportance.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={`hsl(${index * 60}, 70%, 50%)`} />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          <div className="space-y-3">
            {featureImportance.map((feature, index) => (
              <div key={index} className="flex items-center justify-between">
                <span className="text-sm font-medium text-gray-700">{feature.feature}</span>
                <div className="flex items-center space-x-2">
                  <div className="w-24 bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-gradient-to-r from-blue-500 to-purple-600 h-2 rounded-full"
                      style={{ width: `${feature.importance * 100}%` }}
                    ></div>
                  </div>
                  <span className="text-sm text-gray-600 w-12">
                    {(feature.importance * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* 模型控制面板 */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
          <Settings className="w-5 h-5 mr-2 text-gray-600" />
          模型控制面板
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <button className="btn-primary flex items-center justify-center space-x-2">
            <Play className="w-4 h-4" />
            <span>开始训练</span>
          </button>
          <button className="btn-secondary flex items-center justify-center space-x-2">
            <Settings className="w-4 h-4" />
            <span>调整参数</span>
          </button>
          <button className="btn-secondary flex items-center justify-center space-x-2">
            <TrendingUp className="w-4 h-4" />
            <span>导出模型</span>
          </button>
        </div>
      </div>
    </div>
  );
};

export default ModelVisualization;