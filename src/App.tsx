import React, { useState } from 'react';
import { Cloud, BarChart3, Map, Settings, Activity, AlertTriangle } from 'lucide-react';
import Dashboard from './components/Dashboard';
import ModelVisualization from './components/ModelVisualization';
import DataAnalysis from './components/DataAnalysis';
import PredictionInterface from './components/PredictionInterface';

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');

  const tabs = [
    { id: 'dashboard', label: '仪表板', icon: BarChart3 },
    { id: 'model', label: '模型可视化', icon: Activity },
    { id: 'data', label: '数据分析', icon: Map },
    { id: 'prediction', label: '预测界面', icon: AlertTriangle },
  ];

  const renderContent = () => {
    switch (activeTab) {
      case 'dashboard':
        return <Dashboard />;
      case 'model':
        return <ModelVisualization />;
      case 'data':
        return <DataAnalysis />;
      case 'prediction':
        return <PredictionInterface />;
      default:
        return <Dashboard />;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-3">
              <div className="relative">
                <Cloud className="w-8 h-8 text-typhoon-600" />
                <div className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">台风预测模型</h1>
                <p className="text-sm text-gray-500">Typhoon Prediction System</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2 text-sm text-gray-600">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                <span>系统运行正常</span>
              </div>
              <Settings className="w-5 h-5 text-gray-400 hover:text-gray-600 cursor-pointer" />
            </div>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm transition-colors duration-200 ${
                    activeTab === tab.id
                      ? 'border-typhoon-500 text-typhoon-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {renderContent()}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <p className="text-sm text-gray-500">
              © 2025 台风预测模型系统. 基于机器学习的气象预测平台.
            </p>
            <div className="flex items-center space-x-4 text-sm text-gray-500">
              <span>数据更新时间: {new Date().toLocaleString('zh-CN')}</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;