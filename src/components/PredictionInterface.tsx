import React, { useState } from 'react';
import { AlertTriangle, MapPin, Clock, Wind, Thermometer, Droplets, Eye, Play, RefreshCw } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';

const PredictionInterface = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [predictionResult, setPredictionResult] = useState(null);

  // 模拟输入参数
  const [inputParams, setInputParams] = useState({
    seaTemperature: 28.5,
    pressure: 1005,
    windSpeed: 45,
    humidity: 75,
    cloudCover: 60,
    latitude: 15.5,
    longitude: 140.2,
  });

  // 预测路径数据
  const predictionPath = [
    { time: '现在', lat: 15.5, lng: 140.2, intensity: 3.2, probability: 95 },
    { time: '6小时', lat: 16.1, lng: 139.8, intensity: 3.5, probability: 92 },
    { time: '12小时', lat: 16.8, lng: 139.2, intensity: 3.8, probability: 88 },
    { time: '18小时', lat: 17.5, lng: 138.5, intensity: 4.1, probability: 85 },
    { time: '24小时', lat: 18.2, lng: 137.8, intensity: 4.3, probability: 82 },
    { time: '48小时', lat: 20.1, lng: 135.5, intensity: 4.5, probability: 75 },
    { time: '72小时', lat: 22.5, lng: 132.8, intensity: 4.2, probability: 68 },
  ];

  // 强度预测数据
  const intensityForecast = [
    { time: '0h', current: 3.2, predicted: 3.2, confidence: 0.95 },
    { time: '6h', current: null, predicted: 3.5, confidence: 0.92 },
    { time: '12h', current: null, predicted: 3.8, confidence: 0.88 },
    { time: '18h', current: null, predicted: 4.1, confidence: 0.85 },
    { time: '24h', current: null, predicted: 4.3, confidence: 0.82 },
    { time: '48h', current: null, predicted: 4.5, confidence: 0.75 },
    { time: '72h', current: null, predicted: 4.2, confidence: 0.68 },
  ];

  const handlePredict = async () => {
    setIsLoading(true);
    // 模拟API调用
    setTimeout(() => {
      setPredictionResult({
        category: 4,
        maxWindSpeed: 185,
        minPressure: 945,
        riskLevel: 'high',
        affectedAreas: ['冲绳', '台湾', '福建'],
        confidence: 0.87
      });
      setIsLoading(false);
    }, 2000);
  };

  const handleInputChange = (key, value) => {
    setInputParams(prev => ({
      ...prev,
      [key]: parseFloat(value)
    }));
  };

  return (
    <div className="space-y-8">
      {/* 预测控制面板 */}
      <div className="card">
        <h2 className="text-xl font-bold text-gray-900 mb-6 flex items-center">
          <AlertTriangle className="w-6 h-6 mr-2 text-red-600" />
          台风预测系统
        </h2>
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* 输入参数 */}
          <div className="lg:col-span-1">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">输入参数</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  海表温度 (°C)
                </label>
                <div className="flex items-center space-x-2">
                  <Thermometer className="w-4 h-4 text-orange-500" />
                  <input
                    type="number"
                    value={inputParams.seaTemperature}
                    onChange={(e) => handleInputChange('seaTemperature', e.target.value)}
                    className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    step="0.1"
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  气压 (hPa)
                </label>
                <div className="flex items-center space-x-2">
                  <Eye className="w-4 h-4 text-blue-500" />
                  <input
                    type="number"
                    value={inputParams.pressure}
                    onChange={(e) => handleInputChange('pressure', e.target.value)}
                    className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  风速 (m/s)
                </label>
                <div className="flex items-center space-x-2">
                  <Wind className="w-4 h-4 text-green-500" />
                  <input
                    type="number"
                    value={inputParams.windSpeed}
                    onChange={(e) => handleInputChange('windSpeed', e.target.value)}
                    className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  湿度 (%)
                </label>
                <div className="flex items-center space-x-2">
                  <Droplets className="w-4 h-4 text-blue-400" />
                  <input
                    type="number"
                    value={inputParams.humidity}
                    onChange={(e) => handleInputChange('humidity', e.target.value)}
                    className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    纬度
                  </label>
                  <input
                    type="number"
                    value={inputParams.latitude}
                    onChange={(e) => handleInputChange('latitude', e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    step="0.1"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    经度
                  </label>
                  <input
                    type="number"
                    value={inputParams.longitude}
                    onChange={(e) => handleInputChange('longitude', e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    step="0.1"
                  />
                </div>
              </div>

              <button
                onClick={handlePredict}
                disabled={isLoading}
                className="w-full btn-primary flex items-center justify-center space-x-2"
              >
                {isLoading ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    <span>预测中...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4" />
                    <span>开始预测</span>
                  </>
                )}
              </button>
            </div>
          </div>

          {/* 预测结果 */}
          <div className="lg:col-span-2">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">预测结果</h3>
            {predictionResult ? (
              <div className="space-y-4">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="bg-red-50 p-4 rounded-lg">
                    <div className="text-2xl font-bold text-red-600">{predictionResult.category}</div>
                    <div className="text-sm text-gray-600">台风等级</div>
                  </div>
                  <div className="bg-blue-50 p-4 rounded-lg">
                    <div className="text-2xl font-bold text-blue-600">{predictionResult.maxWindSpeed}</div>
                    <div className="text-sm text-gray-600">最大风速 (km/h)</div>
                  </div>
                  <div className="bg-purple-50 p-4 rounded-lg">
                    <div className="text-2xl font-bold text-purple-600">{predictionResult.minPressure}</div>
                    <div className="text-sm text-gray-600">最低气压 (hPa)</div>
                  </div>
                  <div className="bg-green-50 p-4 rounded-lg">
                    <div className="text-2xl font-bold text-green-600">{(predictionResult.confidence * 100).toFixed(1)}%</div>
                    <div className="text-sm text-gray-600">预测置信度</div>
                  </div>
                </div>

                <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                  <div className="flex items-center space-x-2 mb-2">
                    <AlertTriangle className="w-5 h-5 text-yellow-600" />
                    <span className="font-medium text-yellow-800">风险等级: 高</span>
                  </div>
                  <p className="text-sm text-yellow-700">
                    预计影响区域: {predictionResult.affectedAreas.join('、')}
                  </p>
                </div>
              </div>
            ) : (
              <div className="bg-gray-50 rounded-lg p-8 text-center">
                <AlertTriangle className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-500">请输入参数并点击"开始预测"按钮</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* 预测路径 */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
          <MapPin className="w-5 h-5 mr-2 text-green-600" />
          预测路径
        </h3>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div>
            <div className="bg-gradient-to-br from-blue-100 to-green-100 rounded-lg p-6 h-80 relative overflow-hidden">
              <div className="absolute inset-0 opacity-20">
                <svg viewBox="0 0 400 300" className="w-full h-full">
                  <path d="M50,50 Q100,30 150,50 T250,50 Q300,40 350,60 L350,200 Q300,210 250,200 T150,200 Q100,220 50,200 Z" 
                        fill="#3b82f6" opacity="0.3"/>
                </svg>
              </div>
              <svg className="absolute inset-0 w-full h-full">
                <path
                  d="M 100,200 Q 150,150 200,120 T 300,80"
                  stroke="#ef4444"
                  strokeWidth="3"
                  fill="none"
                  className="typhoon-path"
                />
                {predictionPath.slice(0, 5).map((point, index) => (
                  <circle
                    key={index}
                    cx={100 + index * 50}
                    cy={200 - index * 30}
                    r={4 + index}
                    fill="#ef4444"
                    className="data-point"
                  >
                    <title>{point.time}: 强度 {point.intensity}</title>
                  </circle>
                ))}
              </svg>
            </div>
          </div>
          <div className="space-y-3">
            {predictionPath.map((point, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center space-x-3">
                  <div className={`w-3 h-3 rounded-full ${
                    index === 0 ? 'bg-red-500' : 'bg-blue-500'
                  }`}></div>
                  <div>
                    <div className="font-medium text-gray-900">{point.time}</div>
                    <div className="text-sm text-gray-500">
                      {point.lat.toFixed(1)}°N, {point.lng.toFixed(1)}°E
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="font-medium text-gray-900">强度 {point.intensity}</div>
                  <div className="text-sm text-gray-500">{point.probability}% 置信度</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* 强度预测图表 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <Wind className="w-5 h-5 mr-2 text-blue-600" />
            强度预测
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={intensityForecast}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Line 
                type="monotone" 
                dataKey="predicted" 
                stroke="#3b82f6" 
                strokeWidth={3}
                name="预测强度"
              />
              <Line 
                type="monotone" 
                dataKey="current" 
                stroke="#ef4444" 
                strokeWidth={3}
                name="当前强度"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <Clock className="w-5 h-5 mr-2 text-purple-600" />
            置信度变化
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={intensityForecast}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Area 
                type="monotone" 
                dataKey="confidence" 
                stroke="#8b5cf6" 
                fill="#c4b5fd" 
                strokeWidth={2}
                name="置信度"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* 预警信息 */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">预警信息</h3>
        <div className="space-y-4">
          <div className="bg-red-50 border border-red-200 rounded-lg p-4">
            <div className="flex items-center space-x-2 mb-2">
              <AlertTriangle className="w-5 h-5 text-red-600" />
              <span className="font-medium text-red-800">台风红色预警</span>
              <span className="text-sm text-red-600">发布时间: {new Date().toLocaleString('zh-CN')}</span>
            </div>
            <p className="text-sm text-red-700">
              预计未来24-48小时内，台风将以4级强度影响冲绳、台湾等地区，请做好防护准备。
            </p>
          </div>
          
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
            <div className="flex items-center space-x-2 mb-2">
              <Wind className="w-5 h-5 text-yellow-600" />
              <span className="font-medium text-yellow-800">大风预警</span>
            </div>
            <p className="text-sm text-yellow-700">
              预计最大风速可达185km/h，沿海地区需特别注意防范。
            </p>
          </div>

          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <div className="flex items-center space-x-2 mb-2">
              <Droplets className="w-5 h-5 text-blue-600" />
              <span className="font-medium text-blue-800">暴雨预警</span>
            </div>
            <p className="text-sm text-blue-700">
              台风过境期间可能伴随强降雨，累计降雨量可达200-400mm。
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PredictionInterface;