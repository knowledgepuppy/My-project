import React from 'react';
import { TrendingUp, AlertCircle, MapPin, Clock, Wind, Thermometer } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, BarChart, Bar } from 'recharts';

const Dashboard = () => {
  // 模拟台风数据
  const typhoonData = [
    { time: '00:00', pressure: 980, windSpeed: 45, temperature: 28 },
    { time: '06:00', pressure: 975, windSpeed: 52, temperature: 27 },
    { time: '12:00', pressure: 970, windSpeed: 58, temperature: 26 },
    { time: '18:00', pressure: 965, windSpeed: 65, temperature: 25 },
    { time: '24:00', pressure: 960, windSpeed: 72, temperature: 24 },
  ];

  const predictionAccuracy = [
    { model: 'CNN', accuracy: 92.5 },
    { model: 'LSTM', accuracy: 89.3 },
    { model: 'Random Forest', accuracy: 85.7 },
    { model: 'SVM', accuracy: 82.1 },
  ];

  const recentTyphoons = [
    { name: '台风玛娃', category: 5, status: '活跃', lastUpdate: '2小时前' },
    { name: '台风古超', category: 3, status: '减弱', lastUpdate: '4小时前' },
    { name: '台风泰利', category: 2, status: '消散', lastUpdate: '1天前' },
  ];

  return (
    <div className="space-y-8">
      {/* 关键指标卡片 */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">活跃台风</p>
              <p className="text-3xl font-bold text-typhoon-600">3</p>
              <p className="text-sm text-green-600 flex items-center mt-1">
                <TrendingUp className="w-4 h-4 mr-1" />
                较上周 +1
              </p>
            </div>
            <div className="p-3 bg-typhoon-100 rounded-full">
              <Wind className="w-6 h-6 text-typhoon-600" />
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">预测准确率</p>
              <p className="text-3xl font-bold text-green-600">92.5%</p>
              <p className="text-sm text-green-600 flex items-center mt-1">
                <TrendingUp className="w-4 h-4 mr-1" />
                +2.3%
              </p>
            </div>
            <div className="p-3 bg-green-100 rounded-full">
              <AlertCircle className="w-6 h-6 text-green-600" />
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">监测区域</p>
              <p className="text-3xl font-bold text-blue-600">15</p>
              <p className="text-sm text-gray-500 flex items-center mt-1">
                <MapPin className="w-4 h-4 mr-1" />
                西太平洋
              </p>
            </div>
            <div className="p-3 bg-blue-100 rounded-full">
              <MapPin className="w-6 h-6 text-blue-600" />
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">数据更新</p>
              <p className="text-3xl font-bold text-purple-600">实时</p>
              <p className="text-sm text-gray-500 flex items-center mt-1">
                <Clock className="w-4 h-4 mr-1" />
                每6小时
              </p>
            </div>
            <div className="p-3 bg-purple-100 rounded-full">
              <Clock className="w-6 h-6 text-purple-600" />
            </div>
          </div>
        </div>
      </div>

      {/* 图表区域 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* 台风强度变化 */}
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <Wind className="w-5 h-5 mr-2 text-typhoon-600" />
            台风强度变化趋势
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={typhoonData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis yAxisId="left" />
              <YAxis yAxisId="right" orientation="right" />
              <Tooltip />
              <Line 
                yAxisId="left" 
                type="monotone" 
                dataKey="pressure" 
                stroke="#0ea5e9" 
                strokeWidth={3}
                name="气压 (hPa)"
              />
              <Line 
                yAxisId="right" 
                type="monotone" 
                dataKey="windSpeed" 
                stroke="#ef4444" 
                strokeWidth={3}
                name="风速 (m/s)"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* 模型准确率对比 */}
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <BarChart className="w-5 h-5 mr-2 text-green-600" />
            模型预测准确率对比
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={predictionAccuracy}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="model" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="accuracy" fill="#10b981" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* 温度变化区域图 */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
          <Thermometer className="w-5 h-5 mr-2 text-orange-600" />
          海表温度变化
        </h3>
        <ResponsiveContainer width="100%" height={250}>
          <AreaChart data={typhoonData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="time" />
            <YAxis />
            <Tooltip />
            <Area 
              type="monotone" 
              dataKey="temperature" 
              stroke="#f97316" 
              fill="#fed7aa" 
              strokeWidth={2}
              name="温度 (°C)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* 最近台风活动 */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">最近台风活动</h3>
        <div className="space-y-4">
          {recentTyphoons.map((typhoon, index) => (
            <div key={index} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
              <div className="flex items-center space-x-4">
                <div className={`w-3 h-3 rounded-full ${
                  typhoon.status === '活跃' ? 'bg-red-500 animate-pulse' :
                  typhoon.status === '减弱' ? 'bg-yellow-500' : 'bg-gray-400'
                }`}></div>
                <div>
                  <h4 className="font-medium text-gray-900">{typhoon.name}</h4>
                  <p className="text-sm text-gray-500">等级 {typhoon.category} · {typhoon.status}</p>
                </div>
              </div>
              <div className="text-right">
                <p className="text-sm text-gray-500">{typhoon.lastUpdate}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;