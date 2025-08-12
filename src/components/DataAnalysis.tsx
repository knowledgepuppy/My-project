import React, { useState } from 'react';
import { Map, Satellite, BarChart3, PieChart, Filter, Download } from 'lucide-react';
import { ComposedChart, Bar, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart as RechartsPieChart, Cell, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';

const DataAnalysis = () => {
  const [selectedRegion, setSelectedRegion] = useState('western-pacific');
  const [timeRange, setTimeRange] = useState('month');

  // 模拟地理数据
  const regionData = [
    { region: '西太平洋', typhoons: 15, avgIntensity: 4.2, lat: 15, lng: 140 },
    { region: '东太平洋', typhoons: 8, avgIntensity: 3.8, lat: 15, lng: -120 },
    { region: '北大西洋', typhoons: 12, avgIntensity: 3.5, lat: 25, lng: -60 },
    { region: '印度洋', typhoons: 6, avgIntensity: 3.9, lat: -15, lng: 80 },
  ];

  // 时间序列数据
  const timeSeriesData = [
    { month: '1月', count: 0, intensity: 0, temperature: 26.5 },
    { month: '2月', count: 1, intensity: 2.1, temperature: 26.8 },
    { month: '3月', count: 2, intensity: 2.8, temperature: 27.2 },
    { month: '4月', count: 3, intensity: 3.2, temperature: 28.1 },
    { month: '5月', count: 5, intensity: 3.6, temperature: 28.9 },
    { month: '6月', count: 8, intensity: 4.1, temperature: 29.5 },
    { month: '7月', count: 12, intensity: 4.5, temperature: 30.2 },
    { month: '8月', count: 15, intensity: 4.8, temperature: 30.8 },
    { month: '9月', count: 18, intensity: 4.6, temperature: 30.1 },
    { month: '10月', count: 12, intensity: 4.2, temperature: 29.3 },
    { month: '11月', count: 6, intensity: 3.5, temperature: 28.2 },
    { month: '12月', count: 2, intensity: 2.9, temperature: 27.1 },
  ];

  // 台风等级分布
  const categoryData = [
    { name: '热带低压', value: 25, color: '#22c55e' },
    { name: '热带风暴', value: 30, color: '#3b82f6' },
    { name: '强热带风暴', value: 20, color: '#f59e0b' },
    { name: '台风', value: 15, color: '#ef4444' },
    { name: '强台风', value: 8, color: '#8b5cf6' },
    { name: '超强台风', value: 2, color: '#dc2626' },
  ];

  // 雷达图数据
  const radarData = [
    { subject: '风速', A: 120, B: 110, fullMark: 150 },
    { subject: '气压', A: 98, B: 130, fullMark: 150 },
    { subject: '温度', A: 86, B: 130, fullMark: 150 },
    { subject: '湿度', A: 99, B: 100, fullMark: 150 },
    { subject: '云量', A: 85, B: 90, fullMark: 150 },
    { subject: '降水', A: 65, B: 85, fullMark: 150 },
  ];

  return (
    <div className="space-y-8">
      {/* 控制面板 */}
      <div className="card">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <h2 className="text-xl font-bold text-gray-900 flex items-center">
            <BarChart3 className="w-6 h-6 mr-2 text-blue-600" />
            数据分析中心
          </h2>
          <div className="flex items-center space-x-4">
            <select 
              value={selectedRegion}
              onChange={(e) => setSelectedRegion(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="western-pacific">西太平洋</option>
              <option value="eastern-pacific">东太平洋</option>
              <option value="atlantic">北大西洋</option>
              <option value="indian">印度洋</option>
            </select>
            <select 
              value={timeRange}
              onChange={(e) => setTimeRange(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="week">本周</option>
              <option value="month">本月</option>
              <option value="year">本年</option>
            </select>
            <button className="btn-secondary flex items-center space-x-2">
              <Filter className="w-4 h-4" />
              <span>筛选</span>
            </button>
            <button className="btn-primary flex items-center space-x-2">
              <Download className="w-4 h-4" />
              <span>导出</span>
            </button>
          </div>
        </div>
      </div>

      {/* 地理分布 */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
          <Map className="w-5 h-5 mr-2 text-green-600" />
          全球台风分布
        </h3>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <div className="bg-gradient-to-br from-blue-100 to-blue-200 rounded-lg p-6 h-80 relative overflow-hidden">
              <div className="absolute inset-0 opacity-20">
                <svg viewBox="0 0 400 200" className="w-full h-full">
                  {/* 简化的世界地图轮廓 */}
                  <path d="M50,50 Q100,30 150,50 T250,50 Q300,40 350,60 L350,150 Q300,160 250,150 T150,150 Q100,170 50,150 Z" 
                        fill="#3b82f6" opacity="0.3"/>
                </svg>
              </div>
              {regionData.map((region, index) => (
                <div
                  key={index}
                  className="absolute w-4 h-4 bg-red-500 rounded-full animate-pulse"
                  style={{
                    left: `${(region.lng + 180) / 360 * 100}%`,
                    top: `${(90 - region.lat) / 180 * 100}%`,
                    transform: 'translate(-50%, -50%)'
                  }}
                  title={`${region.region}: ${region.typhoons}个台风`}
                >
                  <div className="absolute inset-0 bg-red-400 rounded-full animate-ping"></div>
                </div>
              ))}
            </div>
          </div>
          <div className="space-y-4">
            {regionData.map((region, index) => (
              <div key={index} className="p-4 bg-gray-50 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-medium text-gray-900">{region.region}</h4>
                  <span className="text-sm text-gray-500">{region.typhoons}个</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="flex-1 bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-blue-500 h-2 rounded-full"
                      style={{ width: `${(region.avgIntensity / 5) * 100}%` }}
                    ></div>
                  </div>
                  <span className="text-xs text-gray-600">强度 {region.avgIntensity}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* 时间序列分析 */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
          <Satellite className="w-5 h-5 mr-2 text-purple-600" />
          台风活动时间分析
        </h3>
        <ResponsiveContainer width="100%" height={400}>
          <ComposedChart data={timeSeriesData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="month" />
            <YAxis yAxisId="left" />
            <YAxis yAxisId="right" orientation="right" />
            <Tooltip />
            <Bar yAxisId="left" dataKey="count" fill="#3b82f6" name="台风数量" />
            <Line 
              yAxisId="right" 
              type="monotone" 
              dataKey="intensity" 
              stroke="#ef4444" 
              strokeWidth={3}
              name="平均强度"
            />
            <Line 
              yAxisId="right" 
              type="monotone" 
              dataKey="temperature" 
              stroke="#f59e0b" 
              strokeWidth={3}
              strokeDasharray="5 5"
              name="海表温度"
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* 统计图表 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* 台风等级分布 */}
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <PieChart className="w-5 h-5 mr-2 text-orange-600" />
            台风等级分布
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <RechartsPieChart>
              <Tooltip />
              <RechartsPieChart data={categoryData} cx="50%" cy="50%" outerRadius={80} dataKey="value">
                {categoryData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </RechartsPieChart>
            </RechartsPieChart>
          </ResponsiveContainer>
          <div className="grid grid-cols-2 gap-2 mt-4">
            {categoryData.map((item, index) => (
              <div key={index} className="flex items-center space-x-2">
                <div 
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: item.color }}
                ></div>
                <span className="text-sm text-gray-600">{item.name}</span>
                <span className="text-sm font-medium text-gray-900">{item.value}%</span>
              </div>
            ))}
          </div>
        </div>

        {/* 气象要素雷达图 */}
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">气象要素对比</h3>
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={radarData}>
              <PolarGrid />
              <PolarAngleAxis dataKey="subject" />
              <PolarRadiusAxis />
              <Radar 
                name="当前台风" 
                dataKey="A" 
                stroke="#3b82f6" 
                fill="#3b82f6" 
                fillOpacity={0.3}
                strokeWidth={2}
              />
              <Radar 
                name="历史平均" 
                dataKey="B" 
                stroke="#ef4444" 
                fill="#ef4444" 
                fillOpacity={0.3}
                strokeWidth={2}
              />
              <Tooltip />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* 数据统计表 */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">详细统计数据</h3>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  时间
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  台风数量
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  平均强度
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  海表温度
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  趋势
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {timeSeriesData.slice(6, 10).map((item, index) => (
                <tr key={index} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                    {item.month}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {item.count}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {item.intensity}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {item.temperature}°C
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                      item.count > 10 ? 'bg-red-100 text-red-800' : 
                      item.count > 5 ? 'bg-yellow-100 text-yellow-800' : 
                      'bg-green-100 text-green-800'
                    }`}>
                      {item.count > 10 ? '高' : item.count > 5 ? '中' : '低'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default DataAnalysis;