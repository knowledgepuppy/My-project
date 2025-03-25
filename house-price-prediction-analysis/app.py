import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from volcenginesdkarkruntime import Ark
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import joblib
from flask_cors import CORS
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)
CORS(app)  # 启用CORS支持

# 配置火山引擎
client = Ark(
    ak=os.environ.get("VOLC_ACCESSKEY", 'your_access_key'),
    sk=os.environ.get("VOLC_SECRETKEY", 'your_secret_key'),
)

# 全局变量存储当前数据
current_data = None
model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global current_data
    if 'file' not in request.files:
        return jsonify({'error': '没有上传文件'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '未选择文件'})
    
    if file and file.filename.endswith('.csv'):
        try:
            current_data = pd.read_csv(file)
            # 格式化数据预览
            preview_html = current_data.head().to_html(
                classes='table table-striped table-bordered',
                float_format=lambda x: '{:.3f}'.format(x) if pd.notnull(x) else '',
                na_rep='',
                index=False
            )
            # 格式化统计描述
            description_html = current_data.describe().to_html(
                classes='table table-striped table-bordered',
                float_format=lambda x: '{:.3f}'.format(x) if pd.notnull(x) else '',
                na_rep='',
            )
            
            return jsonify({
                'columns': current_data.columns.tolist(),
                'preview': preview_html,
                'shape': current_data.shape,
                'description': description_html
            })
        except Exception as e:
            return jsonify({'error': f'文件读取错误: {str(e)}'})
    return jsonify({'error': '请上传CSV文件'})

# 在文件开头添加matplotlib配置
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

@app.route('/analyze', methods=['POST'])
def analyze():
    global current_data
    if current_data is None:
        return jsonify({'error': '请先上传数据'})
    
    try:
        # 只处理数值类型的列
        numeric_data = current_data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            return jsonify({'error': '没有可分析的数值类型数据'})

        # 生成基本统计信息
        description = numeric_data.describe()
        description_dict = {}
        for column in description.columns:
            description_dict[column] = {
                stat: float(val) if not pd.isna(val) else 0 
                for stat, val in description[column].items()
            }

        # 生成直方图
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for idx, col in enumerate(numeric_data.columns[:6]):  # 展示前6个特征
            sns.histplot(data=numeric_data, x=col, ax=axes[idx], kde=True)
            axes[idx].set_title(f'{col}的分布')
        
        plt.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight', dpi=300)
        buffer.seek(0)
        hist_plot = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)

        # 修改散点图矩阵生成方式
        plt.figure(figsize=(12, 12))
        # 选择前4个数值列来生成散点图矩阵，避免图表过大
        plot_columns = numeric_data.columns[:4]
        sns.pairplot(numeric_data[plot_columns], diag_kind='kde', plot_kws={'alpha': 0.6})
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
        buffer.seek(0)
        scatter_plot = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        # 生成相关性矩阵热力图
        corr_matrix = numeric_data.corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, 
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   fmt='.2f',
                   ax=ax)
        ax.set_title('特征相关性热力图', fontsize=12)
        
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight', dpi=300)
        buffer.seek(0)
        heatmap = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)

        # 返回结果
        response = jsonify({
            'description': description_dict,  # 现在 description_dict 已定义
            'hist_plot': hist_plot,
            'scatter_plot': scatter_plot,
            'heatmap': heatmap,
            'missing_values': numeric_data.isnull().sum().to_dict(),
            'columns': numeric_data.columns.tolist()
        })
        
        return response
    except Exception as e:
        print(f"错误详情: {str(e)}")
        return jsonify({'error': f'分析错误: {str(e)}'})

@app.route('/train', methods=['POST'])
def train():
    global current_data, model
    if current_data is None:
        return jsonify({'error': '请先上传数据'})
    
    try:
        data = request.get_json()
        target_column = data.get('target_column')
        feature_columns = data.get('feature_columns')
        
        X = current_data[feature_columns]
        y = current_data[target_column]
        
        # 数据预处理
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 训练模型
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        # 保存模型
        joblib.dump((model, scaler, feature_columns), 'model.pkl')
        
        return jsonify({'message': '模型训练完成'})
    except Exception as e:
        return jsonify({'error': f'训练错误: {str(e)}'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 先加载模型和相关组件
        model, scaler, feature_columns = joblib.load('model.pkl')
        
        data = request.get_json()
        input_data = pd.DataFrame(data['input_data'], index=[0])  # 确保创建DataFrame时有一行数据
        
        # 检查是否有缺失的特征，并用训练数据的平均值填充
        missing_features = set(feature_columns) - set(input_data.columns)
        if missing_features:
            # 加载原始数据计算平均值
            training_data = pd.read_csv('Exp05Bboston_house_prices.csv')
            means = training_data[list(missing_features)].mean()
            
            # 为缺失的特征添加平均值
            for feature in missing_features:
                input_data[feature] = means[feature]
        
        # 确保特征列的顺序与训练时一致
        input_data = input_data[feature_columns]
        
        # 预处理输入数据
        X_scaled = scaler.transform(input_data)
        
        # 预测
        predictions = model.predict(X_scaled)
        
        return jsonify({
            'predictions': predictions.tolist(),
            'filled_features': {k: float(means[k]) for k in missing_features} if missing_features else {}
        })
    except Exception as e:
        return jsonify({'error': f'预测错误: {str(e)}'})

@app.route('/ask_ai', methods=['POST'])
def ask_ai():
    global current_data
    if current_data is None:
        return jsonify({'error': '请先上传数据'})
    
    try:
        data = request.get_json()
        question = data['question']
        
        # 准备上下文信息
        context = {
            'columns': current_data.columns.tolist(),
            'shape': current_data.shape,
            'description': current_data.describe().to_dict(),
            'sample': current_data.head().to_dict()
        }
        
        # 调用豆包API
        completion = client.chat.completions.create(
            model="model",
            messages=[
                {
                    "role": "system",
                    "content": "你是一个数据分析助手，专门解答关于数据分析的问题。"
                },
                {
                    "role": "user",
                    "content": f"基于以下数据信息：{context}\n\n问题：{question}"
                }
            ]
        )
        
        return jsonify({'response': completion.choices[0].message.content})
    except Exception as e:
        return jsonify({'error': f'AI回答错误: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)