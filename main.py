import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder  # Import LabelEncoder from sklearn
import xgboost as xgb

# 读取数据
df = pd.read_csv(r"C:\Users\ASUS\Desktop\大二下课程文件\考查-单车.csv")

# 检查空缺值并填充
end_station_mean = df['End station number'].mean()
df['End station number'].fillna(end_station_mean, inplace=True)

start_station_mean = df['Start station number'].mean()
df['Start station number'].fillna(start_station_mean, inplace=True)

# 转换日期格式
df['Start date'] = pd.to_datetime(df['Start date'], format='%m/%d/%Y %H:%M')
df['End date'] = pd.to_datetime(df['End date'], format='%m/%d/%Y %H:%M')

# 生成新特征
df['hour'] = df['Start date'].dt.hour
df['is_peak'] = df['hour'].apply(lambda x: 1 if (6 <= x <= 10) or (16 <= x <= 20) else 0)
df['is_weekend'] = df['Start date'].dt.weekday >= 5

# 特征和目标变量
X = df[['Start station number', 'End station number', 'Bike model', 'hour', 'is_weekend']]
y = df['Total duration (ms)']

# Label Encoding for 'Bike model' column
label_encoder = LabelEncoder()
X['Bike model'] = label_encoder.fit_transform(X['Bike model'])

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 转换数据为DMatrix格式，这是XGBoost所需的数据格式
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 设置XGBoost参数
params = {
    'objective': 'reg:squarederror',  # 回归任务
    'eval_metric': 'rmse',  # 评估指标为均方根误差
    'seed': 42
}

# 训练模型
num_rounds = 1000  # 迭代次数
xgb_model = xgb.train(params, dtrain, num_rounds)

# 预测
y_pred = xgb_model.predict(dtest)

# 评估模型
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f'R² 指标：{r2:.2f}')
print(f'MSE 指标：{mse:.2f}')

# 可视化预测结果
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title('真实值 vs 预测值')
plt.show()
