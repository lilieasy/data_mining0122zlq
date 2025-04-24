import pandas as pd
import matplotlib.pyplot as plt

# 加载测试集预测结果
predictions = pd.read_csv("D:/weka/output/ML_test2.csv")

# 提取实际值、预测值和误差
actual = predictions['actual']
predicted = predictions['predicted']
errors = predictions['error']

# 绘制实际值与预测值的对比曲线
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)  # 第一个子图
plt.plot(actual, label='Actual Stock Index', color='blue', marker='o', markersize=5, linestyle='-')
plt.plot(predicted, label='Predicted Stock Index', color='red', marker='x', markersize=5, linestyle='--')
plt.title('Actual vs Predicted Stock Index (Test Set)')
plt.xlabel('Instance Index')
plt.ylabel('Stock Index Value')
plt.legend()
plt.grid(True)

# 绘制误差分布直方图
plt.subplot(2, 1, 2)  # 第二个子图
plt.hist(errors, bins=10, color='purple', alpha=0.7, edgecolor='black')
plt.title('Prediction Error Distribution')
plt.xlabel('Error (Predicted - Actual)')
plt.ylabel('Frequency')
plt.grid(True)

# 调整布局并保存
plt.tight_layout()
plt.savefig('D:/VScode/python/py/data_mining/test_actual_vs_predicted_with_error.png')
plt.show()