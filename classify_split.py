import pandas as pd
from sklearn.model_selection import train_test_split

# 文件路径
csv_file = "D:/VScode/python/py/data_mining/classify/data.csv"  # 替换为你的CSV文件路径
train_file = "D:/VScode/python/py/data_mining/classify/train.csv"
test_file = "D:/VScode/python/py/data_mining/classify/test.csv"

# 加载数据集
df = pd.read_csv(csv_file)

# 确认目标变量列名为'y'
target_col = 'y'

# 检查数据基本信息
print("原始数据集概览：")
print(f"总样本数: {len(df)}")
print(f"类别分布:\n{df[target_col].value_counts(normalize=True)}")

# 分离特征和目标变量
X = df.drop(columns=[target_col])
y = df[target_col]

# 进行分层抽样划分（80:20）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,  # 20%测试集
    stratify=y,     # 分层抽样
    random_state=42  # 随机种子，确保可重复性
)

# 合并特征和目标变量为DataFrame
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# 保存训练集和测试集
train_df.to_csv(train_file, index=False)
test_df.to_csv(test_file, index=False)

# 打印划分结果
print("\n划分结果：")
print(f"训练集保存为 {train_file} ({len(train_df)} 样本)")
print(f"测试集保存为 {test_file} ({len(test_df)} 样本)")
print(f"训练集类别分布:\n{y_train.value_counts(normalize=True)}")
print(f"测试集类别分布:\n{y_test.value_counts(normalize=True)}")

# 验证无重复和完整性
train_indices = set(X_train.index)
test_indices = set(X_test.index)
if train_indices & test_indices:
    print("警告：训练集和测试集存在重叠！")
else:
    print("验证通过：训练集和测试集无重叠。")
if len(train_df) + len(test_df) == len(df):
    print("验证通过：训练集和测试集合并后为完整数据集。")
else:
    print("错误：训练集和测试集样本数之和不等于原始数据集！")