import pandas as pd

def split_stock_data(input_file, train_file, test_file, train_ratio=0.8):
    """
    按时间顺序将嵌入数据划分为训练集和测试集。
    
    参数：
    input_file: 输入CSV文件路径（嵌入后的数据）
    train_file: 训练集CSV文件路径
    test_file: 测试集CSV文件路径
    train_ratio: 训练集比例（默认0.8）
    """
    # 读取嵌入后的数据
    df = pd.read_csv(input_file)
    
    # 检查列数
    expected_columns = 8  # day_t-6 到 day_t-0 共7列，加上target共8列
    if len(df.columns) != expected_columns:
        raise ValueError(f"预期输入数据有 {expected_columns} 列，但实际有 {len(df.columns)} 列")
    
    # 总行数（不含标题行）
    total_rows = len(df)
    print(f"总数据行数（不含标题行）: {total_rows}")
    print(f"列数: {len(df.columns)}")
    
    # 计算训练集和测试集的行数（覆盖所有数据）
    train_rows = int(total_rows * train_ratio)
    test_rows = total_rows - train_rows  # 确保训练集和测试集之和为总行数
    
    # 按时间顺序划分训练集和测试集
    train_data = df.iloc[:train_rows]
    test_data = df.iloc[train_rows:]
    
    # 打印划分信息
    print(f"训练集行数（不含标题行）: {len(train_data)}")
    print(f"测试集行数（不含标题行）: {len(test_data)}")
    
    # 保存训练集和测试集
    train_data.to_csv(train_file, index=False)
    test_data.to_csv(test_file, index=False)
    print(f"训练集已保存为: {train_file}")
    print(f"测试集已保存为: {test_file}")

if __name__ == "__main__":
    # 文件路径
    input_file = "D:/VScode/python/py/data_mining/stock_embedded_data2.csv"
    train_file = "D:/VScode/python/py/data_mining/stock_train3.csv"
    test_file = "D:/VScode/python/py/data_mining/stock_test3.csv"
    
    # 划分数据集
    split_stock_data(input_file, train_file, test_file, train_ratio=0.8)