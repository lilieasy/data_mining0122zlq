import pandas as pd
import numpy as np

def create_embedding_matrix(data, embed_dim=7):
    """
    将时间序列数据转换为嵌入维度为7的输入输出矩阵。
    
    参数：
    data: pandas Series 或 numpy array，时间序列数据
    embed_dim: 嵌入维度（过去的天数作为输入）
    
    返回：
    X: 输入矩阵（每行包含embed_dim个过去值）
    y: 输出向量（下一天的值）
    """
    X, y = [], []
    for i in range(len(data) - embed_dim):
        X.append(data[i:i + embed_dim])  # 取过去embed_dim天的值作为输入
        y.append(data[i + embed_dim])    # 取下一天的值作为输出
    return np.array(X), np.array(y)

def process_stock_data(input_file, output_file, embed_dim=7):
    """
    处理股票CSV文件，创建嵌入数据以供WEKA分析。
    
    参数：
    input_file: 输入CSV文件路径
    output_file: 输出CSV文件路径
    embed_dim: 嵌入维度（默认7，考虑一周周期）
    """
    # 读取CSV文件
    df = pd.read_csv(input_file)
    
    # 提取“spji”列作为时间序列数据
    time_series = df['spj'].values
    
    # 创建嵌入矩阵
    X, y = create_embedding_matrix(time_series, embed_dim)
    
    # 创建输出CSV的列名：day_t-6, day_t-5, ..., day_t, target
    columns = [f'day_t-{i}' for i in range(embed_dim-1, -1, -1)] + ['target']
    
    # 将X和y合并为一个DataFrame
    output_data = pd.DataFrame(np.column_stack([X, y]), columns=columns)
    
    # 保存为CSV文件
    output_data.to_csv(output_file, index=False)
    print(f"输出CSV文件已保存为：{output_file}")

if __name__ == "__main__":
    # 示例用法
    input_file = "D:/VScode/python/py/data_mining/stock_data2.csv"  # 替换为你的输入CSV文件路径
    output_file = "D:/VScode/python/py/data_mining/stock_embedded_data2.csv"  # 输出文件路径
    embed_dim = 7  # 嵌入维度，考虑一周周期
    
    process_stock_data(input_file, output_file, embed_dim)