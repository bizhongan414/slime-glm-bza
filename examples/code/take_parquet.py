import pandas as pd

file_path = '/gfs/space/chatrl/users/wlw_temp/data/data_aime25/data/train-00000-of-00001.parquet'

try:
    # 读取 parquet 文件
    df = pd.read_parquet(file_path)
    
    # 方法 A: 打印第一行（以 Series 格式显示，包含列名和值）
    print("--- 第一行数据 (Series) ---")
    print(df.iloc[0])
    
    # 方法 B: 转换为字典格式（更易读）
    print("\n--- 第一行数据 (Dict) ---")
    print(df.iloc[0].to_dict())

except Exception as e:
    print(f"读取失败: {e}")