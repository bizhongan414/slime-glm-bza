import pandas as pd
import json
import os

def process_parquet_to_jsonl(input_path, output_path, n):
    """
    1. 读取 Parquet
    2. 解析 verification_info 提取 answer
    3. 构建 messages 字段 (content = prompt)
    4. 复制 n 次并保存为 jsonl
    """
    
    # --- 1. 检查文件 ---
    if not os.path.exists(input_path):
        print(f"错误: 找不到文件 {input_path}")
        return

    print(f"正在读取 Parquet 文件: {input_path} ...")
    
    try:
        df = pd.read_parquet(input_path)
    except Exception as e:
        print(f"读取 Parquet 失败: {e}")
        return

    print(f"读取成功，共 {len(df)} 行数据。正在处理字段...")

    # --- 2. 辅助函数：提取 ground_truth ---
    def extract_answer(verification_info_str):
        try:
            info_dict = json.loads(verification_info_str)
            return info_dict.get('ground_truth')
        except (json.JSONDecodeError, TypeError, AttributeError):
            return None

    # 应用函数创建 'answer' 字段
    if 'verification_info' in df.columns:
        df['answer'] = df['verification_info'].apply(extract_answer)
    else:
        df['answer'] = None

    # --- 3. 转换数据结构并添加 messages 字段 ---
    # 将 DataFrame 转为 Python 字典列表，方便处理嵌套结构
    records = df.to_dict(orient='records')
    
    # 遍历列表，为每一条数据增加 messages 字段
    for record in records:
        # 获取 prompt 内容，如果不存在则为空字符串
        prompt_content = record.get('prompt', "")
        
        # 构建 messages 结构
        record['messages'] = [
            {
                "content": prompt_content,
                "role": "user"
            }
        ]

    print(f"字段处理完成。开始写入文件 (每条数据重复 {n} 次)...")
    
    # --- 4. 写入 JSONL (流式写入) ---
    count = 0
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in records:
                # 序列化当前记录
                json_str = json.dumps(record, ensure_ascii=False)
                
                # 写入 n 次
                for _ in range(n):
                    f.write(json_str + '\n')
                
                count += 1
                if count % 100 == 0:
                    print(f"\r已处理原始数据: {count}/{len(records)} 行", end="")
                    
        print(f"\n\n成功！文件已保存至: {output_path}")
        print(f"原始行数: {len(records)} -> 最终行数: {len(records) * n}")
        
    except IOError as e:
        print(f"\n写入文件失败: {e}")

# ==========================================
# 用户配置区域 (请修改这里)
# ==========================================

if __name__ == "__main__":
    # 1. 输入 Parquet 文件的路径 (支持相对路径或绝对路径)
    INPUT_FILE = "/gfs/space/chatrl/users/wlw_temp/data/data_aime25/data/train-00000-of-00001.parquet"  # 修改为你的实际路径
    
    # 2. 输出 JSONL 文件的路径
    OUTPUT_FILE = "/gfs/space/chatrl/users/wlw_temp/data/data_aime25/data/train-00000-of-00001_avg8.jsonl" # 修改为你想要保存的路径
    
    # 3. 复制次数 N
    REPEAT_N = 8

    # 执行
    process_parquet_to_jsonl(INPUT_FILE, OUTPUT_FILE, REPEAT_N)