import pandas as pd
import os

def process_parquet_to_jsonl(input_path, output_path):
    print(f"1. 正在读取 Parquet 文件: {input_path} ...")
    
    # 读取文件
    try:
        df = pd.read_parquet(input_path, engine='pyarrow')
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return

    print(f"   读取成功，共 {len(df)} 条数据。")
    print("2. 正在提取 answer ...")

    # 定义提取函数：从 reward_model 字典中安全提取 ground_truth
    def extract_answer(reward_model_data):
        if isinstance(reward_model_data, dict):
            return reward_model_data.get('ground_truth')
        return None

    # 应用函数创建新列
    df['answer'] = df['reward_model'].apply(extract_answer)

    # 打印一条数据验证（可选）
    print("   数据预览 (第一条):")
    print(f"   Reward Model: {df['reward_model'].iloc[0]}")
    print(f"   Extracted Answer: {df['answer'].iloc[0]}")

    print(f"3. 正在保存为 JSONL: {output_path} ...")
    
    # 保存为 JSONL 格式
    # orient='records': 将每一行转换为一个对象
    # lines=True: 开启 JSONL 模式（每行一个 JSON 对象）
    # force_ascii=False: 允许保存非 ASCII 字符（如中文），不仅限于 Unicode 编码
    df.to_json(output_path, orient='records', lines=True, force_ascii=False)
    
    print(f"✅ 处理完成！文件已保存至: {output_path}")

if __name__ == "__main__":
    # ================= 配置区域 =================
    
    INPUT_FILE = '/gfs/space/chatrl/users/wlw_temp/data/dapo_17k/data/dapo-math-17k.parquet'          # 输入文件名
    OUTPUT_FILE = '/gfs/space/chatrl/users/wlw_temp/data/dapo_17k/data/dapo-math-17k-with-answer-label.jsonl' # 输出文件名 (.jsonl)
    

    # 执行转换
    process_parquet_to_jsonl(INPUT_FILE, OUTPUT_FILE)