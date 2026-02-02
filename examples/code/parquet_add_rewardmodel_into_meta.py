import pandas as pd
import json
import os
import numpy as np
# --- 配置区 ---/afs/chatrl/users/lyy/data/code_train/DeepCoder-Preview-Dataset_wlw/taco
input_path = '/gfs/space/chatrl/users/lyy/data/code_train/DeepCoder-Preview-Dataset_wlw/taco/train.parquet'
output_path = '/gfs/space/chatrl/users/wlw_temp/wlw/data/slime/DeepCoder-Preview-Dataset_wlw/taco/train.jsonl' # 你可以修改这里
# --------------
class NumpyEncoder(json.JSONEncoder):
    """ 处理 ndarray 等 numpy 数据类型的编码器 """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def process_and_save_jsonl(src, dest):
    try:
        # 1. 读取 Parquet
        print(f"🚀 正在读取源文件: {src}")
        df = pd.read_parquet(src)
        
        # 2. 转换为列表字典
        data_list = df.to_dict(orient='records')
        
        # 3. 确保输出目录存在
        output_dir = os.path.dirname(dest)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        print(f"🔄 正在转换并写入 JSONL 文件...")
        
        # 4. 逐行处理并写入文件
        with open(dest, 'w', encoding='utf-8') as f:
            for i, entry in enumerate(data_list):
                # --- 结构调整逻辑 ---
                rm_content = entry.get("reward_model")
                
                # 在 metadata 中包裹一层 "reward_model"
                entry["metadata"] = {
                    "reward_model": rm_content
                }
                
                # 写入一行 JSON（不带缩进，紧凑格式）并换行
                # ensure_ascii=False 保证中文正常显示
                json_line = json.dumps(entry, ensure_ascii=False, cls=NumpyEncoder)
                f.write(json_line + '\n')
                
                # 记录第一行用于稍后展示
                if i == 0:
                    first_record = entry

        print(f"✅ 转换成功！文件已保存至: {dest}")
        
        # 5. 打印转换后的第一个数据（为了方便查看，打印时使用了缩进）
        print("\n" + "="*40)
        print("👀 预览：第一条数据结构 (JSONL 第一行)")
        print("="*40)
        print(json.dumps(first_record, indent=4, ensure_ascii=False, cls=NumpyEncoder))

    except Exception as e:
        print(f"❌ 发生错误: {e}")

if __name__ == "__main__":
    process_and_save_jsonl(input_path, output_path)