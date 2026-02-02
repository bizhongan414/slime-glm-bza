import json

# 文件路径
file_path = '/gfs/space/chatrl/users/wlw_temp/wlw/data/slime/humaneval_codeinmd/humaneval_codeinmd.jsonl' #'/gfs/space/chatrl/users/lyy/data/code/livecodebench/sp_none_prompt_none/v6.jsonl'

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        # 读取第一行
        first_line = f.readline()
        
        # 增加 strip() 去除首尾空白符（主要是换行符 \n），虽然 json.loads 通常能容忍，但这样更稳健
        if first_line and first_line.strip():
            # 解析
            first_data = json.loads(first_line)
            
            print("=== 第一行数据预览 ===")
            # 打印数据
            print(json.dumps(first_data, indent=4, ensure_ascii=False))
        else:
            print("文件是空的或第一行是空行。")
            
except FileNotFoundError:
    print(f"找不到文件: {file_path}")
except json.JSONDecodeError as e:
    print(f"JSON 解析错误 (可能这一行不是合法的 JSON): {e}")
except Exception as e:
    print(f"发生错误: {e}")