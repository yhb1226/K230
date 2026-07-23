def convert_cn_punct_to_en(file_path):
    # 中文→英文标点映射表
    mapping = {
        "，": ",", "。": ".", "：": ":", "；": ";",
        "（": "(", "）": ")", "【": "[", "】": "]",
        "“": "\"", "”": "\"", "‘": "'", "’": "'",
        "？": "?", "！": "!"
    }
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    # 逐个替换
    for cn, en in mapping.items():
        text = text.replace(cn, en)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"已完成标点修复：{file_path}")

# 修改成你的代码文件路径
convert_cn_punct_to_en("./demo_camera_find_rects.py")