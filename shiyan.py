#
# # 使用示例
# import json
#
# # 读取 JSON 文件
# with open(r"zhiwubindu\captions\train\merged_questionnaires.json", "r", encoding="utf-8") as f:
#     data = json.load(f)
#
# # 确保数据是一个列表
# if isinstance(data, list):
#     converted_data = []
#     id_counter = 0  # 计数ID
#
#     for entry in data:  # 遍历列表中的每个元素
#         if isinstance(entry, dict) and "content" in entry:
#             content = entry["content"]  # 获取content字典
#             if isinstance(content, dict):
#                 for file_name, details in content.items():
#                     if isinstance(details, dict) and "text" in details:
#                         converted_entry = {
#                             "id": id_counter,  # 递增ID
#                             "file_path": file_name,
#                             "captions": [details["text"]],
#                             "captions_bt": [details["text"]]  # 这里暂时复制 captions_bt
#                         }
#                         converted_data.append(converted_entry)
#                         id_counter += 1
#
# # 保存转换后的 JSON
# with open(r"zhiwubindu\captions\train\converted_questionnaires.json", "w", encoding="utf-8") as f:
#     json.dump(converted_data, f, ensure_ascii=False, indent=4)
#
# print("转换完成，数据已保存到 converted_questionnaires.json")

# import json
# import os
#
# # 定义文件夹路径
# folder_path = r"zhiwubindu\captions\train"  # 你的 JSON 文件所在的文件夹
# output_file = r"zhiwubindu\captions\train\merged.json"  # 最终合并后的 JSON 文件名
#
# # 存储合并后的数据
# merged_data = []
#
# # 遍历文件夹下所有 JSON 文件
# for filename in os.listdir(folder_path):
#     if filename.endswith(".json"):  # 只处理 JSON 文件
#         file_path = os.path.join(folder_path, filename)
#
#         # 读取 JSON 文件内容
#         with open(file_path, "r", encoding="utf-8") as f:
#             try:
#                 data = json.load(f)  # 解析 JSON 文件
#                 if isinstance(data, list):
#                     merged_data.extend(data)  # 如果 JSON 是列表，扩展到 merged_data
#                 else:
#                     merged_data.append(data)  # 如果 JSON 是字典，直接追加
#             except json.JSONDecodeError as e:
#                 print(f"解析 {filename} 失败: {e}")
#
# # 将合并后的数据保存到新 JSON 文件
# with open(output_file, "w", encoding="utf-8") as f:
#     json.dump(merged_data, f, ensure_ascii=False, indent=4)
#
# print(f"合并完成，数据已保存到 {output_file}")
import json

# 读取 JSON 文件
with open(r"zhiwubindu\captions\train\merged.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 存储转换后的数据
converted_data = []
id_counter = 0  # 用于编号

# 遍历数据并转换格式
for entry in data:
    if isinstance(entry, dict):  # 确保 entry 是字典
        for file_name, details in entry.items():
            if isinstance(details, dict) and "text" in details and "label" in details:
                converted_entry = {
                    "id": details["label"],  # label 变 id
                    "file_path": file_name,  # 图片文件名
                    "captions": [details["text"]],  # text 变 captions
                    "captions_bt": [details["text"]]  # 这里暂时复制 captions_bt
                }
                converted_data.append(converted_entry)
                id_counter += 1  # 递增 ID

# 保存转换后的 JSON
with open(r"zhiwubindu\captions\train\converted.json", "w", encoding="utf-8") as f:
    json.dump(converted_data, f, ensure_ascii=False, indent=4)

print("转换完成，数据已保存到 converted.json")
#
