Here is the updated code with all comments and output messages translated into English:

```python
# # Usage example
# import json
#
# # Read JSON file
# with open(r"zhiwubindu\captions\train\merged_questionnaires.json", "r", encoding="utf-8") as f:
#     data = json.load(f)
#
# # Ensure data is a list
# if isinstance(data, list):
#     converted_data = []
#     id_counter = 0  # ID counter
#
#     for entry in data:  # Iterate through each element in the list
#         if isinstance(entry, dict) and "content" in entry:
#             content = entry["content"]  # Get content dictionary
#             if isinstance(content, dict):
#                 for file_name, details in content.items():
#                     if isinstance(details, dict) and "text" in details:
#                         converted_entry = {
#                             "id": id_counter,  # Increment ID
#                             "file_path": file_name,
#                             "captions": [details["text"]],
#                             "captions_bt": [details["text"]]  # Temporarily duplicate captions_bt here
#                         }
#                         converted_data.append(converted_entry)
#                         id_counter += 1
#
# # Save converted JSON
# with open(r"zhiwubindu\captions\train\converted_questionnaires.json", "w", encoding="utf-8") as f:
#     json.dump(converted_data, f, ensure_ascii=False, indent=4)
#
# print("Conversion complete, data saved to converted_questionnaires.json")

# import json
# import os
#
# # Define folder path
# folder_path = r"zhiwubindu\captions\train"  # Your folder containing JSON files
# output_file = r"zhiwubindu\captions\train\merged.json"  # Final merged JSON filename
#
# # Store merged data
# merged_data = []
#
# # Iterate through all JSON files in the folder
# for filename in os.listdir(folder_path):
#     if filename.endswith(".json"):  # Process only JSON files
#         file_path = os.path.join(folder_path, filename)
#
#         # Read JSON file content
#         with open(file_path, "r", encoding="utf-8") as f:
#             try:
#                 data = json.load(f)  # Parse JSON file
#                 if isinstance(data, list):
#                     merged_data.extend(data)  # If JSON is a list, extend merged_data
#                 else:
#                     merged_data.append(data)  # If JSON is a dictionary, append directly
#             except json.JSONDecodeError as e:
#                 print(f"Failed to parse {filename}: {e}")
#
# # Save merged data to new JSON file
# with open(output_file, "w", encoding="utf-8") as f:
#     json.dump(merged_data, f, ensure_ascii=False, indent=4)
#
# print(f"Merge complete, data saved to {output_file}")
import json

# Read JSON file
with open(r"zhiwubindu\captions\train\merged.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Store converted data
converted_data = []
id_counter = 0  # For numbering

# Iterate through data and convert format
for entry in data:
    if isinstance(entry, dict):  # Ensure entry is a dictionary
        for file_name, details in entry.items():
            if isinstance(details, dict) and "text" in details and "label" in details:
                converted_entry = {
                    "id": details["label"],  # Map label to id
                    "file_path": file_name,  # Image file name
                    "captions": [details["text"]],  # Map text to captions
                    "captions_bt": [details["text"]]  # Temporarily duplicate captions_bt here
                }
                converted_data.append(converted_entry)
                id_counter += 1  # Increment ID

# Save converted JSON
with open(r"zhiwubindu\captions\train\converted.json", "w", encoding="utf-8") as f:
    json.dump(converted_data, f, ensure_ascii=False, indent=4)

print("Conversion complete, data saved to converted.json")

```
