# import json
# import os
#
# # Define folder path
# folder_path = r"zhiwubindu\captions\test"  # Your folder containing JSON files
# output_file = r"zhiwubindu\captions\test\merged.json"  # Final merged JSON filename
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
#                     merged_data.append(data)  # If JSON is a dict, append directly
#             except json.JSONDecodeError as e:
#                 print(f"Failed to parse {filename}: {e}")
#
# # Save merged data to a new JSON file
# with open(output_file, "w", encoding="utf-8") as f:
#     json.dump(merged_data, f, ensure_ascii=False, indent=4)
#
# print(f"Merge complete, data saved to {output_file}")
import json

# Read JSON file
with open(r"zhiwubindu\captions\test\merged.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Store converted data
converted_data = []
id_counter = 0  # Counter for ID

# Iterate through data and convert format
for entry in data:
    if isinstance(entry, dict):  # Ensure entry is a dictionary
        for file_name, details in entry.items():
            if isinstance(details, dict) and "text" in details and "label" in details:
                converted_entry = {
                    "id": details["label"],  # Map label to id
                    "file_path": file_name,  # Image file name
                    "captions": [details["text"]]  # Map text to captions
                }
                converted_data.append(converted_entry)
                id_counter += 1  # Increment ID

# Save converted JSON
with open(r"zhiwubindu\captions\test\converted.json", "w", encoding="utf-8") as f:
    json.dump(converted_data, f, ensure_ascii=False, indent=4)

print("Conversion complete, data saved to converted.json")
