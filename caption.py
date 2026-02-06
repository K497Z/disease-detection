import json
import re

# Read converted.json
# input_file = r"plantdoc\captions\test\estconverted.json"
input_file = r".\plantdoc\captions\train\converted_augment.json"
output_file = r".\plantdoc\captions\train\converted_reformatted.json"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

step_pattern = re.compile(r"Step\s*\d+\s*:\s*", re.IGNORECASE)


def process_caption(text):
    """General function to process captions and captions_bt uniformly"""
    if step_pattern.search(text):
        steps = step_pattern.split(text)[1:]
        if len(steps) >= 7:
            part1 = " ".join(steps[:3]).strip()
            part2 = " ".join(steps[3:7]).strip()
        else:
            mid = max(len(steps) // 2, 1)
            part1 = " ".join(steps[:mid]).strip()
            part2 = " ".join(steps[mid:]).strip()
    else:
        words = text.split()
        mid = len(words) // 2 if len(words) >= 20 else 0
        part1 = " ".join(words[:mid]).strip() if mid else text
        part2 = " ".join(words[mid:]).strip() if mid else text
    return [part1, part2]


for item in data:
    # Process captions
    original_captions = item["captions"][0]
    item["captions"] = process_caption(original_captions)

    # Process captions_bt (maintain the same structure)
    if "captions_bt" in item and item["captions_bt"]:
        translated_captions = item["captions_bt"][0]
        item["captions_bt"] = process_caption(translated_captions)
    else:
        item["captions_bt"] = ["", ""]  # Handle empty values

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"Conversion complete! New file saved as {output_file}")
