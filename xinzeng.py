# import json
# import random
# import nltk
# import asyncio
# import time
# from deep_translator import GoogleTranslator
# from nltk.corpus import wordnet
#
# # 如果 NLTK 下载失败，手动指定本地路径
# nltk.data.path.append("D:/nltk_data")  # 你可以修改为你的路径
#
# # 读取 JSON 文件
# input_file = r"zhiwubindu\captions\train\converted.json"
# output_file = r"zhiwubindu\captions\train\converted_augment.json"
#
# with open(input_file, "r", encoding="utf-8") as f:
#     data = json.load(f)
#
# translator = GoogleTranslator()
#
# # **异步翻译**
# async def async_translate(text, src="en", dest="fr"):
#     """异步翻译文本"""
#     try:
#         time.sleep(1)  # 降低请求频率，避免 Google 限流
#         translation = await translator.translate(text, src=src, dest=dest)
#         return translation.text
#     except Exception as e:
#         print(f"回译失败: {e}")
#         return text  # 失败时返回原文本
#
# # **回译**
# def back_translate(text, src="en", mid="fr"):
#     """使用 Deep Translator 进行回译（稳定）"""
#     try:
#         time.sleep(1)  # 避免 API 速率限制
#         french_text = GoogleTranslator(source=src, target=mid).translate(text)
#         time.sleep(1)
#         back_translated_text = GoogleTranslator(source=mid, target=src).translate(french_text)
#         return back_translated_text
#     except Exception as e:
#         print(f"回译失败: {e}")
#         return text  # 失败时返回原文本
#
# # **同义词替换**
# def synonym_replacement(text, n=2):
#     words = text.split()
#     for _ in range(n):
#         word_candidates = [w for w in words if wordnet.synsets(w)]
#         if not word_candidates:
#             continue  # 没有找到同义词的单词则跳过
#         word_to_replace = random.choice(word_candidates)
#         synonyms = wordnet.synsets(word_to_replace)
#         if synonyms:
#             new_word = synonyms[0].lemmas()[0].name().replace('_', ' ')
#             words = [new_word if w == word_to_replace else w for w in words]
#     return " ".join(words)
#
# # **随机插入**
# def random_insertion(text, n=2):
#     words = text.split()
#     for _ in range(n):
#         word_candidates = [w for w in words if wordnet.synsets(w)]
#         if not word_candidates:
#             continue
#         word_to_insert = random.choice(word_candidates)
#         synonyms = wordnet.synsets(word_to_insert)
#         if synonyms:
#             new_word = synonyms[0].lemmas()[0].name().replace('_', ' ')
#             insert_pos = random.randint(0, len(words))
#             words.insert(insert_pos, new_word)
#     return " ".join(words)
#
# # **随机交换**
# def random_swap(text, n=2):
#     words = text.split()
#     for _ in range(n):
#         if len(words) < 2:
#             continue
#         idx1, idx2 = random.sample(range(len(words)), 2)
#         words[idx1], words[idx2] = words[idx2], words[idx1]
#     return " ".join(words)
#
# # **随机删除**
# def random_deletion(text, p=0.3):
#     words = text.split()
#     if len(words) <= 1:  # 避免删除所有单词
#         return text
#     new_words = [word for word in words if random.uniform(0, 1) > p]
#     return " ".join(new_words) if new_words else text
#
# # **EDA 增强**
# def eda_augmentation(text):
#     methods = [synonym_replacement, random_insertion, random_swap, random_deletion]
#     num_augmentations = random.randint(1, 2)  # 每个句子至少 1-2 次增强
#     for _ in range(num_augmentations):
#         method = random.choice(methods)
#         text = method(text)
#     return text
#
# # **处理 JSON 数据**
# for item in data:
#     new_captions = []
#     for caption in item["captions"]:
#         # print(1)
#         augmented_caption = back_translate(caption, "fr")  # 先进行回译
#         augmented_caption = eda_augmentation(augmented_caption)  # 然后进行 EDA 增强
#         new_captions.append(augmented_caption)
#     print(item["id"])
#     item["captions_bt"] = new_captions  # 存入增强后的文本
#
# # **保存增强后的 JSON**
# with open(output_file, "w", encoding="utf-8") as f:
#     json.dump(data, f, ensure_ascii=False, indent=4)
#
# print(f"增强完成，数据已保存到 {output_file}")

import json
import time
import random
from deep_translator import GoogleTranslator

# 输入 / 输出文件路径
input_file = r"zhiwubindu\captions\train\converted.json"
output_file = r"zhiwubindu\captions\train\converted_augment.json"

# 读取 JSON 数据
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)


# **回译函数**
def back_translate(text, src="en", retries=3):
    """使用 Google 翻译进行回译，并随机选择中间语言，支持失败重试"""
    mid_langs = ["fr", "de", "es"]  # 可能的中间语言（法语、德语、西班牙语）

    for attempt in range(retries):
        try:
            mid = random.choice(mid_langs)  # 随机选择一个中间语言
            time.sleep(1)  # 避免 API 速率限制

            translated_text = GoogleTranslator(source=src, target=mid).translate(text)
            time.sleep(1)
            back_translated_text = GoogleTranslator(source=mid, target=src).translate(translated_text)

            # **检测是否有变化**
            if text == back_translated_text:
                print(f"⚠️ 回译无变化: {text} → {back_translated_text}")
            else:
                print(f"✅ 回译成功: {text} → {back_translated_text}")

            return back_translated_text  # 返回回译后的文本

        except Exception as e:
            print(f"❌ 第 {attempt + 1} 次回译失败: {e}")
            time.sleep(2)  # 失败后等待再尝试

    print(f"⏳ 最终回译失败，返回原文本: {text}")
    return text  # 失败多次后返回原文本


# **处理 JSON 数据**
for item in data:
    new_captions = []
    for caption in item["captions"]:
        augmented_caption = back_translate(caption, src="en")  # 进行回译
        new_captions.append(augmented_caption)

    print(f"📌 处理 ID: {item['id']}")  # 打印处理进度
    item["captions_bt"] = new_captions  # 存入回译后的文本

# **保存增强后的 JSON**
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"🎉 回译完成，数据已保存到 {output_file}")
