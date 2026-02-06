# import json
# import random
# import nltk
# import asyncio
# import time
# from deep_translator import GoogleTranslator
# from nltk.corpus import wordnet
#
# # If NLTK download fails, manually specify local path
# nltk.data.path.append("D:/nltk_data")  # You can modify this to your path
#
# # Read JSON file
# input_file = r"zhiwubindu\captions\train\converted.json"
# output_file = r"zhiwubindu\captions\train\converted_augment.json"
#
# with open(input_file, "r", encoding="utf-8") as f:
#     data = json.load(f)
#
# translator = GoogleTranslator()
#
# # **Asynchronous Translation**
# async def async_translate(text, src="en", dest="fr"):
#     """Asynchronous text translation"""
#     try:
#         time.sleep(1)  # Reduce request frequency to avoid Google rate limiting
#         translation = await translator.translate(text, src=src, dest=dest)
#         return translation.text
#     except Exception as e:
#         print(f"Back translation failed: {e}")
#         return text  # Return original text on failure
#
# # **Back Translation**
# def back_translate(text, src="en", mid="fr"):
#     """Use Deep Translator for back translation (stable)"""
#     try:
#         time.sleep(1)  # Avoid API rate limits
#         french_text = GoogleTranslator(source=src, target=mid).translate(text)
#         time.sleep(1)
#         back_translated_text = GoogleTranslator(source=mid, target=src).translate(french_text)
#         return back_translated_text
#     except Exception as e:
#         print(f"Back translation failed: {e}")
#         return text  # Return original text on failure
#
# # **Synonym Replacement**
# def synonym_replacement(text, n=2):
#     words = text.split()
#     for _ in range(n):
#         word_candidates = [w for w in words if wordnet.synsets(w)]
#         if not word_candidates:
#             continue  # Skip words with no synonyms found
#         word_to_replace = random.choice(word_candidates)
#         synonyms = wordnet.synsets(word_to_replace)
#         if synonyms:
#             new_word = synonyms[0].lemmas()[0].name().replace('_', ' ')
#             words = [new_word if w == word_to_replace else w for w in words]
#     return " ".join(words)
#
# # **Random Insertion**
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
# # **Random Swap**
# def random_swap(text, n=2):
#     words = text.split()
#     for _ in range(n):
#         if len(words) < 2:
#             continue
#         idx1, idx2 = random.sample(range(len(words)), 2)
#         words[idx1], words[idx2] = words[idx2], words[idx1]
#     return " ".join(words)
#
# # **Random Deletion**
# def random_deletion(text, p=0.3):
#     words = text.split()
#     if len(words) <= 1:  # Avoid deleting all words
#         return text
#     new_words = [word for word in words if random.uniform(0, 1) > p]
#     return " ".join(new_words) if new_words else text
#
# # **EDA Augmentation**
# def eda_augmentation(text):
#     methods = [synonym_replacement, random_insertion, random_swap, random_deletion]
#     num_augmentations = random.randint(1, 2)  # At least 1-2 augmentations per sentence
#     for _ in range(num_augmentations):
#         method = random.choice(methods)
#         text = method(text)
#     return text
#
# # **Process JSON Data**
# for item in data:
#     new_captions = []
#     for caption in item["captions"]:
#         # print(1)
#         augmented_caption = back_translate(caption, "fr")  # Perform back translation first
#         augmented_caption = eda_augmentation(augmented_caption)  # Then perform EDA augmentation
#         new_captions.append(augmented_caption)
#     print(item["id"])
#     item["captions_bt"] = new_captions  # Store augmented text
#
# # **Save Augmented JSON**
# with open(output_file, "w", encoding="utf-8") as f:
#     json.dump(data, f, ensure_ascii=False, indent=4)
#
# print(f"Augmentation complete, data saved to {output_file}")

import json
import time
import random
from deep_translator import GoogleTranslator

# Input / Output file paths
input_file = r"zhiwubindu\captions\train\converted.json"
output_file = r"zhiwubindu\captions\train\converted_augment.json"

# Read JSON data
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)


# **Back Translation Function**
def back_translate(text, src="en", retries=3):
    """Back translation using Google Translate with random intermediate language selection and retry support"""
    mid_langs = ["fr", "de", "es"]  # Possible intermediate languages (French, German, Spanish)

    for attempt in range(retries):
        try:
            mid = random.choice(mid_langs)  # Randomly select an intermediate language
            time.sleep(1)  # Avoid API rate limits

            translated_text = GoogleTranslator(source=src, target=mid).translate(text)
            time.sleep(1)
            back_translated_text = GoogleTranslator(source=mid, target=src).translate(translated_text)

            # **Check for changes**
            if text == back_translated_text:
                print(f"⚠️ Back translation unchanged: {text} → {back_translated_text}")
            else:
                print(f"✅ Back translation successful: {text} → {back_translated_text}")

            return back_translated_text  # Return back-translated text

        except Exception as e:
            print(f"❌ Back translation failed (Attempt {attempt + 1}): {e}")
            time.sleep(2)  # Wait before retrying after failure

    print(f"⏳ Final back translation failed, returning original text: {text}")
    return text  # Return original text after multiple failures


# **Process JSON Data**
for item in data:
    new_captions = []
    for caption in item["captions"]:
        augmented_caption = back_translate(caption, src="en")  # Perform back translation
        new_captions.append(augmented_caption)

    print(f"📌 Processing ID: {item['id']}")  # Print processing progress
    item["captions_bt"] = new_captions  # Store back-translated text

# **Save Augmented JSON**
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"🎉 Back translation complete, data saved to {output_file}")
