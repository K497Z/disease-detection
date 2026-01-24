import os
import shutil

# 设定主目录
train_dir = r"zhiwubindu\images\test"  # 你的 train 目录路径

# 遍历 train 目录下的所有子文件夹
for subdir in os.listdir(train_dir):
    subdir_path = os.path.join(train_dir, subdir)

    # 确保是文件夹
    if os.path.isdir(subdir_path):
        for filename in os.listdir(subdir_path):
            file_path = os.path.join(subdir_path, filename)

            # 检查是否是图片文件（可根据实际情况调整）
            if os.path.isfile(file_path) and filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
                new_path = os.path.join(train_dir, filename)  # 目标路径

                # 如果文件名重复，重命名避免覆盖
                counter = 1
                while os.path.exists(new_path):
                    name, ext = os.path.splitext(filename)
                    new_path = os.path.join(train_dir, f"{name}_{counter}{ext}")
                    counter += 1

                shutil.move(file_path, new_path)  # 移动文件

        # 删除空子文件夹
        os.rmdir(subdir_path)

print("所有图片已移动到 train 目录，并删除了空子文件夹。")
