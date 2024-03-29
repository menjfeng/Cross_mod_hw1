import os
import shutil



#根绝图片名按照小数点将其分割成对应的猫狗数据集
def data_dir(source_dir, target_dir):
    # 获取图片文件列表
    image_files = os.listdir(source_dir)

    # 遍历图片文件
    for image_file in image_files:
        # 获取图片完整路径
        image_path = os.path.join(source_dir, image_file)

        # 根据图片名称进行分类
        filename_parts = image_file.split(".")
        if len(filename_parts) > 1:
            sub_dir = filename_parts[0]
            new_filename = ".".join(filename_parts[1:])
        else:
            sub_dir = "other"
            new_filename = image_file

        # 分类目录路径
        category_dir = os.path.join(target_dir, sub_dir)

        # 创建分类目录（如果不存在）
        os.makedirs(category_dir, exist_ok=True)

        # 目标文件路径
        target_path = os.path.join(category_dir, new_filename)

        # 将图片移动到分类目录下，并修改图片的新名称
        # shutil.move(image_path, target_path)



if __name__ == '__main__':
    data_dir(source_dir="data/train", target_dir="data/train")
    data_dir(source_dir="data/val", target_dir="data/val")