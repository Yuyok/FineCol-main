import os
import json

# caption_mapping_json = '/mnt/SSD/cyx/L-CAD-main/resources/coco_color/test_no_black_and_white.json'  # 划分coco val part 过滤黑白
# caption_mapping_json = '/mnt/SSD/cyx/L-CAD-main/resources/coco_color/train_no_black_and_white.json' # 划分coco val part 过滤黑白
caption_mapping_json = '/mnt/SSD/cyx/L-CAD-main/resources/coco_color/test_real_no_black_and_white.json'  # coco trian part 过滤黑白（可能不完整）
# caption_mapping_json = '/mnt/SSD/cyx/L-CAD-main/resources/cap-img-pairs_val_simple_7k.json'  # multi-instances
# 配置路径
gray_image_dir = '/mnt/SSD/cyx/DATASET/COCO_color/val_1944_gray' #'/mnt/SSD/cyx/DATASET/instance_color/val_gray_7213_resized_512' #'/mnt/SSD/cyx/DATASET/COCO_color/val_1944_gray' #'/mnt/SSD/cyx/DATASET/COCO_color/train_22748_gray'  #'/mnt/SSD/cyx/DATASET/COCO_color/val_1944_gray'  # 存放灰度图像的路径
caption_from_color_dir = '/mnt/SSD/cyx/CogVLM2/output/final_merged_json/val/' #'/mnt/SSD/cyx/CogVLM2/output/final_merged_json/val_instances' # '/mnt/SSD/cyx/CogVLM2/output/final_merged_json/train/'  # 存放JSON描述的路径
# COCO val2017 part
# output_json = '/mnt/SSD/cyx/LLaMA-Factory/data/qwen2_5vl_color_caption_grayscale_pairs_test.json'   # grayscale to color caption (automatic)
output_json = '/mnt/SSD/cyx/LLaMA-Factory/data/coco_val1944_light.json'     # grayscale to color caption (based on coco short caption)
# output_json = '/mnt/SSD/cyx/LLaMA-Factory/data/instances_val_material.json'
# COCO train2017 part
# output_json = '/mnt/SSD/cyx/LLaMA-Factory/data/qwen2_5vl_color_caption_grayscale_pairs_train_real.json'     # grayscale to color caption (automatic)
# output_json = '/mnt/SSD/cyx/LLaMA-Factory/data/qwen2_5vl_color_caption_grayscale_pairs_train_text_based_real.json'     # grayscale to color caption (based on coco short caption)



# 读取包含文件名到文本描述映射的JSON文件
with open(caption_mapping_json, 'r') as f:
    file_to_captions = json.load(f)  # 获取所有需要处理的图像文件名和对应的描述


data = []
for image_name in file_to_captions.keys():  # 根据新增JSON文件中的键值筛选图像
    base_name = image_name.split('.')[0]
    image_path = os.path.join(gray_image_dir, image_name.replace('.jpg', '.png'))
    json_path = os.path.join(caption_from_color_dir, f"{base_name}.json")

    if not os.path.exists(json_path):
        print(f"[跳过] 没找到描述: {json_path}")
        continue

    with open(json_path, 'r') as f:
        print(json_path)
        caption_from_color = json.load(f)
        if isinstance(caption_from_color, dict):
            caption_from_color_long = caption_from_color['whole_prompt']['COLOR']
            caption_from_coco_short = caption_from_color['short_prompts']

            caption_from_style_long = caption_from_color['whole_prompt']['STYLE']
            caption_from_light_long = caption_from_color['whole_prompt']['LIGHT']
            caption_from_material_long = caption_from_color['whole_prompt']['MATERIAL']
            # print(caption_from_coco_short)
        else:
            print('caption_from_color has no instance')
            continue

    # 用于灰度图生成COLOR  自动
    # example = {
    #     "messages": [
    #         {
    #             "content": "<image>[GrayColorExplain] Describe the real-world colors of each object in this grayscale photo. Write a concise, objective, and visually grounded description of the image in full colors. Describe each object exactly once, using realistic and semantically appropriate colors.",
    #             "role": "user"
    #         },
    #         {
    #             "content": caption_from_color_long.strip(),
    #             "role": "assistant"
    #         }
    #     ],
    #     "images": [image_path]
    # }

    # 用于灰度图生成COLOR  给定 COCO的短caption
    # example = {
    #     "messages": [
    #         {
    #             "content": f"<image>[GrayColorExplainFromGivenPrompt] You are given a grayscale image and some related captions from the original colorful image: ```{caption_from_coco_short}```, these captions provided important color clues. You need to reference them and Describe the real-world colors of each object in this grayscale photo. Write a concise, objective, and visually grounded description of the image in full colors. Describe each object exactly once, using realistic and semantically appropriate colors.",
    #             "role": "user"
    #         },
    #         {
    #             "content": caption_from_color_long.strip(),
    #             "role": "assistant"
    #         }
    #     ],
    #     "images": [image_path]
    # }


    # 用于灰度图生成STYLE
    # example = {
    #     "messages": [
    #         {
    #             "content": f"<image>[GrayStyleExplain] Describe the real-world style of this grayscale photo if it is a colorful image in real world. Write a concise, overall, and visually grounded description of the image in appropriate style.",
    #             "role": "user"
    #         },
    #         {
    #             "content": caption_from_style_long.strip(),
    #             "role": "assistant"
    #         }
    #     ],
    #     "images": [image_path]
    # }

    # 用于灰度图生成LIGHT
    example = {
        "messages": [
            {
                "content": f"<image>[GrayLightExplain] Describe the real-world lighting of this grayscale photo if it is a colorful image in real world. Write a concise, overall, and visually grounded description of the image in appropriate lighting.",
                "role": "user"
            },
            {
                "content": caption_from_light_long.strip(),
                "role": "assistant"
            }
        ],
        "images": [image_path]
    }

    # 用于灰度图生成MATERIAL
    # example = {
    #     "messages": [
    #         {
    #             "content": f"<image>[GrayMaterialExplain] Describe the real-world material of each object in this grayscale photo. Write a concise, objective, and visually grounded description of the image in full colors. Describe each object exactly once, using realistic and semantically appropriate material.",
    #             "role": "user"
    #         },
    #         {
    #             "content": caption_from_material_long.strip(),
    #             "role": "assistant"
    #         }
    #     ],
    #     "images": [image_path]
    # }
    print(example)


    data.append(example)

# 保存 JSON
with open(output_json, 'w') as f:
    json.dump(data, f, indent=2)

print(f"已生成数据，共 {len(data)} 条，保存为 {output_json}")
