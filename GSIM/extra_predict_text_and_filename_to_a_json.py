import os
import json
from collections import OrderedDict

# ================== 配置路径 ==================
# original_data_json = '/mnt/SSD/cyx/LLaMA-Factory/data/qwen2_5vl_color_caption_grayscale_pairs_train.json'
original_data_json = '/mnt/SSD/cyx/LLaMA-Factory/data/qwen2_5vl_color_caption_grayscale_pairs_test_real.json'
# qwen_lora_eval_json_path = "/mnt/SSD/cyx/LLaMA-Factory/saves/Qwen2.5-VL-7B-Instruct/lora/eval_2025-09-11-15-08-20_text_based_1944test/generated_predictions.jsonl"  #real text step1400
# qwen_lora_eval_json_path = "/mnt/SSD/cyx/LLaMA-Factory/saves/Qwen2.5-VL-7B-Instruct/lora/eval_2025-09-11-03-49-41_auto_1944test/generated_predictions.jsonl" #real auto step500
qwen_lora_eval_json_path = "/mnt/SSD/cyx/LLaMA-Factory/saves/Qwen2.5-VL-7B-Instruct/lora/eval_2025-12-09-07-37-47-pred_coco_material/generated_predictions.jsonl" #coco style 1500

output_dir = "/mnt/SSD/cyx/DATASET/COCO_color/"
# output_txt = os.path.join(output_dir, "all_captions_sorted.txt")   # 纯文本，每行一个 caption
# output_json = os.path.join(output_dir, "color_captions_sorted_by_filename_real_train_pred_1944test_text_based_step1400.json")  # 保存为有序 JSON text
# output_json = os.path.join(output_dir, "color_captions_sorted_by_filename_real_train_pred_1944test_auto_step500.json")  # 保存为有序 JSON auto
output_json = os.path.join(output_dir, "coco_val1944_material_pred_step1500.json")  # 保存为有序 JSON auto

# os.makedirs(output_dir, exist_ok=True)

# ================== 步骤1: 从原始 JSON 提取文件名并排序 ==================
filename_to_predict = OrderedDict()  # 保持按文件名数字排序的顺序

with open(original_data_json, 'r', encoding='utf-8') as f:
    data_list = json.load(f)

# 提取每个样本的文件名（不含扩展名），并按数字排序
file_info_list = []
for idx, item in enumerate(data_list):
    image_path = item["images"][0]
    filename = os.path.basename(image_path)
    file_id = os.path.splitext(filename)[0]  # 如 '000000331352'
    file_id_num = int(file_id)
    file_info_list.append((file_id_num, file_id, idx))  # (数字ID, 文件名, 原始索引)

# 按数字 ID 排序
file_info_list.sort(key=lambda x: x[0])

# ================== 步骤2: 从 jsonl 读取所有 predict 文本 ==================
predictions = []
with open(qwen_lora_eval_json_path, 'r', encoding='utf-8') as f:
    for line_idx, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
            predict_text = item.get("predict", "").strip()
            predictions.append(predict_text)
        except json.JSONDecodeError as e:
            print(f"第 {line_idx} 行 JSON 解析失败：{e}")
            predictions.append("")

# ================== 步骤3: 按排序后的文件名顺序构建有序字典 ==================
for file_id_num, file_id, original_idx in file_info_list:
    if original_idx < len(predictions):
        caption = predictions[original_idx]
        filename_to_predict[file_id] = caption
    else:
        filename_to_predict[file_id] = ""

# ================== 步骤4: 保存结果 ==================
# # 保存为纯文本（每行一个 caption，按文件名排序）
# with open(output_txt, 'w', encoding='utf-8') as f:
#     for caption in filename_to_predict.values():
#         f.write(caption + '\n')

# 保存为 JSON（保持顺序）
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(filename_to_predict, f, ensure_ascii=False, indent=2)

print(f"✅ 预测文本已按文件名（数字）排序并保存：")
# print(f"    文本文件: {output_txt}")
print(f"    JSON 文件: {output_json}")
print(f"    共 {len(filename_to_predict)} 条数据")
