import json
import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch



def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    # init_extra_nodes()
    # Replaced init_extra_nodes() with:
    loop.run_until_complete(init_extra_nodes())
    loop.close()


from nodes import NODE_CLASS_MAPPINGS



def get_filename_list_to_test(file_path):
    """加载JSON文件并返回内容"""
    with open(file_path, 'r') as f:
        data = json.load(f)
        filename_list_to_test = data.keys()

    return filename_list_to_test

def load_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


import os

split_str = os.getenv("SPLIT")
if split_str is None:
    print("Error: Please set SPLIT before running, e.g.:")
    print("       SPLIT=1 CUDA_VISIBLE_DEVICES=1 python test_all.py")
    sys.exit(1)

split_num = int(split_str)
print(f"Split = {split_num}")

def main():
    import_custom_nodes()


    gray_input_dir = '/mnt/SSD/cyx/DATASET/COCO_color/val_1944_gray'
    whole_captions_dir = '/mnt/SSD/cyx/CogVLM2/output/final_merged_json/val'
    if split_num == -1:
        filename_list_to_test = get_filename_list_to_test('/mnt/SSD/cyx/L-CAD-main/resources/coco_color/test_real_no_black_and_white.json')
    else:
        filename_list_to_test = get_filename_list_to_test(f'/mnt/SSD/cyx/L-CAD-main/resources/coco_color/split_test_captions/part{split_num}.json')

    input_color_caption_pred_from_gray_dir = '/mnt/SSD/cyx/DATASET/COCO_color/color_captions_sorted_by_filename.json'
    input_style_caption_pred_from_gray_dir = '/mnt/SSD/cyx/DATASET/COCO_color/coco_val_style_pred.json'
    input_light_caption_pred_from_gray_dir = '/mnt/SSD/cyx/DATASET/COCO_color/coco_val_light_pred.json'
    input_material_caption_pred_from_gray_dir = '/mnt/SSD/cyx/DATASET/COCO_color/coco_val_material_pred.json'

    with torch.inference_mode():
        unetloader = NODE_CLASS_MAPPINGS["UNETLoader"]()
        unetloader_37 = unetloader.load_unet(
            unet_name="flux1-dev-kontext_fp8_scaled.safetensors", weight_dtype="default"
        )

        dualcliploader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
        dualcliploader_38 = dualcliploader.load_clip(
            clip_name1="clip_l.safetensors",
            clip_name2="t5xxl_fp8_e4m3fn_scaled.safetensors",
            type="flux",
            device="default",
        )

        loraloader = NODE_CLASS_MAPPINGS["LoraLoader"]()
        loraloader_193 = loraloader.load_lora(
            # lora_name="other_caption_colorization-12-18109.safetensors",
            lora_name="COLOR_real-4-25590.safetensors",
            # lora_name="COLOR_real-1-10236.safetensors",
            # lora_name="COLOR_real-0-5118.safetensors",
            # lora_name="no_caption_colorization-8-12537.safetensors",
            # lora_name="colorization-9-13930.safetensors",

            strength_model=1.0,#1.0,  #0.95
            strength_clip=1.0,#1.0,   #0.95
            model=get_value_at_index(unetloader_37, 0),
            clip=get_value_at_index(dualcliploader_38, 0),
        )
        loraloader_238 = loraloader.load_lora(
            lora_name="STYLE_real-4-25590.safetensors",
            # lora_name="STYLE_real-0-5118.safetensors",
            strength_model=0.1,
            strength_clip=0.1,
            model=get_value_at_index(loraloader_193, 0),
            clip=get_value_at_index(loraloader_193, 1),
        )

        loraloader_239 = loraloader.load_lora(
            lora_name="LIGHT_real-4-25590.safetensors",
            # lora_name="LIGHT_real-0-5118.safetensors",
            strength_model=0.1,
            strength_clip=0.1,
            model=get_value_at_index(loraloader_238, 0),
            clip=get_value_at_index(loraloader_238, 1),
        )

        loraloader_240 = loraloader.load_lora(
            lora_name="MATERIAL_real-4-25590.safetensors",
            # lora_name="MATERIAL_real-0-5118.safetensors",
            strength_model=0.1,
            strength_clip=0.1,
            model=get_value_at_index(loraloader_239, 0),
            clip=get_value_at_index(loraloader_239, 1),
        )

        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()

        vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
        vaeloader_39 = vaeloader.load_vae(vae_name="ae.safetensors")

        loadimageoutput = NODE_CLASS_MAPPINGS["LoadImageOutput"]()
        imagestitch = NODE_CLASS_MAPPINGS["ImageStitch"]()
        fluxkontextimagescale = NODE_CLASS_MAPPINGS["FluxKontextImageScale"]()
        vaeencode = NODE_CLASS_MAPPINGS["VAEEncode"]()

        referencelatent = NODE_CLASS_MAPPINGS["ReferenceLatent"]()
        fluxguidance = NODE_CLASS_MAPPINGS["FluxGuidance"]()
        conditioningzeroout = NODE_CLASS_MAPPINGS["ConditioningZeroOut"]()
        ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        image_resize_rgthree = NODE_CLASS_MAPPINGS["Image Resize (rgthree)"]()
        easy_imagesave = NODE_CLASS_MAPPINGS["easy imageSave"]()



        for filename in filename_list_to_test:
            filename = filename.split('.')[0]  # filename = '000000375430'
            # if os.path.exists(f'/mnt/SSD/cyx/ComfyUI/output_all_lora_without_M_module_pred_using_ori_coco_prompt_into_4_lora/{filename}_all_lora_wo_M_module_00001_.png'):
            #     print('skip ' + filename)
            #     continue

            captions_data = load_json_file(os.path.join(whole_captions_dir, f'{filename}.json'))

            input_color_caption_pred_from_gray_data = load_json_file(input_color_caption_pred_from_gray_dir)
            input_style_caption_pred_from_gray_data = load_json_file(input_style_caption_pred_from_gray_dir)
            input_light_caption_pred_from_gray_data = load_json_file(input_light_caption_pred_from_gray_dir)
            input_material_caption_pred_from_gray_data = load_json_file(input_material_caption_pred_from_gray_dir)


            input_color_caption_pred_from_gray = input_color_caption_pred_from_gray_data[filename]
            input_style_caption_pred_from_gray = input_style_caption_pred_from_gray_data[filename]
            input_light_caption_pred_from_gray = input_light_caption_pred_from_gray_data[filename]
            input_material_caption_pred_from_gray = input_material_caption_pred_from_gray_data[filename]


            # print(input_color_caption_pred_from_gray)
            ori_COCO_caption = captions_data['short_prompts'][0]
            COLOR_caption = captions_data['whole_prompt']["COLOR"]
            # print('Processing: ', filename, '\n', COLOR_caption)
            STYLE_caption = captions_data['whole_prompt']["STYLE"]
            LIGHT_caption = captions_data['whole_prompt']["LIGHT"]
            MATERIAL_caption = captions_data['whole_prompt']["MATERIAL"]
            # caption_except_COLOR = 'change to a colorful image' #f"Style is {STYLE_caption}, Light is {LIGHT_caption}, Material is {MATERIAL_caption}"
            # input_color_caption_pred_from_gray_with_style_light_matetial_from_gt = 'a vivid colorful image,' + input_color_caption_pred_from_gray #+ ' STYLE: ' + STYLE_caption #+ ' LIGHTING: ' + LIGHT_caption #+ ' MATERIAL: ' + MATERIAL_caption
            # input_color_caption_pred_from_gray_with_style_light_matetial_from_gray = 'Colorize this to a vivid colorful image following the COLOR:' + input_color_caption_pred_from_gray + ori_COCO_caption + ' ' + input_style_caption_pred_from_gray + ' ' + input_light_caption_pred_from_gray + ' ' + input_material_caption_pred_from_gray
            input_color_caption_pred_from_gray_with_style_light_matetial_from_gray = 'a vivid colorful image,' + input_color_caption_pred_from_gray + ori_COCO_caption + ' STYLE: ' + input_style_caption_pred_from_gray + ' LIGHTING: ' + input_light_caption_pred_from_gray + ' MATERIAL: ' + input_material_caption_pred_from_gray
            # print(input_color_caption_pred_from_gray_with_style_light_matetial_from_gray)
            # print(input_style_caption_pred_from_gray)
            # print(input_light_caption_pred_from_gray)
            # print(input_material_caption_pred_from_gray)

            # 将图像文件路径传递给 load_image_output
            loadimageoutput_142 = loadimageoutput.load_image(
                image=os.path.join(gray_input_dir, filename + '.png')
            )

            imagestitch_146 = imagestitch.stitch(
                direction="right",
                match_image_size=True,
                spacing_width=0,
                spacing_color="white",
                image1=get_value_at_index(loadimageoutput_142, 0),
            )

            fluxkontextimagescale_42 = fluxkontextimagescale.scale(
                image=get_value_at_index(imagestitch_146, 0)
            )

            vaeencode_124 = vaeencode.encode(
                pixels=get_value_at_index(fluxkontextimagescale_42, 0),
                vae=get_value_at_index(vaeloader_39, 0),
            )

            # TODO: 记得修改文本输入
            cliptextencode_6 = cliptextencode.encode(
                text=input_color_caption_pred_from_gray_with_style_light_matetial_from_gray,
                # text=input_color_caption_pred_from_gray_with_style_light_matetial_from_gt,
                #text=' ',
                # text='change to a colorful image',
                # text=ori_COCO_caption,
                clip=get_value_at_index(loraloader_240, 1)
            )
            referencelatent_177 = referencelatent.append(
                conditioning=get_value_at_index(cliptextencode_6, 0),
                latent=get_value_at_index(vaeencode_124, 0),
            )
            fluxguidance_35 = fluxguidance.append(
                guidance=2.5, conditioning=get_value_at_index(referencelatent_177, 0)
            )
            conditioningzeroout_135 = conditioningzeroout.zero_out(
                conditioning=get_value_at_index(cliptextencode_6, 0)
            )



            ksampler_31 = ksampler.sample(
                seed=random.randint(1, 2**64),
                steps=20,
                cfg=1.0,
                sampler_name="euler",
                scheduler="simple",
                denoise=1.0,
                model=get_value_at_index(loraloader_240, 0),
                positive=get_value_at_index(fluxguidance_35, 0),
                negative=get_value_at_index(conditioningzeroout_135, 0),
                latent_image=get_value_at_index(vaeencode_124, 0),
            )

            vaedecode_8 = vaedecode.decode(
                samples=get_value_at_index(ksampler_31, 0),
                vae=get_value_at_index(vaeloader_39, 0),
            )

            image_resize_rgthree_205 = image_resize_rgthree.main(
                measurement="pixels",
                width=512,
                height=512,
                fit="contain",
                method="lanczos",
                image=get_value_at_index(vaedecode_8, 0),
            )

            # 最终保存结果
            easy_imagesave_234 = easy_imagesave.save(
                # filename_prefix=f"{filename}_all_real_text_no_style_light_material",
                # filename_prefix=f"{filename}_all_lora_wo_M_module",
                filename_prefix=f"{filename}_output_COCO_text_all_pred_different_prompt_start",
                only_preview=False,
                images=get_value_at_index(image_resize_rgthree_205, 0),
            )


if __name__ == "__main__":
    main()
