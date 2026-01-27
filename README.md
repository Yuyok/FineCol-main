This is the official PyTorch code of the paper **[FineCol: A Fine-Grained Dataset for Image Colorization with Semantic Diffusion Framework]**.

[![homepage](https://img.shields.io/badge/homepage-GitHub-blue)](https://github.com/yourname/yourproject)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

## 📦 The Constructed Fine-Grained Dataset
[Download Link Here](https://drive.google.com/file/d/16Sa4sdmVPhUfY5SIHITCS2hn-QjsLoSA/view?usp=sharing)

## 🛠️ Setup

### 1. Environment Preparation
Create a conda environment and install the necessary dependencies:
```bash
conda env create -f environment.yaml
conda activate FineCol
```

### 2. External Repositories Preparation
Our project relies on several external frameworks. Please clone the following repositories and prepare them according to their official instructions, and prepare their pre-trained weights:
[T2ITrainer](https://github.com/lrzjason/T2ITrainer), [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

### 3. Code Injection
To enable our specific modules, you need to inject our custom codes into the external repositories downloaded above. 

**[WARNING] The `cp` command will overwrite target files. Please ensure you have backed up the original repositories if needed.**

Assuming your directory structure puts these repos alongside ours, run:

```bash
cp -r ./GSIM/data/* ../path_to_your_llama_repo/data
cp -r hack_codes/* ../path_to_your_comfyui_repo/
```

### 4. Checkpoints & Datasets
Please download the datasets and our pre-trained weights from the links below:

* **Constructed fine-grained textual dataset**: [Download Link Here](https://drive.google.com/file/d/16Sa4sdmVPhUfY5SIHITCS2hn-QjsLoSA/view?usp=sharing)
* **Pre-trained weights**: [Download Link Here](https://drive.google.com/...)
* **Other datasets:** Other datasets used in this project follow the COCO-LC. For more details, please refer to the [COCO-LC Project Page](https://github.com/lyf1212/COCO-LC).

Put all the attribute-adapters checkpoints to ``../path_to_your_comfyui_repo/models/loras``
Put the GSIM checkpoints to ``../path_to_your_llama_repo/saves/Qwen2.5-VL-7B-Instruct/lora``

## 🚀 Usage

### Step 1: Text Prediction
First, run the GSIM to generate fine-grained text prompts based on your input.

```bash
llamafactory-cli webui
```
Lease configure the following parameters:
``
GSIM-LoRA Path, Quantization: None, Batch Size: 16, Cutoff Length: 2048, Top-p=0 & Temperature=1, Save Path, Test Data: Select your test JSON, you can use the config files in ./GSIM/data/
``

Run the following script to generate the final JSON file containing fine-grained text prompts:
```bash
python extra_predict_text_and_filename_to_a_json.py
```


### Step 2: Image Generation
Use the predicted text from Step 1 to guide the image colorization process.

Before running the script, ensure you have configured the following parameters.
``Attribute-adapters weights, Input JSON, Dataset Path, Output Filename in [folder_path.py]``.

```bash
python test.py --SPLIT -1
```


### Training (TODO)


## 📧 Contact
If you have any questions, please feel free to submit an issue.