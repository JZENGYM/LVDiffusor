import ast
import os
import pickle
import argparse
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv
import torch
from groundingdino.util.inference import Model
from ipdb import set_trace
import openai
from openai import OpenAI
from diffusers import DiffusionPipeline



DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### config ###
class Config:
    def __init__(self, args):
        ### SD Config ###
        self.sd_model = "stabilityai/stable-diffusion-xl-base-1.0"
        self.vlm_prompt  = "A well-organized table with one plate, one fork, and one bowl"
        self.save_dir = f"./datasets/{args.scene_name}/vlm_imgs"
        self.num_images = args.num_samples
        
        ### DINO Config ###
        self.DINO_CONFIG_PATH = "./backbones/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        self.DINO_CHECKPOINT_PATH = "./backbones/Grounded-Segment-Anything/groundingdino_swint_ogc.pth"
        self.CLASSES = ['plate', 'fork', 'knife', 'cup', 'spoon', 'glass', 'mug', 'bowl']
        # self.CLASSES = ['laptop', 'mouse', 'keyboard', 'notebook', 'mug', 'green plant']
        self.BOX_THRESHOLD = 0.36
        self.vlm_data_dir = f"./datasets/{args.scene_name}/vlm_data"

        ### GPT Config ###
        # self.opai_key = open("openai_key.txt","r").read().strip()
        self.opai_key = ''
        self.llm_data_dir = f"./datasets/{args.scene_name}/refined_data"
        self.llm_prompt = [f"Utilize your expertis in {args.scene_name} arrangement and spatial reasoning to create an organized layout suitable for {args.function} individuals.\
                            please instruct me on how to rearrange these items {self.CLASSES}. Your response should be in list format, with each line not exceeding twenty words, and no additional information.", 
                            
                            f"The user will give you several bounding boxes of objects on a square image, the bounding box format is [xmin, ymin, xmax, ymax, label], and xmin, ymin, xmax, ymax are normalized to [0, 1]\
                            There are some objects you should remove. this {args.scene_name} setting is only for one {args.function} person. \
                            Use the following step-by-step instructions to respond. \
                            Step 1 - Depending on the number of dining people you infer, list how many of each item is needed \
                            Step 2 - Depending on the number of each item you need, remove the bounding box that contains unnecessary items till the number of each item is consistent with the number you raised in Step 1. \
                            Step 3 - After removing, you should move the remaining coordinate of the bounding box to achieve spatial reasoning and a well-organized {args.scene_name} setting for {args.function} people, and the size of the bounding box you removed should not change. \
                            your answer should be brief. please follow below desired format below stricly without other words: \
                            Number to remove: \
                            Number to move: \
                            Rearrange bounding box: \
                            Num of each object: \
                            Rearrange bounding box: \
                            From: \
                            To: \
                            Modified list: \
                            Here is an example for in-context learning: \
                            User input: [[0.87265625, 0.22708333, 0.9140625, 0.8020833, 'knife'], \
                                        [0.921875, 0.33125, 0.984375, 0.8, 'spoon'], \
                                        [0.046875, 0.27083333, 0.11875, 0.8125, 'fork'], \
                                        [0.1078125, 0.10416667, 0.8828125, 0.83125, 'plate'], \
                                        [0.8203125, 0.0, 0.971875, 0.17083333, 'glass'], \
                                        [0.2578125, 0.25208333, 0.8359375, 0.7125, 'bowl']] \
                            Output: \
                            Modified List: [[0.0578125, 0.22708333, 0.103125, 0.8020833, 'knife'], \
                                            [0.00390625, 0.33125, 0.084375, 0.8, 'spoon'], \
                                            [0.884375, 0.27083333, 0.953125, 0.8125, 'fork'], \
                                            [0.1078125, 0.10416667, 0.8828125, 0.83125, 'plate'], \
                                            [0.5859375, 0.0, 0.7359375, 0.17083333, 'glass'], \
                                            [0.2578125, 0.25208333, 0.8359375, 0.7125, 'bowl']]"]
        
'''
Number to move: 5 (Plate, Knife, Spoon, Fork, and Glass) \
Rearrange Bounding Box: \
Plate: From: [69, 50, 564, 399] To: [69, 50, 564, 399] (No change in size and position) \
Knife: From: [557, 109, 586, 385] To:  [37, 109, 66, 385] (Moved to the left side of the plate) \
Spoon: From: [590, 158, 639, 382] To: [5, 158, 54, 382] (Moved to the far left, to the left of the knife, not overlap a lot with a knife) \
Fork: From: [30, 129, 75, 388] To: [564, 129, 610, 388] (Moved to the right side of the plate) \
Glass: From:  [523, 0, 621, 82] To: [375, 0, 472, 82] (Moved above the plate, at a centralized position) \
Bowl: Unchanged: [165, 121, 526, 344] \
'''
parser = argparse.ArgumentParser()
parser.add_argument('--num_samples', type=int, default=100)
parser.add_argument('--scene_name', default='dinning_table_vanilla')
parser.add_argument('--function', default='right-handed')
# load args
args = parser.parse_args()

config = Config(args)
os.makedirs(config.save_dir, exist_ok=True)
os.makedirs(config.vlm_data_dir, exist_ok=True)
os.makedirs(config.llm_data_dir, exist_ok=True)
label_mapping = {label: index for index, label in enumerate(config.CLASSES)}

### 1. img genereation with SDXL ###
class VLMGenImg:
    def __init__(self, model):
        # self.pipe = DiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        self.pipe = DiffusionPipeline.from_pretrained(model, use_safetensors=True, variant="fp16")
        self.pipe.to(DEVICE)

    def generate_images(self, prompt, num_images):
        image_paths = []
        for i in range(num_images):
            # with torch.autocast('cuda'):
            image = self.pipe(prompt).images[0]
            image_path = os.path.join(config.save_dir, f"GenImg_{i}.png")
            image.save(image_path)
            image_paths.append(image_path)
        return image_paths


### 2. image detect ###
class ObjDetector:
    def __init__(self, DINO_CONFIG_PATH, DINO_CHECKPOINT_PATH):
        self.grounding_dino_model = Model(model_config_path=DINO_CONFIG_PATH, model_checkpoint_path=DINO_CHECKPOINT_PATH)

    def obj_detect(self, image_paths, CLASSES, output_dir):
        # {self.CLASSES}_data = []
        for image_path in image_paths:
            image_dir = os.path.join(config.save_dir, image_path)
            image = cv2.imread(image_dir)
            detections, boxes = self.grounding_dino_model.predict_with_classes(
                image=image,
                classes=CLASSES,
                box_threshold=config.BOX_THRESHOLD,
                text_threshold=config.BOX_THRESHOLD
            )

            box_annotator = sv.BoxAnnotator()
            labels = [
                # f"{CLASSES[class_id]} {confidence:0.2f}"
                f"{CLASSES[class_id]}" 
                for _, _, confidence, class_id, _ 
                in detections]
            annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)
            # save the annotated grounding dino image
            cv2.imwrite(os.path.join(output_dir, f"groundingdino_annotated_{image_path}.png"), annotated_frame)

            bblabels = [
                class_id
                for _, _, _, class_id, _ 
                in detections]
            boxes = boxes.numpy()
            bbox_data = []
            for i in range(len(bblabels)):
                bbox_data.append(np.append(boxes[i], labels[i]))
            bbox_data = list(bbox_data)

            with open(os.path.join(output_dir, f"{image_path}.pickle"), 'wb') as f:
                pickle.dump(bbox_data, f)


### 3. LLM refine ### 
class GPTRefine:
    def __init__(self, openai_key):
        self.client = OpenAI(api_key=openai_key, base_url=f"https://frostsnowjh.com/v1")

    def layout_refine(self, prompts, vlm_data_dir, llm_data_dir):
        messages = []
        for prompt in prompts:
            # set_trace()
            messages.append({"role": "user", "content": [{'type': 'text', 'text': prompt}]})
            response = self.client.chat.completions.create(
                model="gpt-4o", 
                messages=messages,
            )
            answer = response.choices[0].message.content
            print(answer)
            messages.append({"role": "system", "content": answer})

        for file in os.listdir(vlm_data_dir):
            if file.endswith(".pickle"):
                with open(os.path.join(vlm_data_dir, file), 'rb') as f:
                    bbox_data_vlm = pickle.load(f)
                    ndarry2list = [array.tolist() for array in bbox_data_vlm]
                    json_data = json.dumps(ndarry2list)
                    messages.append({"role": "user", "content": [{'type': 'text', 'text': json_data}]})
                    response = self.client.chat.completions.create(
                        model="gpt-4o", 
                        messages=messages
                    )
                    answer = response.choices[0].message.content
                    
                    # with open(os.path.join(llm_data_dir, f"{file}.txt"), 'w') as f:
                    #     f.write(answer)
                    
                    # save refined bbox data
                    answer = answer.lower()
                    start_index = answer.index("modified list:") + len("modified list:")
                    cleaned_text = answer[start_index:].strip()
                    # list_of_lists = json.loads(cleaned_text)
                    list_of_lists = ast.literal_eval(cleaned_text)
                    res = []
                    for item in list_of_lists:
                        item[-1] = label_mapping[item[-1]]
                        item = np.array(item)
                        res.append(item)

                    with open(os.path.join(llm_data_dir, f"{file}"), 'wb') as f:
                        pickle.dump(res, f)
                    # delete the last user message
                    messages.pop()

        # response = self.client.ChatCompletion.create(
        #     model="gpt-4", 
        #     messages=messages
        # )
    # return response.choices[0].message['content']



if __name__ == "__main__":
    ### Generate imgs with SDXL ###
    index_to_class = {index: class_name for index, class_name in enumerate(config.CLASSES)}
    category_dir = f"datasets/{args.scene_name}/category"
    os.makedirs(category_dir, exist_ok=True)
    with open(os.path.join(category_dir, 'category_dict.pkl'), 'wb') as f:
        pickle.dump(index_to_class, f)
    vlm_gen = VLMGenImg(config.sd_model)
    image_paths = vlm_gen.generate_images(config.vlm_prompt, config.num_images)

    ### Detection ###
    image_paths = [f for f in os.listdir(config.save_dir) if f.endswith('.png')]
    obj_detector = ObjDetector(config.DINO_CONFIG_PATH, config.DINO_CHECKPOINT_PATH)
    obj_detector.obj_detect(image_paths, config.CLASSES, config.vlm_data_dir)

    ### LLM refine ###
    gpt_refine = GPTRefine(config.opai_key)
    gpt_refine.layout_refine(config.llm_prompt, config.vlm_data_dir, config.llm_data_dir)
    print("Done!")
