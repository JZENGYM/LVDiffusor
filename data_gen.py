import os
import pickle
import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv
import torch
from groundingdino.util.inference import Model
# from ipdb import set_trace
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
        self.opai_key = ''
        self.llm_data_dir = f"./datasets/{args.scene_name}/refined_data"
        self.llm_prompt = [f"Utilize your expertis in {args.scene_name} arrangement and spatial reasoning to create an organized layout suitable for {args.function} individuals. \
                            please instruct me on how to rearrange these items {self.CLASSES}. Your response should be in list format, with each line not exceeding twenty words, and no additional information.", 
                            
                            f"The user will give you several bounding boxes of objects on a width*height = 640*480 image, the bounding box format is [xmin, ymin, xmax, ymax, label]. \
                            There are some objects you should remove. this {args.scene_name} setting is only for one {args.function} person. \
                            Use the following step-by-step instructions to respond. \
                            Step 1 - Depending on the number of dining people you infer, list how many of each item is needed \
                            Step 2 - Depending on the number of each item you need, remove the bounding box that contains unnecessary items till the number of each item is consistent with the number you raised in Step 1. \
                            Step 3 - After removing, you should move the remaining coordinate of the bounding box to achieve spatial reasoning and a well-organized {args.scene_name} setting for {args.function} people, and the size of the bounding box you removed should not change. \
                            your answer should be brief. please follow below desired format below: \
                            Number to remove: \
                            Number to move: \
                            Rearrange bounding box: \
                            Num of each object: \
                            Rearrange bounding box: \
                            From: \
                            To: \
                            Modified list: \
                            Here is an example for in-context learning: \
                            User input: [[557, 109, 586, 385, 'knife'], \
                            [590, 158, 639, 382, 'spoon'], \
                            [30, 129, 75, 388, 'fork'], \
                            [69, 50, 564, 399, 'plate'], \
                            [523, 0, 621, 82, 'glass'], \
                            [165, 121, 526, 344, 'bowl']] \
                            Output: \
                            Number to move: 5 (Plate, Knife, Spoon, Fork, and Glass) \
                            Rearrange Bounding Box: \
                            Plate: From: [69, 50, 564, 399] To: [69, 50, 564, 399] (No change in size and position) \
                            Knife: From: [557, 109, 586, 385] To:  [37, 109, 66, 385] (Moved to the left side of the plate) \
                            Spoon: From: [590, 158, 639, 382] To: [5, 158, 54, 382] (Moved to the far left, to the left of the knife, not overlap a lot with a knife) \
                            Fork: From: [30, 129, 75, 388] To: [564, 129, 610, 388] (Moved to the right side of the plate) \
                            Glass: From:  [523, 0, 621, 82] To: [375, 0, 472, 82] (Moved above the plate, at a centralized position) \
                            Bowl: Unchanged: [165, 121, 526, 344] \
                            Modified List: [[37, 109, 66, 385, 'knife'], [5, 158, 54, 382, 'spoon'], [564, 129, 610, 388, 'fork'], [69, 50, 564, 399, 'plate'], [375, 0, 472, 82, 'glass'], [165, 121, 526, 344, 'bowl']]", 

                            f"To assist with your {args.scene_name} rearrangement task for a {args.function} person, \
                            I will need the bounding box coordinates and labels of the items on the table. \
                            Once you provide these details, I can guide you through the steps to organize the items according to the requirements you’ve outlined. \
                            Please share the bounding box information for the items on your {args.scene_name}."] # 3 prompts for step-by-step instructions and in-context learning
        
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
                bbox_data.append(np.append(boxes[i], CLASSES[bblabels[i]]))
            bbox_data = list(bbox_data)

            with open(os.path.join(output_dir, f"{image_path}.pickle"), 'wb') as f:
                pickle.dump(bbox_data, f)


### 3. LLM refine ### 
class GPTRefine:
    def __init__(self, openai_key):
        # openai.api_key = openai_key
        self.client = OpenAI(api_key=openai_key)

    def layout_refine(self, prompts, vlm_data_dir, llm_data_dir):
        messages = []
        for prompt in prompts:
            messages.append({"role": "user", "content": prompt})
            response = self.client.completions.create(
                model="gpt-4", 
                prompt=prompt,
            )
            answer = response.choices[0].text
            messages.append({"role": "assistant", "content": answer})

        for file in os.listdir(vlm_data_dir):
            if file.endswith(".pickle"):
                with open(os.path.join(vlm_data_dir, file), 'rb') as f:
                    bbox_data_vlm = pickle.load(f)
                    messages.append({"role": "user", "content": bbox_data_vlm})
                    response = self.client.completions.create(
                        model="gpt-4", 
                        messages=messages
                    )
                    answer = response.choices[0].text
                    with open(os.path.join(llm_data_dir, f"{file}.pickle"), 'wb') as f:
                        pickle.dump(answer, f)
                    # delete the last user message
                    messages.pop()

        # response = self.client.ChatCompletion.create(
        #     model="gpt-4", 
        #     messages=messages
        # )
    # return response.choices[0].message['content']



if __name__ == "__main__":
    index_to_class = {index: class_name for index, class_name in enumerate(config.CLASSES)}
    category_dir = f"datasets/{args.scene_name}/category"
    os.makedirs(category_dir, exist_ok=True)
    with open(os.path.join(category_dir, 'category_dict.pkl'), 'wb') as f:
        pickle.dump(index_to_class, f)
    vlm_gen = VLMGenImg(config.sd_model)
    image_paths = vlm_gen.generate_images(config.vlm_prompt, config.num_images)

    image_paths = [f for f in os.listdir(config.save_dir) if f.endswith('.png')]
    obj_detector = ObjDetector(config.DINO_CONFIG_PATH, config.DINO_CHECKPOINT_PATH)
    obj_detector.obj_detect(image_paths, config.CLASSES, config.vlm_data_dir)

    gpt_refine = GPTRefine(config.opai_key)
    gpt_refine.layout_refine(config.llm_prompt, config.vlm_data_dir, config.llm_data_dir)
    print("Done!")
