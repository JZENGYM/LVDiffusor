import argparse
import functools
import os
import pickle
import cv2
import numpy as np
import torch

from config import config
from utils.datasets import GraphDataset
from ipdb import set_trace
from PIL import Image
from score_matching.sampler import ode_sampler
from score_matching.sde import init_sde, marginal_prob_std_v2
from score_nets import ArrangeScoreModelGNN, ScoreModelGNN
from torch_geometric.loader import DataLoader
from groundingdino.util.inference import Model
from segment_anything import SamPredictor, sam_model_registry
import supervision as sv
from utils.misc import exists_or_mkdir
from utils.visualisations import images_to_gif
from utils.bbox_vis import MaskAnnotate, BoxAnnotate, rgb_vis, Rotated_rgb_vis


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()
parser.add_argument('--input_rgb', default='')
parser.add_argument('--scene_name', default='dinning_table')
def get_data_root(scene_name):
        return f'./datasets/{scene_name}/test_data'
parser.add_argument('--data_root', default=get_data_root(parser.parse_known_args().scene_name),
                    help='The root directory of the test data.')
parser.add_argument('--rotation', default=False)

parser.add_argument('--DINO_CONFIG_PATH', default="./backbones/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
parser.add_argument('--DINO_CHECKPOINT_PATH', default="./backbones/Grounded-Segment-Anything/groundingdino_swint_ogc.pth")
parser.add_argument('--CLASSES', nargs='+', default=['plate', 'fork', 'knife', 'cup', 'spoon', 'glass', 'mug', 'bowl'])
parser.add_argument('--BOX_THRESHOLD', type=float, default=0.36)
parser.add_argument('--SAM_ENCODER_VERSION', default="vit_h")
parser.add_argument('--SAM_CHECKPOINT_PATH', default="./backbones/Grounded-Segment-Anything/sam_vit_h_4b8939.pth")
args = parser.parse_args()

# objects detection and segmentation
class ObjDetector:
    def __init__(self, DINO_CONFIG_PATH, DINO_CHECKPOINT_PATH, SAM_ENCODER_VERSION, SAM_CHECKPOINT_PATH):
        self.grounding_dino_model = Model(model_config_path=DINO_CONFIG_PATH, model_checkpoint_path=DINO_CHECKPOINT_PATH)
        sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
        sam.to(device=device)
        self.sam_predictor = SamPredictor(sam)

    def segment(self, sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        # set_trace()
        return np.array(result_masks)

    def obj_detect(self, scene_name, data_root, input_rgb, CLASSES):
        # {self.CLASSES}_data = []
        if input_rgb:
            image = cv2.imread(input_rgb)
            detections, boxes = self.grounding_dino_model.predict_with_classes(
                image=image,
                classes=CLASSES,
                box_threshold=args.BOX_THRESHOLD,
                text_threshold=args.BOX_THRESHOLD
            )

            box_annotator = sv.BoxAnnotator()
            labels = [
                # f"{CLASSES[class_id]} {confidence:0.2f}"
                f"{CLASSES[class_id]}" 
                for _, _, confidence, class_id, _ 
                in detections]

            annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

            bblabels = [
                class_id
                for _, _, _, class_id, _ 
                in detections]
            boxes = boxes.numpy()
            

            # convert detections to masks
            detections.mask = self.segment(
                sam_predictor=self.sam_predictor,
                image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                xyxy=detections.xyxy
            )
            masks = detections.mask
            mask_annotator = MaskAnnotate()
            # return scene, masks_sort, mask_label, sorted_index
            annotated_image, color_masks, masks_label, sorted_index = mask_annotator.annotate(scene=image.copy(), detections=detections)

            # mask rgb -> grey
            masks_sorted = []
            for img in color_masks:
                mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                masks_sorted.append(mask)

            bbox_data = []
            for i in range(len(bblabels)):
                bbox_data.append(np.append(boxes[i], bblabels[i]))
            bbox_data = list(bbox_data)

            # os.makedirs(f'./datasets/{scene_name}/data/', exist_ok=True)
            with open(os.path.join(data_root, f'{scene_name}.pickle'), 'wb') as f:
                pickle.dump(bbox_data, f)
    

        return annotated_frame, bbox_data, masks, masks_sorted, masks_label, sorted_index



# load test data
def get_dataloaders(rotation, scene_name, data_root):
    dataset= GraphDataset(scene_name, data_root, rotation=rotation, base_noise_scale=config.base_noise_scale)
    dataloader_vis = DataLoader(dataset, batch_size=2**2, shuffle=True, num_workers=4)
    return dataloader_vis

# load score model
def get_score_network(configs, marginal_prob_std_fn, state_dim):
    if configs.env_type == 'NoTableGeo':     
        score_net = ArrangeScoreModelGNN(
            marginal_prob_std_fn,
            hidden_dim=configs.hidden_dim_gf,
            embed_dim=configs.embed_dim_gf,
            state_dim=state_dim,
        )
    elif configs.env_type == 'TableGeoCon':
        score_net = ScoreModelGNN(
            marginal_prob_std_fn, 
            num_classes=configs.num_classes, 
            hidden_dim=configs.hidden_dim_gf,
            embed_dim=configs.embed_dim_gf,
            device=device,
        )
    else:
        raise ValueError(f"Mode {configs.env_type} not recognized.")
    return score_net


def eval_model(configs, rgb_input, masks, bbox_data, scene_name, data_root, rotation):

    # get dataloaders
    dataloader_vis = get_dataloaders(rotation, scene_name, data_root)

    # load rgb
    rgb = cv2.imread(rgb_input)
    # # load category list
    with open(f"datasets/{scene_name}/category/category_dict.pkl", 'rb') as f:
        category_dict = pickle.load(f)


    # init SDE-related params
    sigma = configs.sigma
    marginal_prob_std_fn = functools.partial(marginal_prob_std_v2, sigma=sigma)

    # create models
    state_dim = 4 if rotation==True else 2
    score = get_score_network(configs, marginal_prob_std_fn, state_dim)

    #load pretrained model
    # with open('./logs/test/score.pt', 'rb') as f:
    #     score = pickle.load(f)
    ckpt_dir = f'./logs/ckpt/{scene_name}.ckpt'
    score.load_state_dict(torch.load(ckpt_dir))
    score.to(device)
    prior_fn, marginal_prob_fn_v2, sde_fn, sampling_eps = init_sde(configs.sde_mode)

    ref_batch = next(iter(dataloader_vis))
    res_batch, res_video = ode_sampler(
        score,
        ref_batch,
        sde_fn,
        prior_fn,
        eps=sampling_eps,
        t0=1.0,
        batch_size=2**2,
        use_rotation=rotation,
    ) 
    print("inference done!")
    # save target layout
    pos = res_batch.x.cpu().numpy()
    pos[:, :2] = (pos[:, :2] + 1.0) / 2.0

    # imgs of all samping step
    print("rendering ...")
    imgs = []
    for i in range(len(res_video)):
        if rotation==True:
            img = Rotated_rgb_vis(res_video[i], category_dict, bbox_data, rgb, masks)
        else:
            img, _ = rgb_vis(res_video[i], category_dict, bbox_data, rgb, masks)    
        imgs.append(img)
    
    # print(f'steps = {len(imgs)}')
    print("saving videos ...")
    video_save_path = f'./logs/{args.scene_name}/eval_video'
    exists_or_mkdir(video_save_path)
    fps = len(imgs)//5
    test = [Image.fromarray(img[:, :, ::-1].astype(np.uint8), mode='RGB') for img in imgs]
    
    images_to_gif(os.path.join(video_save_path, f'{scene_name}.gif'), test, fps=200)

    cv2.imwrite(os.path.join(video_save_path, f'eval_initial.jpg'), imgs[0])
    cv2.imwrite(os.path.join(video_save_path, f'eval_ending.jpg'), imgs[-1])

if __name__ == "__main__":
    
    obj_detector = ObjDetector(args.DINO_CONFIG_PATH, args.DINO_CHECKPOINT_PATH, args.SAM_ENCODER_VERSION, args.SAM_CHECKPOINT_PATH)
    annotated_frame, bbox_data, masks, masks_sorted, masks_label, sorted_index = obj_detector.obj_detect(args.scene_name, args.data_root, args.input_rgb, args.CLASSES)
    eval_model(config, args.input_rgb, masks, bbox_data, args.scene_name, args.data_root, args.rotation)
