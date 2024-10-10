### Generate Diffusor Data with Large Models ###
CUDA_VISIBLE_DEVICES=0 python data_gen.py --scene_name dinning_table --num_samples 100

### Training ###
CUDA_VISIBLE_DEVICES=0 python training.py --rotation False --scene_name dinning_table

### Evaluation ###
CUDA_VISIBLE_DEVICES=0 python eval_model.py --input_rgb ./assets/rgb.png --scene_name dinning_table --rotation False