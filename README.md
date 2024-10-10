# LVDiffusor: Distilling Functional Rearrangement Priors from Large Models into Diffusor

![LVDiffusor Logo](assets/pipeline.png)

LVDiffusor distills **functional arrangement knowledge** from **large models** into a **diffusion model** to generate **well-organized** and **compatible layouts** from everyday cluttered scenes.

[![Website](https://img.shields.io/badge/Website-orange.svg )](https://sites.google.com/view/lvdiffusion)
[![Arxiv](https://img.shields.io/badge/Arxiv-green.svg )](https://arxiv.org/abs/2312.01474)

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)
- [Contact](#contact)


## Installation

To install the LVDiffusor, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/LVDiffusor.git
    ```
2. Navigate to the project directory:
    ```bash
    cd LVDiffusor
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Detection Submodule Installation ###

The detection module in our work depends on [GroundingSAM](https://github.com/IDEA-Research/Grounded-Segment-Anything). Please check the original repo, download checkpoints for DINO and SAM, and complete the setup for [local deployment](https://github.com/IDEA-Research/Grounded-Segment-Anything/issues/75).

You should set the environment variable manually as follows if you want to build a local GPU environment for Grounded-SAM:
```bash
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export CUDA_HOME=/path/to/cuda-11.3/
```

Install Segment Anything:

```bash
python -m pip install -e segment_anything
```

Install Grounding DINO:

```bash
pip install --no-build-isolation -e GroundingDINO
```


Install diffusers:

```bash
pip install --upgrade diffusers[torch]
```

Install osx:

```bash
git submodule update --init --recursive
cd grounded-sam-osx && bash install.sh
```

Install RAM & Tag2Text:

```bash
git clone https://github.com/xinyu1205/recognize-anything.git
pip install -r ./recognize-anything/requirements.txt
pip install -e ./recognize-anything/
```

Download the pretrained weights
```bash
cd Grounded-Segment-Anything

wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

## Usage

### Dataset Generation with VLM and LLM ###
We use GPT4 as the large language model, please prepare your keys for requesting the openai api. The config in [data_gen.py](data_gen.py) can be edited for prompting the VLM and LLM.

Generate the layout dataset:
```bash
python data_gen.py --num_samples 100
```

### Training ###
```bash
python training.py --rotation False --scene_name dinning_table
```

### Evaluation ###
To use LVDiffusor, download the checkpoints and run the following command:
```bash
python eval_model.py --input_rgb ./assets/rgb.png --scene_name dinning_table --rotation False
```


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Citation

Please cite our paper using the following BibTeX entry if you find it helpful:

```bibtex
@ARTICLE{lvdiffusor2024,
    author={Zeng, Yiming and Wu, Mingdong and Yang, Long and Zhang, Jiyao and Ding, Hao and Cheng, Hui and Dong, Hao},
    journal={IEEE Robotics and Automation Letters}, 
    title={LVDiffusor: Distilling Functional Rearrangement Priors From Large Models Into Diffusor}, 
    year={2024},
    volume={9},
    number={10},
    pages={8258-8265}
}
```

## Contact

For any questions or inquiries, please contact:

Yiming Zeng: [zengym27@mail2.sysu.edu.cn](mailto:zengym27@mail2.sysu.edu.cn)

Mingdong Wu: [wmingd@pku.edu.cn](mailto:wmingd@pku.edu.cn)
