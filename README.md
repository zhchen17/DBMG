<h1 align="center"> Dual-Branch Multi-Granularity Network with Structured Contrastive Ranking for Cross-Modal Retrieval </h1>






### Step 1: Set up the environment

We recommend using Conda to manage virtual environments. This project requires `torch_npu` for NPU support and Python 3.8+.

```bash
conda create -n dbmg_env python=3.8
conda activate dbmg_env
pip install -r requirements.txt

# Install torch_npu (make sure Ascend toolkit is installed properly)
pip install torch_npu -f https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/pytorch/{version}/torch_npu.html
```

### Step 2: Download the Pre-trained CLIP Model and Datasets.
```bash
You can download the pre-trained CLIP model used in this paper from the following link: https://huggingface.co/openai/clip-vit-base-patch32
Wikipedia: http://www.svcl.ucsd.edu/projects/crossmodal/
NUS-WIDE: http://lms.comp.nus.edu.sg/research/NUS-WIDE.htm
XMediaNet: http://www.icst.pku.edu.cn/mipl/XMedia/
Pascal-Sentence: https://vision.cs.uiuc.edu/pascal-sentences/
```

### Step 3: Start the training

Download the required datasets from the following links and place them in the ./data/ directory:
```bash
python train.py
