# Let Your Video Listen to Your Music! -- Beat-Aligned, Content-Preserving Video Editing with Arbitrary Music

<div align="center">
  <p><b>Accepted by ACM MM 2025</b></p>

  <p>
    <a href="https://zhangxinyu-xyz.github.io/">Xinyu Zhang</a><sup>1,3</sup>,
    <a href="https://donggong1.github.io/">Dong Gong</a><sup>2</sup>,
    <a href="https://zichengduan.github.io/">Zicheng Duan</a><sup>3</sup>,
    <a href="https://researchers.adelaide.edu.au/profile/anton.vandenhengel">Anton van den Hengel</a><sup>3</sup>,
    <a href="https://lingqiao-adelaide.github.io/lingqiaoliu.github.io/">Lingqiao Liu</a><sup>3</sup>
  </p>

  <p>
    <sup>1</sup>University of Auckland,
    <sup>2</sup>University of New South Wales,
    <sup>3</sup>The University of Adelaide
  </p>

  <p>
    <a href="https://zhangxinyu-xyz.github.io/MVAA/"><img src="https://img.shields.io/badge/Project-Page-green"></a>
    <a href="https://dl.acm.org/doi/abs/10.1145/3746027.3758140"><img src="https://img.shields.io/badge/ACM%20MM%2025-Paper-blue"></a>
    <a href="https://arxiv.org/abs/2506.18881"><img src="https://img.shields.io/badge/arXiv-2506.18881-b31b1b"></a>
  </p>
</div>

> MVAA (Music-Video Auto-Alignment) is a beat-aligned, content-preserving video editing framework. Given an arbitrary music track and an input video, MVAA aligns visual motion to musical rhythm while preserving the original semantic content.

---

## 📖 Table of Contents

- [🔥 Update Log](#update-log)
- [🛠️ Method Overview](#method-overview)
- [🚀 Getting Started](#getting-started)
- [🏃 Running Scripts](#running-scripts)
- [📊 Evaluation](#evaluation)
- [🤝 Citation](#citation)
- [🙏 Acknowledgement](#acknowledgement)

---

<a id="update-log"></a>
## 🔥 Update Log

- **[2026/03/28]** Try our "Let"-series work: [Let Your Image Move with Your Motion!](https://arxiv.org/abs/2603.01000), accepted by CVPR 2026.
- **[2026/03/28]** Codebase for MVAA released and cleaned for reproducible training/inference/evaluation.
- **[2025/07/31]** MVAA is accepted by ACM MM 2025.

---

<a id="method-overview"></a>
## 🛠️ Method Overview

Aligning video motion to music beats is a practical but underexplored problem in autonomous video editing.  
MVAA addresses this with a **two-step framework**:

1. **Beat-to-Motion Alignment**  
   Greedy Monotonic Matching: each music beat is matched to a unique motion peak.
2. **Auxiliary Video Completion Model**  
   A frame-conditioned diffusion model synthesizes coherent intermediate frames while preserving original content.

To balance quality and efficiency, MVAA adopts a **two-stage adaptation strategy**:

- **Pretraining** on a small set of videos to learn general motion priors.
- **Rapid test-time fine-tuning** for video-specific adaptation (can be done within several minutes on a single modern GPU).

<div align="center">
  <br><br>
  <img src="assets/main_method.gif" alt="MVAA main method demo" width="95%">
</div>

---

<a id="getting-started"></a>
## 🚀 Getting Started

### Clone this repo

```
git clone https://github.com/zhangxinyu-xyz/MVAA.git
```

The structure of the repo:
```text
MVAA/
├── finetune/        # training and test-time adaptation
├── inference/       # inference scripts and demos
├── pipelines/       # diffusion pipeline and custom components
├── preprocess/      # music/video preprocessing and alignment tools
├── postprocess/     # music/video postprocessing
├── utils/  
├── assets/          
└── README.md
```

### Environment Setup

```bash
conda create -n MVAA python=3.12
conda activate MVAA
pip install -r requirements.txt
```

### Data Preparation

You need:

- data/videos.txt  # all video list, the path is the related path
- data/musics.txt  # all music list, the path is the related path

Recommended preprocessing:

1. Extract music beat timestamps.
2. Analyze video motion statistics.
3. Build alignment targets for keyframe insertion.

The repository provides helper scripts under `preprocess/` for these steps.

### Checkpoints

Please download and prepare:

- The required base video diffusion backbone (e.g., CogVideoX I2V variant used in your experiments).
- MVAA-related checkpoints produced by your training or released weights.

Place checkpoints in your preferred local directory and update the paths in scripts accordingly.

---

<a id="running-scripts"></a>
## 🏃 Running Scripts

Below are common usage patterns. Please adapt paths to your local environment.

### 1) Preprocess

```bash
bash preprocess/run_preprocess.sh
```

This step prepares beat/motion alignment information needed for training or inference.

### 2) Training / Adaptation

Single-/multi-video training entrypoints are under `finetune/`, for example:

```bash
bash finetune/train_mvaa_single_video.sh  ## single video training
bash finetune/train_mvaa_multiple_videos.sh  ## multiple videos training
```

You can also use the test-time adaptation variant:

```bash
bash finetune/train_mvaa_single_video_testtime.sh  ## single video test-time training
bash finetune/train_mvaa_multiple_videos_testtime.sh  ## multiple videos test-time training
```

### 3) Inference and Evaluation

Run MVAA inference and generate beat-aligned videos:

```bash
bash inference/run_MVAA_eval.sh  ## single video inference
bash inference/run_MVAA_eval_multivideo.sh  ## single video inference and evaluation
```
---

<a id="citation"></a>
## 🤝 Citation

If you find this project useful, please cite:

```bibtex
@article{zhang2025let,
  title={Let Your Video Listen to Your Music!},
  author={Zhang, Xinyu and Gong, Dong and Duan, Zicheng and Hengel, Anton van den and Liu, Lingqiao},
  journal={arXiv preprint arXiv:2506.18881},
  year={2025}
}
```

---

<a id="acknowledgement"></a>
## 🙏 Acknowledgement

This project builds upon open-source video diffusion and multimedia processing ecosystems, especially [CogVideoX](https://github.com/zai-org/CogVideo).  
We sincerely thank all contributors from the related communities.
