# Sketch2Image — Bringing Sketches to Life with Generative AI

Turns hand-drawn pen sketches into realistic or stylised images with **Stable Diffusion 1.5 + ControlNet**, and measures how well each result keeps the sketch's structure.

![Input sketches (left) and generated images in four styles](outputs/figures/grid_styles.png)

---

## 1. How it works

```mermaid
flowchart LR
    A[Photo of pen sketch] --> B[Preprocessing<br/>adaptive threshold,<br/>denoise, resize ×8]
    B --> C[Control image<br/>white lines on black]
    C --> D[ControlNet<br/>trainable copy of UNet encoder]
    P[Text prompt<br/>subject + style] --> E[CLIP text encoder]
    E --> F
    E --> D
    D -- residuals × conditioning scale --> F[SD 1.5 UNet<br/>frozen]
    N[Random latent noise<br/>seed] --> F
    F -- 25 UniPC denoising steps --> G[VAE decoder]
    G --> H[Generated image 512 px]
```

**Stable Diffusion (latent diffusion).** A VAE compresses a 512×512 image into a 64×64×4 latent. A UNet is trained to predict the noise in a noisy latent, given the text embedding from CLIP (cross-attention). Generation starts from pure noise and removes it step by step (we use the UniPC solver, 25 steps). The VAE decoder then turns the final latent into pixels.

**How the sketch steers the image: ControlNet.** ControlNet is a trainable copy of the UNet encoder that takes the control image (our sketch) as an extra input. Its feature maps go through *zero-initialised* 1×1 convolutions and are **added to the skip connections and middle block of the frozen SD UNet at every denoising step**. The original model weights are never changed, and the sketch only adds a spatial "where things go" signal. The **conditioning scale** multiplies these residuals:
- 0 → plain text-to-image (the sketch is ignored)
- 1 → the strength used in training
- more than 1 → the sketch is enforced harder

**Classifier-free guidance (CFG).** At each step the UNet is run with and without the prompt (the "without" branch uses the negative prompt). The difference is amplified by the *guidance scale*. A higher value follows the prompt more closely but gives over-saturated images.

**Baseline: img2img (SDEdit).** Instead of ControlNet, the sketch itself is noised to a level set by `strength` and then denoised with the prompt. The structure is kept only through the starting image, so there is a direct trade-off: low strength still looks like a pencil drawing, high strength loses the layout. Comparing this with ControlNet shows why a separate conditioning network is needed.

**Three ControlNet variants compared:**

| Control | Model | Control image | Trained on |
|---|---|---|---|
| `scribble` | `control_v11p_sd15_scribble` | sketch lines, thickened (dilated ×2) | synthesised rough scribbles |
| `lineart` | `control_v11p_sd15_lineart` | sketch lines, thin | line drawings extracted from images |
| `canny` | `control_v11p_sd15_canny` | Canny edges of the sketch photo | Canny edge maps |

---

## 2. Repository structure

```
sketch2image/
├── sketch2img.py          preprocessing + Generator class + single-image CLI
├── metrics.py             sketch-preservation metric + CLIP score
├── run_experiments.py     runs all experiments in config.yaml → outputs/results.csv
├── make_figures.py        comparison grids, curves, LaTeX tables
├── app.py                 Gradio web UI
├── config.yaml            models, sketches, prompts, default settings, experiments
├── Sketch2Image.ipynb     Colab/Kaggle notebook that runs everything
├── requirements.txt
├── sketches/              input sketches (my own drawings)
└── outputs/               control images, generated images, figures, tables
```

---

## 3. Installation

Tested on Google Colab with a free **T4 GPU** (16 GB). Kaggle works too: choose GPU **T4** and turn on *Internet* in the notebook settings.

**Easiest: open `Sketch2Image.ipynb` in Colab** (*Runtime → Change runtime type → T4 GPU*) and run the cells from top to bottom.

**Manual / local** (NVIDIA GPU with ≥ 8 GB VRAM, Python ≥ 3.10):
```bash
git clone https://github.com/<your-username>/sketch2image.git
cd sketch2image
pip install torch --index-url https://download.pytorch.org/whl/cu121   # if torch is not installed
pip install -r requirements.txt
```
On the first run, the models (~9 GB) are downloaded automatically from the Hugging Face Hub.

---

## 4. Usage

```bash
# one image
python sketch2img.py --sketch sketches/sketch1.png \
    --prompt "a futuristic skyscraper, photorealistic" --control scribble --cond-scale 1.0 --out out.png

# all experiments (~70 images, ~10 min on a T4), then the figures and tables
python run_experiments.py
python make_figures.py

# only some experiments
python run_experiments.py --only styles cond_scale

# web UI
python app.py --share
```
Generated images are cached in `outputs/images/`. If Colab disconnects, re-run and it continues where it stopped.

---

## 5. Reproducibility: main settings

| Setting | Value |
|---|---|
| Base model | `stable-diffusion-v1-5/stable-diffusion-v1-5` (fp16) |
| ControlNets | `lllyasviel/control_v11p_sd15_{scribble, lineart, canny}` |
| Scheduler | UniPC multistep, 25 steps |
| Resolution | longest side 512 px, both sides multiples of 8 |
| Guidance scale | 7.5 |
| ControlNet conditioning scale | 1.0 |
| img2img strength | 0.8 |
| Seed | 42 (PyTorch generator on the GPU) |
| Negative prompt | `lowres, blurry, bad anatomy, deformed, …` (see `sketch2img.py`) |
| Prompt | `"<subject>, <style text>"`, both defined in `config.yaml` |
| Safety checker | disabled (all content is self-made sketches) |

All experiment settings are in `config.yaml`. Each experiment changes **one** setting and keeps the others at the defaults above.

| Experiment | Setting varied | Values |
|---|---|---|
| `styles` | style text in the prompt | photo, concept art, watercolor, clay |
| `cond_scale` | ControlNet conditioning scale | 0, 0.3, 0.6, 0.9, 1.2, 1.5 |
| `guidance` | CFG scale | 2, 4, 7.5, 11, 15 |
| `control` | ControlNet type | scribble, lineart, canny |
| `img2img` | img2img strength (baseline, no ControlNet) | 0.5, 0.7, 0.85, 1.0 |
| `seeds` | random seed | 1, 2, 3, 4 |

Note: exact pixel values can differ slightly between GPU types and library versions, even with the same seed.

---

## 6. Evaluation metrics

- **Sketch recall @5 px (higher is better).** The sketch lines are skeletonised to 1-px centre lines. Edges are detected in the generated image with Canny (thresholds set automatically from the median intensity). The metric is the fraction of sketch pixels that have a generated edge within 5 px (~1 % of the image size).
- **Chamfer distance (px, lower is better).** The mean distance from the sketch centre lines to the nearest generated edge, computed with a distance transform.
- Both metrics are one-directional on purpose. A good result *adds* texture and details, so extra edges in the output are not penalised.
- **CLIP score (higher is better).** 100 × cosine similarity between the CLIP ViT-B/16 image and prompt embeddings ([Hessel et al., 2021](https://arxiv.org/abs/2104.08718)). It measures how well the image matches the text.
- Sanity checks on a test sketch:
  - the sketch scored against itself gives recall = 1.0;
  - the sketch shifted by 20 px drops to 0.26–0.60;
  - an image with no edges gives 0.

---

## 7. Results

*(Add the figures from `outputs/figures/` and a short discussion here once the experiments are run.)*

---

## 8. References

**Papers**
1. Rombach, R. et al. *High-Resolution Image Synthesis with Latent Diffusion Models.* CVPR 2022. [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)
2. Zhang, L., Rao, A., Agrawala, M. *Adding Conditional Control to Text-to-Image Diffusion Models* (ControlNet). ICCV 2023. [arXiv:2302.05543](https://arxiv.org/abs/2302.05543)
3. Ho, J., Salimans, T. *Classifier-Free Diffusion Guidance.* 2022. [arXiv:2207.12598](https://arxiv.org/abs/2207.12598)
4. Meng, C. et al. *SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations.* ICLR 2022. [arXiv:2108.01073](https://arxiv.org/abs/2108.01073)
5. Zhao, W. et al. *UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models.* NeurIPS 2023. [arXiv:2302.04867](https://arxiv.org/abs/2302.04867)
6. Radford, A. et al. *Learning Transferable Visual Models From Natural Language Supervision* (CLIP). ICML 2021. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
7. Hessel, J. et al. *CLIPScore: A Reference-free Evaluation Metric for Image Captioning.* EMNLP 2021. [arXiv:2104.08718](https://arxiv.org/abs/2104.08718)
8. Canny, J. *A Computational Approach to Edge Detection.* IEEE TPAMI, 1986.

**Models (Hugging Face)**
- Stable Diffusion v1.5: [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) (CreativeML OpenRAIL-M licence)
- ControlNet v1.1 by L. Zhang: [scribble](https://huggingface.co/lllyasviel/control_v11p_sd15_scribble), [lineart](https://huggingface.co/lllyasviel/control_v11p_sd15_lineart), [canny](https://huggingface.co/lllyasviel/control_v11p_sd15_canny)
- CLIP ViT-B/16: [openai/clip-vit-base-patch16](https://huggingface.co/openai/clip-vit-base-patch16)

**Libraries**
- [🤗 Diffusers](https://github.com/huggingface/diffusers)
- [🤗 Transformers](https://github.com/huggingface/transformers)
- [PyTorch](https://pytorch.org)
- [OpenCV](https://opencv.org)
- [scikit-image](https://scikit-image.org)
- [Gradio](https://gradio.app)
- pandas, matplotlib, PyYAML

**Sketches.** All sketches in `sketches/` were drawn by me (Haroon Ahmed) for this project.

**AI assistance.** Code structure and parts of the implementation were written with the help of an AI assistant (Claude, Anthropic). I reviewed, ran and tested all code, ran all experiments, and wrote the analysis.
