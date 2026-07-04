# Fall Detection

Human activity recognition and fall detection from video, using pose-based deep learning
models (2D/3D CNN, Vision Transformer, ST-GCN, TCN+Transformer) and a CLIP-based
vision-language model (zero-shot and few-shot), served through a Flask web app.

## Demo

[![Watch the demo](https://img.youtube.com/vi/WSCsatEl120/maxresdefault.jpg)](https://youtu.be/WSCsatEl120)


## Overview

The pipeline extracts 17-point human skeletons from video with YOLOv11-pose, splits the
skeleton sequence into overlapping time windows, and classifies each window with one of
several interchangeable models:

- **KTH activity models** — 2D CNN (ResNet / LeNet), 3D CNN (Simple / Deep), Vision
  Transformer, ST-GCN, and TCN+Transformer, trained on the KTH dataset's six activities
  (walking, jogging, running, boxing, hand-waving, hand-clapping).
- **Supervised fall detection models** — the same 2D CNN / 3D CNN / ViT architectures,
  but trained directly on binary Fall vs. No-Fall labels instead of KTH activities.
- **VLM fall detection** — CLIP (ViT-B/32) used two ways:
  - *Zero-shot*: contrastive scoring against a balanced bank of ~200 fall / non-fall
    text prompts, no training required.
  - *Few-shot*: a lightweight linear classifier trained on top of frozen CLIP image
    embeddings from a handful of labeled examples.
- **Simple pose-based heuristic** — a rule-based fall detector using joint geometry,
  included as a fast baseline.

All window-based methods use 32-frame windows with 50% overlap and a majority vote across
windows for the final video-level decision.

## Results

40-video balanced test set (20 fall / 20 non-fall), single-person fall videos vs. KTH
activities as non-fall, produced by `evaluation/evaluate_all.py`:

| Method                | Accuracy | Precision | Recall | F1   | Avg. inference / window |
|-----------------------|:--------:|:---------:|:------:|:----:|:------------------------:|
| Few-shot VLM          | 100%     | 1.00      | 1.00   | 1.00 | ~0.042 s |
| 3D CNN (Simple)       | 100%     | 1.00      | 1.00   | 1.00 | ~0.146 s |
| Vision Transformer    | 100%     | 1.00      | 1.00   | 1.00 | ~0.150 s |
| 2D CNN (ResNet)       | 97.5%    | 1.00      | 0.95   | 0.97 | ~0.147 s |
| Zero-shot VLM         | 92.5%    | 0.87      | 1.00   | 0.93 | ~0.040 s |
| Rule-based heuristic  | 70%      | 1.00      | 0.40   | 0.57 | ~0.165 s |

All methods use 32-frame windows with 50% overlap, with a majority vote across a video's
windows for the final accuracy/precision/recall/F1. "Avg. inference / window" is measured
differently depending on the method, so it isn't a strict apples-to-apples number:

- **CNN / ViT / rule-based**: pose is extracted once per video with YOLOv11-pose
  (`yolo11n-pose.pt`, the nano variant), then that one-time cost is divided evenly across
  the video's windows and added to each window's classifier (or rule) forward pass.
- **VLM (zero-shot / few-shot)**: only 1 frame is sampled per 32-frame window (not all 32),
  encoded once with CLIP, then classified — contrastively against a fixed fall/non-fall
  prompt bank for zero-shot, or with a small linear classifier for few-shot. The prompt
  bank itself is encoded once up front and reused across all windows, not re-encoded per
  window.

## Project Structure

```
├── src/                       Core models, training, skeleton extraction, preprocessing
├── vlm/                       CLIP wrapper, zero-shot inference, few-shot training
├── evaluation/                Evaluation pipeline, prompt bank, CNN/VLM training scripts
├── configs/                   YAML configs for models, training, extraction, VLM
├── webapp/                    Flask web app (UI + inference for all model types)
└── requirements.txt           All dependencies (core + web app)
```

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

YOLOv11-pose weights (`yolo11n-pose.pt`) are downloaded automatically by `ultralytics` on
first use.

Model checkpoints are not included in this repository. Train your own with the scripts in
`src/` (KTH activity models) and `evaluation/` (fall/no-fall CNN and few-shot VLM models),
or point `webapp/config.py` / `configs/vlm_config.yaml` at your own checkpoint paths.

## Running the Web App

```bash
cd webapp
python app.py
```

Then open `http://localhost:5000`, upload a video, and choose a model to run inference.

## Preparing Training Data

Datasets aren't included in this repository, and the KTH activity models train on
pre-extracted skeletons, not raw video, so there's a one-time data prep step before
`src/train.py` will find anything.

**KTH activity models:**

1. Download the [KTH action dataset](https://www.csc.kth.se/cvap/actions/) and arrange it as
   `<kth_root>/{walking,jogging,running,boxing,handwaving,handclapping}/*.avi`.
2. Extract skeletons with YOLOv11-pose:
   ```bash
   python src/skeleton_extraction.py <kth_root> ./data yolov11
   ```
3. Preprocess the extracted skeletons into fixed-length training clips:
   ```bash
   python src/preprocessing.py ./data/yolov11 ./data/yolov11/preprocessed
   ```
   `src/train.py` (below) auto-detects `./data/yolov11/preprocessed` once it exists.

**Supervised fall/no-fall models and the few-shot VLM classifier** don't need a separate
preprocessing step — `evaluation/train_cnn_models.py` and `evaluation/train_few_shot_vlm.py`
extract skeletons/embeddings directly from your video files. You'll need your own fall
videos (single-person, one fall per video) and non-fall videos (KTH works, or your own);
point `--fall_base_dir` / `--non_fall_videos_dir` at those folders.

## Training and Evaluation

```bash
# Train a KTH activity model (after the data prep above)
python src/train.py --model_type 2dcnn_resnet --config configs/training_config_2dcnn_resnet.yaml

# Train a supervised fall/no-fall CNN
python evaluation/train_cnn_models.py --model_type 2dcnn_resnet \
  --fall_base_dir <path/to/fall/videos> --non_fall_videos_dir <path/to/non_fall/videos>

# Train the few-shot VLM classifier
python evaluation/train_few_shot_vlm.py \
  --fall_base_dir <path/to/fall/videos> --non_fall_videos_dir <path/to/non_fall/videos>

# Run the full evaluation suite (all methods, one shared test set)
python evaluation/evaluate_all.py --shared_test_set <path> --few_shot_classifier <path> \
  --cnn_checkpoints 2dcnn_resnet:<path> 3dcnn_simple:<path> vit:<path>
```

## Citation

If you use this work, please cite:

```bibtex
@incollection{Shahabian2026b,
    author = "Shahabian Alashti, Mohamad Reza and Ghamati, Khashayar and Zaraki, Abolfazl and Zhang, Baobing and Holthaus, Patrick and Alingal Meethal, Shadiya and Velmurugan, Vignesh and Lakatos, Gabriella and Dickinson, Angela and Amirabdollahian, Farshid",
    title = "{Vision–Language Model for Fall Detection in Socially Assistive Robotics: Zero-Shot Prompting and Few-Shot Calibration}",
    year = "in press",
    booktitle = "International Conference on Social Robotics (ICSR 2026)",
    series = "Lecture Notes in Computer Science",
    address = "London, UK",
    publisher = "Springer"
}
```

## License

MIT — see [LICENSE](LICENSE).
