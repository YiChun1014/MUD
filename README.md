# MUD-Net

## Environment

Create the Conda environment from the provided environment file:

```bash
cd /path/to/MUD-Net
conda env create -f environment.yaml
conda activate mud
```

The environment installs OpenAI CLIP from its official GitHub repository. On
the first run, `clip.load("ViT-B/32")` may also download the pretrained CLIP
weights if they are not already cached.

## Data preparation

Place the prepared data in the `data/` directory:

```text
data/
|-- AVVP_train.csv
|-- AVVP_val_pd.csv
|-- AVVP_test_pd.csv
`-- feats/
    |-- vggish/
    |-- res152/
    |-- r2plus1d_18/
    |-- CLIP/
    |   |-- features/
    |   `-- segment_pseudo_labels/
    `-- CLAP/
        |-- features/
        `-- segment_pseudo_labels/
```

### Pre-extracted features

Download the audio features (VGGish), 2D visual features (ResNet152), and 3D
visual features (ResNet (2+1)D) from
[AVVP-ECCV20](https://github.com/YapengTian/AVVP-ECCV20). Put them in
`data/feats/` so that the extracted directories are:

```text
data/feats/vggish/
data/feats/res152/
data/feats/r2plus1d_18/
```

The annotation files `AVVP_train.csv`, `AVVP_val_pd.csv`, and
`AVVP_test_pd.csv` are expected directly under `data/`.

### CLIP features and segment-level pseudo labels

Download [`CLIP.zip`](https://huggingface.co/datasets/NTUBarista/valor_features/blob/main/CLIP.zip),
put it in `data/feats/`, and unzip it there. The resulting directory must
contain:

```text
data/feats/CLIP/features/
data/feats/CLIP/segment_pseudo_labels/
```

### CLAP features and segment-level pseudo labels

Download [`CLAP.zip`](https://huggingface.co/datasets/NTUBarista/valor_features/blob/main/CLAP.zip),
put it in `data/feats/`, and unzip it there. The resulting directory must
contain:

```text
data/feats/CLAP/features/
data/feats/CLAP/segment_pseudo_labels/
```

The default MUD-Net configuration uses the CLAP audio features, CLIP visual
features, and ResNet (2+1)D spatiotemporal features.

## Testing

The best-performing checkpoint is provided. Run the following command to
directly reproduce the results reported in the paper:

```bash
bash code/test_mud.sh
```

## Training

If you want to retrain MUD-Net, activate the environment and run:

```bash
conda activate mud
bash code/train_mud.sh
```

The checkpoints are written to `models/model_name`.
