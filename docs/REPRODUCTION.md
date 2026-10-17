# Reproduction and review notes

## What changed

- README commands now match the flat repository layout.
- Evaluation checkpoint/output paths are CLI options; pruning no longer selects cuda:7 by default.
- Missing OpenAI CLIP and TensorBoard dependencies are declared. Tokenization can use CLIP's bundled BPE file.
- Analysis detaches tensors before NumPy conversion, validates projection inputs, fixes image-only formatting and mismatched plot column names, and creates output directories.
- Failed pruning evaluations now fail the run rather than silently writing incomplete success-looking CSVs.

## Important limits

No GPU training or paper-number reproduction was performed in this refresh. CPU regression tests cover projection metrics and plotting; syntax checks cover Python files. Dataset downloads and checkpoint-dependent evaluation require your environment.

The original similarity-based caption generator still contains machine-specific input/output settings and can build large similarity matrices. Inspect it before use; it is not equivalent to uniform random caption shuffling. Paper misalignment probability and a generator's swap ratio should not be conflated.

The training CSV contains image-relative-path and caption in its first two columns. Images must end in .png, .jpg or .jpeg. A typical launch is:

```bash
torchrun --nproc_per_node=4 train.py --train-data data/pairs.csv \
  --root data/images --imagenet-root data/imagenet \
  --resume checkpoints/cc3m_laclip.pt --output-dir results/finetune \
  --model CLIP_VITB16 --epochs 10 --batch-size 256 --lr 1e-3 --wd 0.5
```

Verify the printed missing/unexpected checkpoint keys: the legacy loader uses strict=False, and its distributed prefix assumptions are not corrected by this refresh. Missing requested checkpoint files now fail immediately. This command is a configuration example, not a verified training run.

Evaluation CLI imports could not be smoke-tested in the review environment because torchvision is not installed. The analysis tests use synthetic tensors/mocked checkpoints, not a full model forward pass.

Projection-only fine-tuning requires a compatible pretrained checkpoint. Supply `--resume PATH` explicitly; otherwise `train.py` may start from initialization or auto-resume a local output checkpoint. Read its checkpoint-loading behavior before comparing runs. Its root-level modules are the current entry points. CUDA/distributed execution, dataset CSV schema, ImageNet validation and training hyperparameters must be prepared separately.

The reciprocal 'per_neuron_purity' diagnostic can be infinite for exactly orthogonal projections; the primary interpretable metric is average absolute cosine similarity. Zero vectors are rejected because their cosine similarity is undefined.

## Figure provenance

- `overview.png`: conceptual illustration approved for the project page, based on the user's talk. Not a paper result.
- `paper-figure1.png`, `paper-figure2.png`: original NeurIPS 2025 paper panels, extracted for the project page; no data redrawn.
- Old figures remain in `image/` for historical compatibility; README uses the paper-derived versions.

No unpublished LLaVA implementation is invented or implied.
