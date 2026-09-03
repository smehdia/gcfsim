# GCFSim

Train a goal-conditioned functional similarity model on one app graph, then retrieve a graph node for a query screenshot and goal.

## Recommended: standalone script

`gcfsim_standalone.py` is **self-contained** (no `fine_tune_v2` imports). Everything needed for load → train → retrieve lives in that one file.

```bash
python gcfsim_standalone.py \
  --graph /path/to/app \
  --screenshot shot.png \
  --goal "Open settings"
```

Stdout is the top-1 node id. Progress / checkpoint info goes to stderr.

**Dependencies:** `torch`, `transformers`, `sentence-transformers`, `Pillow`, `numpy`.

There is also `gcfsim.py`, a thin wrapper around the repo’s `fine_tune_v2` experiment code (same CLI shape). Prefer `gcfsim_standalone.py` for a portable demo.

## Inputs

| Flag | Required | What it is |
|------|----------|------------|
| `--graph` | yes | Directory of one explored app (see layout below) |
| `--screenshot` | yes | Query screenshot (png/jpg) |
| `--goal` | yes | Natural-language goal, e.g. `"Open settings"` |
| `--device` | no | `cuda` if available, else `cpu` |
| `--epochs` | no | Training epochs (default `20`) |
| `--ckpt-dir` | no | Embeddings + `best.pt` (default `outputs/gcfsim_standalone/<app>/`) |
| `--reuse` | no | Load `best.pt` instead of retraining |
| `--top-k` | no | If `> 1`, also print `node_id<TAB>score` for the top k |

`--graph` must be a **single app directory**. App id is the directory name.

### Graph directory

```
<app>/
  graph.json                      # nodes + links
  user_intents.json               # per-node intent strings (goals + targets)
  node_level_information.json     # optional page descriptions
  edge_level_information.json     # optional action descriptions
  screenshots/
    <node_id>.jpg                 # or .png
```

`user_intents.json` is required for training: unique intent texts become goals, and nodes that list an intent become its original targets.

## Pipeline

```
graph dir + screenshot + goal
        │
        ▼
 1. Load AppGraph (≤24 goals)
        │
        ▼
 2. Frozen SigLIP embeddings
    • every node screenshot
    • every training goal text
        │
        ▼
 3. Semantic target expansion (τ=0.95)
    T_g^sem = original targets ∪ nodes whose intents match the goal
        │
        ▼
 4. Continuation teacher (GCFSim / manual9)
    classes: states that share an optimal next-action signature toward T_g^sem
        │
        ▼
 5. Train SimpleGoalMetric  z(s, g) = F_θ(x_s, x_g)
        │
        ▼
 6. Retrieve: embed query (screenshot, goal), cosine-NN over graph nodes
        │
        ▼
    node_id
```

1. **Load.** `graph.json` + `user_intents.json` → nodes, edges, goals.
2. **Encode.** Frozen SigLIP (`google/siglip-base-patch16-384`) embeds screenshots and goal texts (cached under `--ckpt-dir`).
3. **Expand targets.** `all-mpnet-base-v2` scores node intents against each goal. Nodes with cosine ≥ `0.95` join the teacher target set (original targets are never dropped).
4. **Teacher.** Full recursive continuation classes: for each non-terminal state, the signature is the set of `(broad_action_intent(e), successor_class)` over optimal edges toward the expanded targets. Same signature ⇒ same class.
5. **Train.** `SimpleGoalMetric` fuses image + goal into `z(s, g)`. Pairs: same-class positives, same-distance / visual hard negatives, random negatives, goal-flips. Loss: BCE + SupCon + flip ranking. Checkpoint: `--ckpt-dir/best.pt`.
6. **Retrieve.** Query screenshot and `--goal` are SigLIP-encoded, mapped through `z(·, g)`, and ranked by cosine against every node with a screenshot. The query goal need not match a training goal id.

On a second run, pass `--reuse` (and the same `--ckpt-dir`) to skip training.

## Output

- **stdout:** top-1 `node_id`
- **stderr:** pipeline progress and checkpoint path
- **`--top-k N`:** first line is still top-1 id; then `N` lines of `node_id<TAB>cosine`

## Defaults

- Max goals per app: `24`
- Epochs: `20`
- Seed: `42`
- Target expansion threshold: `0.95`
- Pairs per goal (approx): `400`
