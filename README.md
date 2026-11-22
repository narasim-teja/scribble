# Scribble on the brain

A web page where a picture is a language model. Each pixel is one weight. The
page runs the model straight from the picture, and you can paint on it to
damage the model's writing, then heal it.

Start with [docs/idea.md](docs/idea.md) (what and why), then
[docs/technical.md](docs/technical.md) (how, with milestones).

## Layout

| Folder | What |
|---|---|
| `docs/` | The idea and the technical brief |
| `train/` | Python: data preparation, the one-bit model, training, sampling |
| `bench/` | Benchmark behind the engine-language decision |

## Training

```
cd train
uv sync
uv run python data.py                                   # download and encode TinyStories
uv run python train.py --vocab bpe512 --minutes 120 --name bpe512-4m
uv run python sample.py bpe512-4m --flip 0 0.001 0.01 0.05
uv run pytest && uv run ruff check .
```

## Credits

The brain shape is a modified version of FreeSurfer's fsaverage6 surface, used
under the FreeSurfer Software License (`web/public/licenses/fsaverage.txt`).

## The web page

`web/` is a Vite and TypeScript project, run with Bun.

- `src/model/` runs the model from its picture (no browser needed).
- `src/worker/` runs it off the main thread and takes paint strokes.
- `src/view/` draws the weights on a 3D brain; `src/ui/` is the text panel.

The page needs data files in `web/public/data/`, all git-ignored:

    cd train
    uv run python export.py bpe512-4m      # the model as a picture
    uv run python reference.py bpe512-4m   # answers the engine must match
    cp runs/bpe512-4m/brain.png ../web/public/data/
    cd .. && uv run --no-project --with nilearn python web/tools/build_mesh.py

    cd web
    bun install
    bun run dev      # the page
    bun test         # engine against Python's answers
    bun run build    # type-check and bundle into dist/
    bun run kernel   # rebuild src/model/kernel.wasm from web/kernel (needs Rust)
