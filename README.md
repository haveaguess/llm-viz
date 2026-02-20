
# LLM Visualization — Annotated with nanoGPT Source

**Live demo:** [llm-viz-annotated.vercel.app/llm](https://llm-viz-annotated.vercel.app/llm)

This is a fork of [Brendan Bycroft's LLM Visualization](https://github.com/bbycroft/llm-viz) with added nanoGPT code annotations throughout the 3D walkthrough. Each phase of the visualization now includes the corresponding Python source from [Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT), so you can follow the animated forward pass alongside the actual code.

## What was added

All 10 walkthrough phases are annotated with inline code snippets from nanoGPT's `model.py`:

| Phase | Concept | Key code shown |
|-------|---------|---------------|
| 0. Intro | Model overview | `GPTConfig`, `GPT.forward` complete flow |
| 1. Preliminary | Code structure | Class overview (GPT, Block, MLP, etc.) |
| 2. Embedding | Token + position embedding | `wte(idx)`, `wpe(pos)`, `tok_emb + pos_emb` |
| 3. LayerNorm | Layer normalization | `F.layer_norm(...)`, `self.ln_1(x)` |
| 4. Self-Attention | Q, K, V and attention | `c_attn`, `q @ k.T`, masking, softmax, `att @ v` |
| 5. Softmax | Normalization to probabilities | `F.softmax(att)`, `F.softmax(logits)` |
| 6. Projection | Head reassembly + residual | `view(B,T,C)`, `c_proj(y)`, residual add |
| 7. MLP | Feed-forward network | `c_fc` -> `gelu` -> `c_proj` -> dropout |
| 8. Transformer | Block stacking | `Block.forward`, `for block in self.transformer.h` |
| 9. Output | Logits, sampling, loss | `lm_head(x)`, `multinomial`, `cross_entropy` |

### Annotation features

- **Forward-pass first**: Code snippets lead with the forward pass being animated, with class definitions collapsed for reference
- **Granular per-step snippets**: Compact single-line code snippets appear at each animation step (e.g., separate snippets for token embed, position embed, and addition)
- **Concrete tensor shapes**: Shapes match the visualization's model — `(1, 11, 48)` instead of abstract `(B, T, C)`
- **Variable name mapping**: Commentary bridges visualization names to code names (e.g., "input embedding (`x`)", "Q vectors (`q`)")
- **Python syntax highlighting**: Keywords, strings, comments, builtins, and numbers are colour-coded
- **Collapsible snippets**: Click the header to expand/collapse

### Visualization model

The 3D walkthrough uses a tiny 85K-parameter GPT that sorts 6 letters (A, B, C):

| Parameter | Value |
|-----------|-------|
| `n_embd` (C) | 48 |
| `n_head` | 3 |
| `n_layer` | 3 |
| `block_size` (T) | 11 |
| `vocab_size` | 3 |

A matching nanoGPT training config is at [`config/train_shakespeare_char_bbycroft.py`](https://github.com/haveaguess/nanoGPT_playgroup_202602/blob/master/config/train_shakespeare_char_bbycroft.py) in our nanoGPT playgroup repo.

## Running locally

1. Install dependencies: `yarn`
2. Start the dev server: `yarn dev`
3. Open [http://localhost:3002/llm](http://localhost:3002/llm)

## Deployment

Deployed to Vercel (free tier, auto-detects Next.js):

```sh
npx vercel --prod
```

## Credits

- Original visualization: [Brendan Bycroft](https://github.com/bbycroft/llm-viz)
- nanoGPT: [Andrej Karpathy](https://github.com/karpathy/nanoGPT)
- Code annotations: Added by this fork
