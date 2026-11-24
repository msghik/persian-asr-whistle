# cat-ext-stubs

`import cat` (CAT's `cat/__init__.py`) imports `cat.lm` and `cat.rnnt`, which import the
compiled extensions `gather` (torch-gather) and `warp_rnnt`; `cat/ctc/train.py` imports
`ctcdecode`. None of them is *called* on the CTC training path we use:

| module | imported by | called when | our pipeline |
|---|---|---|---|
| `ctcdecode` | `cat/ctc/train.py:19`, `cat/ctc/decode.py:25` | `--eval-error-rate` (train.py:70-76); `cat.ctc.decode` | never set / not used (we decode with `persian_asr.decode`) |
| `gather` | `cat/lm/train.py:17` | neural LM training | not used (n-gram LM via KenLM `lmplz`) |
| `warp_rnnt` | `cat/rnnt/*` | RNN-T training | not used |

Why stubs instead of building them: ctcdecode bundles OpenFST 1.6.7, which does not compile
under C++20, and PyTorch >= 2.10-ish headers require C++20 (verified locally with torch 2.15:
`#error C++20 or later compatible compiler is required`, then OpenFST allocator errors).
The CUDA ones also need nvcc at install time. Every stub raises on use, so nothing silently changes.
