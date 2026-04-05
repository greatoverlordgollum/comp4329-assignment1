# Stage 1 Work Log (30 March 2026)

## Sequential actions and outcomes

4. Inspected training stack for hard runtime crashes.
   - Read `TrainTools/train.py`, `TrainTools/train_utils.py`, `Losses/loss.py`, schedulers, and data/eval loaders.
   - Found critical bugs:
     - Invalid `argparse.Namespace` construction.
     - `.backward()` called on `loss.item()` (float).
     - NLL loss input/target argument order swapped.
     - Notebook config used `scheduler_name="none"`, but scheduler registry lacked `none`.

5. Applied first set of runtime fixes.
   - `TrainTools/train.py`: fixed `argparse.Namespace(**kwargs)` construction.
   - `TrainTools/train_utils.py`: changed to `loss.backward()`, clipped gradients before optimizer step.
   - `Losses/loss.py`: corrected `F.nll_loss(p1, y1)` and `F.nll_loss(p2, y2)` ordering.
   - `Schedulers/scheduler.py`: added explicit `none` scheduler alias.
   - `Schedulers/lambda_scheduler.py`: fixed LR rule from `base_lr + factor` to `base_lr * factor`.

6. Updated data setup compatibility for modern spaCy.
   - `Tools/download.py`: default model changed from `en` to `en_core_web_sm`.
   - Added fallback behavior for legacy `en` requests.
   - Notebook dependency cell updated to `%pip install -r requirements.txt -q` and spaCy download uses `en_core_web_sm`.

7. Ran first terminal smoke test (1-step train) to validate true runtime behavior.
   - Command executed minimal `train(...)` with local `_data` and `scheduler_name='none'`.
   - New crash found in positional encoding dimensions within `Models/encoder.py`.

8. Investigated model stack and fixed multiple forward-pass blockers.
   - Read `Models/encoder.py`, `Models/qanet.py`, `Models/embedding.py`, `Models/conv.py`, `Models/attention.py`, `Models/heads.py`, normalization modules, and optimizers.
   - Applied runtime-critical fixes:
     - `Models/encoder.py`
       - Rewrote positional encoding tensor construction to correct shapes.
       - Fixed multi-head reshape/permute order.
       - Fixed normalization index from `self.norms[i+1]` to `self.norms[i]`.
       - Restored residual connection after self-attention (`self.self_att(...) + res`).
     - `Models/embedding.py`
       - Corrected highway transpose to `x.transpose(1, 2)`.
       - Fixed char embedding permute to `[B, d_char, L, char_len]` (`permute(0, 3, 1, 2)`).
     - `Models/conv.py`
       - Corrected Conv1d sliding window axis from `unfold(1, ...)` to `unfold(2, ...)`.
       - Fixed Conv2d width padding height dimension after prior padding.
       - Corrected depthwise-separable ordering to `pointwise(depthwise(x))`.
     - `Models/qanet.py`
       - Fixed swapped word/char embedding lookups for context/question.
       - Fixed CQAttention mask argument order.
     - `Models/attention.py`
       - Fixed C2Q attention batch matmul (matmul = matrix multiplication) (`A = torch.bmm(S1, Q)`).
     - `Models/heads.py`
       - Fixed pointer concatenation dim (`dim=1` for both starts/ends).
     - `Models/Normalizations/layernorm.py`
       - Fixed mean/var keepdim broadcasting.
       - Corrected affine equation to `x_norm * weight + bias`.
     - `Models/Normalizations/groupnorm.py`
       - Fixed group reshape ordering to `[B, G, C//G, ...]`.

9. Fixed optimizer runtime bugs that could crash alternative configs.
   - `Optimizers/adam.py`
     - Corrected state keys (`exp_avg`, `exp_avg_sq`).
     - Corrected second-moment update to `addcmul_(grad, grad, ...)`.
     - Corrected bias-correction exponents to powers (`beta**t`).
   - `Optimizers/sgd_momentum.py`
     - Fixed velocity state key mismatch (`velocity` consistently).
     - Corrected momentum update to additive form (`v = mu*v + grad`).

10. Fixed evaluation prediction extraction bug.
    - `EvaluateTools/eval_utils.py`: argmax dimension corrected from `dim=0` to `dim=1` for batch-wise spans.

12. Fixed scheduler serialization compatibility.
    - `Schedulers/scheduler.py`: replaced inline lambda with module-level `_constant_one` function for `lambda`/`none` schedulers.

14. Validated evaluate entrypoint and fixed checkpoint loading mismatches.
    - First evaluate smoke test revealed API mismatch in my test call; corrected to `test_num_batches`.
    - Then found checkpoint key mismatch and PyTorch 2.6+ load behavior issue.
    - `EvaluateTools/evaluate.py` fixes:
      - Load model state from `model_state` with fallback to `model`.
      - Set `torch.load(..., weights_only=False)` for local trusted checkpoint compatibility.


## Files changed in Stage 1
- `Tools/download.py`
- `requirements.txt`
- `assignment1.ipynb`
- `TrainTools/train.py`
- `TrainTools/train_utils.py`
- `Losses/loss.py`
- `Schedulers/scheduler.py`
- `Schedulers/lambda_scheduler.py`
- `Models/encoder.py`
- `Models/embedding.py`
- `Models/conv.py`
- `Models/qanet.py`
- `Models/attention.py`
- `Models/heads.py`
- `Models/Normalizations/layernorm.py`
- `Models/Normalizations/groupnorm.py`
- `Optimizers/adam.py`
- `Optimizers/sgd_momentum.py`
- `EvaluateTools/eval_utils.py`
- `EvaluateTools/evaluate.py`

---

## NaN Loss Fix Log (30 March 2026, follow-up)

17. Isolated primary numerical instability source.
    - Inspected custom dropout implementation in `Models/dropout.py`.
    - Found incorrect inverted-dropout scaling:
      - Before: `x * mask / p`
      - Correct: `x * mask / (1 - p)`
    - The old formula amplifies activations by ~10x at `p=0.1`, which can quickly destabilize logits and produce NaNs.

18. Patched dropout to stable inverted-dropout behavior.
    - `Models/dropout.py` updated:
      - compute `keep_prob = 1.0 - p`
      - return zeros when `keep_prob <= 0.0`
      - scale by `1 / keep_prob`

Loss is now finite and all the pipeline runs at this point. Not well, but it runs.

## Additional files changed in follow-up
- `Models/dropout.py`
