import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional

AI_RATE = 15
DECISION_IDS = torch.tensor([1,2,3,4,5,6,7,8,9], dtype=torch.long)
PAD_ID = 0  # used only as a harmless placeholder if needed

def extract_last_outcomes_from_history(history_tokens: List[int], ai_rate: int = AI_RATE) -> List[int]:
    """
    From the last complete 15-token block in history:
      block = [LTO4][OUT10][PREV_DEC1]
    outcomes are positions 4..13 (0-based within block).
    """
    assert len(history_tokens) >= ai_rate and len(history_tokens) % ai_rate == 0, \
        "history_tokens must be non-empty and a multiple of 15."
    last_block = history_tokens[-ai_rate:]
    return last_block[4:14]  # length 10

@torch.no_grad()
def sample_decision_from_logits(
    logits_last_pos: torch.Tensor,      # (V,)
    decision_ids: torch.Tensor,         # (9,)
    temperature: float = 1.0,
    greedy: bool = False,
) -> int:
    """
    Restrict logits to decision_ids and sample (or argmax).
    Returns decision token in {1..9}.
    """
    device = logits_last_pos.device
    ids = decision_ids.to(device)
    dec_logits = logits_last_pos[ids]  # (9,)

    if temperature != 1.0:
        dec_logits = dec_logits / max(temperature, 1e-8)

    probs = F.softmax(dec_logits, dim=-1)

    if greedy:
        k = torch.argmax(probs).item()
    else:
        k = torch.multinomial(probs, 1).item()

    return int(ids[k].item())

@torch.no_grad()
def generate_campaign28_step1_fixed_outcomes(
    model,
    history_tokens: List[int],          # Campaigns 1–27 AggregateInput tokens (multiple of 15)
    lto28_tokens: List[int],            # 4 integers: Campaign 28 firm action tokens
    fixed_outcomes_after_step0: List[int],  # 10 integers used for steps >= 1 (frozen outcomes)
    device: torch.device,
    init_prev_dec: Optional[int] = None,    # warm-start previous decision; if None, user should supply
    max_steps28: int = 500,
    stop_decision: int = 9,
    temperature: float = 1.0,
    greedy: bool = False,
) -> Dict[str, Any]:
    """
    Step 1: outcomes are fixed (step0 from history; step>=1 from fixed_outcomes_after_step0).
    Generates decisions until decision==9 or until max_steps28.

    Returns:
      - seq_campaign28: token stream for campaign 28 blocks (each 15 tokens)
      - seq_full: history_tokens + seq_campaign28
      - decisions28: sampled decision list (includes terminal 9 if hit)
      - stopped: bool
      - stop_step: int|None  (0-based within campaign 28 decisions)
    """
    assert len(lto28_tokens) == 4, "lto28_tokens must have length 4."
    assert len(fixed_outcomes_after_step0) == 10, "fixed_outcomes_after_step0 must have length 10."
    assert len(history_tokens) >= AI_RATE and len(history_tokens) % AI_RATE == 0, \
        "history_tokens must be non-empty and a multiple of 15."

    model.eval()

    # Step0 outcomes come from the last observed block in campaign 27
    outcomes_step0 = extract_last_outcomes_from_history(history_tokens)

    # Previous decision token that will be placed in the 15th slot of step0.
    # You should warm-start this with the last *actual* decision in campaign 27.
    # If you cannot provide it, you can fall back to the last slot token in history,
    # but that is conceptually "previous decision for the last observed step", not the last actual decision.
    if init_prev_dec is None:
        init_prev_dec = int(history_tokens[-1])  # fallback; recommend overriding this

    prev_dec = int(init_prev_dec)

    seq_full = list(history_tokens)
    seq_c28: List[int] = []
    decisions28: List[int] = []

    stopped = False
    stop_step: Optional[int] = None

    for t in range(max_steps28):
        if t == 0:
            out10 = outcomes_step0
        else:
            out10 = fixed_outcomes_after_step0

        # Construct the 15-token block: [LTO4][OUT10][PREV_DEC]
        block = list(lto28_tokens) + list(out10) + [prev_dec]

        seq_full.extend(block)
        seq_c28.extend(block)

        # Run model on the full prefix and read logits at the decision-position of this step.
        x = torch.tensor(seq_full, dtype=torch.long, device=device).unsqueeze(0)  # (1,T)
        logits = model(x)                                                         # (1,T,V)
        logits_at_decpos = logits[0, -1, :]                                       # (V,)

        dec = sample_decision_from_logits(
            logits_at_decpos, DECISION_IDS, temperature=temperature, greedy=greedy
        )
        decisions28.append(dec)

        # stop rule
        if dec == stop_decision:
            stopped = True
            stop_step = t
            break

        # update prev_dec for next step
        prev_dec = dec

    return {
        "seq_campaign28": seq_c28,
        "seq_full": seq_full,
        "decisions28": decisions28,
        "stopped": stopped,
        "stop_step": stop_step,
    }
