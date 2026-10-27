"""R48: write one customer's occasion rows as a chat transcript for a zero-shot LLM.

The model is told exactly what ProductGPT is given -- the products on offer, the
products obtained since the previous occasion, every earlier decision, and the
attribute row of each product (the table ProductGPT embeds) -- as text.  It is
NOT told the game's rules (pity, the 50/50 guarantee), shown worked examples,
or given campaign ids or the clock.  That is what "most basic zero-shot" means.

Each occasion is one user turn followed by one assistant turn whose whole
content is the decision letter.  Attention is causal, so a single forward pass
over the transcript gives, at every answer, the model's next-token distribution
given only the turns before it: the same as asking the occasions one at a time.
scripts/test_r48_zeroshot.py checks that equivalence and the absence of
look-ahead.

The chat format is Qwen3's ChatML in non-thinking mode, written out by hand so
every assistant turn is formatted exactly like a fresh generation prompt (the
tokenizer's template treats the last turn differently from earlier ones); the
test compares it with tokenizer.apply_chat_template.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

FIRST, LAST = 13, 56            # level-6 product token ids
UNK_PROD = 59
LETTERS = "ABCDEFGHI"           # decision 1..9 -> answer letter

OPTIONS = (
    "1 draw on the Regular banner",
    "10 draws on the Regular banner",
    "1 draw on Featured banner A",
    "10 draws on Featured banner A",
    "1 draw on Featured banner B",
    "10 draws on Featured banner B",
    "1 draw on the Weapon banner",
    "10 draws on the Weapon banner",
    "no draw during this 24-hour period",
)

# parametric.state.ELEMENTS / WTYPES order, in the game's own words
ELEMENT_NAMES = ("Cryo", "Geo", "Hydro", "Pyro", "Electro", "Anemo", "Dendro")
WTYPE_NAMES = ("sword", "claymore", "bow", "catalyst", "polearm")

IM_START, IM_END = "<|im_start|>", "<|im_end|>"
ASSISTANT_PREFIX = f"{IM_START}assistant\n<think>\n\n</think>\n\n"


def describe_product(pid: int, prod: Dict[str, np.ndarray]) -> str:
    rarity = int(prod["rarity"][pid])
    is_char = int(prod["fig"][pid]) == 1
    parts = [f"{rarity}-star" if rarity else "unrated",
             "limited" if int(prod["lto"][pid]) else "standard",
             "character" if is_char else "weapon"]
    e, w = int(prod["element"][pid]), int(prod["wtype"][pid])
    if is_char:
        if e >= 0:
            parts.append(ELEMENT_NAMES[e])
        parts.append("female" if int(prod["female"][pid]) else "male")
        if w >= 0:
            parts.append(f"uses a {WTYPE_NAMES[w]}")
    elif w >= 0:
        parts.append(WTYPE_NAMES[w])
    return f"P{pid}: " + ", ".join(parts)


def system_prompt(prod: Dict[str, np.ndarray]) -> str:
    legend = "\n".join(describe_product(p, prod) for p in range(FIRST, LAST + 1))
    options = "\n".join(f"{LETTERS[k]}: {OPTIONS[k]}" for k in range(9))
    return (
        "You will read the purchase record of one player of a mobile game in which "
        "players pay for random draws of characters and weapons. The record is a "
        "sequence of occasions. At each occasion the player either buys draws on one "
        "banner or makes no draw for 24 hours.\n\n"
        "Banners: the Regular banner is always open. Featured banner A and Featured "
        "banner B each feature one limited character; banner B is not always open. "
        "The Weapon banner features two limited weapons. What is featured changes "
        "from time to time.\n\n"
        f"Products are coded P{FIRST} to P{LAST}:\n{legend}\n\n"
        "At each occasion you are told what is featured (whenever it changes) and "
        "what the player obtained since the previous occasion. Predict the player's "
        f"decision at that occasion with one letter:\n{options}\n\n"
        "Answer with the letter only."
    )


def _items(ids: Sequence[int]) -> str:
    """'P13, P20 x2, an unlisted item' -- duplicates counted, order of first appearance."""
    counts: Dict[str, int] = {}
    for v in ids:
        v = int(v)
        if FIRST <= v <= LAST:
            key = f"P{v}"
        elif v == UNK_PROD:
            key = "an unlisted item"
        else:
            continue                      # PAD and structural tokens
        counts[key] = counts.get(key, 0) + 1
    return ", ".join(k if n == 1 else f"{k} x{n}" for k, n in counts.items())


def _offer(lto_row: Sequence[int]) -> str:
    def slot(v):
        v = int(v)
        return f"P{v}" if FIRST <= v <= LAST else None
    a, b, w1, w2 = (slot(v) for v in lto_row)
    weapons = ", ".join(x for x in (w1, w2) if x) or "nothing listed"
    return (f"Featured now: banner A {a or 'nothing listed'}; "
            f"banner B {b or 'closed'}; Weapon banner {weapons}.")


def user_turn(content: str) -> str:
    return f"{IM_START}user\n{content}{IM_END}\n"


def transcript(label: np.ndarray, prev: np.ndarray, lto: np.ndarray, obtained: np.ndarray,
               inv_init: np.ndarray, length: int, system: str
               ) -> Tuple[str, List[int]]:
    """Return (text, answer_char_index per occasion) for one customer's window.

    label/prev (S,), lto (S,4), obtained (S,10), inv_init (44,) as in the R46
    occasion table.  Every row t < length must carry a decision 1..9.
    """
    text = f"{IM_START}system\n{system}{IM_END}\n"
    answers: List[int] = []
    for t in range(int(length)):
        d = int(label[t])
        if not 1 <= d <= 9:
            raise ValueError(f"row {t} inside the window has label {d}")
        parts = [f"Occasion {t + 1}."]
        if t == 0:
            owned = [p for k, n in enumerate(inv_init) for p in [FIRST + k] * int(n)]
            if owned:
                parts.append(f"Owned before this record: {_items(owned)}.")
            if 1 <= int(prev[0]) <= 9:
                parts.append(f"Decision just before this record: {LETTERS[int(prev[0]) - 1]}.")
        if t == 0 or not np.array_equal(lto[t], lto[t - 1]):
            parts.append(_offer(lto[t]))
        got = _items(obtained[t])
        if got:
            parts.append(f"Obtained since the previous occasion: {got}.")
        text += user_turn(" ".join(parts)) + ASSISTANT_PREFIX
        answers.append(len(text))
        text += LETTERS[d - 1] + IM_END + "\n"
    return text, answers
