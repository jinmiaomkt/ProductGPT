import random

def random_check(prob):
    """Return True with probability = prob (e.g. 0.5 for 50%)."""
    return (random.random() < prob)

# -------------------------------------------------------------------------
# Data structures for each banner: pity counters, "guaranteed" flags, etc.
# -------------------------------------------------------------------------

def init_pity_state():
    """Initialize a dictionary to hold pity/guarantee states for all banners."""
    return {
        "regular": {
            "pulls_since_4star": 0,
            "pulls_since_5star": 0,
            # No guaranteed promotional concept in the standard banner:
            "guaranteed_5star_promotional": False, 
            "guaranteed_4star_promotional": False
        },
        "figure": {
            "pulls_since_4star": 0,
            "pulls_since_5star": 0,
            # If you lose the 50/50 on a 5★, next time is guaranteed promotional
            "guaranteed_5star_promotional": False,
            # If you fail to get one of the "featured" 4★, next 4★ is guaranteed
            "guaranteed_4star_promotional": False
        },
        "weapon": {
            "pulls_since_4star": 0,
            "pulls_since_5star": 0,
            # If you lose the 75/25 for a 5★, next 5★ is guaranteed promotional
            "guaranteed_5star_promotional": False,
            # If you lose the 75/25 for a 4★, next 4★ is guaranteed featured
            "guaranteed_4star_promotional": False
        }
    }

# -------------------------------------------------------------------------
# Banner-specific draw logic
# -------------------------------------------------------------------------

def draw_once_regular(state):
    """
    Simulate a single pull on the Regular (Wanderlust Invocation) banner.
    Returns a string describing the rarity and whether it was "featured" or not.
    Adjust as needed to track actual items.
    """
    state["pulls_since_4star"] += 1
    state["pulls_since_5star"] += 1

    # ===== Check forced 5★ at 90 pity =====
    if state["pulls_since_5star"] >= 90:
        # forced 5★
        state["pulls_since_5star"] = 0
        state["pulls_since_4star"] = 0
        return "5★ (Regular pity)"

    # ===== Check forced 4★ at 10 pity (with small chance to upgrade to 5★) =====
    if state["pulls_since_4star"] >= 10:
        # There is a small chance (0.6%) to "upgrade" to a 5★:
        if random_check(0.006):
            # got a 5★
            state["pulls_since_5star"] = 0
            state["pulls_since_4star"] = 0
            return "5★ (Upgrade from 4★ pity, Regular)"
        else:
            # 4★ guaranteed
            state["pulls_since_4star"] = 0
            # 5★ pity not reset
            return "4★ (Regular pity)"

    # ===== Otherwise, roll base rates: 0.6% for 5★, 5.1% for 4★, else 3★ =====
    r = random.random()
    if r < 0.006:
        # 5★
        result = "5★ (Regular normal)"
        state["pulls_since_5star"] = 0
        state["pulls_since_4star"] = 0
        return result
    elif r < 0.006 + 0.051:
        # 4★
        result = "4★ (Regular normal)"
        state["pulls_since_4star"] = 0
        # 5★ pity continues
        return result
    else:
        # 3★
        return "3★ (Regular)"

def draw_once_figure(state):
    """
    Simulate a single pull on the Figure (Character Event) banner.
    Incorporates:
      - 90-pull pity for 5★
      - 10-pull pity for 4★
      - 50/50 for the promotional 5★
      - 50/50 for the featured 4★
    """
    # Pity counters
    state["pulls_since_4star"] += 1
    state["pulls_since_5star"] += 1

    # ===== forced 5★ at 90 =====
    if state["pulls_since_5star"] >= 90:
        state["pulls_since_5star"] = 0
        state["pulls_since_4star"] = 0
        return get_5star_figure(state, forced=True)

    # ===== forced 4★ at 10 (chance to upgrade to 5★ is 0.6%) =====
    if state["pulls_since_4star"] >= 10:
        if random_check(0.006):
            # 5★
            state["pulls_since_5star"] = 0
            state["pulls_since_4star"] = 0
            return get_5star_figure(state, forced=True)
        else:
            # guaranteed 4★
            state["pulls_since_4star"] = 0
            return get_4star_figure(state, forced=True)

    # ===== normal roll: 0.6% 5★, 5.1% 4★, rest 3★ =====
    r = random.random()
    if r < 0.006:
        # 5★
        state["pulls_since_5star"] = 0
        state["pulls_since_4star"] = 0
        return get_5star_figure(state, forced=False)
    elif r < 0.006 + 0.051:
        # 4★
        state["pulls_since_4star"] = 0
        return get_4star_figure(state, forced=False)
    else:
        return "3★ (Figure)"

def draw_once_weapon(state):
    """
    Simulate a single pull on the Weapon Event banner.
    Incorporates:
      - 80-pull pity for 5★
      - 10-pull pity for 4★
      - 75/25 for promotional 5★
      - 75/25 for featured 4★
      (In real Genshin, there's also an "Epitomized Path" system. This code omits that.)
    """
    state["pulls_since_4star"] += 1
    state["pulls_since_5star"] += 1

    # ===== forced 5★ at 80 =====
    if state["pulls_since_5star"] >= 80:
        state["pulls_since_5star"] = 0
        state["pulls_since_4star"] = 0
        return get_5star_weapon(state, forced=True)

    # ===== forced 4★ at 10 (chance to upgrade to 5★ is about 0.7%) =====
    # (In the official text: "probability of winning 5★ item through the 4★ guarantee = 0.7%")
    if state["pulls_since_4star"] >= 10:
        if random_check(0.007):
            # 5★
            state["pulls_since_5star"] = 0
            state["pulls_since_4star"] = 0
            return get_5star_weapon(state, forced=True)
        else:
            # guaranteed 4★
            state["pulls_since_4star"] = 0
            return get_4star_weapon(state, forced=True)

    # ===== normal roll: 0.7% for 5★, 6.0% for 4★, rest 3★ =====
    r = random.random()
    if r < 0.007:
        # 5★
        state["pulls_since_5star"] = 0
        state["pulls_since_4star"] = 0
        return get_5star_weapon(state, forced=False)
    elif r < 0.007 + 0.06:
        # 4★
        state["pulls_since_4star"] = 0
        return get_4star_weapon(state, forced=False)
    else:
        return "3★ (Weapon)"

# -------------------------------------------------------------------------
# "Get 5★" or "Get 4★" helper functions, implementing 50/50 or 75/25
# -------------------------------------------------------------------------

def get_5star_figure(state, forced=False):
    """
    For the figure (character event) banner:
      - If guaranteed_5star_promotional == True, then get the banner character (e.g. Klee).
      - Otherwise 50% chance for promotional, 50% chance off-banner
      - If you lose the 50%, set guaranteed_5star_promotional = True for next time
    """
    if state["guaranteed_5star_promotional"]:
        # Must get promotional
        state["guaranteed_5star_promotional"] = False
        return "5★ Figure [Promotional guaranteed]"
    else:
        # 50/50
        if random_check(0.5):
            return "5★ Figure [Promotional 50%]"
        else:
            # Off-banner
            state["guaranteed_5star_promotional"] = True
            return "5★ Figure [Off-banner, lost 50/50]"

def get_4star_figure(state, forced=False):
    """
    For the figure (character event) banner’s 4★:
      - If guaranteed_4star_promotional == True, then guaranteed featured 4★
      - Otherwise 50% chance for a featured 4★ vs. off-banner
      - If off-banner, then guaranteed_4star_promotional = True next time
    """
    if state["guaranteed_4star_promotional"]:
        state["guaranteed_4star_promotional"] = False
        return "4★ Figure [Featured guaranteed]"
    else:
        # 50% chance for featured
        if random_check(0.5):
            return "4★ Figure [Featured 50%]"
        else:
            # off-banner
            state["guaranteed_4star_promotional"] = True
            return "4★ Figure [Off-banner, lost 50%]"

def get_5star_weapon(state, forced=False):
    """
    For the weapon banner:
      - If guaranteed_5star_promotional == True, then guaranteed promotional weapon
      - Otherwise 75% chance promotional, 25% off-banner
      - If off-banner, guaranteed_5star_promotional = True next time
    """
    if state["guaranteed_5star_promotional"]:
        state["guaranteed_5star_promotional"] = False
        return "5★ Weapon [Promotional guaranteed]"
    else:
        if random_check(0.75):
            return "5★ Weapon [Promotional 75%]"
        else:
            state["guaranteed_5star_promotional"] = True
            return "5★ Weapon [Off-banner, lost 75%]"

def get_4star_weapon(state, forced=False):
    """
    For the weapon banner:
      - If guaranteed_4star_promotional == True, guaranteed featured 4★
      - Otherwise 75% chance featured, 25% off-banner
      - If off-banner, next 4★ is guaranteed featured
    """
    if state["guaranteed_4star_promotional"]:
        state["guaranteed_4star_promotional"] = False
        return "4★ Weapon [Featured guaranteed]"
    else:
        if random_check(0.75):
            return "4★ Weapon [Featured 75%]"
        else:
            state["guaranteed_4star_promotional"] = True
            return "4★ Weapon [Off-banner, lost 75%]"

# -------------------------------------------------------------------------
# Putting it all together: single-pull or 10-pull
# -------------------------------------------------------------------------

def pull_once(decision, pity):
    """
    decision: integer 1..6 indicating which banner + how many pulls
    pity: the user’s overall pity state dictionary

    Return a list of the results (strings).
    """
    results = []
    if decision == 1:
        # single draw, regular
        item = draw_once_regular(pity["regular"])
        results.append(item)
    elif decision == 2:
        # 10 draws, regular
        for _ in range(10):
            item = draw_once_regular(pity["regular"])
            results.append(item)
    elif decision == 3:
        # single draw, figure
        item = draw_once_figure(pity["figure"])
        results.append(item)
    elif decision == 4:
        # 10 draws, figure
        for _ in range(10):
            item = draw_once_figure(pity["figure"])
            results.append(item)
    elif decision == 5:
        # single draw, weapon
        item = draw_once_weapon(pity["weapon"])
        results.append(item)
    elif decision == 6:
        # 10 draws, weapon
        for _ in range(10):
            item = draw_once_weapon(pity["weapon"])
            results.append(item)

    return results

def interpret_decision(whether_draw, which_pool, how_many):
    """
    Map your separate columns into the 0..6 style codes:
      0 = no buy
      1 = single regular
      2 = 10 regular
      3 = single figure
      4 = 10 figure
      5 = single weapon
      6 = 10 weapon
    """
    if whether_draw == 0:
        return 0
    else:
        # Regular = 200, Figure = 301, Weapon = 302
        if which_pool == 200:  # regular
            if how_many == 1:
                return 1
            elif how_many == 10:
                return 2
        elif which_pool == 301:  # figure
            if how_many == 1:
                return 3
            elif how_many == 10:
                return 4
        elif which_pool == 302:  # weapon
            if how_many == 1:
                return 5
            elif how_many == 10:
                return 6
    # If missing or zero, return 0
    return 0

# -------------------------------------------------------------------------
# Example: Running a sequence of decisions
# -------------------------------------------------------------------------

if __name__ == "__main__":
    # Initialize pity states for one user
    pity_state = init_pity_state()

    # Suppose we have some predicted or observed decisions:
    # We'll make up some data: (WhetherDraw, WhichPoolDraw, HowManyDraw)
    # 1) Draw from figure pool (single)
    # 2) Draw from figure pool (ten-pull)
    # 3) No draw
    # 4) Draw from weapon pool (ten-pull)
    example_decisions = [
        (1, 301, 1),
        (1, 301, 10),
        (0, 0, 0),
        (1, 302, 10),
    ]

    # Simulate
    for (wd, wp, hm) in example_decisions:
        dec = interpret_decision(wd, wp, hm)
        if dec == 0:
            print("No pull this time.")
            continue
        results = pull_once(dec, pity_state)
        print(f"Decision {dec}, results:")
        for r in results:
            print("  ", r)

    print("\nFinal pity state:", pity_state)
