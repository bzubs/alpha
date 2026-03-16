from scorer import compute_l1_score
from memory_utils import find_weakest_cluster
def try_promote(candidate, memory, prompt_embedding, current_step=0):
    if not memory.l1:
        memory.l1.append(candidate)
        return

    if candidate in memory.l1:
        return

    # negative similarity floor
    candidate_sim = candidate.similarity(prompt_embedding)
    if candidate_sim < 0.1:
        return

    # cooldown check
    COOLDOWN_STEPS = 5
    if (current_step - getattr(candidate, 'last_promoted_step', -99)) < COOLDOWN_STEPS:
        return

    if candidate in memory.l2:
        memory.l2.remove(candidate)

    old_score = compute_l1_score(memory.l1, prompt_embedding)
    PROMOTION_MARGIN = 0.15

    weakest = find_weakest_cluster(memory.l1, prompt_embedding)
    simulated = memory.l1.copy()
    simulated.remove(weakest)
    simulated.append(candidate)
    new_score = compute_l1_score(simulated, prompt_embedding)

    if new_score > old_score + PROMOTION_MARGIN:
        if len(memory.l1) < 3:
            memory.l1.append(candidate)
        else:
            memory.l1.remove(weakest)
            memory.add_to_l2(weakest)
            memory.l1.append(candidate)

        candidate.last_promoted_step = current_step  # stamp it
        print("Promoted cluster:", candidate.summary_text)