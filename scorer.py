def compute_l1_score(clusters, prompt_embedding):

    if not clusters:
        return 0.0

    weighted_sim = 0
    total_tokens = 0

    for c in clusters:

        sim = c.similarity(prompt_embedding)

        weighted_sim += sim * c.token_size
        total_tokens += c.token_size

    return weighted_sim / total_tokens