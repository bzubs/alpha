def find_weakest_cluster(clusters, prompt_embedding):

    weakest = None
    weakest_sim = float("inf")

    for c in clusters:

        sim = c.similarity(prompt_embedding)

        if sim < weakest_sim:
            weakest_sim = sim
            weakest = c

    return weakest

