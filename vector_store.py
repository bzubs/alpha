class VectorStore:

    def __init__(self):

        self.clusters = []

    def add(self, cluster):

        self.clusters.append(cluster)

    def search(self, embedding):

        best = None
        best_sim = 0

        for c in self.clusters:

            sim = c.similarity(embedding)

            if sim > best_sim:
                best_sim = sim
                best = c

        if best_sim > 0.4:
            return best

        return None