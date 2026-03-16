from cluster import Cluster


class ClusterManager:

    def __init__(self, embedder, threshold=0.6):

        self.embedder = embedder
        self.threshold = threshold
        self.clusters = []

    def add_memory(self, text):

        emb = self.embedder.embed(text)

        best_cluster = None
        best_sim = -1

        for cluster in self.clusters:

            sim = cluster.similarity(emb)

            if sim > best_sim:
                best_sim = sim
                best_cluster = cluster

        if best_sim > self.threshold:

            best_cluster.add_memory(text, emb)
            return best_cluster

        else:

            cluster = Cluster(
                embedding=emb,
                text=text,
                token_size=len(text.split())
            )

            self.clusters.append(cluster)

            return cluster