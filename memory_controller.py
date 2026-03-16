from scorer import compute_l1_score
from promotion import try_promote
from cluster_manager import ClusterManager
from memory_utils import find_weakest_cluster


class MemoryController:

    def __init__(self, embedder, memory_levels, vector_store):

        self.embedder = embedder
        self.memory = memory_levels
        self.vector_store = vector_store

        self.cluster_manager = ClusterManager(embedder)

        self.l1_threshold = 0.3  # was 0.15

    def process_prompt(self, prompt, current_step=0):  # add current_step
        prompt_emb = self.embedder.embed(prompt)
        l1_score = compute_l1_score(self.memory.l1, prompt_emb)
        print("L1 score:", l1_score)

        if l1_score >= self.l1_threshold:
            return self.memory.l1

        candidate = self.search_l2(prompt_emb)
        if not candidate:
            candidate = self.vector_store.search(prompt_emb)

        if candidate:
            try_promote(candidate, self.memory, prompt_emb, current_step)  # pass step
            while self.memory.l1_token_count() > self.memory.l1_limit_tokens:
                weakest = find_weakest_cluster(self.memory.l1, prompt_emb)
                self.memory.l1.remove(weakest)
                self.memory.add_to_l2(weakest)

        return self.memory.l1

    def search_l2(self, prompt_emb):

        best = None
        best_sim = 0

        for cluster in self.memory.l2:

            sim = cluster.similarity(prompt_emb)

            if sim > best_sim:
                best = cluster
                best_sim = sim

        if best_sim > 0.35:
            return best

        return None

    def store_memory(self, text):
        cluster = self.cluster_manager.add_memory(text)
        # sync cluster_manager clusters into vector_store
        if cluster not in self.vector_store.clusters:
            self.vector_store.add(cluster)