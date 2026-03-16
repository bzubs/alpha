import numpy as np


class Cluster:

    def __init__(self, embedding, text, token_size):

        self.centroid_embedding = embedding
        self.summary_text = text
        self.token_size = token_size

        self.member_memories = [text]
        self.n_members = 1

        # In Cluster.__init__
        self.promotion_count = 0
        self.last_promoted_step = -1

    def similarity(self, embedding):
        return float(np.dot(self.centroid_embedding, embedding) /
             (np.linalg.norm(self.centroid_embedding) * np.linalg.norm(embedding)))

    def add_memory(self, text, embedding):
        if text in self.member_memories:
            return  # deduplicate exact matches
        
        self.member_memories.append(text)
        self.summary_text = " | ".join(self.member_memories[-3:])  # separator makes it readable

        n = self.n_members
        new_centroid = (self.centroid_embedding * n + embedding) / (n + 1)
        self.centroid_embedding = new_centroid / np.linalg.norm(new_centroid)
        self.n_members += 1
        self.token_size = len(self.summary_text.split())