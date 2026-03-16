import random

from embedder import Embedder
from memory_lev import MemoryLevels
from vector_store import VectorStore
from memory_controller import MemoryController


class ConversationSimulator:

    def __init__(self):

        self.topics = {
            "transformers": [
                "Explain transformer architecture",
                "What is self attention",
                "How positional encoding works",
                "Difference between BERT and GPT"
            ],

            "kubernetes": [
                "Explain kubernetes pods",
                "What is service mesh",
                "How load balancing works in k8s",
                "What is kube proxy"
            ],

            "startups": [
                "How to raise seed funding",
                "What is product market fit",
                "How to build MVP",
                "Startup growth strategies"
            ]
        }

    def generate(self, length=50):

        convo = []

        topic = random.choice(list(self.topics.keys()))

        for _ in range(length):

            if random.random() < 0.2:
                topic = random.choice(list(self.topics.keys()))

            text = random.choice(self.topics[topic])

            convo.append({
                "text": text,
                "topic": topic
            })

        return convo


class Metrics:

    def __init__(self):

        self.total = 0
        self.correct = 0
        self.total_tokens = 0

    def update(self, correct, tokens):

        self.total += 1

        if correct:
            self.correct += 1

        self.total_tokens += tokens

    def report(self):

        return {
            "retrieval_accuracy": self.correct / self.total,
            "avg_tokens": self.total_tokens / self.total
        }


class ExperimentRunner:

    def __init__(self):

        embedder = Embedder()
        memory_levels = MemoryLevels()
        vector_store = VectorStore()

        self.controller = MemoryController(
            embedder,
            memory_levels,
            vector_store
        )

        self.simulator = ConversationSimulator()
        self.metrics = Metrics()

    def run(self, length=100):

        conversation = self.simulator.generate(length)

        for step, turn in enumerate(conversation):

            text = turn["text"]

            # Process prompt
            l1_clusters = self.controller.process_prompt(text, step)

            # Get embedding of prompt
            prompt_emb = self.controller.embedder.embed(text)

            # ----- Retrieval accuracy check -----
            correct = False

            for c in l1_clusters:

                sim = c.similarity(prompt_emb)

                if sim > 0.7:
                    correct = True
                    break

            # ----- Token usage -----
            tokens = sum(c.token_size for c in l1_clusters)

            self.metrics.update(correct, tokens)

            # ----- Store new memory -----
            self.controller.store_memory(text)

        return self.metrics.report()