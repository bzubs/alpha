
class MemoryLevels:

    def __init__(self, l1_limit_tokens=1000, l2_limit=50):

        self.l1 = []
        self.l2 = []

        self.l1_limit_tokens = l1_limit_tokens
        self.l2_limit = l2_limit

    def l1_token_count(self):

        return sum(c.token_size for c in self.l1)

    def add_to_l2(self, cluster):

        self.l2.insert(0, cluster)

        if len(self.l2) > self.l2_limit:
            self.l2.pop()