# MemTrans Architecture Deep Dive

## Table of Contents
1. [Memory Hierarchy](#memory-hierarchy)
2. [Cluster Representation](#cluster-representation)
3. [Promotion Logic](#promotion-logic)
4. [Eviction Logic](#eviction-logic)
5. [Scoring Mechanism](#scoring-mechanism)
6. [Caching Strategy](#caching-strategy)
7. [Full Step-by-Step Example](#full-step-by-step-example)

---

## Memory Hierarchy

The system maintains memory across three levels, optimized for both relevance and efficiency:

### **L1: Active Memory (Hot Set)**
- **Purpose:** Clusters currently injected into LLM context
- **Constraints:**
  - Max 4 clusters (configurable `l1_max_clusters`)
  - Max 1000 tokens total (configurable `l1_limit_tokens`)
  - Strictly maintained via token budget + cluster count caps
- **Scoring:** Weighted by similarity to current prompt (recency-adjusted)
- **Use Case:** Real-time context for immediate relevance

### **L2: Warm Memory (Recent Cache)**
- **Purpose:** Recently used clusters no longer in active context
- **Constraints:**
  - Max 50 clusters (configurable `l2_limit`)
  - FIFO with oldest removed when limit exceeded
  - Prioritizes recently-promoted clusters
- **Retrieval:** Ranked by recency decay (half-life model)
- **Use Case:** Quick recovery of recently-relevant context

### **Vector Store: Long-Term Memory (Archive)**
- **Purpose:** Full persistent semantic archive
- **Implementation:** FAISS IndexFlatIP (cosine similarity search)
- **Retrieval:** Top-k by raw embedding similarity (no decay)
- **Use Case:** Global semantic search across all history

```
Prompt comes in
    ↓
Check L1 (fast, in-memory)
    ↓ (not good enough)
Retrieve from L2 (recent, decayed)
    ↓ (empty/weak)
Search Vector Store (global, no decay)
    ↓
Promote best candidate to L1
    ↓
(Evict weaker L1 members if needed)
    ↓
Return L1 for context injection
```

---

## Cluster Representation

### **Cluster Class**

```python
class Cluster:
    centroid_embedding      # (128,) normalized vector (L2 norm = 1)
    summary_text            # Latest 3 memories joined with " | "
    token_size              # Cached token count
    member_memories         # Full list of grouped memories
    n_members               # Count for incremental centroid updates
    promotion_count         # Tracks promotions to L1
    last_promoted_step      # Timestamp for recency decay
```

### **Cluster Operations**

#### Add Memory
When new memory added to existing cluster:
1. Append to `member_memories`
2. Update `summary_text` (keep last 3 for readability)
3. **Incremental centroid update:**
   ```
   new_centroid = (old_centroid * n + new_embedding) / (n + 1)
   new_centroid = normalize(new_centroid)
   ```
   - O(1) per addition (avoid recomputing from scratch)
4. Update `token_size` based on word count

#### Similarity
```
sim(cluster, prompt) = dot_product(centroid, prompt_embedding)
```
- Both normalized to unit vectors (cosine similarity)
- Cached per step to avoid O(n²) recalculation

#### Cluster Creation
New cluster formed when incoming memory's similarity to all existing clusters < threshold (0.6)

---

## Promotion Logic

**Goal:** Move a cluster from L2/Vector Store → L1 if it improves overall prompt coverage

### **Decision Tree**

```
Is L1 empty?
├─ YES → Append candidate immediately
└─ NO → Check constraints:

    Is candidate in L1?
    ├─ YES → Skip (already active)
    └─ NO → Continue:

        Candidate relevance < 0.2?
        ├─ YES → Reject (too weak)
        └─ NO → Continue:

            Candidate too similar to any L1 member (>0.8)?
            ├─ YES → Reject (diversity constraint)
            └─ NO → Continue:

                Cooldown check: promoted < 5 steps ago?
                ├─ YES → Reject (cooldown active)
                └─ NO → Continue:

                    Score improvement check:
                    ├─ new_score > old_score + 0.15 → PROMOTE
                    └─ NO → REJECT
```

### **Key Constraints**

#### 1. **Relevance Floor**
```
CANDIDATE_MIN_SIM = 0.2
if candidate.similarity(prompt) < 0.2:
    return False  # Too weak to be useful
```
Prevents irrelevant clusters from polluting L1

#### 2. **Diversity Constraint**
```
DIVERSITY_SIM_THRESH = 0.8
for existing in L1:
    if existing.similarity(prompt) > 0.8:
        return False  # Already redundant
```
Prevents near-duplicate clusters wasting token budget

#### 3. **Cooldown**
```
COOLDOWN_STEPS = 5
if (current_step - last_promoted_step) < 5:
    return False  # Recently active, let other clusters get priority
```
Ensures fair rotation of clusters; prevents thrashing

#### 4. **Margin Requirement**
```
PROMOTION_MARGIN = 0.15
if new_score > old_score + 0.15:
    return True  # Must be meaningful improvement
```
Avoids marginal promotions that add noise

### **Promotion Strategy: Replace vs Append**

#### Append (if L1 not full)
```
new_score = (current_weighted + candidate_sim * candidate.token_size) 
          / (current_total_tokens + candidate.token_size)
```

#### Replace (if L1 full)
```
Remove: weakest cluster by similarity
new_score = (current_weighted - weakest_sim * weakest.token_size 
           + candidate_sim * candidate.token_size) 
          / (current_total_tokens - weakest.token_size + candidate.token_size)
```

The weakest cluster is identified as:
```
weakest = argmin over L1 of cluster.similarity(prompt)
```

### **Incremental Scoring**

Instead of recalculating all similarities every step:

```python
# Cache from previous promotions
l1_sims = {cluster: weighted_sim for cluster in L1}

# On new candidate:
new_weighted = current_weighted - weakest_sim * weakest.token_size \
             + candidate_sim * candidate.token_size
new_total = current_total_tokens - weakest.token_size + candidate.token_size
new_score = new_weighted / new_total

# O(1) instead of O(|L1|²)
```

### **Recency Weighting**

To prefer recent clusters over stale ones:

```
RECENCY_LAMBDA = 0.05  # decay rate
age = current_step - cluster.last_promoted_step
recency_weight = exp(-RECENCY_LAMBDA * age)
weighted_sim = raw_sim * recency_weight
```

Example:
- Age 0: weight = 1.0 (full strength)
- Age 10: weight = 0.61
- Age 20: weight = 0.37
- Age 100: weight = 0.006 (essentially 0)

---

## Eviction Logic

Two concurrent mechanisms keep L1 within constraints:

### **1. Token Budget Enforcement**

After each promotion, if `l1_token_count() > l1_limit_tokens`:

```python
while l1_token_count() > l1_limit_tokens:
    weakest = argmin over L1 of cluster.similarity(prompt)
    L1.remove(weakest)
    L2.add(weakest)
    # Update incremental scores
```

- Greedy: removes weakest until budget satisfied
- **O(n log n)** in worst case, but rarely triggers multiple times per step

### **2. Cluster Count Cap**

After processing all candidates:

```python
while len(L1) > max_clusters (4):
    victim = find_weakest_cluster(L1, prompt)
    L1.remove(victim)
    L2.add(victim)
```

- Hard limit of 4 clusters regardless of tokens
- Ensures bounded memory footprint

### **When Eviction Happens**

1. **Immediately after promotion** (if tokens exceeded)
2. **After all promotions processed** (cluster count exceeded)
3. **Before returning L1** (final cleanup)

**Eviction Order:**
- Find weakest cluster by current prompt similarity
- Move to L2 (not discarded)
- Update L2 recency (moves to front)
- If L2 full, oldest L2 cluster dropped

---

## Scoring Mechanism

### **L1 Score: Weighted Average Similarity**

```
L1_score = sum(cluster_sim * token_size for cluster in L1) 
         / sum(token_size for cluster in L1)
```

Interpretation: On average, how well does L1 explain the prompt?

### **Evaluation Thresholds**

```python
l1_threshold = 0.3         # Min score to avoid retrieval
L1_FILTER_THRESHOLD = 0.4  # Min sim to include in scoring
CANDIDATE_MIN_SIM = 0.2    # Min sim to consider promoting
```

### **Score Computation**

Before deciding to promote:

```
1. Compute current L1 score (weighted avg of recency-adjusted sims)
2. Simulate adding candidate (or replacing weakest)
3. Compute new L1 score
4. Compare: new - old > 0.15?
```

If L1_score already > 0.3, stop retrieving (query well-covered by current context).

---

## Caching Strategy

### **Similarity Cache: Per-Step**

```python
# At start of step
_SIM_CACHE = {}  # Clear cache

# During retrieval/promotion
def sim_cache_get(cluster, embedding):
    key = (id(cluster), id(embedding))
    if key in _SIM_CACHE:
        return _SIM_CACHE[key]  # O(1) lookup
    
    sim = compute_similarity(cluster, embedding)
    _SIM_CACHE[key] = sim
    return sim
```

**Why:** 
- Same prompt embedding compared to many clusters
- Same clusters compared to same embedding
- Reusing `id()` as key ensures pointer identity (fast hashing)

**Lifetime:** Single step (cleared before next step)

### **LRU Cache: Embeddings**

In `Embedder` class:

```python
@lru_cache(maxsize=10000)
def _embed_impl(text: str) -> np.ndarray:
    return model.encode(text, normalize_embeddings=True)

def embed(self, text):
    return self._cached_encode(text)
```

**Why:**
- Same instructions/prompts embed to same vectors
- Don't recompute embeddings for repeated text
- 10k cache holds ~1M tokens of history

---

## Full Step-by-Step Example

### **Setup**
```
Prompt: "Add JWT authentication to the API"
Current Step: 3
Current L1: [
    Cluster A: "Create Flask app with GET /users" (sim=0.35)
    Cluster B: "Add POST /users endpoint" (sim=0.32)
    Cluster C: "Add email validation" (sim=0.25)
]
L2: [old_cluster_D, old_cluster_E]
Vector Store: [full archive of 50+ clusters]
```

### **Step 1: Score L1**

```
L1_FILTER_THRESHOLD = 0.4

Recency weight for each:
  Cluster A: age 5 → weight = exp(-0.05 * 5) = 0.78
  Cluster B: age 4 → weight = exp(-0.05 * 4) = 0.82
  Cluster C: age 2 → weight = exp(-0.05 * 2) = 0.90

Weighted sims:
  A: 0.35 * 0.78 * 200 tokens = 54.6
  B: 0.32 * 0.82 * 150 tokens = 39.4
  C: 0.25 * 0.90 * 100 tokens = 22.5

L1_score = (54.6 + 39.4 + 22.5) / (200 + 150 + 100) = 116.5 / 450 = 0.26
```

**Result:** 0.26 < 0.3 threshold → Need retrieval

### **Step 2: Retrieve Candidates**

**From L2:**
```
old_cluster_D: raw_sim=0.45, age=10
  → weighted = 0.45 * exp(-0.05*10) = 0.45 * 0.60 = 0.27 < 0.35 threshold
  
old_cluster_E: raw_sim=0.52, age=8
  → weighted = 0.52 * exp(-0.05*8) = 0.52 * 0.67 = 0.35 (marginal)

Candidates from L2: [old_cluster_E] (only above threshold)
```

**If L2 insufficient, search Vector Store:**
```
Top-3 by cosine similarity:
  JWT_cluster: sim=0.58 (summary: "JWT token auth with login")
  Auth_cluster: sim=0.51 (summary: "Basic auth headers")
  OAuth_cluster: sim=0.43 (summary: "OAuth2 integration")
```

**Final Candidates:** [old_cluster_E, JWT_cluster, Auth_cluster, OAuth_cluster]

### **Step 3: Try Promote JWT_cluster**

```
Check constraints:

1. Relevance floor?
   jwt_cluster.sim = 0.58 > 0.2 ✓

2. Already in L1?
   jwt_cluster not in L1 ✓

3. Diversity check (sim > 0.8 with any L1)?
   jwt_cluster vs A: 0.15 < 0.8 ✓
   jwt_cluster vs B: 0.18 < 0.8 ✓
   jwt_cluster vs C: 0.12 < 0.8 ✓

4. Cooldown (last_promoted_step)?
   jwt_cluster.last_promoted_step = -1 (never promoted) ✓

5. Score improvement?
   current_weighted = 116.5, current_total = 450
   old_score = 0.26
   
   L1 has 3/4 slots, so append:
   new_weighted = 116.5 + (0.58 * 0.95 * 300_tokens) = 116.5 + 165.3 = 281.8
   new_total = 450 + 300 = 750
   new_score = 281.8 / 750 = 0.376
   
   improvement = 0.376 - 0.26 = 0.116 < 0.15 ✗ (just barely fails!)
```

**Result:** Rejection (marginal improvement)

### **Step 4: Try Promote Auth_cluster**

```
Check constraints:

1. Relevance floor?
   auth_cluster.sim = 0.51 > 0.2 ✓

2-4. All pass ✓

5. Score improvement?
   new_weighted = 116.5 + (0.51 * 300) = 116.5 + 153 = 269.5
   new_total = 750
   new_score = 269.5 / 750 = 0.359
   
   improvement = 0.359 - 0.26 = 0.099 < 0.15 ✗
```

**Result:** Rejection

### **Step 5: Try Promote Auth_cluster (No L1 Space)**

Hypothetical: After earlier promotions, L1 full:

```
L1 full (4 clusters), need to replace.
Weakest in L1: Cluster C (sim=0.225 after recency)

Score if replacing:
  Remove: -0.225 * 100 = -22.5
  Add: +0.51 * 300 = +153
  new_weighted = 116.5 - 22.5 + 153 = 247
  new_total = 450 - 100 + 300 = 650
  new_score = 247 / 650 = 0.38
  
  improvement = 0.38 - 0.26 = 0.12 < 0.15 ✗
```

**Result:** Still rejected

### **Step 6: Early Exit or Continue**

If current_score after promotions >= 0.3, **stop** (query well-covered).
Otherwise, continue trying remaining candidates.

### **Step 7: Eviction (If Triggered)**

After all promotions, check constraints:

```
L1_token_count = 450 tokens
l1_limit_tokens = 1000

token_count < limit ✓

len(L1) = 3 clusters
l1_max_clusters = 4

count < limit ✓
```

**Result:** No eviction needed

### **Step 8: Return L1**

```
Final L1 passed to agent context:
[
    Cluster A: "Create Flask app with GET /users"
    Cluster B: "Add POST /users endpoint"
    Cluster C: "Add email validation"
]

Formatted as:
"Create Flask app with GET /users | Add POST /users endpoint | Add email validation"
```

---

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| **Embed Text** | O(seq_len) | Cached via LRU if repeated |
| **Add Memory** | O(1) amortized | Incremental centroid update |
| **Compute Similarity** | O(dim) | Usually O(1) with cache |
| **Retrieve from L2** | O(L2 size) | Typically ~50 clusters |
| **Search Vector Store** | O(n log k) | FAISS with k=5 top-k |
| **Promote Candidate** | O(L1 constraints) | Check diversity, cooldown |
| **Evict Cluster** | O(L1 size) | Find weakest → remove |
| **Full Step** | O(n_candidates * L1) | Usually < 100ms |

---

## Configuration Tuning

### For Aggressive Memory (More Context)
```python
l1_limit_tokens = 2000        # Bigger budget
l1_max_clusters = 6           # More clusters
CANDIDATE_MIN_SIM = 0.15      # Lower bar to promote
PROMOTION_MARGIN = 0.10       # Easier to meet
```

### For Conservative Memory (Tight Budget)
```python
l1_limit_tokens = 500         # Tight budget
l1_max_clusters = 2           # Minimal clusters
CANDIDATE_MIN_SIM = 0.3       # Higher bar
PROMOTION_MARGIN = 0.25       # Must be significant
```

### For Fast-Paced Tasks
```python
COOLDOWN_STEPS = 2            # Rotate quickly
RECENCY_LAMBDA = 0.1          # Decay faster
```

### For Stability
```python
COOLDOWN_STEPS = 10           # Stable clusters
RECENCY_LAMBDA = 0.02         # Slow decay
DIVERSITY_SIM_THRESH = 0.7    # Strict diversity
```

