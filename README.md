# EchoLLM: Evidence-Checked Hierarchical Ontology 🔍

**Embedding Clarity for Hallucination Optimization with Large Language Models**

---

## Overview

EchoLLM addresses a critical limitation of Large Language Models in knowledge graph construction: **factual hallucination**. When LLMs generate knowledge triples, they often produce fluent but incorrect assertions—plausible-sounding facts that have no grounding in the source text. This proves particularly dangerous in scientific and enterprise applications where factual accuracy is non-negotiable.

EchoLLM solves this through an **evidence-grounded verification pipeline** that validates every extracted triple against its source material. Rather than trusting the LLM's output at face value, the system retrieves supporting evidence and applies formal logical verification before accepting any fact into the knowledge graph.

### Key Contributions

- **Statistically Validated Precision Gains**: 11% absolute improvement in precision (0.65 → 0.76) on the CaRB benchmark
- **Dramatic Hallucination Reduction**: 55% decrease in false positives (1229 → 558 triples filtered)
- **No Domain-Specific Rules**: Works across diverse text types without requiring custom ontology definitions
- **Full Auditability**: Every decision is logged with confidence scores and supporting evidence
- **Practical Accessibility**: Generates human-interpretable RDF ontologies ready for deployment

---

## Architecture Overview

EchoLLM implements a five-stage pipeline that transforms raw text into verified knowledge:

```
Raw Text
   ↓
[1] Preprocessing 
   ↓
[2] Triple Extraction (LLM-based)
   ↓
[3] Hybrid Retrieval & NLI Verification
   ↓
[4] Entity Clustering
   ↓
[5] Ontology Construction
   ↓
Validated Knowledge Graph + Hierarchy
```

Each stage is designed for transparency: at any point, you can inspect what the system accepted and why.

---

## Stage 1: Preprocessing — The Minimal Triad 📝

Before triple extraction, raw text undergoes three lightweight but crucial operations:

### 1.1 Normalization
Removes boilerplate (navigation links, timestamps, metadata) and standardizes Unicode characters to ensure consistent input representation.

**Example:**
```
Input:  "Click here to read more! ✓ Urban agriculture..."
Output: "Urban agriculture..."
```

### 1.2 Segmentation
Splits text into single-verb clauses, isolating predicates and reducing ambiguity for the extraction step.

**Example:**
```
Input:  "Urban agriculture provides food and benefits biodiversity"
Output: 
  • Clause 1: "Urban agriculture provides food"
  • Clause 2: "Urban agriculture benefits biodiversity"
```

```

**Example Token Table (Urban Agriculture sentence):**

| Token | POS | Syntactic Role 
|-------|-----|---|--------|
| Urban | ADJ | amod | 
| agriculture | NOUN | nsubj | 
| provides | VERB | root | 
| food | NOUN | obj | 

These weights are passed to the LLM as context, nudging it to focus on informationally dense portions of the text. This lightweight preprocessing balances data quality assurance without over-engineering.

---

## Stage 2: Triple Extraction — LLM Selection 🤖

EchoLLM evaluated four prominent 7-8B parameter language models to identify the best triple extractor:

### Model Evaluation Results

| Model | Precision | Recall | F1 | Key Observations |
|-------|-----------|--------|-----|------------------|
| **Llama3-8B** | 0.47 | 0.44 | **0.45** | **Consistent formatting, high instruction adherence** |
| Mistral-7B | 0.47 | 0.44 | **0.45** | Accurate but unstructured outputs |
| ChatGPT-4o-mini | 0.42 | 0.40 | 0.38 | Occasional summarization instead of extraction |
| DeepSeek-7B | 0.30 | 0.29 | 0.30 | Frequent incomplete triples |

While Mistral-7B matched Llama3-8B numerically (both F1 = 0.45), **Llama3-8B was selected for superior instruction adherence**—a qualitative factor crucial for downstream verification. The model consistently produced clean, structured output without extraneous text, reducing parsing errors and enabling more reliable verification.


```


### Relaxed Matching Criteria

During evaluation, EchoLLM employs **relaxed matching rules** to accommodate minor variations in predicate and object phrasing without penalizing semantically equivalent extractions. This recognizes that different LLMs may phrase identical concepts differently (e.g., "benefits" vs. "enhances").

#### Predicate Similarity Groups

Seven predicate groups capture semantic equivalence:

| Group | Included Predicates |
|-------|---------------------|
| **P1** (Benefit/Improvement) | benefits, enhances, improves, promotes, optimizes, helps |
| **P2** (Usage/Leveraging) | uses, utilizes, leverages, integrates, employs |
| **P3** (Causation/Enablement) | leads to, causes, results in, enables, allows, powers |
| **P4** (Focus/Addressing) | is focused on, focuses on, addresses, targets |
| **P5** (Inclusion/Composition) | includes, involves, has components, comprises, consists of |
| **P6** (Impact/Change) | affects, impacts, influences, reshapes, transforms, changes |
| **P7** (Provision/Offering) | provides, offers, supplies |

#### Object/Subject Concept Groups

Fourteen concept groups handle acronyms, synonyms, and domain-specific terminology:

| Group | Included Terms |
|-------|---------------|
| **C1** (Diversity) | diversity, biodiversity |
| **C2** (Produce) | produce, local produce, fresh local produce, crops |
| **C3** (Retail/Commerce) | retail, stores, e-commerce, commerce, market |
| **C4** (Data/Information) | data, information, knowledge |
| **C5** (Energy) | renewable energy, renewable sources, clean energy |
| **C6** (Challenges/Issues) | challenges, issues, concerns, limitations, drawbacks |
| **C7** (Ethics/Responsibility) | ethics, bias, privacy, security, responsibility, regulation |
| **C8** (Acronyms - General AI) | ai, artificial intelligence, machine learning |
| **C9** (Acronyms - Vehicles) | ev, electric vehicle, electric vehicles, autonomous vehicles, self-driving cars |
| **C10** (Acronyms - Process Automation) | rpa, robotic process automation |
| **C11** (Acronyms - Language) | nlg, natural language generation |
| **C12** (Acronyms - Reality) | ar, augmented reality, vr, virtual reality, immersive experiences |
| **C13** (Computing Types) | quantum computing, edge computing, cloud computing |
| **C14** (Healthcare Context) | healthcare, medical applications, diagnosis, treatment, patient care, drug discovery |

These groups ensure fair evaluation across models with different linguistic tendencies while maintaining semantic integrity.

---

## Stage 3: Hybrid Retrieval & NLI Verification 🔐

This is the critical validation layer that distinguishes EchoLLM from naive LLM-only approaches. For each extracted triple, the system retrieves supporting evidence and applies logical inference verification.

### 3.1 Hybrid Retrieval

Each triple is queried against the source text using two complementary retrieval methods:

**BM25 (Lexical Matching):**
- Excels at keyword matching and exact term overlap
- Captures precise terminology but misses paraphrases
- Computationally efficient for candidate pruning

**all-MiniLM-L6-v2 (Semantic Embeddings):**
- Understands semantic relationships and paraphrasing
- Fails on rare or domain-specific terms
- Computationally heavier but captures conceptual meaning

Results are fused using **Reciprocal Rank Fusion (RRF)** with k=60, which combines ranked lists without requiring training data. This approach is unsupervised, has been validated in specialized domains (medicine, law show 9-20% F1 gains), and requires no domain-specific tuning.

**Query Construction:**
- **For BM25**: Subject + Predicate + Object concatenated, tokenized, lemmatized, stop words removed
- **For Dense**: Subject, Predicate, Object individually encoded, embeddings averaged (ensures predicate's semantic weight influences search direction)

**Result**: Top 3 candidate sentences retrieved per triple

### 3.2 NLI-Based Logical Verification

For each candidate sentence, two verification methods are applied:

#### Method 1: Lexical Consistency Check
Confirms that the triple's subject and object appear in the candidate sentence.
- **If successful**: confidence = 0.95 (high baseline confidence)
- **If unsuccessful**: move to NLI verification

#### Method 2: Natural Language Inference (NLI)
Uses **BART-Large-MNLI** to test whether the sentence logically entails the triple.

**Verification Formula:**

The triple is verbalized as a hypothesis (e.g., "Urban agriculture benefits biodiversity") and the retrieved sentence serves as the premise. The NLI model computes: *Does the premise logically entail this hypothesis?*

- **Entailment Score > 0.7**: Accept triple (confidence = NLI score)
- **Entailment Score ≤ 0.7**: Reject triple

**Context Expansion for Pronouns:**
When pronouns or anaphoric references obscure meaning (e.g., "it provides benefits"), the premise is expanded to include the preceding sentence, providing broader textual context for more accurate NLI assessment.

### 3.3 Verification Algorithm

```
For each triple t:
  Retrieve top 3 candidate sentences C using hybrid search
  
  For each candidate sentence s in C:
    1. Check lexical match (subject & object present)
       If match → confidence = 0.95, mark as verified
    
    2. Convert triple to NLI hypothesis
       Compute entailment score p_nli via BART-Large-MNLI
       If p_nli > 0.7 → confidence = p_nli, mark as verified
    
    3. Select highest confidence match
       Record: verification method, confidence score, supporting sentence
  
  If confidence > threshold:
    Accept triple into validated set
  Else:
    Discard (log as unsupported)
```

**Output**: 
- ✅ Validated triples (with supporting evidence and confidence scores)
- ❌ Rejected triples (logged for analysis)
- 📋 Verification logs (enables debugging and transparency)

---

## Stage 4: Entity Clustering — Inferring Class Hierarchies 🏗️

Once triples are verified, their constituent entities (subjects and objects) are analyzed to discover semantic groupings, forming the basis for the ontology hierarchy.

### 4.1 Entity Embedding

All unique entities from validated triples are encoded into dense vectors using **bert-base-uncased**, capturing semantic meaning in a 768-dimensional space.

**Example**: 
- "anti-carcinogenic properties" → [0.12, -0.45, ..., 0.78]
- "anti-inflammatory properties" → [0.11, -0.44, ..., 0.79]
- (cosine similarity ≈ 0.93 → highly semantically related)

Embeddings are z-score normalized to ensure similarity measures are interpretable and not skewed by extreme values.

### 4.2 Comprehensive Clustering Algorithm Evaluation

EchoLLM evaluated **six clustering algorithms** on a 30-entity flavonoid research corpus to identify the most semantically coherent approach:

#### Algorithm Summary Statistics

| Algorithm | Total Clusters | Entities Clustered | Noise/Outliers | Largest Cluster Size | Exemplars |
|-----------|----------------|-------------------|----------------|---------------------|-----------|
| **HAC** (Hierarchical Agglomerative Clustering) | 21 | 30 | 0 | 1 | N/A |
| **HDBSCAN** (Hierarchical Density-Based) | 1 | 12 | 18 | 12 | N/A |
| **Affinity Propagation** | 4 | 30 | 0 | 9 | 4 |
| **Spectral Clustering** | 3 | 30 | 0 | 11 | N/A |
| **DBSCAN** (Density-Based) | 5 | 17 | 13 | 4 | N/A |
| **DPC** (Density Peaks Clustering) | 24 | 26 | 4 | 1 | N/A |

**Key Observations:**
- **HAC**: Over-fragmented (21 clusters, most with single entities)
- **HDBSCAN**: Marked 60% of entities as noise (unacceptable semantic loss)
- **DBSCAN**: Marked 43% as noise; remaining clusters too small
- **DPC**: Severe over-fragmentation (24 clusters for 26 entities)
- **Affinity Propagation & Spectral Clustering**: Both achieved full coverage with reasonable cluster counts

#### Detailed Clustering Results (Selected Entities)

Below is a comparative view showing how each algorithm clustered semantically related terms:

| Entity | HAC | HDBSCAN | Affinity Prop. | Spectral | DBSCAN | DPC |
|--------|-----|---------|----------------|----------|--------|-----|
| **Flavonoids** | 14 | NOISE | 0 | 0 | 0 | NOISE |
| **Research on Flavonoids** | 14 | NOISE | 0 | 0 | 0 | 0 |
| **Characterization of flavonoids** | 14 | NOISE | 0 | 0 | 0 | 7 |
| **Identification of flavonoids** | 14 | NOISE | 0 | 0 | 0 | 11 |
| **Isolation of flavonoids** | 15 | NOISE | 0 | 0 | 0 | 16 |
| **Studying functions of flavonoids** | 14 | NOISE | 0 | 0 | 0 | 20 |
| | | | | | | |
| **Anti-carcinogenic properties** | 18 | NOISE | 1 | 3 | NOISE | 1 |
| **Anti-inflammatory properties** | 19 | NOISE | 1 | 3 | NOISE | 2 |
| **Anti-mutagenic properties** | 16 | NOISE | 1 | 3 | NOISE | 3 |
| **Anti-oxidative properties** | 17 | NOISE | 1 | 3 | NOISE | 4 |
| | | | | | | |
| **Bark** | 9 | NOISE | 3 | 1 | NOISE | 5 |
| **Flowers** | 3 | NOISE | 3 | 1 | NOISE | 8 |
| **Fruits** | 5 | NOISE | 3 | 1 | NOISE | 9 |
| **Grains** | 7 | NOISE | 3 | 1 | NOISE | 10 |
| **Roots** | 8 | NOISE | 3 | 1 | NOISE | 18 |
| **Stems** | 4 | NOISE | 3 | 1 | NOISE | 19 |
| **Tea** | 1 | NOISE | 3 | 1 | NOISE | 21 |
| **Vegetables** | 6 | NOISE | 3 | 1 | NOISE | 22 |
| **Wine** | 2 | NOISE | 3 | 1 | NOISE | 23 |
| | | | | | | |
| **Indispensable component in cosmetic applications** | 12 | NOISE | 4 | 4 | NOISE | 12 |
| **Indispensable component in medicinal applications** | 10 | NOISE | 4 | 4 | 1 | 13 |
| **Indispensable component in nutraceutical applications** | 11 | NOISE | 4 | 4 | NOISE | 14 |
| **Indispensable component in pharmaceutical applications** | 10 | NOISE | 4 | 4 | 1 | 15 |
| | | | | | | |
| **Capacity to modulate key cellular enzyme function** | 20 | NOISE | 2 | 2 | NOISE | 6 |
| **Potential drugs in preventing chronic diseases** | 13 | NOISE | 4 | 0 | NOISE | 17 |

**Semantic Analysis:**
- **Flavonoid-related terms** (rows 1-6): Only Affinity Propagation and Spectral Clustering unified these conceptually related entities into single clusters (both assigned Cluster 0). HAC fragmented them into 14 & 15; HDBSCAN/DBSCAN/DPC marked as noise or over-split.
  
- **"Anti-..." properties** (rows 8-11): Affinity Propagation (Cluster 1) and Spectral Clustering (Cluster 3) both correctly grouped these four semantically identical terms. HAC assigned each to different clusters (16-19).

- **Natural sources** (rows 13-21): Both AP (Cluster 3) and SC (Cluster 1) unified plant parts and food/beverage sources. HAC fragmented into 9 separate clusters.

- **Application domains** (rows 23-26): AP (Cluster 4) and SC (Cluster 4) correctly grouped "indispensable component in..." terms despite different application contexts.

### 4.3 Internal Validation Metrics

To select between Affinity Propagation and Spectral Clustering, three complementary metrics were evaluated:

#### Silhouette Score (higher is better: -1 to +1)
```
s(i) = [b(i) − a(i)] / max{a(i), b(i)}
```
where:
- a(i) = average distance within cluster
- b(i) = distance to nearest other cluster

Balances internal cohesion and separation without assumptions about cluster shape.

#### Davies-Bouldin Index (lower is better)
Quantifies average similarity between each cluster and its closest neighbor. Penalizes overlap and cluster imbalance. Assumes centroid-representable clusters (valid for semantic embedding spaces).

#### Calinski-Harabasz Score (higher is better)
```
CH = [trace(B_k) / (k−1)] / [trace(W_k) / (N−k)]
```
where:
- B_k = between-cluster dispersion matrix
- W_k = within-cluster dispersion matrix
- N = total entities, k = number of clusters

Normalized by cluster count to prevent bias toward fragmentation. Computationally efficient.

### 4.4 Final Algorithm Comparison

| Metric | Affinity Propagation | Spectral Clustering (k=3) |
|--------|---------------------|--------------------------|
| Silhouette Score ↑ | 0.41 | **0.56** ✅ |
| Davies-Bouldin ↓ | 1.02 | **0.71** ✅ |
| Calinski-Harabasz ↑ | 214 | **297** ✅ |

**Spectral Clustering** outperformed across all three metrics, indicating:
- **Tighter clusters** (higher Silhouette)
- **Better separation** (lower Davies-Bouldin)
- **Stronger inter-cluster variance** (higher Calinski-Harabasz)

The unanimous metric agreement provides robust confidence in the selection. The joint evaluation prevents overfitting to any single metric and ensures fair comparison between AP (auto k=4) and SC (optimized k=3).

**Selected Algorithm**: **Spectral Clustering with k=3**

---

## Stage 5: Ontology Construction 🌳

Validated triples and clustered entities are synthesized into a structured RDF/OWL ontology.

### 5.1 Class Hierarchy Generation

For each cluster:
1. Compute mean semantic similarity of each entity to all other cluster members
2. Select entity with highest mean similarity → designate as `owl:Class` (cluster representative)
3. Remaining cluster members become `rdfs:subClassOf` this parent class

**Example Output:**
```turtle
# Cluster 0: Flavonoid Research Activities
Class: FlavonoidResearch
  SubClass: Characterization of flavonoids
  SubClass: Identification of flavonoids
  SubClass: Isolation of flavonoids
  SubClass: Studying functions of flavonoids

# Cluster 3: Antioxidant Properties
Class: AntioxidantProperties
  SubClass: Anti-carcinogenic properties
  SubClass: Anti-inflammatory properties
  SubClass: Anti-mutagenic properties
  SubClass: Anti-oxidative properties

# Cluster 1: Natural Sources
Class: NaturalSources
  SubClass: Bark
  SubClass: Flowers
  SubClass: Fruits
  SubClass: Grains
  SubClass: Roots
  SubClass: Stems
  SubClass: Tea
  SubClass: Vegetables
  SubClass: Wine
```

### 5.2 Semantic Annotations

Each class receives:
- **rdfs:label**: The entity string normalized to CamelCase (e.g., "AntioxidantProperties")
- **rdfs:comment**: A contextual description generated from supporting sentences retrieved during verification

The rdfs:comment generation is **human-in-the-loop**: machine-generated summaries (via Llama API using entity + originating paragraph) are presented for review and optionally refined before inclusion. This ensures annotation quality while maintaining efficiency.

**Example Annotation:**
```turtle
Class: AntioxidantProperties
  rdfs:label "Antioxidant Properties"
  rdfs:comment "Flavonoids exhibit a broad spectrum of biological activities including 
                anti-carcinogenic, anti-inflammatory, anti-mutagenic, and anti-oxidative 
                properties, serving as potential therapeutic agents in chronic disease 
                prevention."
```

Domain experts in our evaluation accepted **100% of auto-generated rdfs:comment entries** without modification, validating the quality of machine-generated annotations.

---

## Evaluation Results 📊

EchoLLM was evaluated on the **CaRB (Comprehensive Assessment of Relation Extraction Benchmark)**, a large-scale open information extraction dataset comprising diverse text domains.

### Quantitative Performance

| Metric | Direct LLM Baseline | EchoLLM | Absolute Change | Relative Change |
|--------|---------------------|---------|-----------------|-----------------|
| **Precision** | 0.65 | **0.76** | **+0.11** | **+17%** ✅ |
| **Recall** | 0.74 | 0.57 | -0.17 | -23% |
| **F1-Score** | 0.69 | 0.65 | -0.04 | -6% |
| **False Positives** | 1229 | **558** | **-671** | **-55%** ✅ |
| **True Positives** | 2232 | 1722 | -510 | -23% |

### Statistical Significance

A **McNemar's test** confirmed the precision improvement is not due to random chance:
```
χ²(1) = 319.07, p < 0.001 (highly statistically significant)
```

This test evaluates whether the proportion of discordant cases (triples where one method succeeds and the other fails) differs significantly between methods.

### Contingency Table Analysis

The McNemar's breakdown reveals the precision-recall trade-off mechanism:

| | **LLM: Correct** | **LLM: Incorrect** |
|---|---|---|
| **EchoLLM: Accepted** | 1571 (both correct) | 151 (EchoLLM recovers) |
| **EchoLLM: Rejected** | 661 (EchoLLM filters correct) | 626 (both incorrect) |

**Key Insights:**
- **Cell (1,1)**: 1571 triples correctly identified by both methods (consensus)
- **Cell (1,2)**: 151 correct triples **recovered by EchoLLM** that the baseline missed (demonstrates selective high-confidence extraction beyond baseline recall)
- **Cell (2,1)**: 661 correct triples **over-filtered by EchoLLM** (source of recall reduction)
- **Cell (2,2)**: 626 incorrect triples rejected by both (confirms baseline had substantial false positives)

The statistical significance (p < 0.001) demonstrates that despite filtering 661 correct triples, the **simultaneous recovery of 151 missed triples plus elimination of 671 false positives** represents a beneficial trade-off for accuracy-critical applications.

### Trade-off Interpretation

The precision-recall trade-off reflects EchoLLM's **design philosophy**: prioritize factual correctness over coverage. 

**Why this matters:**
- In scientific publishing, a single hallucinated fact undermines trust in the entire knowledge graph
- In enterprise applications (legal, medical, financial), false positives create liability
- In automated decision-making, precision directly impacts downstream system reliability

The 11% precision gain with 55% false positive reduction justifies the recall reduction for applications where **factual accuracy is paramount**.

### Qualitative Results

Domain experts reviewed EchoLLM-generated ontologies and confirmed:
- ✅ **Tighter class hierarchies**: No spurious groupings; semantically coherent clusters
- ✅ **Coherent entity relationships**: Triples reflect genuine source text assertions
- ✅ **Annotation quality**: All auto-generated rdfs:comment entries accepted without modification
- ✅ **Interpretability**: Ontology structure aligned with domain expert expectations

**Example Comparison** (Urban Agriculture domain):

| Aspect | Direct LLM | EchoLLM |
|--------|-----------|---------|
| Class structure | Mixed unrelated concepts | Clear semantic groupings |
| Triple accuracy | 65% correct | 76% correct |
| False relationships | 35% hallucinated | 24% hallucinated |
| Hierarchy depth | Shallow, over-generalized | Appropriate specificity |

---

## Runtime and Scalability ⚡

Processing 35 abstracts (~2,500 words) requires approximately **11 minutes** on standard hardware (MacBook M3 Pro, CPU-only NLI inference):

| Stage | Approximate Time | Percentage of Total |
|-------|------------------|---------------------|
| Preprocessing | ~30 sec | 5% |
| Triple extraction | ~2 min | 18% |
| **Retrieval & verification** | **~7 min** | **64%** |
| Entity clustering | ~1 min | 9% |
| Ontology construction | ~30 sec | 5% |

### Computational Bottleneck

The **NLI verification step** dominates runtime due to:
1. BART-Large-MNLI requires forward passes for each (triple, candidate sentence) pair
2. Context expansion adds additional NLI evaluations for pronouns
3. Sequential verification (each triple processed individually)

**Scalability Improvements:**
- **GPU acceleration**: Reduces NLI inference time by 5-10x
- **Batch verification**: Group multiple triples for parallel NLI evaluation
- **Candidate pruning**: Stricter retrieval thresholds reduce verification load
- **Smaller NLI models**: Trade-off between speed and verification accuracy

For production deployments on larger corpora (thousands of documents), GPU-accelerated batch processing is recommended.

---

## Design Principles 💡

### 1. **Evidence-First Validation**
Every triple must prove its support in the source text. There are no shortcuts to plausibility. The system never "trusts" LLM output—it demands textual grounding.

### 2. **Transparency Over Complexity**
The pipeline surfaces failure modes (extraction failures, retrieval gaps, verification rejections) rather than hiding them. Every decision includes:
- Confidence score
- Verification method used (lexical vs. NLI)
- Supporting sentence
- Accept/reject rationale

This enables human review and iterative refinement.

### 3. **No Domain-Specific Rules**
The system requires no ontology templates, entity type definitions, or relation schemas. It works across diverse text types:
- Academic abstracts
- Technical reports
- News articles
- Legal documents
- Medical literature

No reconfiguration needed—same pipeline, same hyperparameters.

### 4. **Flexible Model Choices**
While EchoLLM defaults to:
- **LLM**: Llama3-8B
- **NLI Model**: BART-Large-MNLI
- **Clustering**: Spectral Clustering (k=3)
- **Embeddings**: bert-base-uncased, all-MiniLM-L6-v2

...these components are **modular and swappable**. Alternative LLMs, NLI models, clustering algorithms, and embedding models can be substituted based on domain requirements, computational constraints, or emerging model capabilities.

### 5. **Auditability and Reproducibility**
Every decision is logged with:
- Input text
- Extracted triples
- Retrieval results
- Verification scores
- Final acceptance status

This enables:
- **Debugging**: Trace why specific triples were accepted/rejected
- **Compliance**: Demonstrate factual grounding for regulated industries
- **Research**: Analyze failure modes and improve components
- **Trust**: Users can inspect supporting evidence for any fact in the knowledge graph

---

## Comparison with Naive LLM-Only Approaches

### Baseline: Direct LLM Extraction
```
Raw Text → LLM (extract triples) → Knowledge Graph
```

**Problems:**
- No verification of triple accuracy → 35% false positives
- Hallucinated triples accepted at face value
- No transparency into which facts are supported
- Fluent but factually incorrect relationships pass through unchecked

**Example Hallucination:**
```
Input: "Urban agriculture provides fresh produce."
LLM Output: [Urban agriculture, increases, property values]  ❌ (not in source)
```

### EchoLLM: Evidence-Grounded Pipeline
```
Raw Text → LLM (extract) → Retrieval (find evidence) → NLI (verify logic) → 
Clustering (organize) → Knowledge Graph
```

**Advantages:**
- Every triple verified against source text → 76% precision
- 55% fewer false positives
- Full auditability (confidence scores + supporting sentences)
- Hierarchical organization of concepts
- Semantic clustering reveals implicit ontology structure

**Same Example:**
```
Input: "Urban agriculture provides fresh produce."
LLM Output: [Urban agriculture, increases, property values]
Retrieval: No supporting sentence found
NLI: Cannot verify entailment
Result: REJECTED (logged as unsupported) ✅
```

---

## Limitations & Future Work 🔮

### Known Limitations

#### 1. Precision-Recall Trade-off
The focus on precision necessarily reduces recall. The NLI entailment threshold (0.7) can be lowered to recover more triples, but this reintroduces hallucinations. The current threshold balances these competing objectives based on CaRB benchmark analysis.

**Mitigation**: Domain-specific threshold tuning based on application tolerance for false positives vs. false negatives.

#### 2. Domain Shift in NLI
BART-Large-MNLI is trained on general English (MNLI corpus). Performance may degrade on highly specialized domains:
- Medical terminology (e.g., drug-gene interactions)
- Legal jargon (e.g., statutory interpretations)
- Scientific notation (e.g., chemical formulas)
- Proprietary ontologies (e.g., enterprise-specific terminology)

**Mitigation**: Fine-tune NLI models on domain-specific corpora or use domain-adapted NLI models (e.g., BioBERT for biomedical text).

#### 3. Predicate Complexity
The system assumes single-word or simple multi-word predicates. Complex relational structures require extensions:
- **N-ary relations**: (Drug, treats, Disease, with dosage, 50mg)
- **Temporal constraints**: (Company, acquired, Startup, in year, 2023)
- **Modal qualifications**: (Model, may predict, Outcome)

**Mitigation**: Extend triple schema to RDF* or property graphs with edge attributes.

#### 4. Pronoun Resolution
While context expansion partially addresses anaphora, limitations remain:
- **Multi-sentence dependencies**: "The company launched a product. It succeeded. This boosted revenue." (ambiguous "this")
- **Distant pronouns**: Pronouns referring to entities 3+ sentences away
- **Complex coreference**: Multiple entities with same pronoun (e.g., "he" referring to multiple people)

**Mitigation**: Integrate dedicated coreference resolution models (e.g., SpanBERT-based coreference).

#### 5. Computational Bottleneck
NLI verification scales linearly with number of triples × candidate sentences. For large corpora (10,000+ documents), this becomes prohibitive without GPU acceleration or parallelization.

**Mitigation**: Batch processing, GPU acceleration, or approximate retrieval methods (FAISS).

### Future Directions

#### Short-Term Enhancements
- **Domain Adaptation**: Fine-tune NLI on biomedical, legal, financial corpora
- **GPU Optimization**: Implement batch NLI verification for 5-10x speedup
- **Threshold Tuning Interface**: Allow users to adjust precision-recall balance interactively
- **Coreference Integration**: Add SpanBERT or Longformer-based pronoun resolution

#### Medium-Term Extensions
- **Multi-Hop Reasoning**: Chain triples for transitive inference (A→B, B→C ⇒ A→C)
- **Temporal Reasoning**: Add support for time-dependent facts and event sequences
- **Uncertainty Quantification**: Express confidence intervals for borderline triples
- **Incremental Updates**: Support knowledge graph updates without full reprocessing

#### Long-Term Vision
- **Interactive Refinement**: User feedback loops to refine clustering and class hierarchies
- **Multi-Lingual Support**: Extend to non-English texts via multilingual models
- **Hybrid Human-AI Workflows**: Active learning for efficient expert annotation
- **Knowledge Graph Completion**: Predict missing triples based on graph structure

---

## Installation & Usage

### Prerequisites
```
Python 3.9+
PyTorch 2.0+
transformers (Hugging Face)
scikit-learn
stanza
sentence-transformers
rank_bm25
rdflib (for RDF/OWL generation)
```

### Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/echollm.git
cd echollm

# Install dependencies
pip install -r requirements.txt

# Download Stanza models
python -c "import stanza; stanza.download('en')"

# Run sample workflow
python examples/sample_workflow.py --input sample_text.txt --output ontology.owl
```

### Configuration Options

Create a `config.yaml` file to customize pipeline behavior:

```yaml
# Model Selection
llm_model: "meta-llama/Llama-3-8B"
nli_model: "facebook/bart-large-mnli"
embedding_model: "bert-base-uncased"
retrieval_model: "sentence-transformers/all-MiniLM-L6-v2"

# Hyperparameters
temperature: 0.5
max_tokens: 500
nli_threshold: 0.7
lexical_confidence: 0.95
top_k_retrieval: 3
rrf_k: 60

# Clustering
clustering_algorithm: "spectral"  # Options: spectral, affinity_propagation
num_clusters: 3  # For spectral clustering; auto-determined for affinity propagation
validation_metrics: ["silhouette", "davies_bouldin", "calinski_harabasz"]

# Output
output_format: "owl"  # Options: owl, rdf, jsonld
include_annotations: true
human_in_loop: true  # Review rdfs:comment before finalizing
```

### Example Usage

```python
from echollm import EchoLLMPipeline

# Initialize pipeline
pipeline = EchoLLMPipeline(config_path="config.yaml")

# Process text
text = """
Flavonoids are natural substances with variable phenolic structures 
found in fruits, vegetables, grains, bark, roots, stems, flowers, 
tea and wine. They exhibit anti-carcinogenic, anti-inflammatory, 
anti-mutagenic, and anti-oxidative properties.
"""

# Extract verified knowledge graph
results = pipeline.process(text)

# Access results
print(f"Extracted {len(results.triples)} verified triples")
print(f"Identified {len(results.clusters)} semantic clusters")
print(f"Average confidence: {results.avg_confidence:.2f}")

# Export ontology
results.export("flavonoid_ontology.owl", format="owl")

# Inspect verification logs
for triple, log in results.verification_logs.items():
    print(f"{triple}")
    print(f"  Method: {log.method}")
    print(f"  Confidence: {log.confidence}")
    print(f"  Supporting: {log.supporting_sentence}")
```

---



## Citation [IN PRESS]

If you use EchoLLM in your research, please cite:

```bibtex
@inproceedings{echollm2025,
  title={EchoLLM: Evidence-Checked Hierarchical Ontology},
  author={Dalal, Aryan Singh and McGinty, Hande},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025},
  publisher={AAAI Press}
}
```

---

## Acknowledgments

This work was developed at Kansas State University's Department of Computer Science. We thank:
- Domain experts who reviewed ontology outputs and provided constructive feedback
- The open-source community for foundational tools (Stanza, Hugging Face Transformers, scikit-learn)
- CaRB benchmark maintainers for providing standardized evaluation datasets

---

## License

MIT License - see LICENSE file for details

---

## Contact

For questions, issues, or collaboration inquiries:

**Aryan Singh Dalal**  
Department of Computer Science  
Kansas State University  
📧 aryan.dalal@ksu.edu

**Hande McGinty**  
Department of Computer Science  
Kansas State University  
📧 hande@ksu.edu

---

## Additional Resources

- 📄 **Full Paper**: [Link to AAAI published version]
- 📊 **CaRB Benchmark**: [https://github.com/dair-iitd/CaRB](https://github.com/dair-iitd/CaRB)
- 📖 **Extended Technical Report**: [Link to detailed methodology documentation]
- 🔗 **Related Work**: See `docs/related_work.md` for positioning within RAG and ontology engineering literature
- 🎥 **Video Tutorial**: [Link to walkthrough demonstration]
- 💬 **Discussion Forum**: [Link to community discussion]

---

**Last Updated**: November 2025  
**Version**: 1.0.0
