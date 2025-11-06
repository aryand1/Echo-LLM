# EchoLLM: Evidence-Grounded Ontology Construction from Text 🔍

**Embedding Clarity for Hallucination Optimization with Large Language Models**

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

## Stage 1: Preprocessing  📝

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


**Example Token Table (Urban Agriculture sentence):**

| Token | POS | Syntactic Role | 
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

### Extraction Prompt

The LLM receives a structured directive:

```
Convert each numbered sentence into [Subject, Predicate, Object] triples.
Return only triples under a header 'Triples:'
```

This minimalist prompt enforces atomic, sentence-level extraction without complex reasoning. Combined with the preprocessing weights, it steers Llama3-8B toward extracting factual relationships rather than summarizing or inferring cross-sentence connections.

**Example Output:**
```
Input: "Urban agriculture provides food and benefits biodiversity"

Output:
Triples:
[Urban agriculture, provides, food]
[Urban agriculture, benefits, biodiversity]
```

---

## Stage 3: Hybrid Retrieval & NLI Verification 🔐

This is the critical validation layer that distinguishes EchoLLM from naive LLM-only approaches. For each extracted triple, the system retrieves supporting evidence and applies logical inference verification.

### 3.1 Hybrid Retrieval

Each triple is queried against the source text using two complementary retrieval methods:

**BM25 (Lexical Matching):**
- Excel at keyword matching and exact term overlap
- Capture precise terminology but miss paraphrases
- Computationally efficient for candidate pruning

**all-MiniLM-L6-v2 (Semantic Embeddings):**
- Understand semantic relationships and paraphrasing
- Fail on rare or domain-specific terms
- Computationally heavier but capture conceptual meaning

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

### 4.2 Clustering Algorithm Selection

EchoLLM compared two leading clustering algorithms using multiple internal validation metrics:

**Algorithm Comparison on "Anti-..." Properties Test Case:**

| Algorithm | anti-carc | anti-infl | anti-oxid | anti-muta | Result |
|-----------|-----------|-----------|-----------|-----------|--------|
| **Affinity Propagation** | Cluster 1 | Cluster 1 | Cluster 1 | Cluster 1 | ✅ Unified |
| **Spectral Clustering** | Cluster 3 | Cluster 3 | Cluster 3 | Cluster 3 | ✅ Unified |
| HAC | Cluster 18 | Cluster 19 | Cluster 17 | Cluster 16 | ❌ Fragmented |
| HDBSCAN | Noise | Noise | Noise | Noise | ❌ Failed |

Both Affinity Propagation (AP) and Spectral Clustering (SC) successfully grouped semantically related terms; others either over-fragmented or treated terms as noise.

### 4.3 Internal Validation Metrics

To select between AP and SC, three complementary metrics were evaluated:

**Silhouette Score** (higher is better: -1 to +1)
```
s(i) = [b(i) − a(i)] / max{a(i), b(i)}
```
where a(i) = average distance within cluster, b(i) = distance to nearest other cluster.
- Balances internal cohesion and separation
- No assumptions about cluster shape

**Davies-Bouldin Index** (lower is better)
- Quantifies average similarity between each cluster and its closest neighbor
- Penalizes overlap and cluster imbalance
- Assumes centroid-representable clusters (valid for semantic spaces)

**Calinski-Harabasz Score** (higher is better)
```
CH = trace(B_k) / (k−1) ÷ trace(W_k) / (N−k)
```
where B_k = between-cluster dispersion, W_k = within-cluster dispersion.
- Normalized by cluster count (prevents bias toward fragmentation)
- Computationally efficient

### 4.4 Final Algorithm Comparison

| Metric | Affinity Propagation | Spectral Clustering (k=3) |
|--------|---------------------|--------------------------|
| Silhouette Score ↑ | 0.41 | **0.56** |
| Davies-Bouldin ↓ | 1.02 | **0.71** |
| Calinski-Harabasz ↑ | 214 | **297** |

**Spectral Clustering** outperformed across all metrics, producing tighter, more interpretable clusters. The joint evaluation prevents overfitting and ensures fair comparison between AP (auto k=4) and SC (optimized k=3).

---

## Stage 5: Ontology Construction 🌳

Validated triples and clustered entities are synthesized into a structured RDF/OWL ontology.

### 5.1 Class Hierarchy Generation

For each cluster:
1. Select the entity with highest mean semantic similarity to other cluster members → designate as `owl:Class`
2. Other cluster members become `rdfs:subClassOf` this parent class

**Example Output:**
```
Class: AntioxidantProperties
  SubClass: anti-carcinogenic properties
  SubClass: anti-inflammatory properties
  SubClass: anti-oxidative properties
```

### 5.2 Semantic Annotations

Each class receives:
- **rdfs:label**: The entity string (e.g., "AntioxidantProperties")
- **rdfs:comment**: A contextual description generated from supporting sentences

The rdfs:comment generation is human-in-the-loop: machine-generated summaries are reviewed and optionally refined before inclusion.

---

## Evaluation Results 📊

EchoLLM was evaluated on the **CaRB (Comprehensive Assessment of Relation Extraction Benchmark)**, a large-scale open information extraction dataset.

### Quantitative Performance

| Metric | Direct LLM | EchoLLM | Change |
|--------|-----------|---------|--------|
| **Precision** | 0.65 | 0.76 | +11% ✅ |
| **Recall** | 0.74 | 0.57 | -17% |
| **F1-Score** | 0.69 | 0.65 | -4% |
| **False Positives** | 1229 | 558 | -55% ✅ |
| **True Positives** | 2232 | 1722 | -510 |

### Statistical Significance

A **McNemar's test** confirmed the precision improvement is not due to random chance:
```
χ²(1) = 319.07, p < 0.001 (highly significant)
```

### Trade-off Analysis

The precision-recall trade-off reflects EchoLLM's design philosophy: **prioritize factual correctness over coverage**. While some correct triples are filtered (recall drop), the system identifies 151 triples missed by the direct LLM baseline, demonstrating selective recovery of high-confidence facts.

**Contingency Table (McNemar's Breakdown):**

| | LLM Found & Correct | LLM Found & Incorrect |
|---|---|---|
| **EchoLLM Accepted** | 1571 | 151 |
| **EchoLLM Rejected** | 661 | 626 |

For applications where accuracy is paramount (scientific publishing, enterprise KGs), the 11% precision gain with 55% false positive reduction justifies the recall reduction.

### Qualitative Results

Domain experts reviewed EchoLLM-generated ontologies and confirmed:
- ✅ Tighter class hierarchies (no spurious groupings)
- ✅ Coherent relationships between entities
- ✅ All auto-generated rdfs:comment entries accepted without modification

---

## Runtime and Scalability ⚡

Processing 35 abstracts (~2,500 words) requires approximately **11 minutes** on MacBook M3 Pro (CPU-only NLI inference):

| Stage | Time |
|-------|------|
| Triple extraction | ~2 min |
| Retrieval & verification | ~7 min (80% of runtime) |
| Entity clustering | ~1 min |
| Ontology construction | ~1 min |

The **NLI verification step** is the computational bottleneck due to BART-Large-MNLI's inference cost. GPU acceleration significantly improves throughput; exact timings depend on corpus size, batch processing optimization, and hardware.

---

## Design Principles 💡

### 1. **Evidence-First Validation**
Every triple must prove its support in the source text. There are no shortcuts to plausibility.

### 2. **Transparency Over Complexity**
The pipeline surfaces failure modes (extraction failures, retrieval gaps, verification rejections) rather than hiding them. This enables human review and iterative refinement.

### 3. **No Domain-Specific Rules**
The system requires no ontology templates, entity type definitions, or relation schemas. It works across diverse text types (academic abstracts, technical reports, news articles) without reconfiguration.

### 4. **Flexible Model Choices**
While EchoLLM defaults to Llama3-8B, BART-Large-MNLI, and Spectral Clustering, these components can be swapped. The architecture supports alternative LLMs, NLI models, and clustering algorithms.

### 5. **Auditability**
Every decision is logged with confidence scores and supporting evidence. This enables traceability, debugging, and compliance requirements in regulated domains.

---

## Comparison with Naive LLM-Only Approaches

### Direct LLM Baseline
```
Raw Text → LLM (extract triples) → Knowledge Graph
```
**Problems:**
- No verification of triple accuracy
- Hallucinated triples accepted at face value
- 65% precision (35% false positives)
- No transparency into which facts are supported

### EchoLLM
```
Raw Text → LLM (extract) → Retrieval (find evidence) → NLI (verify logic) → Clustering (organize) → Knowledge Graph
```
**Advantages:**
- Every triple verified against source text
- 76% precision (55% fewer false positives)
- Full auditability and transparency
- Hierarchical organization of concepts

---

## Limitations & Future Work 🔮

### Known Limitations

1. **Precision-Recall Trade-off**: The focus on precision necessarily reduces recall. Recall can be improved by lowering the NLI entailment threshold (0.7), but this reintroduces hallucinations. The current threshold balances these competing objectives.

2. **Domain Shift in NLI**: BART-Large-MNLI is trained on general English. Performance may degrade on highly specialized domains (medical terminology, legal jargon, proprietary ontologies) where NLI models have limited training data.

3. **Predicate Complexity**: The system assumes single-word or simple multi-word predicates. Complex relational structures (n-ary relations, temporal constraints, modal qualifications) require extensions to the triple model.

4. **Pronoun Resolution**: While context expansion partially addresses anaphora, complex multi-sentence dependencies or distant pronouns may not be fully resolved by the expansion heuristic.

### Future Directions

- **Domain Adaptation**: Fine-tune NLI models on domain-specific corpora (biomedical, legal, financial)
- **Multi-Hop Reasoning**: Extend verification to chains of inferences (if A→B and B→C, then A→C)
- **Temporal Reasoning**: Add support for time-dependent facts and event sequences
- **Scalability Optimization**: GPU acceleration and batch verification for larger corpora
- **Interactive Refinement**: User feedback loops to refine clustering and class hierarchies

---

## Repository Structure

```
echollm/
├── preprocessing/
│   └── minimal_triad.py         # Normalization, segmentation, weighting
├── extraction/
│   └── llm_extractor.py         # LLM-based triple generation
├── retrieval_verification/
│   ├── hybrid_retriever.py      # BM25 + dense retrieval + RRF
│   └── nli_verifier.py          # BART-Large-MNLI verification logic
├── clustering/
│   └── entity_clustering.py     # Spectral clustering + metrics
├── ontology_construction/
│   └── ontology_builder.py      # RDF/OWL generation
├── evaluation/
│   └── benchmark.py             # CaRB dataset evaluation
└── examples/
    └── sample_workflow.md       # Step-by-step usage guide
```

---

## Installation & Usage

### Prerequisites
- Python 3.9+
- torch, transformers (for LLM & NLI models)
- scikit-learn (clustering & metrics)
- stanza (preprocessing & POS tagging)
- sentence-transformers (semantic embeddings)
- rank_bm25 (lexical retrieval)

### Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/echollm.git
cd echollm

# Install dependencies
pip install -r requirements.txt

# Run sample workflow
python examples/sample_workflow.py --input sample_text.txt --output ontology.owl
```

### Detailed Configuration

Refer to the configuration guide in `docs/configuration.md` for:
- LLM model selection and hyperparameters
- NLI threshold tuning
- Clustering algorithm alternatives
- Output format options (RDF, OWL, JSON-LD)

---

## Citation

If you use EchoLLM in your research, please cite:

```bibtex
@article{echollm2025,
  title={EchoLLM: Embedding Clarity for Hallucination Optimization with Large Language Models},
  author={Dalal, Aryan Singh and McGinty, Hande},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025}
}
```

---

## Acknowledgments

This work was developed at Kansas State University's Department of Computer Science. We thank domain experts who reviewed ontology outputs and provided constructive feedback on annotation quality.

---

## License

MIT License (see LICENSE file)

---

## Contact

For questions, issues, or collaboration inquiries:
- **Aryan Singh Dalal**: aryan.dalal@ksu.edu
- **Hande McGinty**: hande@ksu.edu

---

## Additional Resources

- **Full Paper**: [Link to AAAI published version]
- **Benchmark Dataset**: [Link to CaRB benchmark]
- **Extended Technical Report**: [Link to detailed methodology documentation]
- **Related Work**: See `docs/related_work.md` for positioning within RAG and ontology engineering literature

f
