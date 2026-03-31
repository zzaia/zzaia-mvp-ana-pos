# ANA-POS — Semantic Search over STJ Súmulas

## 🎓 **Academic Project Information**
This repository contains one of three major postgraduate MVP projects developed for the **Data Science, Machine Learning and Data Engineering** program at **PUC-Rio University**. The project demonstrates practical application of NLP and semantic search techniques to Brazilian legal document analysis.

## 🎯 **Domain & Problem Statement**

### **Domain: Brazilian Legal NLP — Superior Court of Justice (STJ)**
The Brazilian judicial system produces thousands of binding legal summaries (*Súmulas*) that consolidate the STJ's settled jurisprudence on specific legal matters. These documents span dozens of legal areas and sub-areas, making manual search for relevant precedents time-consuming and imprecise.

### **Problem Definition**
This project addresses the central question:

> **"Which query is most likely to appear as a case theme in the STJ Súmulas document?"**

The pipeline answers this through semantic similarity analysis, estimating which Portuguese legal query aligns most closely with the themes present in the STJ Súmulas corpus.

### **What is a Súmula?**
A **Súmula** is a numbered binding legal summary issued by a Brazilian court that consolidates its jurisprudence on a specific legal matter. The STJ Súmulas are issued by the *Superior Tribunal de Justiça* — the court responsible for harmonising the interpretation of federal law across Brazil. Each Súmula distils the court's settled position into a single authoritative statement, grouped by legal area (e.g., Bancário, Administrativo, Processual Civil).

### **Business Context**
- **Legal Research Efficiency**: Rapid semantic retrieval of relevant Súmulas without keyword matching
- **Jurisprudence Analysis**: Identify which legal themes are most densely represented in the corpus
- **Query Relevance Ranking**: Probabilistically rank queries by occurrence likelihood in the document
- **Cross-Area Reasoning**: Surface thematically related Súmulas even when wording differs

## 📊 **Dataset & Methodology**

The project processes the official **STJ Súmulas PDF** (`Súmulas - STJ.pdf`), a 2,642-page document containing over 500 Súmulas grouped by legal area. After extraction and segmentation, the pipeline produces **288 labeled Súmula segments** used for semantic indexing.

## 🔬 **Technical Implementation**

### **Core Notebook**

#### **Main Pipeline**
- **File**: [`mvp-ana-pos-main.ipynb`](./mvp-ana-pos-main.ipynb)
- **Purpose**:
  - 10-step NLP pipeline from raw PDF to probabilistic query ranking
  - BERTimbau semantic embeddings (1024-dimensional)
  - FAISS cosine similarity search across the full corpus
  - KDE + Softmax + Z-score probabilistic query occurrence ranking
  - Cosine similarity distribution visualization per query

### **Pipeline Steps**

| Step | Component | Role |
|------|-----------|------|
| 0 | `PdfReader` | Extract raw text with OCR fallback |
| 1 | `EncodingNormalizer` | Repair mojibake and encoding artifacts |
| 2 | `BoilerplateRemover` | Mark repeated headers as delimiters via TF-IDF |
| 3 | `SentenceSegmenter` | Split into per-Súmula blocks |
| 4 | `CitationNormalizer` | Replace citations with typed tokens |
| 5 | `SumulaLabeler` | Annotate area + sub-area + Súmula number |
| 6 | `EmbeddingGenerator` | BERTimbau mean-pooled vectors (1024-dim) |
| 7 | `SearchIndexBuilder` | FAISS cosine similarity search with checkpoint |
| 8 | `SimilarityVisualizer` | Cosine similarity histograms per query |
| 9 | `ProbabilityEstimator` | KDE + Softmax + Z-score query occurrence ranking |

### **Supporting Infrastructure**
- **Azure Integration**: [`pipeline/scripts/azure_utils.py`](./pipeline/scripts/azure_utils.py)
  - Cloud-based dataset and checkpoint storage
  - `AzureBlobDownloader` — downloads PDF and `.pkl` checkpoints from Azure Blob Storage (`mvpanasup` container)

## 🛠 **Technologies & Tools**

- **NLP & Embeddings**: HuggingFace Transformers, BERTimbau (`neuralmind/bert-large-portuguese-cased`)
- **Vector Search**: FAISS (`faiss-cpu`)
- **PDF Processing**: pdfplumber, pytesseract, Pillow
- **Text Processing**: ftfy, scikit-learn (TF-IDF)
- **Statistics**: NumPy, SciPy (Gaussian KDE, softmax)
- **Visualization**: Matplotlib
- **Cloud Integration**: Azure Blob Storage
- **Development**: Jupyter Notebooks, Python 3.8+

## 📈 **Key Results & Achievements**

- **Winner Query**: `"banco cobrou taxa de juros abusiva no contrato"` ranked first across all three probabilistic metrics (softmax, KDE, z-score) with mean cosine similarity of **0.6051**
- **Runner-up**: `"servidor público foi demitido por ato administrativo ilegal"` (mean similarity 0.5942), confirming administrative law as the second most represented theme
- **Corpus Coverage**: 288 labeled Súmula segments across 20+ legal areas, embedded in 1024-dimensional BERTimbau space
- **Semantic Search**: FAISS index enables sub-second retrieval of the full corpus per query
- **Checkpoint System**: Directory-based `.pkl` checkpoints skip expensive BERTimbau inference on re-runs

## 🤖 AI-Assisted Development

This project showcases the integration of AI-powered development workflows using a **customized agentic system** built on Claude Code.

### Custom AI Workspace
- **Repository**: [zzaia-agentic-workspace](https://github.com/zzaia/zzaia-agentic-workspace)
- **Toolset**: Claude Code with custom agents, slash commands, and templates
- **Workflow**: Multi-repository management, git worktrees, automated task orchestration

## 🚀 **Getting Started**

### **Quick Start with Google Colab**

Run the main notebook directly in your browser without any local setup:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zzaia/zzaia-mvp-ana-pos/blob/main/mvp-ana-pos-main.ipynb)

**Benefits of using Colab:**
- ✅ No local installation required
- ✅ Free GPU access for BERTimbau inference
- ✅ Pre-configured Python environment
- ✅ Direct integration with GitHub
- ✅ Easy sharing and collaboration

**Note**: The notebook will automatically clone the repository and install required dependencies when run in Colab.

### **Prerequisites (Local Setup)**
```bash
pip install pdfplumber pytesseract Pillow ftfy scikit-learn
pip install transformers torch numpy matplotlib faiss-cpu scipy
pip install azure-storage-blob jupyter
```

### **Running the Pipeline**
1. Place `Súmulas - STJ.pdf` in `datasets/` (or let the Azure fetch cell download it automatically)
2. Execute `mvp-ana-pos-main.ipynb` from top to bottom
3. Checkpoints are saved to `pipeline/checkpoints/` — re-runs skip cached steps automatically

### **Cloud Integration**
Dataset and checkpoints are retrieved from Azure Blob Storage when not present locally:
```python
from azure_utils import AzureBlobDownloader
downloader = AzureBlobDownloader("your_url", "your_container")
```

## 📚 **Academic Context**

This project represents the intersection of:
- **Applied NLP**: Transformer-based semantic search on Portuguese legal text
- **Data Engineering**: Modular pipeline with checkpoint-based caching and cloud integration
- **Domain Expertise**: Understanding of Brazilian legal structure and STJ jurisprudence
- **Business Analytics**: Probabilistic ranking of legal query themes with interpretable metrics

## 🏛 **Institution**
**Pontifícia Universidade Católica do Rio de Janeiro (PUC-Rio)**
Postgraduate Program in Data Science, Machine Learning and Data Engineering

## 📝 **License**
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 **Contributing**
This is an academic project. For questions or collaboration inquiries, please open an issue or contact the project maintainers.

##
> *"I can do all this through him who gives me strength."*
>
> **— Philippians 4:13**
