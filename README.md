# 📄 Extractive & Abstractive Text Summarization — A Deep Learning NLP Approach

An end-to-end NLP system that implements and compares two paradigms of automatic text summarization:
- **Extractive summarization** using the graph-based TextRank algorithm
- **Abstractive summarization** using pre-trained transformer models (T5 and BART)

Both approaches are integrated into a unified GUI application.

---

## 📋 Table of Contents

- [Problem Statement](#problem-statement)
- [Approach](#approach)
- [Models & Algorithms](#models--algorithms)
- [Project Structure](#project-structure)
- [Technologies & Libraries](#technologies--libraries)
- [How to Run](#how-to-run)
- [Results](#results)
- [Key Findings](#key-findings)
- [Future Work](#future-work)

---

## Problem Statement

The exponential growth of digital text (news, research papers, social media) makes manual reading of long documents impractical. **Automatic text summarization** aims to condense documents while preserving key information and overall meaning.

Two fundamentally different approaches exist:
- **Extractive**: Select and return the most important sentences directly from the source text
- **Abstractive**: Generate new sentences that paraphrase the key points (like how a human would summarize)

This project implements, evaluates, and contrasts both approaches.

---

## Approach

### Extractive Summarization — TextRank
TextRank is an unsupervised graph-based algorithm inspired by Google's PageRank. The pipeline:

1. **Sentence tokenization** — split the document into individual sentences
2. **TF-IDF / word embedding vectorization** — represent each sentence as a numerical vector
3. **Similarity matrix construction** — compute cosine similarity between all sentence pairs
4. **Graph construction** — sentences as nodes, similarity scores as edge weights
5. **PageRank scoring** — iteratively compute importance scores for each sentence
6. **Extraction** — return top-N highest-scoring sentences

### Abstractive Summarization — T5 & BART
Pre-trained transformer models fine-tuned on summarization datasets:

| Model | Architecture | Pre-training Objective | Dataset |
|-------|-------------|------------------------|---------|
| **T5** (Text-to-Text Transfer Transformer) | Encoder-Decoder | Text-to-text reframing | CNN/DailyMail |
| **BART** (Bidirectional Auto-Regressive Transformer) | Encoder-Decoder | Denoising auto-encoder | CNN/DailyMail, XSum |

Both models generate summaries using beam search decoding with configurable `max_length`, `min_length`, and `num_beams`.

---

## Models & Algorithms

### TextRank (Extractive)
```
Input Text → Sentence Segmentation → Vectorization → Cosine Similarity Matrix 
→ Graph (Networkx) → PageRank → Top-N Sentences → Summary
```

### T5 (Abstractive)
```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
model = T5ForConditionalGeneration.from_pretrained("t5-small")
inputs = tokenizer("summarize: " + article, return_tensors="pt", max_length=512)
summary_ids = model.generate(inputs["input_ids"], max_length=150, num_beams=4)
```

### BART (Abstractive)
```python
from transformers import BartForConditionalGeneration, BartTokenizer
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
inputs = tokenizer(article, return_tensors="pt", max_length=1024)
summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=200)
```

---

## Project Structure

```
Extractive-and-Abstractive-Text-Summarization/
│
├── Extractive_Summarization_using_Text_Rank_Algorithm.ipynb  # TextRank from scratch
├── Abstractive_summarization_using_T5_and_BART.ipynb         # T5 + BART experiments
│
├── Final_Extractive_Summarizer.ipynb     # Cleaned extractive pipeline
├── Final_Abstractive_summarizer_gui.ipynb # Abstractive with GUI interface
├── Merged_both_models_with_all_types.ipynb # Unified system: all 3 models + GUI
│
├── Documentation.docx     # Full project report
├── Presentation.pptx      # Project presentation slides
└── README.md
```

---

## Technologies & Libraries

| Category | Library / Tool |
|----------|---------------|
| NLP & Transformers | `transformers` (HuggingFace), `nltk`, `spacy` |
| ML Framework | `torch` (PyTorch) |
| Summarization Models | `T5ForConditionalGeneration`, `BartForConditionalGeneration` |
| Graph Algorithm | `networkx` (TextRank graph) |
| Vectorization | `sklearn.feature_extraction.text` (TF-IDF) |
| GUI | `tkinter` |
| General | `numpy`, `pandas` |

---

## How to Run

### Install Dependencies
```bash
pip install transformers torch nltk networkx scikit-learn
```

### Extractive Summarization
Open and run `Final_Extractive_Summarizer.ipynb` in Jupyter. Provide an input text string; the notebook outputs the top extracted sentences.

### Abstractive Summarization (T5 / BART)
Open `Abstractive_summarization_using_T5_and_BART.ipynb`. The first run will download model weights (~850 MB for BART).

### Unified GUI Application
Run `Merged_both_models_with_all_types.ipynb` to launch the Tkinter interface where users can:
- Paste any text
- Select summarization method (Extractive / T5 / BART)
- Configure summary length
- View the generated summary

---

## Results

| Method | Approach | Pros | Cons |
|--------|----------|------|------|
| TextRank | Extractive | Fast, no GPU needed, preserves original wording | Cannot paraphrase; misses implicit meaning |
| T5-small | Abstractive | Lightweight, generates fluent new text | Less accurate than large models |
| BART-large-cnn | Abstractive | Best quality summaries, strong coherence | Slow inference, requires GPU for large docs |

**Key metric**: ROUGE scores were used to evaluate summary quality against reference summaries.
- BART consistently outperformed T5 on ROUGE-1 and ROUGE-L for news articles
- TextRank performed well on well-structured documents (reports, articles) but poorly on informal text

---

## Key Findings

1. **No single best method** — the optimal choice depends on document type, length, and available compute
2. **Pre-trained transformers dramatically outperform classical NLP** for abstractive tasks with zero fine-tuning
3. **TextRank is surprisingly competitive** for formal, well-structured text where the most important sentences are already well-formed
4. **GUI integration** makes the system accessible for non-technical users and demonstrates end-to-end product thinking

---

## Future Work

- Fine-tune BART/T5 on domain-specific data (scientific papers, legal documents)
- Add support for multi-document summarization
- Implement ROUGE evaluation pipeline for quantitative comparison
- Explore newer models: PEGASUS, LED (Longformer Encoder-Decoder) for long documents

---

## References

- Mihalcea, R. & Tarau, P. (2004). TextRank: Bringing Order into Text. *EMNLP 2004*
- Lewis, M. et al. (2020). BART: Denoising Sequence-to-Sequence Pre-training. *ACL 2020*
- Raffel, C. et al. (2020). Exploring the Limits of Transfer Learning with T5. *JMLR*
- HuggingFace Transformers: [https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)

---

*This project was developed as part of coursework at Gujarat University, demonstrating applied deep learning for NLP tasks.*
