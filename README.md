# 🧠 Enhanced AI Text Detector

An **explainable AI text detector** that distinguishes between human-written and AI-generated content with high accuracy. Built using a fine-tuned **DistilBERT** model, this tool incorporates advanced features like **text humanization**, **synonym replacement**, and **sentiment analysis** — all in an interactive **Gradio interface**.

![Accuracy Badge](https://img.shields.io/badge/Accuracy-97%25-brightgreen)
![License Badge](https://img.shields.io/badge/License-MIT-blue)
![Python Badge](https://img.shields.io/badge/Made%20With-Python%203.8%2B-yellow)

---

## 🚀 Project Overview

With the rise of large language models (LLMs) like ChatGPT, it's becoming increasingly important to **detect AI-generated content**. This project offers a unified and transparent system to:

- **Detect AI-generated text**
- **Explain predictions via SHAP**
- **Humanize AI text**
- **Suggest context-aware synonyms**
- **Analyze sentiment**

It achieves **97–98% validation accuracy** on the **HC3 dataset**.

---

## 🔍 Key Features

- **🔠 Text Classification**  
  Fine-tuned DistilBERT model trained to detect AI vs. human-written text.

- **💡 Explainability**  
  SHAP-based visualizations highlighting important words using Plotly.

- **✍️ Text Humanization (CRAG)**  
  Corrective Retrieval-Augmented Generation using SentenceTransformer + FAISS + BART-large.

- **🧠 Synonym Replacement**  
  BART-based context-aware synonyms with web context (DuckDuckGo) + WordNet.

- **♻️ ReAct Loop**  
  Replaces SHAP-highlighted words with synonyms and reclassifies up to 3 times to improve prediction confidence.

- **🎭 Sentiment Analysis**  
  Understands the emotional tone of the text to support downstream decisions.

- **🖥️ Gradio Interface**  
  Full-featured UI for classification, synonym selection, and humanization visualization.

---

## 🧪 Novelty

Unlike tools such as **DetectGPT**, this system offers:

- ✅ **Transparency** with token-level SHAP explanations  
- 🔄 **Interactive refinement** through the ReAct loop  
- 🧬 **Text enhancement** using CRAG and context-aware generation  
- 🌐 **Web + WordNet-informed synonyms**  

---

## 🛠️ Technologies Used

- `PyTorch`
- `Transformers` (Hugging Face)
- `Gradio`
- `SHAP`
- `SentenceTransformers`
- `FAISS`
- `BART`
- `WordNet`
- `DuckDuckGo API`

---

## 📂 Dataset & Training

- **Dataset**: [HC3 Dataset (Hugging Face)](https://huggingface.co/datasets/Hello-SimpleAI/HC3)  
  → 3000 samples (1500 Human, 1500 AI)

- **Training Configuration**:  
  - Model: `distilbert-base-uncased`  
  - Epochs: `5`  
  - Batch Size: `8`  
  - Learning Rate: `2e-5`  
  - Precision: Mixed  
  - Hardware: Trained on Kaggle GPU  

- **Model Checkpoint**:  
  Saved to `/kaggle/working/ai_text_detector_model`

---

## ⚙️ Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Varunsk777/Enhanced-AI-text-Detector.git
cd Enhanced-AI-text-Detector
