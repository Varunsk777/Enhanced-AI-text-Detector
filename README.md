AI Text Detector
An explainable AI text detector that distinguishes between human-written and AI-generated text using a fine-tuned DistilBERT model, with advanced features for text humanization, synonym replacement, and sentiment analysis. Built with PyTorch, Transformers, SHAP, BART, and a Gradio interface, this project achieves ~97-98% validation accuracy on the HC3 dataset.
Project Overview
With the rise of large language models (LLMs) like ChatGPT, distinguishing human from AI-generated text is crucial for publishers, educators, and content creators. This project addresses this challenge by providing a unified, transparent, and user-friendly system that not only detects AI-generated text but also explains predictions, refines text to sound more human-like, and supports interactive text enhancement.
Key Features

Text Classification: Fine-tuned DistilBERT to classify text as human or AI-generated with high accuracy.
Explainability: Uses SHAP (Shapley Additive Explanations) to highlight influential tokens, visualized as interactive Plotly bar charts.
Text Humanization: Employs Corrective Retrieval-Augmented Generation (CRAG) with SentenceTransformer, FAISS, and BART-large to transform AI text into natural, human-like prose.
Synonym Generation: Generates context-aware synonyms using BART-base, informed by web context (DuckDuckGo) and WordNet, integrated into a Gradio interface.
ReAct Loop: Refines low-confidence predictions by replacing key words (identified by SHAP) with synonyms, re-classifying up to three times.
Sentiment Analysis: Analyzes text tone for enhanced contextual understanding.
Interactive Interface: Gradio-based UI for text classification, synonym selection, and visualization.

Novelty
Unlike tools like DetectGPT, which only detect AI text, this system:

Provides explainability through SHAP visualizations.
Enhances text with humanization using CRAG, blending BART paraphrasing, web context, and WordNet synonyms.
Supports human-in-the-loop correction via interactive synonym replacement.
Adapts to noisy, real-world data with robust context retrieval (FAISS) and corrective actions.

Implementation

Dataset: 3000 balanced samples (1500 human, 1500 AI) from the HC3 dataset (Hugging Face).
Training: Fine-tuned distilbert-base-uncased for 5 epochs (batch size: 8, learning rate: 2e-5) using mixed precision on Kaggle’s GPU.
Model Storage: Saved to /kaggle/working/ai_text_detector_model.
Technologies: PyTorch, Transformers, SHAP, SentenceTransformers, FAISS, BART, WordNet, DuckDuckGo API, Gradio.

Getting Started

Clone the Repository:git clone https://github.com/Varunsk777/llm-project.git
cd llm-project


Install Dependencies:pip install torch transformers scikit-learn matplotlib seaborn tqdm datasets numpy gradio faiss-cpu


Run the Notebook:Open llm-final_code.ipynb in Jupyter Notebook and execute the cells to train the model, test predictions, or launch the Gradio interface.

Future Work

Expand the dataset to include more diverse text sources.
Integrate additional LLMs for enhanced paraphrasing.
Optimize FAISS retrieval for larger context databases.

Acknowledgments

HC3 Dataset by Hello-SimpleAI.
Hugging Face Transformers, SHAP, and Gradio communities.

Explore the code to dive into advanced NLP for AI text detection and humanization!
