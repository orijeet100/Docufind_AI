
# 🤖 Documind AI: Advanced Document & Website QA System

<a href="[[https://your-destination-link.com]](https://drive.google.com/file/d/13p0rVHTKuL0Brra_bVWwJC9XxcmOIPSA/view?usp=drive_link))">
  <img src="documind_thumbnail" alt="Thumbnail" width="80%">
</a>

[View this video post]([https://www.linkedin.com/feed/update/urn:li:ugcPost:7345461036101124096](https://drive.google.com/file/d/13p0rVHTKuL0Brra_bVWwJC9XxcmOIPSA/view?usp=drive_link)])


A sophisticated **Retrieval-Augmented Generation (RAG)** system that provides intelligent question-answering capabilities for both uploaded documents and web content. This project demonstrates the power of combining modern language models with semantic search to create accurate, context-aware responses.

## 🎯 Project Overview

This project implements a comprehensive RAG system that:
- **Processes multiple document formats** (PDF, DOCX, TXT, XLSX, PPTX, CSV)
- **Scrapes and analyzes web content** using FireCrawl
- **Compares RAG vs Non-RAG performance** using state-of-the-art evaluation metrics
- **Provides an intuitive Streamlit interface** for easy interaction
- **Implements advanced evaluation frameworks** for model performance assessment

## 🏗️ Architecture & Technology Stack

### Core Technologies
- **Frontend**: Streamlit 1.41.0
- **Backend**: Python with FastAPI
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Generation Model**: `meta-llama/Llama-3.2-1B-Instruct`
- **Web Scraping**: FireCrawl API
- **Document Processing**: PyMuPDF, python-docx, python-pptx

### Key Libraries
- **Transformers**: Hugging Face model integration
- **Torch**: Deep learning framework
- **Pandas**: Data manipulation
- **Phoenix**: LLM evaluation framework
- **Guardrails**: AI safety and validation
- **Phi**: Agent framework for LLM orchestration

## 📊 Model Performance & Evaluation

### RAG vs Non-RAG Comparison Results

The project includes comprehensive evaluation comparing RAG-enhanced responses against baseline non-RAG responses:

#### BLEU Score Analysis
- **RAG Performance**: BLEU score of 0.027 (2.7%)
- **Non-RAG Performance**: BLEU score of 0.0 (0%)

#### Key Findings
1. **Significant Improvement**: RAG approach shows measurable improvement over baseline
2. **Precision Enhancement**: RAG achieves higher precision scores across n-gram levels
3. **Contextual Accuracy**: Better relevance and factual accuracy in responses
4. **Length Optimization**: More appropriate response lengths with better brevity penalty

### Evaluation Framework
- **BLEU Score**: Measures translation quality and response similarity
- **Precision Metrics**: N-gram precision analysis (1-gram to 4-gram)
- **Brevity Penalty**: Ensures appropriate response length
- **Length Ratio**: Optimal balance between response and reference length

## 🚀 Features

### Document Processing
- **Multi-format Support**: PDF, DOCX, TXT, XLSX, PPTX, CSV
- **Text Extraction**: Advanced parsing for complex document structures
- **Preview Functionality**: Built-in document viewer for uploaded files
- **Batch Processing**: Handle multiple documents simultaneously

### Web Content Analysis
- **Intelligent Scraping**: FireCrawl integration for comprehensive web content extraction
- **Markdown Processing**: Structured content extraction
- **URL Validation**: Robust error handling for invalid URLs
- **Content Preview**: Real-time website content display

### Question-Answering System
- **Semantic Search**: FAISS-based similarity search for relevant document retrieval
- **Context-Aware Responses**: LLM-generated answers based on retrieved context
- **Custom Queries**: User-defined questions with intelligent file targeting
- **Example Questions**: Pre-built question templates for common use cases

### Advanced Features
- **Session Management**: Persistent state across interactions
- **Response Tracking**: Detailed logging of question-answer pairs
- **Distance Metrics**: Similarity scores for transparency
- **File Attribution**: Source document identification for answers

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, for faster processing)
- Hugging Face account and API token

### Environment Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd LLM_final_project
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
# Create .env file
MODEL_PATH=sentence-transformers/all-MiniLM-L6-v2
HF_MODEL_PATH=meta-llama/Llama-3.2-1B-Instruct
FIRECRAWL_API=your_firecrawl_api_key
HUGGINGFACE_TOKEN=your_hf_token
```

5. **Run the application**
```bash
streamlit run app.py
```

## 📖 Usage Guide

### Document Analysis
1. Click "Document Insights" button
2. Upload one or more documents (PDF, DOCX, TXT, XLSX, PPTX, CSV)
3. Preview documents using the preview buttons
4. Ask questions using:
   - Custom text input
   - Pre-built example questions
   - File-specific queries

### Website Analysis
1. Click "Website Insights" button
2. Enter a valid URL
3. Wait for content scraping and processing
4. Ask questions about the website content
5. Use example questions or custom queries

### Question Types
- **General Questions**: "What is the main topic?"
- **Specific Queries**: "What are the key points about [topic]?"
- **File-Specific**: "What conclusions can be drawn from [filename]?"
- **Custom Questions**: Any natural language question

## 🔬 Technical Implementation

### RAG Pipeline
1. **Document Processing**: Text extraction and chunking
2. **Embedding Generation**: Vector representation using sentence transformers
3. **Index Creation**: FAISS index for similarity search
4. **Query Processing**: Question embedding and retrieval
5. **Context Assembly**: Relevant document retrieval
6. **Answer Generation**: LLM response with retrieved context

### Evaluation Methodology
- **Dataset**: Custom QA dataset with diverse topics
- **Metrics**: BLEU, precision, recall, F1-score
- **Comparison**: RAG vs non-RAG baseline
- **Validation**: Human evaluation and automated metrics

## 📈 Performance Insights

### Key Advantages of RAG Approach
1. **Improved Accuracy**: 2.7% BLEU score vs 0% baseline
2. **Better Context**: Relevant document retrieval
3. **Reduced Hallucination**: Grounded responses in source material
4. **Enhanced Relevance**: Semantic similarity matching

### Technical Optimizations
- **Efficient Embedding**: MiniLM model for fast processing
- **Scalable Search**: FAISS for high-performance similarity search
- **Memory Management**: Optimized document chunking
- **Parallel Processing**: Batch document handling

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for:
- Code style and standards
- Testing requirements
- Documentation updates
- Feature proposals

## 📄 License

This project is licensed under the GPL License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Hugging Face** for transformer models and evaluation tools
- **Facebook Research** for FAISS similarity search
- **Streamlit** for the web application framework
- **FireCrawl** for web scraping capabilities
- **Phoenix** for LLM evaluation framework

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the documentation
- Review the evaluation results in the Jupyter notebook

---
