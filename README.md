# Research Fields Classification System

## Research Fields Classification based on Abstract of the Papers using Support Vector Machine

A comprehensive text mining and machine learning platform that classifies academic papers into 5 distinct research fields using advanced Natural Language Processing (NLP), multiple vectorization methods, and Support Vector Machine (SVM) classifiers. Features a complete step-by-step ML pipeline with educational visualizations.

## 🎯 Project Overview

This Flask-based web application automatically categorizes academic papers into one of five research domains:
- **Artificial Intelligence** - Machine learning, neural networks, computer vision, NLP
- **Distributed Systems** - Cloud computing, microservices, consensus algorithms
- **Image Processing** - Digital image analysis, computer vision, medical imaging
- **Networking & Cybersecurity** - Network security, cryptography, threat detection
- **Software Engineering** - Development methodologies, testing, DevOps, architecture

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- pip package manager

### Installation
1. **Clone and navigate to project**:
   ```bash
   git clone <repository-url>
   cd text_mining_project-V2
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data** (first time only):
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

4. **Start the application**:
   ```bash
   python app.py
   ```

5. **Access the web interface**:
   Open your browser to `http://127.0.0.1:5000`

## 🌐 Application Pages

### **Home Page**
- Landing page with updated title reflecting the SVM-based classification approach
- Quick access to all system features

### **Workflow Page** 
- Interactive step-by-step ML pipeline (4 steps: Preprocessing → Binary Vectorization → TF-IDF → Classification)
- Real-time text processing with visual feedback
- **Binary Vectorization**: Shows each word as 0s and 1s representing presence/absence
- SVM kernel selection (Linear vs Polynomial)

### **ML Pipeline Studio** ⭐ NEW
- Complete 6-step machine learning pipeline for custom dataset processing
- File upload with Excel support and validation
- **Binary Vectorization**: Words represented as binary digits (0s and 1s)
- Step-by-step progress with detailed visualizations
- Comprehensive model evaluation with confusion matrix

### **Explore Dataset Page**
- Browse all 3,000+ papers in the collection
- Enhanced with vectorization analysis section
- Compare different vectorization methods with interactive charts
- Detailed performance metrics by research field

### **Accuracy Chart Page**
- Interactive performance visualizations
- Model comparison across different configurations
- Detailed statistical analysis

### **Corpus & Vectorization Page**
- Enhanced corpus analysis with method selection
- TF-IDF analysis and vectorization methods
- Interactive charts showing method performance
- Research field-specific analysis

## 📊 Features & Capabilities

### 🆕 New Enhanced Features (v2.0)

#### **Step-by-Step Data Loading Pipeline**
- **Interactive File Upload**: Support for Excel files (.xlsx, .xls) with automatic CSV conversion
- **6-Step ML Pipeline**: Complete workflow from data upload to model evaluation
- **Real-time Progress Tracking**: Visual progress indicators for each step
- **Data Validation**: Automatic validation of required columns and data format

#### **Advanced Text Vectorization**
- **TF-IDF Vectorization**: Term Frequency-Inverse Document Frequency analysis
- **Binary Vectorization**: Word presence/absence representation
- **Interactive Visualizations**: Charts showing vector dimensions, performance metrics
- **Performance Analysis**: Detailed TF-IDF analysis with word importance scores

#### **Enhanced ML Pipeline**
- **Text Preprocessing**: Advanced preprocessing with before/after comparisons
- **SVM Kernel Selection**: Choose between Linear and Polynomial kernels
- **Comprehensive Evaluation**: Confusion matrix, precision, recall, F1-score
- **Performance Dashboard**: Interactive charts and detailed classification reports

#### **Educational Interface**
- **Step-by-Step Learning**: Understand each phase of the ML pipeline
- **Method Comparison**: Compare different approaches side-by-side
- **Visual Analytics**: Interactive charts using Chart.js
- **Responsive Design**: Works on desktop, tablet, and mobile devices

### Core Features
- **Real-time Classification**: Instant paper categorization from title and abstract
- **Interactive Workflow**: Step-by-step visualization of the ML pipeline
- **Dual Kernel Support**: Linear and Polynomial SVM kernels
- **Performance Metrics**: Comprehensive accuracy, precision, recall, and F1-score analysis
- **Dataset Exploration**: Browse and analyze papers by research field
- **TF-IDF**: Examine feature importance and word weights

### Web Interface Pages
- **Home**: Main classification interface
- **Workflow**: Interactive ML pipeline demonstration
- **Insights**: Performance metrics and model comparison
- **Corpus**: TF-IDF analysis and feature exploration
- **Dataset Explorer**: Browse papers by category

## 🛠 Technical Architecture

### Machine Learning Pipeline
1. **Text Preprocessing**:
   - Lowercasing and special character removal
   - Tokenization using NLTK
   - Stop word removal
   - Lemmatization for word normalization

2. **Feature Extraction**:
   - TF-IDF vectorization
   - MaxAbsScaler normalization
   - Configurable feature limits

3. **Classification Models**:
   - Linear SVM (primary model)
   - Polynomial SVM (degree=2, comparative model)
   - Probability estimation enabled

### Dataset Structure
```
data/
├── ai_data.xlsx                    # AI/ML papers
├── distribution_data.xlsx          # Distributed systems papers
├── image_processing_data.xlsx      # Image processing papers
├── networking_cybersecurity_data.xlsx # Security papers
└── se_data.xlsx                    # Software engineering papers
```

## 📁 Project Structure

```
text_mining_project-V2/
├── app.py                          # Main Flask application
├── preprocessing.py                # Data preprocessing utilities
├── dataset_expansion_tool.py       # Dataset generation tools
├── gap_analysis.py                 # Terminology analysis
├── requirements.txt                # Python dependencies
├── static/
│   └── style.css                   # Web interface styling
├── templates/                      # HTML templates
│   ├── home.html                   # Main interface
│   ├── workflow.html               # ML pipeline demo
│   ├── insights.html               # Performance metrics
│   ├── corpus.html                 # TF-IDF 
│   └── explore_dataset.html        # Dataset browser
└── data/                           # Excel datasets (5 categories)
```

## 🔧 Utility Scripts

### Dataset Management
- **`dataset_expansion_tool.py`**: Generate synthetic paper data for testing
- **`gap_analysis.py`**: Analyze terminology coverage across datasets
- **`check_data_structure.py`**: Validate dataset integrity

### Model Analysis
- **`svm_model_comparison.py`**: Compare different SVM configurations
- **`svm_decision_boundary_demo.py`**: Visualize decision boundaries
- **`simple_svm_comparison.py`**: Basic model performance comparison

### Data Processing
- **`calculate_counts.py`**: Generate dataset statistics
- **`analyze_dataset.py`**: Comprehensive dataset analysis

## 📈 API Endpoints

### Classification Endpoints
- `POST /predict` - Classify paper by title/abstract
- `POST /preprocess` - Show preprocessing steps
- `POST /calculate_tfidf` - Calculate TF-IDF values
- `POST /classify` - Perform classification with kernel selection

### Analysis Endpoints
- `GET /metrics` - Model performance metrics
- `GET /corpus_data` - TF-IDF corpus analysis
- `GET /api/papers/<field>` - Get papers by research field
- `GET /api/datasets` - Dataset information and statistics

## 🎯 Model Performance

The system achieves high accuracy through:
- **Balanced Training Data**: Equal representation across all 5 categories
- **Advanced Preprocessing**: NLTK-based text normalization
- **Optimized Features**: TF-IDF with stop word removal
- **Dual Kernel Approach**: Linear and polynomial SVM comparison

Performance metrics are dynamically calculated based on:
- Dataset size (paper count)
- Train/test split ratio
- Cross-validation results

## 🛡 Security & Data Handling

- No malicious code patterns detected
- Defensive security focus
- Local data processing (no external API calls)
- Safe file handling with existence checks
- Input validation and sanitization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 Dependencies

Core packages (see `requirements.txt`):
- Flask - Web framework
- scikit-learn - Machine learning
- pandas - Data manipulation
- nltk - Natural language processing
- openpyxl - Excel file handling

## 🔍 Usage Examples

### Command Line Classification
```python
from app import preprocess_simple, linear_pipeline, label_encoder

text = "Machine learning algorithms for image recognition"
processed = preprocess_simple(text)
prediction = linear_pipeline.predict([processed])[0]
category = label_encoder.inverse_transform([prediction])[0]
print(f"Category: {category}")
```

### Web Interface
1. Navigate to the home page
2. Enter paper title and abstract
3. Select SVM kernel (Linear/Polynomial)
4. View classification results with confidence scores

## 🔬 Research Applications

This system can be used for:
- Academic paper organization
- Literature review automation
- Research trend analysis
- Conference paper categorization
- Knowledge domain mapping
