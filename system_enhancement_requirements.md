# Academic Paper Classification System - Enhancement Requirements

## 🎯 Main Objectives

### 1. Home Page Title Update
- **Current**: Generic title
- **New**: "Research Fields Classification based on Abstract of the Papers using Support Vector Machine"
- **Purpose**: Clear description of the system's core functionality

### 2. New Data Loading Page Creation
Create a comprehensive data loading interface with step-by-step ML pipeline visualization.

## 📊 Data Loading Page Specifications

### File Upload Requirements
- **Upload Button**: For dataset files
- **Supported Formats**: Excel (.xlsx, .xls) → automatically convert to CSV
- **Dataset Size**: 10-3000 papers per file
- **Required Columns**: 
  - Title
  - Abstract  
  - Category (matching existing SVM model categories)
- **Categories**: AI, Distributed Systems, Image Processing, Networking & Cybersecurity, Software Engineering

### Step-by-Step Pipeline Workflow

#### Step 1: Data Display
- **Feature**: Show uploaded file contents on page
- **Display**: Table view with paper titles, abstracts, and categories
- **Summary**: Dataset statistics (total papers, category distribution)

#### Step 2: Text Preprocessing
- **Button**: "Preprocess Data"
- **Function**: Process all papers' titles and abstracts
- **Operations**:
  - Text cleaning and normalization
  - Tokenization using NLTK
  - Stop word removal
  - Lemmatization
- **Output**: Display before/after preprocessing comparison

#### Step 3: Vectorization (NEW STAGE)
- **Button**: "Vectorize Text"
- **Purpose**: Convert text to numerical vectors for computer understanding
- **Methods Available**:
  - **Binary Vectorization**: Word presence/absence representation
  - **Bag of Words**: Simple word frequency vectors
  - **TF-IDF preparation**: Foundation for text representation
- **Output**: Vector statistics and dimensionality information

#### Step 4: Text Representation Calculation
- **Button**: "Calculate Text Representation"
- **Primary Method**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Alternative Methods**:
  - Count Vectorizer
  - N-gram features
  - Other TF-IDF alternatives for comparison
- **Selection**: Dropdown to choose method
- **Output**: Feature importance analysis and comparison metrics

#### Step 5: SVM Classification
- **Button**: "Train SVM Classifier"
- **Kernel Options**:
  - Linear SVM
  - Polynomial SVM
- **Selection**: Radio buttons or dropdown for kernel choice
- **Process**: Train model on processed data
- **Output**: Training progress and model parameters

#### Step 6: Accuracy & Evaluation
- **Button**: "Calculate Accuracy"
- **Metrics**:
  - Overall accuracy score
  - Confusion matrix visualization
  - Precision, recall, F1-score per category
  - Classification report
- **Output**: Comprehensive performance dashboard

## 🔄 Integration with Existing Pages

### Workflow Page Enhancements
- **Add**: Vectorization step before TF-IDF calculation
- **Feature**: Text representation technique selection
- **Options**: TF-IDF, Count Vectorizer, N-grams
- **UI**: Same interface design as data loading page

### Explore Dataset Page Enhancements
- **Add**: Vectorization analysis section
- **Feature**: Text representation comparison tools
- **Options**: Multiple vectorization methods side-by-side
- **UI**: Consistent with data loading and workflow pages

### Corpus Page Updates
- **Enhancement**: Include vectorization method selection
- **Feature**: Compare different text representation techniques
- **Analysis**: Show how different methods affect feature importance

## 🎨 User Interface Requirements

### Design Consistency
- **Principle**: Use same UI components across all pages
- **Elements**: 
  - Consistent button styling
  - Uniform progress indicators
  - Matching color schemes
  - Similar layout structures

### Interactive Elements
- **Progress Bar**: Show completion status for each step
- **Status Indicators**: Visual feedback for completed steps
- **Results Display**: Expandable sections for detailed outputs
- **Method Selection**: Clear dropdown/radio button interfaces

### Responsive Design
- **Mobile**: Ensure functionality on smaller screens
- **Tablet**: Optimize layout for medium screens
- **Desktop**: Full feature utilization on large displays

## 🛠 Technical Implementation Details

### New Components Required

#### Vectorization Module
```python
# TF-IDF implementation
# Bag of Words vectorizer
# Vector comparison utilities
# Dimensionality analysis tools
```

#### Enhanced UI Components
```html
<!-- Step-by-step progress tracker -->
<!-- Method selection interfaces -->
<!-- Results visualization panels -->
<!-- File upload with preview -->
```

#### Data Processing Pipeline
```python
# File upload handler (Excel → CSV)
# Sequential processing workflow
# State management between steps
# Results caching and display
```

### Integration Points
- **Flask Routes**: New endpoints for each processing step
- **JavaScript**: Dynamic UI updates and progress tracking
- **Data Storage**: Session-based state management
- **Visualization**: Charts and graphs for results display

## 📋 Acceptance Criteria

### Functional Requirements
- [ ] Home page title updated correctly
- [ ] New data loading page accessible from navigation
- [ ] File upload accepts Excel files and converts to CSV
- [ ] All 6 processing steps work sequentially
- [ ] Vectorization step integrated before TF-IDF
- [ ] Method selection available for text representation
- [ ] SVM kernel selection functional
- [ ] Accuracy calculation with confusion matrix display
- [ ] Integration completed in workflow and explore dataset pages

### Technical Requirements
- [ ] Consistent UI design across all pages
- [ ] Proper error handling for file uploads
- [ ] Progress tracking for long-running operations
- [ ] Responsive design implementation
- [ ] Performance optimization for large datasets
- [ ] Session management for data persistence

### User Experience Requirements
- [ ] Intuitive step-by-step workflow
- [ ] Clear visual feedback for each operation
- [ ] Informative results display
- [ ] Easy method comparison capabilities
- [ ] Seamless navigation between pages
- [ ] Helpful error messages and guidance

## 🚀 Implementation Priority

### Phase 1: Core Infrastructure
1. Update home page title
2. Create basic data loading page structure
3. Implement file upload functionality
4. Add vectorization module

### Phase 2: Processing Pipeline
1. Implement 6-step processing workflow
2. Add method selection interfaces
3. Create results visualization components
4. Integrate with existing pages

### Phase 3: Enhancement & Polish
1. Ensure UI consistency across all pages
2. Add advanced comparison features
3. Optimize performance and user experience
4. Comprehensive testing and refinement

## 📊 Expected Outcomes

### Enhanced User Experience
- Clear understanding of ML pipeline steps
- Interactive learning through step-by-step processing
- Ability to compare different methods and techniques
- Comprehensive results analysis and interpretation

### Educational Value
- Visual demonstration of text processing concepts
- Understanding of vectorization importance
- Comparison of different ML approaches
- Real-time feedback on model performance

### System Improvements
- More flexible text representation options
- Better integration between system components
- Enhanced visualization and analysis capabilities
- Improved workflow for research and experimentation