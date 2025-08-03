# Recipe and Nutrition RAG with Dietary Restrictions

A comprehensive RAG (Retrieval-Augmented Generation) system that processes recipes, nutritional information, and dietary guidelines to provide personalized meal suggestions based on health conditions and preferences.

## 🍽️ Features

### Core Functionality
- **Recipe Database Processing**: Comprehensive recipe database with nutritional analysis
- **Dietary Restriction Management**: Support for various dietary restrictions and allergies
- **Health Condition-Based Recommendations**: Personalized meal suggestions based on health conditions
- **Nutritional Goal Optimization**: Track and optimize nutritional goals
- **Ingredient Substitution**: Smart ingredient substitution suggestions
- **Dynamic Recipe Integration**: Real-time recipe generation and external API integration

### Technical Features
- **Vector Database**: Custom in-memory vector store with persistence
- **Embedding Models**: Lightweight bag-of-words embedding for semantic search
- **RAG Pipeline**: Context-aware generation with relevance scoring
- **Dynamic System Stats**: Real-time analytics and performance tracking
- **Personalized Nutrition Analysis**: User-specific nutritional summaries and charts

### Prerequisites
- Python 3.8+
- pip

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd newnervesparks
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

4. **Access the application**
Open your browser and go to `http://localhost:8501` (or the port shown in terminal)

## 📁 Project Structure

```
newnervesparks/
├── app.py                          # Main Streamlit application
├── requirements.txt                 # Python dependencies
├── README.md                       # Project documentation
├── data/
│   ├── recipes.json               # Recipe database
│   ├── nutritional_data.json      # Nutritional information
│   └── dietary_guidelines.json    # Dietary restrictions and guidelines
├── src/
│   ├── __init__.py
│   ├── data_processor.py          # Data processing utilities
│   ├── simple_embedding_model.py  # Lightweight embedding model
│   ├── simple_vector_store.py     # Custom vector database
│   ├── rag_pipeline.py            # RAG pipeline implementation
│   ├── dietary_analyzer.py        # Dietary restriction analysis
│   ├── substitution_engine.py     # Ingredient substitution logic
│   └── recipe_integration.py      # Dynamic recipe generation
├── tests/
│   ├── __init__.py
│   └── test_rag_pipeline.py
└── evaluation/
    └── ragas_evaluation.py        # RAGAS evaluation metrics
```

## 🎯 Usage

### 1. Recipe Search
- Search for recipes by ingredients, cuisine type, or dietary preferences
- Get personalized recommendations based on your health profile
- Dynamic recipe generation for expanded options

### 2. Dietary Restriction Management
- Set your dietary restrictions (vegetarian, vegan, gluten-free, etc.)
- Specify food allergies and intolerances
- Get filtered recommendations that respect your restrictions

### 3. Health Condition-Based Recommendations
- Input your health conditions (diabetes, heart disease, etc.)
- Receive meal suggestions optimized for your health needs
- Track nutritional goals and progress

### 4. Ingredient Substitution
- Get smart ingredient substitution suggestions
- Maintain nutritional value while accommodating restrictions
- Discover new ingredients and cooking methods

### 5. Nutrition Analysis
- Personalized nutritional summaries based on your profile
- Interactive charts showing nutritional distributions
- Goal tracking and optimization suggestions

### 6. Dynamic Integration
- Generate new recipes on-demand
- View system analytics and performance metrics
- Real-time statistics and coverage analysis

## 🔧 Technical Implementation

### RAG Pipeline
1. **Data Ingestion**: Process recipe database with nutritional information
2. **Chunking**: Intelligent text chunking for optimal retrieval
3. **Embedding**: Generate embeddings using lightweight bag-of-words model
4. **Vector Storage**: Store in custom in-memory vector store with persistence
5. **Retrieval**: Semantic search with relevance scoring
6. **Generation**: Context-aware response generation

### Dietary Analysis
- **Restriction Mapping**: Map dietary restrictions to recipe attributes
- **Conflict Resolution**: Handle conflicting dietary requirements
- **Nutritional Optimization**: Balance taste and health requirements

### Dynamic Features
- **Real-time Recipe Generation**: Create new recipes based on queries
- **System Analytics**: Live statistics and performance metrics
- **Personalized Nutrition**: User-specific nutritional analysis


## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request



