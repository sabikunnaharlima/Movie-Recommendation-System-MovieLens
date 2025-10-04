# MovieLens Recommender System

A collaborative filtering recommender system built with scikit-learn's Non-negative Matrix Factorization (NMF) algorithm for movie recommendations.

## 🎬 Features

- **Collaborative Filtering**: Uses NMF to find latent factors in user-item interactions
- **Movie Recommendations**: Provides personalized movie recommendations for users
- **Model Persistence**: Save and load trained models using joblib
- **Standalone Usage**: Use saved models independently without retraining
- **Complete Dataset**: Includes 27 movies with proper titles and metadata

## 📊 Dataset

The system uses a sample MovieLens dataset containing:
- **27 movies** with titles and genres
- **30 ratings** from 3 users
- **Rating scale**: 0.5 to 5.0 stars

### Sample Movies
- Toy Story (1995)
- Jumanji (1995)
- Heat (1995)
- The Usual Suspects (1995)
- Star Wars: Episode IV - A New Hope (1977)
- The Lion King (1994)
- The Shawshank Redemption (1994)
- Forrest Gump (1994)
- The Matrix (1999)
- And more...

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install pandas scikit-learn numpy joblib
```

### 2. Run the Notebook
```bash
jupyter notebook recommender.ipynb
```

### 3. Use the Saved Model
```python
import joblib
import numpy as np

# Load the saved model
model_data = joblib.load('recommender_model.pkl')

# Extract components
user_factors = model_data['user_factors']
item_factors = model_data['item_factors']
user_item_matrix = model_data['user_item_matrix']
movies = model_data['movies']
ratings = model_data['ratings']

# Generate recommendations for a user
def get_recommendations(user_id, num_recommendations=5):
    # Implementation details in the notebook
    pass
```

## 📁 Project Structure

```
DataSynthis_ML_JobTask/
├── recommender.ipynb          # Main Jupyter notebook
├── recommender_model.pkl      # Trained model file
├── data/
│   └── ml-latest-small/
│       ├── movies.csv         # Movie metadata
│       ├── ratings.csv        # User ratings
│       ├── tags.csv           # User tags
│       └── README.html        # Dataset documentation
├── .gitignore                 # Git ignore file
└── README.md                  # This file
```

## 🔧 Algorithm Details

### Non-negative Matrix Factorization (NMF)
- **Purpose**: Decompose user-item rating matrix into user and item factor matrices
- **Components**: 10 latent factors
- **Advantages**: 
  - Handles sparse data well
  - Provides interpretable factors
  - No negative values (realistic for ratings)

### Recommendation Process
1. **Matrix Creation**: Build user-item rating matrix
2. **Factorization**: Apply NMF to get user and item factors
3. **Prediction**: Calculate predicted ratings for unrated items
4. **Ranking**: Sort by predicted ratings and return top N recommendations

## 📈 Model Performance

- **Model Type**: sklearn.decomposition._nmf.NMF
- **User Factors Shape**: (3, 10) - 3 users, 10 components
- **Item Factors Shape**: (10, 20) - 10 components, 20 movies
- **Model Size**: ~9KB

## 🎯 Example Recommendations

### User 1 Recommendations:
1. Jumanji (1995)
2. Waiting to Exhale (1995)
3. Sabrina (1995)

### User 2 Recommendations:
1. Jumanji (1995)
2. Waiting to Exhale (1995)
3. Sabrina (1995)

### User 3 Recommendations:
1. Toy Story (1995)
2. Grumpier Old Men (1995)
3. Heat (1995)

## 🛠️ Development

### Adding New Users
To add new users to the system:
1. Add their ratings to `data/ml-latest-small/ratings.csv`
2. Retrain the model
3. Save the updated model

### Adding New Movies
To add new movies:
1. Add movie metadata to `data/ml-latest-small/movies.csv`
2. Add ratings for the new movies
3. Retrain the model

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note**: This is a demonstration project using a small sample dataset. For production use, consider using the full MovieLens dataset and implementing additional features like content-based filtering, hybrid approaches, and evaluation metrics.
