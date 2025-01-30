# Life Expectancy Prediction using Machine Learning

## 📌 Project Overview
This project predicts **life expectancy** across different countries using various **machine learning models**, including **Linear Regression, Decision Tree, Random Forest, SVM, KNN, AdaBoost, and XGBoost**. It also applies **Principal Component Analysis (PCA)** to improve performance on large datasets.

---

## 📊 Features
✅ **Data Preprocessing** – Handling missing values, encoding categorical variables, and feature scaling.  
✅ **Exploratory Data Analysis (EDA)** – Visualizing trends and correlations in life expectancy data.  
✅ **Machine Learning Models** – Training multiple models for accurate predictions.  
✅ **PCA Transformation** – Reducing dimensionality to improve model performance on large datasets.  
✅ **Hyperparameter Tuning** – Optimizing model parameters using GridSearchCV.  
✅ **Model Comparison** – Evaluating models using RMSE, R² Score, and other performance metrics.  

---

## 📂 Repository Structure
```
📦 Life-Expectancy-Prediction
│── 📂 data
│   ├── life_expectancy.csv  # Dataset file
│── 📂 notebooks
│   ├── 01_data_preprocessing.ipynb  # Data cleaning and preprocessing
│   ├── 02_exploratory_analysis.ipynb  # Data visualization and EDA
│   ├── 03_model_training.ipynb  # Training different ML models
│   ├── 04_pca_analysis.ipynb  # PCA for dimensionality reduction
│   ├── 05_model_comparison.ipynb  # Comparing model performance
│── 📂 models
│   ├── trained_models.pkl  # Saved trained models
│── 📂 src
│   ├── data_loader.py  # Load and preprocess dataset
│   ├── model_trainer.py  # Train ML models
│   ├── pca_transform.py  # Apply PCA transformation
│   ├── evaluate.py  # Model evaluation metrics
│── 📂 results
│   ├── model_performance.csv  # Performance metrics of different models
│   ├── feature_importance.png  # Feature importance visualization
│── 📜 requirements.txt  # Required Python libraries
│── 📜 README.md  # Project documentation
│── 📜 main.py  # Main script to run the project
```

---

## 🛠 Installation
```bash
# Clone the repository
git clone https://github.com/your-username/Life-Expectancy-Prediction.git
cd Life-Expectancy-Prediction

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

---

## 🔄 Data Preprocessing
- Handles missing values using median imputation.
- Encodes categorical features (e.g., Country, Status).
- Scales numerical variables using StandardScaler.

---

## 🤖 Machine Learning Models
- **Linear Regression**
- **Decision Tree Regressor**
- **Random Forest Regressor**
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**
- **AdaBoost Regressor**
- **XGBoost Regressor**

Each model is trained and evaluated using performance metrics such as:
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **R² Score**

---

## 🎯 PCA Transformation
PCA is applied to reduce dimensionality, improving efficiency on large datasets. 

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X)
```

---

## 🚀 Running the Project
```bash
python main.py
```

---

## 📊 Model Performance Comparison
After training, models are compared based on evaluation metrics. The best-performing model is saved for future predictions.

---

## 📌 Contributors
- **Your Name** - [GitHub Profile](https://github.com/your-username)

Feel free to contribute to this project by submitting issues or pull requests!

---

## 📜 License
This project is open-source and available under the **MIT License**.
