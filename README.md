# InsuranceModel: Insurance Premium Prediction

This project is a Flask-based web application that predicts insurance premiums using a neural network model trained on the [Medical Cost Personal Dataset](https://gist.githubusercontent.com/meperezcuello/82a9f1c1c473d6585e750ad2e3c05a41/raw/d42d226d0dd64e7f5395a0eec1b9190a10edbc03/Medical_Cost.csv). The app features a user-friendly frontend with dropdowns and input fields, a backend for real-time predictions, and is deployed on Vercel for scalability. Below is a detailed overview of the project’s development, structure, and deployment process.

---

## Project Overview

The goal was to create an end-to-end machine learning application that:
1. Trains a neural network regression model to predict insurance premiums based on features: `age`, `sex`, `bmi`, `children`, `smoker`, and `region`.
2. Provides a web interface for users to input data and receive predictions.
3. Deploys the app on Vercel as a serverless function for public access.

Key milestones included training the model, integrating it into Flask, optimizing performance, and resolving deployment challenges.

---

## Features

- **Input Features**: `age` (numeric), `sex` (male/female), `bmi` (numeric), `children` (numeric), `smoker` (yes/no), `region` (southwest/southeast/northwest/northeast).
- **Frontend**: A clean HTML form with dropdowns for categorical inputs (`sex`, `smoker`, `region`) and text fields for numeric inputs, styled with basic CSS.
- **Backend**: Flask app that preprocesses user input, predicts premiums using a pre-trained TensorFlow model, and displays results.
- **Preprocessing**: Uses `ColumnTransformer` with `MinMaxScaler` for numeric features and `OneHotEncoder` for categorical features.
- **Deployment**: Hosted on Vercel with a serverless architecture.

---

## Development Process

### 1. Model Training
The model was trained in a Jupyter Notebook (`01_Neural_NET_Regression.ipynb`) on Google Colab using TensorFlow and scikit-learn.

- **Dataset**: Loaded from a public gist (`Medical_Cost.csv`) with 1,338 rows and 7 columns (`age`, `sex`, `bmi`, `children`, `smoker`, `region`, `charges`).
- **Preprocessing**:
  - Split into features (`X`) and target (`y = charges`).
  - Applied `ColumnTransformer`:
    - `MinMaxScaler` on `age`, `bmi`, `children`.
    - `OneHotEncoder` on `sex`, `smoker`, `region`.
  - Split into training (80%) and test (20%) sets with `train_test_split`.
- **Model Architecture**:
  - A `Sequential` neural network with layers: `[Dense(100), Dense(10), Dense(1)]`.
  - Compiled with `mae` loss, `Adam` optimizer (learning rate 0.01), and `mae` metric.
  - Trained for 100 epochs with `tf.random.set_seed(42)` for reproducibility.
- **Evaluation**: Tested on `X_test_normal` with MAE as the metric.
- **Saving**: Initially saved as `insurance_model.pkl` with `pickle`, later switched to `insurance_model.h5` for better compatibility.

### 2. Initial Flask App
The first Flask app (`main.py`) was built to serve predictions locally:
- Loaded the dataset and refitted the `ColumnTransformer` on every startup.
- Used `render_template` to serve `index.html` with a form and display predictions.
- Encountered slow loading due to dataset reload and refitting.

### 3. Frontend Design
Created `templates/index.html`:
- Form with:
  - Numeric inputs: `age`, `bmi`, `children`.
  - Dropdowns: `sex` (male/female), `smoker` (yes/no), `region` (southwest/southeast/northwest/northeast).
- Basic CSS for layout and styling.
- Displays prediction or error messages dynamically.

### 4. Deployment Challenges on Vercel
Deployed to Vercel but faced several issues:
- **Missing Dependencies**: Initial `ModuleNotFoundError: No module named 'flask'` due to missing `requirements.txt`. Fixed by adding it.
- **Size Limit**: Local `venv` (250 MB) exceeded Vercel’s 250 MB free tier limit. Excluded it with `.gitignore`.
- **Version Mismatches**:
  - `tensorflow==2.17.0` with `numpy==2.0.2` caused `_ARRAY_API not found` due to model being trained with older NumPy (1.x).
  - Downgraded to `tensorflow==2.15.0`, but hit `ModuleNotFoundError: No module named 'tensorflow.compat'` due to model expecting a newer version.
  - Settled on `tensorflow==2.17.0` with `numpy==1.26.4` to match Colab’s likely environment.
- **Dependency Conflicts**: `tensorflow-macos==2.15.0` conflicted with newer `keras`, `ml-dtypes`, and `tensorboard`. Fixed by uninstalling conflicting versions and locking to compatible ones.

### 5. Performance Optimization
The app was slow due to redundant data loading and transformer fitting:
- **Solution**: Saved the fitted `ColumnTransformer` as `column_transformer.pkl` during training.
- **Updated `main.py`**:
  - Loaded `column_transformer.pkl` and `insurance_model.h5` once at startup.
  - Transformed user input directly without reloading the dataset or refitting.
- **Result**: Reduced request time from ~1-5 seconds to ~0.1-0.5 seconds.

---

## Final Project Structure
```
insurance_app/
├── main.py                  # Flask app with optimized backend
├── requirements.txt         # Dependencies
├── vercel.json              # Vercel config
├── insurance_model.h5       # Trained neural network model
├── column_transformer.pkl   # Pre-fitted ColumnTransformer
├── templates/
│   └── index.html          # Frontend HTML
├── .gitignore              # Excludes venv/
└── README.md               # This file
```

---

## Installation and Setup

### Prerequisites
- Python 3.9+
- Git
- Vercel CLI (`npm i -g vercel`)

### Local Setup
1. **Clone the Repository**:
   ```bash
   git clone <your-repo-url>
   cd insurance_app
   ```
2. **Create Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate     # Windows
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run Locally**:
   ```bash
   python main.py
   ```
   Visit `http://127.0.0.1:5000` in your browser.

### Deployment to Vercel
1. **Initialize Git** (if not already done):
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   ```
2. **Deploy**:
   ```bash
   vercel --prod
   ```
   Follow prompts to link your Vercel project.
3. **Access**: Visit the provided Vercel URL (e.g., `https://insurance-app.vercel.app`).

---

## Usage
1. Open the app in a browser.
2. Fill the form:
   - Enter `age`, `bmi`, and `children` as numbers.
   - Select `sex`, `smoker`, and `region` from dropdowns.
3. Click "Calculate Premium" to see the predicted insurance premium in dollars.

---

## Dependencies (`requirements.txt`)
```
Flask==3.1.0
Jinja2==3.1.6
Werkzeug==3.1.3
tensorflow==2.17.0
scikit-learn==1.6.1
pandas==2.2.3
numpy==1.26.4
```

---

## Challenges and Solutions
1. **Slow Loading**:
   - **Problem**: Dataset reload and `ColumnTransformer` fitting on every startup.
   - **Solution**: Saved the fitted transformer as `column_transformer.pkl` and loaded it once.
2. **Vercel Deployment Errors**:
   - **Problem**: `ModuleNotFoundError` and size limits.
   - **Solution**: Added `requirements.txt`, excluded `venv` with `.gitignore`.
3. **TensorFlow Version Mismatches**:
   - **Problem**: `_ARRAY_API not found` with NumPy 2.0 and `tensorflow.compat` errors.
   - **Solution**: Locked to `tensorflow==2.17.0` and `numpy==1.26.4`, considered retraining with HDF5.
4. **Dependency Conflicts**:
   - **Problem**: `tensorflow-macos==2.15.0` vs. newer sub-dependencies.
   - **Solution**: Uninstalled conflicting versions and aligned all dependencies.

---

## Future Improvements
- **Model Size**: Compress `insurance_model.h5` with `gzip` or convert to TensorFlow Lite for faster loading.
- **Frontend**: Enhance UI with Bootstrap or JavaScript validation.
- **Caching**: Add a caching layer (e.g., Redis) for frequent predictions.
- **API**: Expose a REST API endpoint for programmatic access.

---

## Acknowledgments
- Dataset: [Medical Cost Personal Dataset](https://gist.githubusercontent.com/meperezcuello/82a9f1c1c473d6585e750ad2e3c05a41/raw/d42d226d0dd64e7f5395a0eec1b9190a10edbc03/Medical_Cost.csv).
- Tools: TensorFlow, scikit-learn, Flask, Vercel.
- Guidance: Thanks to Grok (xAI) for troubleshooting assistance!
