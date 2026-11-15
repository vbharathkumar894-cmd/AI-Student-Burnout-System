# 🎓 AI Student Burnout Early Warning System

## Project Overview
The AI Student Burnout Early Warning System is a machine learning-based educational application designed to identify patterns associated with academic strain among students.

It analyzes study hours, sleep duration, attendance, breaks, self-reported stress and assignment workload to generate an illustrative risk category.

## Problem Statement
Students may experience academic strain due to excessive study hours, insufficient sleep, high workloads and stress. This project explores how machine learning can help identify patterns that may indicate a need for better study-life balance.

## Objectives
- Analyze student academic and lifestyle patterns.
- Identify low, medium and high academic strain risk categories.
- Train and compare machine learning models.
- Provide supportive suggestions.
- Build an interactive Streamlit dashboard.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit

## Machine Learning Algorithms
1. Decision Tree Classifier
2. Random Forest Classifier

## Input Features
- Study Hours
- Sleep Hours
- Attendance
- Breaks
- Stress Level
- Assignment Workload

## Project Modules
1. Dataset Generation
2. Exploratory Data Analysis
3. Machine Learning Model Training
4. Model Evaluation
5. Streamlit Dashboard
6. Academic Strain Risk Prediction

## Dataset
This project uses 1,000 synthetically generated student records. The target labels are generated using predefined demonstration rules.

## How to Run the Project

### 1. Clone the repository
```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the project folder
```bash
cd AI_Student_Burnout_System
```

### 3. Create a virtual environment
```bash
python3 -m venv venv
```

### 4. Activate the environment
```bash
source venv/bin/activate
```

### 5. Install dependencies
```bash
pip install -r requirements.txt
```

### 6. Generate the dataset
```bash
python3 dataset.py
```

### 7. Run EDA
```bash
python3 eda.py
```

### 8. Train the machine learning models
```bash
python3 model.py
```

### 9. Launch the application
```bash
streamlit run app.py
```

## Expected Outcome
An interactive dashboard that allows users to enter student details, view an illustrative academic strain risk category and receive supportive suggestions.

## Limitations
- The dataset is synthetic.
- Target labels are rule-generated.
- Model performance does not establish real-world burnout prediction capability.
- The application is not a medical or psychological diagnostic tool.

## Future Enhancements
- Use ethically collected real-world student data.
- Improve model validation.
- Add personalized wellness resources.
- Introduce longitudinal student trend analysis.

## Disclaimer
This project is developed for educational and demonstration purposes only.
