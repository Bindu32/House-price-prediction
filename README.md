#  House Price Prediction

This project aims to predict house prices based on multiple real estate features such as square footage, location, number of bedrooms/bathrooms, and more.  
It was built as part of **TechMaven Project 1**, and follows a full machine learning pipeline from data preprocessing to model deployment.

---

##  Problem Statement

Given various attributes of houses (like location, number of bedrooms, square footage, etc.), predict the **price** of a house using machine learning techniques.

The goal is to build a **regression model** that can accurately estimate house prices using both categorical and numerical data.  
The solution includes:
- Data preprocessing  
- Visualization  
- Model building  
- Hyperparameter tuning  
- Evaluation  
- Final model saving for real-world use  

---

##  Tech Stack

- **Python**
- **Pandas**, **NumPy** – Data manipulation
- **Matplotlib**, **Seaborn** – Visualization
- **Scikit-learn** – Pipelines, models, preprocessing
- **XGBoost** – Final selected model
- **Joblib** – Model serialization

---

##  Workflow Overview

```mermaid
graph TD
A[Load Dataset] --> B[Data Cleaning & Feature Engineering]
B --> C[Exploratory Data Analysis]
C --> D[Train-Test Split]
D --> E[Model Training: Linear, RF, XGB]
E --> F[GridSearchCV Tuning]
F --> G[Model Evaluation]
G --> H[Model Saving & Prediction]


