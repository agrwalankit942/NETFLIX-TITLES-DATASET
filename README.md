🎬 Netflix Content Analyzer using Machine Learning
📌 Project Overview

This project performs Exploratory Data Analysis (EDA) and Machine Learning classification on the Netflix Titles dataset.
The goal is to analyze Netflix content trends and predict whether a title is a Movie or a TV Show using two machine learning models.

The project also includes an interactive Streamlit web application for real-time predictions.

🎯 Objectives

Clean and preprocess Netflix titles data

Perform exploratory data analysis and visualizations

Apply Machine Learning algorithms:

Decision Tree (structured data)

Naive Bayes (text data)

Save and reload trained models

Deploy models using an interactive Streamlit UI

🧹 Data Cleaning & Feature Engineering

Converted date_added to datetime format

Filled missing values in:

director

cast

country

rating

Created new features:

is_movie (target variable)

release_decade

year_added

Extracted numeric values from duration

Applied TF-IDF vectorization on description

📊 Exploratory Data Analysis (EDA)

Pie chart: Movie vs TV Show distribution

Line plot: Content released per year

Bar chart: Top 10 genres

Heatmap: Country vs content volume

🤖 Machine Learning Models
1️⃣ Decision Tree Classifier

Type: Supervised Learning

Input Features:

Release year

Duration (minutes)

Target: Movie (1) or TV Show (0)

Advantage: Easy to interpret

2️⃣ Naive Bayes Classifier

Type: Supervised Learning (NLP)

Input Feature: Description text

Technique: TF-IDF Vectorization

Advantage: Performs well on text data

🎨 Streamlit Web Application

The Streamlit app allows users to:

Choose between Decision Tree and Naive Bayes models

Enter structured data or text descriptions

Get real-time predictions with a clean UI
