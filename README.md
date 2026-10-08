# Smart Retail Analytics

## Project Overview

Smart Retail Analytics is an end-to-end data science project designed to analyze retail sales data, extract business insights, and predict high-value sales transactions.

The project combines data engineering, exploratory data analysis, statistical analysis, feature engineering, machine learning, and interactive visualization into a complete data science pipeline.

The objective is to help retail businesses better understand their sales performance and identify high-value transactions that may require particular attention.

---

## Problem Statement

Retail businesses generate large amounts of sales data, but raw transaction data does not directly provide actionable business insights.

The main challenges addressed by this project are:

- Understanding sales and revenue performance
- Identifying the best-performing products and categories
- Analyzing sales trends over time
- Understanding the distribution of transaction values
- Identifying factors associated with high-value sales
- Building a machine learning model capable of classifying high-value transactions

These insights can support better business decisions related to product performance, sales monitoring, and commercial strategy.

---

## Target Users

This project is designed for:

- Store managers
- Business owners
- Retail analysts
- Data analysts
- Data scientists

The dashboard provides both business-level insights and machine learning predictions.

---

## Project Objectives

The main objectives are to:

- Analyze retail sales performance
- Identify revenue drivers
- Explore product and category performance
- Analyze daily, weekly, and monthly sales trends
- Perform statistical analysis of sales transactions
- Engineer meaningful features for machine learning
- Build and compare classification models
- Predict whether a transaction is a high-value sale
- Provide an interactive dashboard for business users

---

## Dataset

The project uses product information and sales transactions stored in a PostgreSQL database.

Product data is collected from the Fake Store API:

- Product ID
- Product name
- Category
- Price

Sales transactions contain:

- Sale ID
- Product ID
- Quantity
- Sale date
- Total transaction value

The sales transactions are synthetically generated for project and modeling purposes.

> Note: The current dataset contains 409 transactions and approximately one month of sales data. Therefore, the machine learning results are considered exploratory and should be validated on a larger real-world dataset before production use.

---

## Data Pipeline

The project follows an end-to-end data science pipeline:

```text
API / Sales Data
       ↓
PostgreSQL Database
       ↓
Data Cleaning
       ↓
Data Transformation
       ↓
Exploratory Data Analysis
       ↓
Statistical Analysis
       ↓
Feature Engineering
       ↓
Train / Test Split
       ↓
Model Training
       ↓
Cross-Validation
       ↓
Hyperparameter Tuning
       ↓
Model Evaluation
       ↓
Prediction
       ↓
Streamlit Dashboard