# Python Final Project: Health Expenditure and Malnutrition Death Rate (2021)

**Name:** Elena Comellas  
**Course:** Python Programming Mini-Project  
**Date:** 2025  

---

##  Project Summary

This project investigates the relationship between **healthcare expenditure** (as a % of GDP) and **malnutrition-related death rates**, using global data from 2021. Countries are categorized into **GDP tertiles** (Low, Medium, High) to analyze whether the impact of healthcare spending varies by income level.

The analysis includes:
- Data cleaning and merging across three public datasets
- GDP-based grouping
- Descriptive statistics and visualizations
- Group-specific linear regressions
- Breusch-Pagan tests for heteroskedasticity
- Economic interpretation of results

---

## Datasets Used

Download the following CSV files and place them in the same folder as the script:

- `death-rate-from-malnutrition-ghe.csv`  
- `total-healthcare-expenditure-gdp.csv`  
- `gdp-per-capita-worldbank.csv`  

All files can be found on **Our World in Data** or **World Bank Data** websites.

---

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/elecom301/Python_Final_Exam.git
cd Python_Final_Exam
