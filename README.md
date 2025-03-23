# 🛒 E-Commerce Transaction Purchasing Analysis

## 📌 Project Overview
This project aims to analyze the impact of **product categories** and **payment methods** on **purchase amounts** in e-commerce transactions.  
Using statistical analysis and machine learning models, I try to explore key factors influencing purchasing behavior.

## 🛠 Tools & Technologies Used
### **Programming & Libraries**
- **Python:**
  - `pandas`, `numpy` – Data manipulation and analysis
  - `seaborn`, `matplotlib` – Data visualization
  - `scipy.stats`, `statsmodels.api` – Statistical analysis
  - `sklearn.model_selection` – Train-test splitting
  - `sklearn.ensemble`, `sklearn.metrics` – Machine learning models & evaluation
- **Power BI** – Interactive dashboards and visualizations

## 🔍 Predictive Models Implemented
- **Logistic Regression** – Classifies purchase amounts into high or low categories
- **Linear Regression** – Predicts the exact purchase amount
- **Random Forest** – Improves prediction accuracy and handles non-linearity

## 📊 Overview of the Dataset

🔗 **Link to the dataset:** [E-Commerce Transactions Dataset](https://www.kaggle.com/datasets/smayanj/e-commerce-transactions-dataset?resource=download)

This dataset consists of **50,000 fictional e-commerce transaction records**, making it ideal for **data analysis, visualization, and machine learning experiments**. It contains information about **user demographics, product categories, purchase amounts, payment methods, and transaction dates**, allowing us to explore consumer behavior and sales trends.

### 📂 Dataset Columns:

| Column Name         | Description |
|---------------------|-------------|
| **Transaction_ID**  | Unique identifier for each transaction |
| **User_Name**       | Randomly generated user name |
| **Age**            | Age of the user (range: **18 to 70**) |
| **Country**        | Country where the transaction took place (**randomly chosen from 10 countries**) |
| **Product_Category** | Category of the purchased item (**e.g., Electronics, Clothing, Books**) |
| **Purchase_Amount** | Total amount spent on the transaction (**randomly generated between $5 and $1000**) |
| **Payment_Method**  | Method used for payment (**e.g., Credit Card, PayPal, UPI**) |
| **Transaction_Date** | Date of the purchase (**randomly selected within the past two years**) |

---

## 🚀 Next Steps: Project Implementation
I will now proceed with **data analysis, preprocessing, and predictive modeling** using various machine learning techniques.

## 📊 Processing and Statistical Analysis with Python

### 📥 Import Required Libraries:
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```
## 📥 Loading the Dataset  
To begin, we **load the main dataset** and a **supplemental dataset** containing country codes.  
This will allow us to add a new column (**country-code**) to the original dataset for potential later use.

```python
# Load the main dataset
df = pd.read_csv('ecommerce_transactions.csv')

# Load the supplemental dataset for country codes
country_code = pd.read_csv('https://raw.githubusercontent.com/lukes/ISO-3166-Countries-with-Regional-Codes/refs/heads/master/all/all.csv')
```
At this step, i also load a suplemental data table from github for adding a new column called country-code to the original data set for later use (if neccessary)

## 🔍 Examining the Dataset  

### 📝 Display basic information about the dataset:
```python
# Display dataset info
df.info()
```
Result:
RangeIndex: 50000 entries, 0 to 49999
Data columns (total 8 columns):
| #  | Column Name        | Non-Null Count | Data Type  |
|----|-------------------|---------------|-----------|
| 0  | Transaction_ID    | 50,000        | int64     |
| 1  | User_Name         | 50,000        | object    |
| 2  | Age              | 50,000        | int64     |
| 3  | Country          | 50,000        | object    |
| 4  | Product_Category | 50,000        | object    |
| 5  | Purchase_Amount  | 50,000        | float64   |
| 6  | Payment_Method   | 50,000        | object    |
| 7  | Transaction_Date | 50,000        | object    |

**Memory Usage:** 3.1+ MB  
**Data Types:**  
- **Integer (int64):** Transaction_ID, Age  
- **Float (float64):** Purchase_Amount  
- **Object (string):** User_Name, Country, Product_Category, Payment_Method, Transaction_Date

### 📝 Display the 10 sample rows of the dataset:
```python
df.sample(10)
```
Result:
| Transaction_ID | User_Name        | Age | Country    | Product_Category | Purchase_Amount | Payment_Method     | Transaction_Date |
|---------------|-----------------|-----|-----------|------------------|-----------------|-------------------|-----------------|
| 26213        | Emma Anderson    | 27  | Brazil    | Beauty           | 587.88          | Net Banking       | 2023-09-12      |
| 27196        | Liam Rodriguez   | 38  | UK        | Clothing         | 628.72          | Cash on Delivery  | 2025-01-12      |
| 35443        | Sophia Anderson  | 44  | UK        | Clothing         | 106.35          | Net Banking       | 2023-11-25      |
| 28672        | James Harris     | 50  | UK        | Books            | 198.89          | Cash on Delivery  | 2024-07-27      |
| 7354         | Sophia Allen     | 28  | Germany   | Toys             | 169.50          | Debit Card        | 2023-07-10      |
| 16395        | Liam Harris      | 64  | Australia | Beauty           | 937.62          | Credit Card       | 2023-06-11      |
| 8153         | Liam Rodriguez   | 67  | Canada    | Electronics       | 741.47          | UPI               | 2024-04-03      |
| 45796        | Isabella Lewis   | 65  | Brazil    | Grocery          | 423.74          | Cash on Delivery  | 2023-09-23      |
| 17513        | Isabella Harris  | 48  | Germany   | Books            | 487.19          | Net Banking       | 2025-01-20      |
| 28359        | Olivia Hall      | 68  | Canada    | Toys             | 346.41          | UPI               | 2025-01-31      |

Fortunately, there are no missing values in the dataset. However, for easier analysis, I need to convert User_Name from an object to a string and Transaction_Date from an object to a datetime format. Additionally, I will numerize the "Product_Category", "Payment_Method", and "Country" columns.

## Convert User_Name from an object to a string and Transaction_Date from an object to a datetime format
```python
df["User_Name"] = df["User_Name"].astype('str')
df["Transaction_Date"] = pd.to_datetime(df["Transaction_Date"], format="%Y-%m-%d")
```

## Numerize the "Product_Category", "Payment_Method" columns
```python
# One-hot encode "Product_Category" with 1s and 0s
df = df.join(pd.get_dummies(df["Product_Category"]).astype(int))

# One-hot encode "Payment_Method" with 1s and 0s
df = df.join(pd.get_dummies(df["Payment_Method"]).astype(int))

# Display sample rows to verify
df.sample(5)
```
Result:
| Transaction_ID | User_Name       | Age | Country | Product_Category | Purchase_Amount | Payment_Method  | Transaction_Date | Beauty | Books | ... | Grocery | Home & Kitchen | Sports | Toys | Cash on Delivery | Credit Card | Debit Card | Net Banking | PayPal | UPI |
|---------------|----------------|-----|---------|------------------|----------------|----------------|----------------|--------|-------|-----|---------|---------------|--------|------|------------------|-------------|------------|-------------|--------|-----|
| 2704          | Noah Walker     | 21  | Brazil  | Electronics      | 467.66         | Credit Card    | 2025-02-08     | 0      | 0     | ... | 0       | 0             | 0      | 0    | 0                | 1           | 0          | 0           | 0      | 0   |
| 2872          | Emma Harris     | 47  | France  | Grocery         | 648.53         | Debit Card     | 2024-02-05     | 0      | 0     | ... | 1       | 0             | 0      | 0    | 0                | 0           | 1          | 0           | 0      | 0   |
| 46592         | Oliver Rodriguez| 45  | France  | Toys            | 247.32         | PayPal         | 2024-08-15     | 0      | 0     | ... | 0       | 0             | 0      | 1    | 0                | 0           | 0          | 0           | 1      | 0   |
| 2209          | Sophia Hall     | 55  | Canada  | Electronics      | 551.26         | Credit Card    | 2023-11-12     | 0      | 0     | ... | 0       | 0             | 0      | 0    | 0                | 1           | 0          | 0           | 0      | 0   |
| 36536         | Liam Walker     | 47  | UK      | Books           | 329.76         | PayPal         | 2025-02-21     | 0      | 1     | ... | 0       | 0             | 0      | 0    | 0                | 0           | 0          | 0           | 1      | 0   |

At this step, I do not modify any existing values in the original dataset. Instead, I add new boolean columns (0 for False and 1 for True) to represent the features of the selected categorical columns.

## Numerize the "Country" columns



