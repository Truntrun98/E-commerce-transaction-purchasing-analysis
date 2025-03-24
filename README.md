# 🛒 E-Commerce Transaction Purchasing Analysis

## 📌 Project Overview
This project aims to analyze the impact of **product categories** and **payment methods**, **age** on **purchase amounts** in e-commerce transactions.  
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
At the beginning, I mentioned the supplemental dataset. Initially, I considered adding new boolean columns for each unique country, but this would result in too many columns, making the dataset harder to interpret. Instead, I decided to convert country names into their respective international codes. To achieve this, I used a CSV file from GitHub containing country codes and merged its "country-code" column with the original dataset, matching country names from the supplemental dataset with the "Country" column in the original dataset.
```python
# Merge df with country_code to add 'country-code' next to 'Country'
df_merged = df.merge(country_code[['name', 'country-code']],
                      left_on='Country',
                      right_on='name',
                      how='left')

# Drop the redundant 'name' column from country_code
df_merged.drop(columns=['name'], inplace=True)

df = df_merged

df.info()
```
Result:
RangeIndex: 50000 entries, 0 to 49999
Data columns (total 23 columns):
| #  | Column            | Non-Null Count | Dtype   |
|----|------------------|--------------|--------|
| 0  | Transaction_ID    | 50000 non-null | int64  |
| 1  | User_Name         | 50000 non-null | object |
| 2  | Age               | 50000 non-null | int64  |
| 3  | Country           | 50000 non-null | object |
| 4  | Product_Category  | 50000 non-null | object |
| 5  | Purchase_Amount   | 50000 non-null | float64 |
| 6  | Payment_Method    | 50000 non-null | object |
| 7  | Transaction_Date  | 50000 non-null | object |
| 8  | Beauty            | 50000 non-null | int64  |
| 9  | Books             | 50000 non-null | int64  |
| 10 | Clothing          | 50000 non-null | int64  |
| 11 | Electronics       | 50000 non-null | int64  |
| 12 | Grocery           | 50000 non-null | int64  |
| 13 | Home & Kitchen    | 50000 non-null | int64  |
| 14 | Sports            | 50000 non-null | int64  |
| 15 | Toys              | 50000 non-null | int64  |
| 16 | Cash on Delivery  | 50000 non-null | int64  |
| 17 | Credit Card       | 50000 non-null | int64  |
| 18 | Debit Card        | 50000 non-null | int64  |
| 19 | Net Banking       | 50000 non-null | int64  |
| 20 | PayPal            | 50000 non-null | int64  |
| 21 | UPI               | 50000 non-null | int64  |
| 22 | country-code      | 40070 non-null | float64 |

I discovered that there are missing values in the country-code column. To address this issue, I need to identify which countries are missing and investigate the cause of this discrepancy. Once identified, I will take appropriate steps to fix the problem.
```python
# Display rows where "country-code" is null
df_null_countries = df[df["country-code"].isna()]

# Print unique countries
df_null_countries["Country"].unique()
```
Result:
array(['USA', 'UK'], dtype=object)

It turns out that the missing values correspond to specific countries. This may be due to the main dataset using shortened country names. To verify this, I will check if these countries exist in the supplemental dataset by searching for values that contain "United States" or "United Kingdom."
```python
# Filter rows where the "name" column contains 'United States' or 'United Kingdom' (case-insensitive)
df_missing_countries = country_code[country_code["name"].str.contains("United States|United Kingdom", case=False, na=False)]

# Print the result
df_missing_countries
```
Result: 
| #   | Name                                           | Alpha-2 | Alpha-3 | Country Code | ISO 3166-2   | Region   | Sub-Region       | Intermediate Region | Region Code | Sub-Region Code | Intermediate Region Code |
|-----|-----------------------------------------------|---------|---------|--------------|--------------|----------|------------------|---------------------|-------------|----------------|-------------------------|
| 234 | United Kingdom of Great Britain and Northern Ireland | GB      | GBR     | 826          | ISO 3166-2:GB | Europe   | Northern Europe  | NaN                 | 150.0       | 154.0          | NaN                     |
| 235 | United States of America                     | US      | USA     | 840          | ISO 3166-2:US | Americas | Northern America | NaN                 | 19.0        | 21.0           | NaN                     |
| 236 | United States Minor Outlying Islands        | UM      | UMI     | 581          | ISO 3166-2:UM | Oceania  | Micronesia       | NaN                 | 9.0         | 57.0           | NaN                     |

I identified the issue in the supplemental dataset, which caused a mismatch with the main dataset. To resolve this, I decided to correct the country names in the supplemental dataset and then rematch them with the main dataset.
```python
# Define the mapping of old names to new names
country_name_fix = {
    "United States of America": "USA",
    "United Kingdom of Great Britain and Northern Ireland": "UK"
}

# Apply the changes to the "name" column
country_code["name"] = country_code["name"].replace(country_name_fix)
```
Rematch
```python
# Merge df with country_code to add 'country-code' next to 'Country'
df_merged = df.merge(country_code[['name', 'country-code']],
                      left_on='Country',
                      right_on='name',
                      how='left',
                      suffixes=('', '_new'))  # Add suffix to avoid conflicts

# If "country-code_new" exists, replace the old "country-code" with it
if "country-code_new" in df_merged.columns:
    df_merged["country-code"] = df_merged["country-code_new"]
    df_merged.drop(columns=["country-code_new"], inplace=True)  # Drop temporary column

# Drop the redundant 'name' column from country_code
df_merged.drop(columns=['name'], inplace=True)

# Update df with the merged result
df = df_merged

# Display dataframe info
df.info()
```
Result:
RangeIndex: 50000 entries, 0 to 49999
Data columns (total 23 columns):
| #  | Column            | Non-Null Count | Dtype   |
|----|------------------|---------------|--------|
| 0  | Transaction_ID   | 50000 non-null | int64  |
| 1  | User_Name        | 50000 non-null | object |
| 2  | Age             | 50000 non-null | int64  |
| 3  | Country         | 50000 non-null | object |
| 4  | Product_Category | 50000 non-null | object |
| 5  | Purchase_Amount  | 50000 non-null | float64 |
| 6  | Payment_Method  | 50000 non-null | object |
| 7  | Transaction_Date | 50000 non-null | object |
| 8  | Beauty          | 50000 non-null | int64  |
| 9  | Books           | 50000 non-null | int64  |
| 10 | Clothing        | 50000 non-null | int64  |
| 11 | Electronics     | 50000 non-null | int64  |
| 12 | Grocery         | 50000 non-null | int64  |
| 13 | Home & Kitchen  | 50000 non-null | int64  |
| 14 | Sports         | 50000 non-null | int64  |
| 15 | Toys           | 50000 non-null | int64  |
| 16 | Cash on Delivery | 50000 non-null | int64  |
| 17 | Credit Card     | 50000 non-null | int64  |
| 18 | Debit Card      | 50000 non-null | int64  |
| 19 | Net Banking     | 50000 non-null | int64  |
| 20 | PayPal         | 50000 non-null | int64  |
| 21 | UPI            | 50000 non-null | int64  |
| 22 | country-code   | 50000 non-null | int64  |

Finally, I have resolved the issue with missing values. Now, the dataset is fully prepared for further analysis. 🚀
# Correlation table and map
I selected all the numeric columns for analysis, excluding the country-code column. Although country-code is numerical, its values do not have a meaningful order or impact on correlations.
## Metrix table
```python
 df[[
    "Beauty", "Books", "Clothing", "Electronics", "Grocery", "Home & Kitchen",
    "Sports", "Toys", "Cash on Delivery", "Credit Card", "Debit Card",
    "Net Banking", "PayPal", "UPI", "Purchase_Amount", "Age"
]].corr()
```
Result:
|                     | Beauty   | Books    | Clothing | Electronics | Grocery  | Home & Kitchen | Sports   | Toys     | Cash on Delivery | Credit Card | Debit Card | Net Banking | PayPal   | UPI      | Purchase_Amount | Age      |
|---------------------|---------|---------|---------|-------------|---------|--------------|---------|---------|-----------------|------------|-----------|------------|---------|---------|----------------|---------|
| **Beauty**         | 1.000000 | -0.140601 | -0.140228 | -0.141460   | -0.140112 | -0.140035     | -0.141358 | -0.142381 | 0.002333         | -0.006852  | 0.002276   | -0.003003  | 0.005546 | -0.000319 | 0.000148        | -0.000986 |
| **Books**          | -0.140601 | 1.000000 | -0.142556 | -0.143809   | -0.142439 | -0.142360     | -0.143705 | -0.144746 | -0.000606        | 0.004670   | 0.000830   | 0.000451   | -0.007614 | 0.002234  | 0.007520        | -0.004914 |
| **Clothing**       | -0.140228 | -0.142556 | 1.000000 | -0.143428   | -0.142061 | -0.141983     | -0.143324 | -0.144362 | 0.003581         | -0.006254  | 0.000807   | 0.005161   | -0.000320 | -0.002941 | 0.008363        | -0.002966 |
| **Electronics**    | -0.141460 | -0.143809 | -0.143428 | 1.000000   | -0.143310 | -0.143231     | -0.144584 | -0.145631 | -0.000974        | -0.005397  | -0.002432  | -0.003124  | 0.007978  | 0.003931  | -0.009662       | 0.005726  |
| **Grocery**        | -0.140112 | -0.142439 | -0.142061 | -0.143310   | 1.000000 | -0.141865     | -0.143206 | -0.144242 | 0.003020         | 0.001477   | 0.004139   | 0.001962   | -0.002201 | -0.008351 | -0.000753       | 0.004043  |
| **Home & Kitchen** | -0.140035 | -0.142360 | -0.141983 | -0.143231   | -0.141865 | 1.000000     | -0.143127 | -0.144163 | 0.001403         | 0.003106   | -0.005612  | 0.000320   | 0.004005  | -0.003180 | -0.003211       | -0.002887 |
| **Sports**         | -0.141358 | -0.143705 | -0.143324 | -0.144584   | -0.143206 | -0.143127     | 1.000000 | -0.145525 | -0.010245        | 0.006624   | -0.003024  | -0.001447  | -0.004296 | 0.012336  | 0.004075        | -0.001552 |
| **Toys**           | -0.142381 | -0.144746 | -0.144362 | -0.145631   | -0.144242 | -0.144163     | -0.145525 | 1.000000 | 0.001567         | 0.002518   | 0.003034   | -0.000318  | -0.003014 | -0.003783 | -0.006384       | 0.003466  |
| **Cash on Delivery** | 0.002333  | -0.000606 | 0.003581  | -0.000974   | 0.003020  | 0.001403      | -0.010245 | 0.001567  | 1.000000         | -0.201109  | -0.201762  | -0.199132  | -0.200238 | -0.203528 | 0.006080        | -0.005431 |
| **Credit Card**    | -0.006852 | 0.004670  | -0.006254 | -0.005397   | 0.001477  | 0.003106      | 0.006624  | 0.002518  | -0.201109        | 1.000000   | -0.199975  | -0.197369  | -0.198465  | -0.201726 | -0.005758       | -0.000274 |
| **Debit Card**     | 0.002276  | 0.000830  | 0.000807  | -0.002432   | 0.004139  | -0.005612     | -0.003024 | 0.003034  | -0.201762        | -0.199975  | 1.000000   | -0.198009  | -0.199109  | -0.202380 | 0.005776        | -0.004817 |
| **Net Banking**    | -0.003003 | 0.000451  | 0.005161  | -0.003124   | 0.001962  | 0.000320      | -0.001447 | -0.000318 | -0.199132        | -0.197369  | -0.198009  | 1.000000   | -0.196514  | -0.199743 | -0.009308       | 0.007910  |
| **PayPal**         | 0.005546  | -0.007614 | -0.000320 | 0.007978    | -0.002201 | 0.004005      | -0.004296 | -0.003014 | -0.200238        | -0.198465  | -0.199109  | -0.196514  | 1.000000  | -0.200852 | 0.004223        | 0.000047  |
| **UPI**            | -0.000319 | 0.002234  | -0.002941 | 0.003931    | -0.008351 | -0.003180     | 0.012336  | -0.003783 | -0.203528        | -0.201726  | -0.202380  | -0.199743  | -0.200852 | 1.000000  | -0.001103       | 0.002639  |
| **Purchase_Amount** | 0.000148  | 0.007520  | 0.008363  | -0.009662   | -0.000753 | -0.003211     | 0.004075  | -0.006384 | 0.006080         | -0.005758  | 0.005776   | -0.009308  | 0.004223  | -0.001103 | 1.000000        | -0.003585 |
| **Age**            | -0.000986 | -0.004914 | -0.002966 | 0.005726    | 0.004043  | -0.002887     | -0.001552 | 0.003466  | -0.005431        | -0.000274  | -0.004817  | 0.007910   | 0.000047  | 0.002639  | -0.003585       | 1.000000  |

## Map
```python
# Select only the relevant numeric columns for correlation
correlation_matrix = df[[
    "Beauty", "Books", "Clothing", "Electronics", "Grocery", "Home & Kitchen",
    "Sports", "Toys", "Cash on Delivery", "Credit Card", "Debit Card",
    "Net Banking", "PayPal", "UPI", "Purchase_Amount", "Age"
]].corr()

# Set figure size
plt.figure(figsize=(10, 8))

# Draw heatmap with warm colors
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, linecolor="gray")

# Set title
plt.title("Correlation Heatmap", fontsize=14)

# Show the plot
plt.show()
```
Result:
![image](https://github.com/user-attachments/assets/7ac383b3-d4c2-41dc-9cd3-c2b4f5b79af2)

I conducted an analysis using Purchase_Amount as the primary factor. By examining its correlations with other variables, I identified several insights. While most features showed weak correlations, there were slight positive associations with Clothing, Books, and Sports, suggesting that customers spending in these categories tend to have marginally higher purchase amounts. Conversely, Electronics and Toys exhibited weak negative correlations, indicating that purchases in these categories might slightly lower the total spending. Additionally, payment methods like Cash on Delivery and Debit Card showed a slight positive correlation with spending, whereas Credit Card and Net Banking had weak negative correlations. Overall, no single feature had a strong influence on Purchase_Amount, highlighting the need for further investigation into other potential factors affecting customer spending behavior.
| **Feature**          | **Correlation with Purchase_Amount** | **Interpretation** |
|----------------------|--------------------------------------|--------------------|
| **Books**           | 0.007520  | Slight positive correlation; users buying books may spend slightly more. |
| **Clothing**        | 0.008363  | Slight positive correlation; clothing purchases are associated with slightly higher spending. |
| **Grocery**         | -0.000753 | Almost no correlation; grocery spending does not significantly impact total purchase amount. |
| **Electronics**     | -0.009662 | Weak negative correlation; electronic purchases slightly lower total spending. |
| **Sports**         | 0.004075  | Weak positive correlation; sports-related purchases slightly increase spending. |
| **Toys**           | -0.006384 | Weak negative correlation; toy purchases slightly decrease total spending. |
| **Cash on Delivery** | 0.006080  | Weak positive correlation; Cash on Delivery payments may be linked to slightly higher spending. |
| **Credit Card**     | -0.005758 | Weak negative correlation; credit card users tend to spend slightly less. |
| **Debit Card**      | 0.005776  | Weak positive correlation; debit card users may spend slightly more. |
| **Net Banking**     | -0.009308 | Weak negative correlation; users paying via net banking may spend slightly less. |
| **PayPal**         | 0.004223  | Weak positive correlation; PayPal payments are associated with slightly higher spending. |
| **UPI**            | -0.001103 | No significant correlation. |
| **Age**            | -0.003585 | Almost no correlation; age does not significantly impact purchase amount. |

Unfortunately, the results showed very weak or no correlation, which was not in line with my expectations. Given this, I will now proceed to testing predictive models to explore whether machine learning can uncover hidden patterns in the data that simple correlation analysis could not detect.
# Logistic Regression model testing
```python
# Define features (X) and target variable (y)
X = df[[  
    "Beauty", "Books", "Clothing", "Electronics", "Grocery", "Home & Kitchen",
    "Sports", "Toys", "Cash on Delivery", "Credit Card", "Debit Card",
    "Net Banking", "PayPal", "UPI", "Age"
]]

# Convert Purchase_Amount into a binary variable (above median = 1, below median = 0)
y = (df["Purchase_Amount"] > df["Purchase_Amount"].median()).astype(int)

# Split the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Add a constant for the intercept in training and testing sets
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

# Fit the logistic regression model using training data
model = sm.Logit(y_train, X_train).fit()

# Display the regression results
print(model.summary())
```
Result:
![image](https://github.com/user-attachments/assets/99d4c1b9-9422-4a3b-b45e-a1f7e6d4dedb)
# Interpreting the result
### Model Fit & Significance  

- **Pseudo R-squared**: 0.0001963 → This is very low, meaning the model explains almost none of the variability in `Purchase_Amount`.  
- **Log-Likelihood**: -27720.  
- **LLR p-value**: 0.6204 → A high p-value suggests that the overall model is not statistically significant.  

### Coefficients & Statistical Significance  

- The coefficients represent the **log-odds change** in the probability of `Purchase_Amount` increasing due to each variable.  
- **Extremely large standard errors** (e.g., `6.29e+05`) indicate possible **multicollinearity** or **scaling issues**.  
- **P-values of 1.000** for most predictors mean they are **not statistically significant**.  
- Some payment methods (**Cash on Delivery, Credit Card, etc.**) have missing standard errors (`NaN`), indicating issues in the data or model specification.  

### Key Findings  

- **No strong predictors**: None of the independent variables significantly predict `Purchase_Amount`.  
- **Age has a coefficient of 0.0001** with a high p-value (0.847), meaning it has **no meaningful effect**.  
- **Large standard errors** suggest **data scaling issues** or **high correlation between predictors**.

### Conclusion:
The dataset did not perform well with the Logistic Regression model for predicting purchase amount based on the given factors. As a result, I have decided to move forward with testing the Linear Regression model to assess its suitability for this dataset.

# Logistic Regression model testing
```python
# Define dependent variable (target)
y = df["Purchase_Amount"]

# Define independent variables (predictors)
X = df[[
    "Beauty", "Books", "Clothing", "Electronics", "Grocery", "Home & Kitchen",
    "Sports", "Toys", "Cash on Delivery", "Credit Card", "Debit Card",
    "Net Banking", "PayPal", "UPI", "Age"
]]

# Add a constant for the intercept
X = sm.add_constant(X)

# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the OLS regression model
model = sm.OLS(y_train, X_train).fit()

# Print the summary
print(model.summary())
```
Result:
![image](https://github.com/user-attachments/assets/4da343b1-821e-4eae-93ff-3269bbaa05e7)
# Interpreting the result
## Model Fit & Significance

- **R-squared: 0.000** → The model explains virtually none of the variance in `Purchase_Amount`, meaning that the independent variables do not effectively predict purchase amount.  
- **Adj. R-squared: 0.000** → Adjusted for the number of predictors, the explanatory power remains insignificant.  
- **F-statistic: 1.454** (p-value = **0.126**) → The overall model is not statistically significant, meaning the predictors collectively do not explain variations in purchase amount.  

## Coefficients & Statistical Significance

### Intercept (`const`):  
- **393.0025** → The baseline purchase amount when all other variables are zero.  

### Product Categories (Beauty, Books, Clothing, etc.):  
- All have **positive coefficients**, suggesting that purchasing in these categories slightly increases total purchase amount.  
- The coefficients range from **42.0158 (Electronics)** to **55.0438 (Clothing)**.  
- All p-values (**<0.001**) indicate statistical significance.  

### Payment Methods (Cash on Delivery, Credit Card, etc.):  
- All payment methods have **positive coefficients**, implying that customers using these payment methods tend to spend slightly more.  
- **Cash on Delivery (71.3612)** and **PayPal (69.1386)** have the highest positive impact.  
- All p-values (**<0.001**) indicate statistical significance.  

### Age:  
- **Coefficient: -0.0747**, suggesting that age has almost no impact on purchase amount.  
- **p-value = 0.427** → Not statistically significant.  

### Key Findings

- **Weak Model Performance**: The **R-squared value is 0**, meaning the model does not explain variations in purchase amount.  
- **Statistically Significant Coefficients**: Many independent variables are statistically significant (**p < 0.001**), but their effect sizes are small.  
- **Age Has No Impact**: The coefficient is close to zero and not statistically significant.  
- **Multicollinearity Risk**: The **Condition Number (9.05e+16)** is extremely high, suggesting strong multicollinearity, which may affect coefficient estimates.  

### Conclusion

This linear regression model does not effectively predict `Purchase_Amount`, as indicated by the near-zero **R-squared**. While product categories and payment methods show statistical significance, their impact is minimal, and **multicollinearity is a major concern**. Further model tuning or alternative approaches may be necessary.

# Random Forest model testing
```python
# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf.fit(X_train, y_train)

# Predict on test set
y_pred = rf.predict(X_test)

# Evaluate performance
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R-squared (R²):", r2_score(y_test, y_pred))
```
Result:
Mean Absolute Error (MAE): 0.5004856277687583
Mean Squared Error (MSE): 0.2676863389512165
R-squared (R²): -0.07093765336990554

# Interpreting the result:
## Model Performance Metrics (Random Forest)

### 1. Mean Absolute Error (MAE): **0.5005**
- On average, the model’s predictions deviate by **0.5005 units** from the actual **purchase amount**.
- A lower MAE indicates better accuracy in predicting **purchase amount**, but this value suggests the model may not be very precise.

### 2. Mean Squared Error (MSE): **0.2677**
- MSE measures the average squared difference between actual and predicted **purchase amounts**.
- Since errors are squared, larger mistakes have a greater impact.
- A lower MSE is preferred, but this value suggests that the model is struggling to predict **purchase amount** accurately.

### 3. R-squared (R²): **-0.0709**
- **This is a major concern.** A negative R² means the model performs **worse than simply predicting the average purchase amount for every observation**.
- Ideally, R² should be close to **1** (which means a strong ability to explain variations in **purchase amount**).
- A negative R² suggests that the **Random Forest model does not effectively capture patterns in purchase amount based on the given features**.
### Conclusion

The Random Forest model does not effectively predict **purchase amount** based on the given features. Key takeaways:

- The **negative R² (-0.0709)** indicates that the model performs **worse than a simple average-based prediction**, meaning it fails to capture meaningful patterns in the data.
- The **MAE (0.5005) and MSE (0.2677)** suggest that predictions are not very precise, with significant errors.
- Despite Random Forest being a powerful algorithm, its poor performance here may be due to:
  - **Weak relationships between features and purchase amount**.
  - **Feature selection issues** (e.g., irrelevant or redundant features).
  - **Lack of enough meaningful data patterns** that the model can learn from.

# General conclusion:
After applying three different modeling methods to predict purchase amount, the results were not promising. Given the lack of strong predictive power, I have decided to take the main raw dataset into Power BI for visualization. This will allow me to analyze trends, uncover patterns, and make predictions based on historical records.

# Power BI analysis
At first glance, the distribution of Payment Method and Product Category based on Purchase Amount appears quite similar, making it difficult to extract meaningful insights from their differences.
![image](https://github.com/user-attachments/assets/d8a6d0ae-322a-4145-8ea2-931b1fb3e031)
**Fig 1: Purchase by payment method**

![image](https://github.com/user-attachments/assets/7ba9c58c-2776-462d-8974-5cdd5f77de95)
**Fig 2: Purchase by category**


However, after segmenting customers into age ranges, I found that the majority of revenue comes from the 41–65 age group.
![image](https://github.com/user-attachments/assets/ce25716f-885f-4a95-8738-509ec4f545d8)
**Fig 3: Purchase by age range**


Moreover, after diving deeper into the average order value by product category, I found that Books and Home & Kitchen generated the highest average revenue in 2025. Additionally, Home & Kitchen had the highest growth rate in average order value over the past three years.
![image](https://github.com/user-attachments/assets/e2ceb537-98e2-40db-a81a-94f94e44feb3)
**Fig 4: Average order value by category**


For individuals aged 41–65, the Books category stands out as the leading preference, showcasing the fastest growth rate among all segments. Interestingly, Toys emerge as the second most appealing category for this demographic, maintaining steady growth in average order value over time.
![image](https://github.com/user-attachments/assets/5f3852f1-f747-4bed-a1b6-500b89458e87)
**Fig 5: Average order value by category in the group age of 41 - 65**


For the 26–40 age group, most categories have shown no significant change over time, except for Sports, which has declined dramatically over the past three years.
![image](https://github.com/user-attachments/assets/2e15089a-a996-48e3-9a34-2733060d83b1)
**Fig 6: Average order value by category in the group age of 26 - 40**

In contrast to the 26–40 age group, young consumers aged 18–25 show a growing trend in purchasing Sports, Clothing, Electronics, and Beauty products, while other categories have declined in average order value.
![image](https://github.com/user-attachments/assets/2b024e74-df08-47b4-bc6c-a03e33c63041)
**Fig 7: Average order value by category in the group age of 18 - 25**

In the 65+ age group, the focus is primarily on purchasing Sports, Electronics, and Grocery products.
![image](https://github.com/user-attachments/assets/1041cdfd-72db-48c6-b328-e1e841e60f1c)
**Fig 8: Average order value by category in the group age of above 65**

# Conclusions from thea above illustrations

## Dominant Customer Segment  
- The **41–65 age group** contributes the most to overall revenue, making them a crucial target audience.  
- Within this group, **Books** and **Home & Kitchen** drive significant revenue, with **Books** showing the highest growth rate.  

## Shifting Category Preferences by Age  
- **26–40 age group:** Stable purchasing behavior, except for **Sports**, which has declined sharply.  
- **18–25 age group:** Growing interest in **Sports, Clothing, Electronics, and Beauty**, while other categories are losing traction.  
- **65+ age group:** Focused on **Sports, Electronics, and Grocery**, indicating demand for practical and entertainment-related products.  

## Category Growth Trends  
- **Home & Kitchen** experienced the **fastest-growing average order value** over the past three years, signaling an opportunity for further expansion.  
- **Toys** are gaining steady traction among the **41–65 age group**, suggesting a potential market for gifts or family-oriented purchases.  
- **Sports category decline** among **26–40-year-olds** may indicate shifting interests, requiring strategy adjustments.

# Recommendations for E-Commerce Business  

## Prioritize High-Value Customer Segments  
- Focus marketing efforts on the **41–65 age group**, as they generate the most revenue.  
- Enhance promotions for **Books** and **Home & Kitchen** products, leveraging their high growth.  

## Optimize Product Strategies by Age Group  
- **18–25:** Invest in **targeted promotions** for **Sports, Clothing, Electronics, and Beauty** products through influencer marketing and social media campaigns.  
- **26–40:** Investigate the reasons behind the **decline in Sports sales** and consider repositioning or bundling strategies.  
- **65+:** Improve the shopping experience for **Sports, Electronics, and Grocery** categories, possibly with **loyalty programs** or **senior-friendly promotions**.  

## Leverage Seasonal & Personalized Marketing  
- Offer **personalized recommendations** based on purchase behavior, especially for the **41–65 segment**.  
- Run **seasonal discounts** and **bundled offers** for high-growth categories like **Home & Kitchen** and **Toys**.  

## Address the Declining Categories  
- Conduct deeper analysis on why **Sports is declining** in the **26–40** group. Consider repositioning it with **fitness trends** or **home workout solutions**.  
- Identify **declining categories** in the **18–25 group** and explore **product innovation** or **rebranding**.  


