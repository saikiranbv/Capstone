# Retailrocket recommender system

## Overview

The objective of this project is to Understanding user behavior through implicit signals like clicks and views is crucial for building effective recommender systems in real-world e-commerce platforms. Being able to predict preferences from limited interaction data helps personalize user experience, boosting engagement and conversions. Simultaneously, detecting and removing abnormal traffic is essential to ensure the recommendations are based on genuine customer behavior and do not suffer from noise or bias—leading to better decision-making and higher ROI for businesses.

Here is a link to the Jupyter Notebook https://github.com/saikiranbv/Capstone/blob/main/capstone.ipynb

## Table of Contents

1. [Business Understanding](#business-understanding)
2. [Data Analysis](#data-analysis)
3. [Visualization](#vizualization)
4. [Modeling](#modeling)
5. [Business Insights](#business-insights)

## Business Understanding

The primary objective is to accurately predict item properties associated with “addtocart” events based on a visitor’s prior “view” behavior, and cdetect and filter out abnormal user activity to improve the performance of recommender systems

## Data Analysis

The data will be sourced from the publicly available RetailRocket Recommender System Dataset https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset which includes:

	events.csv – records user interactions (view, add to cart, transaction)
	item_properties_part1.csv and item_properties_part2.csv – describe time-stamped item properties  ( These are not loaded into the data folder due to thier size. Please refer to the kaggle site). 
	category_tree.csv – describes the hierarchical relationships between product categories

Here are the number of records in each file
Events data: 2,756,101
Category data: 1669
Item data: 20,275,902

Number of unique visitors = 1,407,580
Number of unique items = 235,061
Number of unique transactions = 17,673

## Visualization

As the Item data is hashed and category data is only a parent child relationship, most of the analysis is from Events data. 

### The Top 10 users are 
![image](https://github.com/saikiranbv/CapstoneEDA/blob/main/images/Top_ten_Users.png)

### The distribution by event type
![image](https://github.com/saikiranbv/CapstoneEDA/blob/main/images/Event_type_count.png)

### The distribution by event type in a Pie chart
![image](https://github.com/saikiranbv/CapstoneEDA/blob/main/images/Event_tupe_pie.png)

### The Ten items that are viewed are 
![image](https://github.com/saikiranbv/CapstoneEDA/blob/main/images/Top_ten_Viewed_Items.png)

### The Ten items that are added to cart are 
![image](https://github.com/saikiranbv/CapstoneEDA/blob/main/images/Top_ten_Items_AddedtoCart.png)

### The Ten items that are purchased are 
![image](https://github.com/saikiranbv/CapstoneEDA/blob/main/images/Top_ten_sold_items.png)

### The distribution of Customers returning vs New
![image](https://github.com/saikiranbv/CapstoneEDA/blob/main/images/ReturningUsersvsNew.png)


## Modeling

Linear Regression: Used as a baseline model for comparison.

Random Forest: Employed to capture non-linear relationships and improve predictive performance.

Here are the results:

| Model                 | Test Accuracy | Precision | Recall  | F1 Score |
|----------------------|---------------|-----------|---------|----------|
| Logistic Regression  | 0.9916        | 0.8125    | 0.0055  | 0.0109   |
| Random Forest        | 0.9870        | 0.1235    | 0.0897  | 0.1040   |

### Linear Regression:
![image](https://github.com/saikiranbv/CapstoneEDA/blob/main/images/Linear_Regression.png)

### Random Forest:
![image](https://github.com/saikiranbv/CapstoneEDA/blob/main/images/Random_Forest.png)

Linear Regression
|Metric    | Value   | Comment                                                            |
|----------|---------|--------------------------------------------------------------------|
|Accuracy  | 0.9916  | Very high — but likely due to class imbalance                      |
|Precision | 0.8125  | High precision — model is rarely wrong when it predicts a purchase |
|Recall    | 0.0055  | Very low — model misses most purchases                             |
|F1-Score  | 0.0109  | Almost useless — poor balance between precision and recall         |

Random Forest

|Metric    | Value   | Comment                                                            |
|----------|---------|--------------------------------------------------------------------|
|Accuracy  | 0.9870  | Also very high due to imbalance                                    | 
|Precision | 0.1235  | Low — but better than random                                       |  
|Recall    | 0.0897  | Catches more purchases than Linear Regression                      | 
|F1-Score  | 0.1040  | Better than Linear Regression, but still weak overall              |

The models are trained on a dataset where purchases are very rare compared to non-purchases.
High accuracy is misleading as the models mostly predict "no purchase", which is correct most of the time only because purchases are rare.
We can't rely on these models yet for: Accurately predicting purchases and Generating reliable product recommendations

### Next we compared the following Models from the surprise library
	KNNBasic
	
	SVD - Singular Value Decomposition 
	
	NMF - Nonnegative matrix factorization 
	
	CoClustering 
	
	SlopeOne (collaborative filtering algorithms)

## Business Insights

Here are the results 
|Algorithm     |Mean MSE      |Std MSE           |Training Time (s)   |Testing Time (s)   |
|--------------|--------------|------------------|--------------------|-------------------|
|KNNBasic      |14029.739822  |237.368458        | 114.583982         | 0.315372          |
|SVD           |14029.739822  |227.553549        |  28.670742         | 0.066912          |
|NMF           |14029.739822  |259.631642        | 110.013319         | 0.095390          |
|CoClustering  |14029.739822  |249.112258        | 211.242633         | 0.177622          |
|SlopeOne      |14036.135630  |221.138585        |   0.495969         | 0.206490          |

Based on the results, here is the summary

|Category                       |Best Option	|Justification                            |
|-------------------------------|---------------|-----------------------------------------|
|Best Accuracy    	        |KNNBasic	|Lowest MSE + variance                    |
|Fastest Model	                |SVD	        |Good accuracy with minimal time          |
|Most Consistent	        |KNNBasic	|Lowest Std Dev                           |
|Least Tuning Effort	        |SlopeOne	|No hyperparameters, but accuracy is worse| 
|Most Computationally Intensive	|CoClustering	|Longest training time                    |


KNNBasic provides the most accurate predictions, meaning it will likely suggest items that users will actually enjoy. If we deploy this model, 
users are more likely to engage with the recommended items — leading to increased conversions and sales

Personalized Product Recommendations We can now predict how likely a customer is to buy a given product again or how often they would.

### typical Use case: "Customers also bought" or "Recommended for you"

Algorithms like KNNBasic or SVD can recommend products based on similar users or purchase patterns. Impact: Higher conversion rates, increased AOV (Average Order Value), and customer satisfaction.

We can also identify: Which products are repeat-purchased (sticky, high-loyalty) Which products are often co-purchased (cross-sell opportunities) Which users are heavy buyers of specific SKUs (VIP targeting)

We can use the code pred = best_algorithms['KNNBasic'].predict(visitorid='user123', itemid='item456') and based on the value predict. 
