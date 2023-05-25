
# Sales-Prediction
This is a case study of predicting future sales using machine learning algorithms. The goal is to build a model that can predict the sales  based on historical sales data.<br>

# Prerequisites
<h3>To run this project, you will need the following:<br></h3>

Python 3.x<br>
Jupyter Notebook<br>
scikit-learn library<br>
pandas library<br>
numpy library<br>

# Sales Prediction (Case Study)
 given some information about students like:<br>
 
 1. TV: Advertising cost spent in dollars for advertising on TV;<br>
 2. Radio: Advertising cost spent in dollars for advertising on Radio;<br>
 3. Newspaper: Advertising cost spent in dollars for advertising on Newspaper;<br>
 4. Sales: Number of units sold; <br>
 
 # How  did I do?

<h3>The dataset I am using for the sales prediction task is downloaded from Kaggle. Now let’s start with this task by importing the necessary Python libraries and dataset:<br></h3>

import pandas as pd<br>
import numpy as np<br>
import plotly.express as px<br>
import plotly.graph_objects as go<br>
from sklearn.model_selection import train_test_split<br>
from sklearn.linear_model import LinearRegression<br>
from sklearn import metrics<br>

data=pd.read_csv('advertising.csv')<br>
data.head()<br>
![Screenshot (3026)](https://user-images.githubusercontent.com/110754364/235335072-1a13544e-d9f2-44d5-b469-3e9fb76535f3.png)



<h3>Now before moving forward, let’s have a look at whether this dataset contains any null values or not:<br></h3>

data.isnull().sum()<br>

![Screenshot (3027)](https://user-images.githubusercontent.com/110754364/235335076-1a2963b3-8987-4771-9db6-e67e42d0228a.png)


<h3>So this dataset doesn’t have any null values. Now let’s visualize the relationship between the amount spent on advertising on TV and units sold:</h3><br>
figure=px.scatter(data_frame=data,x='Sales',y='TV',trendline='ols')<br>
figure.show()<br>

![Screenshot (3028)](https://user-images.githubusercontent.com/110754364/235335082-5b3f33b7-b2ec-475c-b4dc-a2bfb1eb65ee.png)


<h3>Now let’s visualize the relationship between the amount spent on advertising on newspapers and units sold:</h3><br>

![Screenshot (3030)](https://user-images.githubusercontent.com/110754364/235335084-4f465b92-2b11-425c-abed-ee17255038c0.png)




<h3>Now let’s visualize the relationship between the amount spent on advertising on radio and units sold:</h3><br>
figure=px.scatter(data_frame=data,x='Sales',y='Newspaper',trendline='ols')<br>
figure.show()<br>

![Screenshot (3031)](https://user-images.githubusercontent.com/110754364/235335157-154c2dcc-a8db-4feb-a05d-a4b32477873c.png)

<h3>Out of all the amount spent on advertising on various platforms, I can see that the amount spent on advertising the product on TV results in more sales of the product.</h3>

 # Future Sales Prediction Mode
 
 <h3>Now in this section, I will train a machine learning model to predict the future sales of a product. But before I train the model, let’s split the data into training and test sets</h3><br>
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)<br>
lr=LinearRegression()<br>
lr.fit(X_train,y_train)<br>
y_pred1 = lr.predict(X_test)<br>
score=metrics.r2_score(y_test,y_pred1)<br>
score<br>

# Result 
0.9059011844150826<br>

![sales_new](https://user-images.githubusercontent.com/110754364/235335342-7f190ca1-19e1-49ca-a74b-7fd618b518ec.png)







