#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

# In[169]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


#  # Data Collection and Understanding:

# In[164]:


df = pd.read_csv("C:\\Users\\franc\\Desktop\\capstone project\\customer_train.csv")
df.head()


# In[165]:


df.shape


# In[233]:


df.dropna(inplace=True)


# In[166]:


df.isna().sum()


# In[170]:


df_cleaned = df.dropna(subset=['Income'])


# In[171]:


df_cleaned.dtypes


# In[172]:


del df_cleaned['Unnamed: 0']


# In[173]:


df=df_cleaned


# In[174]:


df.head(20)


# In[175]:


df.columns


# In[176]:


df.describe()


# In[177]:


df['Marital_Status'].unique()


# In[178]:


Mar_Stat={
    'Lajang': 'single',
    'Bertunangan':'Engaged',
    'Menikah':'Married',
    'Cerai':'Divorced',
    'Janda':'Widowed',
    'Duda':'Widowed'
}
    


# In[179]:


for i,j in Mar_Stat.items():
    df.loc[df['Marital_Status']==i,'Marital_Status']=j
df['Marital_Status'].unique()


# In[180]:


df.duplicated().sum()


# # Exploratory Data Analysis (EDA)

# In[181]:


sns.histplot(df['Income'], bins=20, color='blue', edgecolor='black')
plt.xlabel('Income')
plt.ylabel('Frequency')
plt.title('Income Distribution of Customers')
plt.show()


# In[182]:


plt.hist(df['Recency'], bins=20, color='red', edgecolor='black')
plt.xlabel('Recency')
plt.ylabel('Frequency')
plt.title('Income Distribution of Customers')
plt.show()


# In[183]:


plt.scatter(df['Income'], df['Recency'])

# Add labels and a title
plt.xlabel('Income')
plt.ylabel('Recency')

# Show the plot
plt.show()


# # Campaign Response Metrics

# In[184]:


sns.histplot(df['NumWebVisitsMonth'], bins=20, color='yellow', edgecolor='black')
plt.xlabel('NumWebVisitsMonth')
plt.ylabel('Frequency')
plt.title('count of WebVisitsMonth of Customers')
plt.show()


# In[185]:


import datetime
current_year = datetime.datetime.now().year
df['Age'] = current_year - df['Year_Birth']
df


# In[186]:


# Define a Seaborn color palette with different colors for each category
palette = sns.color_palette("husl", len(df['Education'].unique()))

# Create a bar plot to visualize the distribution of customers by Education
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Education', palette=palette)

plt.title('Distribution of customers by Education')
plt.xlabel('Education')
plt.ylabel('Customer Count')
plt.xticks(rotation=45)

plt.show()


# In[187]:


plt.figure(figsize=(8, 6))
sns.countplot(x=df['Age'])

plt.title('Distribution of customer by Age')
plt.xlabel('Age')
plt.ylabel('Customer Count')
plt.xticks(rotation=45)

plt.show()


# # Outliers using zscore 

# In[188]:


m=df.Income.mean()
m


# In[189]:


s=df.Income.std()
s


# In[190]:


df['zscore']=(df.Income-m)/s


# In[191]:


outliers=df[(df.zscore<-3)|(df.zscore>3)]
outliers


# # correlation

# In[192]:


df_corr = df[['ID','Income','Recency','MntCoke','MntFruits','MntMeatProducts',
              'AcceptedCmp4','AcceptedCmp5','AcceptedCmp1','AcceptedCmp2','Complain',
              'Z_Revenue','Response',
              'Age','zscore']].corr()
df_corr


# In[193]:


sns.heatmap(df_corr)


# In[194]:


df.describe()


# # Encoding

# In[195]:


categorical_columns = df.select_dtypes(include=['object', 'category']).columns
categorical_columns


# In[196]:


cols_to_encode =['Education', 'Marital_Status', 'Dt_Customer']


# In[197]:


new_encoded_cols_names = []
for col in cols_to_encode:
    new_encoded_cols_names += [f"is_{category}" for category in df[col].unique().tolist()]
one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_cols = one_hot_encoder.fit_transform(df[cols_to_encode])

df_encoded = pd.DataFrame(encoded_cols, columns=new_encoded_cols_names)
df_one_hot_encoded = df.join(df_encoded)

df_one_hot_encoded.head()
    


# # Data Splitting & feature scaling

# In[198]:


numeric_columns = df_one_hot_encoded.select_dtypes(include=['number'])


# In[199]:


scaler = StandardScaler()
scaler.fit(numeric_columns)
scaled_data = pd.DataFrame(scaler.transform(numeric_columns), columns=numeric_columns.columns)

scaled_data.head()


# # Feature Engineering

# In[200]:


total_spending = df[['MntCoke','MntFruits', 'MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProds']].sum()

print(total_spending)


# In[201]:


df.columns


# In[ ]:


# Perform one-hot encoding for categorical variables
df = pd.get_dummies(df, columns=["Education", "Marital_Status"], prefix=["Educ", "Marital"])


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

# Define features and target variable
features = df.drop(columns=["Response"])
target = df["Response"]

# Initialize the Random Forest classifier
clf = RandomForestClassifier()

# Fit the model to the data
clf.fit(features, target)

# Get feature importances
feature_importances = clf.feature_importances_

# Create a DataFrame to display feature importances
feature_importance_df = pd.DataFrame({'Feature': features.columns, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Visualize feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)

plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Feature')

plt.show()


# In[ ]:


from sklearn.manifold import TSNE
for i in range(10,50):
    tsne = TSNE(n_components=2, random_state=1, perplexity=i)
    df_tsne = tsne.fit_transform(scaled_data)
    
df_tsne


# #  Classification Model

# In[218]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
x = df.drop('Response', axis=1)  # Features
y = df['Response']  # Target variable
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# # SVM

# In[220]:


from sklearn.svm import SVC


# In[ ]:


model = SVC(kernel='linear', C=1)
model.fit(x_train, y_train)
y_pred=model.predict(x_test)
accuracy_score(y_test, y_pred)


# # Customer Segmentation

# In[209]:


from sklearn.cluster import KMeans

# Select features for clustering
clustering_features = df[['Income', 'Age']]

# Choose the number of clusters (k)
k = 4

# Initialize K-means clustering
kmeans = KMeans(n_clusters=k, random_state=42)

# Fit K-means to the data
df['Cluster'] = kmeans.fit_predict(clustering_features)


# In[211]:


# Visualize the clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Income', y='Age', hue='Cluster', palette='viridis', s=100)

plt.title('Customer Segmentation')
plt.xlabel('Income')
plt.ylabel('Customer Age')
plt.legend(title='Cluster')

plt.show()


# # Campaign Response Analysis¶

# In[212]:


# Calculate response rates for each campaign
campaign_columns = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response']
response_rates = df[campaign_columns].mean() * 100

# Visualize response rates
plt.figure(figsize=(10, 6))
sns.barplot(x=response_rates.index, y=response_rates.values)

plt.title('Campaign Response Rates')
plt.ylabel('Response Rate (%)')
plt.xticks(rotation=45)

plt.show()


# # t-SNE

# In[229]:


from sklearn.manifold import TSNE


# In[ ]:


for i in range(10,50):
    tsne = TSNE(n_components=2, random_state=1, perplexity=i)
    df_tsne = tsne.fit_transform(scaled_data)
    
df_tsne


# In[ ]:




