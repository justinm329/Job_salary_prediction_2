# imports necessary for preprocessing
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
%matplotlib inline

# NLP Imports
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import CountVectorizer
import re
# Code to download corpora
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

# Imports to create Neural Net and metrics associated
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import random
import tensorflow as tf


# Load data in with read_csv
file_path = "../Data/Train_rev1.csv"
df = pd.read_csv(file_path)
df.head()

# check the basic information of the data set, dtypes, null values, column names
df.info()

# check the total null values  in each column since the above shows there are missing value
# based on this those 3 features should be dropped before removing the null values
df.isnull().sum()

# Drop the columns that we won't need, Company, ContractType, ContractTime, SalaryRaw, LocationRaw, ID
new_df = df.drop(columns = ["Company", "ContractType", "ContractTime", "SalaryRaw", "LocationRaw", "Id", "SourceName"])
new_df.head()

# check datatypes again
new_df.dtypes

# drop the null values of this dataframe
new_df.dropna(inplace = True)

# make sure the shape looks good, this will be helpful later on
new_df.shape

# slice the dataframe so there are less rows, the above indicates there are over 200k rows this will take a while for a ML model to run
sliced_df = new_df.loc[:5000, :]

# create a varialbe for wordnetlemmatizer
lemmatizer = WordNetLemmatizer()

def clean_text(article):
    sw = set(stopwords.words('english'))
    sw_addons = {"k", "uk","also"} 
    # Substitute everything that is not a letter with an empty string
    regex = re.compile("[^a-zA-Z ]")
    # we sub in an extra character for anything that is not a character from the
    # above line of code
    re_clean = regex.sub('', article)
    # tokenize each word in the sentence
    words = word_tokenize(re_clean)
    # obtain the root word for each word 
    lem = [lemmatizer.lemmatize(word) for word in words]
    # obtain an output that is all lowercase and not in the stop words
    output = [word.lower() for word in lem if word.lower() not in sw.union(sw_addons)]
    output = ' '.join(output)
    return output

# test function on sliced df to make sure it is correct
clean_text(sliced_df["FullDescription"][0])

# create new column that has the clean description of the job
sliced_df['CleanDescription'] = sliced_df['FullDescription'].apply(clean_text)
sliced_df.head()

# Calculating the COUNT for the working corpus.
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words="english", min_df=.12)
count_vectorizer = vectorizer.fit_transform(sliced_df["CleanDescription"])
words_df = pd.DataFrame(count_vectorizer.toarray(), columns=vectorizer.get_feature_names())
words_df.head()

# since there are words that were missed with the stop words, I want to give each word an equal weight of one. I want to do this because
# I do not want the word "said" to out way the word "engineer" as an example
# Filter the dataframe so each word has a weight of 1 
filtered_df_2 = words_df.replace(list(range(1,100)),1)
filtered_df_2.head()

# combine the two dataframes
combined_df = pd.concat([sliced_df, filtered_df_2], axis = 1)
combined_df.head()

# drop the null values from the new dataframe
combined_df.dropna(inplace = True)

# drop the 2 description columns as we no longer need them
combined_df = combined_df.drop(columns = ["FullDescription", "CleanDescription", "Title", "LocationNormalized"])
combined_df.head()

# use get dummies to turn the category columns into number columns
encoded_df = pd.get_dummies(combined_df)
encoded_df.head()

# split the dataset into X and y
X = encoded_df.drop(columns = ["SalaryNormalized"])
y = encoded_df["SalaryNormalized"].values.reshape(-1,1)

# look at the shape of each data set
X.shape
y.shape

# lets import train test split to split the data up
X_train, X_test, y_train, y_test = train_test_split(X,
                                                   y,
                                                   random_state=78)
# use MinMaxScaler to scale the date
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

# scale the training data
x_scaler.fit(X_train)
y_scaler.fit(y_train)
X_train_scaled = x_scaler.transform(X_train)
X_test_scaled = x_scaler.transform(X_test)
y_train_scaled = y_scaler.transform(y_train)
y_test_scaled =y_scaler.transform(y_test)

# This will act as a random state variable....it might not be exact but it will be close to previous run
seed_value = 0
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# first we need to define the number of hidden nodes and input features
# start with a shallow network and expand from there
# I am going to start with the number of columns as the input features
number_input_columns = X.shape[1]
number_hidden_nodes = (X.shape[1]*2)



# Create NN
neural_network = Sequential()

# create the input latter
neural_network.add(Dense(units = number_input_columns, input_dim = number_input_columns, activation = "tanh", kernel_initializer='normal' ))
# create hidden layer
neural_network.add(Dense(units = number_hidden_nodes,  activation = "tanh", kernel_initializer='normal'))

# 2nd hidden layer
neural_network.add(Dense(units = number_hidden_nodes,  activation = "tanh", kernel_initializer='normal'))

# create output layer
neural_network.add(Dense(units = 1, activation = "linear", kernel_initializer='normal'))

# complie the NN and we can see how many total parameters along with how many parameters are in each layer
neural_network.compile(loss="mean_absolute_error", optimizer="adam", metrics=["mean_absolute_error"])
neural_network.summary()

# Fit the model to the training sets
nn_model = neural_network.fit(X_train_scaled, y_train_scaled,validation_split=0.2, epochs=20)

# Train vs test for shallow net
plt.plot(nn_model.history["loss"])
plt.plot(nn_model.history["val_loss"])
plt.title("loss_function - Training Vs. Validation - 2 hidden layers")
plt.legend(["train", "validation"])
plt.show()
plt.close()

# evaluate the model
neural_network.evaluate(X_test_scaled, y_test_scaled)

# make predictions on X_test_scaled
predictions = neural_network.predict(X_test_scaled)

predicted_salaries = y_scaler.inverse_transform(predictions)
real_salaries = y_scaler.inverse_transform(y_test_scaled.reshape(-1, 1))

salaries = pd.DataFrame({
    "Real": real_salaries.ravel(),
    "Predicted": predicted_salaries.ravel()
})
salaries.head()

