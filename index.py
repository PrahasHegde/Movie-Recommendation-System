#imports
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import matplotlib.pyplot as plt
import seaborn as sns



movies = pd.read_csv('movies.csv')
print(movies.head())
print(movies.shape)
print(movies.info())
# print(movies.isnull().sum())

#we'll use some cols from dataset for our project
needed_cols = ['genres', 'keywords', 'tagline', 'cast', 'director']
df = movies[needed_cols]
# print(df.isnull().sum())

#we need to replace the null value with an empty string.
for col in needed_cols:
  df[col] = df[col].fillna('')

print(df.isnull().sum())

"""We will give the user a prompt like “Enter your favorite movie name: ”.
 After the user enters a movie name he/she will be recommended other movies similar to that of the movie they entered"""

#combine features
combined_cols = df['genres'] + " " + df['keywords'] + " " + df['tagline'] + " " + df['cast'] + " " + df['director']
print(combined_cols)
#It’s a single string consisting of all the details of the movie from its genres, taglines, keywords, cast, and directors.
print(combined_cols[0])


#convert cols into numbers using the TfidfVectorizer() 

vectorizer = TfidfVectorizer()
vectorized_combined_cols = vectorizer.fit_transform(combined_cols)
print(vectorized_combined_cols)

"""use the cosine_similarity(). It will give us a matrix(like a correlation matrix) 
It carries the percentage of similarity each movie has with each other."""

similarity = cosine_similarity(vectorized_combined_cols)
np.set_printoptions(threshold=10) #This will print only the first 10 rows from the numpy array
print(similarity)


#Now we are going to create a list that will have all the movie names in it.
movie_titles = movies['title'].tolist()
print(movie_titles)

"""difflib, this library will help us find similar words in a list
if the user enters something like, “ironman2” it will find the more similar words from the list and prints them"""

user_input = input("Enter your favorite movie to get similar recommendations: ")
close_movie_names = difflib.get_close_matches(user_input, movie_titles)
print(close_movie_names)

#1st match is closest match
close_match = close_movie_names[0]
print(close_match)


#we need to find the index value of the movie.
index_of_the_movie = movies[movies.title == close_match]['index'].values[0]
print(index_of_the_movie)


#get all the similarity scores this movie has with all other movies using the “similarity” we created before
similarity_score = list(enumerate(similarity[index_of_the_movie]))
print(similarity_score)


#we are going to sort the movies in descending order so that more similar movies are at the top.
sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 
print(sorted_similar_movies)


#use a for loop and get all the titles of the movies and print them out
print('Movies suggested for you : \n')

i = 1

for movie in sorted_similar_movies:
  index = movie[0]
  title_from_index = movies[movies.index==index]['title'].values[0]
  if (i<30):
    print(i, '.',title_from_index)
    i+=1

##############################################################################
# Get the top 10 recommended movies
top_n = 10
top_movies = sorted_similar_movies[:top_n]

# Extract movie titles and similarity scores
titles = [movies.iloc[movie[0]]['title'] for movie in top_movies]
scores = [movie[1] for movie in top_movies]

# Plot the bar chart
plt.figure(figsize=(10, 6))
plt.barh(titles, scores, color='skyblue')
plt.xlabel('Similarity Score')
plt.title('Top 10 Movie Recommendations')
plt.gca().invert_yaxis()
plt.show()

#####################################################################################################
#heatmap
# Reduce the size of the similarity matrix for visualization purposes
reduced_similarity = similarity[:100, :100]

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(reduced_similarity, cmap='coolwarm', xticklabels=False, yticklabels=False)
plt.title('Similarity Matrix Heatmap (First 100 Movies)')
plt.show()
########################################################################################################

# Plot the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(range(len(sorted_similar_movies)), [score[1] for score in sorted_similar_movies], alpha=0.6)
plt.xlabel('Movies')
plt.ylabel('Similarity Score')
plt.title('Similarity Score Distribution')
plt.show()
