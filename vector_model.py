#the nltk has classes to deal with natural language
import math
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#the numpy package has too many classes do deal with data and math
import numpy as np
from numpy.linalg import norm

#the dataset or collection
dataset = [
    "Earth orbits Sun, and the Moon orbits Earth. The Moon is a natural satellite that reflects the Sun's light.",
    "The Sun is the biggest celestial body in our solar system. The Sun is a star. The Sun is beautiful.",
    "Earth is the third planet in our solar system.",
    "The Sun orbits the Milky Way galaxy!"
]

#a variable to store the stopwords
stopWords = stopwords.words('english')
#the lemmatizer 
lemmatizer = WordNetLemmatizer()
#an array to store the clean dataset, after the preprocessing
cleanDataset = []

#store the clean dataset
cleanDataset = []

#perform the precprocessing
#for each doc present in dataset, do...
for doc in dataset:
    #tokenize (a doc is an array of words) the doc and convert to lower case
    tokenDoc = word_tokenize(doc.lower())
    #a variable to store a clean doc
    cleanDoc = []
    #for each word in tokenDoc, do...
    for word in tokenDoc:
        #if the word is alphanumeric and is not present in stopwords, then...
        if(word.isalnum() and word not in stopWords):
            #extracts the lemma and add it to cleanDoc array
            cleanWord = lemmatizer.lemmatize(word)
            cleanDoc.append(cleanWord)
    #add the processed doc to cleanDataset array
    cleanDataset.append(cleanDoc)

#show clean Dataset
print(cleanDataset)

#define a function to calculate the cosine similarity
#we are using the numpy methods
def cosineSimilarity(query, doc):
    sim = np.dot(query,doc)/(norm(query)*norm(doc))
    if(math.isnan(sim)):
        return 0
    else:
        return sim


#an array to store the results of query
results = []
#define the query
query = ['sun', 'star']
#a variable to define the id of document in the results
id = 1

#the script to calculate the similary for all docs
#for each doc in cleanDataset, do...
for doc in cleanDataset:
    #create the vocabulary of doc
    vocabulary = list(set(doc))

    #create two arrays with zeros to represents the query and the doc
    #vectors must have the same size to calculate similarity
    vecDoc = [0] * len(vocabulary)
    vecQuery = [0] * len(vocabulary)

    #for each word in doc, do...
    for word in doc:
        #fill the vecDoc with a weight number considering the term frequency of the word
        vecDoc[vocabulary.index(word)] +=1

    #do the same with query, but, when the term of query is not present in doc, the term frequency is zero. 
    for word in query:
        if word in vocabulary:
            vecQuery[vocabulary.index(word)] += 1

    #calculate the cosine similariy
    cosine = cosineSimilarity(vecDoc, vecQuery)
    #add the result in results list
    results.append({"id":id, "score" : round(cosine, 5)})
    #increment the id
    id+=1
#show the results
print(results)

#a function to sort the results by score
def mySort(obj):
    return obj['score']

#sort the result in descending way by similarity score
results.sort(key=mySort, reverse=True)

#show the results
for result in results:
    print(result)