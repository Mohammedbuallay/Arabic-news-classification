from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from tashaphyne.stemming import ArabicLightStemmer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import pandas as pd
ArListem = ArabicLightStemmer()

def text_stem(input_text):
    temp_text=[]
    for word in input_text.split(" "):
        stem = ArListem.light_stem(word)
        stem = ArListem.get_stem()
        temp_text.append(stem)
    temp_text = pd.DataFrame([" ".join(temp_text)])  
    return temp_text

def get_sim_tfidf(text1,text2): 
    from sklearn.feature_extraction.text import TfidfVectorizer
    Tfidf_vect = TfidfVectorizer()
    text1 = text_stem(text1)      
    text2 = text_stem(text2)
    Tfidf_vect.fit(text1[0]+text2[0])
    text1 = Tfidf_vect.transform(text1[0])
    text2 = Tfidf_vect.transform(text2[0])
    #print (text1,text2)
    return round(cosine_similarity(text1,text2)[0][0],2)

def get_vectors(text1,text2):
    vectorizer = CountVectorizer()
    vectorizer.fit([text1,text2])
    return vectorizer.transform([text1,text2]).toarray()    

def get_sim_vector_count(text1,text2): 
    vectors = get_vectors(text1,text2)
    return round(cosine_similarity(vectors)[0][1]*100,2)

