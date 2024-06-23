import streamlit as st
import pandas as pd
import scattertext as stx
import spacy
import pyLDAvis.gensim_models
import plotly.express as px
from wordcloud import WordCloud
from gensim import corpora, models
from textblob import TextBlob
import networkx as nx
import matplotlib.pyplot as plt

# Load SpaCy model
nlp = spacy.load('en_core_web_sm')

def main():
    st.title("Contact Center Interaction Analysis")

    st.markdown("""
    This app visualizes the terms used in contact center interactions, 
    differentiating between customer and agent text.
    """)

    uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.write("Data Preview:")
        st.write(df.head())

        text_column = st.selectbox("Select the text column", df.columns)

        if st.button("Analyze"):
            perform_analysis(df, text_column)

def perform_analysis(df, text_column):
    st.write(f"Performing analysis on column: {text_column}")
    
    # Sentiment Analysis
    sentiment_analysis(df, text_column)
    
    # Network Graph
    network_graph(df, text_column)
    
    # Word Cloud
    create_wordcloud(df, text_column)
    
    # LDavis Topic Modeling
    lda_topic_modeling(df, text_column)

def sentiment_analysis(df, text_column):
    st.subheader("Sentiment Analysis")
    df['sentiment'] = df[text_column].apply(lambda x: TextBlob(x).sentiment.polarity)
    fig = px.histogram(df, x='sentiment', nbins=50, title="Sentiment Polarity Distribution")
    st.plotly_chart(fig)

def network_graph(df, text_column):
    st.subheader("Network Graph")
    docs = [nlp(text) for text in df[text_column]]
    edges = []
    for doc in docs:
        for token in doc:
            for child in token.children:
                edges.append((token.text, child.text))
    graph = nx.Graph(edges)
    plt.figure(figsize=(10, 10))
    nx.draw(graph, with_labels=True, node_size=20, font_size=10)
    st.pyplot(plt)

def create_wordcloud(df, text_column):
    st.subheader("Word Cloud")
    text = " ".join(review for review in df[text_column])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

def lda_topic_modeling(df, text_column):
    st.subheader("Topic Modeling")
    texts = df[text_column].apply(lambda x: [token.lemma_ for token in nlp(x) if not token.is_stop and token.is_alpha])
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)
    lda_display = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary, sort_topics=False)
    st.write(pyLDAvis.display(lda_display))

if __name__ == "__main__":
    main()
