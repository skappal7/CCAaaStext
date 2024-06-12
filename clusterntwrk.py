import streamlit as st
import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import networkx as nx
import plotly.graph_objects as go

# Function to preprocess data
def preprocess_data(data):
    # Add Sentiment Score
    data['sentiment_score'] = data['content'].apply(lambda x: TextBlob(x).sentiment.polarity)
    
    # Perform Topic Modeling
    vectorizer = CountVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(data['content'])
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(X)
    
    # Assign Topics
    topic_labels = ['Topic 1', 'Topic 2', 'Topic 3', 'Topic 4', 'Topic 5']
    data['topic'] = lda.transform(X).argmax(axis=1)
    data['topic'] = data['topic'].apply(lambda x: topic_labels[x])
    
    # Extract Keywords
    top_n_words = 5
    def extract_keywords(text, n=top_n_words):
        vectorizer = CountVectorizer(stop_words='english', max_features=n)
        X = vectorizer.fit_transform([text])
        keywords = vectorizer.get_feature_names_out()
        return ', '.join(keywords)

    data['keywords'] = data['content'].apply(lambda x: extract_keywords(x))
    
    return data

# Function to create network graph
def create_network_graph(data):
    G = nx.Graph()

    # Add nodes and edges based on topics and keywords
    for i, row in data.iterrows():
        keywords = row['keywords'].split(', ')
        for word in keywords:
            if not G.has_node(word):
                G.add_node(word, size=row['sentiment_score']*100, sentiment=row['sentiment_score'])
            for other_word in keywords:
                if word != other_word:
                    if not G.has_edge(word, other_word):
                        G.add_edge(word, other_word, weight=row['sentiment_score'])

    pos = nx.spring_layout(G)
    edge_x = []
    edge_y = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_size = []
    node_color = []
    node_text = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_size.append(G.nodes[node]['size'])
        sentiment = G.nodes[node]['sentiment']
        if sentiment > 0.5:
            node_color.append('green')
        elif sentiment < -0.5:
            node_color.append('red')
        else:
            node_color.append('orange')
        node_text.append(node)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=node_size,
            color=node_color,
            colorbar=dict(
                thickness=15,
                title='Sentiment',
                xanchor='left',
                titleside='right'
            )),
        text=node_text)

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Network Graph',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40)))

    return fig

# Streamlit app layout
st.title("Review Data Network Graph")
st.write("Upload your review data in CSV format with 'reviewId', 'Date', and 'content' columns as mandatory.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        # Check for mandatory columns
        if all(column in data.columns for column in ['reviewId', 'Date', 'content']):
            data = preprocess_data(data)
            fig = create_network_graph(data)
            st.plotly_chart(fig)
        else:
            st.error("CSV file must contain 'reviewId', 'Date', and 'content' columns.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
