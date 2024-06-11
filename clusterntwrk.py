import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import nltk

# Set the page configuration first
st.set_page_config(layout="wide")

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Custom CSS for styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #07B1FC !important;
        color: white;
    }
    .stButton>button {
        background-color: #06516F !important;
        color: white;
    }
    .stButton>button:hover {
        background-color: #0098DB !important;
        color: white;
    }
    .stNumberInput input {
        border: 2px solid #06516F !important;
    }
    .stNumberInput input:focus {
        border: 2px solid #0098DB !important;
    }
    </style>
""", unsafe_allow_html=True)

# Function to preprocess text
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english')).union(set(['and', 'or', 'but', 'if', 'because', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'with', 'is', 'are', 'was', 'were', 'of', 'for']))
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return tokens

# Function to perform sentiment analysis
def sentiment_analysis(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# Function to determine sentiment icon
def sentiment_icon(sentiment):
    if sentiment > 0.1:
        return "😊"
    elif sentiment < -0.1:
        return "😞"
    else:
        return "😐"

# Function to create a network graph
def create_network_graph(reviews_tokens, keyword=None, min_occurrence=1):
    G = nx.Graph()
    word_counts = {}
    word_sentiments = {}

    for tokens, sentiment in reviews_tokens:
        for word in tokens:
            if word not in word_counts:
                word_counts[word] = 0
                word_sentiments[word] = []
            word_counts[word] += 1
            word_sentiments[word].append(sentiment)

    for tokens in reviews_tokens:
        for i in range(len(tokens[0])):
            for j in range(i + 1, len(tokens[0])):
                if (keyword is None or keyword == "All" or tokens[0][i] == keyword or tokens[0][j] == keyword) and word_counts[tokens[0][i]] >= min_occurrence and word_counts[tokens[0][j]] >= min_occurrence:
                    G.add_edge(tokens[0][i], tokens[0][j])

    for word in word_counts:
        if word in G and word_counts[word] >= min_occurrence:
            avg_sentiment = sum(word_sentiments[word]) / len(word_sentiments[word])
            G.nodes[word]['size'] = word_counts[word]
            G.nodes[word]['sentiment'] = avg_sentiment
            G.nodes[word]['icon'] = sentiment_icon(avg_sentiment)

    return G

# Function to filter reviews by sentiment
def filter_reviews_by_sentiment(reviews, sentiment):
    if sentiment == "Positive":
        return reviews[reviews['sentiment'] > 0.1]
    elif sentiment == "Negative":
        return reviews[reviews['sentiment'] < -0.1]
    else:
        return reviews[(reviews['sentiment'] >= -0.1) & (reviews['sentiment'] <= 0.1)]

# Function to perform clustering
def perform_clustering(reviews):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(reviews['text'])
    kmeans = KMeans(n_clusters=5, random_state=0).fit(X)
    reviews['cluster'] = kmeans.labels_
    return reviews, kmeans, vectorizer

# Function to detect anomalies
def detect_anomalies(reviews, kmeans, vectorizer):
    anomaly_scores = kmeans.transform(vectorizer.transform(reviews['text']))
    reviews['anomaly_score'] = anomaly_scores.min(axis=1)
    return reviews

# Streamlit UI
st.title("Customer Reviews Network Graph with Clustering and Anomaly Detection")

with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        reviews = pd.read_csv(uploaded_file)
        reviews['tokens'] = reviews['text'].apply(preprocess_text)
        reviews['sentiment'] = reviews['text'].apply(sentiment_analysis)
        sentiment_filter = st.selectbox("Select Sentiment", ["All", "Positive", "Negative", "Neutral"])
        keyword_options = ["All"] + sorted(set(word for tokens in reviews['tokens'] for word in tokens if word.isalpha()))
        keyword = st.selectbox("Select Keyword", keyword_options)
        node_size_scale = st.slider("Adjust Node Size", min_value=1, max_value=20, value=10)
        min_occurrence = st.slider("Minimum Word Occurrence", min_value=1, max_value=20, value=1)
        page_size = st.slider("Page Size", min_value=5, max_value=50, value=10)
        show_anomalies = st.checkbox("Show Anomalies")

        # Filter reviews by sentiment
        if sentiment_filter != "All":
            reviews = filter_reviews_by_sentiment(reviews, sentiment_filter)

        # Perform clustering
        reviews, kmeans, vectorizer = perform_clustering(reviews)

        # Detect anomalies
        if show_anomalies:
            reviews = detect_anomalies(reviews, kmeans, vectorizer)

# Display the review table with pagination
if uploaded_file is not None:
    page_number = st.number_input("Page Number", min_value=1, max_value=(len(reviews) // page_size) + 1, step=1)
    start_index = (page_number - 1) * page_size
    end_index = start_index + page_size
    st.write(reviews.iloc[start_index:end_index])

    # Create network graph
    reviews_tokens = list(zip(reviews['tokens'], reviews['sentiment']))
    G = create_network_graph(reviews_tokens, keyword, min_occurrence)
    pos = nx.spring_layout(G)

    # Create Plotly figure
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
    node_text = []
    node_size = []
    node_color = []

    accent_colors = ['#06516F', '#0098DB', '#FAAF3B', '#333333', '#979797']

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{node}<br>Count: {G.nodes[node]['size']}<br>{G.nodes[node]['icon']}")
        node_size.append(G.nodes[node]['size'] * node_size_scale)
        sentiment = G.nodes[node]['sentiment']
        if sentiment > 0.1:
            node_color.append('green')
        elif sentiment < -0.1:
            node_color.append('red')
        else:
            node_color.append('gray')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        textposition="bottom center",
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale=accent_colors,
            size=node_size,
            color=node_color,
            line=dict(width=2, color='#06516F'),
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    hover_text = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        text=node_text,
        hoverinfo='text',
        marker=dict(
            size=node_size,
            opacity=0
        ))

    fig = go.Figure(data=[edge_trace, node_trace, hover_text],
                    layout=go.Layout(
                        title='Network graph of customer reviews',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False))
                    )

    fig.update_layout(
        dragmode='zoom',  # Enable zoom
        clickmode='event+select'
    )

    st.plotly_chart(fig)

    # Sentiment distribution
    positive_reviews = reviews[reviews['sentiment'] > 0.1].shape[0]
    negative_reviews = reviews[reviews['sentiment'] < -0.1].shape[0]
    neutral_reviews = reviews[(reviews['sentiment'] >= -0.1) & (reviews['sentiment'] <= 0.1)].shape[0]

    st.markdown(f"### Sentiment Distribution")
    st.markdown(f"😊 Positive Reviews: {positive_reviews}")
    st.markdown(f"😞 Negative Reviews: {negative_reviews}")
    st.markdown(f"😐 Neutral Reviews: {neutral_reviews}")

    # Cluster distribution
    st.markdown(f"### Cluster Distribution")
    cluster_counts = reviews['cluster'].value_counts().sort_index()
    for cluster, count in cluster_counts.items():
        st.markdown(f"Cluster {cluster}: {count} reviews")

    if show_anomalies:
        st.markdown(f"### Anomalies")
        anomaly_reviews = reviews[reviews['anomaly_score'] > reviews['anomaly_score'].quantile(0.95)]
        st.write(anomaly_reviews)
