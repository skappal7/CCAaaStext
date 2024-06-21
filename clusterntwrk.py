import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
import nltk

# Set the page configuration first
st.set_page_config(layout="wide")

# Download necessary NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

download_nltk_data()

# Function to preprocess text
@st.cache_data
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english')).union(set(['and', 'or', 'but', 'if','also','yhis','yrs', 'because', 'ca','would','let','abt','ac','the', 'a', 'an', 'in', 'on', 'at', 'to', 'with', 'is', 'are', 'was', 'were', 'of', 'for']))
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return tokens

# Function to perform sentiment analysis
@st.cache_data
def sentiment_analysis(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# Function to determine sentiment type
def sentiment_type(sentiment):
    if sentiment > 0.1:
        return "Positive"
    elif sentiment < -0.1:
        return "Negative"
    else:
        return "Neutral"

# Function to create a network graph
@st.cache_data
def create_network_graph(reviews, keyword=None, min_occurrence=1):
    G = nx.Graph()
    word_counts = {}
    word_sentiments = {}

    for _, row in reviews.iterrows():
        for word in row['tokens']:
            if word not in word_counts:
                word_counts[word] = 0
                word_sentiments[word] = []
            word_counts[word] += 1
            word_sentiments[word].append(row['sentiment'])

    for word, count in word_counts.items():
        if count >= min_occurrence and (keyword is None or keyword == "All" or word == keyword):
            G.add_node(word)

    for _, row in reviews.iterrows():
        for i in range(len(row['tokens'])):
            for j in range(i + 1, len(row['tokens'])):
                word1, word2 = row['tokens'][i], row['tokens'][j]
                if G.has_node(word1) and G.has_node(word2):
                    if G.has_edge(word1, word2):
                        G[word1][word2]['weight'] += 1
                    else:
                        G.add_edge(word1, word2, weight=1)

    for word in G.nodes():
        G.nodes[word]['size'] = word_counts[word]
        avg_sentiment = sum(word_sentiments[word]) / len(word_sentiments[word])
        G.nodes[word]['sentiment'] = avg_sentiment
        G.nodes[word]['sentiment_type'] = sentiment_type(avg_sentiment)

    return G

# Function to filter reviews by sentiment
def filter_reviews_by_sentiment(reviews, sentiment):
    if sentiment == "Positive":
        return reviews[reviews['sentiment'] > 0.1]
    elif sentiment == "Negative":
        return reviews[reviews['sentiment'] < -0.1]
    else:
        return reviews[(reviews['sentiment'] >= -0.1) & (reviews['sentiment'] <= 0.1)]

# Function to calculate word frequency trend
@st.cache_data
def calculate_word_frequency_trend(reviews, word):
    word_reviews = reviews[reviews['tokens'].apply(lambda x: word in x)]
    trend_data = word_reviews.groupby('Date').size().reset_index(name='count')
    trend_data['total'] = reviews.groupby('Date').size().reset_index(name='total')['total']
    trend_data['frequency'] = trend_data['count'] / trend_data['total'] * 100
    return trend_data

# Streamlit UI
st.title("Customer Reviews Network Graph")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    @st.cache_data
    def load_data(file):
        reviews = pd.read_csv(file)
        reviews['tokens'] = reviews['Text'].apply(preprocess_text)
        reviews['sentiment'] = reviews['Text'].apply(sentiment_analysis)
        reviews['Date'] = pd.to_datetime(reviews['Date'])
        return reviews

    reviews = load_data(uploaded_file)

    # Controls at the top
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        sentiment_filter = st.selectbox("Select Sentiment", ["All", "Positive", "Negative", "Neutral"])
    with col2:
        keyword_options = ["All"] + sorted(set(word for tokens in reviews['tokens'] for word in tokens if word.isalpha()))
        keyword = st.selectbox("Select Keyword", keyword_options)
    with col3:
        node_size_scale = st.slider("Adjust Node Size", min_value=1, max_value=20, value=10)
    with col4:
        min_occurrence = st.slider("Minimum Word Occurrence", min_value=1, max_value=20, value=1)

    if sentiment_filter != "All":
        reviews = filter_reviews_by_sentiment(reviews, sentiment_filter)

    G = create_network_graph(reviews, keyword, min_occurrence)
    pos = nx.spring_layout(G)

    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')

    node_x, node_y, node_size, node_color, node_text = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_size.append(G.nodes[node]['size'] * node_size_scale)
        sentiment = G.nodes[node]['sentiment']
        node_color.append('green' if sentiment > 0.1 else 'red' if sentiment < -0.1 else 'gray')
        node_text.append(f"Word: {node}<br>Count: {G.nodes[node]['size']}<br>Sentiment: {G.nodes[node]['sentiment_type']}")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=node_size,
            color=node_color,
            line=dict(width=2, color='#06516F'),
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    # Use columns to create side-by-side layout
    col1, col2 = st.columns([2, 1])

    with col1:
        selected_point = plotly_events(fig, click_event=True)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Word Trend")
        if selected_point:
            selected_word = list(G.nodes())[selected_point[0]['pointIndex']]
            trend_data = calculate_word_frequency_trend(reviews, selected_word)
            
            # Create trend chart
            trend_fig = px.line(trend_data, x='Date', y='frequency', title=f"Trend for '{selected_word}'")
            trend_fig.update_layout(yaxis_title="Frequency (%)")
            st.plotly_chart(trend_fig, use_container_width=True)

            # Display trend data as a table
            st.write(trend_data)
        else:
            st.write("Click on a node in the graph to view its trend.")

    # Sentiment distribution
    positive_reviews = reviews[reviews['sentiment'] > 0.1].shape[0]
    negative_reviews = reviews[reviews['sentiment'] < -0.1].shape[0]
    neutral_reviews = reviews[(reviews['sentiment'] >= -0.1) & (reviews['sentiment'] <= 0.1)].shape[0]

    st.markdown(f"### Sentiment Distribution")
    st.markdown(f"😊 Positive Reviews: {positive_reviews}")
    st.markdown(f"😞 Negative Reviews: {negative_reviews}")
    st.markdown(f"😐 Neutral Reviews: {neutral_reviews}")
