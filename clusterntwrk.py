import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
import nltk
import base64

# Set the page configuration first
st.set_page_config(layout="wide")

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Custom CSS for styling (unchanged)
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
    stop_words = set(stopwords.words('english')).union(set(['and', 'or', 'but', 'if','also','yhis','yrs', 'because', 'ca','would','let','abt','ac','the', 'a', 'an', 'in', 'on', 'at', 'to', 'with', 'is', 'are', 'was', 'were', 'of', 'for']))
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

# Function to calculate trend data
def calculate_trend(word, reviews):
    word_reviews = reviews[reviews['tokens'].apply(lambda x: word in x)]
    trend_data = word_reviews.groupby('Date')['sentiment'].mean().reset_index()
    trend_data['count'] = word_reviews.groupby('Date').size().values
    return trend_data

# Function to create trend image
def create_trend_image(trend_data):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
    
    fig.add_trace(go.Scatter(x=trend_data['Date'], y=trend_data['sentiment'], mode='lines+markers', name='Sentiment'), row=1, col=1)
    fig.add_trace(go.Bar(x=trend_data['Date'], y=trend_data['count'], name='Count'), row=2, col=1)
    
    fig.update_layout(height=300, width=400, title_text="Word Trend", showlegend=False)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Sentiment", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    
    img_bytes = fig.to_image(format="png")
    encoding = base64.b64encode(img_bytes).decode()
    return encoding

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

# Streamlit UI
st.title("Customer Reviews Network Graph")

with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        reviews = pd.read_csv(uploaded_file)
        reviews['tokens'] = reviews['Text'].apply(preprocess_text)
        reviews['sentiment'] = reviews['Text'].apply(sentiment_analysis)
        reviews['Date'] = pd.to_datetime(reviews['Date'])  # Ensure 'Date' column is datetime
        sentiment_filter = st.selectbox("Select Sentiment", ["All", "Positive", "Negative", "Neutral"])
        keyword_options = ["All"] + sorted(set(word for tokens in reviews['tokens'] for word in tokens if word.isalpha()))
        keyword = st.selectbox("Select Keyword", keyword_options)
        node_size_scale = st.slider("Adjust Node Size", min_value=1, max_value=20, value=10)
        min_occurrence = st.slider("Minimum Word Occurrence", min_value=1, max_value=20, value=1)
        page_size = st.slider("Page Size", min_value=5, max_value=50, value=10)

        # Filter reviews by sentiment
        if sentiment_filter != "All":
            reviews = filter_reviews_by_sentiment(reviews, sentiment_filter)

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
    hover_data = []

    accent_colors = ['#06516F', '#0098DB', '#FAAF3B', '#333333', '#979797']

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        trend_data = calculate_trend(node, reviews)
        trend_image = create_trend_image(trend_data)
        hover_data.append(f"<b>{node}</b><br>Count: {G.nodes[node]['size']}<br>{G.nodes[node]['icon']}<br><img src='data:image/png;base64,{trend_image}'>")
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
        text=hover_data,
        hovertemplate='%{text}',
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

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Network graph of customer reviews',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    fig.update_layout(
        dragmode='zoom',  # Enable zoom
        clickmode='event+select',
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Rockwell"
        )
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
