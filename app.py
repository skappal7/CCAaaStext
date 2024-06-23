import streamlit as st
import pandas as pd
import scattertext as stx
import spacy

# Load SpaCy model
nlp = spacy.load('en_core_web_sm')

def main():
    st.title("Scattertext Visualization App")

    st.markdown("""
    This app allows you to upload a CSV, Excel, or text file and visualize the text data using Scattertext.
    """)

    uploaded_file = st.file_uploader("Upload your CSV, Excel, or text file", type=["csv", "xlsx", "txt"])

    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_table(uploaded_file, header=None, names=["text"])
        
        st.write("Data Preview:")
        st.write(df.head())

        if uploaded_file.name.endswith(('.csv', '.xlsx')):
            text_column = st.selectbox("Select the text column", df.columns)
            text_data = df[text_column]
        else:
            text_data = df["text"]

        if st.button("Analyze"):
            scattertext_visualization(text_data)

def scattertext_visualization(text_data):
    st.write("Generating Scattertext visualization...")
    
    # Create a Scattertext corpus
    corpus = stx.CorpusFromPandas(pd.DataFrame({"text": text_data}), category_col="text", text_col="text", nlp=nlp).build()
    
    # Create the Scattertext visualization
    html = stx.produce_scattertext_explorer(corpus, category='text', category_name='Text', not_category_name='None',
                                            width_in_pixels=1000, metadata=text_data)
    
    # Display the Scattertext visualization in Streamlit
    st.components.v1.html(html, width=1000, height=700, scrolling=True)

if __name__ == "__main__":
    main()
