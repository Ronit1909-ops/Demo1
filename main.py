import streamlit as st
import pandas as pd
import numpy as np
import faiss
import google.generativeai as genai
from dotenv import load_dotenv
import os
import tempfile
from sentence_transformers import SentenceTransformer
import time

# Load environment variables
load_dotenv()

# Configure Gemini API
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    st.error("Please set your GEMINI_API_KEY in the .env file")
    st.stop()

genai.configure(api_key=api_key)

# Initialize the sentence transformer model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

def create_faiss_index(texts):
    """Create FAISS index from text data"""
    embeddings = model.encode(texts)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    return index, embeddings

def search_similar_texts(query, index, texts, k=5):
    """Search for similar texts using FAISS"""
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding.astype('float32'), k)
    return [texts[i] for i in indices[0]]

def process_dataframe(df):
    """Process dataframe to create text chunks with better context"""
    text_chunks = []
    # Add column information first
    columns_info = []
    for col in df.columns:
        if df[col].dtype in ['object', 'category']:
            unique_values = df[col].unique()
            columns_info.append(f"Column '{col}' has {len(unique_values)} unique values: {', '.join(map(str, unique_values))}")
        elif df[col].dtype in ['int64', 'float64']:
            columns_info.append(f"Column '{col}' is numeric with range from {df[col].min()} to {df[col].max()}")
    
    text_chunks.append("\n".join(columns_info))
    
    # Add sample data
    for _, row in df.iterrows():
        chunk = " | ".join([f"{col}: {val}" for col, val in row.items()])
        text_chunks.append(chunk)
    return text_chunks

def generate_response(query, context, df):
    """Generate response using Gemini API with improved context"""
    # Prepare data summary
    data_summary = f"""
    Dataset Summary:
    - Total rows: {len(df)}
    - Columns: {', '.join(df.columns)}
    - Column types: {', '.join([f"{col}: {dtype}" for col, dtype in df.dtypes.items()])}
    """
    
    # Add unique value counts for categorical columns
    categorical_info = []
    for col in df.select_dtypes(include=['object', 'category']).columns:
        unique_values = df[col].unique()
        categorical_info.append(f"Column '{col}' has {len(unique_values)} unique values: {', '.join(map(str, unique_values))}")
    
    if categorical_info:
        data_summary += "\nCategorical Columns Information:\n" + "\n".join(categorical_info)
    
    prompt = f"""You are a data analysis assistant. Please analyze the following data and answer the question accurately.

    {data_summary}

    Relevant data context:
    {context}

    Question: {query}

    Please provide a detailed and accurate answer based on the actual data. If the question is about counting unique values or categories, make sure to count them precisely from the data.
    
    Answer:"""
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "I apologize, but I encountered an error while processing your request. Please try again."

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        border: none;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
        padding: 10px;
    }
    .uploadedFile {
        background-color: #e8f5e9;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .response-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.title("📊 CSV Chatbot")
st.markdown("### Upload your data and chat with it!")

# File upload
uploaded_file = st.file_uploader("📂 Upload your CSV or Excel file", type=['csv', 'xlsx'])

if uploaded_file is not None:
    with st.spinner('Loading and processing your data...'):
        # Read the file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Display the dataframe preview with a nice container
        st.markdown('<div class="uploadedFile">', unsafe_allow_html=True)
        st.write("### Preview of your data (first 5 rows):")
        st.dataframe(df.head(), use_container_width=True)
        st.write(f"### Total rows in dataset: {len(df)}")
        
        # Show column information
        st.write("### Column Information:")
        for col in df.columns:
            if df[col].dtype in ['object', 'category']:
                unique_values = df[col].unique()
                st.write(f"- {col}: {len(unique_values)} unique values")
            elif df[col].dtype in ['int64', 'float64']:
                st.write(f"- {col}: Numeric (range: {df[col].min()} to {df[col].max()})")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Process the data and create FAISS index
        with st.spinner('Creating vector database...'):
            text_chunks = process_dataframe(df)
            index, embeddings = create_faiss_index(text_chunks)
        
        st.success('Data processing complete! You can now chat with your data.')
    
    # Chat interface
    st.markdown("### 💬 Chat with your data")
    user_query = st.text_input("Ask a question about your data:", placeholder="e.g., How many unique values are in column X?")
    
    if user_query:
        with st.spinner('Searching for relevant information...'):
            # Search for relevant context
            similar_texts = search_similar_texts(user_query, index, text_chunks)
            context = "\n".join(similar_texts)
        
        with st.spinner('Generating response...'):
            # Generate response with the full dataframe
            response = generate_response(user_query, context, df)
            
            # Display response in a nice container
            st.markdown('<div class="response-box">', unsafe_allow_html=True)
            st.write("### 🤖 Response:")
            st.write(response)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Data analysis section
        if "analyze" in user_query.lower() or "analysis" in user_query.lower():
            st.markdown("### 📈 Data Analysis")
            with st.spinner('Generating analysis...'):
                # Add basic data analysis
                st.write("#### Basic Statistics:")
                st.dataframe(df.describe(), use_container_width=True)
                
                # Add visualization options
                st.write("#### Visualizations:")
                for column in df.select_dtypes(include=['float64', 'int64']).columns:
                    st.write(f"##### {column}")
                    st.bar_chart(df[column]) 