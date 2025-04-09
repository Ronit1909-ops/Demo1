# CSV Chatbot with Gemini AI

This application allows you to upload CSV or Excel files and interact with them using natural language queries powered by Google's Gemini AI. The application uses FAISS for efficient similarity search and provides data analysis capabilities.

## Features

- Upload CSV or Excel files
- Convert data to FAISS vector database
- Chat with your data using natural language
- Get data analysis and insights
- Visualize data through charts

## Setup

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory and add your Gemini API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```
   You can get a free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run main.py
   ```
2. Upload your CSV or Excel file
3. Ask questions about your data in natural language
4. Get insights and analysis

## Example Questions

- "What are the main trends in this data?"
- "Show me the analysis of column X"
- "What is the average value of column Y?"
- "Find the highest value in column Z"

## Note

This application uses the free version of Gemini API, which has rate limits. Make sure to check the current limits and pricing on Google's website. 