import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from langchain.llms import OpenAI
from langchain.agents import create_pandas_dataframe_agent

st.set_page_config(page_title="BizBot - Business Insight Chatbot", layout="wide")
st.title("📊 BizBot: Your Business Data Chatbot")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")

if uploaded_file and openai_api_key:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Preview of Uploaded Data")
    st.dataframe(df.head())

    llm = OpenAI(openai_api_key=openai_api_key, temperature=0)
    agent = create_pandas_dataframe_agent(llm, df, verbose=True)

    st.markdown("### 🧠 Ask a Question About Your Data")
    query = st.text_input("e.g., What are the top-selling products?")

    if query:
        with st.spinner("Analyzing..."):
            try:
                response = agent.run(query)
                st.success("✅ Response:")
                st.write(response)
            except Exception as e:
                st.error(f"❌ Error: {e}")

    st.markdown("### 📈 Quick Data Visualization")
    column = st.selectbox("Select a column to visualize", df.columns)
    if st.button("Generate Histogram"):
        plt.figure(figsize=(10, 5))
        df[column].hist(bins=20)
        st.pyplot(plt)
else:
    st.info("👆 Upload a CSV file and enter your OpenAI API key to start.")
