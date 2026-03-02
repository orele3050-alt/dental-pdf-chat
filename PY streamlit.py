import streamlit as st
import os

# תיקון הכרחי ל-SQLite בשרתי Streamlit
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Dental Chat", page_icon="🦷")
st.title("🦷 צ'אט עם מאמרים דנטליים")

# קבלת המפתח מה-Secrets
api_key = st.secrets.get("OPENAI_API_KEY")

if not api_key:
    st.error("אנא הגדר את ה-API Key ב-Secrets של Streamlit")
    st.stop()

# חיבור לבסיס הנתונים שהעלית (chroma.sqlite3 נמצא בתיקיית השורש)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
vectorstore = Chroma(persist_directory=".", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=api_key)

template = "ענה בעברית על בסיס המידע:\n{context}\n\nשאלה: {question}"
prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])), 
     "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

if user_input := st.chat_input("שאל על המאמרים..."):
    with st.chat_message("user"):
        st.write(user_input)
    with st.chat_message("assistant"):
        with st.spinner("סורק..."):
            st.write(rag_chain.invoke(user_input))
