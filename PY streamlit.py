import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# --- הגדרת המפתח החדש שעבד בבדיקה ---
MY_NEW_KEY = "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"

# בדיקה חכמה לשימוש ב-Secrets בענן
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
else:
    api_key = MY_NEW_KEY


# --- הגדרות דף ---
st.set_page_config(page_title="עוזר ה-PDF החכם", page_icon="📚")

st.markdown("""<style>.stApp {direction: RTL; text-align: right;}</style>""", unsafe_allow_html=True)
st.title("📚 צ'אט עם מסמכי ה-PDF שלך")

# --- טעינת בסיס הנתונים מהענן ---
db_path = "vectorstore_db"

@st.cache_resource
def get_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if os.path.exists(db_path):
        return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    else:
        st.error("שגיאה: תיקיית vectorstore_db לא נמצאה ב-GitHub!")
        st.stop()

vector_db = get_vector_db()

# --- ניהול הצ'אט ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("שאל אותי על המסמכים..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # חיפוש במסמכים
        docs = vector_db.similarity_search(prompt, k=3)
        context = "\n".join([d.page_content for d in docs])
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)
        full_prompt = f"Context:\n{context}\n\nQuestion: {prompt}\nענה בעברית על סמך ההקשר."
        
        try:
            response = llm.invoke(full_prompt).content
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"שגיאה: {e}")
