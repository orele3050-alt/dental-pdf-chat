import streamlit as st
import os
import pdfplumber
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

# --- הגדרות API ---
# בדיקה חכמה: אם אנחנו בענן נמשוך מה-Secrets, אם במחשב נשתמש במפתח החדש
MY_NEW_KEY = "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"

try:
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    else:
        api_key = MY_NEW_KEY
except:
    api_key = MY_NEW_KEY

os.environ["OPENAI_API_KEY"] = api_key

# --- הגדרות דף ---
st.set_page_config(page_title="עוזר ה-PDF החכם", page_icon="📚", layout="centered")

# עיצוב בסיסי לתמיכה בעברית (RTL)
st.markdown("""
    <style>
    .stApp {
        direction: RTL;
        text-align: right;
    }
    div[data-testid="stChatMessageContent"] {
        text-align: right;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📚 צ'אט עם מסמכי ה-PDF שלך")
st.subheader("מערכת RAG לחיפוש וניתוח מסמכים")

# --- הגדרות נתיבים ---
# בענן, אנחנו מסתמכים על תיקיית ה-DB שהעלית ל-GitHub
db_path = "vectorstore_db"

# פונקציה לטעינה של בסיס הנתונים
@st.cache_resource
def get_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # טעינת בסיס נתונים קיים מהתיקייה שהעלית ל-GitHub
    if os.path.exists(db_path):
        return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    else:
        st.error("בסיס הנתונים (vectorstore_db) לא נמצא ב-GitHub. וודא שהעלית את התיקייה.")
        st.stop()

# אתחול המערכת
with st.spinner("מאתחל את בסיס הנתונים..."):
    vector_db = get_vector_db()

# --- ניהול שיחה ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# הצגת היסטוריית ההודעות
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# קלט מהמשתמש
if prompt := st.chat_input("שאל אותי משהו על המסמכים..."):
    # הוספת הודעת המשתמש
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # יצירת תשובה
    with st.chat_message("assistant"):
        with st.spinner("מעבד תשובה..."):
            # חיפוש במסמכים
            docs = vector_db.similarity_search(prompt, k=5)
            
            context_list = []
            for d in docs:
                src = d.metadata.get('source', 'מקור לא ידוע')
                context_list.append(f"--- מקור: {src} ---\n{d.page_content}")
            
            context = "\n\n".join(context_list)
            
            # קריאה ל-LLM
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            
            full_prompt = f"""
            אתה עוזר אינטליגנטי ומקצועי. 
            1. אם המשתמש שואל שאלת נימוסין, ענה לו בנימוס.
            2. אם המשתמש שואל שאלה מקצועית, ענה בפירוט ובעברית רהוטה על סמך ההקשר (Context) המצורף בלבד.
            3. אם התשובה אינה מופיעה במסמכים, ציין זאת בנימוס.
            4. בסוף כל תשובה מקצועית, ציין בפירוט מאילו קבצי מקור נלקח המידע.

            Context:
            {context}

            Question:
            {prompt}
            """
            
            try:
                response = llm.invoke(full_prompt).content
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"שגיאה בתקשורת עם המודל: {e}")
