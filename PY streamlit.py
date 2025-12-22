import streamlit as st
import os
import pdfplumber
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

# --- הגדרות API ---
# החלף את ה-API_KEY במפתח האמיתי שלך (sk-...)
os.environ["OPENAI_API_KEY"] = "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"

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
folder_path = r"C:\Users\elnatan_u\Downloads\drive-download-20251221T143827Z-1-001"
db_path = "vectorstore_db"

# פונקציה לטעינה או בנייה של בסיס הנתונים
@st.cache_resource
def get_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # טעינת בסיס נתונים קיים אם יש
    if os.path.exists(db_path):
        return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    
    # סריקת התיקייה במידה ולא קיים אינדקס
    if not os.path.exists(folder_path):
        st.error(f"התיקייה לא נמצאה בנתיב: {folder_path}")
        st.stop()
        
    docs = []
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    
    if not pdf_files:
        st.error("לא נמצאו קבצי PDF בתיקייה.")
        st.stop()

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, filename in enumerate(pdf_files):
        status_text.text(f"מעבד קובץ {i+1} מתוך {len(pdf_files)}: {filename}")
        full_path = os.path.join(folder_path, filename)
        try:
            with pdfplumber.open(full_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_content = page.extract_text()
                    if page_content:
                        text += page_content + "\n"
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = text_splitter.split_text(text)
            
            for chunk in chunks:
                docs.append(Document(page_content=chunk, metadata={"source": filename}))
        except Exception as e:
            st.warning(f"שגיאה בעיבוד {filename}: {e}")
        
        progress_bar.progress((i + 1) / len(pdf_files))

    if not docs:
        st.error("לא הצלחתי לחלץ טקסט מהקבצים.")
        st.stop()

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(db_path)
    status_text.text("✅ בסיס הנתונים מוכן לשימוש!")
    return vectorstore

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
            # חיפוש במסמכים - שליפת 5 הקטעים הכי רלוונטיים
            docs = vector_db.similarity_search(prompt, k=5)
            
            context_list = []
            for d in docs:
                src = d.metadata.get('source', 'מקור לא ידוע')
                context_list.append(f"--- מקור: {src} ---\n{d.page_content}")
            
            context = "\n\n".join(context_list)
            
            # קריאה ל-LLM עם ה-Prompt המשופר
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            
            full_prompt = f"""
            אתה עוזר אינטליגנטי ומקצועי. 
            1. אם המשתמש שואל שאלת נימוסין (כמו היי, מה קורה?, מי אתה?), ענה לו בנימוס והסבר שאתה כאן כדי לעזור לו לנתח את המסמכים המקצועיים שלו.
            2. אם המשתמש שואל שאלה מקצועית, ענה בפירוט ובעברית רהוטה על סמך ההקשר (Context) המצורף בלבד.
            3. אם התשובה אינה מופיעה במסמכים, ציין זאת בנימוס.
            4. בסוף כל תשובה מקצועית שמסתמכת על המסמכים, ציין בפירוט מאילו קבצי מקור נלקח המידע.

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