import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="PDF Q&A Bot", page_icon="📄", layout="centered")

# ---------- Custom CSS ----------
st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 10px;
    color: #1f2937;
}
.sub-text {
    font-size: 18px;
    color: #4b5563;
    margin-bottom: 18px;
}
.answer-box {
    background-color: #f8fafc;
    border: 1px solid #dbe4ee;
    border-radius: 12px;
    padding: 18px 20px;
    margin-top: 10px;
    margin-bottom: 15px;
}
.answer-title {
    font-size: 20px;
    font-weight: 600;
    color: #111827;
    margin-bottom: 10px;
}
.source-box {
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 12px;
}
.source-title {
    font-weight: 600;
    color: #2563eb;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Student Advisor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-text">Ask a question about your document.</div>',
    unsafe_allow_html=True
)

@st.cache_resource
def load_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore

vectorstore = load_vectorstore()

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant.
Answer the user's question using ONLY the context provided below.
If the answer is not contained in the context, respond with exactly:
I’m sorry, I am only authorized to talk about the provided document.
Do not use outside knowledge.

Context:
{context}

Question:
{question}

Answer:
"""
)

llm = ChatOpenAI(model="gpt-4", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt_template}
)

question = st.text_input(
    "Ask a question about your documents:",
    placeholder="e.g. What are Illegal U-Turns?"
)

if question:
    with st.spinner("Searching documents..."):
        result = qa_chain.invoke({"query": question})

    # Answer section
    st.markdown(
        f"""
        <div class="answer-box">
            <div class="answer-title">Answer</div>
            <div>{result["result"]}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Sources section
    with st.expander("View source passages"):
        docs = result["source_documents"][:3]
        for i, doc in enumerate(docs, start=1):
            page = doc.metadata.get("page", "N/A")
            content = doc.page_content[:500] + "..."
            st.markdown(
                f"""
                <div class="source-box">
                    <div class="source-title">Source {i} (Page {page})</div>
                    <div>{content}</div>
                </div>
                """,
                unsafe_allow_html=True
            )