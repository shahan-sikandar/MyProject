import os
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq


# -------------------
# Load Vectorstore
# -------------------
DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


# -------------------
# Custom Prompt
# -------------------
def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])


# -------------------
# Load LLM
# -------------------
HF_TOKEN = os.environ.get("HF_TOKEN")

def load_llm(temperature):
    return ChatGroq(
        api_key=HF_TOKEN,
        model="llama-3.1-8b-instant",  # Free Groq-hosted model
        temperature=temperature,
        max_tokens=512,
    )


# -------------------
# Hybrid Retriever (PDF + CSV)
# -------------------
def hybrid_retriever(vectorstore, query, k=3):
    """Retrieve from PDFs + ensure at least 1 CSV result if available."""
    results = vectorstore.as_retriever(search_kwargs={"k": k}).get_relevant_documents(query)

    # Search specifically in CSV docs
    csv_hits = [
        doc for doc in vectorstore.similarity_search(query, k=5)
        if doc.metadata.get("source") == "self_care_tips.csv"
    ]
    if csv_hits:
        results.append(csv_hits[0])  # Always add at least one CSV tip
    return results


# -------------------
# Streamlit App
# -------------------
def main():
    st.set_page_config(
    page_title="MediBot – AI Medical Assistant",
    page_icon="🩺",
    layout="centered"
    )

    st.markdown("""
                <div style="text-align:center;">
                <h1>🩺 MediBot</h1>
                <h4>AI-Powered Mental Health Assistant</h4>
                <p style="color:gray;">Final Year Project – Medical AI</p>
                </div>
                """, unsafe_allow_html=True)

    st.warning(
    "⚠️ This chatbot provides educational information only and is NOT a substitute for professional medical advice."
    )


    with st.sidebar:
        st.header("📘 Project Information")
        st.write("Final Year Project")
        st.write("Domain: Medical AI / NLP")
        st.write("Tools: Streamlit, LangChain, FAISS, Groq LLM")

        st.divider()
        st.header("⚙️ Model Settings")

        temperature = st.slider(
        "Creativity (Temperature)",
        0.0, 1.0, 0.5
        )

        k_value = st.slider(
        "Documents Retrieved (k)",
        1, 10, 3
        )

        show_sources = st.checkbox("Show Source Documents", value=True)

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.experimental_rerun()


    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    # Chat input
    prompt = st.chat_input("Ask me about mental health, self-care, or interventions...")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer the user's question.
        If you don’t know the answer, just say that you don’t know — don’t make things up.
        Only answer from the given context.

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk please.
        """

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("❌ Failed to load the vector store")
                return

            retriever = vectorstore.as_retriever(search_kwargs={"k": k_value})

            document_chain = create_stuff_documents_chain(
                                    llm=load_llm(temperature),
                                    prompt=set_custom_prompt(CUSTOM_PROMPT_TEMPLATE),
                                   )

            qa_chain = create_retrieval_chain(
                                       retriever,
                                       document_chain
                                      )

            with st.spinner("🔍 Analyzing medical knowledge..."):
                    result = qa_chain.invoke({"input": prompt})
                    response = result["answer"]





            # Show assistant reply
            st.chat_message("assistant").markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

            # Show sources if enabled
            if show_sources and docs:
                with st.expander("📚 Source Documents Used"):
                    for i, doc in enumerate(docs, 1):
                        source = doc.metadata.get("source", "Unknown")
                        page = doc.metadata.get("page", "-")
                        if source.endswith(".csv"):
                            st.markdown(f"**{i}.** `{source}` (Self-care Tip)")
                        else:
                            st.markdown(f"**{i}.** `{source}` (Page {page})")


        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")


if __name__ == "__main__":
    main()



