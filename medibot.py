# import os
# import streamlit as st

# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA

# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import PromptTemplate
# from langchain_groq import ChatGroq


# ## Uncomment the following files if you're not using pipenv as your virtual environment manager
# #from dotenv import load_dotenv, find_dotenv
# #load_dotenv(find_dotenv())


# DB_FAISS_PATH="vectorstore/db_faiss"
# @st.cache_resource
# def get_vectorstore():
#     embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
#     db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
#     return db


# def set_custom_prompt(custom_prompt_template):
#     prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
#     return prompt

# HF_TOKEN=os.environ.get("HF_TOKEN")
# def load_llm():
#     llm = ChatGroq(
#         api_key=HF_TOKEN,
#         model="llama-3.1-8b-instant",   # You can also try "mixtral-8x7b-32768"
#         temperature=0.5,
#         max_tokens=512
#     )
#     return llm


# def main():
#     st.title("Ask Chatbot!")

#     if 'messages' not in st.session_state:
#         st.session_state.messages = []

#     for message in st.session_state.messages:
#         st.chat_message(message['role']).markdown(message['content'])

#     prompt=st.chat_input("Pass your prompt here")

#     if prompt:
#         st.chat_message('user').markdown(prompt)
#         st.session_state.messages.append({'role':'user', 'content': prompt})

#         CUSTOM_PROMPT_TEMPLATE = """
#                 Use the pieces of information provided in the context to answer user's question.
#                 If you dont know the answer, just say that you dont know, dont try to make up an answer. 
#                 Dont provide anything out of the given context

#                 Context: {context}
#                 Question: {question}

#                 Start the answer directly. No small talk please.
#                 """
        
#         #HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3" # PAID
#         #HF_TOKEN=os.environ.get("HF_TOKEN")  

#         #TODO: Create a Groq API key and add it to .env file
        
#         try: 
#             vectorstore=get_vectorstore()
#             if vectorstore is None:
#                 st.error("Failed to load the vector store")

#             qa_chain = RetrievalQA.from_chain_type(
#                 llm=ChatGroq(
#                     api_key=os.environ["HF_TOKEN"],
#                     model="llama-3.1-8b-instant",  # free, fast Groq-hosted model
#                     temperature=0.0,
#                     max_tokens=512,
#                 ),
#                 chain_type="stuff",
#                 retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
#                 return_source_documents=True,
#                 chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
#             )

#             response=qa_chain.invoke({'query':prompt})

#             result=response["result"]
#             source_documents=response["source_documents"]
#             result_to_show=result+"\nSource Docs:\n"+str(source_documents)
#             #response="Hi, I am MediBot!"
#             st.chat_message('assistant').markdown(result_to_show)
#             st.session_state.messages.append({'role':'assistant', 'content': result_to_show})

#         except Exception as e:
#             st.error(f"Error: {str(e)}")

# if __name__ == "__main__":
#     main()


# import os
# import streamlit as st

# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import PromptTemplate
# from langchain_groq import ChatGroq


# # -------------------
# # Load Vectorstore
# # -------------------
# DB_FAISS_PATH = "vectorstore/db_faiss"

# @st.cache_resource
# def get_vectorstore():
#     embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
#     return db


# # -------------------
# # Custom Prompt
# # -------------------
# def set_custom_prompt(custom_prompt_template):
#     return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])


# # -------------------
# # Load LLM
# # -------------------
# HF_TOKEN = os.environ.get("HF_TOKEN")

# def load_llm(temperature):
#     return ChatGroq(
#         api_key=HF_TOKEN,
#         model="llama-3.1-8b-instant",  # Free Groq-hosted model
#         temperature=temperature,
#         max_tokens=512,
#     )


# # -------------------
# # Hybrid Retriever (PDF + CSV)
# # -------------------
# def hybrid_retriever(vectorstore, query, k=3):
#     """Retrieve from PDFs + ensure at least 1 CSV result if available."""
#     results = vectorstore.as_retriever(search_kwargs={"k": k}).get_relevant_documents(query)

#     # Search specifically in CSV docs
#     csv_hits = [
#         doc for doc in vectorstore.similarity_search(query, k=5)
#         if doc.metadata.get("source") == "self_care_tips.csv"
#     ]
#     if csv_hits:
#         results.append(csv_hits[0])  # Always add at least one CSV tip
#     return results


# # -------------------
# # Streamlit App
# # -------------------
# def main():
#     st.set_page_config(
#     page_title="MediBot – AI Medical Assistant",
#     page_icon="🩺",
#     layout="centered"
#     )

#     st.markdown("""
#                 <div style="text-align:center;">
#                 <h1>🩺 MediBot</h1>
#                 <h4>AI-Powered Mental Health Assistant</h4>
#                 <p style="color:gray;">Final Year Project – Medical AI</p>
#                 </div>
#                 """, unsafe_allow_html=True)

#     st.warning(
#     "⚠️ This chatbot provides educational information only and is NOT a substitute for professional medical advice."
#     )


#     with st.sidebar:
#         st.header("📘 Project Information")
#         st.write("Final Year Project")
#         st.write("Domain: Medical AI / NLP")
#         st.write("Tools: Streamlit, LangChain, FAISS, Groq LLM")

#         st.divider()
#         st.header("⚙️ Model Settings")

#         temperature = st.slider(
#         "Creativity (Temperature)",
#         0.0, 1.0, 0.5
#         )

#         k_value = st.slider(
#         "Documents Retrieved (k)",
#         1, 10, 3
#         )

#         show_sources = st.checkbox("Show Source Documents", value=True)

#     if st.button("🧹 Clear Chat"):
#         st.session_state.messages = []
#         st.rerun()


#     # Chat history
#     if "messages" not in st.session_state:
#         st.session_state.messages = []

#     for message in st.session_state.messages:
#         st.chat_message(message["role"]).markdown(message["content"])

#     # Chat input
#     prompt = st.chat_input("Ask me about mental health, self-care, or interventions...")

#     if prompt:
#         st.chat_message("user").markdown(prompt)
#         st.session_state.messages.append({"role": "user", "content": prompt})

#         CUSTOM_PROMPT_TEMPLATE = """
#         Use the pieces of information provided in the context to answer the user's question.
#         If you don’t know the answer, just say that you don’t know — don’t make things up.
#         Only answer from the given context.

#         Context: {context}
#         Question: {question}

#         Start the answer directly. No small talk please.
#         """

#         try:
#             vectorstore = get_vectorstore()
#             if vectorstore is None:
#                 st.error("❌ Failed to load the vector store")
#                 return

#             qa_chain = RetrievalQA.from_chain_type(
#                 llm=load_llm(temperature),
#                 chain_type="stuff",
#                 retriever=vectorstore.as_retriever(search_kwargs={"k": k_value}),
#                 return_source_documents=True,
#                 chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)},
#             )

#             # Use hybrid retriever to include PDFs + CSVs
#             with st.spinner("🔍 Analyzing medical knowledge..."):
#                 docs = hybrid_retriever(vectorstore, prompt, k_value)
#                 response = qa_chain.combine_documents_chain.run(
#                            input_documents=docs,
#                            question=prompt
#                         )


#             # Show assistant reply
#             st.chat_message("assistant").markdown(response)
#             st.session_state.messages.append({"role": "assistant", "content": response})

#             # Show sources if enabled
#             if show_sources and docs:
#                 with st.expander("📚 Source Documents Used"):
#                     for i, doc in enumerate(docs, 1):
#                         source = doc.metadata.get("source", "Unknown")
#                         page = doc.metadata.get("page", "-")
#                         if source.endswith(".csv"):
#                             st.markdown(f"**{i}.** `{source}` (Self-care Tip)")
#                         else:
#                             st.markdown(f"**{i}.** `{source}` (Page {page})")


#         except Exception as e:
#             st.error(f"⚠️ Error: {str(e)}")


# if __name__ == "__main__":
#     main()

# import os
# import chainlit as cl
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import PromptTemplate
# from langchain_groq import ChatGroq

# # -------------------
# # Configuration
# # -------------------
# DB_FAISS_PATH = "vectorstore/db_faiss"
# HF_TOKEN = os.environ.get("HF_TOKEN")

# # Fixed settings (Removing sliders for a "Consumer Product" feel)
# TEMPERATURE = 0
# K_VALUE = 5

# # -------------------
# # Helper Functions
# # -------------------
# def get_vectorstore():
#     """Load the FAISS vector store."""
#     embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
#     return db

# def load_llm():
#     """Load the Groq LLM."""
#     return ChatGroq(
#         api_key=HF_TOKEN,
#         model="llama-3.1-8b-instant",
#         temperature=TEMPERATURE,
#         max_tokens=1024,
#     )

# def set_custom_prompt():
#     """Define the Safety Triage Prompt with EXACT File Mapping."""
#     template = """
#     You are MediBot, a specialized mental health assistant.
#     You have access to 5 specific WHO Clinical Manuals. Use them strictly based on the user's need:

#     RULES:
#     1. IF CRISIS/SUICIDE/HOPELESSNESS: 
#        - Use 'Psychological First Aid' (Source: Psychological First Aid Guide.pdf).
#        - Your ONLY goal is to Listen, De-escalate, and Link to help.
#        - Do NOT suggest complex activities or planning.

#     2. IF SYMPTOMS/DIAGNOSIS: 
#        - Use 'mhGAP Intervention Guide' (Source: mhGAP_Intervention_Guide.pdf).

#     3. IF STRESS/ANXIETY: 
#        - Use exercises from 'Doing What Matters' (Source: Doing What Matters in Times of Stress.pdf).

#     4. IF PREGNANCY: 
#        - Use 'Thinking Healthy' (Source: Thinking Healthy.pdf).

#     5. IF PRACTICAL PROBLEMS (and user is calm): 
#        - Use 'Problem Management Plus' (Source: Problem Management Plus (PM+).pdf).

#     If you don't know the answer, just say that you don't know. Do NOT make up an answer.
    
#     Context: {context}
#     Question: {question}

#     Start the answer directly. No small talk. Be empathetic but professional.
#     """
#     return PromptTemplate(template=template, input_variables=["context", "question"])

# # -------------------
# # Chainlit Events
# # -------------------

# @cl.on_chat_start
# async def start():
#     """Initializes the bot when a user connects."""
    
#     # 1. Send loading message
#     msg = cl.Message(content="Starting MediBot...")
#     await msg.send()

#     # 2. Setup Components
#     vectorstore = await cl.make_async(get_vectorstore)()
#     llm = load_llm()

#     # 3. Create the QA Chain (Corrected)
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=vectorstore.as_retriever(search_kwargs={"k": K_VALUE}),
#         return_source_documents=True,
#         chain_type_kwargs={"prompt": set_custom_prompt()},
#     )

#     # 4. Save to Session (So main() can find them)
#     cl.user_session.set("vectorstore", vectorstore)
#     cl.user_session.set("qa_chain", qa_chain)

#     # 5. Final Welcome Message
#     msg.content = "👋 **Hello! I am MediBot.**\nI can help you with mental health questions using your medical documents.\n\n*Note: I am an AI assistant, not a doctor.*"
#     await msg.update()


# @cl.on_message
# async def main(message: cl.Message):
#     """Refreshed execution for every user message."""
    
#     # 1. Retrieve Components
#     qa_chain = cl.user_session.get("qa_chain")
#     vectorstore = cl.user_session.get("vectorstore")

#     # 2. Run Retrieval (Standard Vector Search)
#     # We do this manually first to get the docs for the UI
#     docs = await cl.make_async(vectorstore.similarity_search)(message.content, k=K_VALUE)
    
#     # DEBUG PRINT (Visible in Terminal)
#     print(f"\n🔍 DEBUG: Found {len(docs)} documents for query: '{message.content}'")

#     # 3. Run the Chain
#     res = await cl.make_async(qa_chain.combine_documents_chain.run)(
#         input_documents=docs,
#         question=message.content
#     )

#     # 4. Process Sources for UI
#     source_elements = []
#     found_sources = set()

#     if docs:
#         for doc in docs:
#             source_name = doc.metadata.get("source", "Unknown")
#             page_num = doc.metadata.get("page", "N/A")
            
#             # Clean Citation Label
#             label = f"{source_name} (Pg {page_num})"
            
#             if label not in found_sources:
#                 source_elements.append(
#                     cl.Text(content=doc.page_content, name=label, display="inline")
#                 )
#                 found_sources.add(label)

#     # 5. Append Sources list to the bottom of the answer
#     if found_sources:
#         res += "\n\n**📚 Clinical References:**\n"
#         for name in found_sources:
#             res += f"* {name}\n"
#     else:
#         res += "\n\n*No specific clinical documents found for this query.*"

#     # 6. Send Response
#     await cl.Message(content=res, elements=source_elements).send()

import streamlit as st
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# -------------------
# 1. Configuration & Setup
# -------------------
st.set_page_config(page_title="MediBot - WHO AI", page_icon="🩺")

# Load Environment Variables
load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
DB_FAISS_PATH = "vectorstore/db_faiss"

# Constants
TEMPERATURE = 0
K_VALUE = 5

# -------------------
# 2. Caching System (The Speed Fix) 🚀
# -------------------
# We use @st.cache_resource so these heavy models load ONLY ONCE
@st.cache_resource
def load_chain():
    print("⚙️ Loading Medical AI Models... (Cached)")
    
    # 1. Load Embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 2. Load Vector Store
    try:
        vectorstore = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"❌ Failed to load Database: {e}")
        st.stop()
        
    # 3. Load LLM
    llm = ChatGroq(
        api_key=HF_TOKEN,
        model="llama-3.1-8b-instant",
        temperature=TEMPERATURE,
        max_tokens=1024
    )
    
    # 4. Define Prompt
    template = """
    You are MediBot, a specialized mental health assistant.
    You have access to 5 specific WHO Clinical Manuals. Use them strictly based on the user's need:

    RULES:
    1. IF CRISIS/SUICIDE/HOPELESSNESS: 
       - Use 'Psychological First Aid' (Source: Psychological First Aid Guide.pdf).
       - Your ONLY goal is to Listen, De-escalate, and Link to help.
       - Do NOT suggest complex activities or planning.

    2. IF SYMPTOMS/DIAGNOSIS: 
       - Use 'mhGAP Intervention Guide' (Source: mhGAP_Intervention_Guide.pdf).

    3. IF STRESS/ANXIETY: 
       - Use exercises from 'Doing What Matters' (Source: Doing What Matters in Times of Stress.pdf).

    4. IF PREGNANCY: 
       - Use 'Thinking Healthy' (Source: Thinking Healthy.pdf).

    5. IF PRACTICAL PROBLEMS (and user is calm): 
       - Use 'Problem Management Plus' (Source: Problem Management Plus (PM+).pdf).

    If you don't know the answer, just say that you don't know. Do NOT make up an answer.
    
    Context: {context}
    Question: {question}

    Start the answer directly. No small talk. Be empathetic but professional.
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # 5. Create Chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": K_VALUE}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return chain

# Load the chain (Cached)
qa_chain = load_chain()

# -------------------
# 3. Session State (Chat History)
# -------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 **Hello! I am MediBot.**\nI can help you with mental health questions using WHO medical documents.\n\n*Note: I am an AI assistant, not a doctor.*"}
    ]

# -------------------
# 4. Streamlit UI
# -------------------
st.title("🩺 MediBot: Mental Health Assistant")
st.markdown("---")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -------------------
# 5. Chat Logic
# -------------------
if user_query := st.chat_input("How can I help you today?"):
    
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing Clinical Protocols..."):
            
            # Run Chain
            response = qa_chain.invoke({"query": user_query})
            result_text = response["result"]
            source_docs = response["source_documents"]

            # Process Sources
            found_sources = set()
            citations = []
            
            for doc in source_docs:
                source_name = doc.metadata.get("source", "Unknown")
                page_num = doc.metadata.get("page", "N/A")
                label = f"{source_name} (Pg {page_num})"
                
                if label not in found_sources:
                    citations.append(f"* {label}")
                    found_sources.add(label)

            # Append Sources to Text
            if citations:
                full_response = result_text + "\n\n**📚 Clinical References:**\n" + "\n".join(citations)
            else:
                full_response = result_text + "\n\n*No specific clinical documents found.*"

            # Display Response
            st.markdown(full_response)
            
            # Save to History
            st.session_state.messages.append({"role": "assistant", "content": full_response})