import streamlit as st # type: ignore
import logging, json
from tools.patient_tool import get_patient_report
from agents.crew import receptionist, clinical, llm  # as defined above
from langchain_community.embeddings import HuggingFaceEmbeddings # type: ignore
from langchain_community.vectorstores import Chroma # type: ignore
from langchain import LLMChain, PromptTemplate

# Logging
logging.basicConfig(filename="logs/system.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

st.title("Post-Discharge Nephrology AI Assistant")
st.caption("This is an AI assistant for educational purposes only. Always consult healthcare professionals.")

if "chat" not in st.session_state:
    st.session_state.chat = [{"role":"assistant","content":"Hello! I'm your post-discharge care assistant. What's your name?"}]

for m in st.session_state.chat:
    if m["role"]=="assistant":
        st.markdown(f"**Assistant:** {m['content']}")
    else:
        st.markdown(f"**You:** {m['content']}")

user_input = st.text_input("Type here...")

if st.button("Send"):
    st.session_state.chat.append({"role":"user","content":user_input})
    # simple routing: if user says name, receptionist fetches report
    if len(user_input.split()) <= 3 and any(word.lower() in user_input.lower() for word in ["john", "doe", "smith", "name"]):
        # treat as name lookup
        res = get_patient_report(user_input.strip())
        if res.get("error"):
            reply = res.get("message")
        else:
            report = res["report"]
            reply = f"Hi {report['patient_name']} — I found a discharge on {report['discharge_date']} for {report['primary_diagnosis']}. How are you feeling? Are you following your meds?"
            logging.info(f"Receptionist retrieved report for {report['patient_name']}")
    else:
        # route to clinical: use RAG retriever + fallback to web if needed (simplified)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectordb = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
        retriever = vectordb.as_retriever(search_kwargs={"k":3})
        # use the retriever to get context (LangChain pattern)
        docs = retriever.get_relevant_documents(user_input)
        # Build prompt including source text from docs
        doc_text = "\n\n".join([f"[SOURCE: {d.metadata.get('source','page')}] {d.page_content[:800]}" for d in docs])
        prompt = f"You are a clinical assistant. Use the following references to answer question. References:\n{doc_text}\n\nQuestion: {user_input}\nAnswer with short guidance and cite sources."
        prompt_template = PromptTemplate(input_variables=["text"], template=prompt)
        chain = LLMChain(llm=llm, prompt=prompt_template)
        answer = chain.run({"text": user_input})
        # If the answer finds no doc evidence, fallback to web search
        if not docs:
            web = __import__("tools.web_search_tool").web_search_tool.web_search
            webres = web(user_input)
            answer = f"Web fallback used. See results: {webres}"
            logging.info("Clinical used web search fallback")
        reply = answer
    st.session_state.chat.append({"role":"assistant","content":reply})
    st.experimental_rerun()
