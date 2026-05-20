import streamlit as st
from dotenv import load_dotenv
import tempfile
import os

load_dotenv()

from rag.loader import load_pdf
from rag.embedding import get_vector_store
from rag.chain import execute_chain, get_sources



st.set_page_config(page_title="DocuMind", page_icon="🧠", layout="wide")

st.title("🧠 DocuMind")
st.caption("Chat with any PDF using AI")

# ---- SIDEBAR ----
with st.sidebar:
    st.header("📄 Upload your PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file:
        if "vector_store" not in st.session_state:
            with st.spinner("Processing PDF..."):

                # save uploaded file temporarily to disk
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                # loader → embedding → store in session
                chunks = load_pdf(tmp_path)
                st.session_state.vector_store = get_vector_store(chunks)
                os.unlink(tmp_path)  # clean up temp file

            st.success(f"✅ Ready! Loaded {len(chunks)} chunks.")
        else:
            st.info("PDF already processed. Ask away!")

    # reset button
    if st.button("🗑️ Clear and upload new PDF"):
        for key in ["vector_store", "messages"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# ---- CHAT AREA ----

# initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# handle new question
if question := st.chat_input("Ask something about your PDF..."):

    if "vector_store" not in st.session_state:
        st.warning("Please upload a PDF first.")
    else:
        # show user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                retriever = st.session_state.vector_store.as_retriever(
                    search_kwargs={"k": 4}
                )
                answer = execute_chain(retriever, question)
                sources = get_sources(retriever, question)

            st.markdown(answer)

            # show sources
            with st.expander("📚 Sources"):
                for i, doc in enumerate(sources):
                    page = doc.metadata.get("page", "?")
                    st.markdown(f"**Chunk {i+1} — Page {page + 1}**")
                    st.caption(doc.page_content[:300] + "...")

        st.session_state.messages.append({"role": "assistant", "content": answer})