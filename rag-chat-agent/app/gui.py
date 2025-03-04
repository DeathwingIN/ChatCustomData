import os
import streamlit as st
from data_loader import process_documents
from rag_pipeline import generate_response

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_db" not in st.session_state:
    with st.spinner("Initializing system..."):
        st.session_state.vector_db = process_documents()

def main():
    st.title("RAG Chat Agent")
    st.caption("Hybrid AI Assistant - Uses both general knowledge and your documents")
    
    # Chat container
    chat_container = st.container()
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Controls")
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
        
        if st.button("Refresh Knowledge Base"):
            with st.spinner("Updating knowledge..."):
                st.session_state.vector_db = process_documents()
            st.session_state.chat_history = []
            st.rerun()
    
    # File upload handling
    with st.sidebar:
        uploaded_files = st.file_uploader(
            "Upload documents (PDF/TXT)",
            type=["pdf", "txt"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            # Save uploaded files
            for file in uploaded_files:
                file_type = "pdf" if file.type == "application/pdf" else "txt"
                save_dir = f"uploads/{file_type}"
                os.makedirs(save_dir, exist_ok=True)
                file_path = os.path.join(save_dir, file.name)
                
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
            
            # Refresh knowledge base
            with st.spinner("Processing new documents..."):
                st.session_state.vector_db = process_documents()
            st.success("Documents processed!")
            st.session_state.chat_history = []
    
    # Chat interface
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
   # Input handling
    if prompt := st.chat_input("Ask anything..."):
        # Clear previous answer immediately
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.spinner("Analyzing knowledge base..."):
            try:
                # Directly show context-based response
                response = generate_response(prompt, st.session_state.vector_db)
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": response
                })
                
            except Exception as e:
                st.error(f"Error processing request: {str(e)}")
        
        st.rerun()

if __name__ == "__main__":
    main()