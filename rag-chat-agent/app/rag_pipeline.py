from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

def should_use_rag(response_text):
    """Determine if we need to consult RAG based on initial response"""
    uncertainty_keywords = ["don't know", "not sure", "no information", "unclear"]
    return any(keyword in response_text.lower() for keyword in uncertainty_keywords)

def generate_response(query, vector_db):
    """Hybrid response generator with chat memory"""
    try:
        # Initialize core components
        llm = ChatOllama(
            model="deepseek-r1:1.5b",
            temperature=0.7,
            max_tokens=2000,
            num_ctx=4096
        )
        output_parser = StrOutputParser()

        # First try base LLM response
        base_template = """You are a helpful AI assistant. Answer this question:
        Question: {question}
        Answer:"""
        base_prompt = PromptTemplate.from_template(base_template)
        base_chain = base_prompt | llm | output_parser
        initial_response = base_chain.invoke({"question": query})
        
        # Check if RAG needed
        if not vector_db or not should_use_rag(initial_response):
            return initial_response
        
        # If RAG needed
        docs = vector_db.similarity_search(query, k=5)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Enhanced RAG prompt
        rag_template = """Answer the question combining your knowledge with this context:
        Context: {context}
        
        Question: {question}
        Your previous answer was: {initial_response}
        Improve or confirm your answer using the context:"""
        
        rag_prompt = PromptTemplate(
            template=rag_template,
            input_variables=["context", "question", "initial_response"]
        )
        
        rag_chain = rag_prompt | llm | output_parser
        final_response = rag_chain.invoke({
            "context": context,
            "question": query,
            "initial_response": initial_response
        })
        
        return f"{final_response}\n\n(Source: Knowledge Base)"
    
    except Exception as e:
        return f"Error generating response: {str(e)}"