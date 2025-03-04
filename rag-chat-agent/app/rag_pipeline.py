from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def generate_response(query, vector_db):
    """Prioritize vector DB content while maintaining general knowledge"""
    try:
        llm = ChatOllama(
            model="deepseek-r1:1.5b",
            temperature=0.3,  # Lower temperature for more factual responses
            max_tokens=2000,
            num_ctx=4096
        )
        output_parser = StrOutputParser()

        # Always check vector DB first
        if vector_db:
            docs = vector_db.similarity_search(query, k=8)  # Increased context chunks
            context = "\n\n".join([d.page_content for d in docs])
            
            # Enhanced prompt template
            rag_template = """You must prioritize this context when answering:
            Context: {context}
            
            Question: {question}
            
            If context is relevant:
            - Give detailed response using context
            - Never say "based on context"
            - If needed, add general knowledge to enhance answer
            
            If context is irrelevant:
            - Answer normally using your knowledge
            """
            
            rag_prompt = PromptTemplate(
                template=rag_template,
                input_variables=["context", "question"]
            )
            
            rag_chain = (
                {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                | rag_prompt
                | llm
                | output_parser
            )
            return rag_chain.invoke({"context": context, "question": query})
        
        # Fallback to general knowledge
        base_template = """Answer this question: {question}"""
        base_prompt = PromptTemplate.from_template(base_template)
        base_chain = base_prompt | llm | output_parser
        return base_chain.invoke({"question": query})
    
    except Exception as e:
        return f"Error generating response: {str(e)}"