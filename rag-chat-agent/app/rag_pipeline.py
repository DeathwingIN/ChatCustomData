from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def is_relevant_context(query, context_text):
    """Determine if context is actually relevant to the query"""
    relevance_prompt = PromptTemplate.from_template(
        "Determine if this context is relevant to the query. Answer ONLY 'yes' or 'no'.\n"
        "Query: {query}\nContext: {context}\nRelevant?"
    )
    
    llm = ChatOllama(model="deepseek-r1:1.5b", temperature=0)
    chain = relevance_prompt | llm | StrOutputParser()
    
    response = chain.invoke({
        "query": query,
        "context": context_text[:1000]  # First 1000 chars for efficiency
    }).strip().lower()
    
    return "yes" in response

def generate_response(query, vector_db):
    """Intelligent hybrid response generator"""
    try:
        llm = ChatOllama(
            model="deepseek-r1:1.5b",
            temperature=0.5,
            max_tokens=2000,
            num_ctx=4096
        )
        output_parser = StrOutputParser()

        # Try vector DB first with relevance check
        if vector_db:
            docs = vector_db.similarity_search_with_relevance_scores(query, k=5)
            
            # Filter documents with minimum similarity score
            relevant_docs = [doc for doc, score in docs if score > 0.25]
            context = "\n\n".join([d.page_content for d in relevant_docs[:3]])  # Top 3

            if context and is_relevant_context(query, context):
                response_template = """[Priority to Context] Answer using this information:
                {context}
                
                [Question] {question}
                
                Requirements:
                - If context is about a person/organization, it's likely the main subject
                - Never mention you're referring to documents
                - Add general knowledge only if needed for completeness
                """
                
                prompt = PromptTemplate(
                    template=response_template,
                    input_variables=["context", "question"]
                )
                chain = prompt | llm | output_parser
                return chain.invoke({"context": context, "question": query})

        # General knowledge fallback
        base_template = """Answer this concisely: {question}"""
        base_prompt = PromptTemplate.from_template(base_template)
        return (base_prompt | llm | output_parser).invoke({"question": query})

    except Exception as e:
        return f"Error generating response: {str(e)}"