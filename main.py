from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
# from vector import retriever

model = OllamaLLM(model="llama3.2")

template = """
You are a helpful assistant. Answer the following question based on the context provided.

Context: {context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    question = input("Enter your question (or 'q' to quit): ")
    if question == 'q':
        break

    result = chain.invoke({"context": [], "question": question})
    print(result)

