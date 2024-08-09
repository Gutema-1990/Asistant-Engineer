from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.utils import Secret
from haystack.dataclasses import Document
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore

from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter

def pipeline_setup(question):
    pipeline = Pipeline()
    document_store = InMemoryDocumentStore()
    pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder())
    pipeline.add_component("writer", DocumentWriter(document_store))
    pipeline.connect("embedder", "writer")

    pipeline.run({"embedder": {"documents": [Document(content="Sushi is a traditional dish from Japan"),
                                            Document(content="Tacos are popular in Mexico"),
                                            Document(content="Pizza originated in Italy")]}})
    pipeline.connect("embedder", "writer")

    embedder = SentenceTransformersTextEmbedder()
    retriever = InMemoryEmbeddingRetriever(document_store=document_store)

    template = """Given the following context, answer the question.Context:
    {% for document in documents %}{{ document.content }}{% endfor %}
    Question: {{query}}"""
    prompt_builder = PromptBuilder(template=template)
    # generator = OpenAIGenerator(model="gpt-4") 


    generator = OpenAIGenerator(
        model="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
        api_base_url="http://10.2.125.37:1234/v1",
        api_key=Secret.from_token("lm-studio")
    )

    rag_pipeline = Pipeline()
    rag_pipeline.add_component("embedder", embedder)
    rag_pipeline.add_component("retriever", retriever)
    rag_pipeline.add_component("prompt_builder", prompt_builder)
    rag_pipeline.add_component("generator", generator)

    rag_pipeline.connect("embedder", "retriever")
    rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder", "generator")

    # question = "What is the traditional dish of Japan?"
    response = rag_pipeline.run({"embedder": {"text": question},
    "prompt_builder": {"query": question}})
    return response


import streamlit as st

# Streamlit app layout
st.title("Simple Q&A Chatbot")
st.write("Ask me a question!")

# Function to generate responses
def chatbot_response(question):
    response = pipeline_setup(question)
    
    return response

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# User input
user_input = st.chat_input("Type your question here:")
# React to user input
if user_input :
    # Display user message in chat message container
    st.chat_message("user").markdown(user_input)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    response = chatbot_response(user_input)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response['generator']['replies'][0])
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response['generator']['replies'][0]})





# if user_input:
#     # Get bot response
#     response = chatbot_response(user_input)
#     st.write(f"**Bot:** {response['generator']['replies'][0]}")