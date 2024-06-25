import os
import requests
import streamlit as st
import weaviate
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import cohere
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationSummaryMemory
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Weaviate
from trulens_eval import Feedback, Huggingface, Tru, TruChain
from weaviate.embedded import EmbeddedOptions

COHERE_API_KEY = os.environ.get('COHERE_API_KEY')
HUGGINGFACE_ACCESS_TOKEN = os.environ.get('ACCESS_TOKEN')

hugs = Huggingface()
tru = Tru()
chat = cohere.ChatCohere(COHERE_API_KEY)

chain_recorder = None
conversation = None

def handle_conversation(user_input):
    input_dict = {"question": user_input}
    try:
        with chain_recorder as recording:
            response = conversation(input_dict)
            return response.get("answer", "No answer generated")
    except Exception as e:
        return f"An error occured: {e}"

st.sidebar.title("Configuration")
url = st.sidebar.text_input("Enter URL")
submit_button = st.sidebar.button("Submit")

if 'initiated' not in st.session_state:
    st.session_state['initiated'] = False
    st.session_state['messages'] = []

if submit_button or st.session_state['initiated']:
    st.session_state['initiated'] = True

    if url and not conversation:
        # Load and process the document
        res = requests.get(url)
        with open("state_of_the_union.txt", "w") as f:
            f.write(res.text)

        loader = TextLoader('./state_of_the_union.txt')
        documents = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)

        client = weaviate.Client("http://localhost:8000")
        vectorstore = Weaviate.from_documents(client=client, documents=chunks, embedding=OpenAIEmbeddings(), by_text=False)
        
        retriever = vectorstore.as_retriever()

        llm = ChatVertexAI()
        template = """You are an assistant for question-answering tasks..."""
        prompt = ChatPromptTemplate.from_template(template)
        memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)
        conversation = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)

        chain_recorder = TruChain(
            conversation,
            app_id="RAG-System",
            feedbacks=[
                Feedback(hugs.language_match).on_input_output(),
                Feedback(hugs.not_toxic).on_output(),
                Feedback(hugs.pii_detection).on_input(),
                Feedback(hugs.positive_sentiment).on_output(),
            ]
        )

st.title("RAG System powered by TruLens and Vertex AI")

    # Display chat messages
for message in st.session_state.messages:
    st.write(f"{message['role']}: {message['content']}")

    # User input and response handling
    user_prompt = st.text_input("Your question:", key="user_input")
    if st.button("Send", key="send_button"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        # Generate and display assistant response
        assistant_response = handle_conversation(user_prompt)

        # Update and display chat history
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        st.write(f"Assistant: {assistant_response}")

    # Run TruLens dashboard
        tru.run_dashboard()
    else:
        st.write("Please enter a URL and click submit to load the application.")

tru.run_dashboard()