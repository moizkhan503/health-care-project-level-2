# ==========================
# Import necessary libraries
# ==========================
import streamlit as st                             # For building the web app
from langchain_groq import ChatGroq               # LangChain wrapper for chatting with Groq's LLMs
from langchain_text_splitters import RecursiveCharacterTextSplitter  # For splitting large text into smaller chunks
from langchain_openai import OpenAIEmbeddings     # For generating text embeddings using OpenAI
from langchain_community.vectorstores import FAISS # For storing and searching embeddings efficiently
import faiss                                      # Core FAISS library for vector similarity search
import pymupdf4llm                                # For reading PDF files and converting them to text

# ==========================
# Set up the Streamlit app
# ==========================
# Configure the page title and icon
st.set_page_config(page_title="🤖 Healthcare Assistant", page_icon="🏥")

# Display the main title at the top of the page
st.title("🤖 Healthcare Assistant 🏥")

# =====================================
# Initialize the chat history
# =====================================
# Check if "messages" is already in session state
# (session_state keeps data between user interactions)
if "messages" not in st.session_state:
    # If not, initialize with a welcome message from the assistant
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I'm your healthcare assistant. How can I help you today? 💊"
        }
    ]

# =====================================
# Display the chat messages so far
# =====================================
# Loop through all saved messages and display them in the chat window
for message in st.session_state.messages:
    with st.chat_message(message["role"]):         # Sets the message style (assistant/user)
        st.markdown(message["content"])            # Display the message text

# =====================================
# Initialize the Large Language Model
# =====================================
# Create a ChatGroq object to talk to Groq's LLaMA-4 model
llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.5                                 # Controls creativity of responses
)

# =====================================
# Sidebar: File uploader for PDF
# =====================================
# Create a file uploader in the sidebar that accepts PDF files
uploaded_file = st.sidebar.file_uploader(
    "Upload a healthcare document (PDF)", type="pdf"
)

# =====================================
# Process the uploaded PDF, if any
# =====================================
if uploaded_file:
    with st.spinner("Processing your document..."):
        # -------------------------------
        # Save the uploaded PDF locally
        # -------------------------------
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # -------------------------------
        # Convert PDF to plain text
        # -------------------------------
        text = pymupdf4llm.to_markdown("temp.pdf")

        # -------------------------------
        # Split long text into smaller chunks
        # This helps keep context small enough for the LLM
        # -------------------------------
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,       # Length of each text chunk
            chunk_overlap=200      # Overlapping characters between chunks for context
        )
        docs = text_splitter.split_text(text)   # Returns a list of text chunks

        # -------------------------------
        # Generate embeddings for each chunk
        # -------------------------------
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # -------------------------------
        # Create a vector store (FAISS) from the embeddings
        # This allows fast similarity search later
        # -------------------------------
        vector_store = FAISS.from_texts(docs, embeddings)

        # Notify the user that the document has been processed
        st.sidebar.success("Document processed! You can now ask questions about it.")

# =====================================
# Chat input box at the bottom
# =====================================
# Display a text input field for the user to type a question
if prompt := st.chat_input("Ask your health question..."):

    # -------------------------------
    # Save the user's question to chat history
    # -------------------------------
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    # Display the user's question in the chat window
    with st.chat_message("user"):
        st.markdown(prompt)

    # =====================================
    # Generate and display the assistant's answer
    # =====================================
    with st.chat_message("assistant"):
        # Show a spinner while waiting for the AI to reply
        with st.spinner("Thinking...", show_time=True):

            # -------------------------------
            # If a PDF was uploaded:
            # Search for the most relevant chunks in the document
            # -------------------------------
            if uploaded_file:
                # Search top 2 similar chunks based on the user's question
                relevant_docs = vector_store.similarity_search(prompt, k=2)

                # Combine the contents of those chunks into a single context string
                context = "\n\n".join(
                    [doc.page_content for doc in relevant_docs]
                )

                # Construct the full prompt for the LLM:
                # include the context plus the user's question
                full_prompt = (
                    f"Answer based on this context:\n{context}\n\n"
                    f"Question: {prompt}\nAnswer:"
                )

                # Send the prompt to the Groq LLM and get the answer
                response = llm.invoke(full_prompt)

            else:
                # -------------------------------
                # If no document uploaded:
                # Just send the question to the LLM directly
                # -------------------------------
                response = llm.invoke(
                    f"You are a helpful healthcare assistant. "
                    f"Provide accurate medical advice. "
                    f"Question: {prompt}"
                )

            # Display the assistant's response in the chat window
            st.markdown(response.content)

    # -------------------------------
    # Save the assistant's answer to the chat history
    # -------------------------------
    st.session_state.messages.append(
        {"role": "assistant", "content": response.content}
    )
