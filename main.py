import streamlit as st

# -----------------------
# Import LangChain pieces
# -----------------------
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_chroma import Chroma as ChromaFactory
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
    get_query_constructor_prompt,
)

def main():
    st.title("Self-Query Retriever (Perplexity.ai)")

    # Retrieve your Perplexity API key from Streamlit secrets
    if "PERPLEXITY_API_KEY" not in st.secrets:
        st.error("Please set the PERPLEXITY_API_KEY in your Streamlit secrets.")
        return

    perplexity_api_key = st.secrets["PERPLEXITY_API_KEY"]

    # ---------------------------
    # Prepare embeddings & docs
    # ---------------------------
    embeddings = OpenAIEmbeddings(
        openai_api_key=perplexity_api_key,
        openai_api_base="https://api.perplexity.ai"
    )

    docs = [
        Document(
            page_content=(
                "A poor but big-hearted man takes orphans into his home. "
                "After discovering his scientist father's invisibility device, "
                "he rises to the occasion and fights to save his children and all of India "
                "from the clutches of a greedy gangster."
            ),
            metadata={
                "year": 2006,
                "director": "Rakesh Roshan",
                "rating": 7.1,
                "genre": "science fiction"
            },
        ),
        Document(
            page_content=(
                "The story of six young Indians who assist an English woman to film a documentary "
                "on the freedom fighters from their past, and the events that lead them to "
                "relive the long-forgotten saga of freedom."
            ),
            metadata={
                "year": 2006,
                "director": "Rakeysh Omprakash Mehra",
                "rating": 9.1,
                "genre": "drama"
            },
        ),
        Document(
            page_content=(
                "A depressed wealthy businessman finds his life changing "
                "after he meets a spunky and care-free young woman."
            ),
            metadata={
                "year": 2007,
                "director": "Anurag Basu",
                "rating": 6.8,
                "genre": "romance"
            },
        ),
        Document(
            page_content=(
                "A schoolteacher's world turns upside down when he realizes that his former student, "
                "who is now a world-famous artist, may have plagiarized his work."
            ),
            metadata={
                "year": 2023,
                "director": "R. Balki",
                "rating": 7.8,
                "genre": "drama"
            },
        ),
        Document(
            page_content=(
                "A man returns to his country in order to marry his childhood sweetheart "
                "and proceeds to create misunderstanding between the families."
            ),
            metadata={
                "year": 1995,
                "director": "Aditya Chopra",
                "rating": 8.1,
                "genre": "romance"
            },
        ),
        Document(
            page_content=(
                "The story of an Indian army officer guarding a picket alone "
                "in the Kargil conflict between India and Pakistan."
            ),
            metadata={
                "year": 2003,
                "director": "J.P. Dutta",
                "rating": 7.9,
                "genre": "war"
            },
        ),
        Document(
            page_content=(
                "Three young men from different parts of India arrive in Mumbai, seeking fame and fortune."
            ),
            metadata={
                "year": 1975,
                "director": "Ramesh Sippy",
                "rating": 8.2,
                "genre": "action"
            },
        ),
        Document(
            page_content=(
                "A simple man from a village falls in love with his new neighbor. "
                "He enlists the help of his musical-theater friends to woo the lovely girl-next-door "
                "away from her music teacher."
            ),
            metadata={
                "year": 1990,
                "director": "Sooraj Barjatya",
                "rating": 7.7,
                "genre": "musical"
            },
        ),
        Document(
            page_content=(
                "A young mute girl from Pakistan loses herself in India with no way to head back. "
                "A devoted man undertakes the task to get her back to her homeland "
                "and unite her with her family."
            ),
            metadata={
                "year": 2015,
                "director": "Kabir Khan",
                "rating": 8.0,
                "genre": "drama"
            },
        ),
        Document(
            page_content=(
                "Three idiots embark on a quest for a lost buddy. This journey takes them on a hilarious "
                "and meaningful adventure through memory lane and gives them a chance "
                "to relive their college days."
            ),
            metadata={
                "year": 2009,
                "director": "Rajkumar Hirani",
                "rating": 9.4,
                "genre": "comedy"
            },
        ),
    ]

    # Create (or load) the Chroma vectorstore
    vectorstore = Chroma.from_documents(docs, embeddings)

    # -----------------------------------
    # Initialize the ChatOpenAI (Perplexity)
    # -----------------------------------
    llm = ChatOpenAI(
        model="llama-3.1-sonar-small-128k-online",
        openai_api_key=perplexity_api_key,
        openai_api_base="https://api.perplexity.ai",
        temperature=0
    )

    # --------------------------------
    # Setup metadata for query parsing
    # --------------------------------
    metadata_field_info = [
        AttributeInfo(
            name="genre",
            description="The genre of the movie.",
            type="string",
        ),
        AttributeInfo(
            name="year",
            description="The year the movie was released",
            type="integer",
        ),
        AttributeInfo(
            name="director",
            description="The name of the movie director",
            type="string",
        ),
        AttributeInfo(
            name="rating",
            description="A 1-10 rating for the movie",
            type="float"
        ),
    ]

    document_content_description = "Brief summary of a movie"

    # -------------------------------------------
    # Build the Self-Query Retriever
    # -------------------------------------------
    retriever = SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vectorstore,
        document_content_description=document_content_description,
        metadata_field_info=metadata_field_info,
    )

    # This chain allows us to see how the query is being interpreted
    query_constructor_prompt = get_query_constructor_prompt(
        document_content_description, metadata_field_info
    )
    output_parser = StructuredQueryOutputParser.from_components()

    # -------------
    # Streamlit UI
    # -------------
    st.write(
        "Enter a natural language query about the type of movie you want to watch.\n\n"
        "Examples:\n"
        " - *I want to watch a movie rated higher than 8.*\n"
        " - *I want to watch a movie by Rajkumar Hirani which is about college life.*\n"
        " - *I want to watch a movie rated higher than 9.0 which has suspense plot.*\n"
    )

    user_query = st.text_input("Your query:")

    if st.button("Retrieve"):
        if user_query.strip():
            # Use the retriever to get relevant documents
            retrieved_docs = retriever.invoke(user_query)

            st.subheader("Retrieved Documents")
            if not retrieved_docs:
                st.write("No matching documents found.")
            else:
                for i, doc in enumerate(retrieved_docs):
                    st.markdown(f"**Document #{i+1}**")
                    st.write(f"**Content:** {doc.page_content}")
                    st.write(f"**Metadata:** {doc.metadata}")
                    st.write("---")
        else:
            st.warning("Please enter a query above.")

if __name__ == "__main__":
    main()
