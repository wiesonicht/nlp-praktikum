from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import init_chat_model

class BasicModel:
    def __init__(self, embedding_model_name="thenlper/gte-base", llm_model="llama3.2:1b", llm_provider="ollama", vector_store_path="data/basic-model/"):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            encode_kwargs={"normalize_embeddings": True},
        )

        self.llm = init_chat_model(
            model=llm_model,
            model_provider=llm_provider
        )

        self.vector_store = FAISS.load_local(
            folder_path=vector_store_path,
            embeddings=self.embedding_model,
            allow_dangerous_deserialization=True
        )

    def retrieve(self, query: str):
        """Retrieve information related to a query."""
        retrieved_docs = self.vector_store.similarity_search(query, k=2)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    def generate(self, query: str):
        context, _ = self.retrieve(query)
        system_message_content = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "This is relevant context:"
            f"{context}"
            "\n\n"
            "This is the user query:"
            f"{query}"
        )

        return self.llm.invoke(system_message_content).content

    def chat(self):
        """Interactive chat loop."""
        while True:
            user_input = input("USER: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting chat.")
                break
            print(self.generate(user_input))
