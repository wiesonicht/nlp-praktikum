from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import MessagesState, StateGraph
from langchain.chat_models import init_chat_model
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition

EMBEDDING_MODEL_NAME = "thenlper/gte-small"

llm = init_chat_model(
    model="llama3.2:1b",
    model_provider="ollama"
)

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    encode_kwargs={"normalize_embeddings": True},
)

VECTOR_STORE = FAISS.load_local(
    folder_path="data",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)


# @tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = VECTOR_STORE.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

def generate(query: str):
    context = retrieve("Tell me something about ultraviolet B radiation")
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

    return llm.invoke(system_message_content).content

while True:
    user_input = input("USER: ")
    print(generate(user_input))



# def query_or_respond(state: MessagesState):
#     """Decide whether to retrieve or respond directly"""
#     response = llm.invoke([
#         *state["messages"],
#         SystemMessage("Always decide whether to use the 'retrieve' tool when answering questions. "
#                      "Use it for any factual or information-seeking requests.")
#     ])
#
#     # Check if response suggests needing information
#     if "retrieve" in response.content.lower():
#         return {"messages": [response, ToolMessage(tool_call_id="retrieve_123", content=response.content)]}
#     return {"messages": [response]}
#
#
# # def query_or_respond(state: MessagesState):
# #     """Generate tool call for retrieval or respond."""
# #     llm_with_tools = llm.bind_tools([retrieve])
# #     response = llm_with_tools.invoke(state["messages"])
# #     return {"messages": [response]}
#
# tools = ToolNode([retrieve])
#
# def generate(state: MessagesState):
#     """Generate answer."""
#     recent_tool_messages = []
#     for message in reversed(state["messages"]):
#         if message.type == "tool":
#             recent_tool_messages.append(message)
#         else:
#             break
#     tool_messages = recent_tool_messages[::-1]
#
#     docs_content = "\n\n".join(doc.content for doc in tool_messages)
#     system_message_content = (
#         "You are an assistant for question-answering tasks. "
#         "Use the following pieces of retrieved context to answer "
#         "the question. If you don't know the answer, say that you "
#         "don't know. Use three sentences maximum and keep the "
#         "answer concise."
#         "\n\n"
#         f"{docs_content}"
#     )
#     conversation_messages = [
#         message
#         for message in state["messages"]
#         if message.type in ("human", "system")
#         or (message.type == "ai" and not message.tool_calls)
#     ]
#     prompt = [SystemMessage(system_message_content)] + conversation_messages
#
#     response = llm.invoke(prompt)
#     return {"messages": [response]}
#
# graph_builder = StateGraph(MessagesState)
#
# graph_builder.add_node(query_or_respond)
# graph_builder.add_node(tools)
# graph_builder.add_node(generate)
#
# graph_builder.set_entry_point("query_or_respond")
# graph_builder.add_conditional_edges(
#     "query_or_respond",
#     tools_condition,
#     {END: END, "tools": "tools"},
# )
# graph_builder.add_edge("tools", "generate")
# graph_builder.add_edge("generate", END)
#
# graph = graph_builder.compile()
#
# # Specify an ID for the thread
# config = {"configurable": {"thread_id": "abc123"}}
#
#
# while True:
#     input_message = input("USER: ")
#
#     for step in graph.stream(
#         {"messages": [{"role": "user", "content": input_message}]},
#         stream_mode="values",
#     ):
#         step["messages"][-1].pretty_print()
