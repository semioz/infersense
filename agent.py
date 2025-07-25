import logging
import os

from langchain.tools.retriever import create_retriever_tool
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
)
from langfuse import get_client
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from openinference.instrumentation.langchain import LangChainInstrumentor
from supabase.client import Client, create_client

from tool import tools

LangChainInstrumentor().instrument()
#todo
#os.environ["LANGFUSE_PUBLIC_KEY"]
#os.environ["LANGFUSE_SECRET_KEY"]
#os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com"

logger = logging.getLogger(__name__)
langfuse = get_client()

with open("system_prompt.txt", encoding="utf-8") as f:
    system_prompt = f.read()
print(system_prompt)

sys_msg = SystemMessage(content=system_prompt)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

supabase: Client = create_client(
    os.environ.get("SUPABASE_URL"),
    os.environ.get("SUPABASE_SERVICE_KEY"))
vector_store = SupabaseVectorStore(
    client=supabase,
    embedding= embeddings,
    table_name="documents",
    query_name="match_documents_langchain",
)
create_retriever_tool = create_retriever_tool(
    retriever=vector_store.as_retriever(),
    name="Question Search",
    description="A tool to retrieve similar questions from a vector store.",
)

def build_graph(provider: str = "groq"):
    """Build the graph"""
    if provider == "groq":
        llm = ChatGroq(model="qwen/qwen3-32b", temperature=0)
    elif provider == "huggingface":
        llm = ChatHuggingFace(
            llm=HuggingFaceEndpoint(
                repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                task="text-generation",
                max_new_tokens=1024,
                temperature=0,
            ),
            verbose=True,
        )
    else:
        raise ValueError("Invalid provider. Choose 'groq' or 'huggingface'.")
    llm_with_tools = llm.bind_tools(tools)

    # Node
    def assistant(state: MessagesState):
        """Assistant node"""
        return {"messages": [llm_with_tools.invoke(state["messages"])]}


    def retriever(state: MessagesState):
        """Retriever node"""
        similar_question = vector_store.similarity_search(state["messages"][0].content)
        example_msg = HumanMessage(
            content=f"Here I provide a similar question and answer for reference: \n\n{similar_question[0].page_content}",
        )
        return {"messages": [sys_msg] + state["messages"] + [example_msg]}


    builder = StateGraph(MessagesState)
    builder.add_node("retriever", retriever)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "retriever")
    builder.add_edge("retriever", "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )
    builder.add_edge("tools", "assistant")

    return builder.compile()


if __name__ == "__main__":
    question = "If Ada Lovelace was born in 1815 and Charles Babbage died in 1871, how old was she when he died?"
    graph = build_graph(provider="groq")
    messages = [HumanMessage(content=question)]
    messages = graph.invoke({"messages": messages})
    for m in messages["messages"]:
        m.pretty_print()
