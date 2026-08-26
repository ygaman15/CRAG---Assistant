"""
rag_engine.py
Hybrid Retrieval (BM25 + Dense) -> Cross-Encoder Re-rank -> CRAG relevance grading -> corrective Tavily web fallback (ALSO graded) -> generation.
"""
import os
from typing import List, Literal
from typing_extensions import TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

from langgraph.graph import END, StateGraph, START

class GraphState(TypedDict):
    question: str
    documents: List[Document]
    web_search_needed: bool
    web_used: bool
    generation: str

class HybridCRAGPipeline:
    """
    Hybrid Retrieval (BM25 + Dense) -> Cross-Encoder Re-rank -> CRAG grading -> corrective web fallback (also graded) -> generation.
    """
    RECALL_K = 8
    GRADE_K = 5
    GEN_K = 3

    def __init__(self, chunks: List[Document]):
        # 1. Dense (semantic) retrieval
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={
                "device": "cpu",
                "model_kwargs": {"low_cpu_mem_usage": False}
            }
        )
        self.vectorstore = Chroma.from_documents(
            documents=chunks, embedding=self.embeddings
        )
        self.vector_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.RECALL_K}
        )

        # 2. Sparse (keyword) retrieval
        self.bm25_retriever = BM25Retriever.from_documents(chunks)
        self.bm25_retriever.k = self.RECALL_K

        # 3. Cross-encoder re-ranker (the precision judge)
        self.reranker = HuggingFaceCrossEncoder(
            model_name="BAAI/bge-reranker-base", 
            model_kwargs={
                "device": "cpu",
                "automodel_args": {"low_cpu_mem_usage": False}
            }
        )

       
        # 4. Two LLMs: small fast grader + strong generator
        self.grader_llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="openai/gpt-oss-20b", # <--- Changed this line
            temperature=0.0,
        )
        self.gen_llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="openai/gpt-oss-120b",
            temperature=0.0,
        )

        # 5. Search Tool & Graph Compilation
        self.web_search_tool = TavilySearchResults(k=3)
        self.graph = self._build_graph()

    def _hybrid_retrieve_and_rerank(self, query: str, top_n: int) -> List[Document]:
        dense_docs = self.vector_retriever.invoke(query)
        sparse_docs = self.bm25_retriever.invoke(query)

        # fuse both retrievers + remove duplicates by content
        seen, candidates = set(), []
        for doc in dense_docs + sparse_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                candidates.append(doc)
        
        if not candidates:
            return []

        # cross-encoder scores each candidate for true relevance
        pairs = [[query, doc.page_content] for doc in candidates]
        scores = self.reranker.score(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_n]]

    def _grade_document(self, doc: Document, question: str) -> bool:
        grader_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a strict relevance grader. Decide if the passage is useful "
                       "to answer the question. Reply with exactly one word: YES or NO."),
            ("human", "Question: {question}\n\nPassage:\n{context}"),
        ])
        chain = grader_prompt | self.grader_llm | StrOutputParser()
        ans = chain.invoke(
            {"question": question, "context": doc.page_content}
        ).strip().upper()
        return ans.startswith("Y")

    def _node_retrieve(self, state: GraphState):
        ranked = self._hybrid_retrieve_and_rerank(state["question"], top_n=self.GRADE_K)
        return {"documents": ranked, "question": state["question"]}

    def _node_grade_documents(self, state: GraphState):
        question = state["question"]
        relevant = [d for d in state["documents"] if self._grade_document(d, question)]
        return {
            "documents": relevant,
            "web_search_needed": len(relevant) == 0,
        }

    def _node_web_search(self, state: GraphState):
        question = state["question"]
        results = self.web_search_tool.invoke({"query": question})
        web_docs = [
            Document(
                page_content=r.get("content", ""),
                metadata={"source": "Tavily", "page": "web"},
            )
            for r in results if r.get("content")
        ]

        # CRAG must be a closed loop: grade web hits too, keep only relevant ones
        relevant_web = [d for d in web_docs if self._grade_document(d, question)]
        return {
            "documents": state.get("documents", []) + relevant_web,
            "web_used": len(relevant_web) > 0,
        }

    def _node_generate(self, state: GraphState):
        question = state["question"]
        docs = state["documents"][:self.GEN_K]

        # HARD GUARD: no evidence survived grading -> never call the LLM
        if not docs:
            return {
                "generation": "I'm sorry, I don't know the answer based on the "
                              "provided documents or web search."
            }

        context_str = "\n\n".join(
            f"--- [Source: {d.metadata.get('source', '?')} | "
            f"Page: {d.metadata.get('page', '?')}] ---\n{d.page_content}"
            for d in docs
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert research assistant. Answer using ONLY the context below. "
                       "If the context is insufficient, say you don't know. Cite sources when useful."),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ])
        chain = qa_prompt | self.gen_llm | StrOutputParser()
        return {
            "generation": chain.invoke(
                {"context": context_str, "question": question}
            )
        }

    def _decide_route(self, state: GraphState) -> Literal["web_search_node", "generate_node"]:
        return "web_search_node" if state["web_search_needed"] else "generate_node"

    def _build_graph(self):
        b = StateGraph(GraphState)
        b.add_node("retrieve_node", self._node_retrieve)
        b.add_node("grade_documents_node", self._node_grade_documents)
        b.add_node("web_search_node", self._node_web_search)
        b.add_node("generate_node", self._node_generate)

        b.add_edge(START, "retrieve_node")
        b.add_edge("retrieve_node", "grade_documents_node")
        b.add_conditional_edges(
            "grade_documents_node",
            self._decide_route,
            {
                "web_search_node": "web_search_node",
                "generate_node": "generate_node",
            },
        )
        b.add_edge("web_search_node", "generate_node")
        b.add_edge("generate_node", END)
        return b.compile()

    def run(self, question: str):
        return self.graph.invoke({"question": question})