# backend/core/rag_manager.py
# ============================================
# Advanced RAG System
# ============================================
# Implements:
# - Hierarchical chunking (parent/child)
# - Contextual retrieval (Anthropic technique)
# - Hybrid search (semantic + BM25)
# - CRAG (Corrective RAG)
# - GraphRAG for cross-document reasoning
# - Agentic RAG routing
# ============================================

import os
import json
import asyncio
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, AsyncGenerator, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# Configuration
VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "data/vectorstores")
RAG_CHUNK_SIZE_PARENT = int(os.getenv("RAG_CHUNK_SIZE_PARENT", "2048"))
RAG_CHUNK_SIZE_CHILD = int(os.getenv("RAG_CHUNK_SIZE_CHILD", "512"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
ENABLE_HYBRID_SEARCH = os.getenv("ENABLE_HYBRID_SEARCH", "true").lower() == "true"
ENABLE_CONTEXTUAL_RETRIEVAL = os.getenv("ENABLE_CONTEXTUAL_RETRIEVAL", "true").lower() == "true"
ENABLE_CRAG = os.getenv("ENABLE_CRAG", "true").lower() == "true"
ENABLE_GRAPHRAG = os.getenv("ENABLE_GRAPHRAG", "false").lower() == "true"


@dataclass
class Chunk:
    """Document chunk with metadata."""
    id: str
    content: str
    context_prefix: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)


@dataclass
class RetrievalResult:
    """Result from retrieval."""
    chunk: Chunk
    score: float
    relevance_grade: Optional[str] = None  # For CRAG


@dataclass
class RAGResponse:
    """Response from RAG query."""
    answer: str
    sources: List[Dict[str, Any]]
    retrieval_strategy: str
    chunks_used: int
    confidence: float


# ============================================
# DOCUMENT EXTRACTION
# ============================================

class DocumentExtractor:
    """Extract text from various document formats."""
    
    @staticmethod
    async def extract_from_pdf(filepath: str) -> Tuple[str, Dict[str, Any]]:
        """Extract text from PDF with structure preservation."""
        try:
            import fitz  # PyMuPDF
            
            loop = asyncio.get_event_loop()
            
            def _extract():
                doc = fitz.open(filepath)
                text_parts = []
                metadata = {
                    "page_count": len(doc),
                    "title": doc.metadata.get("title", ""),
                    "author": doc.metadata.get("author", "")
                }
                
                for page_num, page in enumerate(doc):
                    # Extract text with layout preservation
                    text = page.get_text("text", sort=True)
                    if text.strip():
                        text_parts.append(f"\n--- Page {page_num + 1} ---\n{text}")
                
                doc.close()
                return "\n".join(text_parts), metadata
            
            return await loop.run_in_executor(None, _extract)
            
        except Exception as e:
            print(f"PDF extraction error: {e}")
            return "", {"error": str(e)}
    
    @staticmethod
    async def extract_from_docx(filepath: str) -> Tuple[str, Dict[str, Any]]:
        """Extract text from DOCX."""
        try:
            from docx import Document
            
            loop = asyncio.get_event_loop()
            
            def _extract():
                doc = Document(filepath)
                text_parts = []
                
                for para in doc.paragraphs:
                    if para.text.strip():
                        # Check for heading styles
                        if para.style.name.startswith('Heading'):
                            level = para.style.name.replace('Heading ', '')
                            text_parts.append(f"\n{'#' * int(level) if level.isdigit() else '#'} {para.text}\n")
                        else:
                            text_parts.append(para.text)
                
                # Extract from tables
                for table in doc.tables:
                    for row in table.rows:
                        row_text = " | ".join(cell.text.strip() for cell in row.cells)
                        if row_text.strip():
                            text_parts.append(row_text)
                
                return "\n".join(text_parts), {"type": "docx"}
            
            return await loop.run_in_executor(None, _extract)
            
        except Exception as e:
            print(f"DOCX extraction error: {e}")
            return "", {"error": str(e)}
    
    @staticmethod
    async def extract_from_file(filepath: str) -> Tuple[str, Dict[str, Any]]:
        """Extract text from any supported file."""
        ext = Path(filepath).suffix.lower()
        
        if ext == '.pdf':
            return await DocumentExtractor.extract_from_pdf(filepath)
        elif ext == '.docx':
            return await DocumentExtractor.extract_from_docx(filepath)
        elif ext in ['.txt', '.md', '.html']:
            async with asyncio.get_event_loop().run_in_executor(
                None, lambda: open(filepath, 'r', encoding='utf-8').read()
            ) as f:
                content = f
            return content, {"type": ext[1:]}
        else:
            return "", {"error": f"Unsupported format: {ext}"}


# ============================================
# HIERARCHICAL CHUNKING
# ============================================

class HierarchicalChunker:
    """Create hierarchical chunks with context augmentation."""
    
    def __init__(
        self,
        parent_size: int = RAG_CHUNK_SIZE_PARENT,
        child_size: int = RAG_CHUNK_SIZE_CHILD,
        overlap: int = RAG_CHUNK_OVERLAP
    ):
        self.parent_size = parent_size
        self.child_size = child_size
        self.overlap = overlap
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate (4 chars per token)."""
        return len(text) // 4
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    async def create_chunks(
        self,
        text: str,
        source: str,
        generate_context: bool = True
    ) -> List[Chunk]:
        """Create hierarchical chunks from text."""
        sentences = self._split_into_sentences(text)
        
        # First pass: Create parent chunks
        parent_chunks = []
        current_parent = []
        current_tokens = 0
        
        for sentence in sentences:
            sent_tokens = self._estimate_tokens(sentence)
            
            if current_tokens + sent_tokens > self.parent_size and current_parent:
                parent_text = " ".join(current_parent)
                parent_id = hashlib.md5(parent_text.encode()).hexdigest()[:12]
                
                parent_chunks.append({
                    "id": parent_id,
                    "text": parent_text,
                    "sentences": current_parent.copy()
                })
                
                # Keep overlap
                overlap_sents = []
                overlap_tokens = 0
                for s in reversed(current_parent):
                    overlap_tokens += self._estimate_tokens(s)
                    if overlap_tokens > self.overlap:
                        break
                    overlap_sents.insert(0, s)
                
                current_parent = overlap_sents
                current_tokens = overlap_tokens
            
            current_parent.append(sentence)
            current_tokens += sent_tokens
        
        # Don't forget last parent
        if current_parent:
            parent_text = " ".join(current_parent)
            parent_id = hashlib.md5(parent_text.encode()).hexdigest()[:12]
            parent_chunks.append({
                "id": parent_id,
                "text": parent_text,
                "sentences": current_parent
            })
        
        # Second pass: Create child chunks with context
        all_chunks = []
        
        for parent in parent_chunks:
            # Generate context for parent
            parent_context = ""
            if generate_context and ENABLE_CONTEXTUAL_RETRIEVAL:
                parent_context = await self._generate_context_prefix(
                    parent["text"], source
                )
            
            parent_chunk = Chunk(
                id=parent["id"],
                content=parent["text"],
                context_prefix=parent_context,
                metadata={
                    "source": source,
                    "type": "parent",
                    "created_at": datetime.now().isoformat()
                },
                children_ids=[]
            )
            
            # Create child chunks
            child_sentences = []
            child_tokens = 0
            
            for sentence in parent["sentences"]:
                sent_tokens = self._estimate_tokens(sentence)
                
                if child_tokens + sent_tokens > self.child_size and child_sentences:
                    child_text = " ".join(child_sentences)
                    child_id = hashlib.md5(child_text.encode()).hexdigest()[:12]
                    
                    # Child context inherits from parent + its own
                    child_context = f"[From: {source}] [Parent context: {parent_context[:200]}...]" if parent_context else f"[From: {source}]"
                    
                    child_chunk = Chunk(
                        id=child_id,
                        content=child_text,
                        context_prefix=child_context,
                        metadata={
                            "source": source,
                            "type": "child",
                            "parent_id": parent["id"],
                            "created_at": datetime.now().isoformat()
                        },
                        parent_id=parent["id"]
                    )
                    
                    all_chunks.append(child_chunk)
                    parent_chunk.children_ids.append(child_id)
                    
                    child_sentences = []
                    child_tokens = 0
                
                child_sentences.append(sentence)
                child_tokens += sent_tokens
            
            # Last child chunk
            if child_sentences:
                child_text = " ".join(child_sentences)
                child_id = hashlib.md5(child_text.encode()).hexdigest()[:12]
                
                child_context = f"[From: {source}]"
                if parent_context:
                    child_context = f"[From: {source}] [Parent: {parent_context[:200]}...]"
                
                child_chunk = Chunk(
                    id=child_id,
                    content=child_text,
                    context_prefix=child_context,
                    metadata={
                        "source": source,
                        "type": "child",
                        "parent_id": parent["id"]
                    },
                    parent_id=parent["id"]
                )
                
                all_chunks.append(child_chunk)
                parent_chunk.children_ids.append(child_id)
            
            all_chunks.append(parent_chunk)
        
        return all_chunks
    
    async def _generate_context_prefix(self, text: str, source: str) -> str:
        """Generate context prefix using LLM (Anthropic's contextual retrieval)."""
        try:
            from backend.services import ollama_service
            
            prompt = f"""Summarize this text chunk in 1-2 sentences for search context.
Focus on: key topics, entities, technical terms.

Source: {source}
Text: {text[:1000]}

Context summary:"""
            
            response = await ollama_service.generate_text(
                prompt,
                temperature=0.3,
                max_tokens=100
            )
            
            return response.strip()
            
        except Exception as e:
            print(f"Context generation error: {e}")
            return ""


# ============================================
# HYBRID SEARCH (Semantic + BM25)
# ============================================

class HybridSearcher:
    """Hybrid search combining dense vectors and BM25."""
    
    def __init__(self):
        self.bm25 = None
        self.corpus = []
        self.chunk_map: Dict[str, Chunk] = {}
    
    def index_chunks(self, chunks: List[Chunk]):
        """Build BM25 index."""
        from rank_bm25 import BM25Okapi
        
        self.corpus = []
        self.chunk_map = {}
        
        for chunk in chunks:
            # Combine content and context for indexing
            full_text = f"{chunk.context_prefix} {chunk.content}"
            tokens = full_text.lower().split()
            self.corpus.append(tokens)
            self.chunk_map[chunk.id] = chunk
        
        self.bm25 = BM25Okapi(self.corpus)
    
    def bm25_search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using BM25."""
        if not self.bm25:
            return []
        
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k results
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        chunk_ids = list(self.chunk_map.keys())
        
        for idx in top_indices:
            if idx < len(chunk_ids) and scores[idx] > 0:
                results.append((chunk_ids[idx], scores[idx]))
        
        return results
    
    def hybrid_search(
        self,
        semantic_results: List[Tuple[str, float]],
        bm25_results: List[Tuple[str, float]],
        semantic_weight: float = 0.6
    ) -> List[Tuple[str, float]]:
        """Combine semantic and BM25 results."""
        scores = {}
        
        # Normalize and combine
        max_semantic = max([s for _, s in semantic_results], default=1.0)
        max_bm25 = max([s for _, s in bm25_results], default=1.0)
        
        for chunk_id, score in semantic_results:
            norm_score = score / max_semantic if max_semantic > 0 else 0
            scores[chunk_id] = scores.get(chunk_id, 0) + norm_score * semantic_weight
        
        for chunk_id, score in bm25_results:
            norm_score = score / max_bm25 if max_bm25 > 0 else 0
            scores[chunk_id] = scores.get(chunk_id, 0) + norm_score * (1 - semantic_weight)
        
        # Sort by combined score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return ranked


# ============================================
# CRAG (Corrective RAG)
# ============================================

class CRAGEvaluator:
    """Corrective RAG - evaluate and improve retrieval."""
    
    async def grade_relevance(
        self,
        query: str,
        chunks: List[Chunk]
    ) -> List[Tuple[Chunk, str, float]]:
        """Grade retrieved chunks for relevance."""
        try:
            from backend.services import ollama_service
            
            graded = []
            
            for chunk in chunks:
                prompt = f"""Grade the relevance of this retrieved document to the query.
Respond with ONLY one word: "relevant", "partially_relevant", or "irrelevant"

Query: {query}

Document: {chunk.content[:500]}

Grade:"""
                
                response = await ollama_service.generate_text(
                    prompt,
                    temperature=0.1,
                    max_tokens=10
                )
                
                grade = response.strip().lower()
                
                # Map grade to score
                score_map = {
                    "relevant": 1.0,
                    "partially_relevant": 0.5,
                    "partially": 0.5,
                    "irrelevant": 0.0
                }
                
                score = score_map.get(grade, 0.5)
                graded.append((chunk, grade, score))
            
            return graded
            
        except Exception as e:
            print(f"CRAG grading error: {e}")
            return [(c, "unknown", 0.5) for c in chunks]
    
    async def reformulate_query(self, original_query: str) -> str:
        """Reformulate query for better retrieval."""
        try:
            from backend.services import ollama_service
            
            prompt = f"""The following query didn't retrieve good results. 
Reformulate it to be more specific and searchable.
Return ONLY the reformulated query, nothing else.

Original query: {original_query}

Reformulated query:"""
            
            response = await ollama_service.generate_text(
                prompt,
                temperature=0.3,
                max_tokens=100
            )
            
            return response.strip()
            
        except Exception as e:
            print(f"Query reformulation error: {e}")
            return original_query


# ============================================
# KNOWLEDGE GRAPH (GraphRAG)
# ============================================

class KnowledgeGraphBuilder:
    """Build knowledge graph for GraphRAG."""
    
    def __init__(self):
        self.graph = None
    
    async def load_graph(self, path: str):
        """Load existing graph."""
        import networkx as nx
        
        graph_file = Path(path) / "knowledge_graph.json"
        if graph_file.exists():
            with open(graph_file) as f:
                data = json.load(f)
                self.graph = nx.node_link_graph(data)
        else:
            self.graph = nx.DiGraph()
    
    async def save_graph(self, path: str):
        """Save graph to file."""
        import networkx as nx
        
        Path(path).mkdir(parents=True, exist_ok=True)
        graph_file = Path(path) / "knowledge_graph.json"
        
        with open(graph_file, 'w') as f:
            json.dump(nx.node_link_data(self.graph), f)
    
    async def extract_entities_and_relations(
        self,
        text: str
    ) -> Tuple[List[Dict], List[Dict]]:
        """Extract entities and relations using LLM."""
        try:
            from backend.services import ollama_service
            
            prompt = f"""Extract entities and relationships from this text.
Return JSON with two arrays: "entities" and "relations".

Text: {text[:2000]}

Format:
{{
  "entities": [
    {{"name": "entity_name", "type": "PERSON/ORG/CONCEPT/ROBOT_PART/API"}}
  ],
  "relations": [
    {{"source": "entity1", "target": "entity2", "type": "relation_type"}}
  ]
}}

JSON:"""
            
            response = await ollama_service.generate_text(
                prompt,
                temperature=0.2,
                max_tokens=500
            )
            
            # Parse JSON
            import re
            match = re.search(r'\{[\s\S]*\}', response)
            if match:
                data = json.loads(match.group())
                return data.get("entities", []), data.get("relations", [])
            
        except Exception as e:
            print(f"Entity extraction error: {e}")
        
        return [], []
    
    async def add_to_graph(self, chunks: List[Chunk]):
        """Add chunks to knowledge graph."""
        import networkx as nx
        
        if self.graph is None:
            self.graph = nx.DiGraph()
        
        for chunk in chunks:
            entities, relations = await self.extract_entities_and_relations(chunk.content)
            
            # Add entities as nodes
            for entity in entities:
                self.graph.add_node(
                    entity["name"],
                    type=entity.get("type", "UNKNOWN"),
                    source=chunk.metadata.get("source", "")
                )
            
            # Add relations as edges
            for rel in relations:
                if rel["source"] in self.graph and rel["target"] in self.graph:
                    self.graph.add_edge(
                        rel["source"],
                        rel["target"],
                        type=rel.get("type", "RELATED_TO")
                    )
    
    async def query_graph(
        self,
        entity: str,
        hops: int = 2
    ) -> List[Dict]:
        """Query graph for related entities."""
        import networkx as nx
        
        if self.graph is None or entity not in self.graph:
            return []
        
        # Get neighbors within N hops
        related = []
        visited = set()
        queue = [(entity, 0)]
        
        while queue:
            current, depth = queue.pop(0)
            
            if current in visited or depth > hops:
                continue
            
            visited.add(current)
            
            if current != entity:
                node_data = self.graph.nodes[current]
                related.append({
                    "entity": current,
                    "type": node_data.get("type"),
                    "depth": depth
                })
            
            for neighbor in self.graph.neighbors(current):
                if neighbor not in visited:
                    queue.append((neighbor, depth + 1))
        
        return related


# ============================================
# MAIN RAG MANAGER
# ============================================

class RAGManager:
    """Main RAG manager combining all components."""
    
    def __init__(self):
        self.extractor = DocumentExtractor()
        self.chunker = HierarchicalChunker()
        self.searcher = HybridSearcher()
        self.crag = CRAGEvaluator()
        self.graph_builder = KnowledgeGraphBuilder()
        self.vectorstore = None
        self.vectorstore_path = None
    
    async def process_document(
        self,
        filepath: str,
        progress_callback: Optional[Callable[[str], Any]] = None
    ) -> str:
        """Process document and create vectorstore."""
        if progress_callback:
            await progress_callback("Extracting text...")
        
        # Extract text
        text, metadata = await self.extractor.extract_from_file(filepath)
        
        if not text:
            raise ValueError("Failed to extract text from document")
        
        if progress_callback:
            await progress_callback("Creating hierarchical chunks...")
        
        # Create chunks
        source = Path(filepath).name
        chunks = await self.chunker.create_chunks(text, source)
        
        if progress_callback:
            await progress_callback(f"Created {len(chunks)} chunks. Generating embeddings...")
        
        # Generate embeddings
        from backend.services import ollama_service
        
        for i, chunk in enumerate(chunks):
            full_text = f"{chunk.context_prefix} {chunk.content}"
            chunk.embedding = await ollama_service.generate_embedding(full_text)
            
            if progress_callback and i % 10 == 0:
                await progress_callback(f"Embedded {i}/{len(chunks)} chunks...")
        
        # Build indexes
        if progress_callback:
            await progress_callback("Building search indexes...")
        
        # FAISS vectorstore
        await self._build_faiss_index(chunks)
        
        # BM25 index
        self.searcher.index_chunks(chunks)
        
        # Knowledge graph (if enabled)
        if ENABLE_GRAPHRAG:
            if progress_callback:
                await progress_callback("Building knowledge graph...")
            await self.graph_builder.add_to_graph(chunks)
        
        # Save to disk
        vectorstore_path = self._get_vectorstore_path(source)
        await self._save_vectorstore(vectorstore_path, chunks)
        
        if progress_callback:
            await progress_callback("Processing complete!")
        
        return vectorstore_path
    
    async def _build_faiss_index(self, chunks: List[Chunk]):
        """Build FAISS index from chunks."""
        try:
            import faiss
            import numpy as np
            
            embeddings = [c.embedding for c in chunks if c.embedding]
            
            if not embeddings:
                return
            
            dimension = len(embeddings[0])
            index = faiss.IndexFlatL2(dimension)
            
            embeddings_np = np.array(embeddings).astype('float32')
            index.add(embeddings_np)
            
            self.vectorstore = {
                "index": index,
                "chunks": {c.id: c for c in chunks if c.embedding}
            }
            
        except ImportError:
            print("FAISS not installed, using simple search")
    
    def _get_vectorstore_path(self, source: str) -> str:
        """Get vectorstore path for a source."""
        safe_name = "".join(c if c.isalnum() else "_" for c in source)
        return str(Path(VECTORSTORE_PATH) / safe_name)
    
    async def _save_vectorstore(self, path: str, chunks: List[Chunk]):
        """Save vectorstore to disk."""
        Path(path).mkdir(parents=True, exist_ok=True)
        
        # Save chunks
        chunks_data = []
        for chunk in chunks:
            chunks_data.append({
                "id": chunk.id,
                "content": chunk.content,
                "context_prefix": chunk.context_prefix,
                "metadata": chunk.metadata,
                "embedding": chunk.embedding,
                "parent_id": chunk.parent_id,
                "children_ids": chunk.children_ids
            })
        
        with open(Path(path) / "chunks.json", 'w') as f:
            json.dump(chunks_data, f)
        
        # Save FAISS index
        if self.vectorstore:
            import faiss
            faiss.write_index(
                self.vectorstore["index"],
                str(Path(path) / "index.faiss")
            )
        
        # Save knowledge graph
        if ENABLE_GRAPHRAG:
            await self.graph_builder.save_graph(path)
        
        self.vectorstore_path = path
    
    async def load_vectorstore(self, path: str):
        """Load vectorstore from disk."""
        # Load chunks
        with open(Path(path) / "chunks.json") as f:
            chunks_data = json.load(f)
        
        chunks = []
        for data in chunks_data:
            chunk = Chunk(
                id=data["id"],
                content=data["content"],
                context_prefix=data.get("context_prefix", ""),
                metadata=data["metadata"],
                embedding=data.get("embedding"),
                parent_id=data.get("parent_id"),
                children_ids=data.get("children_ids", [])
            )
            chunks.append(chunk)
        
        # Build BM25 index
        self.searcher.index_chunks(chunks)
        
        # Load FAISS index
        index_path = Path(path) / "index.faiss"
        if index_path.exists():
            import faiss
            index = faiss.read_index(str(index_path))
            self.vectorstore = {
                "index": index,
                "chunks": {c.id: c for c in chunks}
            }
        
        # Load knowledge graph
        if ENABLE_GRAPHRAG:
            await self.graph_builder.load_graph(path)
        
        self.vectorstore_path = path
    
    async def query(
        self,
        question: str,
        top_k: int = RAG_TOP_K,
        use_crag: bool = ENABLE_CRAG
    ) -> RAGResponse:
        """Query the RAG system."""
        if not self.vectorstore:
            return RAGResponse(
                answer="No documents loaded. Please upload a document first.",
                sources=[],
                retrieval_strategy="none",
                chunks_used=0,
                confidence=0.0
            )
        
        # Semantic search
        from backend.services import ollama_service
        
        query_embedding = await ollama_service.generate_embedding(question)
        
        import numpy as np
        query_np = np.array([query_embedding]).astype('float32')
        
        distances, indices = self.vectorstore["index"].search(query_np, top_k * 2)
        
        chunk_ids = list(self.vectorstore["chunks"].keys())
        semantic_results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(chunk_ids):
                # Convert distance to similarity score
                score = 1 / (1 + distances[0][i])
                semantic_results.append((chunk_ids[idx], score))
        
        # Hybrid search
        strategy = "semantic"
        if ENABLE_HYBRID_SEARCH:
            bm25_results = self.searcher.bm25_search(question, top_k * 2)
            combined = self.searcher.hybrid_search(semantic_results, bm25_results)
            chunk_ids_ranked = [cid for cid, _ in combined[:top_k]]
            strategy = "hybrid"
        else:
            chunk_ids_ranked = [cid for cid, _ in semantic_results[:top_k]]
        
        # Get chunks
        chunks = [
            self.vectorstore["chunks"][cid]
            for cid in chunk_ids_ranked
            if cid in self.vectorstore["chunks"]
        ]
        
        # CRAG evaluation
        if use_crag and chunks:
            graded = await self.crag.grade_relevance(question, chunks)
            
            # Filter out irrelevant chunks
            relevant_chunks = [c for c, g, s in graded if s >= 0.5]
            
            # If too few relevant, reformulate and retry
            if len(relevant_chunks) < 2:
                new_query = await self.crag.reformulate_query(question)
                if new_query != question:
                    return await self.query(new_query, top_k, use_crag=False)
            
            chunks = relevant_chunks if relevant_chunks else chunks
            strategy += "+crag"
        
        # Build context
        context_parts = []
        sources = []
        
        for chunk in chunks:
            context_parts.append(f"{chunk.context_prefix}\n{chunk.content}")
            sources.append({
                "chunk_id": chunk.id,
                "source": chunk.metadata.get("source", "unknown"),
                "type": chunk.metadata.get("type", "unknown")
            })
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Generate answer
        prompt = f"""Answer the question based on the following context. 
If the context doesn't contain the answer, say so clearly.
Cite specific parts of the context when possible.

Context:
{context}

Question: {question}

Answer:"""
        
        answer = await ollama_service.generate_text(prompt, temperature=0.3)
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            retrieval_strategy=strategy,
            chunks_used=len(chunks),
            confidence=0.8 if chunks else 0.0
        )
    
    async def query_stream(
        self,
        question: str,
        top_k: int = RAG_TOP_K
    ) -> AsyncGenerator[str, None]:
        """Query with streaming response."""
        # Get context (reuse query logic)
        response = await self.query(question, top_k, use_crag=False)
        
        if not response.sources:
            yield "No documents loaded. Please upload a document first."
            return
        
        # Get chunks for context
        chunks = [
            self.vectorstore["chunks"][s["chunk_id"]]
            for s in response.sources
            if s["chunk_id"] in self.vectorstore["chunks"]
        ]
        
        context = "\n\n---\n\n".join([
            f"{c.context_prefix}\n{c.content}" for c in chunks
        ])
        
        prompt = f"""Answer the question based on the following context.
Cite specific parts when possible.

Context:
{context}

Question: {question}

Answer:"""
        
        from backend.services import ollama_service
        
        async for token in ollama_service.generate_text_stream(prompt, temperature=0.3):
            yield token
    
    def list_vectorstores(self) -> List[str]:
        """List available vectorstores."""
        path = Path(VECTORSTORE_PATH)
        if not path.exists():
            return []
        
        return [d.name for d in path.iterdir() if d.is_dir()]
    
    def delete_vectorstore(self, name: str) -> bool:
        """Delete a vectorstore."""
        import shutil
        
        path = Path(VECTORSTORE_PATH) / name
        if path.exists():
            shutil.rmtree(path)
            return True
        return False


# ============================================
# SINGLETON INSTANCE
# ============================================

rag_manager = RAGManager()
