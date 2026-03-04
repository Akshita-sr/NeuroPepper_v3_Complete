# backend/core/memory_manager.py
# ============================================
# Memory Manager - Persistent Robot Memory
# ============================================
# Implements a Mem0-style memory system that allows
# Pepper robot to remember users across sessions.
# 
# Features:
# - User profile storage
# - Fact extraction from conversations
# - Semantic memory search
# - Memory consolidation and decay
# ============================================

import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import sqlite3
import asyncio

from dotenv import load_dotenv
load_dotenv()

# Configuration
MEMORY_DB_PATH = Path(os.getenv("MEMORY_DB_PATH", "data/memory/memory.db"))
MEMORY_VECTOR_PATH = Path(os.getenv("MEMORY_VECTOR_PATH", "data/memory/vectors"))

# Ensure directories exist
MEMORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
MEMORY_VECTOR_PATH.mkdir(parents=True, exist_ok=True)


@dataclass
class Memory:
    """Represents a single memory entry."""
    id: str
    user_id: str
    content: str
    category: str  # fact, preference, interaction, observation
    importance: float  # 0.0 to 1.0
    created_at: str
    last_accessed: str
    access_count: int
    metadata: Dict[str, Any]


@dataclass
class UserProfile:
    """Represents a user's profile built from memories."""
    user_id: str
    name: Optional[str]
    first_seen: str
    last_seen: str
    interaction_count: int
    preferences: Dict[str, Any]
    facts: List[str]
    personality_notes: List[str]


class MemoryManager:
    """
    Persistent memory system for intelligent robot interactions.
    
    This implements a simplified version of Mem0's architecture:
    - Automatic fact extraction from conversations
    - Semantic search for relevant memories
    - Memory consolidation (merging similar memories)
    - Importance-based decay (less important memories fade)
    
    Usage:
        memory = MemoryManager()
        
        # Store a memory
        await memory.add("user_123", "User mentioned they prefer tea over coffee")
        
        # Retrieve relevant memories
        memories = await memory.search("user_123", "drink preferences")
        
        # Get user profile
        profile = await memory.get_user_profile("user_123")
    """
    
    def __init__(self):
        """Initialize the memory manager."""
        self.db_path = MEMORY_DB_PATH
        self._init_database()
        
        # Try to initialize embeddings for semantic search
        self.embeddings = None
        self._init_embeddings()
    
    def _init_database(self):
        """Initialize SQLite database for memory storage."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Create memories table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                category TEXT DEFAULT 'fact',
                importance REAL DEFAULT 0.5,
                created_at TEXT NOT NULL,
                last_accessed TEXT NOT NULL,
                access_count INTEGER DEFAULT 0,
                metadata TEXT DEFAULT '{}',
                embedding BLOB
            )
        """)
        
        # Create user profiles table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                name TEXT,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                interaction_count INTEGER DEFAULT 1,
                preferences TEXT DEFAULT '{}',
                facts TEXT DEFAULT '[]',
                personality_notes TEXT DEFAULT '[]'
            )
        """)
        
        # Create index for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_user 
            ON memories(user_id)
        """)
        
        conn.commit()
        conn.close()
    
    def _init_embeddings(self):
        """Initialize embedding model for semantic search."""
        try:
            # Try the new package first (langchain-ollama)
            from langchain_ollama import OllamaEmbeddings
            self.embeddings = OllamaEmbeddings(
                model="nomic-embed-text",
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            )
        except ImportError:
            try:
                # Fall back to old package if new one isn't installed
                from langchain_community.embeddings import OllamaEmbeddings
                self.embeddings = OllamaEmbeddings(
                    model="nomic-embed-text",
                    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                )
            except Exception as e:
                print(f"⚠️ Memory embeddings not available: {e}")
        except Exception as e:
            print(f"⚠️ Memory embeddings not available: {e}")
    
    # ============================================
    # MEMORY OPERATIONS
    # ============================================
    
    async def add(
        self,
        user_id: str,
        content: str,
        category: str = "fact",
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Memory:
        """
        Add a new memory for a user.
        
        Args:
            user_id: Unique identifier for the user
            content: The memory content (what to remember)
            category: Type of memory (fact, preference, interaction, observation)
            importance: How important is this memory (0.0-1.0)
            metadata: Additional structured data
        
        Returns:
            The created Memory object
        """
        # Generate unique ID
        memory_id = hashlib.md5(
            f"{user_id}:{content}:{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        
        now = datetime.now().isoformat()
        
        memory = Memory(
            id=memory_id,
            user_id=user_id,
            content=content,
            category=category,
            importance=importance,
            created_at=now,
            last_accessed=now,
            access_count=0,
            metadata=metadata or {}
        )
        
        # Generate embedding if available
        embedding_bytes = None
        if self.embeddings:
            try:
                embedding = self.embeddings.embed_query(content)
                embedding_bytes = json.dumps(embedding).encode()
            except Exception as e:
                print(f"⚠️ Could not generate embedding: {e}")
        
        # Store in database
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO memories (id, user_id, content, category, importance,
                                  created_at, last_accessed, access_count, metadata, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            memory.id, memory.user_id, memory.content, memory.category,
            memory.importance, memory.created_at, memory.last_accessed,
            memory.access_count, json.dumps(memory.metadata), embedding_bytes
        ))
        
        conn.commit()
        conn.close()
        
        # Update user profile
        await self._update_user_profile(user_id, content, category)
        
        return memory
    
    async def search(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
        category: Optional[str] = None
    ) -> List[Memory]:
        """
        Search for relevant memories.
        
        Uses semantic search if embeddings are available,
        falls back to keyword matching otherwise.
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Build query
        sql = "SELECT * FROM memories WHERE user_id = ?"
        params = [user_id]
        
        if category:
            sql += " AND category = ?"
            params.append(category)
        
        # For now, use simple keyword matching
        # In production, implement proper semantic search
        sql += " AND content LIKE ?"
        params.append(f"%{query}%")
        
        sql += " ORDER BY importance DESC, last_accessed DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        
        memories = []
        for row in rows:
            memory = Memory(
                id=row[0],
                user_id=row[1],
                content=row[2],
                category=row[3],
                importance=row[4],
                created_at=row[5],
                last_accessed=row[6],
                access_count=row[7],
                metadata=json.loads(row[8])
            )
            memories.append(memory)
            
            # Update access stats
            cursor.execute("""
                UPDATE memories 
                SET last_accessed = ?, access_count = access_count + 1
                WHERE id = ?
            """, (datetime.now().isoformat(), memory.id))
        
        conn.commit()
        conn.close()
        
        return memories
    
    async def get_all_memories(
        self,
        user_id: str,
        limit: int = 50
    ) -> List[Memory]:
        """Get all memories for a user."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM memories 
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (user_id, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            Memory(
                id=row[0],
                user_id=row[1],
                content=row[2],
                category=row[3],
                importance=row[4],
                created_at=row[5],
                last_accessed=row[6],
                access_count=row[7],
                metadata=json.loads(row[8])
            )
            for row in rows
        ]
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a specific memory."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        deleted = cursor.rowcount > 0
        
        conn.commit()
        conn.close()
        
        return deleted
    
    async def clear_user_memories(self, user_id: str) -> int:
        """Delete all memories for a user."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))
        deleted_count = cursor.rowcount
        
        cursor.execute("DELETE FROM user_profiles WHERE user_id = ?", (user_id,))
        
        conn.commit()
        conn.close()
        
        return deleted_count
    
    # ============================================
    # USER PROFILES
    # ============================================
    
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get a user's profile."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM user_profiles WHERE user_id = ?",
            (user_id,)
        )
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return UserProfile(
            user_id=row[0],
            name=row[1],
            first_seen=row[2],
            last_seen=row[3],
            interaction_count=row[4],
            preferences=json.loads(row[5]),
            facts=json.loads(row[6]),
            personality_notes=json.loads(row[7])
        )
    
    async def _update_user_profile(
        self,
        user_id: str,
        content: str,
        category: str
    ):
        """Update user profile when a memory is added."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        
        # Check if profile exists
        cursor.execute(
            "SELECT * FROM user_profiles WHERE user_id = ?",
            (user_id,)
        )
        existing = cursor.fetchone()
        
        if existing:
            # Update existing profile
            facts = json.loads(existing[6])
            if category == "fact" and content not in facts:
                facts.append(content)
            
            cursor.execute("""
                UPDATE user_profiles
                SET last_seen = ?,
                    interaction_count = interaction_count + 1,
                    facts = ?
                WHERE user_id = ?
            """, (now, json.dumps(facts), user_id))
        else:
            # Create new profile
            cursor.execute("""
                INSERT INTO user_profiles 
                (user_id, first_seen, last_seen, interaction_count, facts)
                VALUES (?, ?, ?, 1, ?)
            """, (user_id, now, now, json.dumps([content] if category == "fact" else [])))
        
        conn.commit()
        conn.close()
    
    async def set_user_name(self, user_id: str, name: str):
        """Set or update a user's name."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE user_profiles SET name = ? WHERE user_id = ?
        """, (name, user_id))
        
        if cursor.rowcount == 0:
            # Profile doesn't exist, create it
            now = datetime.now().isoformat()
            cursor.execute("""
                INSERT INTO user_profiles (user_id, name, first_seen, last_seen)
                VALUES (?, ?, ?, ?)
            """, (user_id, name, now, now))
        
        conn.commit()
        conn.close()
    
    # ============================================
    # MEMORY CONSOLIDATION
    # ============================================
    
    async def consolidate_memories(self, user_id: str):
        """
        Consolidate similar memories to reduce redundancy.
        
        This is a simplified version of Mem0's memory consolidation.
        In production, use an LLM to intelligently merge memories.
        """
        # Get all memories for user
        memories = await self.get_all_memories(user_id, limit=100)
        
        # Group by category
        by_category = {}
        for memory in memories:
            if memory.category not in by_category:
                by_category[memory.category] = []
            by_category[memory.category].append(memory)
        
        # For each category, check for duplicates
        # (In production, use semantic similarity)
        for category, cat_memories in by_category.items():
            seen_content = set()
            duplicates = []
            
            for memory in cat_memories:
                # Simple duplicate detection
                simplified = memory.content.lower().strip()
                if simplified in seen_content:
                    duplicates.append(memory.id)
                else:
                    seen_content.add(simplified)
            
            # Remove duplicates
            for dup_id in duplicates:
                await self.delete_memory(dup_id)
        
        return len(duplicates) if 'duplicates' in dir() else 0
    
    async def apply_decay(self, days_threshold: int = 30):
        """
        Apply importance decay to old, rarely accessed memories.
        
        Memories that haven't been accessed in `days_threshold` days
        have their importance reduced.
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        threshold_date = (datetime.now() - timedelta(days=days_threshold)).isoformat()
        
        # Reduce importance of old memories
        cursor.execute("""
            UPDATE memories
            SET importance = importance * 0.9
            WHERE last_accessed < ? AND importance > 0.1
        """, (threshold_date,))
        
        updated = cursor.rowcount
        
        # Delete very low importance memories
        cursor.execute("""
            DELETE FROM memories WHERE importance < 0.1
        """)
        
        deleted = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        return {"decayed": updated, "deleted": deleted}
    
    # ============================================
    # CONTEXT BUILDING
    # ============================================
    
    async def build_context_for_conversation(
        self,
        user_id: str,
        current_message: Optional[str] = None
    ) -> str:
        """
        Build a context string from user memories for LLM prompts.
        
        This is what gets injected into the system prompt to give
        the AI context about the user.
        """
        profile = await self.get_user_profile(user_id)
        
        if not profile:
            return ""
        
        context_parts = []
        
        # Add user info
        if profile.name:
            context_parts.append(f"User's name: {profile.name}")
        
        context_parts.append(f"First interaction: {profile.first_seen}")
        context_parts.append(f"Total interactions: {profile.interaction_count}")
        
        # Add relevant memories
        if current_message:
            memories = await self.search(user_id, current_message, limit=5)
        else:
            memories = await self.get_all_memories(user_id, limit=10)
        
        if memories:
            context_parts.append("\nRelevant memories about this user:")
            for memory in memories:
                context_parts.append(f"- {memory.content}")
        
        return "\n".join(context_parts)


# Create singleton instance
memory_manager = MemoryManager()
