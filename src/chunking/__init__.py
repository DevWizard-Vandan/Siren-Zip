"""Chunking and Neural GOP extraction package."""

from src.chunking.chunk_orchestrator import ChunkOrchestrator, ChunkTrainingResult
from src.chunking.video_splitter import ChunkMetadata, VideoSplitter

__all__ = ["VideoSplitter", "ChunkMetadata", "ChunkOrchestrator", "ChunkTrainingResult"]
