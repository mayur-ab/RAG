from src.service import RAGPipelineService

# Global Singleton Service instance
_rag_service_instance: RAGPipelineService = None


def get_rag_service() -> RAGPipelineService:
    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = RAGPipelineService()
    return _rag_service_instance
