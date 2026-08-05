from typing import List, Tuple
from src.metadata.schema import Document


class SecurityFilter:
    """Enforces document security and Role-Based Access Control (RBAC) filtering."""

    @staticmethod
    def filter_documents(
        results: List[Tuple[Document, float]],
        user_roles: List[str]
    ) -> List[Tuple[Document, float]]:
        filtered = []
        for doc, score in results:
            allowed = getattr(doc.metadata, "allowed_roles", ["user", "admin"])
            # Match if user possesses at least one authorized role
            if any(role in allowed for role in user_roles):
                filtered.append((doc, score))
        return filtered
