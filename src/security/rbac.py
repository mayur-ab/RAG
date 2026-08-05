from typing import List


class RBACEngine:
    """Role-Based Access Control Security Engine."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def check_access(self, user_roles: List[str], required_roles: List[str]) -> bool:
        if not self.enabled:
            return True
        if "admin" in user_roles:
            return True
        return any(role in required_roles for role in user_roles)
