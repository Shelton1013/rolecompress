"""RoleCompress: compressing long-video tokens by cross-modal information role.

Public API is intentionally small; see rolecompress.roles for the core definitions.
"""
from .roles import Role, RoleBudget, assign_role_from_margins, allocate_frames

__all__ = ["Role", "RoleBudget", "assign_role_from_margins", "allocate_frames"]
__version__ = "0.1.0"
