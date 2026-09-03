"""What an authenticated user is allowed to do.

Deliberately one function and a handful of constants rather than a policy
engine — the whole rule set fits on a screen, which is the point. Roles come
from LDAP group membership; stewardship is scoped *per pool* because the
dedicated LLM nodes and the general partition have different caretakers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .identity import Principal
from .settings import settings

# ── Actions ──────────────────────────────────────────────────────────────────

VIEW = "view"
CREATE = "create"
EDIT = "edit"
CANCEL = "cancel"
EXTEND = "extend"
LOCK = "lock"
UNLOCK = "unlock"

#: Actions that change a deployment. A locked deployment refuses all of them
#: unless the caller is privileged — that is what makes `locked` mean
#: "this is production" rather than merely "read-only for the owner".
MUTATING = frozenset({EDIT, CANCEL, EXTEND})

#: Actions only an admin or the owning pool's operators may take.
PRIVILEGED_ONLY = frozenset({LOCK, UNLOCK})


def _split(raw: str | None) -> list[str]:
    return [p.strip() for p in (raw or "").split(",") if p.strip()]


@runtime_checkable
class OwnedResource(Protocol):
    """Structural type for anything ownership applies to.

    `Lease` satisfies this today; `Deployment` will satisfy it unchanged when
    the two are split apart.
    """

    owner_sub: str | None
    owner_group: str | None
    locked: bool
    pool: str | None


@dataclass(frozen=True)
class User:
    """A principal with roles resolved."""

    sub: str
    display_name: str
    email: str = ""
    groups: frozenset[str] = field(default_factory=frozenset)
    is_admin: bool = False
    operator_pools: frozenset[str] = field(default_factory=frozenset)
    is_user: bool = True
    via: str = "unknown"

    @property
    def is_local_admin(self) -> bool:
        return self.via == "local"


def operator_pools_for(groups: frozenset[str]) -> frozenset[str]:
    """Pools this user stewards, from `pools[].operators` in cluster config.

    Phase 1 reads this from `cluster.yaml`. Until that lands, the mapping comes
    from POOL_OPERATORS as `pool:group[,pool:group...]`, so the rule and its
    tests are already in place when the config arrives.
    """
    pools: set[str] = set()
    for item in _split(settings.pool_operators):
        pool, _, group = item.partition(":")
        if pool and group and group.strip() in groups:
            pools.add(pool.strip())
    return frozenset(pools)


def build_user(principal: Principal) -> User:
    """Attach roles to an authenticated principal."""
    groups = principal.groups
    admin_groups = set(_split(settings.admin_groups))
    user_groups = set(_split(settings.user_groups))

    # The break-glass account is an admin by construction — it exists precisely
    # for the case where group lookup is impossible.
    is_admin = principal.is_local_admin or bool(admin_groups & groups)
    # An empty USER_GROUPS means "any authenticated identity is a user".
    is_user = is_admin or not user_groups or bool(user_groups & groups)

    return User(
        sub=principal.sub,
        display_name=principal.display_name or principal.sub,
        email=principal.email,
        groups=groups,
        is_admin=is_admin,
        operator_pools=operator_pools_for(groups),
        is_user=is_user,
        via=principal.via,
    )


ANONYMOUS = User(sub="", display_name="", is_user=False, via="anonymous")


def is_privileged(user: User, resource: OwnedResource | None) -> bool:
    """Admin everywhere, or operator of the pool the resource lives in."""
    if user.is_admin:
        return True
    pool = getattr(resource, "pool", None) if resource is not None else None
    return bool(pool) and pool in user.operator_pools


def owns(user: User, resource: OwnedResource) -> bool:
    """Individual ownership, or membership of the owning group.

    Group ownership is what stops "only the owner may edit" from failing the
    moment someone is on holiday or a team shares a service model.
    """
    if not user.sub:
        return False
    if resource.owner_sub and user.sub == resource.owner_sub:
        return True
    group = resource.owner_group
    return bool(group) and group in user.groups


def can(user: User, action: str, resource: OwnedResource) -> bool:
    """The entire authorization rule."""
    if not user.sub:
        return False

    privileged = is_privileged(user, resource)

    if action in PRIVILEGED_ONLY:
        return privileged
    if privileged:
        return True
    if action == VIEW:
        return True
    if getattr(resource, "locked", False) and action in MUTATING:
        return False
    return owns(user, resource)


def can_create(user: User, pool: str | None = None) -> bool:
    """Creation has no resource to own, so it is a role check."""
    if not user.sub:
        return False
    if user.is_admin:
        return True
    if pool and pool in user.operator_pools:
        return True
    return user.is_user


def describe_denial(user: User, action: str, resource: OwnedResource) -> str:
    """A message that says what to do about it, not just 'forbidden'."""
    if not user.sub:
        return "Not authenticated."
    if action in PRIVILEGED_ONLY:
        return f"Only admins or operators of this pool can {action} a deployment."
    if getattr(resource, "locked", False) and action in MUTATING:
        locked_by = getattr(resource, "locked_by", None)
        reason = getattr(resource, "locked_reason", None)
        detail = f" by {locked_by}" if locked_by else ""
        detail += f": {reason}" if reason else ""
        return (
            f"This deployment is locked{detail}. "
            "Ask an admin or a pool operator to unlock it."
        )
    return (
        f"You can only {action} deployments you own. "
        "Ask the owner, an admin, or a pool operator."
    )
