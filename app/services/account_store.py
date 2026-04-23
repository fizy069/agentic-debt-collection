"""Account store — maps borrower IDs to system-side account records.

Production implementations would query a CRM or database.  The bundled
``InMemoryAccountStore`` provides deterministic data for development,
testing, and evaluation without external dependencies.
"""

from __future__ import annotations

import logging
from typing import Protocol

from app.models.pipeline import AccountRecord

logger = logging.getLogger(__name__)


class AccountStore(Protocol):
    """Read-only lookup interface for account records."""

    def get(self, borrower_id: str) -> AccountRecord | None: ...

    def list_ids(self) -> list[str]: ...


_SEED_ACCOUNTS: list[AccountRecord] = [
    AccountRecord(
        borrower_id="B-TEST-001",
        account_reference="ACCT-9876",
        date_of_birth="1985-03-14",
        debt_amount=4500.00,
        currency="USD",
        days_past_due=45,
        notes="Credit card balance, minimum payments missed.",
    ),
    AccountRecord(
        borrower_id="B-TEST-002",
        account_reference="ACCT-5432",
        date_of_birth="1972-11-22",
        debt_amount=12000.00,
        currency="USD",
        days_past_due=90,
        notes="Personal loan, 3 months delinquent.",
    ),
    AccountRecord(
        borrower_id="B-TEST-003",
        account_reference="ACCT-1122",
        date_of_birth="1990-07-08",
        debt_amount=2200.50,
        currency="USD",
        days_past_due=30,
        notes="Medical bill, recently defaulted.",
    ),
    AccountRecord(
        borrower_id="B-TEST-004",
        account_reference="ACCT-7788",
        date_of_birth="1968-01-30",
        debt_amount=8750.00,
        currency="USD",
        days_past_due=120,
        notes="Auto loan deficiency balance after repossession.",
    ),
    AccountRecord(
        borrower_id="B-TEST-005",
        account_reference="ACCT-3344",
        date_of_birth="1995-09-02",
        debt_amount=950.00,
        currency="USD",
        days_past_due=60,
        notes="Utility bill sent to collections.",
    ),
]


class InMemoryAccountStore:
    """Dev/test store seeded with sample accounts."""

    def __init__(self, accounts: list[AccountRecord] | None = None) -> None:
        seed = accounts if accounts is not None else _SEED_ACCOUNTS
        self._accounts: dict[str, AccountRecord] = {a.borrower_id: a for a in seed}

    def get(self, borrower_id: str) -> AccountRecord | None:
        return self._accounts.get(borrower_id)

    def put(self, account: AccountRecord) -> None:
        """Insert or replace an account (useful for eval scenario seeding)."""
        self._accounts[account.borrower_id] = account

    def list_ids(self) -> list[str]:
        return list(self._accounts.keys())


_default_store: InMemoryAccountStore | None = None


def get_account_store() -> InMemoryAccountStore:
    """Return the singleton in-memory account store."""
    global _default_store
    if _default_store is None:
        _default_store = InMemoryAccountStore()
    return _default_store
