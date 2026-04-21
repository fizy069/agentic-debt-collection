"""Tests for token budget enforcement and section accounting."""

from __future__ import annotations

import pytest

from app.services.token_budget import (
    MAX_CONTEXT_TOKENS,
    MAX_HANDOFF_TOKENS,
    ContextBudgetReport,
    count_tokens,
    enforce_context_budget,
    truncate_to_budget,
)


class TestCountTokens:
    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_simple_text(self):
        tokens = count_tokens("Hello world")
        assert tokens > 0
        assert tokens < 10

    def test_longer_text_more_tokens(self):
        short = count_tokens("Hello")
        long = count_tokens("Hello world, this is a much longer sentence with many words")
        assert long > short


class TestTruncateToBudget:
    def test_short_text_unchanged(self):
        text = "Short text"
        result = truncate_to_budget(text, 100)
        assert result == text

    def test_long_text_truncated(self):
        text = "word " * 500
        result = truncate_to_budget(text, 50)
        assert count_tokens(result) <= 50

    def test_exact_budget(self):
        text = "Hello world"
        tokens = count_tokens(text)
        result = truncate_to_budget(text, tokens)
        assert result == text


class TestEnforceContextBudget:
    def test_within_budget_unchanged(self):
        system = "System prompt"
        user = "User prompt"
        s, u = enforce_context_budget(system, user, max_total=2000)
        assert s == system
        assert u == user

    def test_over_budget_truncates_user(self):
        system = "System prompt"
        user = "word " * 500
        s, u = enforce_context_budget(system, user, max_total=100)
        assert s == system
        total = count_tokens(s) + count_tokens(u)
        assert total <= 100

    def test_system_never_truncated(self):
        system = "A " * 100
        user = "B " * 100
        s, u = enforce_context_budget(system, user, max_total=50)
        assert s == system

    def test_exact_budget_no_truncation(self):
        system = "Hello"
        user = "World"
        total = count_tokens(system) + count_tokens(user)
        s, u = enforce_context_budget(system, user, max_total=total)
        assert s == system
        assert u == user


class TestContextBudgetReport:
    def test_add_sections(self):
        report = ContextBudgetReport()
        report.add("system", "System prompt text")
        report.add("user", "User prompt text")
        assert len(report.sections) == 2
        assert report.total_tokens > 0

    def test_to_metadata(self):
        report = ContextBudgetReport(limit=2000)
        report.add("system", "System prompt")
        report.overflow_detected = True
        report.overflow_summary_used = True
        report.handoff_tokens = 150

        meta = report.to_metadata()
        assert meta["budget_limit"] == 2000
        assert meta["overflow_detected"] is True
        assert meta["overflow_summary_used"] is True
        assert meta["handoff_tokens"] == 150
        assert "system" in meta["sections"]

    def test_empty_report(self):
        report = ContextBudgetReport()
        meta = report.to_metadata()
        assert meta["total_tokens"] == 0
        assert meta["overflow_detected"] is False

    def test_section_tokens_sum(self):
        report = ContextBudgetReport()
        t1 = report.add("a", "Hello world")
        t2 = report.add("b", "Goodbye world")
        assert report.total_tokens == t1 + t2


class TestBudgetConstants:
    def test_context_limit(self):
        assert MAX_CONTEXT_TOKENS == 2000

    def test_handoff_limit(self):
        assert MAX_HANDOFF_TOKENS == 500
