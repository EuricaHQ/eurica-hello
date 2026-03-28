# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Backend (Python/FastAPI)
```bash
cd backend

# Run the server
source .venv/bin/activate
uvicorn main:app --reload

# Run all tests
python -m pytest test_*.py

# Run a single test file
python test_flows.py
python test_guard_v2_signals.py

# Simulate multi-participant conversation without an API key (uses MockLLM)
python simulate_conversation.py
```

Tests requiring real LLM calls (`test_e2e_smoke.py`, `test_openai_interpret.py`, `test_critical_llm_advisory.py`) need `OPENAI_API_KEY` set in `backend/.env`.

### Frontend (Next.js)
```bash
cd frontend-web
npm run dev       # Dev server at http://localhost:3000
npm run build
npm run lint
```

> **Warning:** This project uses Next.js 16 which has breaking changes. Before editing frontend code, read the relevant guide in `frontend-web/node_modules/next/dist/docs/`.

## Architecture

Eurica is a multi-party **Decision Coordinator** — a system where a group of participants collaboratively reach a decision. The spec in `spec/` (currently v2.10.6) is the authoritative design document.

### Core Principle: LLM is advisory, state machine is authoritative

The system enforces strict separation:
1. **LLM interprets** user messages → structured `Signals` (preferences, constraints, objections, confirmations, etc.)
2. **`signal_mapper`** deterministically maps `Signals` → `Event` (no LLM involvement)
3. **State machine** runs the `Event` through pure guard functions → new `State` + `Actions`

The LLM never makes decisions. It only classifies and interprets.

### State Machine (`backend/machine/`)
- `states.py` — EFSM states: `CLARIFYING → COLLECTING → AGGREGATING → RESOLVING → AVOIDING → VALIDATING → DECIDING → DECIDED / INFEASIBLE`
- `events.py` — Events: `RESPONSE_RECEIVED`, `AGGREGATION_COMPLETED`, `VALIDATION_COMPLETED`, `DECISION_CONFIRMED`, `DECISION_REJECTED`, `AVOIDANCE_DETECTED`
- `context.py` — `DecisionContext` dataclass: accumulates all signals across the session (preferences, constraints, conflicts, objections, semantic signals, etc.)
- `transition.py` — Core FSM: guard functions (pure predicates on context) + transition table mapping `(state, event) → (new_state, actions, updated_context)`
- `actions.py` — `ActionType` enum describing what the system should do next (ask, aggregate, validate, resolve, propose, finalize, etc.)

After a transition, the system may emit additional system events (e.g., auto-confirm if guards pass). Max 3 system event iterations to prevent loops.

### LLM Layer (`backend/llm/`)
- `interface.py` — Abstract `LLM` class with `interpret()`, `generate()`, `evaluate_critical_participants()`
- `openai_llm.py` — GPT-4o-mini implementation with strict signal schema validation and safe defaults
- `mock_llm.py` — Used in tests that don't need a real API key

### Interpreter / Signal Layer (`backend/interpreter/`)
- `signals.py` — `Signals` dataclass: the structured output of LLM interpretation
- `signal_mapper.py` — Deterministic `map_signals_to_event()` with priority order: confirmation → rejection → avoidance → default `RESPONSE_RECEIVED`

### API Layer (`backend/api/routes.py`)
- POST `/chat` — receives `MessageRequest` (decision_id, participant, message), runs full pipeline, returns `MessageResponse` (state, reply, actions_executed)
- Routes mirror the FSM's `_confirmation_required()` guard to decide whether to auto-emit `DECISION_CONFIRMED` or wait for explicit user confirmation

### Multi-Dimensional Conflict Detection
Preferences carry an optional `dimension` field (e.g., `day`, `time`, `cuisine`). Conflict detection is dimension-aware: preferences on different dimensions are **complementary**, not conflicting. This prevents false positives (spec v2.10.6).

### Frontend (`frontend-web/`)
Simple Next.js chat UI. `app/page.tsx` handles message input and POSTs to `http://127.0.0.1:8000/chat`. Backend URL is hardcoded.

### Mobile (`eurica-mobile/`)
Expo-based React Native project, minimal implementation.

## STRICT DEVELOPMENT RULES (CRITICAL)

Spec Authority:
- The specification (spec/*.md) is the single source of truth
- If code and spec differ → ALWAYS follow spec
- NEVER introduce behavior not grounded in the spec

Determinism:
- The system must remain fully deterministic
- NO hidden heuristics
- NO implicit assumptions

State Machine Safety:
- NEVER change:
  - states
  - transitions
  - guard logic
  without explicit spec reference

- Guard priority MUST be preserved exactly

LLM Constraints:
- LLM is interpretation only
- LLM must NOT:
  - decide transitions
  - override system logic
  - introduce new behavior

Implementation Style:
- Prefer minimal, surgical changes
- DO NOT refactor broadly unless explicitly requested
- Show diffs before large changes

Conflict & Dimension Rules:
- Conflicts must be dimension-aware
- Cross-dimension preferences are complementary
- Missing dimension must NOT create conflict

Testing Discipline:
- Always update tests when behavior changes
- Run full test suite before completion
- Add tests for edge cases (especially conflicts & dimensions)

Current Focus:
- Dimension completeness (NOT yet enforced)
- Prevent premature decisions