---
title: Warehouse Logistics OpenEnv
emoji: 🚚
colorFrom: blue
colorTo: green
sdk: docker
sdk_version: "0.0.0"
python_version: "3.11"
app_file: app.py
pinned: false
app_port: 7860
---

# Warehouse Logistics OpenEnv

Warehouse Logistics OpenEnv is a real-world style agent environment for order fulfillment. It models tasks humans actually perform in operations teams: stock checks, address validation, allocation, shipping, and rerouting across warehouses.

The environment is designed to be useful for training and evaluating agents in a practical workflow, not a toy game. It follows the OpenEnv interface with typed Pydantic models, `reset()`, `step()`, and `state()`, plus an `openenv.yaml` manifest for validation.

## Why This Environment

Agent benchmarks are strongest when the task is concrete, deterministic, and easy to score. Warehouse operations fit that pattern well: the state is structured, the actions are bounded, and progress can be measured at each step. This environment uses that setting to provide incremental reward shaping and deterministic graders.

## Core API

### Observation Space

Each observation includes:

```python
{
    "orders": [
        {
            "order_id": "ORD-001",
            "sku": "SKU-001",
            "quantity": 10,
            "address": "123 Main St, New York, NY",
            "status": "pending|stock_checked|address_validated|allocated|shipped|failed",
            "allocated_warehouse": "NYC-01|null"
        }
    ],
    "warehouses": [
        {
            "warehouse_id": "NYC-01",
            "city": "New York",
            "stock_level": {"SKU-001": 50},
            "capacity": 1000
        }
    ],
    "step_count": 5,
    "processed_orders": 2,
    "failed_orders": 0,
    "last_action_result": {...}
}
```

### Action Space

Agents may choose one of five actions:

```python
{
    "action_type": "check_stock|validate_address|allocate|ship|reroute",
    "order_id": "ORD-001",
    "warehouse": "NYC-01|null",
    "message": "Optional reasoning"
}
```

### Reward Structure

The environment provides shaped rewards for intermediate progress and a final deterministic score from the grader. Final scores are strictly bounded inside `(0, 1)` so they remain valid for the submission validator.

## Tasks

### Easy

Single-order stock check. The agent should identify the available warehouse, allocate correctly, and finish the order with minimal steps.

### Medium

Batch processing with address validation. The agent must validate addresses, reject bad ones, and allocate valid orders efficiently.

### Hard

Multi-warehouse rerouting. The agent must handle stock shortages, reroute to alternate warehouses, and avoid failed allocations.

## Baseline Scores

The baseline runner in `inference.py` evaluates all three tasks and prints structured logs required by the hackathon submission format.

Expected baseline range:

- Easy: around 0.7
- Medium: around 0.6
- Hard: around 0.5

## Setup

### Environment Variables

Set these values before running the baseline:

- `API_BASE_URL` with a default value
- `MODEL_NAME` with a default value
- `HF_TOKEN` required

### Local Run

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python inference.py
```

### Docker Run

```bash
docker build -t warehouse-logistics-env .
docker run -p 7860:7860 -e API_BASE_URL=https://api.openai.com/v1 -e MODEL_NAME=gpt-4.1-mini -e HF_TOKEN=your_token warehouse-logistics-env
```

## OpenEnv Compliance

This repository includes:

- Typed Pydantic models for observations, actions, and rewards
- `step(action) -> (observation, reward, done, info)`
- `reset() -> initial observation`
- `state() -> current observation`
- `openenv.yaml` metadata
- A working Dockerfile for Hugging Face Spaces deployment

## Tag

This project is intended to be published with the `openenv` tag.
