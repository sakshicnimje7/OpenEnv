# Warehouse Logistics AI Environment

**E-Commerce Logistics AI Environment (Warehouse Manager Simulator)**

A production-ready OpenEnv-compliant environment for training AI agents in warehouse logistics and order fulfillment. Simulates real-world scenarios including inventory management, address validation, order routing, and multi-warehouse coordination.

## Features

- ✅ **OpenEnv Specification Compliant**: Full implementation of `reset()`, `step()`, `state()` interface
- ✅ **Pydantic Type Safety**: All models use Pydantic v2 with full type hints
- ✅ **3 Difficulty Levels**: Easy (single order), Medium (batch processing), Hard (multi-warehouse routing)
- ✅ **Shaped Rewards**: Intermediate rewards guide agent learning, not binary rewards
- ✅ **Deterministic Grading**: Each task has a deterministic grader returning scores 0.0-1.0
- ✅ **OpenAI Integration**: Baseline agent uses OpenAI API with async execution
- ✅ **Dockerized**: Production-ready Dockerfile with dependencies
- ✅ **Comprehensive Logging**: Exact logging format [START], [STEP], [END]

## Project Structure

```
logistics-env/
├── env/
│   ├── __init__.py
│   ├── environment.py       # Main environment (OpenEnv compliant)
│   ├── models.py            # Pydantic data models
│   ├── tasks.py             # Task definitions (easy/medium/hard)
│   ├── grader.py            # Deterministic task graders
│   └── utils.py             # Helper utilities
├── inference.py             # Baseline agent with OpenAI
├── test_environment.py      # No-API task demo runner
├── verify_setup.py          # Dependency and environment checks
├── openenv.yaml             # Environment configuration
├── requirements.txt         # Python dependencies
├── Dockerfile               # Container configuration
├── .env.example             # Environment variable template
├── .gitignore               # Git ignore rules
├── GETTING_STARTED.md       # Quick start guide
├── FILE_STRUCTURE.md        # Detailed file inventory
├── DELIVERY_SUMMARY.md      # Delivery and verification summary
├── INDEX.md                 # Navigation index
└── README.md                # This file
```

## Installation

### Local Setup

```bash
# Clone or download the project
cd logistics-env

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your OpenAI API key
```

### Docker Setup

```bash
# Build Docker image
docker build -t warehouse-logistics-env .

# Run container
docker run -e OPENAI_API_KEY=your_key \
           -e TASK_DIFFICULTY=easy \
           warehouse-logistics-env
```

## Observation Space

The observation returned at each step includes:

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
        },
        # ... more orders
    ],
    "warehouses": [
        {
            "warehouse_id": "NYC-01",
            "city": "New York",
            "stock_level": {"SKU-001": 50, "SKU-002": 30},
            "capacity": 1000
        },
        # ... more warehouses
    ],
    "step_count": 5,
    "processed_orders": 2,
    "failed_orders": 0,
    "last_action_result": {...}
}
```

## Action Space

Agents can take 5 types of actions:

```python
{
    "action_type": "check_stock|validate_address|allocate|ship|reroute",
    "order_id": "ORD-001",
    "warehouse": "NYC-01",  # Optional, required for allocate/reroute
    "message": "Optional reasoning"
}
```

### Action Types

| Action | Purpose | Reward | Requirements |
|--------|---------|--------|--------------|
| `CHECK_STOCK` | Verify inventory availability | +0.2 | Valid order & warehouse |
| `VALIDATE_ADDRESS` | Check delivery address validity | +0.15/-0.15 | Valid order |
| `ALLOCATE` | Reserve stock from warehouse | +0.25/-0.15 | Stock available, warehouse specified |
| `SHIP` | Complete order shipment | +0.3 | Order allocated |
| `REROUTE` | Move allocation to different warehouse | +0.2 | Stock at target warehouse |

## Task Descriptions

### Task 1: Easy - Single Order Stock Check

**Goal**: Process a single order by checking stock availability.

**Scenario**:
- 1 order for 10 units of SKU-001
- 2 warehouses (NYC has stock, LA out of stock)
- Agent must check stock and allocate from NYC

**Success Criteria**:
- Order status reaches `SHIPPED`
- Agent efficiently finds available warehouse
- Score: 0.0-1.0 based on completion and efficiency

**Baseline Score**: ~0.7

### Task 2: Medium - Batch Processing with Address Validation

**Goal**: Process batch of orders with address validation.

**Scenario**:
- 5 mixed orders
- 3 warehouses with varying inventory
- Some orders have invalid addresses
- Agent must validate and allocate appropriately

**Success Criteria**:
- Valid addresses processed correctly
- Invalid addresses rejected/failed
- Orders allocated efficiently
- All valid orders shipped

**Baseline Score**: ~0.6

### Task 3: Hard - Multi-Warehouse Routing with Rerouting

**Goal**: Handle out-of-stock scenarios requiring rerouting across warehouses.

**Scenario**:
- 4 orders requesting premium SKU
- First warehouse has insufficient stock
- Agent must detect shortage and reroute to backup warehouse
- Multi-warehouse decision making required

**Success Criteria**:
- Correct rerouting when primary warehouse out of stock
- Load balancing across warehouses
- All possible orders shipped
- Efficient path to completion

**Baseline Score**: ~0.5

## Reward Design

Rewards are **shaped** (continuous, intermediate) not binary:

```
Per-Step Rewards:
- Correct stock check: +0.2
- Valid address: +0.15
- Invalid address: -0.15
- Successful allocation: +0.25
- Failed allocation: -0.15
- Shipment: +0.3
- Reroute success: +0.2
- Invalid action: -0.1

Final Reward (Grader-based):
- Range: 0.0 (complete failure) to 1.0 (perfect)
- Deterministic scoring based on:
  - Orders completed
  - Efficiency (steps taken)
  - Actions taken
  - Warehouse utilization
```

## Grading System

Each task uses a deterministic grader that scores 0.0-1.0:

### Easy Task Grader
```python
score = 0.0
if order is stock_checked: += 0.5
if order is shipped: += 0.5
efficiency_penalty: -= (steps - 2) * 0.05
final_score = clamp(0.0, 1.0)
```

### Medium Task Grader
```python
score = 0.0
score += # of validate_address actions * 0.1
score += # of allocated orders * 0.15
score += # of shipped orders * 0.2
efficiency_penalty: -= (steps - 20) * 0.01
final_score = clamp(0.0, 1.0)
```

### Hard Task Grader
```python
score = 0.0
score += # of reroute actions * 0.15
score += # of shipped orders * 0.2
score += min(allocated_from_alternate * 0.25, 0.5)
score -= failed_orders * 0.1
efficiency_bonus: += max(0, (50 - steps) * 0.01)
final_score = clamp(0.0, 1.0)
```

## Quick Start

### Run Easy Task Locally

```python
from env import WarehouseLogisticsEnvironment, Action, ActionType

# Create environment
env = WarehouseLogisticsEnvironment(task_difficulty="easy")
obs = env.reset()

# Take a step
action = Action(
    action_type=ActionType.CHECK_STOCK,
    order_id="ORD-001",
    warehouse=None,
    message="Checking primary warehouse"
)

obs, reward, done, info = env.step(action)

print(f"Reward: {reward.value}")
print(f"Orders completed: {obs.processed_orders}")
print(f"Done: {done}")
```

### Run Baseline Inference

```bash
# Set up environment variables
export OPENAI_API_KEY=sk-...
export TASK_DIFFICULTY=easy
export MODEL_NAME=gpt-3.5-turbo

# Run inference with logging
python inference.py
```

Output format:
```
[START]
Task: easy
Orders: 1
Warehouses: 2

[STEP]
Step: 1
Action: check_stock
Reward: 0.200
Processed: 0/1

[STEP]
Step: 2
Action: allocate
Reward: 0.250
Processed: 0/1

[END]
Total Reward: 0.725
Completed: 1/1
Steps: 3
```

### Run in Docker

```bash
docker run -it \
  -e OPENAI_API_KEY=sk-... \
  -e TASK_DIFFICULTY=medium \
  warehouse-logistics-env
```

## Configuration

### Environment Variables

```bash
OPENAI_API_KEY          # Your OpenAI API key
MODEL_NAME              # Model (default: gpt-3.5-turbo)
API_BASE_URL            # Custom API base URL (optional)
HF_TOKEN                # Hugging Face token (optional, for compatibility)
TASK_DIFFICULTY         # Task difficulty: easy, medium, hard
```

### YAML Configuration

Edit `openenv.yaml` to customize:
- Task parameters
- Observation/action spaces
- Reward shaping
- Expected baseline scores
- Grading criteria

## Development

### Add New Task

```python
# In env/tasks.py
@staticmethod
def create_custom_task():
    orders = [...]
    warehouses = [...]
    config = TaskConfig(...)
    return orders, warehouses, config
```

### Add New Action Type

```python
# In env/models.py
class ActionType(str, Enum):
    NEW_ACTION = "new_action"

# In env/environment.py
def _action_new_action(self, order, action):
    # Implement action logic
    return Reward(...)
```

### Custom Grader

```python
# In env/grader.py
class CustomGrader(TaskGrader):
    def grade(self, orders, warehouses, action_history, step_count):
        # Implement scoring logic
        return score  # 0.0-1.0
```

## API Reference

### WarehouseLogisticsEnvironment

```python
class WarehouseLogisticsEnvironment:
    def __init__(self, task_difficulty: str = "easy")
    def reset() -> Observation
    def step(action: Action) -> Tuple[Observation, Reward, bool, Dict]
    def state() -> Observation
    def get_task_info() -> Dict
    def get_episode_summary() -> Dict
```

### Core Models

```python
class Observation
class Action
class Reward
class Order
class WarehouseLocation
class TaskConfig
```

All models include complete docstrings and type hints.

## Performance Benchmarks

Expected baseline performance with gpt-3.5-turbo:

| Task | Expected Score | Avg Episodes to Converge |
|------|-----------------|-------------------------|
| Easy | 0.70 | 3-5 |
| Medium | 0.60 | 10-15 |
| Hard | 0.50 | 20-30 |

## Testing

```bash
# Run environment tests
python -m pytest tests/

# Run single task
python -c "from env import *; env = WarehouseLogisticsEnvironment('easy'); obs = env.reset(); print(env.get_task_info())"

# Check code quality
python -m pylint env/
python -m mypy env/
```

## Code Quality

- ✅ Type hints on all functions
- ✅ Docstrings for all methods
- ✅ No placeholder code
- ✅ Fully deterministic for reproducibility
- ✅ Production-ready error handling
- ✅ Clean modular architecture

## Logging Format

The inference script logs exactly as specified:

```
[START]
<task info>

[STEP]
<action results>

[STEP]
...

[END]
<episode summary>
```

## Environment Variables Template

Create `.env`:
```bash
OPENAI_API_KEY=sk-your-key-here
MODEL_NAME=gpt-3.5-turbo
API_BASE_URL=
HF_TOKEN=
TASK_DIFFICULTY=easy
```

## Troubleshooting

### ImportError: No module named 'openai'
```bash
pip install --upgrade openai
```

### CUDA out of memory
Environment runs on CPU by default. No GPU required.

### OpenAI API errors
- Verify API key is valid
- Check rate limits
- Use `gpt-3.5-turbo` for faster inference

### Address validation too strict
Edit `validate_address()` in `env/utils.py` to adjust rules.

## Citation

```bibtex
@article{warehouse-logistics-env,
  title={Warehouse Logistics AI Environment},
  author={AI Systems Engineering},
  year={2024}
}
```

## License

MIT License - See LICENSE file

## Support

For issues, questions, or contributions:
- Check the troubleshooting section
- Review docstrings in source code
- Inspect `openenv.yaml` for configuration details

---

**Production-Ready** | **OpenEnv Compliant** | **Deterministic** | **Type-Safe**
