# GETTING STARTED GUIDE

This guide will get you up and running with the Warehouse Logistics AI Environment in 5 minutes.

## Prerequisites

- Python 3.9+
- pip (Python package manager)
- Optional: OpenAI API key (for inference)
- Optional: Docker (for containerized deployment)

## Quick Start (No API Key Required)

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Setup

```bash
python verify_setup.py
```

Expected output:
```
✅ Python 3.9.x
✅ Project Structure
✅ Dependencies
✅ Module Imports
✅ Environment Creation
```

### 3. Run Basic Test (No API Needed)

```bash
python test_environment.py
```

This runs all 3 tasks:
- Easy: Single order stock check
- Medium: Batch processing with address validation
- Hard: Multi-warehouse routing

Output shows rewards and completion status.

## Advanced: OpenAI Integration

### 1. Get OpenAI API Key

1. Go to https://platform.openai.com/api-keys
2. Create new secret key
3. Copy the key

### 2. Configure Environment

```bash
# Copy template
cp .env.example .env

# Edit .env with your API key
# OPENAI_API_KEY=sk-your-key-here
```

### 3. Run Inference

```bash
python inference.py
```

Output follows exact format:
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

[END]
Total Reward: 0.725
Completed: 1/1
Steps: 3
```

### 4. Try Different Tasks

```bash
# Medium difficulty
TASK_DIFFICULTY=medium python inference.py

# Hard difficulty
TASK_DIFFICULTY=hard python inference.py
```

## Docker Deployment

### Build Image

```bash
docker build -t warehouse-logistics .
```

### Run Container

```bash
docker run -it \
  -e OPENAI_API_KEY=sk-your-key \
  -e TASK_DIFFICULTY=easy \
  warehouse-logistics
```

## Project Structure Overview

- **env/**: Core environment implementation
  - `environment.py`: Main OpenEnv-compliant environment
  - `models.py`: Pydantic data models
  - `tasks.py`: Task definitions (easy/medium/hard)
  - `grader.py`: Deterministic scoring
  - `utils.py`: Helper utilities

- **inference.py**: Baseline agent using OpenAI LLM

- **test_environment.py**: Test without API key

- **openenv.yaml**: Environment configuration

- **README.md**: Full documentation

## Example: Custom Agent

```python
from env import WarehouseLogisticsEnvironment, Action, ActionType

# Create environment
env = WarehouseLogisticsEnvironment(task_difficulty="easy")
obs = env.reset()

# Get task info
print(f"Task: {env.get_task_info()}")

# Your agent loop
for step in range(10):
    # Your agent decision logic here
    action = Action(
        action_type=ActionType.CHECK_STOCK,
        order_id=obs.orders[0].order_id,
        warehouse=None
    )
    
    # Take step
    obs, reward, done, info = env.step(action)
    
    print(f"Step {step}: Reward = {reward.value:.3f}")
    
    if done:
        break

# Get summary
summary = env.get_episode_summary()
print(f"Final reward: {summary['episode_reward']:.3f}")
print(f"Orders completed: {summary['completed_orders']}/{summary['total_orders']}")
print(f"Steps taken: {summary['total_steps']}")
```

## Troubleshooting

### Issue: ModuleNotFoundError: No module named 'pydantic'

**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: OpenAI API Error

**Solution:**
1. Check API key is correct in `.env`
2. Verify API key has sufficient credits
3. Check API key is not expired

### Issue: Task terminates immediately

**Solution:**
- This is expected - environment may finish in 1-2 steps for easy task
- Run `test_environment.py` to see detailed step-by-step execution

## Performance Tips

1. **Start with Easy**: Understand mechanics before challenging tasks
2. **Read Rewards**: Shaped rewards provide learning signal
3. **Check Error Messages**: Detailed info in reward.reason
4. **Save History**: Track action_history for analysis

## Next Steps

1. ✅ Run `verify_setup.py` - Verify everything works
2. ✅ Run `test_environment.py` - See environment in action
3. ✅ Read `README.md` - Full reference documentation
4. ✅ Test `inference.py` - Try OpenAI integration
5. ✅ Build custom agent - Create your own solution

## Key Concepts

### OpenEnv Specification

Environment follows OpenEnv spec:
```python
obs = env.reset()  # Get initial observation
obs, reward, done, info = env.step(action)  # Execute action
```

### Observation

Contains current state:
- `orders`: All orders with status
- `warehouses`: Inventory levels
- `step_count`: Current step
- `processed_orders`: Completed count

### Action

Agent chooses action:
- `action_type`: Type of action
- `order_id`: Target order
- `warehouse`: Target warehouse (optional)

### Reward

Feedback to agent:
- `value`: Numeric reward (-1 to +1)
- `breakdown`: Component analysis
- `reason`: Explanation

## Performance Benchmarks

With gpt-3.5-turbo baseline:

| Task | Expected Score | Notes |
|------|-----------------|-------|
| Easy | 0.70 | Usually 3-5 steps |
| Medium | 0.60 | Requires address validation |
| Hard | 0.50 | Multi-warehouse challenge |

## Support

- Check `README.md` for detailed docs
- Review docstrings in source code
- Run `test_environment.py` for examples
- Inspect `openenv.yaml` for configuration

---

**Ready to begin?** Run `python test_environment.py` now!
