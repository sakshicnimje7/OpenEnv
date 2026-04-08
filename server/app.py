"""HTTP server entrypoint for deploying the OpenEnv environment as a service."""

import os
from threading import Lock
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from env import Action, ActionType, WarehouseLogisticsEnvironment


class ResetRequest(BaseModel):
    """Optional payload for resetting with a specific task difficulty."""

    task_difficulty: str = Field(default="easy", pattern="^(easy|medium|hard)$")


class StepRequest(BaseModel):
    """Request model for environment step actions."""

    action_type: ActionType
    order_id: str
    warehouse: Optional[str] = None
    message: Optional[str] = ""


class EnvironmentService:
    """Thread-safe wrapper around a single environment instance."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._task_difficulty = os.getenv("TASK_DIFFICULTY", "easy")
        self._env = WarehouseLogisticsEnvironment(task_difficulty=self._task_difficulty)
        self._env.reset()

    def reset(self, task_difficulty: Optional[str] = None) -> Dict[str, Any]:
        """Reset environment and optionally switch difficulty."""
        with self._lock:
            if task_difficulty and task_difficulty != self._task_difficulty:
                self._task_difficulty = task_difficulty
                self._env = WarehouseLogisticsEnvironment(task_difficulty=task_difficulty)

            observation = self._env.reset()
            return observation.model_dump()

    def state(self) -> Dict[str, Any]:
        """Return current environment state."""
        with self._lock:
            return self._env.state().model_dump()

    def step(self, request: StepRequest) -> Dict[str, Any]:
        """Execute one step and return OpenEnv-compatible payload."""
        with self._lock:
            action = Action(
                action_type=request.action_type,
                order_id=request.order_id,
                warehouse=request.warehouse,
                message=request.message or "",
            )
            observation, reward, done, info = self._env.step(action)
            return {
                "observation": observation.model_dump(),
                "reward": reward.model_dump(),
                "done": done,
                "info": info,
            }

    def task_info(self) -> Dict[str, Any]:
        """Return current task metadata."""
        with self._lock:
            return self._env.get_task_info()

    def episode_summary(self) -> Dict[str, Any]:
        """Return episode summary."""
        with self._lock:
            return self._env.get_episode_summary()


app = FastAPI(
    title="Warehouse Logistics OpenEnv Service",
    version="1.0.0",
    description="Service wrapper exposing reset/step/state for warehouse logistics tasks.",
)
service = EnvironmentService()


@app.get("/")
def root() -> Dict[str, Any]:
    """Basic liveness endpoint."""
    return {
        "status": "ok",
        "service": "warehouse-logistics-env",
        "openenv": True,
    }


@app.get("/health")
def health() -> Dict[str, str]:
    """Healthcheck endpoint for deployment probes."""
    return {"status": "healthy"}


@app.post("/reset")
def reset(payload: Optional[ResetRequest] = None) -> Dict[str, Any]:
    """Reset environment and return initial observation."""
    difficulty = payload.task_difficulty if payload else None
    try:
        observation = service.reset(task_difficulty=difficulty)
        return {"observation": observation}
    except Exception as exc:  # pragma: no cover - defensive API guard
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/state")
def state() -> Dict[str, Any]:
    """Return current state without stepping."""
    try:
        return {"observation": service.state()}
    except Exception as exc:  # pragma: no cover - defensive API guard
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/step")
def step(payload: StepRequest) -> Dict[str, Any]:
    """Execute one action against the environment."""
    try:
        return service.step(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive API guard
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/task_info")
def task_info() -> Dict[str, Any]:
    """Return task metadata for current episode."""
    return service.task_info()


@app.get("/episode_summary")
def episode_summary() -> Dict[str, Any]:
    """Return episode summary metrics."""
    return service.episode_summary()


def main() -> None:
    """Launch Uvicorn server for local and container runtime."""
    import uvicorn

    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
