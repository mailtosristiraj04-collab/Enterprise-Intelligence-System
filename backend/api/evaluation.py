"""Evaluation API routes."""

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.api.dependencies import get_db_session, get_evaluation_runner
from backend.core.config import Settings
from backend.evaluation.dataset import DatasetValidationError, EvaluationDataset
from backend.evaluation.runner import EvaluationRunner

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


class EvaluationRunRequest(BaseModel):
    """Validated evaluation execution parameters."""

    dataset_path: str | None = Field(default=None, description="Optional JSONL path on the server.")
    k_values: list[int] = Field(default=[5, 10], min_length=1)
    configurations: list[str] | None = None


@router.post("/run", summary="Run retrieval evaluation")
def run_evaluation(
    request: EvaluationRunRequest,
    session: Session = Depends(get_db_session),
    runner: EvaluationRunner = Depends(get_evaluation_runner),
) -> dict[str, Any]:
    default_path = runner.settings.get_evaluation_dataset_path().resolve()
    path = Path(request.dataset_path).resolve() if request.dataset_path else default_path
    if request.dataset_path and not path.is_relative_to(default_path.parent):
        from fastapi import HTTPException

        raise HTTPException(
            status_code=422,
            detail="dataset_path must remain inside the evaluation dataset directory.",
        )
    try:
        dataset = EvaluationDataset.load(path, session)
        report = runner.run(
            session,
            dataset,
            ks=request.k_values,
            configurations=request.configurations,
        )
        runner.persist(session, report)
        return report
    except (DatasetValidationError, ValueError) as exc:
        from fastapi import HTTPException

        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/runs", summary="List evaluation runs")
def list_evaluation_runs(
    session: Session = Depends(get_db_session),
    runner: EvaluationRunner = Depends(get_evaluation_runner),
) -> list[dict[str, Any]]:
    return runner.history(session)


@router.get("/runs/{run_id}", summary="Get an evaluation report")
def get_evaluation_run(
    run_id: str,
    session: Session = Depends(get_db_session),
    runner: EvaluationRunner = Depends(get_evaluation_runner),
) -> dict[str, Any]:
    from fastapi import HTTPException

    report = runner.get(session, run_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Evaluation run not found.")
    return report


@router.get("/metrics", summary="Get the latest evaluation metrics")
def latest_evaluation_metrics(
    session: Session = Depends(get_db_session),
    runner: EvaluationRunner = Depends(get_evaluation_runner),
) -> dict[str, Any]:
    runs = runner.history(session)
    if not runs:
        return {"available": False, "latest_run": None}
    return {"available": True, "latest_run": runner.get(session, runs[0]["run_id"])}
