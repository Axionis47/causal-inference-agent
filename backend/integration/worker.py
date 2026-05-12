"""Delivery composition root. Job identities and recovery schedules are stable."""

import asyncio
from uuid import UUID

from backend.analysis import jobs as analysis_jobs
from backend.chat import jobs as chat_jobs
from backend.integration.readers import Analyses
from backend.jobs import app
from backend.models import now


@app.task(name="chat.respond", queue="chat")
async def respond(turn_id: str) -> None:
    await asyncio.to_thread(chat_jobs.process, UUID(turn_id), Analyses())


@app.periodic(cron="* * * * *")
@app.task(name="chat.recover", queue="chat")
async def recover_chat(timestamp: int) -> None:
    for job in await app.job_manager.get_stalled_jobs(task_name="chat.respond", seconds_since_heartbeat=60):
        await app.job_manager.retry_job(job, retry_at=now())
    failed = await app.job_manager.list_jobs_async(task="chat.respond", status="failed")
    await asyncio.to_thread(chat_jobs.mark_failed, [job.id for job in failed if job.id is not None])



@app.task(name="analysis.run", queue="analysis")
async def analyze(run_id: str) -> None:
    await asyncio.to_thread(analysis_jobs.process, UUID(run_id))


@app.periodic(cron="* * * * *")
@app.task(name="analysis.recover", queue="analysis")
async def recover_analysis(timestamp: int) -> None:
    for job in await app.job_manager.get_stalled_jobs(task_name="analysis.run", seconds_since_heartbeat=60):
        await app.job_manager.retry_job(job, retry_at=now())
    failed = await app.job_manager.list_jobs_async(task="analysis.run", status="failed")
    ids = [job.id for job in failed if job.id is not None]
    if ids:
        await asyncio.to_thread(analysis_jobs.mark_failed, ids)

