"""Optional agent observability with Arize Phoenix (open source, runs locally, no account).

Start the local Phoenix server in a separate terminal:

    PHOENIX_WORKING_DIR=data/phoenix phoenix serve

Then pass ``--trace`` to the CLI or the evaluation runner (or set PHOENIX_TRACING=true). Every LangGraph
node, Gemini call, Policy-agent tool call and token count is exported over OpenTelemetry and shown as a
nested trace at http://localhost:6006. Traces stay on this machine. They contain the redacted brief and
model prompts, so use synthetic briefs when recording a demo.
"""
import os

PROJECT = "ca6123-hdb"


def phoenix_enabled(flag: bool = False) -> bool:
    return flag or os.getenv("PHOENIX_TRACING", "").lower() in {"1", "true", "yes"}


def enable_phoenix(project: str = PROJECT) -> str | None:
    """Instrument LangChain and LangGraph once. Returns the trace URL, or None if Phoenix is unavailable."""
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from phoenix.otel import register
    except ImportError:
        return None
    base = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006").rstrip("/")
    # Simple (unbatched) export so short CLI runs flush every span before the process exits.
    provider = register(project_name=project, endpoint=f"{base}/v1/traces", batch=False, verbose=False)
    instrumentor = LangChainInstrumentor()
    if not instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.instrument(tracer_provider=provider)
    return f"{base}/projects"
