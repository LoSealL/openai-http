"""
OpenTelemetry metrics instruments.

Provides counters, histograms, and gauges for service observability.
"""


try:
    from opentelemetry import metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader

    _provider = MeterProvider(
        metric_readers=[PeriodicExportingMetricReader(ConsoleMetricExporter(), export_interval_millis=60000)]
    )
    metrics.set_meter_provider(_provider)

    _meter = metrics.get_meter("openai_http", "1.0.0")

    request_counter = _meter.create_counter(
        "openai_http.requests.total",
        unit="1",
        description="Total number of HTTP requests",
    )

    request_duration = _meter.create_histogram(
        "openai_http.request.duration_seconds",
        unit="s",
        description="Request duration in seconds",
    )

    tokens_counter = _meter.create_counter(
        "openai_http.tokens.total",
        unit="1",
        description="Total tokens processed",
    )

    error_counter = _meter.create_counter(
        "openai_http.errors.total",
        unit="1",
        description="Total errors by type",
    )

    active_requests_gauge = _meter.create_up_down_counter(
        "openai_http.requests.active",
        unit="1",
        description="Currently active requests",
    )

    def record_request(method: str, endpoint: str, status: int, duration: float) -> None:
        labels = {"method": method, "endpoint": endpoint, "status": str(status)}
        request_counter.add(1, labels)
        request_duration.record(duration, labels)

    def record_tokens(model: str, prompt_tokens: int, completion_tokens: int) -> None:
        tokens_counter.add(prompt_tokens, {"model": model, "type": "prompt"})
        tokens_counter.add(completion_tokens, {"model": model, "type": "completion"})

    def record_error(error_type: str) -> None:
        error_counter.add(1, {"error_type": error_type})

except ImportError:
    def record_request(method: str, endpoint: str, status: int, duration: float) -> None:
        pass

    def record_tokens(model: str, prompt_tokens: int, completion_tokens: int) -> None:
        pass

    def record_error(error_type: str) -> None:
        pass
