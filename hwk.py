from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from starter import rag

import sqlite3
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult


class SQLiteSpanExporter(SpanExporter):

    def __init__(self, db_path="traces.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS spans (
                name TEXT,
                start_time INTEGER,
                end_time INTEGER,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cost REAL
            )
        """)
        self.conn.commit()

    def export(self, spans):
        for span in spans:
            attrs = dict(span.attributes or {})
            self.conn.execute(
                "INSERT INTO spans VALUES (?, ?, ?, ?, ?, ?)",
                (
                    span.name,
                    span.start_time,
                    span.end_time,
                    attrs.get("input_tokens"),
                    attrs.get("output_tokens"),
                    attrs.get("cost"),
                ),
            )
        self.conn.commit()
        return SpanExportResult.SUCCESS

    def shutdown(self):
        self.conn.close()

    def force_flush(self):
        return True

# 1) OTel setup FIRST
provider = TracerProvider()
# provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
provider.add_span_processor(
    SimpleSpanProcessor(SQLiteSpanExporter("traces.db"))
)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("llm-zoomcamp")




class RAGTraced(type(rag)):
    def rag(self, query):
        with tracer.start_as_current_span("rag"):
            return super().rag(query)

    def search(self, query):
        with tracer.start_as_current_span("search"):
            return super().search(query)

    def llm(self, prompt):
        with tracer.start_as_current_span("llm") as span:
            response = super().llm(prompt)

            # OpenAI-style usage object 
            usage = response.usage
            input_tokens = usage.input_tokens
            output_tokens = usage.output_tokens

            # pricing from homework context:
            # input: 0.15 / 1M, output: 0.60 / 1M
            cost = (input_tokens * 0.15 + output_tokens * 0.60) / 1_000_000

            span.set_attribute("input_tokens", input_tokens)
            span.set_attribute("output_tokens", output_tokens)
            span.set_attribute("cost", cost)

            return response


if __name__ == "__main__":
    traced_rag = RAGTraced(rag.index, rag.llm_client, model=rag.model)
    query = "How does the agentic loop keep calling the model until it stops?"
    answer = traced_rag.rag(query)
    print("\nANSWER:\n", answer)


