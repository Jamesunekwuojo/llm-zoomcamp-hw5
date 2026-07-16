from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

# 1) OTel setup FIRST
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("llm-zoomcamp")

# 2) Then import starter
from starter import rag


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

            # OpenAI-style usage object (as used in the course setup)
            usage = response.usage
            input_tokens = usage.input_tokens
            output_tokens = usage.output_tokens

            # gpt-5.4-mini pricing from homework context:
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


