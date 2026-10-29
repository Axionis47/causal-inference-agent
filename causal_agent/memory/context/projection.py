"""A disposable causal working view derived from one immutable context revision."""

from causal_agent.memory import ops
from causal_agent.memory.context.review import Knowledge
from causal_agent.memory.records import Column, Memory


def bindings(knowledge: Knowledge, memory: Memory) -> Memory:
    """Keep catalogue coercion, belief authority and story rules in the existing gate."""
    for statement in knowledge.statements:
        binding = statement.binding
        if binding is None or statement.status == "withdrawn":
            continue
        if Memory.canonical(binding.address) != binding.address:
            raise ValueError("Requirement bindings must use their canonical catalogue address.")
        source = statement.review_source or statement.source
        human = source.kind in {"review", "message"} and statement.status in {"reviewed", "unresolved"}
        if not human:
            # An extraction's mapping never admits an assumption, even if a column was reviewed.
            continue
        if binding.address == "claim:story.text":
            raise ValueError("The verbatim study account is supplied by the desk, not a field edit.")
        errors = ops.apply(
            memory,
            [
                ops.Update(
                    address=binding.address,
                    value=binding.value,
                    status="unknown" if statement.status == "unresolved" else "confirmed",
                    source="user:turn:" + source.id,
                    said=source.text or statement.value,
                    reason=statement.qualifications or statement.reason,
                    evidence=["context:" + statement.id],
                )
            ],
        )
        if errors:
            raise ValueError("; ".join(errors))
    return memory


def validate_bindings(knowledge: Knowledge) -> None:
    """No data or model calls. Validate every submitted field against stable columns."""
    memory = Memory(name="context-review", columns={c.key: Column(name=c.name, key=c.key) for c in knowledge.columns})
    bindings(knowledge, memory)
