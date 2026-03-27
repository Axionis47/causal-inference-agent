"""Adapt existing scientific fixtures to the prepared-request test boundary."""
from inspect import Parameter, signature


def compact_fixture(value, schema, root=None):
    """Serialize canonical scientific fixtures into the actual compact wire shape."""
    root = root or schema
    if "$ref" in schema:
        return compact_fixture(value, root["$defs"][schema["$ref"].split("/")[-1]], root)
    if "anyOf" in schema:
        if value is None:
            return None
        return compact_fixture(value, next(item for item in schema["anyOf"] if item.get("type") != "null"), root)
    if schema.get("type") == "object" and isinstance(value, dict):
        fields = {field["description"].split(". ", 1)[0]: (key, field) for key, field in schema["properties"].items()}
        return {fields[name][0]: compact_fixture(item, fields[name][1], root) if item is not None else None
                for name, item in value.items()}
    if schema.get("type") == "array" and isinstance(value, list):
        return [compact_fixture(item, schema["items"], root) for item in value]
    return value


def model_double(propose):
    def call(request):
        params = signature(propose).parameters
        options = {"images": list(request.images), "max_output_tokens": request.output_limit,
                   "thinking_budget": request.thinking_budget,
                   "reference_decoder": request.reference_decoder, "prepared": request}
        if request.schema.get("description", "").startswith("Use the compact property keys") and request.reference_decoder:
            options["reference_decoder"] = lambda value: request.reference_decoder(compact_fixture(value, request.schema))
        if not any(p.kind == Parameter.VAR_KEYWORD for p in params.values()):
            options = {k: v for k, v in options.items() if k in params}
        return propose(request.response_model, request.system, request.payload, request.model, **options)
    return call
