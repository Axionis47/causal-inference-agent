"""Explicit negative controls, applied only inside fresh integration resources."""

from contextlib import contextmanager

from sqlalchemy import text


def apply_mutation(name, storage, monkeypatch):
    if not name:
        return
    if name.startswith("chat_"):
        import inspect
        import textwrap

        from causal_agent.desk import graph as chat
        from causal_agent.server.context.repository import Repository as Context
        from causal_agent.server.conversations.commands import Repository as Commands

        if name == "chat_owner":
            target, function = Context, "row"
            changes = [("S.revisions.c.owner_id == self.owner", "True"), ("S.revisions.c.version_id == self.version_id", "True")]
        elif name == "chat_stale":
            target, function = Context, "check_revision"
            changes = [('if current["id"] != expected_id:', "if False:")]
        elif name == "chat_duplicate":
            target, function = Commands, "accept"
            changes = [("return dict(existing), False", "return dict(existing), True")]
        elif name == "chat_implicit":
            target, function = chat, "process"
            changes = [
                (
                    'if command.kind == "correct":',
                    'if command.kind == "message":\n        runtime.correct(command.submission_id, [ContextEdit(action="review", target_id="s1")], command.text)\n    if command.kind == "correct":',
                )
            ]
        else:
            raise AssertionError(name)
        fn = getattr(target, function)
        source = textwrap.dedent(inspect.getsource(fn))
        for before, after in changes:
            assert source.count(before) == 1, (name, before)
            source = source.replace(before, after)
        namespace = dict(fn.__globals__)
        exec(compile(source, "<chat-negative-control>", "exec"), namespace)
        monkeypatch.setattr(target, function, namespace[function])
    elif name.startswith("semantic_"):
        import inspect

        from causal_agent.memory.context import extract
        from causal_agent.server.context import service

        def change(module, function, before, after):
            fn = getattr(module, function)
            source = inspect.getsource(fn)
            assert source.count(before) == 1, (function, before)
            namespace = dict(fn.__globals__)
            exec(compile(source.replace(before, after), "<context-negative-control>", "exec"), namespace)
            monkeypatch.setattr(module, function, namespace[function])

        if name == "semantic_owner":
            change(service, "owned", "I.datasets.c.owner_id == owner", "True")
            change(service, "revision_row", "S.revisions.c.owner_id == owner", "True")
        elif name == "semantic_version":
            change(service, "revision_row", "S.revisions.c.version_id == version_id", "True")
        elif name == "semantic_citation":
            change(extract, "validate", "source[cite.start : cite.end] != cite.quote", "False")
        elif name == "semantic_assessment":
            change(service, "publish", "if errors or result.assessment.corrections:", "if errors:")
        elif name == "semantic_participant":
            change(extract, "validate", 'errors.append(f"{relation.id}: unknown {p.kind} participant {p.id}.")', "pass")
            # Publication now also constructs the canonical knowledge revision.
            # Disable its independent identity validation for this deliberate
            # acceptance defect, so the named test must catch the omitted gate.
            change(
                service,
                "publish",
                "knowledge=extracted(result.content, columns, pack_id)",
                "knowledge=Knowledge.model_construct(columns=columns, statements=[], dispositions=[])",
            )
        else:
            raise AssertionError(name)
    elif name.startswith("profile_"):
        import inspect

        from causal_agent.profile import inspection
        from causal_agent.server.profiling import service

        changes = {
            "profile_owner": (service, "owned", "I.datasets.c.owner_id == owner", "True"),
            "profile_version": (service, "column", 'S.column_results.c.version_id == info["version_id"]', "True"),
            "profile_numeric": (inspection, "column", "elif len(valid) == len(nonempty):", "elif valid:"),
            "profile_failure": (inspection, "status", 'return "partial" if any(c.status != "computed" for c in checks) else "complete"', 'return "complete"'),
        }
        module, function, before, after = changes[name]
        source = inspect.getsource(getattr(module, function))
        assert source.count(before) == 1
        namespace = dict(getattr(module, function).__globals__)
        exec(compile(source.replace(before, after), "<profiling-negative-control>", "exec"), namespace)
        monkeypatch.setattr(module, function, namespace[function])
    elif name.startswith("library_"):
        import inspect

        from causal_agent.server.ingestion import library

        changes = {
            "library_owner": ("S.datasets.c.owner_id == owner", "True"),
            "library_pagination": (" < after", " <= after"),
            "library_ready": ('S.operations.c.status == "ready"', "True"),
        }
        before, after = changes[name]
        source = inspect.getsource(library.list_owned)
        assert source.count(before) == 1
        namespace = dict(vars(library))
        exec(compile(source.replace(before, after), "<library-negative-control>", "exec"), namespace)
        monkeypatch.setattr(library, "list_owned", namespace["list_owned"])
    elif name == "cleanup_noop":
        monkeypatch.setattr(storage, "cleanup", lambda: 0)
    elif name == "owner_fk_removed":
        with storage.engine.begin() as db:
            db.execute(text("ALTER TABLE ingestion.contexts DROP CONSTRAINT contexts_version_id_owner_id_fkey"))
    elif name == "context_mutable":
        with storage.engine.begin() as db:
            db.execute(text("DROP TRIGGER immutable ON ingestion.contexts"))
    elif name == "lock_disabled":

        @contextmanager
        def unlocked(owner, key):
            with storage.engine.connect() as db:
                yield db

        monkeypatch.setattr(storage, "locked", unlocked)
    elif name == "verification_skipped":

        def put(key, stream, size, digest):
            storage.objects.client.put_object(Bucket=storage.objects.bucket, Key=key, Body=stream.read())

        monkeypatch.setattr(storage.objects, "put", put)
    elif name == "owner_read_bypassed":
        original = storage.get

        def get(owner, dataset_id, db=None):
            from sqlalchemy import select

            from causal_agent.server.ingestion import schema as S

            with storage.engine.connect() as connection:
                actual = connection.scalar(select(S.datasets.c.owner_id).where(S.datasets.c.id == dataset_id))
            return original(actual, dataset_id, db=db)

        monkeypatch.setattr(storage, "get", get)
    elif name == "profiler_called":
        original = storage.ingest

        def ingest(*args, **kwargs):
            from causal_agent.profile.profiler import profile

            profile("should-never-be-profiled.csv")
            return original(*args, **kwargs)

        monkeypatch.setattr(storage, "ingest", ingest)
    else:
        raise ValueError(f"Unknown mutation: {name}")
