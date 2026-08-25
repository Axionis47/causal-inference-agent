-- One requirement can be open for several scopes of the same design revision
-- (e.g. column.measurement_timing for two columns); the row key must carry the
-- scope (T-012 integration; D-048).
ALTER TABLE design.context_requirements
    DROP CONSTRAINT context_requirements_pkey;
ALTER TABLE design.context_requirements
    ADD PRIMARY KEY (analysis_id, design_revision, requirement_id, scope_id);
