# Project rules

## No past to support

The code supports no earlier layout, name, checkpoint, or artifact.

- When a shape changes, the old shape is deleted, not read. No loader, branch, or fallback for "a file written before".
- Old analyses are deleted, not migrated: designs, threads, checkpoints, run dirs.
- No alias for a renamed route, class, function, or state key.
- A test that only proves an old shape still loads is deleted with the shape.
- The fixture datasets under `data/` and their claims files are data the tests use, not the past. They stay, and a
  field they spell is renamed in the file, not mapped in code.

The global writing and commit rules apply.
