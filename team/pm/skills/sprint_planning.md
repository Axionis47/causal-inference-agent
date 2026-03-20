# Skill: Sprint Planning

Load when: creating tickets or planning sprints.

## Ticket format

```
Title: [TAG] Short imperative description

Labels: <agent-tag>, <priority p1/p2/p3>

User story:
As <who>, I need <what> so that <why>.

done_when:
One sentence. Exact condition that makes this ticket complete.

Acceptance criteria:
- criterion 1
- criterion 2
- criterion 3

Execute with:
  Primary: <subagent_type>
  Collaborators: <subagent_type> (if cross-domain work needed)
  Reason: one line

Test certification required: yes/no
```

## Agent assignment rules

| Subagent type | Owns |
|---------------|------|
| backend | backend/src/**/*.py, backend/requirements.txt |
| frontend | frontend/src/**/*.ts, frontend/src/**/*.tsx |
| qa | backend/tests/*, frontend/src/__tests__/* |
| pm | docs/*, team/* |

If a ticket crosses boundaries, assign primary to the dominant scope
and list the other as collaborator.

## Priority levels

| Priority | Meaning |
|----------|---------|
| p1 | Blocks pipeline from running end-to-end |
| p2 | Degrades analysis quality or user experience |
| p3 | Nice to have, technical debt, documentation |

## Sprint scope

A sprint should have:
- 1-2 p1 tickets (if any exist)
- 3-5 p2 tickets
- 1-2 p3 tickets for breathing room
- Every feature must be testable at sprint end
