# CONTEXT.md

Decisions made, conventions locked, and deferred items. The goal is that
anyone (or Claude) reading this can understand *why* things are the way they
are, not just *what* they are.

_Last updated: [DATE]_

---

## Decisions made

### [Decision area — e.g. "Auth approach"]
**Decision:** [What was decided]  
**Why:** [The reason — constraint, tradeoff, external requirement]  
**Consequence:** [What this means for future work — what it rules out, what it enables]

### [Decision area]
**Decision:**  
**Why:**  
**Consequence:**

---

## Conventions locked (don't change without flagging)

- **[Convention]** — [why it exists / what breaks if you change it]
- **[Convention]** — [e.g. "Status values are lowercase strings — the DB, UI, and email templates all key on the same literals"]

---

## Data shapes / contracts

[Document any shared JSON shapes, enum values, or field names that span
multiple parts of the codebase. Changing these is a multi-file operation.]

```ts
// Example: locked status lifecycle
type Status = "draft" | "submitted" | "approved" | "rejected";
// Order matters — never reorder or rename without updating all consumers.
```

---

## Deferred items

Things that came up but were explicitly pushed to a later phase. If they
reappear, check here first before reopening the discussion.

- **[Item]** — deferred because [reason]. Revisit at [phase / condition].
- **[Item]** — [reason].

---

## Things that surprised us / lessons learned

- [Gotcha or non-obvious thing discovered during build — useful for next sprint]
