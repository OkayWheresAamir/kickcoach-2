# CLAUDE.md

This file provides guidance to Claude Code when working with code in this
repository. It is shared and committed — keep it limited to facts true for
anyone working in this repo. Personal working-style preferences belong in
`CLAUDE.local.md` instead (gitignored, not this file).

## What this project is

[PROJECT NAME] — one-line description of what it does and who it's for.

[2–3 sentences on the core problem it solves and the user types involved.]

## Current focus

[What is the active area of work right now? Which surface or feature set?]

**In scope now:**
- [Feature / screen / module A]
- [Feature / screen / module B]

**Parked — do not edit this sprint:**
- [Module or file that is done/frozen/belongs to a later phase]

## How to treat the plan in PROJECT_STATE.md

- Phase sequencing is fixed; don't reorder it.
- Specific technical implementation choices are defaults, not mandates. If
  you see a better approach, say so explicitly and wait for a decision before
  doing it.
- When in doubt: a working, secure implementation beats matching the plan's
  suggested detail to the letter.

## Commands

```bash
# [Fill in your dev/build/test commands here]
npm run dev      # Start dev server
npm run build    # Production build
npm run lint     # Linter
npm test         # Run tests
```

## Stack

[List your stack here — e.g. Next.js (TypeScript, App Router) · Prisma → Supabase · Tailwind + shadcn/ui · Vercel · Vitest]

## Key docs (read when relevant — don't load all by default)

- `docs/PROJECT_STATE.md` — live status, current phase, open questions. Check first.
- `docs/CONTEXT.md` — decisions made and why, conventions, deferred items.
- [Add other key docs as they exist, e.g. wireframes, schema, API spec]

## Conventions

- [Fill in your file/folder structure — e.g. App Router: pages at `src/app/[route]/page.tsx`]
- TypeScript strict mode — no `any`; use `unknown` then narrow.
- Comments explain *why*, not *what*.
- [Add any naming conventions, branch naming, PR rules, etc.]

## Path alias

[e.g. `@/*` maps to the project root — or remove if not applicable]

## When stuck

- Status / "what's done, what's next" → check `docs/PROJECT_STATE.md` first.
- UI/UX question → [link to wireframes or design doc if you have one].
- Stack question → [your stack here]. Don't introduce new frameworks without flagging it.
