# web — the page

Vite + React + TypeScript, no UI kit. One shell for every route: the datasets in a sidebar on the left (collapse and
expand; remembered), the page in the middle. `/` the datasets, `/new` upload and describe, `/d/:name` the conversation.
On the conversation page a slim strip on the right names the working memory (claims, results, files). Clicking a tab,
a claim mark, a run or a file opens the inspector: a wide pane that pushes the chat left, with full tables, a resizable
splitter, Close and Escape. What is open lives in the URL hash (`#claims`, `#runs/2`, `#files/2/report.md`), so a reload
keeps it. Below 960px the sidebar and the inspector open as overlays.

```bash
npm install
npm run dev      # port 5173, proxies /api to the server on 8000 (or $API_PORT)
npm test         # vitest: the pure helpers (composeAnswer, selection hash, pane bounds, table rows, labels)
npm run build    # web/dist, served by `python -m causal_agent.server`
```

The page talks to `causal_agent/server` only. It polls the session view every 1.5 s while a step runs. Chips answer
the interview's typed questions; `src/compose.ts` quotes each answered question beside its answer so the graph still
reads plain prose. Tokens in `src/styles/tokens.css` are the project's, shared with `docs/design.html`.
