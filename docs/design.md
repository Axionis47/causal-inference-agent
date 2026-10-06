# The page: a design guideline

What a turn, an artifact, a cited address, a question and the handoff look like on the page, and the rules that hold
them together. Anyone changing the page reads this first; a new piece that does not fit one of the shapes below gets its
shape written here before it is built. The wire it draws from is [page.md](page.md); the tokens are
[tokens.css](../web/src/styles/tokens.css).

## What the page is for

The chat is the thing. One column of conversation, the person's words and the desk's replies, in the order they happened.
The desk decides; the page shows. Nothing on the page computes, guesses or fills in: every word in a reply came from the
desk, every number sits beside the address it was read from, every question is one the desk asked.

The inspector is the memory's shelves, opened beside the chat when the person wants to look: the understanding (the
claims and the matrix), the experiments (the runs, their results and files), the journal (every step and what it read).
Artifacts stay in the chat, beside the turn that asked for them.

## The tokens

`tokens.css` is the whole palette and type scale; a component names a token, never a colour. Paper and ink for the page and
the text, three inks for weight (`--ink`, `--ink-2`, `--ink-3`), one rule colour for every line, one panel colour for every
surface, one accent for the one thing on a screen that acts. The monospace face (`--mono`) is for what the machine said:
addresses, column names, values, node names. Everything the person or the desk said is in the text face.

A status is a mark, never a colour alone: settled, open, refuted, a contradiction each have a word or a glyph beside the
colour, so the page reads in grey.

## A turn

```
you · before the run · 14:02
the course was offered first by lunch status, then to anyone who asked

desk · before the run · 14:02
Noted: assignment.kind is drafted as own_choice from what you said [claim:assignment.kind].
```

- The head names who, the phase (before or after the run) and the time. Nothing else.
- The person's words are shown exactly as sent. The desk's reply is prose; a numbered line gets its number set heavier.
- An address in square brackets is drawn as a mark (`.addr`): small, monospace, the brackets dropped, the whole address in
  the hover. A reply never shows a number without its mark somewhere in the same line or the next.
- A turn that asked for nothing ends there. A question the desk attached (under run intent) opens below the reply as the
  composer's question, never inside the reply's prose.

## An artifact beside a reply

An exploration or a picture made from the file lands under the turn that asked for it, as one figure:

```
artifact:ab12cd34  The mean math score is 64.7 under free/reduced lunch and 70.0 under standard. (before any run)
[the image, when one was drawn]
mean_free_reduced 64.70     mean_standard 70.03
lunch         n
standard      645
free/reduced  355
the table: by_lunch.csv
```

- The caption first: the artifact's address, its summary, and whether it is about a design or before any run.
- The image at the column's width when there is one.
- Every fact with its own address in the hover (`artifact:<id>.<fact>`), so what the chat cites later can be found.
- A table's first rows inline, the whole file one click away, with the count of rows not shown.
- Nothing is re-drawn by the page: the image is the file the agent wrote, the facts are its `facts.json`.

## A cited address

The address is the unit of trust. Wherever the desk states a number, the page shows the mark it rests on; wherever a
mark appears, the hover gives the full address. The inspector's claims table shows the same addresses, so a reader can go
from a sentence in the chat to the field it rests on, its status, its source and the sentence the person said. A later
piece may make the mark a link that opens that shelf at that address; until then the mark and the hover are the contract.

## A question with its recommended option

Under run intent the desk asks one thing at a time. The composer shows it as one row:

```
1. Which value in that column means the student got the course?  · asked for assignment
   [completed ✓ recommended]  [none]  [other]
```

- One question, numbered, with the fields it is asked for after the dot.
- The options as chips. The recommended one comes first and says so; the others follow; "other" frees the text box.
- Picking a chip composes the answer beside the question's text (`compose.ts`), so the desk still reads prose.
- A readback ("Is this right?") is the same shape with yes and no as its chips; no opens the text box.
- Never a modal, never a form: the question sits where the next message goes.

## The ready moment and the run

When nothing blocks, the desk shows the design it would run as a reply, the pill reads "ready to run", and the composer's
button reads "Yes, run it". The button sends words ("yes, run it"); the desk reads them like any other turn. Nothing on the
page launches a run.

## The handoff view

What the run will rest on, before the word is given: every field of the handoff with its status mark, its value, the
sentence it rests on and its source, in the order the design reads them. An edit on the view is the person's word: it is
sent as a message naming the field, and the desk writes it through its one write path and shows the field again. The view
never writes a file and never holds a value the memory does not.

## Layout

The datasets in a sidebar on the left; the conversation in the middle, one column, at most 72 characters of prose wide;
the composer pinned beneath it; before a run, the status strip under the composer with one mark per claim kind; the
inspector as a pane on the right, opened from the strip or the tab strip, with its width kept in the URL hash. Below 960
pixels the sidebar and the inspector open as overlays over the conversation, which keeps its width.

## What the page never does

- Decide anything: which question to ask, whether a run is ready, what a number means.
- Show a number without an address, or an address without a hover.
- Put a second column of chat, a modal, or a form between the person and the composer.
- Re-draw an artifact from its data, or compute a table the agent did not write.
- Use a colour as the only carrier of a status.
