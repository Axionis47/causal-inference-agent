# User guide

## Sign in and select data

Precondition: An operator has started the application and configured an account.

1. Open the application URL supplied by the operator.
2. Enter the configured email address and password.
3. Submit the login form.
4. Open the dataset library.
5. Upload a CSV file with a title.
6. Open the saved dataset.
7. Inspect the row preview and original column names.

Expected result: The dataset remains available after a page reload.
The preview displays source values as strings.
It does not infer scientific meanings or change the CSV.

Default intake limits are 10 MiB, 100,000 data rows, and 200 columns.
Headers must be nonempty and unique.
The file must contain at least one data row.

If upload fails, correct the reported input error.
If the result is uncertain after a reload, inspect the dataset list before you upload again.

Use **Data**, **Context**, and **Chat** in the workspace navigation to move between views.
An open conversation also has an **Analysis** view.
The dataset library remains available in the sidebar.

## Prepare and review context

Precondition: A saved dataset is open.

1. Select **Open context workspace**.
2. Add a text source, Markdown source, or pasted note.
3. Select the sources for preparation.
4. Start preparation.
5. Open the resulting draft revision.
6. Inspect the dataset statements and each relevant column.
7. Open citations to inspect their exact source passages.
8. Review, correct, withdraw, or mark statements unresolved.
9. Save a draft or reviewed revision.

Expected result: History contains a new immutable revision.
The earlier revision remains available.

Review means that a review action was recorded.
It does not mean that all statements are known or scientifically established.
The participant map shows shared relationship membership.
It is not the analysis causal graph.

If preparation fails, inspect its saved error and attempt count.
Use retry only when the application permits recovery.
A changed source or exhausted preparation needs a separate preparation.

## Discuss and explore

Precondition: A dataset or saved context revision is open.

1. Select **Open chat** or **Open chat with this context**.
2. Create a conversation.
3. Enter a question.
4. Press Enter to send the message.

Use Shift+Enter to add a newline.
An active input-method composition does not submit the message.

Expected result: The saved turn shows its status and then a complete reply or visible failure.
The conversation retains its exact input references.

Chat uses the full workspace width.
Switch between **Chat** and **Analysis** without losing an unsent message.
After **Inspect selected context**, use **Back to conversation** to return to the same saved chat.

For a chart or numerical exploration, request the computation explicitly.
Inspect the resulting code, method, figure, and limitations.
Use **Use in follow-up** to select a saved exploratory artifact for the next turn.

Exploration can inspect the dataset before analysis.
The analysis records prior exposure.
The later diagnostic restrictions cannot undo information that a person has already seen.

## Apply context changes and an analysis brief

Precondition: The conversation has an appropriate context revision.

1. Select **Update context** mode.
2. Discuss the intended question and any necessary context changes.
3. Ask for an analysis brief with target, treatments, comparison, population, unit, and timing.
4. Inspect the proposed edits and source references.
5. Select **Apply changes** when the proposal is correct.
6. Inspect the saved revision and readiness result.

Expected result: Apply saves the edits, brief, and chat context selection together.
Discussion alone does not save changes.
Earlier turns and analyses retain their previous context.

| Readiness label | Meaning | Next action |
| --- | --- | --- |
| Needs context | Essential references or bindings are missing, unresolved, or inconsistent | Resolve the listed items before Run analysis |
| Ready with assumptions | Required structure is present with recorded uncertainty | Inspect the uncertainty before Run analysis |
| Ready | Required structure is present | Inspect the brief before Run analysis |

Readiness does not prove identification or guarantee execution.
An optional unknown role does not automatically prevent readiness.

## Start and inspect analysis

Precondition: Readiness permits analysis preparation.

1. Select **Analysis** in the workspace navigation.
2. Check the finalized question and selected context revision.
3. Select **Run analysis**.
4. Follow the study, design, and computation progress.
5. Follow the run status, review requests, and available results.
6. If the run stops, inspect its status and recovery action.

Expected result: The run preserves its exact dataset, context, question, brief, and readiness receipt.
Later chat messages cannot enrich that run.

For troubleshooting, expand **Technical details**.
It contains agent summaries, accepted handoffs, saved tool receipts, budgets, and event logs.
Historical runs retain their original checkpoint inspectors.
Use these details to trace a finding to its supporting records or understand a recovery action.

Connection status is separate from scientific progress.
A connected event stream does not mean that an agent has finished.
An unavailable trace does not mean that a task did not run.

## Review and approve the frozen analysis

Precondition: The run reached `awaiting_approval`.

1. Open the frozen scientific specification.
2. Check every requested effect and contrast.
3. Check the population, adjustment, exclusions, missingness, inference, and reporting rules.
4. Inspect the graph, relied-upon assumptions, diagnostics, code, and review findings.
5. Select **Approve frozen analysis** only when the exact packet is acceptable.

Expected result: The application records approval for the displayed snapshot and program hashes.
The worker executes that program against its approved inputs.

If the program needs a correction, select **Request correction** and describe the problem.
The computation agent revises it within the remaining allowance. Accepted study and design findings remain unchanged.
To end the run, reject the displayed revision with a reason. The rejected packet remains saved.

A programming repair exposes a new revision and its diff.
Inspect the repair and approve it separately.
The repair cannot change the frozen scientific design.

## Inspect and discuss results

Precondition: The selected run has saved result artifacts.

1. Read the saved report and result chart.
2. Download the result table, code, and numerical facts.
3. Follow the result references to the frozen specification and assumptions.
4. Select **Discuss these results in chat** and ask about the saved analysis.
5. Inspect the reply's artifact references.

Expected result: Chat discusses only the analysis references authorized for that turn.
It can also discuss frozen context and failed checkpoints before a result exists.
It must not describe a blocked run as a completed estimate.

## Resume after an interruption

1. Reopen the saved dataset and conversation.
2. Select the same analysis run.
3. Inspect its current status and failure record.
4. Use retry only if the interface marks the operation retryable.

Expected result: Recovery preserves accepted work, failed attempts, budgets, and approvals.
It does not create new scientific choices.

Historical incompatible runs remain inspectable.
Their execution and approval controls are unavailable.
See [troubleshooting](operations.md#troubleshooting) for service and provider failures.
