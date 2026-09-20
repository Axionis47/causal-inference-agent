#!/usr/bin/env bash
# Regenerate the forced hand-offs the lane evals run on. Each mines the dataset's note into the memory once (a model call,
# skipped when the memory already holds more than the file's facts), then projects the pack from the memory.
set -euo pipefail
cd "$(dirname "$0")/.."
export LANGSMITH_TRACING=false
H="uv run python -m causal_agent.desk.handoff"

$H gov_transfers --mine --family adjustment --outcome Support --treatment Participation --columns "Support,Participation,Income_Centered,Education,Age" \
  --question "Did receiving the transfer raise support for the government?" \
  --assumption "forced into the adjustment lane as the negative case: the router would route this to discontinuity" \
  -o causal_agent/families/adjustment/evals/handoffs/gov_transfers_forced.json

$H cigar --mine --family diff_in_diff --outcome sales --treatment state --columns "sales,state,year,price,pimin,ndi,pop,cpi" \
  --question "What did California's 1989 tobacco tax do to cigarette sales?" --window "1989 to 1992 against the years before" --target on_treated \
  --assumption "forced into the diff-in-diff lane: the router would route this to synthetic control because one state got the change" \
  -o causal_agent/families/diff_in_diff/evals/handoffs/cigar_forced.json

$H marketing --mine --family diff_in_diff --outcome SalesInThousands --treatment Promotion --columns "SalesInThousands,Promotion,week,MarketSize,LocationID" \
  --question "Which of the three promotions produced the highest weekly sales?" --contrast level_vs_level \
  --assumption "forced into the diff-in-diff lane: the note says there is no week before the promotions began" \
  -o causal_agent/families/diff_in_diff/evals/handoffs/marketing_forced.json

$H card_krueger --mine --family discontinuity --outcome total_emp_nov --treatment state --columns "total_emp_nov,state,total_emp_feb" \
  --question "Did New Jersey's 1992 minimum wage rise reduce fast food employment?" --target on_treated \
  --assumption "forced into the discontinuity lane as the negative case: the notes name a state, not a score with a cutoff" \
  -o causal_agent/families/discontinuity/evals/handoffs/card_krueger_forced.json

$H students --mine --family discontinuity --outcome "math score" --treatment "test preparation course" \
  --columns "math score,test preparation course,reading score,writing score,lunch,parental level of education" \
  --question "Did completing the prep course raise math scores?" \
  --assumption "forced into the discontinuity lane as the negative case: numeric scores exist but no note states a cutoff rule" \
  -o causal_agent/families/discontinuity/evals/handoffs/students_forced.json
