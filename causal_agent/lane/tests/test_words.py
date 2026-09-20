"""What a check asks, in the question's words: every check a lane emits has a sentence, per-column checks fill the column."""

from __future__ import annotations

from causal_agent.common.contracts import CheckResult
from causal_agent.lane import words as W
from causal_agent.specialists.did.knowledge import load_checks as did_checks
from causal_agent.specialists.dowhy.knowledge import load_checks as dowhy_checks
from causal_agent.specialists.rd.knowledge import load_checks as rd_checks

EMITTED = {
    "dowhy": ["arms", "overlap", "separation", "balance", "identification", "adjusts_outside_candidates"],
    "did": ["units", "single_treated_unit", "parallel_untestable", "pre_trends", "staggered"],
    "rd": [
        "sides",
        "effective_rows",
        "density",
        "mass_points",
        "support",
        "compliance",
        "first_stage",
        "first_stage_weak",
        "no_first_stage",
        "covariate_continuity",
    ],
}


def test_every_check_a_lane_emits_has_a_sentence():
    for lane, cfg in (("dowhy", dowhy_checks()), ("did", did_checks()), ("rd", rd_checks())):
        for name in EMITTED[lane]:
            assert W.check_words(cfg, name) != name, (lane, name)


def test_a_per_column_check_fills_the_column_and_an_unknown_check_keeps_its_name():
    cfg = dowhy_checks()
    assert W.check_words(cfg, "balance.lunch", {"lunch": "lunch"}) == "how alike the two arms are on lunch before the adjustment, and after weighting"
    assert "parental level of education" in W.check_words(
        cfg, "balance.parental_level_of_education", {"parental_level_of_education": "parental level of education"}
    )
    assert W.check_words(cfg, "made_up") == "made_up"


def test_say_leads_the_detail_with_the_sentence_and_leaves_the_persons_flags_alone():
    cfg = dowhy_checks()
    rs = [
        CheckResult(contrast="c", name="balance.lunch", level="soft", detail="standardised mean difference 0.16"),
        CheckResult(contrast="all", name="belief.spillover", level="soft", detail="units could reach one another"),
    ]
    W.say(rs, cfg, {"lunch": "lunch"})
    assert rs[0].detail == "how alike the two arms are on lunch before the adjustment, and after weighting: standardised mean difference 0.16"
    assert rs[1].detail == "units could reach one another"
    W.say(rs, cfg, {"lunch": "lunch"})
    assert rs[0].detail.count("how alike") == 1  # said once
