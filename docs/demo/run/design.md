DESIGN
  contrasts    completed vs none (the pack names the level that means the unit got the change)
  graph
    treatment test_preparation_course -> outcome math_score
      lunch: confounder  [claim:assignment.depends_on, col:lunch.when]
      parental_level_of_education: confounder  [claim:assignment.depends_on, col:parental_level_of_education.when]
      gender: outcome driver  [col:gender.when]
      race_ethnicity: outcome driver  [col:race_ethnicity.when]
      excluded reading_score: set at or after the change; the pack forbids adjusting for it [design.forbidden]
      excluded writing_score: set at or after the change; the pack forbids adjusting for it [design.forbidden]
  estimand     backdoor; adjust for parental_level_of_education, lunch
  check        pass check:completed_vs_none.arms  how many rows got the change and how many did not: 358 treated, 642 control
  check        pass check:completed_vs_none.overlap  whether both arms are found across the whole range of what the offer depended on: 100% of rows inside the score range both arms cover (0.28 to 0.44)
  check        pass check:completed_vs_none.separation  whether what the offer depended on all but decides who got the change: a score model tells the arms apart with AUC 0.56
  check        soft check:completed_vs_none.balance.parental_level_of_education  how alike the two arms are on parental level of education before the adjustment, and after weighting: standardised mean difference 0.16 between arms before adjustment, 0.01 after weighting on the score
  check        pass check:completed_vs_none.balance.lunch  how alike the two arms are on lunch before the adjustment, and after weighting: standardised mean difference 0.04 between arms before adjustment, 0.01 after weighting on the score
  estimator    propensity_score_weighting (+ linear_regression as secondary)  params {'weighting_scheme': 'ips_weight', 'min_ps_score': 0.05, 'max_ps_score': 0.95, 'num_simulations': 100}
  refuters     placebo_treatment_refuter, random_common_cause, data_subset_refuter
  target       ate
  frozen at    2026-09-27T23:14:17.250100+00:00