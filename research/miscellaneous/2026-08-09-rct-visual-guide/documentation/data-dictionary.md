# Data dictionary

Definitions transcribed from deposited README; coding/ranges and missingness computed from CSV. SAS numeric `.` and blank fields parse as missing. No deduplication or recoding of endpoints. README says `BTW` where actual variable is `BWT`; use `BWT`. Subsample indicators are insufficiently defined for independent estimand identification.

| Variable | Original name | Definition | Observed coding | Missing | Analytical use |
|---|---|---|---|---:|---|
| Study_ID | Study_ID | Study ID | 4554 distinct text values (not exposed) | 0 | Audit only |
| RCT_Code | RCT_Code | RCT Code | Observed numeric range 1–155; 155 distinct values | 0 | Audit only |
| Sex_of_Infant | Sex_of_Infant | Infant sex | Female; Male | 0 | Baseline descriptive |
| Birth_weight | Birth_weight | Birth weight (original, use BTW instead) | Observed numeric range 280–4500; 87 distinct values | 0 | Audit only |
| Delivered_by | Delivered_by | Delivered by | Dhai; Doctor; Family member; Paramedical worker | 0 | Audit only |
| Delivered_Mode | Delivered_Mode | Delivered Mode | Assisted delivery; Assisted vagin; C-section; Normal vaginal | 0 | Audit only |
| Colostrum_Given | Colostrum_Given | Colostrum given | Discarded; Given to baby | 0 | Audit only |
| Other_than_Breast_milk | Other_than_Breast_milk | Other than breast milk | 0; 1 | 0 | Audit only |
| Completed_Study | Completed_Study | Completed study | Completed study; Infant Died; Lost to Follow-; Withdrew from s | 0 | Audit only |
| Dose1 | Dose1 | Dose1 | AM; NU; PM | 11 | Audit only |
| Dose2 | Dose2 | Dose2 | AM; NU; NULL;; PM | 29 | Audit only |
| Dose3 | Dose3 | Dose3 | AM; NU; PM | 43 | Audit only |
| Dose4 | Dose4 | Dose4 | AM; NU; PM | 52 | Audit only |
| Dose5 | Dose5 | Dose5 | AM; NU; PM | 61 | Audit only |
| Dose6 | Dose6 | Dose6 | AM; NU; PM | 83 | Audit only |
| Dose7 | Dose7 | Dose7 | AM; NU; PM | 80 | Audit only |
| Total_Doses | Total_Doses | Total Doses | 0; 1; 2; 3; 4; 5; 6; 7 | 0 | Audit only |
| Cow_milk_60d | Cow_milk_60d | Cow milk 60d | 0; 1 | 0 | Audit only |
| Honey | Honey | Honey | 0; 1 | 0 | Audit only |
| Gripe_water | Gripe_water | Gripe water | 0 | 0 | Audit only |
| Infant_formula | Infant_formula | Infant formula | 0; 1 | 0 | Audit only |
| Water_boil | Water_boil | Water boil | 0; 1 | 0 | Audit only |
| Water_unboil | Water_unboil | Water unboil | 0 | 0 | Audit only |
| Vitamin_drops | Vitamin_drops | Vitamin drops | 0; 1 | 0 | Audit only |
| Others | Others | Others | 0; 1 | 0 | Audit only |
| Weight_Day_28 | Weight_Day_28 | Weight at day 28 | Observed numeric range 300.0–5900.0; 68 distinct values | 198 | Audit only |
| Weight_Enroll | Weight_Enroll | Weight at enrollment | Observed numeric range 220.0–29000.0; 52 distinct values | 3 | Audit only |
| Mothers_parity | Mothers_parity | Mother’s parity (original, use MOMPAR instead) | 0.0; 1.0; 10.0; 2.0; 20.0; 27.0; 3.0; 30.0; 4.0; 45.0; 5.0; 6.0; 7.0; 8.0; 9.0 | 5 | Audit only |
| Treat_time | Treat_time | Treatment time (original, use ADMIN instead) | 0.0; 1.0; 2.0; 3.0; 4.0; 5.0; 8.0 | 6 | Audit only |
| Twins | Twins | Twins | 0; 1 | 0 | Audit only |
| MOMPAR | MOMPAR | Parity (with outliers/impossible values set to missing) | 1.0; 10.0; 2.0; 3.0; 4.0; 5.0; 6.0; 7.0; 8.0; 9.0 | 15 | Baseline descriptive |
| BWT | BWT | Birth weight (grams) | Observed numeric range 2000–4500; 86 distinct values | 0 | Baseline descriptive |
| wt7 | wt7 | Weight at day 7 | Observed numeric range 2000.0–5500.0; 53 distinct values | 9 | Audit only |
| wt60 | wt60 | Weight at day 60 | Observed numeric range 2600.0–6800.0; 59 distinct values | 250 | Audit only |
| WTDIFF | WTDIFF | Difference between day 60 and birthweight (grams) | Observed numeric range 0.0–4000.0; 111 distinct values | 250 | Audit only |
| MOMAGE | MOMAGE | Mother’s age | Observed numeric range 13.0–43.0; 28 distinct values | 14 | Baseline descriptive |
| TIME | TIME | Age of event | Observed numeric range 4.0–60.0; 57 distinct values | 3852 | Audit only |
| POPBLOCK12 | POPBLOCK12 | Block position (0 if 3, 4; 1 if not 3, 4) | 0; 1 | 0 | Sensitivity |
| POPBLOCK123 | POPBLOCK123 | Block position (0 if 4; 1 if not 4) | 0; 1 | 0 | Sensitivity |
| DZ | DZ | Disease code (see codes below) | 1.0; 10.0; 11.0; 13.0; 15.0; 16.0; 18.0; 19.0; 2.0; 20.0; 4.0; 5.0; 6.0; 7.0; 8.0; 9.0 | 3852 | Audit only |
| SEPSIS | SEPSIS | Sepsis (1=yes; 0=no) | 0; 1 | 0 | Secondary endpoint |
| SEPSISDEATH | SEPSISDEATH | Sepsis or death (1=yes, 0=no) | 0; 1 | 0 | Primary endpoint |
| DEATH | DEATH | Death (1=yes; 0=no) | 0; 1 | 0 | Secondary endpoint |
| SEPSISB | SEPSISB | 1 if disease code=2; 0 otherwise | 0; 1 | 0 | Secondary endpoint |
| SEPSISC | SEPSISC | 1 = disease code 1, 11; 0 = otherwise | 0; 1 | 0 | Secondary endpoint |
| DIARRHEA | DIARRHEA | Diarrhea (1=yes; 0=no) | 0; 1 | 0 | Secondary endpoint |
| LOCINF | LOCINF | Local infection (1=yes; 0=no) | 0; 1 | 0 | Secondary endpoint |
| ABSCESS | ABSCESS | Abscess (1=yes; 0=no) | 0; 1 | 0 | Secondary endpoint |
| OMPHA | OMPHA | Omphalitis (1=yes; 0=no) | 0; 1 | 0 | Secondary endpoint |
| OTHINFECT | OTHINFECT | Other infection (1 = diarrhea, local infection, abscess or omphalitis; 0 = otherwise) | 0; 1 | 0 | Audit only |
| RASH | RASH | Rash (1=yes, 0=no) | 0; 1 | 0 | Audit only |
| JAUNDICE | JAUNDICE | Jaundice (1=yes; 0=no) | 0; 1 | 0 | Audit only |
| WHEEZ | WHEEZ | Wheezing (1=yes; 0=no) | 0; 1 | 0 | Audit only |
| DELIV_MODE | DELIV_MODE | Delivery mode | Assisted/C-section; Unassisted | 0 | Audit only |
| TX | TX | Treatment (0=placebo, 1=synbiotic) | 0; 1 | 0 | Assigned arm |
| GA | GA | Gestational age | 35-36 weeks (preterm); >= 37 weeks (term) | 0 | Baseline descriptive |
| DELIV | DELIV | Delivered | ANM center; Home; Hospital | 0 | Baseline descriptive |
| CULTURE | CULTURE | Blood culture isolates | Acinetobacter baumanii; Citrobacter freundii; E. coli; Enteroc. faecium; Kleb. pneumoniae; Staph. aureus; Strep. viridans | 4523 | Audit only |
| BFEED | BFEED | Breast feeding initiated | 1-12 hours after delivery; 13-24 hours after delivery; > 24 hours after delivery; Immediately < 1 hour after | 1 | Audit only |
| ADMIN | ADMIN | Treatment time (2=2 days or less; 3=3 or more days) | 2.0; 3.0 | 6 | Audit only |
| gram | gram | Gram stain (1=positive; 0=negative) | 0.0; 1.0 | 4523 | Audit only |
| CULTPOS | CULTPOS | Culture positive (1= gram pos or neg; 0 = otherwise) | 0; 1 | 0 | Secondary endpoint |
| GRAMNEG | GRAMNEG | Gram negative (1=yes; 0=no) | 0; 1 | 0 | Secondary endpoint |
| GRAMPOS | GRAMPOS | Gram positive (1=yes, 0=no) | 0; 1 | 0 | Secondary endpoint |
| ALLCULT | ALLCULT | Cultures (0 = gram neg; 1 = gram pos; 2 = otherwise) | 0; 1; 2 | 0 | Audit only |
| LOSTFU | LOSTFU | Lost to follow-up or withdrawn (1=yes, 0=no) | 0; 1 | 0 | Sensitivity |
| STATUS | STATUS | Status (ALIV=alive; DEAD=died; LOST=lost to follow-up; WITH=withdrawn; SEPS=sepsis) | ALIV; DEAD; LOST; SEPS; WITH | 0 | Audit only |
| POPSEP | POPSEP | Subsample | 1.0 | 221 | Audit only |
| POPDEA | POPDEA | Subsample | 1.0 | 220 | Audit only |
| POPBOTH | POPBOTH | Subsample | 1.0 | 211 | Audit only |
| POPOTH | POPOTH | Subsample | 1.0 | 230 | Audit only |
| POPDRR | POPDRR | Subsample | 1.0 | 225 | Audit only |
| POPLOC | POPLOC | Subsample | 1.0 | 229 | Audit only |
| PODRR | PODRR | Subsample |  | 4556 | Audit only |