# Alberta's Export Atlas: validation report

Completed October 6, 2026 on branch `main`. The page is ready for local visual review. No staging, commit, push, raw-source edits, or unrelated page changes were made.

1. **Existing folder:** `research/harmonized-system/3-alberta-trade-by-hs/` already contained `css/style.css`, and `data/`, `js/`, `scripts/` with `.gitkeep` files. This folder was reused; no competing asset folder was created.
2. **Selected raw file:** `ODPFN018_202512N.csv`, 57,771,032 bytes. Its actual header identifies the detailed HS6 domestic-export observations. `ODPFN016_202512N.csv` is HS8; `ODPFN020_202512N.csv` is HS2 and was used only as an independent reconciliation source. The external directory remains read-only and no raw file was copied into the repository.
3. **Relevant actual columns:** `YearMonth/AnnéeMois`, `HS6`, `Country/Pays`, `Province`, `State/État`, `Value/Valeur`, `Quantity/Quantité`, `Unit of Measure/Unité de Mesure`. Monetary amounts alone are parsed as integers; product codes remain strings.
4. **Raw HS6 rows examined:** 1,212,795 (excluding the header).
5. **Alberta 2025 source rows:** 104,986. All twelve monthly periods and all U.S. states are retained. Zero repeated complete observation keys; zero rows deduplicated.
6. **Active genuine destination markets:** 197. This includes separately reported territories/geographic destinations, not just sovereign states. Exclude `ZX` Unknown, `ZZ` High Seas, and `PC` Pacific Islands (an ambiguous legacy grouped label). All excluded codes have zero Alberta value. Retain Antarctica and separately reported Kosovo, Hong Kong, Macao and Taiwan. Country decisions, including zero-flow lookup records, appear in `scripts/build-validation.json`.
7. **Active HS4 headings:** 954.
8. **Active HS6 subheadings:** 2,645.
9. **Alberta annual domestic merchandise exports:** CAD 177,964,624,065.
10. **United States:** CAD 152,042,164,486; 85.433926% of Alberta exports.
11. **China:** CAD 10,192,667,946; 5.727356% of Alberta exports.
12. **Japan:** CAD 2,387,586,321; 1.341607% of Alberta exports.
13. **Top-ten destinations:** CAD 171,118,295,313; 96.152983% of Alberta exports.
14. **Reconciliation:** PASS, exact integer-dollar equality for valid destinations vs Alberta total; Alberta HS4 and HS6; every destination HS4 and HS6; every destination-HS4 and its HS6 children. The separate HS2 file agrees by destination. No negative values. U.S. HS4 2709 = CAD 110,821,617,128; HS4 2711 = CAD 11,573,399,330; together 80.500706% of U.S.-bound exports. Prior approximate targets serve only as checks.
15. **Missing descriptions:** five HS4 headings (`9802`, `9806`, `9807`, `9812`, `9901`); zero HS6 subheadings. Page 1 covers 949/954 active HS4 and 2,639/2,645 active HS6 codes; all unmatched codes are national/special export provisions. StatCan supplies all 2,645 HS6 labels. Leading zeros were validated for 42 active HS4 and 93 active HS6 codes. Explicit HS4 rows or common heading paths confirmed across all children are used. Exact appended tariff-rate notes were removed from heading labels (originals retained in the audit), without changing Page 1. Exact code coverage is not a certification of unchanged wording between editions.
16. **Generated data JSON:** 200 files, 668,267 bytes combined. Initial three files: 408,645 bytes; 197 lazily loaded country files: 259,622 bytes, ranging from 28 to 39,093 bytes. Each filename and exact size is listed below. Country files contain sparse `[HS6 string, integer CAD]` pairs; descriptions are stored once. Browser HS4 aggregation is also independently checked. Audit JSON: `scripts/build-validation.json`, 176,082 bytes.
17. **Files created:** 207 files under the existing Page 3 asset folder. The complete manifest below names every new file. No framework or runtime library was introduced.
18. **Files modified:** `research/harmonized-system/alberta-trade-by-hs.html`, `research/harmonized-system/harmonized-system-index.html`, `research/harmonized-system/3-alberta-trade-by-hs/css/style.css`. The collection index changes only the Page 3 entry and the sentence describing its former planned status. Collection order and the other entries are preserved.
19. **Exact local URL:** http://localhost:8000/research/harmonized-system/alberta-trade-by-hs.html . The local HTTP server is left running, bound to `127.0.0.1`. The machine's `py` launcher reports no registered Python; `python -m http.server 8000 --bind 127.0.0.1` works with Python 3.13.1. Restart from the repository root with `python -m http.server 8000`.
20. **Final git status:** shown below. The pre-existing untracked Page 2 `scripts/source-inspection.txt` is preserved. No files are staged.

## Interaction and visual verification

All five core views are linked: ranked destinations, cumulative concentration, HS4 treemap/precision list, HS6 ribbon/list, and top-four market composition. URL hashes restore country, HS4 and HS6, including back/forward navigation. Top 10/25/All and linear/log controls work. The market-composition comparison includes Top 10/20 and Exclude U.S. No optional rank strip or scatterplot was added: the concentration curve already represents every destination, and the five core views provide enough visual density.

Chrome headless verification passed for United States, China, Japan, Mexico and Netherlands; HS4 2709, 2711, 1205, 3901 and 1001; several HS6 children; partial country searches and product-description searches; Anguilla's CAD 18 flow; clear market/product; reload and back/forward restoration; autocomplete Up/Down/Enter/Escape; all/log destination controls; top-20/exclude-U.S. composition; and reduced motion. Layouts at 1440, 1024, 768 and 390 pixels passed for Japan and Alberta overall without horizontal document overflow. No runtime or HTTP errors. Screenshots were visually reviewed at desktop and mobile sizes and saved outside the repository in the OS temporary directory.

Production-function checks passed for all 197 markets and 15,354 positive destination-HS6 pairs, independent concentration/count checks, leading zeros, exact per-tile area and bounds at four aspect ratios, and WCAG AA contrast for tile/ribbon labels. Static checks passed for all 38 local HTML references, unique IDs/fragments and all 200 data JSON sizes. `git diff --check` passed.

Rerun the checks:

```powershell
python research/harmonized-system/3-alberta-trade-by-hs/scripts/build_data.py
python research/harmonized-system/3-alberta-trade-by-hs/scripts/verify_static.py
node research/harmonized-system/3-alberta-trade-by-hs/scripts/verify_app.cjs
# Optional: localhost:8000 plus isolated headless Chrome on debugging port 9223
node research/harmonized-system/3-alberta-trade-by-hs/scripts/verify_browser.cjs
```

## Data JSON filenames and exact sizes

| Relative to `data/` | Bytes |
| --- | ---: |
| `summary.json` | 41,978 |
| `product-index.json` | 319,932 |
| `alberta-products.json` | 46,735 |
| `countries/US.json` | 39,093 |
| `countries/CN.json` | 7,083 |
| `countries/JP.json` | 5,529 |
| `countries/HK.json` | 2,993 |
| `countries/KR.json` | 3,399 |
| `countries/SG.json` | 4,077 |
| `countries/MX.json` | 3,996 |
| `countries/NL.json` | 6,121 |
| `countries/PE.json` | 2,699 |
| `countries/ID.json` | 2,506 |
| `countries/AE.json` | 6,715 |
| `countries/PA.json` | 553 |
| `countries/BD.json` | 676 |
| `countries/AU.json` | 9,230 |
| `countries/IN.json` | 4,296 |
| `countries/IT.json` | 3,542 |
| `countries/VN.json` | 2,506 |
| `countries/FR.json` | 5,030 |
| `countries/TW.json` | 2,338 |
| `countries/GB.json` | 7,448 |
| `countries/CO.json` | 3,337 |
| `countries/ES.json` | 3,679 |
| `countries/BE.json` | 1,481 |
| `countries/PH.json` | 2,122 |
| `countries/DZ.json` | 886 |
| `countries/BR.json` | 3,019 |
| `countries/NG.json` | 2,042 |
| `countries/MY.json` | 3,095 |
| `countries/MA.json` | 1,149 |
| `countries/EC.json` | 1,499 |
| `countries/CL.json` | 3,240 |
| `countries/TH.json` | 2,658 |
| `countries/GT.json` | 742 |
| `countries/SE.json` | 1,610 |
| `countries/DE.json` | 5,234 |
| `countries/CU.json` | 5,338 |
| `countries/SA.json` | 5,191 |
| `countries/CH.json` | 1,812 |
| `countries/TR.json` | 2,465 |
| `countries/VE.json` | 488 |
| `countries/PK.json` | 1,419 |
| `countries/BG.json` | 548 |
| `countries/NZ.json` | 3,639 |
| `countries/GH.json` | 1,098 |
| `countries/AR.json` | 2,796 |
| `countries/NO.json` | 3,688 |
| `countries/PL.json` | 1,932 |
| `countries/KW.json` | 1,993 |
| `countries/PT.json` | 1,360 |
| `countries/OM.json` | 2,819 |
| `countries/SV.json` | 450 |
| `countries/EG.json` | 1,626 |
| `countries/KE.json` | 1,375 |
| `countries/DK.json` | 1,982 |
| `countries/ZA.json` | 1,943 |
| `countries/LK.json` | 496 |
| `countries/MV.json` | 698 |
| `countries/MZ.json` | 149 |
| `countries/CZ.json` | 908 |
| `countries/UM.json` | 1,804 |
| `countries/CR.json` | 809 |
| `countries/UA.json` | 1,942 |
| `countries/IL.json` | 1,392 |
| `countries/QA.json` | 1,624 |
| `countries/HR.json` | 965 |
| `countries/GR.json` | 982 |
| `countries/FI.json` | 939 |
| `countries/RO.json` | 1,506 |
| `countries/LY.json` | 1,190 |
| `countries/LA.json` | 52 |
| `countries/IQ.json` | 1,418 |
| `countries/DO.json` | 383 |
| `countries/IE.json` | 1,464 |
| `countries/SD.json` | 84 |
| `countries/CM.json` | 527 |
| `countries/CG.json` | 465 |
| `countries/TZ.json` | 814 |
| `countries/UZ.json` | 1,108 |
| `countries/BH.json` | 1,048 |
| `countries/GE.json` | 187 |
| `countries/KZ.json` | 2,221 |
| `countries/IS.json` | 1,136 |
| `countries/GN.json` | 184 |
| `countries/MO.json` | 309 |
| `countries/TG.json` | 115 |
| `countries/RU.json` | 48 |
| `countries/TT.json` | 2,169 |
| `countries/SK.json` | 406 |
| `countries/HU.json` | 1,025 |
| `countries/KH.json` | 172 |
| `countries/LT.json` | 507 |
| `countries/LB.json` | 523 |
| `countries/NP.json` | 457 |
| `countries/LV.json` | 292 |
| `countries/AO.json` | 966 |
| `countries/TM.json` | 677 |
| `countries/UG.json` | 581 |
| `countries/GA.json` | 698 |
| `countries/BO.json` | 1,960 |
| `countries/CI.json` | 451 |
| `countries/AT.json` | 1,142 |
| `countries/ZM.json` | 405 |
| `countries/BS.json` | 715 |
| `countries/ZW.json` | 33 |
| `countries/JO.json` | 349 |
| `countries/MT.json` | 1,026 |
| `countries/PG.json` | 866 |
| `countries/GQ.json` | 748 |
| `countries/SI.json` | 148 |
| `countries/JM.json` | 683 |
| `countries/AL.json` | 574 |
| `countries/SR.json` | 1,051 |
| `countries/CW.json` | 486 |
| `countries/KG.json` | 542 |
| `countries/MN.json` | 637 |
| `countries/UY.json` | 266 |
| `countries/BW.json` | 114 |
| `countries/GY.json` | 544 |
| `countries/BJ.json` | 136 |
| `countries/LU.json` | 513 |
| `countries/CY.json` | 162 |
| `countries/GL.json` | 290 |
| `countries/RW.json` | 308 |
| `countries/RS.json` | 284 |
| `countries/ET.json` | 440 |
| `countries/AZ.json` | 875 |
| `countries/HN.json` | 228 |
| `countries/IR.json` | 50 |
| `countries/SX.json` | 78 |
| `countries/BB.json` | 436 |
| `countries/MM.json` | 171 |
| `countries/SC.json` | 176 |
| `countries/SN.json` | 306 |
| `countries/GM.json` | 236 |
| `countries/TN.json` | 322 |
| `countries/NA.json` | 194 |
| `countries/BN.json` | 370 |
| `countries/TD.json` | 333 |
| `countries/BL.json` | 32 |
| `countries/SL.json` | 178 |
| `countries/MW.json` | 66 |
| `countries/ML.json` | 161 |
| `countries/BF.json` | 113 |
| `countries/KY.json` | 174 |
| `countries/PY.json` | 258 |
| `countries/BZ.json` | 498 |
| `countries/DJ.json` | 81 |
| `countries/EE.json` | 252 |
| `countries/GU.json` | 132 |
| `countries/AQ.json` | 141 |
| `countries/CD.json` | 213 |
| `countries/HT.json` | 49 |
| `countries/ER.json` | 49 |
| `countries/SY.json` | 49 |
| `countries/TC.json` | 375 |
| `countries/MG.json` | 176 |
| `countries/BI.json` | 81 |
| `countries/MD.json` | 108 |
| `countries/VC.json` | 129 |
| `countries/MU.json` | 49 |
| `countries/NI.json` | 145 |
| `countries/DM.json` | 32 |
| `countries/TJ.json` | 64 |
| `countries/YE.json` | 177 |
| `countries/LR.json` | 191 |
| `countries/MK.json` | 64 |
| `countries/FJ.json` | 139 |
| `countries/BA.json` | 64 |
| `countries/WS.json` | 48 |
| `countries/KP.json` | 47 |
| `countries/AM.json` | 48 |
| `countries/BM.json` | 175 |
| `countries/FO.json` | 48 |
| `countries/MR.json` | 96 |
| `countries/CV.json` | 48 |
| `countries/ME.json` | 48 |
| `countries/CX.json` | 31 |
| `countries/PF.json` | 63 |
| `countries/GI.json` | 47 |
| `countries/SS.json` | 48 |
| `countries/AW.json` | 91 |
| `countries/SO.json` | 47 |
| `countries/KN.json` | 47 |
| `countries/GD.json` | 62 |
| `countries/KI.json` | 31 |
| `countries/VG.json` | 44 |
| `countries/AG.json` | 178 |
| `countries/NE.json` | 31 |
| `countries/SB.json` | 31 |
| `countries/GW.json` | 30 |
| `countries/BQ.json` | 30 |
| `countries/CF.json` | 46 |
| `countries/LC.json` | 30 |
| `countries/VU.json` | 45 |
| `countries/SH.json` | 30 |
| `countries/PM.json` | 161 |
| `countries/AI.json` | 28 |

## Complete new-file manifest

```text
research/harmonized-system/3-alberta-trade-by-hs/data/alberta-products.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/AE.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/AG.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/AI.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/AL.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/AM.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/AO.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/AQ.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/AR.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/AT.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/AU.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/AW.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/AZ.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/BA.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/BB.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/BD.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/BE.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/BF.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/BG.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/BH.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/BI.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/BJ.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/BL.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/BM.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/BN.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/BO.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/BQ.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/BR.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/BS.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/BW.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/BZ.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/CD.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/CF.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/CG.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/CH.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/CI.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/CL.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/CM.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/CN.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/CO.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/CR.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/CU.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/CV.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/CW.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/CX.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/CY.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/CZ.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/DE.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/DJ.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/DK.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/DM.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/DO.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/DZ.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/EC.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/EE.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/EG.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/ER.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/ES.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/ET.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/FI.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/FJ.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/FO.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/FR.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/GA.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/GB.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/GD.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/GE.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/GH.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/GI.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/GL.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/GM.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/GN.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/GQ.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/GR.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/GT.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/GU.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/GW.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/GY.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/HK.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/HN.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/HR.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/HT.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/HU.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/ID.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/IE.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/IL.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/IN.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/IQ.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/IR.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/IS.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/IT.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/JM.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/JO.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/JP.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/KE.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/KG.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/KH.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/KI.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/KN.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/KP.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/KR.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/KW.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/KY.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/KZ.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/LA.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/LB.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/LC.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/LK.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/LR.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/LT.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/LU.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/LV.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/LY.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/MA.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/MD.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/ME.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/MG.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/MK.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/ML.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/MM.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/MN.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/MO.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/MR.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/MT.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/MU.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/MV.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/MW.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/MX.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/MY.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/MZ.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/NA.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/NE.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/NG.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/NI.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/NL.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/NO.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/NP.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/NZ.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/OM.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/PA.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/PE.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/PF.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/PG.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/PH.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/PK.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/PL.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/PM.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/PT.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/PY.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/QA.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/RO.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/RS.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/RU.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/RW.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/SA.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/SB.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/SC.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/SD.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/SE.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/SG.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/SH.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/SI.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/SK.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/SL.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/SN.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/SO.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/SR.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/SS.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/SV.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/SX.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/SY.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/TC.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/TD.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/TG.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/TH.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/TJ.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/TM.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/TN.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/TR.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/TT.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/TW.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/TZ.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/UA.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/UG.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/UM.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/US.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/UY.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/UZ.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/VC.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/VE.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/VG.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/VN.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/VU.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/WS.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/YE.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/ZA.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/ZM.json
research/harmonized-system/3-alberta-trade-by-hs/data/countries/ZW.json
research/harmonized-system/3-alberta-trade-by-hs/data/product-index.json
research/harmonized-system/3-alberta-trade-by-hs/data/summary.json
research/harmonized-system/3-alberta-trade-by-hs/js/app.js
research/harmonized-system/3-alberta-trade-by-hs/scripts/build-validation.json
research/harmonized-system/3-alberta-trade-by-hs/scripts/build_data.py
research/harmonized-system/3-alberta-trade-by-hs/scripts/verify_app.cjs
research/harmonized-system/3-alberta-trade-by-hs/scripts/verify_browser.cjs
research/harmonized-system/3-alberta-trade-by-hs/scripts/verify_static.py
research/harmonized-system/3-alberta-trade-by-hs/validation-report.md
```

## Git status

```text
 M research/harmonized-system/3-alberta-trade-by-hs/css/style.css
 M research/harmonized-system/alberta-trade-by-hs.html
 M research/harmonized-system/harmonized-system-index.html
?? research/harmonized-system/2-section-338-hs4-hs6-exposure-canada/scripts/source-inspection.txt
?? research/harmonized-system/3-alberta-trade-by-hs/data/alberta-products.json
?? research/harmonized-system/3-alberta-trade-by-hs/data/countries/
?? research/harmonized-system/3-alberta-trade-by-hs/data/product-index.json
?? research/harmonized-system/3-alberta-trade-by-hs/data/summary.json
?? research/harmonized-system/3-alberta-trade-by-hs/js/app.js
?? research/harmonized-system/3-alberta-trade-by-hs/scripts/build-validation.json
?? research/harmonized-system/3-alberta-trade-by-hs/scripts/build_data.py
?? research/harmonized-system/3-alberta-trade-by-hs/scripts/verify_app.cjs
?? research/harmonized-system/3-alberta-trade-by-hs/scripts/verify_browser.cjs
?? research/harmonized-system/3-alberta-trade-by-hs/scripts/verify_static.py
?? research/harmonized-system/3-alberta-trade-by-hs/validation-report.md
```
