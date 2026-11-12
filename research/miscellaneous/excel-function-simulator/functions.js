((global) => {
  "use strict";

  const catalog = [
    {
      id: "sum",
      name: "SUM",
      category: "Basics",
      syntax: "SUM(number1, [number2], ...)",
      shortDescription: "Adds numbers or values in a range.",
      difficulty: "Beginner",
      exampleFormula: "=SUM(D2:D7)",
      exampleResult: 450000,
      arguments: [
        { name: "number1", description: "First number or range to add." },
        { name: "number2", description: "Optional additional number or range." }
      ]
    },
    {
      id: "average",
      name: "AVERAGE",
      category: "Basics",
      syntax: "AVERAGE(number1, [number2], ...)",
      shortDescription: "Returns the arithmetic mean of numeric values.",
      difficulty: "Beginner",
      exampleFormula: "=AVERAGE(D2:D7)",
      exampleResult: 75000,
      arguments: [
        { name: "number1", description: "First number or range to average." },
        { name: "number2", description: "Optional additional number or range." }
      ]
    },
    {
      id: "min",
      name: "MIN",
      category: "Basics",
      syntax: "MIN(number1, [number2], ...)",
      shortDescription: "Returns the smallest numeric value.",
      difficulty: "Beginner",
      exampleFormula: "=MIN(D2:D7)",
      exampleResult: 64000,
      arguments: [
        { name: "number1", description: "First number or range to inspect." },
        { name: "number2", description: "Optional additional number or range." }
      ]
    },
    {
      id: "max",
      name: "MAX",
      category: "Basics",
      syntax: "MAX(number1, [number2], ...)",
      shortDescription: "Returns the largest numeric value.",
      difficulty: "Beginner",
      exampleFormula: "=MAX(D2:D7)",
      exampleResult: 89000,
      arguments: [
        { name: "number1", description: "First number or range to inspect." },
        { name: "number2", description: "Optional additional number or range." }
      ]
    },
    {
      id: "count",
      name: "COUNT",
      category: "Basics",
      syntax: "COUNT(value1, [value2], ...)",
      shortDescription: "Counts cells containing numeric values.",
      difficulty: "Beginner",
      exampleFormula: "=COUNT(A2:A7)",
      exampleResult: 6,
      arguments: [
        { name: "value1", description: "First value or range to count." },
        { name: "value2", description: "Optional additional value or range." }
      ]
    },
    {
      id: "if",
      name: "IF",
      category: "Logic",
      syntax: "IF(logical_test, value_if_true, value_if_false)",
      shortDescription: "Returns one value when a test is true and another when false.",
      difficulty: "Beginner",
      exampleFormula: "=IF(D2>=75000,\"At target\",\"Below target\")",
      exampleResult: "Below target",
      arguments: [
        { name: "logical_test", description: "Condition to evaluate." },
        { name: "value_if_true", description: "Value returned when the test is true." },
        { name: "value_if_false", description: "Value returned when the test is false." }
      ]
    },
    {
      id: "and",
      name: "AND",
      category: "Logic",
      syntax: "AND(logical1, [logical2], ...)",
      shortDescription: "Returns TRUE only when every test is true.",
      difficulty: "Beginner",
      exampleFormula: "=AND(C3=\"IT\",E3>=5)",
      exampleResult: true,
      arguments: [
        { name: "logical1", description: "First condition to evaluate." },
        { name: "logical2", description: "Optional additional condition." }
      ]
    },
    {
      id: "or",
      name: "OR",
      category: "Logic",
      syntax: "OR(logical1, [logical2], ...)",
      shortDescription: "Returns TRUE when at least one test is true.",
      difficulty: "Beginner",
      exampleFormula: "=OR(C5=\"Finance\",C5=\"HR\")",
      exampleResult: true,
      arguments: [
        { name: "logical1", description: "First condition to evaluate." },
        { name: "logical2", description: "Optional additional condition." }
      ]
    },
    {
      id: "not",
      name: "NOT",
      category: "Logic",
      syntax: "NOT(logical)",
      shortDescription: "Reverses TRUE to FALSE or FALSE to TRUE.",
      difficulty: "Beginner",
      exampleFormula: "=NOT(C2=\"IT\")",
      exampleResult: true,
      arguments: [
        { name: "logical", description: "Condition or logical value to reverse." }
      ]
    },
    {
      id: "countif",
      name: "COUNTIF",
      category: "Conditional",
      syntax: "COUNTIF(range, criteria)",
      shortDescription: "Counts cells that meet one condition.",
      difficulty: "Intermediate",
      exampleFormula: "=COUNTIF(C2:C7,\"Finance\")",
      exampleResult: 3,
      arguments: [
        { name: "range", description: "Cells to test." },
        { name: "criteria", description: "Condition cells must meet." }
      ]
    },
    {
      id: "sumif",
      name: "SUMIF",
      category: "Conditional",
      syntax: "SUMIF(range, criteria, [sum_range])",
      shortDescription: "Adds values whose aligned cells meet one condition.",
      difficulty: "Intermediate",
      exampleFormula: "=SUMIF(C2:C7,\"IT\",D2:D7)",
      exampleResult: 170000,
      arguments: [
        { name: "range", description: "Cells to test." },
        { name: "criteria", description: "Condition cells must meet." },
        { name: "sum_range", description: "Optional aligned cells to add." }
      ]
    },
    {
      id: "averageif",
      name: "AVERAGEIF",
      category: "Conditional",
      syntax: "AVERAGEIF(range, criteria, [average_range])",
      shortDescription: "Averages values whose aligned cells meet one condition.",
      difficulty: "Intermediate",
      exampleFormula: "=AVERAGEIF(C2:C7,\"Finance\",D2:D7)",
      exampleResult: 72000,
      arguments: [
        { name: "range", description: "Cells to test." },
        { name: "criteria", description: "Condition cells must meet." },
        { name: "average_range", description: "Optional aligned cells to average." }
      ]
    },
    {
      id: "countifs",
      name: "COUNTIFS",
      category: "Conditional",
      syntax: "COUNTIFS(criteria_range1, criteria1, [criteria_range2, criteria2], ...)",
      shortDescription: "Counts rows where all conditions are met.",
      difficulty: "Intermediate",
      exampleFormula: "=COUNTIFS(C2:C7,\"Finance\",E2:E7,\">=5\")",
      exampleResult: 1,
      arguments: [
        { name: "criteria_range1", description: "First range to test." },
        { name: "criteria1", description: "Condition for the first range." },
        { name: "criteria_range2, criteria2", description: "Optional additional range and condition pairs." }
      ]
    },
    {
      id: "sumifs",
      name: "SUMIFS",
      category: "Conditional",
      syntax: "SUMIFS(sum_range, criteria_range1, criteria1, ...)",
      shortDescription: "Adds values where all conditions are met.",
      difficulty: "Advanced",
      exampleFormula: "=SUMIFS(D2:D7,C2:C7,\"Finance\",E2:E7,\">=4\")",
      exampleResult: 148000,
      arguments: [
        { name: "sum_range", description: "Cells to add." },
        { name: "criteria_range1", description: "First range to test." },
        { name: "criteria1", description: "Condition for the first range." },
        { name: "criteria_range2, criteria2", description: "Optional additional range and condition pairs." }
      ]
    },
    {
      id: "averageifs",
      name: "AVERAGEIFS",
      category: "Conditional",
      syntax: "AVERAGEIFS(average_range, criteria_range1, criteria1, ...)",
      shortDescription: "Averages values where all conditions are met.",
      difficulty: "Advanced",
      exampleFormula: "=AVERAGEIFS(D2:D7,C2:C7,\"IT\",E2:E7,\">=6\")",
      exampleResult: 85000,
      arguments: [
        { name: "average_range", description: "Cells to average." },
        { name: "criteria_range1", description: "First range to test." },
        { name: "criteria1", description: "Condition for the first range." },
        { name: "criteria_range2, criteria2", description: "Optional additional range and condition pairs." }
      ]
    },
    {
      id: "vlookup",
      name: "VLOOKUP",
      category: "Lookup",
      syntax: "VLOOKUP(lookup_value, table_array, col_index_num, [range_lookup])",
      shortDescription: "Searches the first table column and returns a value from the matched row.",
      difficulty: "Intermediate",
      exampleFormula: "=VLOOKUP(1005,A2:E7,2,FALSE)",
      exampleResult: "Emma",
      arguments: [
        { name: "lookup_value", description: "Value to find in the first column." },
        { name: "table_array", description: "Table containing lookup and return columns." },
        { name: "col_index_num", description: "Return column position within the table." },
        { name: "range_lookup", description: "FALSE for exact or TRUE for approximate match." }
      ]
    },
    {
      id: "hlookup",
      name: "HLOOKUP",
      category: "Lookup",
      syntax: "HLOOKUP(lookup_value, table_array, row_index_num, [range_lookup])",
      shortDescription: "Searches the first table row and returns a value from the matched column.",
      difficulty: "Intermediate",
      exampleFormula: "=HLOOKUP(1004,H7:M9,2,FALSE)",
      exampleResult: "Noah",
      arguments: [
        { name: "lookup_value", description: "Value to find in the first row." },
        { name: "table_array", description: "Table containing lookup and return rows." },
        { name: "row_index_num", description: "Return row position within the table." },
        { name: "range_lookup", description: "FALSE for exact or TRUE for approximate match." }
      ],
      exampleSetup: [
        { cell: "H7", value: 1001 },
        { cell: "I7", value: 1002 },
        { cell: "J7", value: 1003 },
        { cell: "K7", value: 1004 },
        { cell: "L7", value: 1005 },
        { cell: "M7", value: 1006 },
        { cell: "H8", value: "Maya" },
        { cell: "I8", value: "Liam" },
        { cell: "J8", value: "Sofia" },
        { cell: "K8", value: "Noah" },
        { cell: "L8", value: "Emma" },
        { cell: "M8", value: "Lucas" },
        { cell: "H9", value: 72000 },
        { cell: "I9", value: 81000 },
        { cell: "J9", value: 68000 },
        { cell: "K9", value: 64000 },
        { cell: "L9", value: 89000 },
        { cell: "M9", value: 76000 }
      ]
    },
    {
      id: "xlookup",
      name: "XLOOKUP",
      category: "Lookup",
      syntax: "XLOOKUP(lookup_value, lookup_array, return_array, [if_not_found], [match_mode], [search_mode])",
      shortDescription: "Finds a value in one range and returns its aligned value from another.",
      difficulty: "Intermediate",
      exampleFormula: "=XLOOKUP(\"Lucas\",B2:B7,D2:D7)",
      exampleResult: 76000,
      arguments: [
        { name: "lookup_value", description: "Value to find." },
        { name: "lookup_array", description: "Single row or column to search." },
        { name: "return_array", description: "Aligned row or column to return from." },
        { name: "if_not_found", description: "Optional fallback when no match exists." },
        { name: "match_mode", description: "Optional exact, approximate, or wildcard mode." },
        { name: "search_mode", description: "Optional forward or reverse search direction." }
      ]
    },
    {
      id: "match",
      name: "MATCH",
      category: "Lookup",
      syntax: "MATCH(lookup_value, lookup_array, [match_type])",
      shortDescription: "Returns a value's relative position in a row or column.",
      difficulty: "Intermediate",
      exampleFormula: "=MATCH(\"Noah\",B2:B7,0)",
      exampleResult: 4,
      arguments: [
        { name: "lookup_value", description: "Value whose position is needed." },
        { name: "lookup_array", description: "Single row or column to search." },
        { name: "match_type", description: "0 for exact or 1/-1 for approximate match." }
      ]
    },
    {
      id: "index",
      name: "INDEX",
      category: "Lookup",
      syntax: "INDEX(array, row_num, [column_num])",
      shortDescription: "Returns the value at a relative row and column in a range.",
      difficulty: "Intermediate",
      exampleFormula: "=INDEX(B2:B7,5)",
      exampleResult: "Emma",
      arguments: [
        { name: "array", description: "Range containing the return value." },
        { name: "row_num", description: "Relative row position in the range." },
        { name: "column_num", description: "Optional relative column position." }
      ]
    },
    {
      id: "index-match",
      name: "INDEX + MATCH",
      category: "Lookup",
      syntax: "INDEX(return_range, MATCH(lookup_value, lookup_range, 0))",
      shortDescription: "Uses MATCH to locate a row and INDEX to return its aligned value.",
      difficulty: "Advanced",
      exampleFormula: "=INDEX(D2:D7,MATCH(\"Lucas\",B2:B7,0))",
      exampleResult: 76000,
      arguments: [
        { name: "return_range", description: "Range containing the result." },
        { name: "lookup_value", description: "Value to find." },
        { name: "lookup_range", description: "Aligned range searched by MATCH." }
      ]
    },
    {
      id: "len",
      name: "LEN",
      category: "Text",
      syntax: "LEN(text)",
      shortDescription: "Counts the characters in text, including spaces.",
      difficulty: "Beginner",
      exampleFormula: "=LEN(B2)",
      exampleResult: 4,
      arguments: [
        { name: "text", description: "Text or value whose characters are counted." }
      ]
    },
    {
      id: "left",
      name: "LEFT",
      category: "Text",
      syntax: "LEFT(text, [num_chars])",
      shortDescription: "Returns characters from the start of text.",
      difficulty: "Beginner",
      exampleFormula: "=LEFT(B7,2)",
      exampleResult: "Lu",
      arguments: [
        { name: "text", description: "Text to read from the left." },
        { name: "num_chars", description: "Optional number of characters; defaults to 1." }
      ]
    },
    {
      id: "right",
      name: "RIGHT",
      category: "Text",
      syntax: "RIGHT(text, [num_chars])",
      shortDescription: "Returns characters from the end of text.",
      difficulty: "Beginner",
      exampleFormula: "=RIGHT(B4,2)",
      exampleResult: "ia",
      arguments: [
        { name: "text", description: "Text to read from the right." },
        { name: "num_chars", description: "Optional number of characters; defaults to 1." }
      ]
    },
    {
      id: "mid",
      name: "MID",
      category: "Text",
      syntax: "MID(text, start_num, num_chars)",
      shortDescription: "Returns characters from a specified position in text.",
      difficulty: "Beginner",
      exampleFormula: "=MID(\"Vancouver\",4,3)",
      exampleResult: "cou",
      arguments: [
        { name: "text", description: "Text containing the characters to return." },
        { name: "start_num", description: "One-based position of the first character." },
        { name: "num_chars", description: "Number of characters to return." }
      ]
    },
    {
      id: "trim",
      name: "TRIM",
      category: "Text",
      syntax: "TRIM(text)",
      shortDescription: "Removes outer spaces and reduces repeated interior spaces.",
      difficulty: "Beginner",
      exampleFormula: "=TRIM(\"   Maya    Chen   \")",
      exampleResult: "Maya Chen",
      arguments: [
        { name: "text", description: "Text whose ordinary spaces should be cleaned." }
      ]
    },
    {
      id: "upper",
      name: "UPPER",
      category: "Text",
      syntax: "UPPER(text)",
      shortDescription: "Converts text to uppercase.",
      difficulty: "Beginner",
      exampleFormula: "=UPPER(C2)",
      exampleResult: "FINANCE",
      arguments: [
        { name: "text", description: "Text to convert to uppercase." }
      ]
    },
    {
      id: "lower",
      name: "LOWER",
      category: "Text",
      syntax: "LOWER(text)",
      shortDescription: "Converts text to lowercase.",
      difficulty: "Beginner",
      exampleFormula: "=LOWER(C2)",
      exampleResult: "finance",
      arguments: [
        { name: "text", description: "Text to convert to lowercase." }
      ]
    },
    {
      id: "proper",
      name: "PROPER",
      category: "Text",
      syntax: "PROPER(text)",
      shortDescription: "Capitalizes the first letter of each word.",
      difficulty: "Beginner",
      exampleFormula: "=PROPER(\"maya chen\")",
      exampleResult: "Maya Chen",
      arguments: [
        { name: "text", description: "Text to convert to title-style capitalization." }
      ]
    },
    {
      id: "concat",
      name: "CONCAT",
      category: "Text",
      syntax: "CONCAT(text1, [text2], ...)",
      shortDescription: "Combines text values and ranges without a delimiter.",
      difficulty: "Beginner",
      exampleFormula: "=CONCAT(B2,\" - \",C2)",
      exampleResult: "Maya - Finance",
      arguments: [
        { name: "text1", description: "First text, value, or range to combine." },
        { name: "text2", description: "Optional additional text, value, or range." }
      ]
    },
    {
      id: "textjoin",
      name: "TEXTJOIN",
      category: "Text",
      syntax: "TEXTJOIN(delimiter, ignore_empty, text1, [text2], ...)",
      shortDescription: "Joins text values with a delimiter and optional blank filtering.",
      difficulty: "Intermediate",
      exampleFormula: "=TEXTJOIN(\", \",TRUE,B2:B4)",
      exampleResult: "Maya, Liam, Sofia",
      arguments: [
        { name: "delimiter", description: "Text inserted between included values." },
        { name: "ignore_empty", description: "TRUE skips blank values; FALSE includes them." },
        { name: "text1", description: "First text, value, or range to join." },
        { name: "text2", description: "Optional additional text, value, or range." }
      ]
    },
    {
      id: "find",
      name: "FIND",
      category: "Text",
      syntax: "FIND(find_text, within_text, [start_num])",
      shortDescription: "Finds a case-sensitive text position.",
      difficulty: "Intermediate",
      exampleFormula: "=FIND(\"c\",\"Vancouver\")",
      exampleResult: 4,
      arguments: [
        { name: "find_text", description: "Case-sensitive text to locate." },
        { name: "within_text", description: "Text to search." },
        { name: "start_num", description: "Optional one-based position at which to start." }
      ]
    },
    {
      id: "search",
      name: "SEARCH",
      category: "Text",
      syntax: "SEARCH(find_text, within_text, [start_num])",
      shortDescription: "Finds a case-insensitive text position.",
      difficulty: "Intermediate",
      exampleFormula: "=SEARCH(\"C\",\"Vancouver\")",
      exampleResult: 4,
      arguments: [
        { name: "find_text", description: "Text to locate without case sensitivity." },
        { name: "within_text", description: "Text to search." },
        { name: "start_num", description: "Optional one-based position at which to start." }
      ]
    },
    {
      id: "substitute",
      name: "SUBSTITUTE",
      category: "Text",
      syntax: "SUBSTITUTE(text, old_text, new_text, [instance_num])",
      shortDescription: "Replaces matching text, optionally at one occurrence.",
      difficulty: "Intermediate",
      exampleFormula: "=SUBSTITUTE(\"Finance Department\",\"Department\",\"Division\")",
      exampleResult: "Finance Division",
      arguments: [
        { name: "text", description: "Original text." },
        { name: "old_text", description: "Text to find." },
        { name: "new_text", description: "Text that replaces each selected match." },
        { name: "instance_num", description: "Optional positive occurrence number to replace." }
      ]
    },
    {
      id: "replace",
      name: "REPLACE",
      category: "Text",
      syntax: "REPLACE(old_text, start_num, num_chars, new_text)",
      shortDescription: "Replaces characters at a specified text position.",
      difficulty: "Intermediate",
      exampleFormula: "=REPLACE(\"2025-report\",1,4,\"2026\")",
      exampleResult: "2026-report",
      arguments: [
        { name: "old_text", description: "Original text." },
        { name: "start_num", description: "One-based position where replacement begins." },
        { name: "num_chars", description: "Number of original characters to replace." },
        { name: "new_text", description: "Replacement text." }
      ]
    },
    {
      id: "date",
      name: "DATE",
      category: "Date",
      syntax: "DATE(year, month, day)",
      shortDescription: "Builds a numeric date serial from year, month, and day values.",
      difficulty: "Beginner",
      exampleFormula: "=DATE(2026,8,17)",
      exampleResult: 46251,
      exampleNumberFormat: "Date",
      lessonNote: "Excel stores dates as serial numbers. A Date number format turns the numeric value into a calendar date, and adding 1 moves forward one day.",
      exampleSetup: [
        { cell: "H2", value: "=DATE(2026,8,17)" },
        { cell: "H3", value: "=H2+1" }
      ],
      arguments: [
        { name: "year", description: "Year of the date to construct." },
        { name: "month", description: "Month number; values outside 1-12 normalize across years." },
        { name: "day", description: "Day number; values outside the month normalize across dates." }
      ]
    },
    {
      id: "year",
      name: "YEAR",
      category: "Date",
      syntax: "YEAR(serial_number)",
      shortDescription: "Returns the year from a numeric date serial.",
      difficulty: "Beginner",
      exampleFormula: "=YEAR(DATE(2026,8,17))",
      exampleResult: 2026,
      arguments: [
        { name: "serial_number", description: "Numeric date serial or reference to a date-formatted cell." }
      ]
    },
    {
      id: "month",
      name: "MONTH",
      category: "Date",
      syntax: "MONTH(serial_number)",
      shortDescription: "Returns the month number from a numeric date serial.",
      difficulty: "Beginner",
      exampleFormula: "=MONTH(DATE(2026,8,17))",
      exampleResult: 8,
      arguments: [
        { name: "serial_number", description: "Numeric date serial or reference to a date-formatted cell." }
      ]
    },
    {
      id: "day",
      name: "DAY",
      category: "Date",
      syntax: "DAY(serial_number)",
      shortDescription: "Returns the day of the month from a numeric date serial.",
      difficulty: "Beginner",
      exampleFormula: "=DAY(DATE(2026,8,17))",
      exampleResult: 17,
      arguments: [
        { name: "serial_number", description: "Numeric date serial or reference to a date-formatted cell." }
      ]
    },
    {
      id: "today",
      name: "TODAY",
      category: "Date",
      syntax: "TODAY()",
      shortDescription: "Returns the current local calendar date as a numeric date serial.",
      difficulty: "Beginner",
      exampleFormula: "=TODAY()",
      exampleResult: "Current date (recalculates)",
      lessonNote: "TODAY is volatile: it is evaluated again during each full worksheet recalculation.",
      arguments: []
    },
    {
      id: "days",
      name: "DAYS",
      category: "Date",
      syntax: "DAYS(end_date, start_date)",
      shortDescription: "Returns the numeric day difference between two dates.",
      difficulty: "Beginner",
      exampleFormula: "=DAYS(DATE(2026,8,17),DATE(2026,8,10))",
      exampleResult: 7,
      arguments: [
        { name: "end_date", description: "Later or ending date serial." },
        { name: "start_date", description: "Earlier or starting date serial." }
      ]
    },
    {
      id: "edate",
      name: "EDATE",
      category: "Date",
      syntax: "EDATE(start_date, months)",
      shortDescription: "Moves a date by a whole number of months.",
      difficulty: "Intermediate",
      exampleFormula: "=EDATE(DATE(2026,8,17),3)",
      exampleResult: 46343,
      exampleNumberFormat: "Date",
      arguments: [
        { name: "start_date", description: "Date serial to move from." },
        { name: "months", description: "Whole months forward when positive or backward when negative." }
      ]
    },
    {
      id: "eomonth",
      name: "EOMONTH",
      category: "Date",
      syntax: "EOMONTH(start_date, months)",
      shortDescription: "Returns the last day of a month at a chosen offset.",
      difficulty: "Intermediate",
      exampleFormula: "=EOMONTH(DATE(2026,8,17),0)",
      exampleResult: 46265,
      exampleNumberFormat: "Date",
      arguments: [
        { name: "start_date", description: "Date serial whose month anchors the calculation." },
        { name: "months", description: "Whole-month offset from the start date's month." }
      ]
    },
    {
      id: "weekday",
      name: "WEEKDAY",
      category: "Date",
      syntax: "WEEKDAY(serial_number, [return_type])",
      shortDescription: "Returns a weekday number using Sunday- or Monday-based numbering.",
      difficulty: "Intermediate",
      exampleFormula: "=WEEKDAY(DATE(2026,8,17),2)",
      exampleResult: 1,
      arguments: [
        { name: "serial_number", description: "Numeric date serial to classify." },
        { name: "return_type", description: "Optional 1 for Sunday=1 or 2 for Monday=1." }
      ]
    },
    {
      id: "round",
      name: "ROUND",
      category: "Math",
      syntax: "ROUND(number, num_digits)",
      shortDescription: "Rounds a number to a chosen decimal position.",
      difficulty: "Beginner",
      exampleFormula: "=ROUND(12.3456,2)",
      exampleResult: 12.35,
      arguments: [
        { name: "number", description: "Number to round." },
        { name: "num_digits", description: "Decimal places; negative values round left of the decimal point." }
      ]
    },
    {
      id: "roundup",
      name: "ROUNDUP",
      category: "Math",
      syntax: "ROUNDUP(number, num_digits)",
      shortDescription: "Rounds a number away from zero.",
      difficulty: "Beginner",
      exampleFormula: "=ROUNDUP(12.341,2)",
      exampleResult: 12.35,
      arguments: [
        { name: "number", description: "Number to round away from zero." },
        { name: "num_digits", description: "Decimal position at which to round." }
      ]
    },
    {
      id: "rounddown",
      name: "ROUNDDOWN",
      category: "Math",
      syntax: "ROUNDDOWN(number, num_digits)",
      shortDescription: "Rounds a number toward zero.",
      difficulty: "Beginner",
      exampleFormula: "=ROUNDDOWN(12.349,2)",
      exampleResult: 12.34,
      arguments: [
        { name: "number", description: "Number to round toward zero." },
        { name: "num_digits", description: "Decimal position at which to round." }
      ]
    },
    {
      id: "int",
      name: "INT",
      category: "Math",
      syntax: "INT(number)",
      shortDescription: "Rounds down to the nearest integer.",
      difficulty: "Beginner",
      exampleFormula: "=INT(-8.9)",
      exampleResult: -9,
      arguments: [
        { name: "number", description: "Number to round downward." }
      ]
    },
    {
      id: "abs",
      name: "ABS",
      category: "Math",
      syntax: "ABS(number)",
      shortDescription: "Returns a number's absolute value.",
      difficulty: "Beginner",
      exampleFormula: "=ABS(-25)",
      exampleResult: 25,
      arguments: [
        { name: "number", description: "Number whose distance from zero is needed." }
      ]
    },
    {
      id: "mod",
      name: "MOD",
      category: "Math",
      syntax: "MOD(number, divisor)",
      shortDescription: "Returns the remainder using Excel-style divisor sign behavior.",
      difficulty: "Beginner",
      exampleFormula: "=MOD(25,7)",
      exampleResult: 4,
      arguments: [
        { name: "number", description: "Number to divide." },
        { name: "divisor", description: "Number to divide by." }
      ]
    },
    {
      id: "iferror",
      name: "IFERROR",
      category: "Error Handling",
      syntax: "IFERROR(value, value_if_error)",
      shortDescription: "Returns a fallback when an expression produces any spreadsheet error.",
      difficulty: "Intermediate",
      exampleFormula: "=IFERROR(10/0,\"Cannot divide\")",
      exampleResult: "Cannot divide",
      arguments: [
        { name: "value", description: "Primary expression to evaluate." },
        { name: "value_if_error", description: "Fallback evaluated only when an error occurs." }
      ]
    },
    {
      id: "ifna",
      name: "IFNA",
      category: "Error Handling",
      syntax: "IFNA(value, value_if_na)",
      shortDescription: "Returns a fallback only when an expression produces #N/A.",
      difficulty: "Intermediate",
      exampleFormula: "=IFNA(XLOOKUP(\"Olivia\",B2:B7,D2:D7),\"Not found\")",
      exampleResult: "Not found",
      arguments: [
        { name: "value", description: "Primary expression to evaluate." },
        { name: "value_if_na", description: "Fallback evaluated only for #N/A." }
      ]
    },
    {
      id: "median",
      name: "MEDIAN",
      category: "Statistics",
      syntax: "MEDIAN(number1, [number2], ...)",
      shortDescription: "Returns the middle value after numeric observations are ordered.",
      difficulty: "Beginner",
      exampleFormula: "=MEDIAN(D2:D7)",
      exampleResult: 74000,
      arguments: [
        { name: "number1", description: "First number or range containing numeric observations." },
        { name: "number2", description: "Optional additional number or range." }
      ]
    },
    {
      id: "mode-sngl",
      name: "MODE.SNGL",
      category: "Statistics",
      syntax: "MODE.SNGL(number1, [number2], ...)",
      shortDescription: "Returns the most frequently occurring numeric value.",
      difficulty: "Beginner",
      exampleFormula: "=MODE.SNGL(E2:E7)",
      exampleResult: 5,
      arguments: [
        { name: "number1", description: "First number or range to inspect for repeated values." },
        { name: "number2", description: "Optional additional number or range." }
      ]
    },
    {
      id: "stdev-s",
      name: "STDEV.S",
      category: "Statistics",
      syntax: "STDEV.S(number1, [number2], ...)",
      shortDescription: "Estimates standard deviation for a sample using n − 1.",
      difficulty: "Intermediate",
      exampleFormula: "=STDEV.S(D2:D7)",
      exampleResult: 9077.44457432817,
      arguments: [
        { name: "number1", description: "First numeric sample or range." },
        { name: "number2", description: "Optional additional numeric sample or range." }
      ]
    },
    {
      id: "stdev-p",
      name: "STDEV.P",
      category: "Statistics",
      syntax: "STDEV.P(number1, [number2], ...)",
      shortDescription: "Calculates standard deviation for a complete population using n.",
      difficulty: "Intermediate",
      exampleFormula: "=STDEV.P(D2:D7)",
      exampleResult: 8286.53526310404,
      arguments: [
        { name: "number1", description: "First population value or range." },
        { name: "number2", description: "Optional additional population value or range." }
      ]
    },
    {
      id: "var-s",
      name: "VAR.S",
      category: "Statistics",
      syntax: "VAR.S(number1, [number2], ...)",
      shortDescription: "Estimates sample variance using n − 1.",
      difficulty: "Intermediate",
      exampleFormula: "=VAR.S(D2:D7)",
      exampleResult: 82400000,
      arguments: [
        { name: "number1", description: "First numeric sample or range." },
        { name: "number2", description: "Optional additional numeric sample or range." }
      ]
    },
    {
      id: "var-p",
      name: "VAR.P",
      category: "Statistics",
      syntax: "VAR.P(number1, [number2], ...)",
      shortDescription: "Calculates population variance using n.",
      difficulty: "Intermediate",
      exampleFormula: "=VAR.P(D2:D7)",
      exampleResult: 68666666.6666667,
      arguments: [
        { name: "number1", description: "First population value or range." },
        { name: "number2", description: "Optional additional population value or range." }
      ]
    },
    {
      id: "rank-eq",
      name: "RANK.EQ",
      category: "Statistics",
      syntax: "RANK.EQ(number, ref, [order])",
      shortDescription: "Returns a number's rank within a numeric list; tied values receive the same rank.",
      difficulty: "Intermediate",
      exampleFormula: "=RANK.EQ(D7,D2:D7,0)",
      exampleResult: 3,
      arguments: [
        { name: "number", description: "Number whose rank is required." },
        { name: "ref", description: "Numeric reference list used for ranking." },
        { name: "order", description: "Optional 0 for descending; nonzero for ascending." }
      ]
    },
    {
      id: "percentile-inc",
      name: "PERCENTILE.INC",
      category: "Statistics",
      syntax: "PERCENTILE.INC(array, k)",
      shortDescription: "Returns the inclusive k-th percentile with interpolation when necessary.",
      difficulty: "Advanced",
      exampleFormula: "=PERCENTILE.INC(D2:D7,0.75)",
      exampleResult: 79750,
      arguments: [
        { name: "array", description: "Numeric data used to calculate the percentile." },
        { name: "k", description: "Percentile from 0 through 1 inclusive." }
      ]
    },
    {
      id: "quartile-inc",
      name: "QUARTILE.INC",
      category: "Statistics",
      syntax: "QUARTILE.INC(array, quart)",
      shortDescription: "Returns an inclusive quartile from minimum (0) through maximum (4).",
      difficulty: "Intermediate",
      exampleFormula: "=QUARTILE.INC(D2:D7,1)",
      exampleResult: 69000,
      arguments: [
        { name: "array", description: "Numeric data used to calculate the quartile." },
        { name: "quart", description: "Quartile number 0, 1, 2, 3, or 4." }
      ]
    },
    {
      id: "correl",
      name: "CORREL",
      category: "Statistics",
      syntax: "CORREL(array1, array2)",
      shortDescription: "Returns Pearson correlation for two aligned numeric arrays.",
      difficulty: "Advanced",
      exampleFormula: "=CORREL(D2:D7,E2:E7)",
      exampleResult: 0.831467676330196,
      arguments: [
        { name: "array1", description: "First aligned numeric array." },
        { name: "array2", description: "Second aligned numeric array." }
      ]
    },
    {
      id: "covariance-s",
      name: "COVARIANCE.S",
      category: "Statistics",
      syntax: "COVARIANCE.S(array1, array2)",
      shortDescription: "Returns sample covariance for two aligned numeric arrays.",
      difficulty: "Advanced",
      exampleFormula: "=COVARIANCE.S(D2:D7,E2:E7)",
      exampleResult: 13000,
      arguments: [
        { name: "array1", description: "First aligned numeric sample." },
        { name: "array2", description: "Second aligned numeric sample." }
      ]
    },
    {
      id: "pv",
      name: "PV",
      category: "Financial",
      syntax: "PV(rate, nper, pmt, [fv], [type])",
      shortDescription: "Returns the present value of a stream of equal payments using a constant interest rate.",
      difficulty: "Intermediate",
      exampleFormula: "=PV(0.06/12,120,-200)",
      exampleResult: 18014.6906654335,
      exampleNumberFormat: "Currency",
      arguments: [
        { name: "rate", description: "Interest rate per payment period." },
        { name: "nper", description: "Total number of payment periods." },
        { name: "pmt", description: "Payment made each period; cash paid out is normally negative." },
        { name: "fv", description: "Optional future value after the final payment; defaults to 0." },
        { name: "type", description: "Optional 0 for end-of-period payments or 1 for beginning-of-period payments." }
      ]
    },
    {
      id: "fv",
      name: "FV",
      category: "Financial",
      syntax: "FV(rate, nper, pmt, [pv], [type])",
      shortDescription: "Returns the future value of an investment with equal periodic payments and a constant rate.",
      difficulty: "Intermediate",
      exampleFormula: "=FV(0.06/12,120,-200)",
      exampleResult: 32775.8693612916,
      exampleNumberFormat: "Currency",
      arguments: [
        { name: "rate", description: "Interest rate per payment period." },
        { name: "nper", description: "Total number of payment periods." },
        { name: "pmt", description: "Payment made each period; deposits are usually entered as negative cash flows." },
        { name: "pv", description: "Optional present value; defaults to 0." },
        { name: "type", description: "Optional 0 for end-of-period payments or 1 for beginning-of-period payments." }
      ]
    },
    {
      id: "pmt",
      name: "PMT",
      category: "Financial",
      syntax: "PMT(rate, nper, pv, [fv], [type])",
      shortDescription: "Calculates the equal periodic payment required for a loan or annuity.",
      difficulty: "Intermediate",
      exampleFormula: "=PMT(0.05/12,60,20000)",
      exampleResult: -377.42467288022,
      exampleNumberFormat: "Currency",
      arguments: [
        { name: "rate", description: "Interest rate per payment period." },
        { name: "nper", description: "Number of payments." },
        { name: "pv", description: "Present value, such as the amount borrowed." },
        { name: "fv", description: "Optional balance desired after the final payment; defaults to 0." },
        { name: "type", description: "Optional 0 for payments at period end or 1 at period beginning." }
      ]
    },
    {
      id: "npv",
      name: "NPV",
      category: "Financial",
      syntax: "NPV(rate, value1, [value2], ...)",
      shortDescription: "Discounts equally spaced future cash flows back to the present; the first supplied cash flow is one period away.",
      difficulty: "Intermediate",
      exampleFormula: "=NPV(0.1,3000,4200,6800)",
      exampleResult: 11307.2877535687,
      exampleNumberFormat: "Currency",
      arguments: [
        { name: "rate", description: "Discount rate per period." },
        { name: "value1", description: "First future cash flow, discounted one period." },
        { name: "value2", description: "Optional later cash flows at equally spaced periods." }
      ]
    },
    {
      id: "irr",
      name: "IRR",
      category: "Financial",
      syntax: "IRR(values, [guess])",
      shortDescription: "Finds the periodic rate that makes the net present value of equally spaced cash flows equal to zero.",
      difficulty: "Advanced",
      exampleFormula: "=IRR(H2:H5)",
      exampleResult: 0.163405600688989,
      exampleNumberFormat: "Percentage",
      exampleTargetCell: "H7",
      exampleSetup: [
        { cell: "H2", value: "-10000" },
        { cell: "H3", value: "3000" },
        { cell: "H4", value: "4200" },
        { cell: "H5", value: "6800" }
      ],
      arguments: [
        { name: "values", description: "Cash-flow series containing at least one negative and one positive amount." },
        { name: "guess", description: "Optional starting estimate for the rate; defaults to 10%." }
      ]
    },
    {
      id: "xnpv",
      name: "XNPV",
      category: "Financial",
      syntax: "XNPV(rate, values, dates)",
      shortDescription: "Discounts irregularly timed cash flows using their actual dates and a 365-day year.",
      difficulty: "Advanced",
      exampleFormula: "=XNPV(0.1,H2:H5,I2:I5)",
      exampleResult: 2227.43187269064,
      exampleNumberFormat: "Currency",
      exampleTargetCell: "H7",
      exampleSetup: [
        { cell: "H2", value: "-10000" },
        { cell: "H3", value: "3000" },
        { cell: "H4", value: "4200" },
        { cell: "H5", value: "6800" },
        { cell: "I2", value: "=DATE(2026,1,1)" },
        { cell: "I3", value: "=DATE(2026,7,1)" },
        { cell: "I4", value: "=DATE(2027,3,15)" },
        { cell: "I5", value: "=DATE(2028,1,1)" }
      ],
      arguments: [
        { name: "rate", description: "Annual discount rate." },
        { name: "values", description: "Cash flows including the initial amount." },
        { name: "dates", description: "Dates aligned with the cash flows; no date may precede the first date." }
      ]
    },
    {
      id: "xirr",
      name: "XIRR",
      category: "Financial",
      syntax: "XIRR(values, dates, [guess])",
      shortDescription: "Finds the annualized rate that sets XNPV to zero for irregularly dated cash flows.",
      difficulty: "Advanced",
      exampleFormula: "=XIRR(H2:H5,I2:I5)",
      exampleResult: 0.273144698866988,
      exampleNumberFormat: "Percentage",
      exampleTargetCell: "H7",
      exampleSetup: [
        { cell: "H2", value: "-10000" },
        { cell: "H3", value: "3000" },
        { cell: "H4", value: "4200" },
        { cell: "H5", value: "6800" },
        { cell: "I2", value: "=DATE(2026,1,1)" },
        { cell: "I3", value: "=DATE(2026,7,1)" },
        { cell: "I4", value: "=DATE(2027,3,15)" },
        { cell: "I5", value: "=DATE(2028,1,1)" }
      ],
      arguments: [
        { name: "values", description: "Cash flows containing at least one outflow and one inflow." },
        { name: "dates", description: "Actual dates aligned with the cash-flow series." },
        { name: "guess", description: "Optional starting estimate for the annualized rate; defaults to 10%." }
      ]
    },
    {
      id: "ifs",
      name: "IFS",
      category: "Logic",
      syntax: "IFS(logical_test1, value_if_true1, [logical_test2, value_if_true2], ...)",
      shortDescription: "Tests conditions in order and returns the value paired with the first TRUE condition.",
      difficulty: "Intermediate",
      exampleFormula: "=IFS(D2>=80000,\"High\",D2>=70000,\"Medium\",TRUE,\"Low\")",
      exampleResult: "Medium",
      arguments: [
        { name: "logical_test1", description: "First condition to evaluate." },
        { name: "value_if_true1", description: "Value returned if the first condition is TRUE." },
        { name: "logical_test2, value_if_true2", description: "Optional additional condition and result pairs checked in order." }
      ]
    },
    {
      id: "switch",
      name: "SWITCH",
      category: "Logic",
      syntax: "SWITCH(expression, value1, result1, [value2, result2], ..., [default])",
      shortDescription: "Compares one expression with listed values and returns the result for the first match.",
      difficulty: "Intermediate",
      exampleFormula: "=SWITCH(C2,\"Finance\",\"Budget\",\"IT\",\"Technology\",\"Other\")",
      exampleResult: "Budget",
      arguments: [
        { name: "expression", description: "Value evaluated once and compared with each listed case." },
        { name: "value1", description: "First value to compare with the expression." },
        { name: "result1", description: "Result returned if value1 matches." },
        { name: "default", description: "Optional final fallback when no listed value matches." }
      ]
    },
    {
      id: "choose",
      name: "CHOOSE",
      category: "Lookup",
      syntax: "CHOOSE(index_num, value1, [value2], ...)",
      shortDescription: "Returns one item from a list according to a 1-based index number.",
      difficulty: "Beginner",
      exampleFormula: "=CHOOSE(2,\"Low\",\"Medium\",\"High\")",
      exampleResult: "Medium",
      arguments: [
        { name: "index_num", description: "1-based position of the value to return." },
        { name: "value1", description: "First available value." },
        { name: "value2", description: "Optional additional values." }
      ]
    },
    {
      id: "xmatch",
      name: "XMATCH",
      category: "Lookup",
      syntax: "XMATCH(lookup_value, lookup_array, [match_mode], [search_mode])",
      shortDescription: "Returns the relative position of a match, using exact matching by default and optional wildcard or approximate modes.",
      difficulty: "Intermediate",
      exampleFormula: "=XMATCH(\"Noah\",B2:B7)",
      exampleResult: 4,
      arguments: [
        { name: "lookup_value", description: "Value to locate." },
        { name: "lookup_array", description: "One-dimensional row or column to search." },
        { name: "match_mode", description: "Optional 0 exact, -1 next smaller, 1 next larger, or 2 wildcard." },
        { name: "search_mode", description: "Optional 1 first-to-last or -1 last-to-first." }
      ]
    },
    {
      id: "networkdays",
      name: "NETWORKDAYS",
      category: "Date",
      syntax: "NETWORKDAYS(start_date, end_date, [holidays])",
      shortDescription: "Counts Monday-through-Friday workdays between two dates, excluding optional holidays.",
      difficulty: "Intermediate",
      exampleFormula: "=NETWORKDAYS(DATE(2026,8,17),DATE(2026,8,21))",
      exampleResult: 5,
      arguments: [
        { name: "start_date", description: "First date in the inclusive interval." },
        { name: "end_date", description: "Last date in the inclusive interval." },
        { name: "holidays", description: "Optional date or range of dates to exclude." }
      ]
    },
    {
      id: "workday",
      name: "WORKDAY",
      category: "Date",
      syntax: "WORKDAY(start_date, days, [holidays])",
      shortDescription: "Moves forward or backward by a specified number of Monday-through-Friday workdays.",
      difficulty: "Intermediate",
      exampleFormula: "=WORKDAY(DATE(2026,8,17),5)",
      exampleResult: 46258,
      exampleNumberFormat: "Date",
      arguments: [
        { name: "start_date", description: "Starting date." },
        { name: "days", description: "Number of workdays to move; negative values move backward." },
        { name: "holidays", description: "Optional date or range of dates to skip." }
      ]
    },
    {
      id: "let",
      name: "LET",
      category: "Advanced",
      syntax: "LET(name1, value1, [name2, value2], ..., calculation)",
      shortDescription: "Assigns names to intermediate values inside one formula so repeated logic is easier to read and maintain.",
      difficulty: "Advanced",
      exampleFormula: "=LET(bonusRate,0.05,D2*bonusRate)",
      exampleResult: 3600,
      arguments: [
        { name: "name1", description: "Local variable name used only inside this LET formula." },
        { name: "value1", description: "Value or expression assigned to name1." },
        { name: "name2, value2", description: "Optional additional local bindings evaluated in order." },
        { name: "calculation", description: "Final expression evaluated using the local names." }
      ]
    },
    {
      id: "sequence",
      name: "SEQUENCE",
      category: "Dynamic Array",
      syntax: "SEQUENCE(rows, [columns], [start], [step])",
      shortDescription: "Generates a rectangular sequence that spills from one anchor cell.",
      difficulty: "Beginner",
      lessonNote: "The formula lives only in the anchor cell. The remaining values are spill cells controlled by that anchor.",
      exampleFormula: "=SEQUENCE(3,2,10,5)",
      exampleResult: [[10, 15], [20, 25], [30, 35]],
      exampleShape: [3, 2],
      exampleTargetCell: "H15",
      arguments: [
        { name: "rows", description: "Number of rows to generate; must be at least 1." },
        { name: "columns", description: "Optional number of columns; defaults to 1." },
        { name: "start", description: "Optional first value; defaults to 1." },
        { name: "step", description: "Optional amount added for each next value; defaults to 1." }
      ]
    },
    {
      id: "filter",
      name: "FILTER",
      category: "Dynamic Array",
      syntax: "FILTER(array, include, [if_empty])",
      shortDescription: "Returns only rows whose aligned include values are TRUE.",
      difficulty: "Intermediate",
      exampleFormula: "=FILTER(A2:E7,C2:C7=\"Finance\")",
      exampleResult: [
        [1001, "Maya", "Finance", 72000, 4],
        [1003, "Sofia", "Finance", 68000, 3],
        [1006, "Lucas", "Finance", 76000, 5]
      ],
      exampleShape: [3, 5],
      exampleTargetCell: "H15",
      arguments: [
        { name: "array", description: "Rectangular source array whose rows will be returned." },
        { name: "include", description: "One TRUE/FALSE-compatible value aligned with each source row." },
        { name: "if_empty", description: "Optional fallback returned when no rows qualify." }
      ]
    },
    {
      id: "sort",
      name: "SORT",
      category: "Dynamic Array",
      syntax: "SORT(array, [sort_index], [sort_order], [by_col])",
      shortDescription: "Sorts complete rows by a relative column while preserving row alignment.",
      difficulty: "Intermediate",
      exampleFormula: "=SORT(A2:E7,4,-1)",
      exampleResult: [
        [1005, "Emma", "IT", 89000, 8],
        [1002, "Liam", "IT", 81000, 6],
        [1006, "Lucas", "Finance", 76000, 5],
        [1001, "Maya", "Finance", 72000, 4],
        [1003, "Sofia", "Finance", 68000, 3],
        [1004, "Noah", "HR", 64000, 5]
      ],
      exampleShape: [6, 5],
      exampleTargetCell: "H15",
      arguments: [
        { name: "array", description: "Source array to sort." },
        { name: "sort_index", description: "Optional relative column number; defaults to 1." },
        { name: "sort_order", description: "Optional 1 for ascending or -1 for descending." },
        { name: "by_col", description: "Optional TRUE to sort columns; defaults to FALSE." }
      ]
    },
    {
      id: "sortby",
      name: "SORTBY",
      category: "Dynamic Array",
      syntax: "SORTBY(array, by_array1, [sort_order1], [by_array2], [sort_order2])",
      shortDescription: "Sorts an array using one or two separate aligned sort arrays.",
      difficulty: "Intermediate",
      exampleFormula: "=SORTBY(A2:E7,D2:D7,-1)",
      exampleResult: [
        [1005, "Emma", "IT", 89000, 8],
        [1002, "Liam", "IT", 81000, 6],
        [1006, "Lucas", "Finance", 76000, 5],
        [1001, "Maya", "Finance", 72000, 4],
        [1003, "Sofia", "Finance", 68000, 3],
        [1004, "Noah", "HR", 64000, 5]
      ],
      exampleShape: [6, 5],
      exampleTargetCell: "H15",
      arguments: [
        { name: "array", description: "Array whose complete rows are returned." },
        { name: "by_array1", description: "First one-column sort key aligned with the source rows." },
        { name: "sort_order1", description: "Optional direction for the first key: 1 or -1." },
        { name: "by_array2", description: "Optional second aligned sort key." },
        { name: "sort_order2", description: "Optional direction for the second key: 1 or -1." }
      ]
    },
    {
      id: "unique",
      name: "UNIQUE",
      category: "Dynamic Array",
      syntax: "UNIQUE(array, [by_col], [exactly_once])",
      shortDescription: "Returns distinct rows in first-occurrence order.",
      difficulty: "Intermediate",
      exampleFormula: "=UNIQUE(C2:C7)",
      exampleResult: [["Finance"], ["IT"], ["HR"]],
      exampleShape: [3, 1],
      exampleTargetCell: "H15",
      arguments: [
        { name: "array", description: "Source rows or columns to compare." },
        { name: "by_col", description: "Optional TRUE to compare columns instead of rows." },
        { name: "exactly_once", description: "Optional TRUE to keep only values occurring once." }
      ]
    }
  ];

  global.ExcelFunctionCatalog = catalog;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = catalog;
  }
})(typeof window === "undefined" ? globalThis : window);
