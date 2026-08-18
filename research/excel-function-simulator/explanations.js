((global) => {
  "use strict";

  const PURPOSES = {
    SUM: "Adds numeric values.",
    AVERAGE: "Adds numeric values and divides by how many there are.",
    MIN: "Finds the smallest numeric value.",
    MAX: "Finds the largest numeric value.",
    COUNT: "Counts cells containing numeric values.",
    COUNTIF: "Counts cells that meet one condition.",
    SUMIF: "Adds values whose aligned cells meet one condition.",
    AVERAGEIF: "Averages values whose aligned cells meet one condition.",
    COUNTIFS: "Counts positions where every condition is met.",
    SUMIFS: "Adds values where every aligned condition is met.",
    AVERAGEIFS: "Averages values where every aligned condition is met.",
    VLOOKUP: "Searches down the first table column, then returns from another column.",
    HLOOKUP: "Searches across the first table row, then returns from another row.",
    XLOOKUP: "Searches one range and returns the aligned value from another range.",
    XMATCH: "Returns the relative position of a match, using exact matching by default.",
    MATCH: "Returns the relative position of a value in a lookup range.",
    INDEX: "Returns the value at a relative row and column within an array.",
    IF: "Tests a condition and returns one of two possible values.",
    IFS: "Tests condition/result pairs in order and returns the first result whose condition is TRUE.",
    SWITCH: "Compares one expression with listed values and returns the result for the first match.",
    CHOOSE: "Returns one item from a list using a one-based index number.",
    LET: "Assigns local names to intermediate values and uses them in a final calculation.",
    AND: "Returns TRUE only when every logical test is TRUE.",
    OR: "Returns TRUE when at least one logical test is TRUE.",
    NOT: "Reverses a logical value.",
    LEN: "Counts every character in text, including spaces.",
    LEFT: "Returns characters from the start of text.",
    RIGHT: "Returns characters from the end of text.",
    MID: "Returns characters from a one-based position within text.",
    TRIM: "Removes outer spaces and reduces repeated interior spaces.",
    UPPER: "Converts text to uppercase.",
    LOWER: "Converts text to lowercase.",
    PROPER: "Capitalizes the first letter of each word.",
    CONCAT: "Combines text values and ranges in order.",
    TEXTJOIN: "Joins text values with a delimiter and optional blank filtering.",
    FIND: "Returns the one-based position of a case-sensitive text match.",
    SEARCH: "Returns the one-based position of a case-insensitive text match.",
    SUBSTITUTE: "Replaces matching text, either everywhere or at one occurrence.",
    REPLACE: "Replaces a specified number of characters at a one-based position.",
    DATE: "Builds a date serial from year, month, and day values.",
    YEAR: "Returns the year component of a date serial.",
    MONTH: "Returns the month number from a date serial.",
    DAY: "Returns the day of the month from a date serial.",
    TODAY: "Returns the current local calendar date as a date serial.",
    DAYS: "Returns the number of days between two dates.",
    EDATE: "Moves a date forward or backward by whole months.",
    EOMONTH: "Returns the final day of a month at a chosen offset.",
    WEEKDAY: "Returns a weekday number using the selected week numbering system.",
    NETWORKDAYS: "Counts Monday-through-Friday workdays between two dates, excluding optional holidays.",
    WORKDAY: "Moves a date forward or backward by a specified number of workdays.",
    ROUND: "Rounds a number to the requested decimal position.",
    ROUNDUP: "Rounds a number away from zero.",
    ROUNDDOWN: "Rounds a number toward zero.",
    INT: "Rounds a number down to the nearest integer.",
    ABS: "Returns a number's distance from zero.",
    MOD: "Returns the Excel-style remainder after division.",
    IFERROR: "Returns a fallback when the primary expression produces any spreadsheet error.",
    IFNA: "Returns a fallback only when the primary expression produces #N/A.",
    MEDIAN: "Returns the middle value of an ordered numeric data set.",
    "MODE.SNGL": "Returns the most frequently occurring numeric value.",
    "STDEV.S": "Estimates sample standard deviation using n − 1 in the denominator.",
    "STDEV.P": "Calculates population standard deviation using n in the denominator.",
    "VAR.S": "Estimates sample variance using n − 1 in the denominator.",
    "VAR.P": "Calculates population variance using n in the denominator.",
    "RANK.EQ": "Returns a number's rank relative to a numeric list, giving tied values the same rank.",
    "PERCENTILE.INC": "Returns an inclusive percentile, interpolating between ordered values when needed.",
    "QUARTILE.INC": "Returns an inclusive quartile from 0 through 4.",
    CORREL: "Measures the strength and direction of a linear relationship between two numeric arrays.",
    "COVARIANCE.S": "Returns sample covariance for paired numeric observations.",
    PV: "Returns the present value of equal periodic cash flows at a constant rate.",
    FV: "Returns the future value of equal periodic cash flows at a constant rate.",
    PMT: "Calculates the equal periodic payment for a loan or annuity.",
    NPV: "Discounts equally spaced future cash flows back to the present.",
    IRR: "Finds the periodic return that makes net present value equal to zero.",
    XNPV: "Discounts irregular cash flows using their actual dates.",
    XIRR: "Finds the annualized return that makes XNPV equal to zero for irregular cash flows.",
    SEQUENCE: "Generates a rectangular sequence from one anchor formula.",
    FILTER: "Returns rows whose aligned include values are TRUE.",
    SORT: "Reorders complete rows or columns using a relative sort index.",
    SORTBY: "Reorders an array using one or more separate aligned sort arrays.",
    UNIQUE: "Returns the first occurrence of each distinct row or column."
  };

  const ERROR_EXPLANATIONS = {
    "#DIV/0!": {
      title: "Division by zero",
      message: "The formula attempted to divide a number by zero."
    },
    "#NAME?": {
      title: "Unrecognized name",
      message: "The formula contains a function or name that is not recognized."
    },
    "#ERROR!": {
      title: "Formula error",
      message: "The formula could not be parsed or evaluated."
    },
    "#VALUE!": {
      title: "Invalid value or range",
      message: "The formula received an invalid argument or incompatible range dimensions."
    },
    "#N/A": {
      title: "Lookup value not found",
      message: "The lookup searched the selected range but did not find a usable match."
    },
    "#REF!": {
      title: "Invalid reference",
      message: "The requested lookup or INDEX position is outside the selected range."
    },
    "#NUM!": {
      title: "Invalid number",
      message: "A numeric argument is outside the supported range, or an iterative calculation could not find a valid result."
    },
    "#CALC!": {
      title: "Empty array",
      message: "The array calculation produced no results. Supply a fallback value when appropriate."
    },
    "#SPILL!": {
      title: "Spill range blocked",
      message: "The array cannot spill because a destination cell is occupied or outside the worksheet."
    }
  };

  function isNumeric(value) {
    return typeof value === "number" && Number.isFinite(value);
  }

  function calculationValue(value) {
    if (typeof value === "boolean") return value ? "TRUE" : "FALSE";
    if (isNumeric(value)) {
      if (Object.is(value, -0)) return "0";
      if (Number.isInteger(value)) return String(value);
      return String(Number(value.toPrecision(12)));
    }
    if (value === "" || value === null || value === undefined) return "0";
    return String(value);
  }

  function walkAst(node, visitor) {
    if (!node) return;
    visitor(node);

    if (node.type === "unary") {
      walkAst(node.operand, visitor);
    } else if (node.type === "postfix") {
      walkAst(node.operand, visitor);
    } else if (node.type === "binary") {
      walkAst(node.left, visitor);
      walkAst(node.right, visitor);
    } else if (node.type === "comparison") {
      walkAst(node.left, visitor);
      walkAst(node.right, visitor);
    } else if (node.type === "function") {
      node.arguments.forEach((argument) => walkAst(argument, visitor));
    }
  }

  function astDetails(ast, expandRange) {
    const functionNames = [];
    const ranges = [];
    const seenFunctions = new Set();
    const seenRanges = new Set();

    walkAst(ast, (node) => {
      if (node.type === "function" && !seenFunctions.has(node.name)) {
        seenFunctions.add(node.name);
        functionNames.push(node.name);
      }

      if (node.type === "range") {
        const label = `${node.startAddress || node.start}:${node.endAddress || node.end}`;
        if (!seenRanges.has(label)) {
          seenRanges.add(label);
          ranges.push({
            label,
            start: node.start,
            end: node.end,
            references: expandRange(node.start, node.end)
          });
        }
      }
    });

    return { functionNames, ranges };
  }

  function referenceLockDetails(ast) {
    const details = [];
    const seen = new Set();

    function addReference(node) {
      if (!node) return;
      const address = node.address || node.reference;
      const key = `${address}:${Boolean(node.columnAbsolute)}:${Boolean(node.rowAbsolute)}`;
      if (seen.has(key)) return;
      seen.add(key);
      details.push({
        address,
        columnAbsolute: Boolean(node.columnAbsolute),
        rowAbsolute: Boolean(node.rowAbsolute)
      });
    }

    walkAst(ast, (node) => {
      if (node.type === "reference") {
        addReference(node);
      } else if (node.type === "range") {
        addReference(node.startReference);
        addReference(node.endReference);
      }
    });

    return details;
  }

  function referencesInNode(node, expandRange) {
    const references = new Set();

    walkAst(node, (child) => {
      if (child.type === "reference") {
        references.add(child.reference);
      } else if (child.type === "range") {
        expandRange(child.start, child.end).forEach((reference) => references.add(reference));
      }
    });

    return [...references];
  }

  function referenceEntries(references, getCellValue, getCellNumberFormat) {
    return references.map((reference) => {
      const value = getCellValue(reference);
      return {
        reference,
        value,
        numberFormat: getCellNumberFormat ? getCellNumberFormat(reference) : "General",
        numeric: isNumeric(value),
        ignored: value !== "" && !isNumeric(value)
      };
    });
  }

  function functionExpression(node, context) {
    if (["IF", "AND", "OR", "NOT"].includes(node.name)) {
      const argumentsList = node.arguments.map((argument) => (
        expressionParts(argument, context).text
      ));
      return `${node.name}(${argumentsList.join(", ")})`;
    }

    const entries = referenceEntries(
      referencesInNode(node, context.expandRange),
      context.getCellValue
    );
    const values = entries.filter((entry) => entry.numeric).map((entry) => entry.value);
    const formattedValues = values.map(calculationValue);

    if (node.name === "SUM") {
      return `(${formattedValues.join(" + ") || "0"})`;
    }

    if (node.name === "AVERAGE") {
      const sum = values.reduce((total, value) => total + value, 0);
      return `(${calculationValue(sum)} / ${values.length})`;
    }

    if (node.name === "MIN" || node.name === "MAX") {
      return `${node.name}(${formattedValues.join(", ")})`;
    }

    if (node.name === "COUNT") {
      return String(values.length);
    }

    return `${node.name}(...)`;
  }

  function expressionParts(node, context) {
    if (node.type === "number") {
      return { text: calculationValue(node.value), precedence: 5 };
    }

    if (node.type === "string") {
      return { text: node.value, precedence: 5 };
    }

    if (node.type === "boolean") {
      return { text: calculationValue(node.value), precedence: 5 };
    }

    if (node.type === "name") {
      return { text: node.rawName || node.name, precedence: 5 };
    }

    if (node.type === "reference") {
      const value = context.getCellValue(node.reference);
      const numberFormat = context.getCellNumberFormat?.(node.reference) || "General";
      const text = context.formatValue
        ? context.formatValue(value, numberFormat)
        : calculationValue(value);
      return { text, precedence: 5 };
    }

    if (node.type === "range") {
      const values = context.expandRange(node.start, node.end).map((reference) => {
        const value = context.getCellValue(reference);
        const numberFormat = context.getCellNumberFormat?.(reference) || "General";
        return context.formatValue
          ? context.formatValue(value, numberFormat)
          : calculationValue(value);
      });
      return { text: values.join(", "), precedence: 5 };
    }

    if (node.type === "function") {
      return { text: functionExpression(node, context), precedence: 5 };
    }

    if (node.type === "unary") {
      const operand = expressionParts(node.operand, context);
      const text = operand.precedence < 4 ? `(${operand.text})` : operand.text;
      return { text: `${node.operator}${text}`, precedence: 4 };
    }

    if (node.type === "postfix") {
      const operand = expressionParts(node.operand, context);
      const text = operand.precedence < 5 ? `(${operand.text})` : operand.text;
      return { text: `${text}${node.operator}`, precedence: 5 };
    }

    if (node.type === "binary") {
      const precedence = node.operator === "&"
        ? 1
        : (node.operator === "*" || node.operator === "/" ? 3 : 2);
      const symbols = { "+": "+", "-": "-", "*": "×", "/": "/", "&": "&" };
      const left = expressionParts(node.left, context);
      const right = expressionParts(node.right, context);
      const leftText = left.precedence < precedence ? `(${left.text})` : left.text;
      const rightNeedsParentheses = right.precedence < precedence
        || ((node.operator === "-" || node.operator === "/") && right.precedence === precedence);
      const rightText = rightNeedsParentheses ? `(${right.text})` : right.text;
      return {
        text: `${leftText} ${symbols[node.operator]} ${rightText}`,
        precedence
      };
    }

    if (node.type === "comparison") {
      const left = expressionParts(node.left, context);
      const right = expressionParts(node.right, context);
      return {
        text: `${left.text} ${node.operator} ${right.text}`,
        precedence: 0
      };
    }

    return { text: "Unable to describe calculation", precedence: 5 };
  }

  function formulaExpressionParts(node) {
    if (node.type === "number") return { text: calculationValue(node.value), precedence: 5 };
    if (node.type === "boolean") return { text: calculationValue(node.value), precedence: 5 };
    if (node.type === "string") {
      return { text: `"${node.value.replaceAll("\"", "\"\"")}"`, precedence: 5 };
    }
    if (node.type === "name") return { text: node.rawName || node.name, precedence: 5 };
    if (node.type === "reference") return { text: node.address || node.reference, precedence: 5 };
    if (node.type === "range") {
      return {
        text: `${node.startAddress || node.start}:${node.endAddress || node.end}`,
        precedence: 5
      };
    }
    if (node.type === "function") {
      const argumentsList = node.arguments.map((argument) => formulaExpressionParts(argument).text);
      return { text: `${node.name}(${argumentsList.join(", ")})`, precedence: 5 };
    }
    if (node.type === "unary") {
      const operand = formulaExpressionParts(node.operand);
      const text = operand.precedence < 4 ? `(${operand.text})` : operand.text;
      return { text: `${node.operator}${text}`, precedence: 4 };
    }
    if (node.type === "postfix") {
      const operand = formulaExpressionParts(node.operand);
      const text = operand.precedence < 5 ? `(${operand.text})` : operand.text;
      return { text: `${text}%`, precedence: 5 };
    }
    if (node.type === "binary") {
      const precedence = node.operator === "&"
        ? 1
        : (node.operator === "*" || node.operator === "/" ? 3 : 2);
      const left = formulaExpressionParts(node.left);
      const right = formulaExpressionParts(node.right);
      const leftText = left.precedence < precedence ? `(${left.text})` : left.text;
      const rightNeedsParentheses = right.precedence < precedence
        || ((node.operator === "-" || node.operator === "/") && right.precedence === precedence);
      const rightText = rightNeedsParentheses ? `(${right.text})` : right.text;
      return {
        text: `${leftText} ${node.operator} ${rightText}`,
        precedence
      };
    }
    if (node.type === "comparison") {
      const left = formulaExpressionParts(node.left);
      const right = formulaExpressionParts(node.right);
      return { text: `${left.text} ${node.operator} ${right.text}`, precedence: 0 };
    }
    return { text: "Unknown expression", precedence: 5 };
  }

  function criterionComparison(value, criterion) {
    const candidate = value === "" || value === null || value === undefined
      ? "(blank)"
      : calculationValue(value);
    const target = criterion.operand.kind === "blank"
      ? "(blank)"
      : calculationValue(criterion.operand.value);

    if (criterion.usesWildcards) return `${candidate} matches ${criterion.display}`;
    return `${candidate} ${criterion.operator} ${target}`;
  }

  function conditionalExplanation(trace) {
    return {
      functionName: trace.functionName,
      criteria: trace.criteria.map((entry) => ({
        range: entry.range.label,
        references: entry.range.cells.map((cell) => cell.reference),
        criterion: entry.criterion
      })),
      aggregateRange: trace.aggregateRange ? {
        label: trace.aggregateRange.label,
        references: trace.aggregateRange.cells.map((cell) => cell.reference)
      } : null,
      positions: trace.positions.map((position) => ({
        index: position.index,
        checks: position.checks.map((check, checkIndex) => ({
          ...check,
          criterion: trace.criteria[checkIndex].criterion.display,
          comparison: criterionComparison(check.value, trace.criteria[checkIndex].criterion)
        })),
        allMatched: position.allMatched,
        aggregate: position.aggregate
      })),
      includedValues: trace.includedValues,
      summary: trace.summary
    };
  }

  function safeEvaluate(node, evaluateAst) {
    try {
      return evaluateAst(node);
    } catch (error) {
      return error.code || "#ERROR!";
    }
  }

  function logicalTest(node, context) {
    return {
      expression: formulaExpressionParts(node).text,
      calculation: expressionParts(node, context).text,
      result: safeEvaluate(node, context.evaluateAst)
    };
  }

  function logicalExplanation(ast, result, context) {
    if (ast?.type !== "function" || !["IF", "AND", "OR", "NOT"].includes(ast.name)) {
      return null;
    }

    if (ast.name === "IF") {
      if (ast.arguments.length !== 3) return null;
      const conditionNode = ast.arguments[0];
      const conditionResult = safeEvaluate(conditionNode, context.evaluateAst);
      const conditionIsTrue = conditionResult === true
        || (typeof conditionResult === "number" && conditionResult !== 0);
      const nestedTests = conditionNode.type === "function"
        && ["AND", "OR"].includes(conditionNode.name)
        ? conditionNode.arguments.map((argument) => logicalTest(argument, context))
        : [];
      const selectedBranch = conditionIsTrue ? 1 : 2;
      const selectedNode = ast.arguments[selectedBranch];

      return {
        kind: "IF",
        condition: formulaExpressionParts(conditionNode).text,
        comparison: expressionParts(conditionNode, context).text,
        conditionResult,
        tests: nestedTests,
        rule: conditionNode.type === "function" ? PURPOSES[conditionNode.name] || "" : "",
        chosenBranch: conditionIsTrue ? "value_if_true" : "value_if_false",
        returnedValue: result,
        nestedDecision: selectedNode.type === "function" && selectedNode.name === "IF"
          ? logicalExplanation(selectedNode, result, context)
          : null
      };
    }

    if (!ast.arguments.length) return null;
    const tests = ast.arguments.map((argument) => logicalTest(argument, context));
    return {
      kind: ast.name,
      tests,
      rule: PURPOSES[ast.name],
      result
    };
  }

  function purposeFor(ast, functionNames) {
    if (ast?.type === "postfix" && ast.operator === "#") {
      return "Returns the entire dynamic-array spill range owned by the referenced anchor cell.";
    }
    if (ast?.type === "function" && PURPOSES[ast.name]) return PURPOSES[ast.name];
    if (functionNames.length === 1 && PURPOSES[functionNames[0]]) {
      return `${PURPOSES[functionNames[0]]} The function result is then used in an arithmetic expression.`;
    }
    return "Calculates an arithmetic expression using the referenced values.";
  }

  function buildExplanation(options) {
    const {
      formula,
      ast,
      result,
      dependencies = [],
      getCellValue,
      expandRange,
      evaluateAst,
      analyzeConditional,
      analyzeLookup,
      analyzeText,
      analyzeDate,
      analyzeMath,
      analyzeStatistics,
      analyzeFinancial,
      analyzeAdvanced,
      analyzeError,
      analyzeDynamic,
      spill = null,
      spillError = null,
      numberFormat = "General",
      numberFormatOverride = null,
      formatOptions = {},
      displayedResult,
      getCellNumberFormat,
      getSpill,
      formatValue
    } = options;
    const details = ast ? astDetails(ast, expandRange) : { functionNames: [], ranges: [] };
    const entries = referenceEntries(dependencies, getCellValue, getCellNumberFormat);
    const numericEntries = entries.filter((entry) => entry.numeric);
    const ignoredEntries = entries.filter((entry) => entry.ignored);
    const numericValues = numericEntries.map((entry) => entry.value);
    const rootFunction = ast?.type === "function" ? ast.name : null;
    const explanation = {
      formula,
      functionName: details.functionNames.join(", "),
      purpose: purposeFor(ast, details.functionNames),
      ranges: details.ranges,
      referenceGroups: [],
      metrics: [],
      calculation: ast ? expressionParts(ast, {
        getCellValue,
        getCellNumberFormat,
        expandRange,
        formatValue
      }).text : "",
      result,
      numberFormat,
      numberFormatOverride,
      formatOptions,
      underlyingDisplay: formatValue ? formatValue(result, "General") : calculationValue(result),
      displayedResult: displayedResult ?? (formatValue ? formatValue(result, numberFormat, formatOptions) : calculationValue(result)),
      referenceLocks: ast ? referenceLockDetails(ast) : [],
      error: ERROR_EXPLANATIONS[result] || null,
      logical: ast && evaluateAst
        ? logicalExplanation(ast, result, { getCellValue, expandRange, evaluateAst })
        : null,
      conditional: null,
      text: null,
      date: null,
      math: null,
      statistical: null,
      financial: null,
      advanced: null,
      errorHandling: null,
      dynamicArray: null,
      spillReference: null
    };
    explanation.lookup = null;

    if (ast?.type === "postfix" && ast.operator === "#"
      && ast.operand?.type === "reference" && typeof getSpill === "function") {
      try {
        const descriptor = getSpill(ast.operand.reference);
        if (descriptor) {
          explanation.spillReference = {
            anchor: ast.operand.reference,
            range: descriptor.range,
            rows: descriptor.rows,
            columns: descriptor.columns,
            values: descriptor.values.map((row) => row.slice()),
            formats: descriptor.formats?.map((row) => row.slice()) || null,
            references: descriptor.references?.slice() || []
          };
        }
      } catch (error) {
        explanation.spillReference = null;
      }
    }

    if (ast && analyzeConditional) {
      try {
        const trace = analyzeConditional(ast);
        if (trace) explanation.conditional = conditionalExplanation(trace);
      } catch (error) {
        explanation.conditional = null;
      }
    }

    if (ast && analyzeLookup) {
      try {
        explanation.lookup = analyzeLookup(ast);
      } catch (error) {
        explanation.lookup = null;
      }
    }

    if (ast && analyzeText) {
      try {
        explanation.text = analyzeText(ast);
      } catch (error) {
        explanation.text = null;
      }
    }

    if (ast && analyzeDate) {
      try {
        explanation.date = analyzeDate(ast);
      } catch (error) {
        explanation.date = null;
      }
    }

    if (ast && analyzeMath) {
      try {
        explanation.math = analyzeMath(ast);
      } catch (error) {
        explanation.math = null;
      }
    }

    if (ast && analyzeStatistics) {
      try {
        explanation.statistical = analyzeStatistics(ast);
      } catch (error) {
        explanation.statistical = null;
      }
    }

    if (ast && analyzeFinancial) {
      try {
        explanation.financial = analyzeFinancial(ast);
      } catch (error) {
        explanation.financial = null;
      }
    }

    if (ast && analyzeAdvanced) {
      try {
        const trace = analyzeAdvanced(ast);
        if (trace) {
          if (trace.kind === "ifs") {
            explanation.advanced = {
              ...trace,
              branches: trace.branches.map((branch) => ({
                ...branch,
                conditionExpression: formulaExpressionParts(branch.conditionNode).text,
                valueExpression: formulaExpressionParts(branch.valueNode).text
              }))
            };
          } else if (trace.kind === "switch") {
            explanation.advanced = {
              ...trace,
              cases: trace.cases.map((entry) => ({
                ...entry,
                resultExpression: formulaExpressionParts(entry.valueNode).text
              }))
            };
          } else if (trace.kind === "choose") {
            explanation.advanced = {
              ...trace,
              selectedExpression: trace.selectedNode
                ? formulaExpressionParts(trace.selectedNode).text
                : null
            };
          } else if (trace.kind === "let") {
            explanation.advanced = {
              ...trace,
              bindings: trace.bindings.map((binding) => ({
                name: binding.name,
                value: binding.value,
                expression: formulaExpressionParts(binding.valueNode).text
              })),
              calculationExpression: formulaExpressionParts(trace.calculationNode).text
            };
          } else {
            explanation.advanced = trace;
          }
        }
      } catch (error) {
        explanation.advanced = null;
      }
    }

    if (ast && analyzeError) {
      try {
        const trace = analyzeError(ast);
        if (trace) {
          explanation.errorHandling = {
            ...trace,
            primaryExpression: formulaExpressionParts(trace.primaryNode).text,
            fallbackExpression: formulaExpressionParts(trace.fallbackNode).text
          };
        }
      } catch (error) {
        explanation.errorHandling = null;
      }
    }

    if (ast && analyzeDynamic) {
      try {
        const trace = analyzeDynamic(ast);
        if (trace) {
          explanation.dynamicArray = {
            ...trace,
            conditionExpression: trace.conditionNode
              ? formulaExpressionParts(trace.conditionNode).text
              : null,
            spill,
            spillError
          };
        }
      } catch (error) {
        explanation.dynamicArray = null;
      }
    }

    if (explanation.dynamicArray) {
      explanation.referenceGroups = [];
      explanation.metrics = [];
      explanation.calculation = "";
      return explanation;
    }

    if (explanation.lookup) {
      explanation.referenceGroups = [];
      explanation.metrics = [];
      explanation.calculation = "";
      return explanation;
    }

    if (explanation.conditional) {
      const { functionName, includedValues, summary } = explanation.conditional;
      explanation.referenceGroups = [];
      explanation.metrics.push({ label: "Matches", value: summary.matches });

      if (functionName === "SUMIF" || functionName === "SUMIFS") {
        explanation.calculation = includedValues.length
          ? includedValues.map((entry) => calculationValue(entry.value)).join(" + ")
          : "0";
      } else if (functionName === "AVERAGEIF" || functionName === "AVERAGEIFS") {
        explanation.metrics.push({ label: "Sum", value: summary.sum });
        explanation.metrics.push({ label: "Count", value: summary.includedCount });
        explanation.calculation = `${calculationValue(summary.sum)} / ${summary.includedCount}`;
      } else {
        explanation.calculation = `${summary.matches} matching position${summary.matches === 1 ? "" : "s"}`;
      }

      return explanation;
    }

    if (explanation.text) {
      explanation.referenceGroups = [];
      explanation.metrics = [];
      explanation.calculation = "";
      return explanation;
    }

    if (explanation.date) {
      explanation.referenceGroups = [];
      explanation.metrics = [];
      explanation.calculation = "";
      return explanation;
    }

    if (explanation.statistical) {
      explanation.referenceGroups = [];
      explanation.metrics = [];
      explanation.calculation = "";
      return explanation;
    }

    if (explanation.financial) {
      explanation.referenceGroups = [];
      explanation.metrics = [];
      explanation.calculation = "";
      return explanation;
    }

    if (explanation.advanced) {
      explanation.referenceGroups = [];
      explanation.metrics = [];
      explanation.calculation = "";
      return explanation;
    }

    if (explanation.math || explanation.errorHandling) {
      explanation.referenceGroups = [];
      explanation.metrics = [];
      explanation.calculation = "";
      return explanation;
    }

    if (explanation.logical) explanation.calculation = "";

    if (rootFunction === "COUNT") {
      explanation.referenceGroups.push({ label: "Numeric cells found", entries: numericEntries });
      if (ignoredEntries.length) {
        explanation.referenceGroups.push({ label: "Ignored nonnumeric cells", entries: ignoredEntries });
      }
      explanation.metrics.push({ label: "Count", value: numericEntries.length });
      explanation.calculation = `${numericEntries.length} numeric cell${numericEntries.length === 1 ? "" : "s"}`;
      return explanation;
    }

    if (entries.length) {
      const displayedEntries = explanation.logical
        ? entries.map((entry) => ({ ...entry, ignored: false }))
        : entries;
      explanation.referenceGroups.push({ label: "Values", entries: displayedEntries });
    }

    if (rootFunction === "SUM") {
      explanation.calculation = numericValues.map(calculationValue).join(" + ") || "0";
    } else if (rootFunction === "AVERAGE") {
      const sum = numericValues.reduce((total, value) => total + value, 0);
      explanation.metrics.push({ label: "Sum", value: sum });
      explanation.metrics.push({ label: "Count", value: numericValues.length });
      explanation.calculation = `${calculationValue(sum)} / ${numericValues.length}`;
    } else if (rootFunction === "MIN") {
      const smallest = numericValues.length ? Math.min(...numericValues) : 0;
      explanation.metrics.push({ label: "Smallest value", value: smallest });
    } else if (rootFunction === "MAX") {
      const largest = numericValues.length ? Math.max(...numericValues) : 0;
      explanation.metrics.push({ label: "Largest value", value: largest });
    }

    return explanation;
  }

  const api = {
    buildExplanation,
    calculationValue,
    walkAst
  };

  global.FormulaExplanations = api;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
})(typeof window === "undefined" ? globalThis : window);
