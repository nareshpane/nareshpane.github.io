((global) => {
  "use strict";

  const CriteriaEngine = global.CriteriaEngine
    || (typeof require === "function" ? require("./criteria.js") : null);
  const LookupEngine = global.LookupEngine
    || (typeof require === "function" ? require("./lookup.js") : null);
  const ExcelFormatting = global.ExcelFormatting
    || (typeof require === "function" ? require("./formatting.js") : null);
  const ERROR_VALUES = {
    CALC: "#CALC!",
    DIV_ZERO: "#DIV/0!",
    NAME: "#NAME?",
    GENERIC: "#ERROR!",
    VALUE: "#VALUE!",
    NA: "#N/A",
    REF: "#REF!",
    NUM: "#NUM!",
    SPILL: "#SPILL!"
  };

  class FormulaError extends Error {
    constructor(code) {
      super(code);
      this.code = code;
    }
  }

  class FormulaSyntaxError extends Error {}

  function tokenize(source) {
    const tokens = [];
    let position = 0;

    while (position < source.length) {
      const character = source[position];

      if (/\s/.test(character)) {
        position += 1;
        continue;
      }

      if (character === "\"") {
        let value = "";
        let closed = false;
        position += 1;

        while (position < source.length) {
          if (source[position] !== "\"") {
            value += source[position];
            position += 1;
            continue;
          }

          if (source[position + 1] === "\"") {
            value += "\"";
            position += 2;
            continue;
          }

          position += 1;
          closed = true;
          break;
        }

        if (!closed) throw new FormulaSyntaxError("Unterminated text literal");
        tokens.push({ type: "string", value });
        continue;
      }

      const comparison = ["<>", ">=", "<=", "=", ">", "<"]
        .find((operator) => source.startsWith(operator, position));
      if (comparison) {
        tokens.push({ type: "comparison", value: comparison });
        position += comparison.length;
        continue;
      }

      if ("+-*/&():,%".includes(character)) {
        const tokenTypes = {
          "+": "plus",
          "-": "minus",
          "*": "multiply",
          "/": "divide",
          "&": "concatenate",
          "(": "leftParen",
          ")": "rightParen",
          ":": "colon",
          ",": "comma",
          "%": "percent"
        };
        tokens.push({ type: tokenTypes[character], value: character });
        position += 1;
        continue;
      }

      const remaining = source.slice(position);
      const numberMatch = remaining.match(/^(?:\d+(?:\.\d*)?|\.\d+)/);
      if (numberMatch) {
        tokens.push({ type: "number", value: Number(numberMatch[0]) });
        position += numberMatch[0].length;
        continue;
      }

      const identifierMatch = remaining.match(/^[A-Za-z_][A-Za-z0-9_.]*/);
      if (identifierMatch) {
        tokens.push({ type: "identifier", value: identifierMatch[0] });
        position += identifierMatch[0].length;
        continue;
      }

      throw new FormulaSyntaxError(`Unexpected character at position ${position}`);
    }

    tokens.push({ type: "end", value: "" });
    return tokens;
  }

  function isCellReference(value) {
    return /^[A-Za-z]+[1-9]\d*$/.test(value);
  }

  class Parser {
    constructor(tokens) {
      this.tokens = tokens;
      this.position = 0;
    }

    current() {
      return this.tokens[this.position];
    }

    match(type) {
      if (this.current().type !== type) return false;
      this.position += 1;
      return true;
    }

    consume(type) {
      const token = this.current();
      if (token.type !== type) {
        throw new FormulaSyntaxError(`Expected ${type}`);
      }
      this.position += 1;
      return token;
    }

    parse() {
      const expression = this.parseComparison();
      this.consume("end");
      return expression;
    }

    parseComparison() {
      const left = this.parseConcatenation();
      if (this.current().type !== "comparison") return left;

      const operator = this.consume("comparison").value;
      return {
        type: "comparison",
        operator,
        left,
        right: this.parseConcatenation()
      };
    }

    parseConcatenation() {
      let expression = this.parseAdditive();

      while (this.current().type === "concatenate") {
        this.position += 1;
        expression = {
          type: "binary",
          operator: "&",
          left: expression,
          right: this.parseAdditive()
        };
      }

      return expression;
    }

    parseAdditive() {
      let expression = this.parseMultiplicative();

      while (this.current().type === "plus" || this.current().type === "minus") {
        const operator = this.current().value;
        this.position += 1;
        expression = {
          type: "binary",
          operator,
          left: expression,
          right: this.parseMultiplicative()
        };
      }

      return expression;
    }

    parseMultiplicative() {
      let expression = this.parseUnary();

      while (this.current().type === "multiply" || this.current().type === "divide") {
        const operator = this.current().value;
        this.position += 1;
        expression = {
          type: "binary",
          operator,
          left: expression,
          right: this.parseUnary()
        };
      }

      return expression;
    }

    parseUnary() {
      if (this.current().type === "plus" || this.current().type === "minus") {
        const operator = this.current().value;
        this.position += 1;
        return { type: "unary", operator, operand: this.parseUnary() };
      }

      return this.parsePostfix();
    }

    parsePostfix() {
      let expression = this.parsePrimary();
      while (this.match("percent")) {
        expression = { type: "postfix", operator: "%", operand: expression };
      }
      return expression;
    }

    parsePrimary() {
      if (this.current().type === "number") {
        return { type: "number", value: this.consume("number").value };
      }

      if (this.current().type === "string") {
        return { type: "string", value: this.consume("string").value };
      }

      if (this.match("leftParen")) {
        const expression = this.parseComparison();
        this.consume("rightParen");
        return expression;
      }

      if (this.current().type !== "identifier") {
        throw new FormulaSyntaxError("Expected a number, cell reference, or function");
      }

      const identifier = this.consume("identifier").value;
      if (this.match("leftParen")) {
        return this.parseFunctionCall(identifier);
      }

      if (identifier.toUpperCase() === "TRUE" || identifier.toUpperCase() === "FALSE") {
        return { type: "boolean", value: identifier.toUpperCase() === "TRUE" };
      }

      if (!isCellReference(identifier)) {
        throw new FormulaSyntaxError("Invalid cell reference");
      }

      const start = identifier.toUpperCase();
      if (!this.match("colon")) {
        return { type: "reference", reference: start };
      }

      const end = this.consume("identifier").value;
      if (!isCellReference(end)) {
        throw new FormulaSyntaxError("Invalid range reference");
      }

      return { type: "range", start, end: end.toUpperCase() };
    }

    parseFunctionCall(name) {
      const argumentsList = [];

      if (!this.match("rightParen")) {
        do {
          argumentsList.push(this.parseComparison());
        } while (this.match("comma"));
        this.consume("rightParen");
      }

      return {
        type: "function",
        name: name.toUpperCase(),
        arguments: argumentsList
      };
    }
  }

  function parseFormula(input) {
    if (typeof input !== "string" || !input.startsWith("=")) {
      throw new FormulaSyntaxError("Formulas must begin with =");
    }

    const source = input.slice(1);
    if (!source.trim()) {
      throw new FormulaSyntaxError("Formula is empty");
    }

    return new Parser(tokenize(source)).parse();
  }

  function columnIndex(label) {
    return label.split("").reduce(
      (index, character) => (index * 26) + character.charCodeAt(0) - 64,
      0
    ) - 1;
  }

  function columnLabel(index) {
    let label = "";
    let remaining = index + 1;

    while (remaining > 0) {
      const offset = (remaining - 1) % 26;
      label = String.fromCharCode(65 + offset) + label;
      remaining = Math.floor((remaining - 1) / 26);
    }

    return label;
  }

  function parseReference(reference) {
    const match = String(reference).toUpperCase().match(/^([A-Z]+)([1-9]\d*)$/);
    if (!match) throw new FormulaSyntaxError("Invalid cell reference");

    return {
      column: columnIndex(match[1]),
      row: Number(match[2]) - 1
    };
  }

  function expandRange(start, end) {
    const first = parseReference(start);
    const last = parseReference(end);
    const references = [];
    const firstRow = Math.min(first.row, last.row);
    const lastRow = Math.max(first.row, last.row);
    const firstColumn = Math.min(first.column, last.column);
    const lastColumn = Math.max(first.column, last.column);

    for (let row = firstRow; row <= lastRow; row += 1) {
      for (let column = firstColumn; column <= lastColumn; column += 1) {
        references.push(`${columnLabel(column)}${row + 1}`);
      }
    }

    return references;
  }

  function rangeDimensions(start, end) {
    const first = parseReference(start);
    const last = parseReference(end);
    return {
      rows: Math.abs(last.row - first.row) + 1,
      columns: Math.abs(last.column - first.column) + 1
    };
  }

  function isArrayValue(value) {
    return Boolean(value && value.kind === "array");
  }

  function makeArray(rows, columns, values, options = {}) {
    if (!Number.isInteger(rows) || rows < 1 || !Number.isInteger(columns) || columns < 1) {
      throw new FormulaError(ERROR_VALUES.VALUE);
    }
    if (!Array.isArray(values) || values.length !== rows
      || values.some((row) => !Array.isArray(row) || row.length !== columns)) {
      throw new FormulaError(ERROR_VALUES.VALUE);
    }

    const array = {
      kind: "array",
      rows,
      columns,
      values: values.map((row) => row.slice())
    };
    if (options.formats) array.formats = options.formats.map((row) => row.slice());
    if (options.references) array.references = options.references.map((row) => row.slice());
    return array;
  }

  function arrayFromRange(start, end, context) {
    const { rows, columns } = rangeDimensions(start, end);
    const references = expandRange(start, end);
    const legacyValues = context.getCellValue
      ? null
      : context.getRangeValues?.(start, end);
    const values = [];
    const formats = [];
    const sourceReferences = [];

    for (let row = 0; row < rows; row += 1) {
      const valueRow = [];
      const formatRow = [];
      const referenceRow = [];
      for (let column = 0; column < columns; column += 1) {
        const reference = references[(row * columns) + column];
        valueRow.push(context.getCellValue
          ? context.getCellValue(reference)
          : (legacyValues?.[(row * columns) + column] ?? ""));
        formatRow.push(context.getCellNumberFormat?.(reference) || "General");
        referenceRow.push(reference);
      }
      values.push(valueRow);
      formats.push(formatRow);
      sourceReferences.push(referenceRow);
    }

    return makeArray(rows, columns, values, { formats, references: sourceReferences });
  }

  function arrayValues(value) {
    if (!isArrayValue(value)) return [value];
    const flattened = [];
    value.values.forEach((row) => row.forEach((entry) => flattened.push(entry)));
    return flattened;
  }

  function mapArrayValues(value, callback) {
    if (!isArrayValue(value)) return callback(value, 0, 0);
    return makeArray(
      value.rows,
      value.columns,
      value.values.map((row, rowIndex) => row.map(
        (entry, columnIndex) => callback(entry, rowIndex, columnIndex)
      )),
      { formats: value.formats, references: value.references }
    );
  }

  function mapArrayPair(left, right, callback) {
    if (!isArrayValue(left) && !isArrayValue(right)) return callback(left, right, 0, 0);
    if (isArrayValue(left) && isArrayValue(right)
      && (left.rows !== right.rows || left.columns !== right.columns)) {
      throw new FormulaError(ERROR_VALUES.VALUE);
    }
    const template = isArrayValue(left) ? left : right;
    const values = template.values.map((row, rowIndex) => row.map((_, columnIndex) => callback(
      isArrayValue(left) ? left.values[rowIndex][columnIndex] : left,
      isArrayValue(right) ? right.values[rowIndex][columnIndex] : right,
      rowIndex,
      columnIndex
    )));
    return makeArray(template.rows, template.columns, values);
  }

  function isErrorValue(value) {
    return typeof value === "string" && value.startsWith("#");
  }

  function isKnownErrorValue(value) {
    return Object.values(ERROR_VALUES).includes(value);
  }

  function requireNumber(value) {
    if (isErrorValue(value)) throw new FormulaError(value);
    if (typeof value === "number" && Number.isFinite(value)) return value;
    if (value === "" || value === null || value === undefined) return 0;

    if (typeof value === "string" && value.trim() && Number.isFinite(Number(value))) {
      return Number(value);
    }

    throw new FormulaError(ERROR_VALUES.GENERIC);
  }

  function requireLogical(value) {
    if (isErrorValue(value)) throw new FormulaError(value);
    if (typeof value === "boolean") return value;
    if (typeof value === "number" && Number.isFinite(value)) return value !== 0;
    if (value === "" || value === null || value === undefined) return false;
    throw new FormulaError(ERROR_VALUES.GENERIC);
  }

  function compareValues(left, right, operator) {
    if (isErrorValue(left)) throw new FormulaError(left);
    if (isErrorValue(right)) throw new FormulaError(right);
    if (isArrayValue(left) || isArrayValue(right)) {
      throw new FormulaError(ERROR_VALUES.GENERIC);
    }

    let normalizedLeft = left;
    let normalizedRight = right;

    if ((left === "" || left === null || left === undefined) && typeof right === "number") {
      normalizedLeft = 0;
    }
    if ((right === "" || right === null || right === undefined) && typeof left === "number") {
      normalizedRight = 0;
    }
    if (typeof normalizedLeft === "string" && typeof normalizedRight === "string") {
      normalizedLeft = normalizedLeft.toLocaleLowerCase();
      normalizedRight = normalizedRight.toLocaleLowerCase();
    }

    const sameType = typeof normalizedLeft === typeof normalizedRight;
    if (operator === "=") return sameType && normalizedLeft === normalizedRight;
    if (operator === "<>") return !sameType || normalizedLeft !== normalizedRight;
    if (!sameType) throw new FormulaError(ERROR_VALUES.GENERIC);

    if (operator === ">") return normalizedLeft > normalizedRight;
    if (operator === "<") return normalizedLeft < normalizedRight;
    if (operator === ">=") return normalizedLeft >= normalizedRight;
    if (operator === "<=") return normalizedLeft <= normalizedRight;
    throw new FormulaError(ERROR_VALUES.GENERIC);
  }

  function cleanNumericResult(value) {
    if (!Number.isFinite(value) || Number.isInteger(value)) return value;
    return Number(value.toPrecision(15));
  }

  function flattenFunctionValues(nodes, context) {
    const values = [];

    nodes.forEach((node) => {
      const result = evaluateNode(node, context);
      if (isArrayValue(result)) {
        arrayValues(result).forEach((value) => values.push(value));
      } else {
        values.push(result);
      }
    });

    return values;
  }

  function numericFunctionValues(nodes, context) {
    return flattenFunctionValues(nodes, context).filter((value) => {
      if (isErrorValue(value)) throw new FormulaError(value);
      return typeof value === "number" && Number.isFinite(value);
    });
  }

  function textValue(value) {
    if (isKnownErrorValue(value)) throw new FormulaError(value);
    if (isArrayValue(value)) throw new FormulaError(ERROR_VALUES.VALUE);
    if (value === "" || value === null || value === undefined) return "";
    if (typeof value === "boolean") return value ? "TRUE" : "FALSE";
    if (typeof value === "number" && Number.isFinite(value)) return String(value);
    if (typeof value === "string") return value;
    throw new FormulaError(ERROR_VALUES.VALUE);
  }

  function requireArity(nodes, minimum, maximum = minimum) {
    if (nodes.length < minimum || nodes.length > maximum) {
      throw new FormulaError(ERROR_VALUES.VALUE);
    }
  }

  function textArgument(node, context) {
    return textValue(evaluateNode(node, context));
  }

  function textInteger(node, context) {
    const value = evaluateNode(node, context);
    if (isKnownErrorValue(value)) throw new FormulaError(value);
    if (isArrayValue(value) || typeof value === "boolean") {
      throw new FormulaError(ERROR_VALUES.VALUE);
    }
    const number = typeof value === "number" ? value : Number(value);
    if (!Number.isFinite(number) || !Number.isInteger(number)) {
      throw new FormulaError(ERROR_VALUES.VALUE);
    }
    return number;
  }

  function textLogical(node, context) {
    const value = evaluateNode(node, context);
    if (isKnownErrorValue(value)) throw new FormulaError(value);
    if (typeof value === "boolean") return value;
    if (typeof value === "number" && Number.isFinite(value)) return value !== 0;
    if (value === "" || value === null || value === undefined) return false;
    throw new FormulaError(ERROR_VALUES.VALUE);
  }

  function textCharacters(value) {
    return Array.from(value);
  }

  function characterEntries(text, selectedStart = 0, selectedCount = 0) {
    return textCharacters(text).map((character, index) => ({
      character,
      position: index + 1,
      selected: index >= selectedStart && index < selectedStart + selectedCount
    }));
  }

  function textPiecesForNode(node, context) {
    if (node.type === "reference") {
      const value = context.getCellValue(node.reference);
      return [{
        sourceType: "reference",
        source: node.reference,
        reference: node.reference,
        value,
        text: textValue(value)
      }];
    }

    if (node.type === "range") {
      return expandRange(node.start, node.end).map((reference) => {
        const value = context.getCellValue(reference);
        return {
          sourceType: "reference",
          source: reference,
          reference,
          value,
          text: textValue(value)
        };
      });
    }

    const value = evaluateNode(node, context);
    if (isArrayValue(value)) {
      return arrayValues(value).map((entry) => ({
        sourceType: "expression",
        source: null,
        reference: null,
        value: entry,
        text: textValue(entry)
      }));
    }

    return [{
      sourceType: node.type === "string" ? "literal" : "expression",
      source: node.type === "string" ? node.value : null,
      reference: null,
      value,
      text: textValue(value)
    }];
  }

  function flattenedTextPieces(nodes, context) {
    return nodes.flatMap((node) => textPiecesForNode(node, context));
  }

  function findTextPosition(findText, withinText, startNumber, caseSensitive) {
    const source = textCharacters(caseSensitive ? withinText : withinText.toLowerCase());
    const query = textCharacters(caseSensitive ? findText : findText.toLowerCase());
    const startIndex = startNumber - 1;

    for (let index = startIndex; index <= source.length - query.length; index += 1) {
      if (query.every((character, offset) => source[index + offset] === character)) {
        return index + 1;
      }
    }
    return null;
  }

  function replaceOccurrence(text, oldText, newText, instanceNumber) {
    if (oldText === "") return text;
    if (instanceNumber === null) return text.split(oldText).join(newText);

    let fromIndex = 0;
    let occurrence = 0;
    while (fromIndex <= text.length) {
      const index = text.indexOf(oldText, fromIndex);
      if (index < 0) return text;
      occurrence += 1;
      if (occurrence === instanceNumber) {
        return text.slice(0, index) + newText + text.slice(index + oldText.length);
      }
      fromIndex = index + oldText.length;
    }
    return text;
  }

  function resolveRangeArgument(node, context) {
    let start;
    let end;

    if (node.type === "reference") {
      start = node.reference;
      end = node.reference;
    } else if (node.type === "range") {
      start = node.start;
      end = node.end;
    } else {
      throw new FormulaError(ERROR_VALUES.VALUE);
    }

    const first = parseReference(start);
    const last = parseReference(end);
    const references = expandRange(start, end);
    const rowCount = Math.abs(last.row - first.row) + 1;
    const columnCount = Math.abs(last.column - first.column) + 1;
    return {
      label: start === end ? start : `${start}:${end}`,
      start,
      end,
      rowCount,
      columnCount,
      orientation: rowCount === 1 && columnCount === 1
        ? "scalar"
        : (rowCount === 1 ? "row" : (columnCount === 1 ? "column" : "matrix")),
      cells: references.map((reference, index) => ({
        index,
        rowIndex: Math.floor(index / columnCount),
        columnIndex: index % columnCount,
        reference,
        value: context.getCellValue(reference)
      }))
    };
  }

  function requireMatchingDimensions(expected, actual) {
    if (
      expected.rowCount !== actual.rowCount
      || expected.columnCount !== actual.columnCount
    ) {
      throw new FormulaError(ERROR_VALUES.VALUE);
    }
  }

  function criterionFromNode(node, context) {
    const rawValue = evaluateNode(node, context);
    if (isKnownErrorValue(rawValue)) throw new FormulaError(rawValue);
    if (isArrayValue(rawValue)) throw new FormulaError(ERROR_VALUES.VALUE);

    try {
      return CriteriaEngine.compile(rawValue);
    } catch (error) {
      throw new FormulaError(ERROR_VALUES.VALUE);
    }
  }

  function conditionalArguments(name, nodes, context) {
    let aggregateRange = null;
    let criteriaNodes;

    if (name === "COUNTIF") {
      if (nodes.length !== 2) throw new FormulaError(ERROR_VALUES.VALUE);
      criteriaNodes = [[nodes[0], nodes[1]]];
    } else if (name === "SUMIF" || name === "AVERAGEIF") {
      if (nodes.length !== 2 && nodes.length !== 3) {
        throw new FormulaError(ERROR_VALUES.VALUE);
      }
      criteriaNodes = [[nodes[0], nodes[1]]];
      aggregateRange = resolveRangeArgument(nodes[2] || nodes[0], context);
    } else if (name === "COUNTIFS") {
      if (nodes.length < 2 || nodes.length % 2 !== 0) {
        throw new FormulaError(ERROR_VALUES.VALUE);
      }
      criteriaNodes = [];
      for (let index = 0; index < nodes.length; index += 2) {
        criteriaNodes.push([nodes[index], nodes[index + 1]]);
      }
    } else {
      if (nodes.length < 3 || nodes.length % 2 !== 1) {
        throw new FormulaError(ERROR_VALUES.VALUE);
      }
      aggregateRange = resolveRangeArgument(nodes[0], context);
      criteriaNodes = [];
      for (let index = 1; index < nodes.length; index += 2) {
        criteriaNodes.push([nodes[index], nodes[index + 1]]);
      }
    }

    const criteria = criteriaNodes.map(([rangeNode, criterionNode]) => ({
      range: resolveRangeArgument(rangeNode, context),
      criterion: criterionFromNode(criterionNode, context)
    }));
    const dimensions = aggregateRange || criteria[0].range;
    criteria.forEach((entry) => requireMatchingDimensions(dimensions, entry.range));
    if (aggregateRange) requireMatchingDimensions(dimensions, aggregateRange);
    return { aggregateRange, criteria, dimensions };
  }

  const CONDITIONAL_FUNCTIONS = new Set([
    "COUNTIF",
    "SUMIF",
    "AVERAGEIF",
    "COUNTIFS",
    "SUMIFS",
    "AVERAGEIFS"
  ]);

  function runConditionalAggregate(name, nodes, context) {
    const { aggregateRange, criteria, dimensions } = conditionalArguments(name, nodes, context);
    const positions = [];
    const includedValues = [];
    let matches = 0;

    for (let index = 0; index < dimensions.cells.length; index += 1) {
      const checks = criteria.map((entry) => {
        const cell = entry.range.cells[index];
        if (isKnownErrorValue(cell.value)) throw new FormulaError(cell.value);
        return {
          reference: cell.reference,
          value: cell.value,
          matched: CriteriaEngine.matches(cell.value, entry.criterion)
        };
      });
      const allMatched = checks.every((check) => check.matched);
      const aggregateCell = aggregateRange?.cells[index] || null;
      const aggregate = aggregateCell ? {
        reference: aggregateCell.reference,
        value: aggregateCell.value,
        numeric: typeof aggregateCell.value === "number" && Number.isFinite(aggregateCell.value),
        included: false
      } : null;

      if (allMatched) {
        matches += 1;
        if (aggregate) {
          if (isKnownErrorValue(aggregate.value)) throw new FormulaError(aggregate.value);
          if (aggregate.numeric) {
            aggregate.included = true;
            includedValues.push({ reference: aggregate.reference, value: aggregate.value });
          }
        }
      }

      positions.push({ index, checks, allMatched, aggregate });
    }

    const sum = includedValues.reduce((total, entry) => total + entry.value, 0);
    let result;

    if (name === "COUNTIF" || name === "COUNTIFS") {
      result = matches;
    } else if (name === "SUMIF" || name === "SUMIFS") {
      result = sum;
    } else {
      if (!includedValues.length) throw new FormulaError(ERROR_VALUES.DIV_ZERO);
      result = sum / includedValues.length;
    }

    return {
      kind: "conditionalAggregate",
      functionName: name,
      criteria: criteria.map((entry) => ({
        range: entry.range,
        criterion: {
          rawValue: entry.criterion.rawValue,
          display: entry.criterion.display,
          operator: entry.criterion.operator,
          operand: entry.criterion.operand,
          usesWildcards: Boolean(entry.criterion.wildcard)
        }
      })),
      aggregateRange,
      positions,
      includedValues,
      summary: {
        matches,
        includedCount: includedValues.length,
        sum,
        result
      },
      result
    };
  }

  function scalarValue(node, context) {
    const value = evaluateNode(node, context);
    if (isKnownErrorValue(value)) throw new FormulaError(value);
    if (isArrayValue(value)) throw new FormulaError(ERROR_VALUES.VALUE);
    return value;
  }

  function integerValue(node, context) {
    const value = scalarValue(node, context);
    if (typeof value !== "number" || !Number.isInteger(value)) {
      throw new FormulaError(ERROR_VALUES.VALUE);
    }
    return value;
  }

  function requireVector(range) {
    if (range.rowCount !== 1 && range.columnCount !== 1) {
      throw new FormulaError(ERROR_VALUES.VALUE);
    }
    return range.cells.map((cell, index) => ({ ...cell, index }));
  }

  function rangeCell(range, rowIndex, columnIndex) {
    return range.cells[(rowIndex * range.columnCount) + columnIndex];
  }

  function finalizeLookupTrace(trace) {
    if (isKnownErrorValue(trace.result)) throw new FormulaError(trace.result);
    return trace.result;
  }

  function runTableLookup(name, nodes, context) {
    if (nodes.length !== 3 && nodes.length !== 4) {
      throw new FormulaError(ERROR_VALUES.VALUE);
    }

    const lookupValue = scalarValue(nodes[0], context);
    const table = resolveRangeArgument(nodes[1], context);
    const returnIndex = integerValue(nodes[2], context);
    const vertical = name === "VLOOKUP";
    const limit = vertical ? table.columnCount : table.rowCount;
    const exact = nodes.length === 4 ? !requireLogical(scalarValue(nodes[3], context)) : false;

    if (returnIndex < 1 || returnIndex > limit) {
      return {
        kind: "tableLookup",
        functionName: name,
        lookupValue,
        table,
        orientation: vertical ? "vertical" : "horizontal",
        returnIndex,
        search: null,
        matchedBand: [],
        returnCell: null,
        result: ERROR_VALUES.REF,
        error: ERROR_VALUES.REF
      };
    }

    const lookupLane = [];
    const laneLength = vertical ? table.rowCount : table.columnCount;
    for (let index = 0; index < laneLength; index += 1) {
      const cell = vertical ? rangeCell(table, index, 0) : rangeCell(table, 0, index);
      if (isKnownErrorValue(cell.value)) throw new FormulaError(cell.value);
      lookupLane.push({ ...cell, index });
    }

    const search = LookupEngine.search(lookupLane, lookupValue, {
      matchMode: exact ? 0 : -1,
      searchMode: 1
    });
    const selectedIndex = search.selected?.index;
    const returnCell = selectedIndex === undefined ? null : (
      vertical
        ? rangeCell(table, selectedIndex, returnIndex - 1)
        : rangeCell(table, returnIndex - 1, selectedIndex)
    );
    const matchedBand = selectedIndex === undefined ? [] : table.cells.filter((cell) => (
      vertical ? cell.rowIndex === selectedIndex : cell.columnIndex === selectedIndex
    ));
    const result = returnCell ? returnCell.value : ERROR_VALUES.NA;

    return {
      kind: "tableLookup",
      functionName: name,
      lookupValue,
      table,
      orientation: vertical ? "vertical" : "horizontal",
      lookupLane,
      returnIndex,
      exact,
      search,
      matchedBand,
      returnCell,
      result,
      error: isKnownErrorValue(result) ? result : null
    };
  }

  function runXlookup(nodes, context) {
    if (nodes.length < 3 || nodes.length > 6) throw new FormulaError(ERROR_VALUES.VALUE);
    const lookupValue = scalarValue(nodes[0], context);
    const lookupRange = resolveRangeArgument(nodes[1], context);
    const returnRange = resolveRangeArgument(nodes[2], context);
    const lookupEntries = requireVector(lookupRange);
    const returnEntries = requireVector(returnRange);
    if (lookupEntries.length !== returnEntries.length) throw new FormulaError(ERROR_VALUES.VALUE);

    const matchMode = nodes[4] ? integerValue(nodes[4], context) : 0;
    const searchMode = nodes[5] ? integerValue(nodes[5], context) : 1;
    if (![0, -1, 1, 2].includes(matchMode) || ![1, -1].includes(searchMode)) {
      throw new FormulaError(ERROR_VALUES.VALUE);
    }

    lookupEntries.forEach((entry) => {
      if (isKnownErrorValue(entry.value)) throw new FormulaError(entry.value);
    });
    const search = LookupEngine.search(lookupEntries, lookupValue, { matchMode, searchMode });
    const selectedIndex = search.selected?.index;
    let returnCell = null;
    let result;
    let fallbackUsed = false;

    if (selectedIndex === undefined) {
      fallbackUsed = nodes.length >= 4;
      result = fallbackUsed ? scalarValue(nodes[3], context) : ERROR_VALUES.NA;
    } else {
      returnCell = returnEntries[selectedIndex];
      result = returnCell.value;
    }

    return {
      kind: "xlookup",
      functionName: "XLOOKUP",
      lookupValue,
      lookupRange,
      returnRange,
      matchMode,
      searchMode,
      search,
      returnCell,
      fallback: { provided: nodes.length >= 4, used: fallbackUsed },
      result,
      error: isKnownErrorValue(result) ? result : null
    };
  }

  function runMatch(nodes, context) {
    if (nodes.length !== 2 && nodes.length !== 3) throw new FormulaError(ERROR_VALUES.VALUE);
    const lookupValue = scalarValue(nodes[0], context);
    const lookupRange = resolveRangeArgument(nodes[1], context);
    const entries = requireVector(lookupRange);
    const matchType = nodes[2] ? integerValue(nodes[2], context) : 1;
    if (![0, 1, -1].includes(matchType)) throw new FormulaError(ERROR_VALUES.VALUE);
    entries.forEach((entry) => {
      if (isKnownErrorValue(entry.value)) throw new FormulaError(entry.value);
    });
    const search = LookupEngine.search(entries, lookupValue, {
      matchMode: matchType === 0 ? 0 : (matchType === 1 ? -1 : 1),
      searchMode: 1
    });
    const result = search.selected ? search.selected.index + 1 : ERROR_VALUES.NA;
    return {
      kind: "match",
      functionName: "MATCH",
      lookupValue,
      lookupRange,
      matchType,
      search,
      resultPosition: typeof result === "number" ? result : null,
      result,
      error: isKnownErrorValue(result) ? result : null
    };
  }

  function runIndex(nodes, context) {
    if (nodes.length !== 2 && nodes.length !== 3) throw new FormulaError(ERROR_VALUES.VALUE);
    const array = resolveRangeArgument(nodes[0], context);
    const firstIndex = integerValue(nodes[1], context);
    let rowNumber;
    let columnNumber;

    if (array.rowCount === 1 && array.columnCount > 1 && nodes.length === 2) {
      rowNumber = 1;
      columnNumber = firstIndex;
    } else {
      rowNumber = firstIndex;
      columnNumber = nodes[2] ? integerValue(nodes[2], context) : 1;
    }

    const inRange = rowNumber >= 1 && rowNumber <= array.rowCount
      && columnNumber >= 1 && columnNumber <= array.columnCount;
    const selectedCell = inRange ? rangeCell(array, rowNumber - 1, columnNumber - 1) : null;
    const result = selectedCell ? selectedCell.value : ERROR_VALUES.REF;
    return {
      kind: "index",
      functionName: "INDEX",
      array,
      requested: { row: rowNumber, column: columnNumber },
      selectedCell,
      children: [],
      result,
      error: isKnownErrorValue(result) ? result : null
    };
  }

  const LOOKUP_FUNCTIONS = new Set(["VLOOKUP", "HLOOKUP", "XLOOKUP", "MATCH", "INDEX"]);

  function runLookupFunction(name, nodes, context) {
    if (name === "VLOOKUP" || name === "HLOOKUP") return runTableLookup(name, nodes, context);
    if (name === "XLOOKUP") return runXlookup(nodes, context);
    if (name === "MATCH") return runMatch(nodes, context);
    if (name === "INDEX") return runIndex(nodes, context);
    throw new FormulaError(ERROR_VALUES.NAME);
  }

  const TEXT_FUNCTIONS = new Set([
    "LEN",
    "LEFT",
    "RIGHT",
    "MID",
    "TRIM",
    "UPPER",
    "LOWER",
    "PROPER",
    "CONCAT",
    "TEXTJOIN",
    "FIND",
    "SEARCH",
    "SUBSTITUTE",
    "REPLACE"
  ]);

  function runTextFunction(name, nodes, context) {
    if (name === "LEN") {
      requireArity(nodes, 1);
      const text = textArgument(nodes[0], context);
      const result = textCharacters(text).length;
      return {
        kind: "characters",
        functionName: name,
        text,
        characters: characterEntries(text),
        count: result,
        result
      };
    }

    if (name === "LEFT" || name === "RIGHT") {
      requireArity(nodes, 1, 2);
      const text = textArgument(nodes[0], context);
      const count = nodes[1] ? textInteger(nodes[1], context) : 1;
      if (count < 0) throw new FormulaError(ERROR_VALUES.VALUE);
      const characters = textCharacters(text);
      const selectedCount = Math.min(count, characters.length);
      const selectedStart = name === "LEFT" ? 0 : characters.length - selectedCount;
      const result = characters.slice(selectedStart, selectedStart + selectedCount).join("");
      return {
        kind: "characters",
        functionName: name,
        text,
        characters: characterEntries(text, selectedStart, selectedCount),
        start: selectedStart + 1,
        count,
        selectedText: result,
        direction: name === "LEFT" ? "start" : "end",
        result
      };
    }

    if (name === "MID") {
      requireArity(nodes, 3);
      const text = textArgument(nodes[0], context);
      const start = textInteger(nodes[1], context);
      const count = textInteger(nodes[2], context);
      if (start <= 0 || count < 0) throw new FormulaError(ERROR_VALUES.VALUE);
      const characters = textCharacters(text);
      const selectedStart = Math.min(start - 1, characters.length);
      const selectedCount = Math.min(count, Math.max(0, characters.length - selectedStart));
      const result = characters.slice(selectedStart, selectedStart + selectedCount).join("");
      return {
        kind: "characters",
        functionName: name,
        text,
        characters: characterEntries(text, selectedStart, selectedCount),
        start,
        count,
        selectedText: result,
        direction: "position",
        result
      };
    }

    if (name === "TRIM") {
      requireArity(nodes, 1);
      const before = textArgument(nodes[0], context);
      const leading = /^ +/.test(before);
      const trailing = / +$/.test(before);
      const interior = /\S {2,}\S/.test(before);
      const result = before.replace(/^ +| +$/g, "").replace(/ +/g, " ");
      return {
        kind: "trim",
        functionName: name,
        before,
        after: result,
        changes: { leading, interior, trailing },
        result
      };
    }

    if (name === "UPPER" || name === "LOWER" || name === "PROPER") {
      requireArity(nodes, 1);
      const before = textArgument(nodes[0], context);
      let result;
      if (name === "UPPER") result = before.toUpperCase();
      if (name === "LOWER") result = before.toLowerCase();
      if (name === "PROPER") {
        result = before.toLowerCase().replace(
          /(^|[^\p{L}])(\p{L})/gu,
          (_, prefix, character) => prefix + character.toUpperCase()
        );
      }
      return {
        kind: "case",
        functionName: name,
        before,
        transformation: name,
        after: result,
        result
      };
    }

    if (name === "CONCAT") {
      requireArity(nodes, 1, Number.POSITIVE_INFINITY);
      const pieces = flattenedTextPieces(nodes, context).map((piece) => ({
        ...piece,
        included: true
      }));
      const result = pieces.map((piece) => piece.text).join("");
      return { kind: "concat", functionName: name, pieces, result };
    }

    if (name === "TEXTJOIN") {
      requireArity(nodes, 3, Number.POSITIVE_INFINITY);
      const delimiter = textArgument(nodes[0], context);
      const ignoreEmpty = textLogical(nodes[1], context);
      const pieces = flattenedTextPieces(nodes.slice(2), context).map((piece) => ({
        ...piece,
        included: !ignoreEmpty || piece.text !== ""
      }));
      const result = pieces
        .filter((piece) => piece.included)
        .map((piece) => piece.text)
        .join(delimiter);
      return {
        kind: "textjoin",
        functionName: name,
        delimiter,
        ignoreEmpty,
        pieces,
        result
      };
    }

    if (name === "FIND" || name === "SEARCH") {
      requireArity(nodes, 2, 3);
      const findText = textArgument(nodes[0], context);
      const withinText = textArgument(nodes[1], context);
      const start = nodes[2] ? textInteger(nodes[2], context) : 1;
      if (start <= 0 || start > textCharacters(withinText).length) {
        throw new FormulaError(ERROR_VALUES.VALUE);
      }
      const caseSensitive = name === "FIND";
      const position = findTextPosition(findText, withinText, start, caseSensitive);
      const result = position ?? ERROR_VALUES.VALUE;
      return {
        kind: "search",
        functionName: name,
        findText,
        withinText,
        start,
        caseSensitive,
        matchPosition: position,
        characters: characterEntries(
          withinText,
          position ? position - 1 : 0,
          position ? textCharacters(findText).length : 0
        ),
        result,
        error: position ? null : ERROR_VALUES.VALUE
      };
    }

    if (name === "SUBSTITUTE") {
      requireArity(nodes, 3, 4);
      const text = textArgument(nodes[0], context);
      const oldText = textArgument(nodes[1], context);
      const newText = textArgument(nodes[2], context);
      const instance = nodes[3] ? textInteger(nodes[3], context) : null;
      if (instance !== null && instance <= 0) throw new FormulaError(ERROR_VALUES.VALUE);
      const result = replaceOccurrence(text, oldText, newText, instance);
      return {
        kind: "substitute",
        functionName: name,
        before: text,
        oldText,
        newText,
        instance,
        after: result,
        result
      };
    }

    if (name === "REPLACE") {
      requireArity(nodes, 4);
      const text = textArgument(nodes[0], context);
      const start = textInteger(nodes[1], context);
      const count = textInteger(nodes[2], context);
      const newText = textArgument(nodes[3], context);
      if (start <= 0 || count < 0) throw new FormulaError(ERROR_VALUES.VALUE);
      const characters = textCharacters(text);
      const startIndex = Math.min(start - 1, characters.length);
      const replacedCount = Math.min(count, Math.max(0, characters.length - startIndex));
      const result = characters.slice(0, startIndex).join("")
        + newText
        + characters.slice(startIndex + count).join("");
      return {
        kind: "replace",
        functionName: name,
        before: text,
        characters: characterEntries(text, startIndex, replacedCount),
        start,
        count,
        replacedCount,
        newText,
        after: result,
        result
      };
    }

    throw new FormulaError(ERROR_VALUES.NAME);
  }

  function finalizeTextTrace(trace) {
    if (isKnownErrorValue(trace.result)) throw new FormulaError(trace.result);
    return trace.result;
  }

  const DATE_FUNCTIONS = new Set([
    "DATE",
    "YEAR",
    "MONTH",
    "DAY",
    "TODAY",
    "DAYS",
    "EDATE",
    "EOMONTH",
    "WEEKDAY"
  ]);
  const DATE_RESULT_FUNCTIONS = new Set(["DATE", "TODAY", "EDATE", "EOMONTH"]);
  const VOLATILE_FUNCTIONS = new Set(["TODAY"]);

  function dateSerialArgument(node, context) {
    const value = evaluateNode(node, context);
    if (isKnownErrorValue(value)) throw new FormulaError(value);
    if (isArrayValue(value) || typeof value === "boolean") {
      throw new FormulaError(ERROR_VALUES.VALUE);
    }
    const serial = typeof value === "number" ? value : Number(value);
    if (!Number.isFinite(serial)) throw new FormulaError(ERROR_VALUES.VALUE);
    if (!ExcelFormatting.serialToCalendar(serial)) throw new FormulaError(ERROR_VALUES.NUM);
    return serial;
  }

  function dateResultTrace(functionName, serial, extra = {}) {
    const calendar = ExcelFormatting.serialToCalendar(serial);
    if (!calendar) throw new FormulaError(ERROR_VALUES.NUM);
    return {
      functionName,
      result: serial,
      resultFormat: ExcelFormatting.NUMBER_FORMATS.DATE,
      resultDate: ExcelFormatting.formatDateSerial(serial),
      calendar,
      ...extra
    };
  }

  function runDateFunction(name, nodes, context) {
    if (name === "DATE") {
      requireArity(nodes, 3);
      const requestedYear = textInteger(nodes[0], context);
      const year = requestedYear >= 0 && requestedYear <= 1899
        ? requestedYear + 1900
        : requestedYear;
      const month = textInteger(nodes[1], context);
      const day = textInteger(nodes[2], context);
      const serial = ExcelFormatting.calendarToSerial(year, month, day);
      if (serial === null) throw new FormulaError(ERROR_VALUES.NUM);
      return dateResultTrace(name, serial, {
        kind: "date-construction",
        requested: { year, month, day },
        requestedYear
      });
    }

    if (name === "TODAY") {
      requireArity(nodes, 0);
      const serial = context.getCurrentDateSerial
        ? context.getCurrentDateSerial()
        : ExcelFormatting.todaySerial();
      if (serial === null) throw new FormulaError(ERROR_VALUES.NUM);
      return dateResultTrace(name, serial, { kind: "today" });
    }

    if (name === "YEAR" || name === "MONTH" || name === "DAY") {
      requireArity(nodes, 1);
      const serial = dateSerialArgument(nodes[0], context);
      const calendar = ExcelFormatting.serialToCalendar(serial);
      const key = name.toLowerCase();
      return {
        kind: "date-component",
        functionName: name,
        sourceSerial: serial,
        sourceDate: ExcelFormatting.formatDateSerial(serial),
        sourceReference: nodes[0].type === "reference" ? nodes[0].reference : null,
        component: name[0] + name.slice(1).toLowerCase(),
        result: calendar[key],
        resultFormat: ExcelFormatting.NUMBER_FORMATS.GENERAL
      };
    }

    if (name === "DAYS") {
      requireArity(nodes, 2);
      const endSerial = dateSerialArgument(nodes[0], context);
      const startSerial = dateSerialArgument(nodes[1], context);
      return {
        kind: "date-difference",
        functionName: name,
        startSerial,
        startDate: ExcelFormatting.formatDateSerial(startSerial),
        endSerial,
        endDate: ExcelFormatting.formatDateSerial(endSerial),
        difference: Math.floor(endSerial) - Math.floor(startSerial),
        result: Math.floor(endSerial) - Math.floor(startSerial),
        resultFormat: ExcelFormatting.NUMBER_FORMATS.GENERAL
      };
    }

    if (name === "EDATE" || name === "EOMONTH") {
      requireArity(nodes, 2);
      const startSerial = dateSerialArgument(nodes[0], context);
      const months = textInteger(nodes[1], context);
      const result = name === "EDATE"
        ? ExcelFormatting.addMonths(startSerial, months)
        : ExcelFormatting.endOfMonth(startSerial, months);
      if (result === null) throw new FormulaError(ERROR_VALUES.NUM);
      return dateResultTrace(name, result, {
        kind: name === "EDATE" ? "month-shift" : "month-end",
        startSerial,
        startDate: ExcelFormatting.formatDateSerial(startSerial),
        months
      });
    }

    if (name === "WEEKDAY") {
      requireArity(nodes, 1, 2);
      const serial = dateSerialArgument(nodes[0], context);
      const returnType = nodes[1] ? textInteger(nodes[1], context) : 1;
      const result = ExcelFormatting.weekday(serial, returnType);
      if (result === null) throw new FormulaError(ERROR_VALUES.NUM);
      return {
        kind: "weekday",
        functionName: name,
        sourceSerial: serial,
        sourceDate: ExcelFormatting.formatDateSerial(serial),
        returnType,
        weekStarts: returnType === 2 ? "Monday = 1" : "Sunday = 1",
        dayName: ExcelFormatting.weekdayName(serial),
        result,
        resultFormat: ExcelFormatting.NUMBER_FORMATS.GENERAL
      };
    }

    throw new FormulaError(ERROR_VALUES.NAME);
  }

  function finalizeDateTrace(trace) {
    if (isKnownErrorValue(trace.result)) throw new FormulaError(trace.result);
    return trace.result;
  }

  const MATH_FUNCTIONS = new Set(["ROUND", "ROUNDUP", "ROUNDDOWN", "INT", "ABS", "MOD"]);
  const ERROR_FUNCTIONS = new Set(["IFERROR", "IFNA"]);

  function numericArgument(node, context) {
    const value = evaluateNode(node, context);
    if (isKnownErrorValue(value)) throw new FormulaError(value);
    if (isArrayValue(value) || typeof value === "boolean") {
      throw new FormulaError(ERROR_VALUES.VALUE);
    }
    const number = typeof value === "number" ? value : Number(value);
    if (!Number.isFinite(number)) throw new FormulaError(ERROR_VALUES.VALUE);
    return number;
  }

  function shiftDecimal(value, places) {
    const [coefficient, exponent = "0"] = String(value).split("e");
    return Number(`${coefficient}e${Number(exponent) + places}`);
  }

  function roundedNumber(number, digits, mode) {
    if (digits > 308) return number;
    if (digits < -308) return 0;
    const sign = number < 0 ? -1 : 1;
    const shifted = shiftDecimal(Math.abs(number), digits);
    if (!Number.isFinite(shifted)) return number;
    const tolerance = Number.EPSILON * Math.max(1, shifted) * 2;
    let rounded;
    if (mode === "nearest") rounded = Math.floor(shifted + 0.5 + tolerance);
    if (mode === "away") rounded = Math.ceil(shifted - tolerance);
    if (mode === "toward") rounded = Math.floor(shifted + tolerance);
    const result = shiftDecimal(rounded * sign, -digits);
    if (!Number.isFinite(result)) throw new FormulaError(ERROR_VALUES.NUM);
    return Object.is(result, -0) ? 0 : result;
  }

  function runMathFunction(name, nodes, context) {
    if (["ROUND", "ROUNDUP", "ROUNDDOWN"].includes(name)) {
      requireArity(nodes, 2);
      const number = numericArgument(nodes[0], context);
      const digits = textInteger(nodes[1], context);
      const modes = {
        ROUND: ["nearest", "Nearest value"],
        ROUNDUP: ["away", "Away from zero"],
        ROUNDDOWN: ["toward", "Toward zero"]
      };
      const [mode, direction] = modes[name];
      const result = roundedNumber(number, digits, mode);
      return {
        kind: "rounding",
        functionName: name,
        number,
        digits,
        direction,
        result
      };
    }

    if (name === "INT" || name === "ABS") {
      requireArity(nodes, 1);
      const number = numericArgument(nodes[0], context);
      const result = name === "INT" ? Math.floor(number) : Math.abs(number);
      return {
        kind: name === "INT" ? "integer" : "absolute",
        functionName: name,
        number,
        direction: name === "INT" ? "Downward to the nearest integer" : "Distance from zero",
        result
      };
    }

    if (name === "MOD") {
      requireArity(nodes, 2);
      const number = numericArgument(nodes[0], context);
      const divisor = numericArgument(nodes[1], context);
      if (divisor === 0) throw new FormulaError(ERROR_VALUES.DIV_ZERO);
      const quotient = Math.floor(number / divisor);
      const result = cleanNumericResult(number - (divisor * quotient));
      return {
        kind: "modulo",
        functionName: name,
        number,
        divisor,
        quotient,
        result: Object.is(result, -0) ? 0 : result
      };
    }

    throw new FormulaError(ERROR_VALUES.NAME);
  }

  function runErrorHandlingFunction(name, nodes, context, options = {}) {
    requireArity(nodes, 2);
    let primaryResult;
    let error = null;
    try {
      primaryResult = evaluateNode(nodes[0], context);
      if (isKnownErrorValue(primaryResult)) error = primaryResult;
    } catch (caught) {
      if (!(caught instanceof FormulaError)) throw caught;
      error = caught.code;
      primaryResult = caught.code;
    }

    const caught = name === "IFERROR"
      ? isKnownErrorValue(error)
      : error === ERROR_VALUES.NA;
    if (error && !caught && !options.traceUncaught) throw new FormulaError(error);
    const fallbackResult = caught ? evaluateNode(nodes[1], context) : null;
    return {
      kind: "error-handling",
      functionName: name,
      primaryNode: nodes[0],
      fallbackNode: nodes[1],
      primaryResult,
      error,
      caught,
      fallbackResult,
      result: caught ? fallbackResult : primaryResult
    };
  }

  function finalizeMathTrace(trace) {
    if (isKnownErrorValue(trace.result)) throw new FormulaError(trace.result);
    return trace.result;
  }

  const DYNAMIC_ARRAY_FUNCTIONS = new Set(["SEQUENCE", "FILTER", "SORT", "SORTBY", "UNIQUE"]);

  function arrayArgument(node, context) {
    const value = evaluateNode(node, context);
    if (isKnownErrorValue(value)) throw new FormulaError(value);
    return isArrayValue(value) ? value : makeArray(1, 1, [[value]]);
  }

  function optionalLogical(node, context, fallback) {
    return node ? requireLogical(scalarValue(node, context)) : fallback;
  }

  function optionalInteger(node, context, fallback) {
    return node ? integerValue(node, context) : fallback;
  }

  function sourceLabel(node) {
    if (node?.type === "reference") return node.reference;
    if (node?.type === "range") return `${node.start}:${node.end}`;
    return null;
  }

  function subsetArrayRows(array, rowIndexes) {
    return makeArray(
      rowIndexes.length,
      array.columns,
      rowIndexes.map((index) => array.values[index]),
      {
        formats: array.formats ? rowIndexes.map((index) => array.formats[index]) : null,
        references: array.references ? rowIndexes.map((index) => array.references[index]) : null
      }
    );
  }

  function subsetArrayColumns(array, columnIndexes) {
    return makeArray(
      array.rows,
      columnIndexes.length,
      array.values.map((row) => columnIndexes.map((index) => row[index])),
      {
        formats: array.formats
          ? array.formats.map((row) => columnIndexes.map((index) => row[index]))
          : null,
        references: array.references
          ? array.references.map((row) => columnIndexes.map((index) => row[index]))
          : null
      }
    );
  }

  function compareSortValues(left, right) {
    if (isKnownErrorValue(left)) throw new FormulaError(left);
    if (isKnownErrorValue(right)) throw new FormulaError(right);
    const leftBlank = left === "" || left === null || left === undefined;
    const rightBlank = right === "" || right === null || right === undefined;
    if (leftBlank || rightBlank) return leftBlank === rightBlank ? 0 : (leftBlank ? 1 : -1);
    if (typeof left === "number" && typeof right === "number") return left - right;
    if (typeof left === "boolean" && typeof right === "boolean") return Number(left) - Number(right);
    return String(left).localeCompare(String(right), undefined, {
      numeric: true,
      sensitivity: "base"
    });
  }

  function stableOrder(length, comparators) {
    return Array.from({ length }, (_, index) => index).sort((leftIndex, rightIndex) => {
      for (const comparator of comparators) {
        const comparison = comparator(leftIndex, rightIndex);
        if (comparison !== 0) return comparison;
      }
      return leftIndex - rightIndex;
    });
  }

  function uniqueKey(values) {
    return values.map((value) => {
      if (typeof value === "string") return `s:${value.toLocaleLowerCase()}`;
      if (typeof value === "number") return `n:${value}`;
      if (typeof value === "boolean") return `b:${value}`;
      if (value === null || value === undefined || value === "") return "blank:";
      return `${typeof value}:${String(value)}`;
    }).join("\u001f");
  }

  function runDynamicArrayFunction(name, nodes, context) {
    if (name === "SEQUENCE") {
      requireArity(nodes, 1, 4);
      const rows = integerValue(nodes[0], context);
      const columns = optionalInteger(nodes[1], context, 1);
      const start = nodes[2] ? numericArgument(nodes[2], context) : 1;
      const step = nodes[3] ? numericArgument(nodes[3], context) : 1;
      if (rows < 1 || columns < 1) throw new FormulaError(ERROR_VALUES.VALUE);
      const values = Array.from({ length: rows }, (_, row) => (
        Array.from({ length: columns }, (_, column) => (
          cleanNumericResult(start + (((row * columns) + column) * step))
        ))
      ));
      const result = makeArray(rows, columns, values);
      return { kind: "sequence", functionName: name, rows, columns, start, step, result };
    }

    if (name === "FILTER") {
      requireArity(nodes, 2, 3);
      const source = arrayArgument(nodes[0], context);
      const include = arrayArgument(nodes[1], context);
      if (include.rows !== source.rows || include.columns !== 1) {
        throw new FormulaError(ERROR_VALUES.VALUE);
      }
      let conditionSource = null;
      if (nodes[1].type === "comparison") {
        const left = evaluateNode(nodes[1].left, context);
        const right = evaluateNode(nodes[1].right, context);
        conditionSource = isArrayValue(left) ? left : (isArrayValue(right) ? right : null);
      }
      const rowEvaluation = include.values.map((row, index) => ({
        index,
        reference: conditionSource?.references?.[index]?.[0]
          || include.references?.[index]?.[0]
          || null,
        value: row[0],
        conditionValue: conditionSource?.values?.[index]?.[0],
        included: requireLogical(row[0])
      }));
      const includedRows = rowEvaluation.filter((entry) => entry.included).map((entry) => entry.index);
      let result;
      if (includedRows.length) {
        result = subsetArrayRows(source, includedRows);
      } else if (nodes[2]) {
        result = evaluateNode(nodes[2], context);
      } else {
        result = ERROR_VALUES.CALC;
      }
      return {
        kind: "filter",
        functionName: name,
        source,
        sourceLabel: sourceLabel(nodes[0]),
        include,
        includeLabel: sourceLabel(nodes[1]),
        conditionNode: nodes[1],
        rowEvaluation,
        includedRows,
        result
      };
    }

    if (name === "SORT") {
      requireArity(nodes, 1, 4);
      const source = arrayArgument(nodes[0], context);
      const sortIndex = optionalInteger(nodes[1], context, 1);
      const sortOrder = optionalInteger(nodes[2], context, 1);
      const byColumn = optionalLogical(nodes[3], context, false);
      const limit = byColumn ? source.rows : source.columns;
      if (sortIndex < 1 || sortIndex > limit || ![1, -1].includes(sortOrder)) {
        throw new FormulaError(ERROR_VALUES.VALUE);
      }
      const itemCount = byColumn ? source.columns : source.rows;
      const order = stableOrder(itemCount, [
        (left, right) => sortOrder * compareSortValues(
          byColumn ? source.values[sortIndex - 1][left] : source.values[left][sortIndex - 1],
          byColumn ? source.values[sortIndex - 1][right] : source.values[right][sortIndex - 1]
        )
      ]);
      const result = byColumn ? subsetArrayColumns(source, order) : subsetArrayRows(source, order);
      const keyReference = byColumn
        ? source.references?.[sortIndex - 1]?.[0]
        : source.references?.[0]?.[sortIndex - 1];
      const keyCoordinates = keyReference ? parseReference(keyReference) : null;
      const keyHeader = keyCoordinates
        ? context.getCellValue(`${columnLabel(keyCoordinates.column)}1`)
        : null;
      return {
        kind: "sort",
        functionName: name,
        source,
        sourceLabel: sourceLabel(nodes[0]),
        sortIndex,
        sortOrder,
        byColumn,
        keyHeader,
        order,
        result
      };
    }

    if (name === "SORTBY") {
      if (![2, 3, 5].includes(nodes.length)) throw new FormulaError(ERROR_VALUES.VALUE);
      const source = arrayArgument(nodes[0], context);
      const keyCount = nodes.length === 5 ? 2 : 1;
      const keys = [];
      for (let index = 0; index < keyCount; index += 1) {
        const nodeIndex = index === 0 ? 1 : 3;
        const byArray = arrayArgument(nodes[nodeIndex], context);
        if (byArray.rows !== source.rows || byArray.columns !== 1) {
          throw new FormulaError(ERROR_VALUES.VALUE);
        }
        const sortOrder = optionalInteger(nodes[nodeIndex + 1], context, 1);
        if (![1, -1].includes(sortOrder)) throw new FormulaError(ERROR_VALUES.VALUE);
        keys.push({
          array: byArray,
          label: sourceLabel(nodes[nodeIndex]),
          sortOrder
        });
      }
      const order = stableOrder(source.rows, keys.map((key) => (
        (left, right) => key.sortOrder * compareSortValues(
          key.array.values[left][0],
          key.array.values[right][0]
        )
      )));
      return {
        kind: "sortby",
        functionName: name,
        source,
        sourceLabel: sourceLabel(nodes[0]),
        keys,
        order,
        result: subsetArrayRows(source, order)
      };
    }

    if (name === "UNIQUE") {
      requireArity(nodes, 1, 3);
      const source = arrayArgument(nodes[0], context);
      const byColumn = optionalLogical(nodes[1], context, false);
      const exactlyOnce = optionalLogical(nodes[2], context, false);
      const items = byColumn
        ? Array.from({ length: source.columns }, (_, column) => (
          source.values.map((row) => row[column])
        ))
        : source.values;
      const keys = items.map(uniqueKey);
      const counts = keys.reduce((map, key) => map.set(key, (map.get(key) || 0) + 1), new Map());
      const seen = new Set();
      const kept = [];
      keys.forEach((key, index) => {
        if (seen.has(key)) return;
        seen.add(key);
        if (!exactlyOnce || counts.get(key) === 1) kept.push(index);
      });
      if (!kept.length) {
        return {
          kind: "unique",
          functionName: name,
          source,
          sourceLabel: sourceLabel(nodes[0]),
          byColumn,
          exactlyOnce,
          counts,
          kept,
          duplicatesRemoved: items.length,
          result: ERROR_VALUES.CALC
        };
      }
      return {
        kind: "unique",
        functionName: name,
        source,
        sourceLabel: sourceLabel(nodes[0]),
        byColumn,
        exactlyOnce,
        counts,
        kept,
        duplicatesRemoved: items.length - kept.length,
        result: byColumn ? subsetArrayColumns(source, kept) : subsetArrayRows(source, kept)
      };
    }

    throw new FormulaError(ERROR_VALUES.NAME);
  }

  function finalizeDynamicArrayTrace(trace) {
    if (isKnownErrorValue(trace.result)) throw new FormulaError(trace.result);
    return trace.result;
  }

  const functions = {
    SUM(nodes, context) {
      return numericFunctionValues(nodes, context).reduce((total, value) => total + value, 0);
    },
    AVERAGE(nodes, context) {
      const values = numericFunctionValues(nodes, context);
      if (!values.length) throw new FormulaError(ERROR_VALUES.DIV_ZERO);
      return values.reduce((total, value) => total + value, 0) / values.length;
    },
    MIN(nodes, context) {
      const values = numericFunctionValues(nodes, context);
      return values.length ? Math.min(...values) : 0;
    },
    MAX(nodes, context) {
      const values = numericFunctionValues(nodes, context);
      return values.length ? Math.max(...values) : 0;
    },
    COUNT(nodes, context) {
      return numericFunctionValues(nodes, context).length;
    },
    COUNTIF(nodes, context) {
      return runConditionalAggregate("COUNTIF", nodes, context).result;
    },
    SUMIF(nodes, context) {
      return runConditionalAggregate("SUMIF", nodes, context).result;
    },
    AVERAGEIF(nodes, context) {
      return runConditionalAggregate("AVERAGEIF", nodes, context).result;
    },
    COUNTIFS(nodes, context) {
      return runConditionalAggregate("COUNTIFS", nodes, context).result;
    },
    SUMIFS(nodes, context) {
      return runConditionalAggregate("SUMIFS", nodes, context).result;
    },
    AVERAGEIFS(nodes, context) {
      return runConditionalAggregate("AVERAGEIFS", nodes, context).result;
    },
    VLOOKUP(nodes, context) {
      return finalizeLookupTrace(runLookupFunction("VLOOKUP", nodes, context));
    },
    HLOOKUP(nodes, context) {
      return finalizeLookupTrace(runLookupFunction("HLOOKUP", nodes, context));
    },
    XLOOKUP(nodes, context) {
      return finalizeLookupTrace(runLookupFunction("XLOOKUP", nodes, context));
    },
    MATCH(nodes, context) {
      return finalizeLookupTrace(runLookupFunction("MATCH", nodes, context));
    },
    INDEX(nodes, context) {
      return finalizeLookupTrace(runLookupFunction("INDEX", nodes, context));
    },
    LEN(nodes, context) {
      return finalizeTextTrace(runTextFunction("LEN", nodes, context));
    },
    LEFT(nodes, context) {
      return finalizeTextTrace(runTextFunction("LEFT", nodes, context));
    },
    RIGHT(nodes, context) {
      return finalizeTextTrace(runTextFunction("RIGHT", nodes, context));
    },
    MID(nodes, context) {
      return finalizeTextTrace(runTextFunction("MID", nodes, context));
    },
    TRIM(nodes, context) {
      return finalizeTextTrace(runTextFunction("TRIM", nodes, context));
    },
    UPPER(nodes, context) {
      return finalizeTextTrace(runTextFunction("UPPER", nodes, context));
    },
    LOWER(nodes, context) {
      return finalizeTextTrace(runTextFunction("LOWER", nodes, context));
    },
    PROPER(nodes, context) {
      return finalizeTextTrace(runTextFunction("PROPER", nodes, context));
    },
    CONCAT(nodes, context) {
      return finalizeTextTrace(runTextFunction("CONCAT", nodes, context));
    },
    TEXTJOIN(nodes, context) {
      return finalizeTextTrace(runTextFunction("TEXTJOIN", nodes, context));
    },
    FIND(nodes, context) {
      return finalizeTextTrace(runTextFunction("FIND", nodes, context));
    },
    SEARCH(nodes, context) {
      return finalizeTextTrace(runTextFunction("SEARCH", nodes, context));
    },
    SUBSTITUTE(nodes, context) {
      return finalizeTextTrace(runTextFunction("SUBSTITUTE", nodes, context));
    },
    REPLACE(nodes, context) {
      return finalizeTextTrace(runTextFunction("REPLACE", nodes, context));
    },
    DATE(nodes, context) {
      return finalizeDateTrace(runDateFunction("DATE", nodes, context));
    },
    YEAR(nodes, context) {
      return finalizeDateTrace(runDateFunction("YEAR", nodes, context));
    },
    MONTH(nodes, context) {
      return finalizeDateTrace(runDateFunction("MONTH", nodes, context));
    },
    DAY(nodes, context) {
      return finalizeDateTrace(runDateFunction("DAY", nodes, context));
    },
    TODAY(nodes, context) {
      return finalizeDateTrace(runDateFunction("TODAY", nodes, context));
    },
    DAYS(nodes, context) {
      return finalizeDateTrace(runDateFunction("DAYS", nodes, context));
    },
    EDATE(nodes, context) {
      return finalizeDateTrace(runDateFunction("EDATE", nodes, context));
    },
    EOMONTH(nodes, context) {
      return finalizeDateTrace(runDateFunction("EOMONTH", nodes, context));
    },
    WEEKDAY(nodes, context) {
      return finalizeDateTrace(runDateFunction("WEEKDAY", nodes, context));
    },
    ROUND(nodes, context) {
      return finalizeMathTrace(runMathFunction("ROUND", nodes, context));
    },
    ROUNDUP(nodes, context) {
      return finalizeMathTrace(runMathFunction("ROUNDUP", nodes, context));
    },
    ROUNDDOWN(nodes, context) {
      return finalizeMathTrace(runMathFunction("ROUNDDOWN", nodes, context));
    },
    INT(nodes, context) {
      return finalizeMathTrace(runMathFunction("INT", nodes, context));
    },
    ABS(nodes, context) {
      return finalizeMathTrace(runMathFunction("ABS", nodes, context));
    },
    MOD(nodes, context) {
      return finalizeMathTrace(runMathFunction("MOD", nodes, context));
    },
    SEQUENCE(nodes, context) {
      return finalizeDynamicArrayTrace(runDynamicArrayFunction("SEQUENCE", nodes, context));
    },
    FILTER(nodes, context) {
      return finalizeDynamicArrayTrace(runDynamicArrayFunction("FILTER", nodes, context));
    },
    SORT(nodes, context) {
      return finalizeDynamicArrayTrace(runDynamicArrayFunction("SORT", nodes, context));
    },
    SORTBY(nodes, context) {
      return finalizeDynamicArrayTrace(runDynamicArrayFunction("SORTBY", nodes, context));
    },
    UNIQUE(nodes, context) {
      return finalizeDynamicArrayTrace(runDynamicArrayFunction("UNIQUE", nodes, context));
    }
  };

  function evaluateNode(node, context) {
    if (node.type === "number") return node.value;
    if (node.type === "string" || node.type === "boolean") return node.value;
    if (node.type === "reference") return context.getCellValue(node.reference);

    if (node.type === "range") {
      return arrayFromRange(node.start, node.end, context);
    }

    if (node.type === "unary") {
      return mapArrayValues(evaluateNode(node.operand, context), (entry) => {
        const value = requireNumber(entry);
        return node.operator === "-" ? -value : value;
      });
    }

    if (node.type === "postfix") {
      if (node.operator !== "%") throw new FormulaError(ERROR_VALUES.GENERIC);
      return mapArrayValues(
        evaluateNode(node.operand, context),
        (entry) => requireNumber(entry) / 100
      );
    }

    if (node.type === "binary") {
      if (node.operator === "&") {
        return textValue(evaluateNode(node.left, context))
          + textValue(evaluateNode(node.right, context));
      }

      const left = evaluateNode(node.left, context);
      const right = evaluateNode(node.right, context);
      const arrayOperation = isArrayValue(left) || isArrayValue(right);
      return mapArrayPair(left, right, (leftEntry, rightEntry) => {
        const leftNumber = arrayOperation && typeof leftEntry === "boolean"
          ? Number(leftEntry)
          : requireNumber(leftEntry);
        const rightNumber = arrayOperation && typeof rightEntry === "boolean"
          ? Number(rightEntry)
          : requireNumber(rightEntry);
        if (node.operator === "+") return cleanNumericResult(leftNumber + rightNumber);
        if (node.operator === "-") return cleanNumericResult(leftNumber - rightNumber);
        if (node.operator === "*") return cleanNumericResult(leftNumber * rightNumber);
        if (rightNumber === 0) throw new FormulaError(ERROR_VALUES.DIV_ZERO);
        return cleanNumericResult(leftNumber / rightNumber);
      });
    }

    if (node.type === "comparison") {
      return mapArrayPair(
        evaluateNode(node.left, context),
        evaluateNode(node.right, context),
        (left, right) => compareValues(left, right, node.operator)
      );
    }

    if (node.type === "function") {
      if (node.name === "IF") {
        if (node.arguments.length !== 3) throw new FormulaError(ERROR_VALUES.GENERIC);
        const condition = requireLogical(evaluateNode(node.arguments[0], context));
        return evaluateNode(node.arguments[condition ? 1 : 2], context);
      }

      if (node.name === "NOT") {
        if (node.arguments.length !== 1) throw new FormulaError(ERROR_VALUES.GENERIC);
        return !requireLogical(evaluateNode(node.arguments[0], context));
      }

      if (node.name === "AND" || node.name === "OR") {
        if (!node.arguments.length) throw new FormulaError(ERROR_VALUES.GENERIC);
        const values = node.arguments.map((argument) => (
          requireLogical(evaluateNode(argument, context))
        ));
        return node.name === "AND" ? values.every(Boolean) : values.some(Boolean);
      }

      if (ERROR_FUNCTIONS.has(node.name)) {
        return runErrorHandlingFunction(node.name, node.arguments, context).result;
      }

      const implementation = functions[node.name];
      if (!implementation) throw new FormulaError(ERROR_VALUES.NAME);
      return implementation(node.arguments, context);
    }

    throw new FormulaError(ERROR_VALUES.GENERIC);
  }

  function evaluate(ast, context) {
    return evaluateNode(ast, context);
  }

  function analyzeConditionalAggregate(ast, context) {
    if (ast?.type !== "function" || !CONDITIONAL_FUNCTIONS.has(ast.name)) return null;
    return runConditionalAggregate(ast.name, ast.arguments, context);
  }

  function analyzeLookupExpression(ast, context) {
    if (ast?.type !== "function" || !LOOKUP_FUNCTIONS.has(ast.name)) return null;
    const trace = runLookupFunction(ast.name, ast.arguments, context);

    if (ast.name === "INDEX") {
      [ast.arguments[1], ast.arguments[2]].forEach((argument, index) => {
        if (argument?.type === "function" && LOOKUP_FUNCTIONS.has(argument.name)) {
          trace.children.push({
            role: index === 0 ? "row_num" : "column_num",
            trace: runLookupFunction(argument.name, argument.arguments, context)
          });
        }
      });
    }

    return trace;
  }

  function analyzeTextExpression(ast, context) {
    if (ast?.type !== "function" || !TEXT_FUNCTIONS.has(ast.name)) return null;
    return runTextFunction(ast.name, ast.arguments, context);
  }

  function analyzeMathExpression(ast, context) {
    if (ast?.type !== "function" || !MATH_FUNCTIONS.has(ast.name)) return null;
    return runMathFunction(ast.name, ast.arguments, context);
  }

  function analyzeErrorExpression(ast, context) {
    if (ast?.type !== "function" || !ERROR_FUNCTIONS.has(ast.name)) return null;
    return runErrorHandlingFunction(ast.name, ast.arguments, context, { traceUncaught: true });
  }

  function analyzeDynamicArrayExpression(ast, context) {
    if (ast?.type !== "function" || !DYNAMIC_ARRAY_FUNCTIONS.has(ast.name)) return null;
    return runDynamicArrayFunction(ast.name, ast.arguments, context);
  }

  function inferNumberFormat(ast, getCellNumberFormat = () => ExcelFormatting.NUMBER_FORMATS.GENERAL) {
    if (!ast) return ExcelFormatting.NUMBER_FORMATS.GENERAL;
    if (ast.type === "reference") return getCellNumberFormat(ast.reference);
    if (ast.type === "function") {
      return DATE_RESULT_FUNCTIONS.has(ast.name)
        ? ExcelFormatting.NUMBER_FORMATS.DATE
        : ExcelFormatting.NUMBER_FORMATS.GENERAL;
    }
    if (ast.type === "unary") return inferNumberFormat(ast.operand, getCellNumberFormat);
    if (ast.type === "postfix") return ExcelFormatting.NUMBER_FORMATS.GENERAL;
    if (ast.type !== "binary") return ExcelFormatting.NUMBER_FORMATS.GENERAL;

    const left = inferNumberFormat(ast.left, getCellNumberFormat);
    const right = inferNumberFormat(ast.right, getCellNumberFormat);
    const date = ExcelFormatting.NUMBER_FORMATS.DATE;
    const general = ExcelFormatting.NUMBER_FORMATS.GENERAL;
    if (ast.operator === "+" && ((left === date) !== (right === date))) return date;
    if (ast.operator === "-" && left === date && right !== date) return date;
    return general;
  }

  function analyzeDateExpression(ast, context) {
    if (ast?.type === "function" && DATE_FUNCTIONS.has(ast.name)) {
      return runDateFunction(ast.name, ast.arguments, context);
    }

    if (ast?.type !== "binary" || !["+", "-"].includes(ast.operator)) return null;
    const getFormat = context.getCellNumberFormat
      || (() => ExcelFormatting.NUMBER_FORMATS.GENERAL);
    if (inferNumberFormat(ast, getFormat) !== ExcelFormatting.NUMBER_FORMATS.DATE) return null;

    const leftFormat = inferNumberFormat(ast.left, getFormat);
    const dateOnLeft = leftFormat === ExcelFormatting.NUMBER_FORMATS.DATE;
    const dateNode = dateOnLeft ? ast.left : ast.right;
    const numberNode = dateOnLeft ? ast.right : ast.left;
    const startSerial = dateSerialArgument(dateNode, context);
    const amount = requireNumber(evaluateNode(numberNode, context));
    const dayChange = ast.operator === "-" ? -amount : amount;
    const result = evaluateNode(ast, context);
    if (!ExcelFormatting.serialToCalendar(result)) throw new FormulaError(ERROR_VALUES.NUM);
    return dateResultTrace("DATE ARITHMETIC", result, {
      kind: "date-arithmetic",
      startSerial,
      startDate: ExcelFormatting.formatDateSerial(startSerial),
      operation: dayChange >= 0 ? "add" : "subtract",
      days: Math.abs(dayChange),
      signedDays: dayChange
    });
  }

  function containsVolatileFunction(ast) {
    let volatile = false;
    function visit(node) {
      if (!node || volatile) return;
      if (node.type === "function") {
        if (VOLATILE_FUNCTIONS.has(node.name)) {
          volatile = true;
          return;
        }
        node.arguments.forEach(visit);
      } else if (node.type === "unary") {
        visit(node.operand);
      } else if (node.type === "postfix") {
        visit(node.operand);
      } else if (node.type === "binary" || node.type === "comparison") {
        visit(node.left);
        visit(node.right);
      }
    }
    visit(ast);
    return volatile;
  }

  function collectReferences(ast) {
    const references = new Set();

    function visit(node) {
      if (node.type === "reference") {
        references.add(node.reference);
      } else if (node.type === "range") {
        expandRange(node.start, node.end).forEach((reference) => references.add(reference));
      } else if (node.type === "unary") {
        visit(node.operand);
      } else if (node.type === "postfix") {
        visit(node.operand);
      } else if (node.type === "binary") {
        visit(node.left);
        visit(node.right);
      } else if (node.type === "comparison") {
        visit(node.left);
        visit(node.right);
      } else if (node.type === "function") {
        node.arguments.forEach(visit);
      }
    }

    visit(ast);
    return [...references];
  }

  const api = {
    ERROR_VALUES,
    FormulaError,
    FormulaSyntaxError,
    analyzeConditionalAggregate,
    analyzeDateExpression,
    analyzeDynamicArrayExpression,
    analyzeErrorExpression,
    analyzeLookupExpression,
    analyzeMathExpression,
    analyzeTextExpression,
    collectReferences,
    containsVolatileFunction,
    evaluate,
    expandRange,
    inferNumberFormat,
    isArrayValue,
    isSpreadsheetError: isKnownErrorValue,
    makeArray,
    parseFormula,
    parseReference,
    tokenize
  };

  global.FormulaEngine = api;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
})(typeof window === "undefined" ? globalThis : window);
