((global) => {
  "use strict";

  const OPERATORS = ["<>", ">=", "<=", "=", ">", "<"];
  const NUMBER_PATTERN = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$/;

  class CriteriaError extends Error {}

  function isBlank(value) {
    return value === "" || value === null || value === undefined;
  }

  function wildcardPattern(text) {
    let pattern = "";

    for (let index = 0; index < text.length; index += 1) {
      const character = text[index];

      if (character === "~" && index + 1 < text.length) {
        index += 1;
        pattern += text[index].replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      } else if (character === "*") {
        pattern += ".*";
      } else if (character === "?") {
        pattern += ".";
      } else {
        pattern += character.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      }
    }

    return new RegExp(`^${pattern}$`, "i");
  }

  function parseOperand(value) {
    if (value === "") return { kind: "blank", value: "" };
    if (NUMBER_PATTERN.test(value)) return { kind: "number", value: Number(value) };
    if (/^(TRUE|FALSE)$/i.test(value)) {
      return { kind: "boolean", value: value.toUpperCase() === "TRUE" };
    }
    return { kind: "text", value };
  }

  function compile(rawValue) {
    if (typeof rawValue === "number" && Number.isFinite(rawValue)) {
      return {
        rawValue,
        display: String(rawValue),
        operator: "=",
        operand: { kind: "number", value: rawValue },
        wildcard: null
      };
    }

    if (typeof rawValue === "boolean") {
      return {
        rawValue,
        display: rawValue ? "TRUE" : "FALSE",
        operator: "=",
        operand: { kind: "boolean", value: rawValue },
        wildcard: null
      };
    }

    if (isBlank(rawValue)) rawValue = "";
    if (typeof rawValue !== "string") throw new CriteriaError("Invalid criterion");

    const operator = OPERATORS.find((candidate) => rawValue.startsWith(candidate)) || "=";
    const operandText = operator === "=" && !rawValue.startsWith("=")
      ? rawValue
      : rawValue.slice(operator.length);
    const operand = parseOperand(operandText);
    const usesWildcards = operand.kind === "text" && /[*?]/.test(operand.value);

    return {
      rawValue,
      display: rawValue,
      operator,
      operand,
      wildcard: usesWildcards ? wildcardPattern(operand.value) : null
    };
  }

  function orderedComparison(candidate, target, operator) {
    if (operator === ">") return candidate > target;
    if (operator === "<") return candidate < target;
    if (operator === ">=") return candidate >= target;
    if (operator === "<=") return candidate <= target;
    return false;
  }

  function matches(candidate, criterion) {
    const { operator, operand, wildcard } = criterion;

    if (operand.kind === "blank") {
      if (operator === "=") return isBlank(candidate);
      if (operator === "<>") return !isBlank(candidate);
      return false;
    }

    if (operand.kind === "number") {
      if (typeof candidate !== "number" || !Number.isFinite(candidate)) {
        return operator === "<>";
      }
      if (operator === "=") return candidate === operand.value;
      if (operator === "<>") return candidate !== operand.value;
      return orderedComparison(candidate, operand.value, operator);
    }

    if (operand.kind === "boolean") {
      if (typeof candidate !== "boolean") return operator === "<>";
      if (operator === "=") return candidate === operand.value;
      if (operator === "<>") return candidate !== operand.value;
      return orderedComparison(Number(candidate), Number(operand.value), operator);
    }

    if (typeof candidate !== "string") return operator === "<>";
    const candidateText = candidate.toLocaleLowerCase();
    const targetText = operand.value.toLocaleLowerCase();
    const equal = wildcard ? wildcard.test(candidate) : candidateText === targetText;

    if (operator === "=") return equal;
    if (operator === "<>") return !equal;
    return orderedComparison(candidateText, targetText, operator);
  }

  const api = {
    CriteriaError,
    compile,
    isBlank,
    matches
  };

  global.CriteriaEngine = api;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
})(typeof window === "undefined" ? globalThis : window);
