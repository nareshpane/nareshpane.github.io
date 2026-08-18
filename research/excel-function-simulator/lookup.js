((global) => {
  "use strict";

  const CriteriaEngine = global.CriteriaEngine
    || (typeof require === "function" ? require("./criteria.js") : null);

  function isBlank(value) {
    return value === "" || value === null || value === undefined;
  }

  function valuesEqual(left, right) {
    if (isBlank(left) || isBlank(right)) return isBlank(left) && isBlank(right);
    if (typeof left !== typeof right) return false;
    if (typeof left === "string") return left.toLowerCase() === right.toLowerCase();
    return left === right;
  }

  function compareValues(left, right) {
    if (isBlank(left) || isBlank(right)) {
      if (isBlank(left) && isBlank(right)) return 0;
      return null;
    }
    if (typeof left !== typeof right) return null;

    const normalizedLeft = typeof left === "string" ? left.toLowerCase() : left;
    const normalizedRight = typeof right === "string" ? right.toLowerCase() : right;
    if (normalizedLeft === normalizedRight) return 0;
    return normalizedLeft < normalizedRight ? -1 : 1;
  }

  function wildcardMatches(candidate, pattern) {
    if (typeof candidate !== "string" || typeof pattern !== "string") return false;
    return CriteriaEngine.matches(candidate, CriteriaEngine.compile(pattern));
  }

  function search(entries, lookupValue, options = {}) {
    const matchMode = options.matchMode ?? 0;
    const searchMode = options.searchMode ?? 1;
    const ordered = searchMode === -1 ? [...entries].reverse() : [...entries];
    const steps = [];
    let selected = null;
    let best = null;

    for (const entry of ordered) {
      const relation = compareValues(entry.value, lookupValue);
      const exact = matchMode === 2
        ? wildcardMatches(entry.value, lookupValue)
        : valuesEqual(entry.value, lookupValue);
      const step = {
        position: entry.index + 1,
        reference: entry.reference,
        value: entry.value,
        relation: exact ? (matchMode === 2 ? "wildcard-match" : "equal")
          : (relation === null ? "incompatible" : (relation < 0 ? "less" : "greater")),
        matched: exact,
        selected: false
      };
      steps.push(step);

      if (exact) {
        selected = entry;
        step.selected = true;
        break;
      }

      if (matchMode === -1 && relation !== null && relation < 0) {
        if (!best || compareValues(entry.value, best.value) > 0) best = entry;
      } else if (matchMode === 1 && relation !== null && relation > 0) {
        if (!best || compareValues(entry.value, best.value) < 0) best = entry;
      }
    }

    if (!selected && best) {
      selected = best;
      const selectedStep = steps.find((step) => step.position === best.index + 1);
      if (selectedStep) selectedStep.selected = true;
    }

    return {
      lookupValue,
      matchMode,
      searchMode,
      steps,
      selected,
      selectedPosition: selected ? selected.index + 1 : null
    };
  }

  const api = {
    compareValues,
    isBlank,
    search,
    valuesEqual,
    wildcardMatches
  };

  global.LookupEngine = api;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
})(typeof window === "undefined" ? globalThis : window);
