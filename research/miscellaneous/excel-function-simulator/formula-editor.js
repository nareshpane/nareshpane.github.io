(() => {
  "use strict";

  const simulator = window.ExcelSimulator;
  const engine = window.FormulaEngine;
  const catalog = window.ExcelFunctionCatalog;
  const grid = document.querySelector("#spreadsheet-grid");
  const formulaControls = document.querySelector(".formula-controls");
  const formulaInput = document.querySelector("#formula-input");
  const nameBox = document.querySelector("#name-box");
  const functionButton = document.querySelector("#formula-function-button");
  const functions = catalog.filter((entry) => /^[A-Z][A-Z0-9.]*$/.test(entry.name));
  const functionsByName = new Map(functions.map((entry) => [entry.name, entry]));
  const referenceClasses = [
    "formula-edit-reference-1",
    "formula-edit-reference-2",
    "formula-edit-reference-3",
    "formula-edit-reference-4"
  ];

  const state = {
    active: false,
    target: "",
    originalInput: "",
    source: null,
    focusReference: "",
    focusStartInput: "",
    caretStart: 0,
    caretEnd: 0,
    suggestions: [],
    suggestionIndex: 0,
    suggestionKey: "",
    suggestionStart: 0,
    dragStart: "",
    dragEnd: "",
    suppressClick: false,
    highlightedCells: new Set(),
    dragCells: new Set()
  };

  function createPopover(className, role) {
    const element = document.createElement("div");
    element.className = `formula-authoring-popover ${className}`;
    if (role) element.setAttribute("role", role);
    element.hidden = true;
    formulaControls.append(element);
    return element;
  }

  const autocomplete = createPopover("formula-autocomplete", "listbox");
  autocomplete.id = "formula-autocomplete";
  const signature = createPopover("formula-signature", "status");
  signature.id = "formula-signature";
  signature.setAttribute("aria-live", "polite");
  const functionPicker = createPopover("function-picker", "dialog");
  functionPicker.id = "function-picker";
  functionPicker.setAttribute("aria-label", "Insert a function");

  const pickerSearchWrap = document.createElement("label");
  pickerSearchWrap.className = "function-picker-search-wrap";
  const pickerSearchLabel = document.createElement("span");
  pickerSearchLabel.className = "visually-hidden";
  pickerSearchLabel.textContent = "Search functions";
  const pickerSearch = document.createElement("input");
  pickerSearch.className = "function-picker-search";
  pickerSearch.type = "search";
  pickerSearch.placeholder = "Search functions";
  pickerSearch.autocomplete = "off";
  pickerSearchWrap.append(pickerSearchLabel, pickerSearch);
  const pickerResults = document.createElement("div");
  functionPicker.append(pickerSearchWrap, pickerResults);

  function sourceText() {
    if (state.source === formulaInput) return formulaInput.value;
    return state.source?.textContent || "";
  }

  function selectionOffsets(element) {
    if (element === formulaInput) {
      const fallback = formulaInput.value.length;
      return {
        start: formulaInput.selectionStart ?? fallback,
        end: formulaInput.selectionEnd ?? fallback
      };
    }

    const selection = window.getSelection();
    if (!selection.rangeCount || !element.contains(selection.anchorNode)) {
      const fallback = element.textContent.length;
      return { start: fallback, end: fallback };
    }

    const range = selection.getRangeAt(0);
    const beforeStart = range.cloneRange();
    beforeStart.selectNodeContents(element);
    beforeStart.setEnd(range.startContainer, range.startOffset);
    const beforeEnd = range.cloneRange();
    beforeEnd.selectNodeContents(element);
    beforeEnd.setEnd(range.endContainer, range.endOffset);
    return { start: beforeStart.toString().length, end: beforeEnd.toString().length };
  }

  function setSelection(element, start, end = start) {
    if (element === formulaInput) {
      formulaInput.setSelectionRange(start, end);
      return;
    }

    const textNode = element.firstChild || element.appendChild(document.createTextNode(""));
    const range = document.createRange();
    const length = textNode.textContent.length;
    range.setStart(textNode, Math.min(start, length));
    range.setEnd(textNode, Math.min(end, length));
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
  }

  function rememberCaret() {
    if (!state.source) return;
    const offsets = selectionOffsets(state.source);
    state.caretStart = offsets.start;
    state.caretEnd = offsets.end;
  }

  function clearCellClasses(cells, classNames) {
    cells.forEach((cell) => cell.classList.remove(...classNames));
    cells.clear();
  }

  function clearReferenceHighlights() {
    clearCellClasses(state.highlightedCells, referenceClasses);
  }

  function clearDragHighlights() {
    clearCellClasses(state.dragCells, ["formula-drag-reference"]);
  }

  function referencesInFormula(formula) {
    let inString = false;
    let masked = "";

    for (let index = 0; index < formula.length; index += 1) {
      const character = formula[index];
      if (character === "\"") {
        if (inString && formula[index + 1] === "\"") {
          masked += "  ";
          index += 1;
          continue;
        }
        inString = !inString;
        masked += " ";
      } else {
        masked += inString ? " " : character;
      }
    }

    const references = [];
    const pattern = /(?<![A-Z0-9_.])(\$?[A-Z]{1,3}\$?[1-9]\d*)(?:\s*:\s*(\$?[A-Z]{1,3}\$?[1-9]\d*))?(?![A-Z0-9_.])/gi;
    let match;
    while ((match = pattern.exec(masked))) {
      references.push({ start: match[1].toUpperCase(), end: (match[2] || match[1]).toUpperCase() });
    }
    return references;
  }

  function referenceTokenAtCaret(text, caretStart, caretEnd) {
    if (!text.startsWith("=")) return null;
    let inString = false;
    let masked = "";
    for (let index = 0; index < text.length; index += 1) {
      const character = text[index];
      if (character === "\"") {
        if (inString && text[index + 1] === "\"") {
          masked += "  ";
          index += 1;
          continue;
        }
        inString = !inString;
        masked += " ";
      } else {
        masked += inString ? " " : character;
      }
    }

    const pattern = /(?<![A-Z0-9_.])\$?[A-Z]{1,3}\$?[1-9]\d*(?:\s*:\s*\$?[A-Z]{1,3}\$?[1-9]\d*)?(?![A-Z0-9_.])/gi;
    let match;
    let preceding = null;
    while ((match = pattern.exec(masked))) {
      const start = match.index;
      const end = start + match[0].length;
      if (caretStart !== caretEnd && caretStart === start && caretEnd === end) {
        return { start, end, address: text.slice(start, end) };
      }
      if (caretStart >= start && caretStart <= end && caretEnd >= start && caretEnd <= end) {
        return { start, end, address: text.slice(start, end) };
      }
      if (end === caretStart && caretStart === caretEnd) {
        preceding = { start, end, address: text.slice(start, end) };
      }
    }
    return preceding;
  }

  function cycleReferenceAddress(address) {
    if (!address.includes(":")) return engine.cycleReferenceLock(address);
    return address.split(":").map((part) => engine.cycleReferenceLock(part.trim())).join(":");
  }

  function cycleReferenceAtCaret() {
    rememberCaret();
    const reference = referenceTokenAtCaret(
      sourceText(),
      state.caretStart,
      state.caretEnd
    );
    if (!reference) return false;
    const nextAddress = cycleReferenceAddress(reference.address);
    const current = sourceText();
    const next = current.slice(0, reference.start) + nextAddress + current.slice(reference.end);
    const selectionEnd = reference.start + nextAddress.length;
    writeSource(next, reference.start, selectionEnd);
    updateAuthoring();
    return true;
  }

  function visibleRange(start, end) {
    try {
      const first = engine.parseReference(start);
      const last = engine.parseReference(end);
      const firstRow = Math.max(0, Math.min(first.row, last.row));
      const lastRow = Math.min(49, Math.max(first.row, last.row));
      const firstColumn = Math.max(0, Math.min(first.column, last.column));
      const lastColumn = Math.min(25, Math.max(first.column, last.column));
      const cells = [];

      for (let row = firstRow; row <= lastRow; row += 1) {
        for (let column = firstColumn; column <= lastColumn; column += 1) {
          const reference = `${String.fromCharCode(65 + column)}${row + 1}`;
          const cell = simulator.getCellElement(reference);
          if (cell) cells.push(cell);
        }
      }
      return cells;
    } catch (error) {
      return [];
    }
  }

  function highlightFormulaReferences() {
    clearReferenceHighlights();
    if (!state.active) return;

    referencesInFormula(sourceText()).forEach((reference, index) => {
      const className = referenceClasses[index % referenceClasses.length];
      visibleRange(reference.start, reference.end).forEach((cell) => {
        cell.classList.add(className);
        state.highlightedCells.add(cell);
      });
    });
  }

  function isInsideString(text, caret) {
    let inString = false;
    for (let index = 0; index < caret; index += 1) {
      if (text[index] !== "\"") continue;
      if (inString && text[index + 1] === "\"" && index + 1 < caret) {
        index += 1;
      } else {
        inString = !inString;
      }
    }
    return inString;
  }

  function autocompleteContext(text, caret) {
    if (!text.startsWith("=") || isInsideString(text, caret)) return null;
    const match = text.slice(0, caret).match(/([A-Z][A-Z0-9.]*)$/i);
    if (!match) return null;
    const start = caret - match[1].length;
    const previous = text[start - 1] || "";
    if (/[A-Z0-9_.]/i.test(previous)) return null;
    return { prefix: match[1].toUpperCase(), start };
  }

  function closeAutocomplete() {
    autocomplete.hidden = true;
    autocomplete.replaceChildren();
    state.suggestions = [];
    state.suggestionKey = "";
    state.source?.removeAttribute("aria-activedescendant");
  }

  function renderAutocomplete() {
    const context = autocompleteContext(sourceText(), state.caretStart);
    if (!context) {
      closeAutocomplete();
      return false;
    }

    const matches = functions.filter((entry) => entry.name.startsWith(context.prefix)).slice(0, 8);
    if (!matches.length) {
      closeAutocomplete();
      return false;
    }

    const suggestionKey = `${context.start}:${context.prefix}`;
    if (suggestionKey !== state.suggestionKey) state.suggestionIndex = 0;
    state.suggestionKey = suggestionKey;
    state.suggestions = matches;
    state.suggestionStart = context.start;
    state.suggestionIndex = Math.min(state.suggestionIndex, matches.length - 1);
    autocomplete.replaceChildren();

    matches.forEach((entry, index) => {
      const option = document.createElement("button");
      option.type = "button";
      option.id = `formula-suggestion-${index}`;
      option.className = `formula-autocomplete-option${index === state.suggestionIndex ? " active" : ""}`;
      option.dataset.functionName = entry.name;
      option.setAttribute("role", "option");
      option.setAttribute("aria-selected", String(index === state.suggestionIndex));
      const name = document.createElement("span");
      name.className = "formula-autocomplete-name";
      name.textContent = entry.name;
      const description = document.createElement("span");
      description.className = "formula-autocomplete-description";
      description.textContent = entry.shortDescription;
      option.append(name, description);
      autocomplete.append(option);
    });

    autocomplete.hidden = false;
    state.source?.setAttribute("aria-activedescendant", `formula-suggestion-${state.suggestionIndex}`);
    return true;
  }

  function functionContext(text, caret) {
    if (!text.startsWith("=")) return null;
    const stack = [];
    let inString = false;

    for (let index = 1; index < caret; index += 1) {
      const character = text[index];
      if (character === "\"") {
        if (inString && text[index + 1] === "\"" && index + 1 < caret) {
          index += 1;
        } else {
          inString = !inString;
        }
        continue;
      }
      if (inString) continue;

      if (character === "(") {
        let end = index;
        while (end > 0 && /\s/.test(text[end - 1])) end -= 1;
        let start = end;
        while (start > 0 && /[A-Z0-9.]/i.test(text[start - 1])) start -= 1;
        const name = text.slice(start, end).toUpperCase();
        stack.push({ entry: functionsByName.get(name) || null, argumentIndex: 0 });
      } else if (character === ")") {
        stack.pop();
      } else if (character === "," && stack.length) {
        stack[stack.length - 1].argumentIndex += 1;
      }
    }

    for (let index = stack.length - 1; index >= 0; index -= 1) {
      if (stack[index].entry) return stack[index];
    }
    return null;
  }

  function renderSignature() {
    const context = functionContext(sourceText(), state.caretStart);
    if (!context || !autocomplete.hidden) {
      signature.hidden = true;
      signature.replaceChildren();
      return;
    }

    const { entry } = context;
    const activeIndex = Math.min(context.argumentIndex, entry.arguments.length - 1);
    const line = document.createElement("div");
    line.className = "formula-signature-line";
    line.append(document.createTextNode(`${entry.name}(`));
    entry.arguments.forEach((argument, index) => {
      if (index) line.append(document.createTextNode(", "));
      const item = document.createElement("span");
      item.className = `formula-signature-argument${index === activeIndex ? " active" : ""}`;
      const optional = entry.syntax.includes(`[${argument.name}]`)
        || argument.description.toLowerCase().startsWith("optional");
      item.textContent = optional ? `[${argument.name}]` : argument.name;
      line.append(item);
    });
    line.append(document.createTextNode(")"));

    const description = document.createElement("p");
    description.className = "formula-signature-description";
    description.textContent = entry.arguments[activeIndex]?.description || entry.shortDescription;
    signature.replaceChildren(line, description);
    signature.hidden = false;
  }

  function updateAuthoring() {
    if (!state.active) return;
    rememberCaret();
    highlightFormulaReferences();
    renderAutocomplete();
    renderSignature();
  }

  function beginFormulaEdit(source, target, originalInput) {
    if (state.active && state.target !== target) finishFormulaEdit();

    state.active = true;
    state.target = target;
    state.originalInput = originalInput;
    state.source = source;
    simulator.getCellElement(target)?.classList.add("formula-edit-target");
    formulaControls.classList.add("formula-editing");
    rememberCaret();
    updateAuthoring();
  }

  function finishFormulaEdit() {
    if (!state.active) return;
    simulator.getCellElement(state.target)?.classList.remove("formula-edit-target");
    state.source?.removeAttribute("aria-activedescendant");
    state.active = false;
    state.source = null;
    state.target = "";
    state.dragStart = "";
    state.dragEnd = "";
    formulaControls.classList.remove("formula-editing");
    closeAutocomplete();
    signature.hidden = true;
    signature.replaceChildren();
    clearReferenceHighlights();
    clearDragHighlights();
  }

  function writeSource(text, start, end = start) {
    if (state.source === formulaInput) {
      formulaInput.value = text;
    } else {
      state.source.textContent = text;
    }
    setSelection(state.source, start, end);
    state.caretStart = start;
    state.caretEnd = end;
    state.source.dispatchEvent(new Event("input", { bubbles: true }));
  }

  function insertText(text, replaceStart = state.caretStart, replaceEnd = state.caretEnd) {
    const current = sourceText();
    const next = current.slice(0, replaceStart) + text + current.slice(replaceEnd);
    const caret = replaceStart + text.length;
    writeSource(next, caret);
  }

  function acceptSuggestion(name) {
    const entry = name
      ? functionsByName.get(name)
      : state.suggestions[state.suggestionIndex];
    if (!entry) return;
    insertText(`${entry.name}(`, state.suggestionStart, state.caretEnd);
  }

  function moveSuggestion(offset) {
    if (!state.suggestions.length) return;
    state.suggestionIndex = (
      state.suggestionIndex + offset + state.suggestions.length
    ) % state.suggestions.length;
    renderAutocomplete();
    autocomplete.querySelector(".active")?.scrollIntoView({ block: "nearest" });
  }

  function handleAuthoringKeydown(event, isCellSource) {
    if (!state.active || event.target !== state.source) return;

    if (event.key === "F4") {
      event.preventDefault();
      event.stopImmediatePropagation();
      cycleReferenceAtCaret();
    } else if (!autocomplete.hidden && event.key === "ArrowDown") {
      event.preventDefault();
      event.stopImmediatePropagation();
      moveSuggestion(1);
    } else if (!autocomplete.hidden && event.key === "ArrowUp") {
      event.preventDefault();
      event.stopImmediatePropagation();
      moveSuggestion(-1);
    } else if (!autocomplete.hidden && (event.key === "Enter" || event.key === "Tab")) {
      event.preventDefault();
      event.stopImmediatePropagation();
      acceptSuggestion();
    } else if (event.key === "Escape") {
      closeAutocomplete();
      if (isCellSource) {
        window.setTimeout(finishFormulaEdit);
      } else {
        event.preventDefault();
        event.stopImmediatePropagation();
        const target = state.target;
        const originalInput = state.originalInput;
        finishFormulaEdit();
        simulator.setCell(target, originalInput);
        simulator.selectCell(target);
      }
    } else if (event.key === "Enter" || event.key === "Tab") {
      if (isCellSource) {
        window.setTimeout(finishFormulaEdit);
      } else {
        finishFormulaEdit();
      }
    }
  }

  function normalizedRange(start, end) {
    const references = engine.expandRange(start, end);
    if (references.length === 1) return references[0];
    return `${references[0]}:${references[references.length - 1]}`;
  }

  function showDragRange(start, end) {
    clearDragHighlights();
    visibleRange(start, end).forEach((cell) => {
      cell.classList.add("formula-drag-reference");
      state.dragCells.add(cell);
    });
  }

  function isReferenceCell(event) {
    return event.target.closest(".sheet-cell");
  }

  grid.addEventListener("mousedown", (event) => {
    if (!state.active || event.button !== 0) return;
    const cell = isReferenceCell(event);
    if (!cell || cell === state.source) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    rememberCaret();
    state.dragStart = cell.dataset.reference;
    state.dragEnd = state.dragStart;
    showDragRange(state.dragStart, state.dragEnd);
  }, true);

  grid.addEventListener("mouseover", (event) => {
    if (!state.active || !state.dragStart) return;
    const cell = isReferenceCell(event);
    if (!cell || cell.dataset.reference === state.dragEnd) return;
    state.dragEnd = cell.dataset.reference;
    showDragRange(state.dragStart, state.dragEnd);
  }, true);

  grid.addEventListener("mouseup", (event) => {
    if (!state.active || !state.dragStart || event.button !== 0) return;
    const cell = isReferenceCell(event);
    if (cell) state.dragEnd = cell.dataset.reference;
    event.preventDefault();
    event.stopImmediatePropagation();
    const reference = normalizedRange(state.dragStart, state.dragEnd);
    state.dragStart = "";
    state.dragEnd = "";
    state.suppressClick = true;
    window.setTimeout(() => {
      state.suppressClick = false;
    });
    clearDragHighlights();
    insertText(reference);
    state.source.focus({ preventScroll: true });
  }, true);

  document.addEventListener("mouseup", (event) => {
    if (!state.active || !state.dragStart || grid.contains(event.target)) return;
    event.preventDefault();
    const reference = normalizedRange(state.dragStart, state.dragEnd);
    state.dragStart = "";
    state.dragEnd = "";
    clearDragHighlights();
    insertText(reference);
    state.source.focus({ preventScroll: true });
  }, true);

  grid.addEventListener("click", (event) => {
    if (!state.suppressClick) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    state.suppressClick = false;
  }, true);

  grid.addEventListener("keydown", (event) => {
    handleAuthoringKeydown(event, state.source?.classList.contains("sheet-cell"));
  }, true);

  grid.addEventListener("keydown", (event) => {
    const cell = event.target.closest(".sheet-cell.editing");
    if (!state.active && cell?.textContent.startsWith("=")) {
      const target = cell.dataset.reference;
      beginFormulaEdit(cell, target, simulator.getCell(target)?.input || "");
    }
  });

  grid.addEventListener("dblclick", (event) => {
    const cell = event.target.closest(".sheet-cell.editing");
    if (!state.active && cell?.textContent.startsWith("=")) {
      const target = cell.dataset.reference;
      beginFormulaEdit(cell, target, simulator.getCell(target)?.input || "");
    }
  });

  grid.addEventListener("input", (event) => {
    const cell = event.target.closest(".sheet-cell.editing");
    if (!cell) return;
    if (!state.active && cell.textContent.startsWith("=")) {
      const target = cell.dataset.reference;
      beginFormulaEdit(cell, target, simulator.getCell(target)?.input || "");
    } else if (state.active && state.source === cell) {
      if (cell.textContent.startsWith("=")) updateAuthoring();
      else finishFormulaEdit();
    }
  });

  grid.addEventListener("keyup", (event) => {
    if (state.active && event.target === state.source) updateAuthoring();
  });

  grid.addEventListener("click", (event) => {
    if (state.active && event.target === state.source) updateAuthoring();
  });

  grid.addEventListener("blur", (event) => {
    if (!state.active || event.target !== state.source) return;
    window.setTimeout(() => {
      if (document.activeElement === formulaInput) {
        state.source = formulaInput;
        rememberCaret();
        updateAuthoring();
      } else if (functionPicker.contains(document.activeElement)
        || document.activeElement === functionButton) {
        state.source = formulaInput;
      } else if (!functionPicker.contains(document.activeElement)) {
        finishFormulaEdit();
      }
    });
  }, true);

  formulaInput.addEventListener("focus", () => {
    const target = simulator.getActiveReference?.() || nameBox.value;
    if (simulator.isSpillCell?.(target)) return;
    if (state.active && state.target === target) {
      state.source = formulaInput;
      return;
    }
    state.focusReference = target;
    state.focusStartInput = simulator.getCell(target)?.input || "";
    if (formulaInput.value.startsWith("=")) {
      beginFormulaEdit(formulaInput, target, state.focusStartInput);
    }
  });

  formulaInput.addEventListener("input", () => {
    if (!state.active && formulaInput.value.startsWith("=")) {
      beginFormulaEdit(
        formulaInput,
        state.focusReference || simulator.getActiveReference?.() || nameBox.value,
        state.focusStartInput
      );
    } else if (state.active && state.source === formulaInput) {
      if (formulaInput.value.startsWith("=")) updateAuthoring();
      else finishFormulaEdit();
    }
  });

  formulaInput.addEventListener("keydown", (event) => {
    handleAuthoringKeydown(event, false);
  }, true);

  ["click", "keyup", "select"].forEach((eventName) => {
    formulaInput.addEventListener(eventName, () => {
      if (state.active && state.source === formulaInput) updateAuthoring();
    });
  });

  formulaInput.addEventListener("blur", () => {
    window.setTimeout(() => {
      if (!state.active) return;
      if (functionPicker.contains(document.activeElement)
        || autocomplete.contains(document.activeElement)
        || document.activeElement === functionButton) return;
      finishFormulaEdit();
    });
  });

  autocomplete.addEventListener("mousedown", (event) => event.preventDefault());
  autocomplete.addEventListener("click", (event) => {
    const option = event.target.closest("[data-function-name]");
    if (option) acceptSuggestion(option.dataset.functionName);
  });

  function renderFunctionPicker(query = "") {
    const normalizedQuery = query.trim().toUpperCase();
    const matches = functions.filter((entry) => (
      entry.name.includes(normalizedQuery)
      || entry.shortDescription.toUpperCase().includes(normalizedQuery)
    ));
    pickerResults.replaceChildren();
    if (!matches.length) {
      const empty = document.createElement("p");
      empty.className = "function-picker-empty";
      empty.textContent = "No matching functions.";
      pickerResults.append(empty);
      return;
    }

    let category = "";
    matches.forEach((entry) => {
      if (entry.category !== category) {
        category = entry.category;
        const heading = document.createElement("h3");
        heading.className = "function-picker-category";
        heading.textContent = category;
        pickerResults.append(heading);
      }
      const option = document.createElement("button");
      option.type = "button";
      option.className = "function-picker-option";
      option.dataset.functionName = entry.name;
      const name = document.createElement("strong");
      name.textContent = entry.name;
      const description = document.createElement("span");
      description.textContent = entry.shortDescription;
      option.append(name, description);
      pickerResults.append(option);
    });
  }

  function closeFunctionPicker() {
    functionPicker.hidden = true;
    functionButton.setAttribute("aria-expanded", "false");
  }

  functionButton.addEventListener("click", () => {
    if (simulator.isSpillCell?.(simulator.getActiveReference?.() || nameBox.value)) return;
    const opening = functionPicker.hidden;
    if (!opening) {
      closeFunctionPicker();
      if (state.active) {
        state.source.focus({ preventScroll: true });
        setSelection(state.source, state.caretStart, state.caretEnd);
      }
      return;
    }
    pickerSearch.value = "";
    renderFunctionPicker();
    functionPicker.hidden = false;
    functionButton.setAttribute("aria-expanded", "true");
    pickerSearch.focus();
  });

  pickerSearch.addEventListener("input", () => renderFunctionPicker(pickerSearch.value));
  functionPicker.addEventListener("click", (event) => {
    const option = event.target.closest("[data-function-name]");
    if (!option) return;
    const entry = functionsByName.get(option.dataset.functionName);
    if (!entry) return;

    closeFunctionPicker();
    if (state.active) {
      state.source.focus({ preventScroll: true });
      setSelection(state.source, state.caretStart, state.caretEnd);
      insertText(`${entry.name}(`);
    } else {
      const target = simulator.getActiveReference?.() || nameBox.value;
      const originalInput = simulator.getCell(target)?.input || "";
      formulaInput.focus({ preventScroll: true });
      formulaInput.value = `=${entry.name}(`;
      formulaInput.setSelectionRange(formulaInput.value.length, formulaInput.value.length);
      beginFormulaEdit(formulaInput, target, originalInput);
      formulaInput.dispatchEvent(new Event("input", { bubbles: true }));
    }
  });

  functionPicker.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    event.preventDefault();
    closeFunctionPicker();
    if (state.active) {
      state.source.focus({ preventScroll: true });
      setSelection(state.source, state.caretStart, state.caretEnd);
    } else {
      functionButton.focus();
    }
  });

  document.addEventListener("mousedown", (event) => {
    if (functionPicker.hidden
      || functionPicker.contains(event.target)
      || event.target === functionButton) return;
    closeFunctionPicker();
    if (state.active
      && !grid.contains(event.target)
      && !formulaControls.contains(event.target)) {
      finishFormulaEdit();
    }
  });

  window.ExcelFormulaEditor = Object.freeze({
    getContext() {
      if (!state.active) return null;
      const context = functionContext(sourceText(), state.caretStart);
      return context ? {
        functionName: context.entry.name,
        argumentIndex: context.argumentIndex,
        target: state.target
      } : { functionName: null, argumentIndex: null, target: state.target };
    },
    isEditing() {
      return state.active;
    }
  });
})();
