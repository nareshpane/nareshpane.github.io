(() => {
  "use strict";

  const ROW_COUNT = 50;
  const COLUMN_COUNT = 26;
  const COLUMN_LABELS = Array.from(
    { length: COLUMN_COUNT },
    (_, index) => String.fromCharCode(65 + index)
  );

  const sampleRows = [
    ["Employee ID", "Employee", "Department", "Salary", "Years"],
    ["1001", "Maya", "Finance", "72000", "4"],
    ["1002", "Liam", "IT", "81000", "6"],
    ["1003", "Sofia", "Finance", "68000", "3"],
    ["1004", "Noah", "HR", "64000", "5"],
    ["1005", "Emma", "IT", "89000", "8"],
    ["1006", "Lucas", "Finance", "76000", "5"]
  ];

  const cellData = new Map();
  const cellFormatOverrides = new Map();
  let spillCells = new Map();
  let spillRanges = new Map();
  const cellElements = new Map();
  const rowHeaders = [];
  const columnHeaders = [];

  const grid = document.querySelector("#spreadsheet-grid");
  const nameBox = document.querySelector("#name-box");
  const formulaInput = document.querySelector("#formula-input");
  const formulaFunctionButton = document.querySelector("#formula-function-button");
  const selectionStatus = document.querySelector("#selection-status");
  const explorerTitle = document.querySelector("#function-explorer-title");
  const explorerContent = document.querySelector("#explorer-content");
  const numberFormatSelect = document.querySelector("#number-format-select");
  const currencyFormatButton = document.querySelector("#currency-format-button");
  const percentageFormatButton = document.querySelector("#percentage-format-button");
  const numberFormatButton = document.querySelector("#number-format-button");
  const decreaseDecimalButton = document.querySelector("#decrease-decimal-button");
  const increaseDecimalButton = document.querySelector("#increase-decimal-button");
  const highlightedReferences = new Set();
  const outlinedSpillReferences = new Set();
  const explorerNumberFormat = new Intl.NumberFormat("en-US", {
    maximumSignificantDigits: 12
  });

  const state = {
    activeRow: 0,
    activeColumn: 0,
    editingCell: null,
    editStartInput: "",
    formulaStartInput: "",
    currentDateSerial: null
  };

  function cellReference(row, column) {
    return `${COLUMN_LABELS[column]}${row + 1}`;
  }

  function normalizeInput(input) {
    return String(input).replace(/[\r\n]+/g, " ");
  }

  function literalValue(input) {
    const trimmed = input.trim();
    const isNumber = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$/.test(trimmed);
    return isNumber ? Number(trimmed) : input;
  }

  function createCellModel(input) {
    if (!input.startsWith("=")) {
      return {
        input,
        value: literalValue(input),
        type: "literal",
        ast: null,
        dependencies: [],
        inferredNumberFormat: window.ExcelFormatting.NUMBER_FORMATS.GENERAL,
        numberFormatOverride: null,
        numberFormat: window.ExcelFormatting.NUMBER_FORMATS.GENERAL,
        formatOptions: window.ExcelFormatting.normalizeFormatOptions("General")
      };
    }

    try {
      const ast = window.FormulaEngine.parseFormula(input);
      return {
        input,
        value: "",
        type: "formula",
        ast,
        dependencies: window.FormulaEngine.collectReferences(ast),
        inferredNumberFormat: window.ExcelFormatting.NUMBER_FORMATS.GENERAL,
        numberFormatOverride: null,
        numberFormat: window.ExcelFormatting.NUMBER_FORMATS.GENERAL,
        formatOptions: window.ExcelFormatting.normalizeFormatOptions("General")
      };
    } catch (error) {
      return {
        input,
        value: window.FormulaEngine.ERROR_VALUES.GENERIC,
        type: "formula",
        ast: null,
        dependencies: [],
        inferredNumberFormat: window.ExcelFormatting.NUMBER_FORMATS.GENERAL,
        numberFormatOverride: null,
        numberFormat: window.ExcelFormatting.NUMBER_FORMATS.GENERAL,
        formatOptions: window.ExcelFormatting.normalizeFormatOptions("General")
      };
    }
  }

  function getCellModel(row, column) {
    const reference = cellReference(row, column);
    return cellData.get(reference) || spillCells.get(reference) || null;
  }

  function cellInput(row, column) {
    return getCellModel(row, column)?.input || "";
  }

  function calculatedCellValue(reference) {
    return cellData.get(reference)?.value ?? spillCells.get(reference)?.value ?? "";
  }

  function calculatedCellNumberFormat(reference) {
    return spillCells.get(reference)?.numberFormat
      || cellData.get(reference)?.numberFormat
      || window.ExcelFormatting.NUMBER_FORMATS.GENERAL;
  }

  function evaluateAstForExplanation(ast) {
    return window.FormulaEngine.evaluate(ast, {
      getCellValue: calculatedCellValue,
      getRangeValues(start, end) {
        return window.FormulaEngine.expandRange(start, end).map(calculatedCellValue);
      },
      getCellNumberFormat: calculatedCellNumberFormat,
      getCurrentDateSerial: () => state.currentDateSerial
    });
  }

  function analyzeConditionalForExplanation(ast) {
    return window.FormulaEngine.analyzeConditionalAggregate(ast, {
      getCellValue: calculatedCellValue,
      getRangeValues(start, end) {
        return window.FormulaEngine.expandRange(start, end).map(calculatedCellValue);
      }
    });
  }

  function analyzeLookupForExplanation(ast) {
    return window.FormulaEngine.analyzeLookupExpression(ast, {
      getCellValue: calculatedCellValue,
      getRangeValues(start, end) {
        return window.FormulaEngine.expandRange(start, end).map(calculatedCellValue);
      }
    });
  }

  function analyzeTextForExplanation(ast) {
    return window.FormulaEngine.analyzeTextExpression(ast, {
      getCellValue: calculatedCellValue,
      getRangeValues(start, end) {
        return window.FormulaEngine.expandRange(start, end).map(calculatedCellValue);
      }
    });
  }

  function analyzeDateForExplanation(ast) {
    return window.FormulaEngine.analyzeDateExpression(ast, {
      getCellValue: calculatedCellValue,
      getRangeValues(start, end) {
        return window.FormulaEngine.expandRange(start, end).map(calculatedCellValue);
      },
      getCellNumberFormat: calculatedCellNumberFormat,
      getCurrentDateSerial: () => state.currentDateSerial
    });
  }

  function analyzeMathForExplanation(ast) {
    return window.FormulaEngine.analyzeMathExpression(ast, {
      getCellValue: calculatedCellValue,
      getRangeValues(start, end) {
        return window.FormulaEngine.expandRange(start, end).map(calculatedCellValue);
      }
    });
  }

  function analyzeErrorForExplanation(ast) {
    return window.FormulaEngine.analyzeErrorExpression(ast, {
      getCellValue: calculatedCellValue,
      getRangeValues(start, end) {
        return window.FormulaEngine.expandRange(start, end).map(calculatedCellValue);
      },
      getCurrentDateSerial: () => state.currentDateSerial
    });
  }

  function analyzeDynamicForExplanation(ast) {
    return window.FormulaEngine.analyzeDynamicArrayExpression(ast, {
      getCellValue: calculatedCellValue,
      getCellNumberFormat: calculatedCellNumberFormat,
      getCurrentDateSerial: () => state.currentDateSerial
    });
  }

  function displayedValue(model) {
    if (!model) return "";
    if (model.type === "literal"
      && (model.numberFormat === window.ExcelFormatting.NUMBER_FORMATS.GENERAL
        || typeof model.value !== "number")) return model.input;
    return window.ExcelFormatting.formatValue(
      model.value,
      model.numberFormat,
      model.formatOptions
    );
  }

  function renderCell(reference) {
    const cell = cellElements.get(reference);
    if (!cell || cell === state.editingCell) return;

    const projection = spillCells.get(reference);
    const model = projection || cellData.get(reference);
    const display = displayedValue(model);
    cell.textContent = display;
    cell.classList.toggle("spill-cell", Boolean(projection && projection.spillOwner !== reference));
    cell.classList.toggle("spill-anchor", Boolean(projection && projection.spillOwner === reference));
    if (projection) {
      cell.dataset.spillOwner = projection.spillOwner;
      cell.setAttribute("aria-readonly", projection.spillOwner === reference ? "false" : "true");
      cell.setAttribute(
        "aria-label",
        `${reference}: ${display || "blank"}. Spilled from ${projection.spillOwner}`
      );
    } else {
      delete cell.dataset.spillOwner;
      cell.removeAttribute("aria-readonly");
      cell.setAttribute("aria-label", display ? `${reference}: ${display}` : reference);
    }
  }

  function renderAllCells() {
    cellElements.forEach((_, reference) => renderCell(reference));
  }

  function recalculateAll() {
    state.currentDateSerial = window.ExcelFormatting.todaySerial();
    function applyCellFormat(reference, model, inferredNumberFormat) {
      const override = cellFormatOverrides.get(reference) || null;
      model.inferredNumberFormat = inferredNumberFormat;
      model.numberFormatOverride = override?.type || null;
      model.numberFormat = override?.type || inferredNumberFormat;
      model.formatOptions = window.ExcelFormatting.normalizeFormatOptions(
        model.numberFormat,
        override || {}
      );
      return model.numberFormat;
    }

    function worksheetOrder(left, right) {
      const first = window.FormulaEngine.parseReference(left);
      const second = window.FormulaEngine.parseReference(right);
      return first.row - second.row || first.column - second.column;
    }

    const formulaReferences = [...cellData.entries()]
      .filter(([, model]) => model.type === "formula")
      .map(([reference]) => reference)
      .sort(worksheetOrder);
    let spillHints = spillCells;
    let previousSignature = "";

    for (let pass = 0; pass < 5; pass += 1) {
      const nextSpillCells = new Map();
      const nextSpillRanges = new Map();
      const calculationState = new Map();
      const formatState = new Map();

      cellData.forEach((model) => {
        if (model.type !== "formula") return;
        delete model.arrayResult;
        delete model.spillRange;
        delete model.spillRows;
        delete model.spillColumns;
        delete model.spillError;
      });

      function resolveCellNumberFormat(reference) {
        const projection = nextSpillCells.get(reference);
        if (projection && projection.spillOwner !== reference) return projection.numberFormat;
        const model = cellData.get(reference);
        if (!model || model.type !== "formula" || !model.ast) {
          if (model) {
            return applyCellFormat(
              reference,
              model,
              window.ExcelFormatting.NUMBER_FORMATS.GENERAL
            );
          }
          return projection?.numberFormat
            || cellFormatOverrides.get(reference)?.type
            || window.ExcelFormatting.NUMBER_FORMATS.GENERAL;
        }
        if (formatState.get(reference) === "resolving") {
          return window.ExcelFormatting.NUMBER_FORMATS.GENERAL;
        }
        if (formatState.get(reference) === "resolved") return model.numberFormat;
        formatState.set(reference, "resolving");
        const inferredNumberFormat = window.FormulaEngine.inferNumberFormat(
          model.ast,
          resolveCellNumberFormat
        );
        applyCellFormat(reference, model, inferredNumberFormat);
        formatState.set(reference, "resolved");
        return model.numberFormat;
      }

      function spillDestination(owner, rowOffset, columnOffset) {
        const anchor = window.FormulaEngine.parseReference(owner);
        const row = anchor.row + rowOffset;
        const column = anchor.column + columnOffset;
        if (row < 0 || row >= ROW_COUNT || column < 0 || column >= COLUMN_COUNT) return null;
        return cellReference(row, column);
      }

      function spillBlocker(owner, array) {
        for (let row = 0; row < array.rows; row += 1) {
          for (let column = 0; column < array.columns; column += 1) {
            const destination = spillDestination(owner, row, column);
            if (!destination) return { type: "boundary", reference: null, value: null };
            if (destination !== owner && cellData.has(destination)) {
              return {
                type: "cell",
                reference: destination,
                value: cellData.get(destination).value
              };
            }
            const existingProjection = nextSpillCells.get(destination);
            if (existingProjection && existingProjection.spillOwner !== owner) {
              return {
                type: "cell",
                reference: destination,
                value: existingProjection.value
              };
            }
          }
        }
        return null;
      }

      function placeSpill(owner, model, array) {
        const blocker = spillBlocker(owner, array);
        const end = spillDestination(owner, array.rows - 1, array.columns - 1);
        model.arrayResult = array;
        model.spillRows = array.rows;
        model.spillColumns = array.columns;
        model.spillRange = end ? `${owner}:${end}` : null;
        if (blocker) {
          model.spillError = {
            ...blocker,
            requiredRange: end ? `${owner}:${end}` : `${owner}:outside worksheet`
          };
          return false;
        }

        const descriptor = {
          owner,
          start: owner,
          end,
          range: `${owner}:${end}`,
          rows: array.rows,
          columns: array.columns,
          values: array.values.map((row) => row.slice()),
          formats: array.formats?.map((row) => row.slice()) || null,
          references: []
        };
        for (let row = 0; row < array.rows; row += 1) {
          for (let column = 0; column < array.columns; column += 1) {
            const destination = spillDestination(owner, row, column);
            const numberFormat = array.formats?.[row]?.[column]
              || window.ExcelFormatting.NUMBER_FORMATS.GENERAL;
            const projection = {
              input: "",
              value: array.values[row][column],
              type: destination === owner ? "spill-anchor" : "spill",
              ast: null,
              dependencies: [],
              spillOwner: owner,
              spillRowOffset: row,
              spillColumnOffset: column,
              numberFormat,
              inferredNumberFormat: numberFormat,
              numberFormatOverride: null,
              formatOptions: window.ExcelFormatting.normalizeFormatOptions(numberFormat)
            };
            nextSpillCells.set(destination, projection);
            descriptor.references.push(destination);
          }
        }
        nextSpillRanges.set(owner, descriptor);
        return true;
      }

      function evaluateCell(reference) {
        const model = cellData.get(reference);
        if (!model) {
          const projection = nextSpillCells.get(reference);
          if (projection) return projection.value;
          const hint = spillHints.get(reference);
          if (hint) {
            evaluateCell(hint.spillOwner);
            return nextSpillCells.get(reference)?.value ?? "";
          }
          return "";
        }
        if (model.type === "literal") return model.value;
        if (calculationState.get(reference) === "evaluating") {
          return window.FormulaEngine.ERROR_VALUES.GENERIC;
        }
        if (calculationState.get(reference) === "evaluated") return model.value;
        calculationState.set(reference, "evaluating");

        if (!model.ast) {
          model.value = window.FormulaEngine.ERROR_VALUES.GENERIC;
        } else {
          try {
            const result = window.FormulaEngine.evaluate(model.ast, {
              getCellValue: evaluateCell,
              getCellNumberFormat: resolveCellNumberFormat,
              getCurrentDateSerial: () => state.currentDateSerial
            });
            if (window.FormulaEngine.isArrayValue(result)) {
              model.value = placeSpill(reference, model, result)
                ? result.values[0][0]
                : window.FormulaEngine.ERROR_VALUES.SPILL;
            } else {
              model.value = result;
            }
            model.numberFormat = resolveCellNumberFormat(reference);
            if (model.inferredNumberFormat === window.ExcelFormatting.NUMBER_FORMATS.DATE
              && typeof model.value === "number"
              && !window.ExcelFormatting.serialToCalendar(model.value)) {
              model.value = window.FormulaEngine.ERROR_VALUES.NUM;
            }
          } catch (error) {
            model.value = error instanceof window.FormulaEngine.FormulaError
              ? error.code
              : window.FormulaEngine.ERROR_VALUES.GENERIC;
          }
        }
        calculationState.set(reference, "evaluated");
        return model.value;
      }

      formulaReferences.forEach(evaluateCell);
      cellData.forEach((model, reference) => resolveCellNumberFormat(reference));

      nextSpillRanges.forEach((descriptor, owner) => {
        const anchorModel = cellData.get(owner);
        descriptor.references.forEach((reference) => {
          const projection = nextSpillCells.get(reference);
          if (reference === owner && anchorModel?.numberFormatOverride) {
            projection.numberFormat = anchorModel.numberFormat;
            projection.formatOptions = anchorModel.formatOptions;
          }
        });
      });

      const signature = JSON.stringify({
        values: formulaReferences.map((reference) => [reference, cellData.get(reference).value]),
        spills: [...nextSpillRanges.values()].map((spill) => [spill.owner, spill.range, spill.values])
      });
      spillCells = nextSpillCells;
      spillRanges = nextSpillRanges;
      if (signature === previousSignature) break;
      previousSignature = signature;
      spillHints = nextSpillCells;
    }

    renderAllCells();
    updateSpillOutline();
  }

  function storeCellInput(row, column, input) {
    const reference = cellReference(row, column);
    const normalizedInput = normalizeInput(input);

    if (normalizedInput) {
      cellData.set(reference, createCellModel(normalizedInput));
    } else {
      cellData.delete(reference);
    }

    return normalizedInput;
  }

  function setCellInput(row, column, input) {
    const reference = cellReference(row, column);
    const projection = spillCells.get(reference);
    if (projection && projection.spillOwner !== reference) {
      selectionStatus.textContent = `You can't change part of an array. Edit ${projection.spillOwner}.`;
      renderCell(reference);
      return cellData.get(reference)?.input || "";
    }
    const normalizedInput = storeCellInput(row, column, input);

    recalculateAll();
    if (cellElements.size) updateFormulaTrace();
    return normalizedInput;
  }

  function setCellNumberFormat(reference, numberFormat, options = {}) {
    coordinatesForReference(reference);
    const projection = spillCells.get(reference.toUpperCase());
    if (projection && projection.spillOwner !== reference.toUpperCase()) {
      selectionStatus.textContent = `Format the anchor ${projection.spillOwner} instead.`;
      return false;
    }
    if (!Object.values(window.ExcelFormatting.NUMBER_FORMATS).includes(numberFormat)) {
      throw new Error(`Unsupported number format: ${numberFormat}`);
    }
    cellFormatOverrides.set(reference.toUpperCase(), {
      type: numberFormat,
      ...window.ExcelFormatting.normalizeFormatOptions(numberFormat, options)
    });
    recalculateAll();
    if (cellElements.size) updateFormulaTrace();
    return true;
  }

  function getCellFormat(reference) {
    coordinatesForReference(reference);
    const override = cellFormatOverrides.get(reference.toUpperCase());
    return override ? { ...override } : null;
  }

  function coordinatesForReference(reference) {
    const coordinates = window.FormulaEngine.parseReference(reference);
    if (
      coordinates.row < 0
      || coordinates.row >= ROW_COUNT
      || coordinates.column < 0
      || coordinates.column >= COLUMN_COUNT
    ) {
      throw new Error(`Cell reference outside worksheet: ${reference}`);
    }
    return coordinates;
  }

  function setCellInputs(updates) {
    const entries = Array.isArray(updates)
      ? updates
      : Object.entries(updates).map(([cell, value]) => ({ cell, value }));

    entries.forEach((entry) => {
      const reference = entry.cell;
      const input = entry.value;
      const { row, column } = coordinatesForReference(reference);
      const projection = spillCells.get(reference.toUpperCase());
      if (projection && projection.spillOwner !== reference.toUpperCase()) return;
      storeCellInput(row, column, input);
      if (entry.numberFormat) {
        if (!Object.values(window.ExcelFormatting.NUMBER_FORMATS).includes(entry.numberFormat)) {
          throw new Error(`Unsupported number format: ${entry.numberFormat}`);
        }
        cellFormatOverrides.set(reference.toUpperCase(), {
          type: entry.numberFormat,
          ...window.ExcelFormatting.normalizeFormatOptions(entry.numberFormat, entry.formatOptions)
        });
      }
    });
    recalculateAll();
    if (cellElements.size) updateFormulaTrace();
  }

  function clearCellRange(start, end) {
    const references = window.FormulaEngine.expandRange(start, end);
    const spillOwners = new Set();
    references.forEach((reference) => {
      const projection = spillCells.get(reference);
      if (projection) spillOwners.add(projection.spillOwner);
    });
    spillOwners.forEach((owner) => {
      cellData.delete(owner);
      cellFormatOverrides.delete(owner);
    });
    references.forEach((reference) => {
      const { row, column } = coordinatesForReference(reference);
      cellFormatOverrides.delete(reference);
      storeCellInput(row, column, "");
    });
    recalculateAll();
    if (cellElements.size) updateFormulaTrace();
  }

  function seedSampleData() {
    sampleRows.forEach((rowValues, row) => {
      rowValues.forEach((input, column) => {
        cellData.set(cellReference(row, column), createCellModel(input));
      });
    });
    recalculateAll();
  }

  function createGrid() {
    const fragment = document.createDocumentFragment();
    const corner = document.createElement("div");
    corner.className = "corner-cell";
    corner.setAttribute("aria-hidden", "true");
    fragment.append(corner);

    COLUMN_LABELS.forEach((label) => {
      const header = document.createElement("div");
      header.className = "column-header";
      header.textContent = label;
      header.setAttribute("role", "columnheader");
      columnHeaders.push(header);
      fragment.append(header);
    });

    for (let row = 0; row < ROW_COUNT; row += 1) {
      const rowHeader = document.createElement("div");
      rowHeader.className = "row-header";
      rowHeader.textContent = String(row + 1);
      rowHeader.setAttribute("role", "rowheader");
      rowHeaders.push(rowHeader);
      fragment.append(rowHeader);

      for (let column = 0; column < COLUMN_COUNT; column += 1) {
        const reference = cellReference(row, column);
        const display = displayedValue(cellData.get(reference));
        const cell = document.createElement("div");
        cell.className = "sheet-cell";
        cell.dataset.row = String(row);
        cell.dataset.column = String(column);
        cell.dataset.reference = reference;
        cell.textContent = display;
        cell.tabIndex = -1;
        cell.setAttribute("role", "gridcell");
        cell.setAttribute("aria-label", display ? `${reference}: ${display}` : reference);
        cell.setAttribute("aria-selected", "false");
        cell.setAttribute("contenteditable", "false");
        cell.setAttribute("spellcheck", "false");

        if (row === 0 && column < sampleRows[0].length) {
          cell.classList.add("sample-header");
        }

        cellElements.set(reference, cell);
        fragment.append(cell);
      }
    }

    grid.append(fragment);
  }

  function activeCellElement() {
    return cellElements.get(cellReference(state.activeRow, state.activeColumn));
  }

  function scrollCellIntoView(cell) {
    const viewport = grid.parentElement;
    const headerHeight = columnHeaders[0]?.offsetHeight || 0;
    const headerWidth = rowHeaders[0]?.offsetWidth || 0;
    const cellTop = cell.offsetTop - grid.offsetTop;
    const cellLeft = cell.offsetLeft - grid.offsetLeft;

    if (cellTop < viewport.scrollTop + headerHeight) {
      viewport.scrollTop = Math.max(0, cellTop - headerHeight);
    } else if (cellTop + cell.offsetHeight > viewport.scrollTop + viewport.clientHeight) {
      viewport.scrollTop = cellTop + cell.offsetHeight - viewport.clientHeight;
    }

    if (cellLeft < viewport.scrollLeft + headerWidth) {
      viewport.scrollLeft = Math.max(0, cellLeft - headerWidth);
    } else if (cellLeft + cell.offsetWidth > viewport.scrollLeft + viewport.clientWidth) {
      viewport.scrollLeft = cellLeft + cell.offsetWidth - viewport.clientWidth;
    }
  }

  function createExplorerElement(tagName, className, text) {
    const element = document.createElement(tagName);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = text;
    return element;
  }

  function explorerValue(
    value,
    numberFormat = window.ExcelFormatting.NUMBER_FORMATS.GENERAL,
    formatOptions = {}
  ) {
    if (numberFormat !== window.ExcelFormatting.NUMBER_FORMATS.GENERAL) {
      return window.ExcelFormatting.formatValue(value, numberFormat, formatOptions);
    }
    if (typeof value === "boolean") return value ? "TRUE" : "FALSE";
    if (typeof value === "number" && Number.isFinite(value)) {
      return explorerNumberFormat.format(value);
    }
    if (value === "" || value === null || value === undefined) return "(blank)";
    return String(value);
  }

  function explorerFieldName(label) {
    return label.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  }

  function createExplorerSection(label, content, fieldName) {
    const section = createExplorerElement("section", "explorer-section");
    section.dataset.explorerField = fieldName || explorerFieldName(label);
    section.append(createExplorerElement("div", "explorer-label", label));

    if (content instanceof Node) {
      section.append(content);
    } else {
      section.append(createExplorerElement("div", "explorer-value", content));
    }

    return section;
  }

  function createReferenceList(entries) {
    const list = createExplorerElement("div", "explorer-reference-list");

    entries.forEach((entry) => {
      const row = createExplorerElement("div", "explorer-reference-row");
      row.dataset.reference = entry.reference;
      row.append(createExplorerElement("span", "explorer-reference", entry.reference));
      const valueClass = entry.ignored
        ? "explorer-reference-value explorer-ignored"
        : "explorer-reference-value";
      const suffix = entry.ignored ? " (ignored)" : "";
      row.append(createExplorerElement(
        "span",
        valueClass,
        `${explorerValue(entry.value, entry.numberFormat)}${suffix}`
      ));
      list.append(row);
    });

    return list;
  }

  function visibleSpaces(value) {
    return String(value).replaceAll(" ", "·") || "(blank)";
  }

  function quotedText(value) {
    return `"${String(value).replaceAll("\"", "\"\"")}"`;
  }

  function createCharacterStrip(characters) {
    const strip = createExplorerElement("div", "text-character-strip");
    strip.tabIndex = 0;
    strip.setAttribute("role", "group");
    strip.setAttribute("aria-label", characters.map((entry) => (
      `Character ${entry.position}: ${entry.character === " " ? "space" : entry.character}${entry.selected ? ", selected" : ""}`
    )).join("; "));

    characters.forEach((entry) => {
      const cell = createExplorerElement(
        "div",
        `text-character${entry.selected ? " selected" : ""}`
      );
      cell.append(createExplorerElement(
        "span",
        "text-character-value",
        entry.character === " " ? "·" : entry.character
      ));
      cell.append(createExplorerElement("span", "text-character-position", entry.position));
      strip.append(cell);
    });
    return strip;
  }

  function selectedCharacterText(characters) {
    return characters
      .filter((entry) => entry.selected)
      .map((entry) => entry.character === " " ? "·" : entry.character)
      .join(" ") || "(none)";
  }

  function createTextPieceList(pieces) {
    const list = createExplorerElement("div", "text-piece-list");
    pieces.forEach((piece, index) => {
      const row = createExplorerElement(
        "div",
        `text-piece-row${piece.included ? "" : " excluded"}`
      );
      if (piece.reference) row.dataset.reference = piece.reference;
      row.append(createExplorerElement("span", "text-piece-number", index + 1));
      let source;
      if (piece.sourceType === "reference") {
        source = `${piece.source} → ${piece.text === "" ? "(blank)" : piece.text}`;
      } else if (piece.sourceType === "literal") {
        source = quotedText(piece.source);
      } else {
        source = piece.text === "" ? "(blank)" : piece.text;
      }
      row.append(createExplorerElement("span", "text-piece-value", source));
      if (!piece.included) row.append(createExplorerElement("span", "text-piece-status", "skipped"));
      list.append(row);
    });
    return list;
  }

  function createTextChangeList(changes) {
    const list = createExplorerElement("ul", "text-change-list");
    const items = [
      [changes.leading, "Leading spaces removed", "No leading spaces to remove"],
      [changes.interior, "Repeated interior spaces reduced", "Interior spacing unchanged"],
      [changes.trailing, "Trailing spaces removed", "No trailing spaces to remove"]
    ];
    items.forEach(([changed, yes, no]) => {
      list.append(createExplorerElement("li", changed ? "changed" : "unchanged", changed ? yes : no));
    });
    return list;
  }

  function renderTextExplorer(fragment, text) {
    if (text.kind === "characters") {
      fragment.append(createExplorerSection("Text", text.text, "text-source"));
      fragment.append(createExplorerSection(
        "Characters",
        createCharacterStrip(text.characters),
        "text-characters"
      ));
      if (text.functionName === "LEN") {
        fragment.append(createExplorerSection("Character count", explorerValue(text.count), "character-count"));
      } else {
        if (text.functionName === "RIGHT") {
          fragment.append(createExplorerSection(
            "Direction",
            "Take characters from the end; character positions remain left to right.",
            "text-direction"
          ));
        }
        if (text.functionName === "MID") {
          fragment.append(createExplorerSection("Start position", explorerValue(text.start), "start-position"));
          fragment.append(createExplorerSection("Characters requested", explorerValue(text.count), "character-count"));
          fragment.append(createExplorerSection("Selected text", text.selectedText, "selected-text"));
        } else {
          fragment.append(createExplorerSection("Number of characters", explorerValue(text.count), "character-count"));
          fragment.append(createExplorerSection(
            "Selected characters",
            selectedCharacterText(text.characters),
            "selected-characters"
          ));
        }
      }
      return;
    }

    if (text.kind === "trim") {
      fragment.append(createExplorerSection("Before", visibleSpaces(text.before), "text-before"));
      fragment.append(createExplorerSection("After", visibleSpaces(text.after), "text-after"));
      fragment.append(createExplorerSection(
        "Spacing changes",
        createTextChangeList(text.changes),
        "spacing-changes"
      ));
      return;
    }

    if (text.kind === "case") {
      fragment.append(createExplorerSection("Original", text.before, "text-before"));
      fragment.append(createExplorerSection("Transformation", text.transformation, "text-transformation"));
      return;
    }

    if (text.kind === "concat") {
      fragment.append(createExplorerSection("Pieces", createTextPieceList(text.pieces), "text-pieces"));
      const combined = text.pieces.map((piece) => (
        piece.sourceType === "literal"
          ? quotedText(piece.source)
          : (piece.text === "" ? '""' : piece.text)
      )).join(" + ");
      fragment.append(createExplorerSection("Combined", combined, "text-combined"));
      return;
    }

    if (text.kind === "textjoin") {
      fragment.append(createExplorerSection("Delimiter", quotedText(text.delimiter), "text-delimiter"));
      fragment.append(createExplorerSection(
        "Ignore empty",
        text.ignoreEmpty ? "TRUE" : "FALSE",
        "ignore-empty"
      ));
      fragment.append(createExplorerSection("Values", createTextPieceList(text.pieces), "text-pieces"));
      fragment.append(createExplorerSection("Joined result", text.result, "joined-result"));
      return;
    }

    if (text.kind === "search") {
      fragment.append(createExplorerSection("Search for", text.findText, "search-text"));
      fragment.append(createExplorerSection("Inside", text.withinText, "within-text"));
      fragment.append(createExplorerSection(
        "Case-sensitive",
        text.caseSensitive ? "YES" : "NO",
        "case-sensitive"
      ));
      if (text.start > 1) {
        fragment.append(createExplorerSection("Start position", explorerValue(text.start), "start-position"));
      }
      fragment.append(createExplorerSection(
        "Character positions",
        createCharacterStrip(text.characters),
        "text-characters"
      ));
      fragment.append(createExplorerSection(
        "Match begins at character",
        text.matchPosition ? explorerValue(text.matchPosition) : "Not found",
        "match-position"
      ));
      return;
    }

    if (text.kind === "substitute") {
      fragment.append(createExplorerSection("Find", text.oldText, "substitute-find"));
      fragment.append(createExplorerSection("Replace with", text.newText, "substitute-replacement"));
      fragment.append(createExplorerSection(
        "Occurrence",
        text.instance === null ? "All" : explorerValue(text.instance),
        "substitute-occurrence"
      ));
      fragment.append(createExplorerSection("Before", text.before, "text-before"));
      fragment.append(createExplorerSection("After", text.after, "text-after"));
      return;
    }

    if (text.kind === "replace") {
      fragment.append(createExplorerSection("Original", text.before, "text-before"));
      fragment.append(createExplorerSection(
        "Character positions",
        createCharacterStrip(text.characters),
        "text-characters"
      ));
      fragment.append(createExplorerSection("Start", explorerValue(text.start), "start-position"));
      fragment.append(createExplorerSection(
        "Characters replaced",
        explorerValue(text.replacedCount),
        "character-count"
      ));
      fragment.append(createExplorerSection("Replacement", text.newText, "replacement-text"));
    }
  }

  function renderDateExplorer(fragment, date) {
    if (date.kind === "date-construction") {
      fragment.append(createExplorerSection("Year", String(date.requested.year), "date-year"));
      fragment.append(createExplorerSection("Month", String(date.requested.month), "date-month"));
      fragment.append(createExplorerSection("Day", String(date.requested.day), "date-day"));
      fragment.append(createExplorerSection("Calendar result", date.resultDate, "calendar-result"));
      fragment.append(createExplorerSection(
        "Underlying Excel-style serial",
        explorerValue(date.result),
        "date-serial"
      ));
      fragment.append(createExplorerSection(
        "How dates work",
        "Dates are stored as numbers and displayed using a date format.",
        "date-note"
      ));
      return;
    }

    if (date.kind === "today") {
      fragment.append(createExplorerSection("Current calendar date", date.resultDate, "calendar-result"));
      fragment.append(createExplorerSection(
        "Underlying Excel-style serial",
        explorerValue(date.result),
        "date-serial"
      ));
      fragment.append(createExplorerSection(
        "How dates work",
        "TODAY is recalculated from the current calendar date; the date remains numeric internally.",
        "date-note"
      ));
      return;
    }

    if (date.kind === "date-component") {
      if (date.sourceReference) {
        fragment.append(createExplorerSection("Source cell", date.sourceReference, "source-cell"));
      }
      fragment.append(createExplorerSection("Displayed date", date.sourceDate, "source-date"));
      fragment.append(createExplorerSection("Requested component", date.component, "date-component"));
      return;
    }

    if (date.kind === "date-arithmetic") {
      fragment.append(createExplorerSection("Start date", date.startDate, "start-date"));
      fragment.append(createExplorerSection(
        date.operation === "add" ? "Add" : "Subtract",
        `${explorerValue(date.days)} day${date.days === 1 ? "" : "s"}`,
        "day-change"
      ));
      fragment.append(createExplorerSection("Calendar result", date.resultDate, "calendar-result"));
      return;
    }

    if (date.kind === "date-difference") {
      fragment.append(createExplorerSection("Start", date.startDate, "start-date"));
      fragment.append(createExplorerSection("End", date.endDate, "end-date"));
      fragment.append(createExplorerSection(
        "Difference",
        `${explorerValue(date.difference)} day${Math.abs(date.difference) === 1 ? "" : "s"}`,
        "date-difference"
      ));
      return;
    }

    if (date.kind === "month-shift") {
      const direction = date.months < 0 ? "backward" : "forward";
      fragment.append(createExplorerSection("Start date", date.startDate, "start-date"));
      fragment.append(createExplorerSection(
        "Move",
        date.months === 0
          ? "0 months"
          : `${Math.abs(date.months)} month${Math.abs(date.months) === 1 ? "" : "s"} ${direction}`,
        "month-movement"
      ));
      fragment.append(createExplorerSection("Calendar result", date.resultDate, "calendar-result"));
      return;
    }

    if (date.kind === "month-end") {
      fragment.append(createExplorerSection("Start date", date.startDate, "start-date"));
      fragment.append(createExplorerSection("Month offset", explorerValue(date.months), "month-offset"));
      fragment.append(createExplorerSection("End of target month", date.resultDate, "calendar-result"));
      return;
    }

    if (date.kind === "weekday") {
      fragment.append(createExplorerSection("Date", date.sourceDate, "source-date"));
      fragment.append(createExplorerSection("Return type", explorerValue(date.returnType), "return-type"));
      fragment.append(createExplorerSection("Week starts", date.weekStarts, "week-start"));
      fragment.append(createExplorerSection("Day", date.dayName, "weekday-name"));
    }
  }

  function renderMathExplorer(fragment, math) {
    if (math.kind === "rounding") {
      fragment.append(createExplorerSection("Original number", explorerValue(math.number), "original-number"));
      fragment.append(createExplorerSection("Decimal places", explorerValue(math.digits), "decimal-places"));
      if (math.digits < 0) {
        fragment.append(createExplorerSection(
          "Position",
          "Negative decimal places round to the left of the decimal point.",
          "round-position"
        ));
      }
      fragment.append(createExplorerSection("Direction", math.direction, "round-direction"));
      fragment.append(createExplorerSection("Rounded value", explorerValue(math.result), "rounded-value"));
      return;
    }
    if (math.kind === "integer") {
      fragment.append(createExplorerSection("Number", explorerValue(math.number), "original-number"));
      fragment.append(createExplorerSection(
        "How INT works",
        "INT rounds downward to the nearest integer, including for negative numbers.",
        "int-rule"
      ));
      return;
    }
    if (math.kind === "absolute") {
      fragment.append(createExplorerSection("Number", explorerValue(math.number), "original-number"));
      fragment.append(createExplorerSection("Transformation", "Distance from zero", "absolute-rule"));
      return;
    }
    if (math.kind === "modulo") {
      fragment.append(createExplorerSection(
        "Division identity",
        `${explorerValue(math.number)} = ${explorerValue(math.divisor)} × ${explorerValue(math.quotient)} + ${explorerValue(math.result)}`,
        "mod-identity"
      ));
      fragment.append(createExplorerSection("Remainder", explorerValue(math.result), "remainder"));
    }
  }

  function renderErrorHandlingExplorer(fragment, trace) {
    fragment.append(createExplorerSection(
      trace.functionName === "IFNA" ? "Lookup or primary expression" : "Primary expression",
      trace.primaryExpression,
      "primary-expression"
    ));
    fragment.append(createExplorerSection("Primary result", explorerValue(trace.primaryResult), "primary-result"));
    if (trace.caught) {
      fragment.append(createExplorerSection(
        "Error handling",
        trace.functionName === "IFNA"
          ? "IFNA catches #N/A, so the fallback is returned."
          : "An error occurred, so IFERROR returns the fallback.",
        "error-handling-rule"
      ));
      fragment.append(createExplorerSection("Fallback", trace.fallbackExpression, "fallback-expression"));
      fragment.append(createExplorerSection("Returned value", explorerValue(trace.result), "returned-value"));
    } else if (trace.error) {
      fragment.append(createExplorerSection(
        "Error handling",
        `IFNA catches only #N/A, so ${trace.error} remains an error.`,
        "error-handling-rule"
      ));
    } else {
      fragment.append(createExplorerSection(
        "Error handling",
        "No error occurred, so the fallback was not evaluated.",
        "error-handling-rule"
      ));
    }
  }

  function createLogicalTestList(tests) {
    const list = createExplorerElement("div", "explorer-test-list");

    tests.forEach((test) => {
      const item = createExplorerElement("div", "explorer-test");
      item.append(createExplorerElement(
        "div",
        "explorer-test-expression",
        test.expression
      ));
      item.append(createExplorerElement(
        "div",
        "explorer-test-calculation",
        test.calculation
      ));
      item.append(createExplorerElement(
        "div",
        "explorer-test-result",
        explorerValue(test.result)
      ));
      list.append(item);
    });

    return list;
  }

  function createConditionalList(conditional) {
    const list = createExplorerElement("div", "conditional-list");

    conditional.positions.forEach((position) => {
      const status = position.allMatched ? "match" : "no-match";
      const item = createExplorerElement("div", `conditional-position ${status}`);
      item.dataset.matched = String(position.allMatched);
      const heading = createExplorerElement("div", "conditional-position-head");
      heading.append(createExplorerElement("span", "", `Position ${position.index + 1}`));
      heading.append(createExplorerElement(
        "span",
        `conditional-status ${status}`,
        position.allMatched ? "MATCH" : "no match"
      ));
      item.append(heading);

      position.checks.forEach((check) => {
        const checkStatus = check.matched ? "match" : "no-match";
        const row = createExplorerElement("div", "conditional-check");
        row.dataset.reference = check.reference;
        row.append(createExplorerElement(
          "span",
          "conditional-check-reference",
          check.reference
        ));
        row.append(createExplorerElement("span", "", check.comparison));
        row.append(createExplorerElement(
          "span",
          `conditional-check-result ${checkStatus}`,
          check.matched ? "✓" : "×"
        ));
        item.append(row);
      });

      if (position.aggregate) {
        const aggregate = createExplorerElement("div", "conditional-aggregate");
        aggregate.append(createExplorerElement(
          "span",
          "conditional-aggregate-reference",
          position.aggregate.reference
        ));
        const action = position.aggregate.included
          ? `Include ${explorerValue(position.aggregate.value)}`
          : `Exclude ${explorerValue(position.aggregate.value)}`;
        aggregate.append(createExplorerElement("span", "", action));
        item.append(aggregate);
      }

      list.append(item);
    });

    return list;
  }

  function createLookupStepList(search) {
    const list = createExplorerElement("div", "lookup-step-list");
    search.steps.forEach((step) => {
      const selected = step.selected ? " selected" : "";
      const row = createExplorerElement("div", `lookup-step${selected}`);
      row.dataset.reference = step.reference;
      row.dataset.selected = String(step.selected);
      row.append(createExplorerElement("span", "lookup-step-reference", step.reference));
      row.append(createExplorerElement("span", "", explorerValue(step.value)));
      row.append(createExplorerElement(
        "span",
        "lookup-step-status",
        step.selected ? "MATCH" : "no"
      ));
      list.append(row);
    });
    return list;
  }

  function createXlookupAlignment(lookup) {
    const list = createExplorerElement("div", "lookup-alignment-list");
    const returnCells = lookup.returnRange.cells;
    lookup.search.steps.forEach((step) => {
      const selected = step.selected ? " selected" : "";
      const returnCell = returnCells[step.position - 1];
      const row = createExplorerElement("div", `lookup-alignment-row${selected}`);
      row.dataset.reference = step.reference;
      row.dataset.selected = String(step.selected);
      row.append(createExplorerElement(
        "span",
        "lookup-alignment-cell",
        `${step.reference} ${explorerValue(step.value)}`
      ));
      row.append(createExplorerElement(
        "span",
        "lookup-alignment-cell",
        `${returnCell.reference} ${explorerValue(returnCell.value)}`
      ));
      row.append(createExplorerElement(
        "span",
        "lookup-step-status",
        step.selected ? "✓" : ""
      ));
      list.append(row);
    });
    return list;
  }

  function renderLookupExplorer(fragment, lookup) {
    if (lookup.kind === "tableLookup") {
      const laneLabel = lookup.lookupLane?.length
        ? `${lookup.lookupLane[0].reference}:${lookup.lookupLane.at(-1).reference}`
        : "Unavailable";
      fragment.append(createExplorerSection("Lookup value", explorerValue(lookup.lookupValue), "lookup-value"));
      fragment.append(createExplorerSection("Table", lookup.table.label, "lookup-table"));
      fragment.append(createExplorerSection(
        lookup.orientation === "vertical" ? "Search column" : "Search row",
        laneLabel,
        "lookup-range"
      ));
      if (lookup.search) {
        fragment.append(createExplorerSection(
          "Search process",
          createLookupStepList(lookup.search),
          "lookup-search"
        ));
      }
      if (lookup.matchedBand?.length) {
        const label = `${lookup.matchedBand[0].reference}:${lookup.matchedBand.at(-1).reference}`;
        fragment.append(createExplorerSection(
          lookup.orientation === "vertical" ? "Matched row" : "Matched column",
          label,
          "matched-band"
        ));
      }
      fragment.append(createExplorerSection(
        lookup.orientation === "vertical" ? "Requested table column" : "Requested table row",
        explorerValue(lookup.returnIndex),
        "requested-index"
      ));
      if (lookup.returnCell) {
        fragment.append(createExplorerSection(
          "Returned cell",
          lookup.returnCell.reference,
          "returned-cell"
        ));
        fragment.append(createExplorerSection(
          "Returned value",
          explorerValue(lookup.returnCell.value),
          "returned-value"
        ));
      }
      return;
    }

    if (lookup.kind === "xlookup") {
      fragment.append(createExplorerSection("Lookup value", explorerValue(lookup.lookupValue), "lookup-value"));
      fragment.append(createExplorerSection("Lookup range", lookup.lookupRange.label, "lookup-range"));
      fragment.append(createExplorerSection("Return range", lookup.returnRange.label, "return-range"));
      fragment.append(createExplorerSection("Match mode", explorerValue(lookup.matchMode), "match-mode"));
      fragment.append(createExplorerSection("Search mode", explorerValue(lookup.searchMode), "search-mode"));
      fragment.append(createExplorerSection(
        "Aligned search",
        createXlookupAlignment(lookup),
        "lookup-search"
      ));
      fragment.append(createExplorerSection(
        "How XLOOKUP works",
        createExplorerElement(
          "div",
          "explorer-value lookup-note",
          "XLOOKUP can return from either side of the lookup range."
        ),
        "lookup-note"
      ));
      if (lookup.returnCell) {
        fragment.append(createExplorerSection("Returned cell", lookup.returnCell.reference, "returned-cell"));
        fragment.append(createExplorerSection(
          "Returned value",
          explorerValue(lookup.returnCell.value),
          "returned-value"
        ));
      }
      return;
    }

    if (lookup.kind === "match") {
      fragment.append(createExplorerSection("Lookup value", explorerValue(lookup.lookupValue), "lookup-value"));
      fragment.append(createExplorerSection("Lookup range", lookup.lookupRange.label, "lookup-range"));
      fragment.append(createExplorerSection(
        "Search positions",
        createLookupStepList(lookup.search),
        "lookup-search"
      ));
      if (lookup.resultPosition) {
        fragment.append(createExplorerSection(
          "Returned position",
          explorerValue(lookup.resultPosition),
          "returned-position"
        ));
      }
      fragment.append(createExplorerSection(
        "What MATCH returns",
        createExplorerElement(
          "div",
          "explorer-value lookup-note",
          "MATCH returns a relative position, not the cell value."
        ),
        "lookup-note"
      ));
      return;
    }

    if (lookup.kind === "index") {
      fragment.append(createExplorerSection("Array", lookup.array.label, "index-array"));
      fragment.append(createExplorerSection(
        "Requested relative row",
        explorerValue(lookup.requested.row),
        "index-row"
      ));
      fragment.append(createExplorerSection(
        "Requested relative column",
        explorerValue(lookup.requested.column),
        "index-column"
      ));

      if (lookup.children.length) {
        const composition = createExplorerElement("div", "lookup-composition");
        lookup.children.forEach((child, index) => {
          const step = createExplorerElement("div", "lookup-composition-step");
          step.append(createExplorerElement("strong", "", `Step ${index + 1}: ${child.trace.functionName}`));
          step.append(createExplorerElement(
            "span",
            "",
            `${child.trace.lookupRange.label} returned position ${explorerValue(child.trace.result)}.`
          ));
          composition.append(step);
        });
        const finalStep = createExplorerElement("div", "lookup-composition-step");
        finalStep.append(createExplorerElement("strong", "", `Step ${lookup.children.length + 1}: INDEX`));
        finalStep.append(createExplorerElement(
          "span",
          "",
          lookup.selectedCell
            ? `INDEX uses that position and returns ${lookup.selectedCell.reference} = ${explorerValue(lookup.selectedCell.value)}.`
            : "INDEX could not select a valid cell."
        ));
        composition.append(finalStep);
        fragment.append(createExplorerSection("Lookup steps", composition, "lookup-composition"));
      }

      if (lookup.selectedCell) {
        fragment.append(createExplorerSection("Intersection", lookup.selectedCell.reference, "index-intersection"));
        fragment.append(createExplorerSection(
          "Value",
          explorerValue(lookup.selectedCell.value),
          "returned-value"
        ));
      }
    }
  }

  function renderPlaceholderExplorer() {
    explorerTitle.textContent = "Function Explorer";
    const description = createExplorerElement(
      "p",
      "",
      "Select a function to learn how it works."
    );
    const functions = createExplorerElement("ul");
    functions.setAttribute("aria-label", "Supported function examples");
    ["IF", "SUMIF", "COUNTIF", "VLOOKUP", "XLOOKUP"].forEach((name) => {
      functions.append(createExplorerElement("li", "", name));
    });
    explorerContent.replaceChildren(description, functions);
  }

  function renderCellFormatExplorer(reference, model) {
    explorerTitle.textContent = "Cell Explorer";
    const fragment = document.createDocumentFragment();
    fragment.append(createExplorerSection("Cell", reference, "cell"));
    fragment.append(createExplorerSection(
      "Underlying value",
      window.ExcelFormatting.formatValue(model.value),
      "underlying-value"
    ));
    fragment.append(createExplorerSection(
      "Number format",
      window.ExcelFormatting.formatSummary(model.numberFormat, model.formatOptions),
      "number-format"
    ));
    fragment.append(createExplorerSection(
      "Displayed value",
      displayedValue(model),
      "displayed-value"
    ));
    if (model.numberFormat !== "General") {
      fragment.append(createExplorerSection(
        "Formatting note",
        "Formatting changes how the value appears, not the stored value.",
        "format-note"
      ));
    }
    explorerContent.replaceChildren(fragment);
  }

  function renderSpillCellExplorer(reference, projection) {
    const descriptor = spillRanges.get(projection.spillOwner);
    const ownerModel = cellData.get(projection.spillOwner);
    explorerTitle.textContent = "Spill Cell Explorer";
    const fragment = document.createDocumentFragment();
    fragment.append(createExplorerSection("Selected cell", reference, "cell"));
    fragment.append(createExplorerSection("Spilled from", projection.spillOwner, "spill-owner"));
    if (descriptor) {
      fragment.append(createExplorerSection("Spill range", descriptor.range, "spill-range"));
      fragment.append(createExplorerSection(
        "Position in array",
        `Row ${projection.spillRowOffset + 1}, column ${projection.spillColumnOffset + 1}`,
        "spill-position"
      ));
    }
    fragment.append(createExplorerSection(
      "Anchor formula",
      createExplorerElement("code", "explorer-value explorer-formula", ownerModel?.input || ""),
      "formula"
    ));
    fragment.append(createExplorerSection("Value", displayedValue(projection), "result"));
    fragment.append(createExplorerSection(
      "Editing",
      `This value is controlled by ${projection.spillOwner}. Edit or delete the anchor to change the array.`,
      "spill-editing"
    ));
    explorerContent.replaceChildren(fragment);
  }

  function createArrayPreview(array) {
    if (!window.FormulaEngine.isArrayValue(array)) {
      return createExplorerElement("div", "explorer-value", explorerValue(array));
    }
    const preview = createExplorerElement("div", "array-preview");
    const rows = Math.min(array.rows, 8);
    const columns = Math.min(array.columns, 6);
    preview.style.setProperty("--array-columns", String(columns));
    for (let row = 0; row < rows; row += 1) {
      for (let column = 0; column < columns; column += 1) {
        const numberFormat = array.formats?.[row]?.[column] || "General";
        preview.append(createExplorerElement(
          "span",
          "array-preview-cell",
          explorerValue(array.values[row][column], numberFormat)
        ));
      }
    }
    if (array.rows > rows || array.columns > columns) {
      preview.append(createExplorerElement(
        "span",
        "array-preview-more",
        `Previewing ${rows} × ${columns} of ${array.rows} × ${array.columns}`
      ));
    }
    return preview;
  }

  function renderDynamicArrayExplorer(fragment, trace) {
    if (trace.kind === "sequence") {
      fragment.append(createExplorerSection("Rows", String(trace.rows), "array-rows"));
      fragment.append(createExplorerSection("Columns", String(trace.columns), "array-columns"));
      fragment.append(createExplorerSection("Start", explorerValue(trace.start), "array-start"));
      fragment.append(createExplorerSection("Step", explorerValue(trace.step), "array-step"));
    }

    if (trace.kind === "filter") {
      fragment.append(createExplorerSection(
        "Source array",
        trace.sourceLabel || `${trace.source.rows} × ${trace.source.columns} array`,
        "filter-source"
      ));
      fragment.append(createExplorerSection(
        "Filter condition",
        trace.conditionExpression || trace.includeLabel || "Array expression",
        "filter-condition"
      ));
      const evaluations = createExplorerElement("div", "array-evaluation-list");
      trace.rowEvaluation.forEach((entry, index) => {
        const row = createExplorerElement(
          "div",
          `array-evaluation-row ${entry.included ? "included" : "excluded"}`
        );
        const sourceReference = trace.source.references?.[index]?.[0];
        const rowLabel = sourceReference
          ? String(window.FormulaEngine.parseReference(sourceReference).row + 1)
          : String(index + 1);
        row.append(createExplorerElement("span", "array-row-number", rowLabel));
        row.append(createExplorerElement(
          "span",
          "array-row-value",
          explorerValue(entry.conditionValue ?? entry.value)
        ));
        row.append(createExplorerElement(
          "strong",
          "array-row-decision",
          entry.included ? "INCLUDE" : "exclude"
        ));
        evaluations.append(row);
      });
      fragment.append(createExplorerSection("Row evaluation", evaluations, "filter-evaluation"));
      fragment.append(createExplorerSection(
        "Rows returned",
        String(trace.includedRows.length),
        "filter-count"
      ));
    }

    if (trace.kind === "sort") {
      fragment.append(createExplorerSection(
        "Array",
        trace.sourceLabel || `${trace.source.rows} × ${trace.source.columns} array`,
        "sort-source"
      ));
      fragment.append(createExplorerSection(
        trace.byColumn ? "Sort row within array" : "Sort column within array",
        String(trace.sortIndex),
        "sort-index"
      ));
      if (trace.keyHeader !== null && trace.keyHeader !== "") {
        fragment.append(createExplorerSection(
          "This corresponds to",
          explorerValue(trace.keyHeader),
          "sort-key-header"
        ));
      }
      fragment.append(createExplorerSection(
        "Order",
        trace.sortOrder === -1 ? "Descending" : "Ascending",
        "sort-order"
      ));
      fragment.append(createExplorerSection("Before", createArrayPreview(trace.source), "sort-before"));
    }

    if (trace.kind === "sortby") {
      fragment.append(createExplorerSection(
        "Returned array",
        trace.sourceLabel || `${trace.source.rows} × ${trace.source.columns} array`,
        "sortby-source"
      ));
      trace.keys.forEach((key, index) => {
        fragment.append(createExplorerSection(
          trace.keys.length > 1 ? `Sort key ${index + 1}` : "Sort key",
          `${key.label || "Aligned array"} · ${key.sortOrder === -1 ? "Descending" : "Ascending"}`,
          `sortby-key-${index + 1}`
        ));
      });
      fragment.append(createExplorerSection(
        "Why SORTBY",
        "The sort key is a separate aligned array, so it does not need a relative column number.",
        "sortby-note"
      ));
    }

    if (trace.kind === "unique") {
      fragment.append(createExplorerSection(
        "Input",
        createArrayPreview(trace.source),
        "unique-input"
      ));
      fragment.append(createExplorerSection(
        trace.exactlyOnce ? "Values occurring exactly once" : "First occurrences kept",
        window.FormulaEngine.isArrayValue(trace.result)
          ? createArrayPreview(trace.result)
          : explorerValue(trace.result),
        "unique-kept"
      ));
      fragment.append(createExplorerSection(
        "Duplicates removed",
        String(trace.duplicatesRemoved),
        "unique-removed"
      ));
    }

    if (window.FormulaEngine.isArrayValue(trace.result)) {
      fragment.append(createExplorerSection(
        "Result shape",
        `${trace.result.rows} × ${trace.result.columns}`,
        "array-shape"
      ));
      if (trace.kind !== "unique") {
        fragment.append(createExplorerSection("Preview", createArrayPreview(trace.result), "array-preview"));
      }
    }

    const spillRange = trace.spill?.range || trace.spillError?.requiredRange;
    if (spillRange) fragment.append(createExplorerSection("Spill range", spillRange, "spill-range"));
    if (trace.spillError) {
      fragment.append(createExplorerSection(
        "Blocked by",
        trace.spillError.reference || "Worksheet boundary",
        "spill-blocked-by"
      ));
      if (trace.spillError.reference) {
        fragment.append(createExplorerSection(
          "Existing value",
          explorerValue(trace.spillError.value),
          "spill-blocking-value"
        ));
      }
      fragment.append(createExplorerSection(
        "Result",
        window.FormulaEngine.ERROR_VALUES.SPILL,
        "spill-error"
      ));
    }
  }

  function renderFormulaExplorer(explanation) {
    explorerTitle.textContent = "Formula Explorer";
    const fragment = document.createDocumentFragment();
    const formula = createExplorerElement("code", "explorer-value explorer-formula", explanation.formula);
    fragment.append(createExplorerSection("Formula", formula, "formula"));

    if (explanation.functionName) {
      fragment.append(createExplorerSection("Function", explanation.functionName, "function"));
    }

    fragment.append(createExplorerSection("Purpose", explanation.purpose, "purpose"));

    if (explanation.numberFormat !== "General" || explanation.numberFormatOverride) {
      fragment.append(createExplorerSection(
        "Underlying value",
        explanation.underlyingDisplay,
        "underlying-value"
      ));
      fragment.append(createExplorerSection(
        "Number format",
        window.ExcelFormatting.formatSummary(
          explanation.numberFormat,
          explanation.formatOptions
        ),
        "number-format"
      ));
      fragment.append(createExplorerSection(
        "Displayed value",
        explanation.displayedResult,
        "displayed-value"
      ));
      fragment.append(createExplorerSection(
        "Formatting note",
        "Formatting changes appearance; it does not change the stored value.",
        "format-note"
      ));
    }

    if (explanation.ranges.length
      && !explanation.conditional
      && !explanation.lookup
      && !explanation.text
      && !explanation.date
      && !explanation.dynamicArray) {
      const ranges = explanation.ranges.map((range) => range.label).join(", ");
      fragment.append(createExplorerSection("Range", ranges, "range"));
    }

    if (explanation.lookup) {
      renderLookupExplorer(fragment, explanation.lookup);
    }

    if (explanation.conditional) {
      explanation.conditional.criteria.forEach((entry, index) => {
        const suffix = explanation.conditional.criteria.length > 1 ? ` ${index + 1}` : "";
        fragment.append(createExplorerSection(
          `Criteria range${suffix}`,
          entry.range,
          `criteria-range${index + 1}`
        ));
        const criterion = entry.criterion.display === "" ? "(blank)" : entry.criterion.display;
        fragment.append(createExplorerSection(
          `Criterion${suffix}`,
          criterion,
          `criterion${index + 1}`
        ));
      });

      if (explanation.conditional.aggregateRange) {
        const label = explanation.conditional.functionName.startsWith("AVERAGE")
          ? "Average range"
          : "Sum range";
        fragment.append(createExplorerSection(
          label,
          explanation.conditional.aggregateRange.label,
          "aggregate-range"
        ));
      }

      fragment.append(createExplorerSection(
        "Evaluation",
        createConditionalList(explanation.conditional),
        "conditional-evaluation"
      ));

      if (explanation.conditional.includedValues.length) {
        const entries = explanation.conditional.includedValues.map((entry) => ({
          ...entry,
          ignored: false
        }));
        fragment.append(createExplorerSection(
          "Included values",
          createReferenceList(entries),
          "included-values"
        ));
      }
    }

    if (explanation.dynamicArray) {
      renderDynamicArrayExplorer(fragment, explanation.dynamicArray);
    }

    if (explanation.text) {
      renderTextExplorer(fragment, explanation.text);
    }

    if (explanation.date) {
      renderDateExplorer(fragment, explanation.date);
    }

    if (explanation.math) {
      renderMathExplorer(fragment, explanation.math);
    }

    if (explanation.errorHandling) {
      renderErrorHandlingExplorer(fragment, explanation.errorHandling);
    }

    explanation.referenceGroups.forEach((group) => {
      fragment.append(createExplorerSection(
        group.label,
        createReferenceList(group.entries),
        explorerFieldName(group.label)
      ));
    });

    if (explanation.logical?.kind === "IF") {
      fragment.append(createExplorerSection(
        "Condition",
        explanation.logical.condition,
        "condition"
      ));
      if (explanation.logical.tests.length) {
        fragment.append(createExplorerSection(
          "Logical tests",
          createLogicalTestList(explanation.logical.tests),
          "logical-tests"
        ));
      }
      fragment.append(createExplorerSection(
        "Comparison",
        explanation.logical.comparison,
        "comparison"
      ));
      fragment.append(createExplorerSection(
        "Condition result",
        explorerValue(explanation.logical.conditionResult),
        "condition-result"
      ));
      if (explanation.logical.rule) {
        fragment.append(createExplorerSection("Rule", explanation.logical.rule, "rule"));
      }
      fragment.append(createExplorerSection(
        "Chosen branch",
        explanation.logical.chosenBranch,
        "chosen-branch"
      ));
      fragment.append(createExplorerSection(
        "Returned value",
        explorerValue(explanation.logical.returnedValue),
        "returned-value"
      ));
      if (explanation.logical.nestedDecision) {
        const nested = explanation.logical.nestedDecision;
        const nestedContent = createExplorerElement("div", "explorer-test-list");
        nestedContent.append(createLogicalTestList([{
          expression: nested.condition,
          calculation: nested.comparison,
          result: nested.conditionResult
        }]));
        nestedContent.append(createExplorerElement(
          "div",
          "explorer-test-calculation",
          `Chosen: ${nested.chosenBranch}`
        ));
        nestedContent.append(createExplorerElement(
          "div",
          "explorer-test-result",
          `Returned: ${explorerValue(nested.returnedValue)}`
        ));
        fragment.append(createExplorerSection(
          "Nested decision",
          nestedContent,
          "nested-decision"
        ));
      }
    } else if (explanation.logical) {
      fragment.append(createExplorerSection(
        "Logical tests",
        createLogicalTestList(explanation.logical.tests),
        "logical-tests"
      ));
      fragment.append(createExplorerSection("Rule", explanation.logical.rule, "rule"));
    }

    explanation.metrics.forEach((metric) => {
      fragment.append(createExplorerSection(
        metric.label,
        explorerValue(metric.value),
        explorerFieldName(metric.label)
      ));
    });

    if (explanation.calculation) {
      fragment.append(createExplorerSection(
        "Calculation",
        explanation.calculation,
        "calculation"
      ));
    }

    if (explanation.error) {
      const error = createExplorerElement("div", "explorer-value");
      error.append(createExplorerElement(
        "div",
        "explorer-error-title",
        explanation.error.title
      ));
      error.append(createExplorerElement(
        "div",
        "explorer-error-message",
        explanation.error.message
      ));
      fragment.append(createExplorerSection("Explanation", error, "error"));
    }

    const resultText = explorerValue(
      explanation.result,
      explanation.numberFormat,
      explanation.formatOptions
    );
    const result = createExplorerElement(
      "div",
      "explorer-value explorer-result",
      resultText
    );
    fragment.append(createExplorerSection("Result", result, "result"));
    explorerContent.replaceChildren(fragment);
  }

  function clearReferenceHighlights() {
    highlightedReferences.forEach((reference) => {
      cellElements.get(reference)?.classList.remove(
        "formula-reference",
        "criteria-reference",
        "aggregate-reference",
        "matched-reference",
        "table-reference",
        "lookup-reference",
        "return-reference",
        "scanned-reference",
        "selected-lookup-reference",
        "selected-return-reference",
        "array-source-reference",
        "array-include-reference",
        "array-included-reference",
        "array-sort-key-reference"
      );
    });
    highlightedReferences.clear();
  }

  function updateSpillOutline() {
    outlinedSpillReferences.forEach((reference) => {
      cellElements.get(reference)?.classList.remove(
        "spill-outline-top",
        "spill-outline-right",
        "spill-outline-bottom",
        "spill-outline-left",
        "spill-selected-anchor",
        "spill-blocker"
      );
    });
    outlinedSpillReferences.clear();

    const selected = cellReference(state.activeRow, state.activeColumn);
    const projection = spillCells.get(selected);
    const owner = projection?.spillOwner || selected;
    const descriptor = spillRanges.get(owner);
    if (descriptor) {
      descriptor.references.forEach((reference) => {
        const cell = cellElements.get(reference);
        const entry = spillCells.get(reference);
        if (!cell || !entry) return;
        if (entry.spillRowOffset === 0) cell.classList.add("spill-outline-top");
        if (entry.spillRowOffset === descriptor.rows - 1) cell.classList.add("spill-outline-bottom");
        if (entry.spillColumnOffset === 0) cell.classList.add("spill-outline-left");
        if (entry.spillColumnOffset === descriptor.columns - 1) cell.classList.add("spill-outline-right");
        if (reference === owner) cell.classList.add("spill-selected-anchor");
        outlinedSpillReferences.add(reference);
      });
      return;
    }

    const blocker = cellData.get(selected)?.spillError?.reference;
    if (blocker) {
      cellElements.get(blocker)?.classList.add("spill-blocker");
      outlinedSpillReferences.add(blocker);
    }
  }

  function updateFormulaTrace() {
    clearReferenceHighlights();
    updateSpillOutline();
    const selectedReference = cellReference(state.activeRow, state.activeColumn);
    const model = getCellModel(state.activeRow, state.activeColumn);

    if (!model) {
      renderPlaceholderExplorer();
      return;
    }
    if (model.type === "spill") {
      renderSpillCellExplorer(selectedReference, model);
      return;
    }
    if (model.type !== "formula") {
      renderCellFormatExplorer(
        cellReference(state.activeRow, state.activeColumn),
        model
      );
      return;
    }

    const explanation = window.FormulaExplanations.buildExplanation({
      formula: model.input,
      ast: model.ast,
      result: model.value,
      dependencies: model.dependencies,
      getCellValue: calculatedCellValue,
      expandRange: window.FormulaEngine.expandRange,
      evaluateAst: evaluateAstForExplanation,
      analyzeConditional: analyzeConditionalForExplanation,
      analyzeLookup: analyzeLookupForExplanation,
      analyzeText: analyzeTextForExplanation,
      analyzeDate: analyzeDateForExplanation,
      analyzeMath: analyzeMathForExplanation,
      analyzeError: analyzeErrorForExplanation,
      analyzeDynamic: analyzeDynamicForExplanation,
      spill: spillRanges.get(selectedReference) || null,
      spillError: model.spillError || null,
      numberFormat: model.numberFormat,
      numberFormatOverride: model.numberFormatOverride,
      formatOptions: model.formatOptions,
      displayedResult: displayedValue(model),
      getCellNumberFormat: calculatedCellNumberFormat,
      formatValue: window.ExcelFormatting.formatValue
    });

    model.dependencies.forEach((reference) => {
      const cell = cellElements.get(reference);
      if (!cell) return;
      cell.classList.add("formula-reference");
      highlightedReferences.add(reference);
    });

    if (explanation.conditional) {
      explanation.conditional.criteria.forEach((entry) => {
        entry.references.forEach((reference) => {
          cellElements.get(reference)?.classList.add("criteria-reference");
        });
      });
      explanation.conditional.aggregateRange?.references.forEach((reference) => {
        cellElements.get(reference)?.classList.add("aggregate-reference");
      });
      explanation.conditional.positions.forEach((position) => {
        if (!position.allMatched) return;
        position.checks.forEach((check) => {
          cellElements.get(check.reference)?.classList.add("matched-reference");
        });
        if (position.aggregate?.included) {
          cellElements.get(position.aggregate.reference)?.classList.add("matched-reference");
        }
      });
    }

    if (explanation.lookup) {
      const lookup = explanation.lookup;
      if (lookup.kind === "tableLookup") {
        lookup.table.cells.forEach((entry) => {
          cellElements.get(entry.reference)?.classList.add("table-reference");
        });
        lookup.lookupLane?.forEach((entry) => {
          cellElements.get(entry.reference)?.classList.add("lookup-reference");
        });
        lookup.search?.steps.forEach((step) => {
          cellElements.get(step.reference)?.classList.add("scanned-reference");
          if (step.selected) {
            cellElements.get(step.reference)?.classList.add("selected-lookup-reference");
          }
        });
        lookup.matchedBand?.forEach((entry) => {
          cellElements.get(entry.reference)?.classList.add("matched-reference");
        });
        if (lookup.returnCell) {
          cellElements.get(lookup.returnCell.reference)?.classList.add(
            "return-reference",
            "selected-return-reference"
          );
        }
      } else if (lookup.kind === "xlookup") {
        lookup.lookupRange.cells.forEach((entry) => {
          cellElements.get(entry.reference)?.classList.add("lookup-reference");
        });
        lookup.returnRange.cells.forEach((entry) => {
          cellElements.get(entry.reference)?.classList.add("return-reference");
        });
        lookup.search.steps.forEach((step) => {
          cellElements.get(step.reference)?.classList.add("scanned-reference");
          if (step.selected) {
            cellElements.get(step.reference)?.classList.add("selected-lookup-reference");
          }
        });
        if (lookup.returnCell) {
          cellElements.get(lookup.returnCell.reference)?.classList.add("selected-return-reference");
        }
      } else if (lookup.kind === "match") {
        lookup.lookupRange.cells.forEach((entry) => {
          cellElements.get(entry.reference)?.classList.add("lookup-reference");
        });
        lookup.search.steps.forEach((step) => {
          cellElements.get(step.reference)?.classList.add("scanned-reference");
          if (step.selected) {
            cellElements.get(step.reference)?.classList.add("selected-lookup-reference");
          }
        });
      } else if (lookup.kind === "index") {
        lookup.array.cells.forEach((entry) => {
          cellElements.get(entry.reference)?.classList.add("table-reference");
        });
        lookup.children.forEach((child) => {
          child.trace.lookupRange?.cells.forEach((entry) => {
            cellElements.get(entry.reference)?.classList.add("lookup-reference");
          });
          child.trace.search?.steps.forEach((step) => {
            if (step.selected) {
              cellElements.get(step.reference)?.classList.add("selected-lookup-reference");
            }
          });
        });
        if (lookup.selectedCell) {
          cellElements.get(lookup.selectedCell.reference)?.classList.add(
            "return-reference",
            "selected-return-reference"
          );
        }
      }
    }

    if (explanation.dynamicArray) {
      const trace = explanation.dynamicArray;
      trace.source?.references?.flat().forEach((reference) => {
        cellElements.get(reference)?.classList.add("array-source-reference");
      });
      if (trace.kind === "filter") {
        trace.include.references?.flat().forEach((reference) => {
          cellElements.get(reference)?.classList.add("array-include-reference");
        });
        trace.includedRows.forEach((rowIndex) => {
          trace.source.references?.[rowIndex]?.forEach((reference) => {
            cellElements.get(reference)?.classList.add("array-included-reference");
          });
        });
      }
      if (trace.kind === "sort" && trace.source.references) {
        if (trace.byColumn) {
          trace.source.references[trace.sortIndex - 1]?.forEach((reference) => {
            cellElements.get(reference)?.classList.add("array-sort-key-reference");
          });
        } else {
          trace.source.references.forEach((row) => {
            cellElements.get(row[trace.sortIndex - 1])?.classList.add("array-sort-key-reference");
          });
        }
      }
      if (trace.kind === "sortby") {
        trace.keys.forEach((key) => {
          key.array.references?.flat().forEach((reference) => {
            cellElements.get(reference)?.classList.add("array-sort-key-reference");
          });
        });
      }
    }

    renderFormulaExplorer(explanation);
  }

  function updateSelectionDisplay() {
    const reference = cellReference(state.activeRow, state.activeColumn);
    const projection = spillCells.get(reference);
    const child = projection && projection.spillOwner !== reference;
    nameBox.value = reference;
    formulaInput.value = child
      ? (cellData.get(projection.spillOwner)?.input || "")
      : cellInput(state.activeRow, state.activeColumn);
    formulaInput.readOnly = Boolean(child);
    formulaFunctionButton.disabled = Boolean(child);
    formulaInput.classList.toggle("spill-formula-readonly", Boolean(child));
    selectionStatus.textContent = child
      ? `Selected: ${reference} · Spilled from ${projection.spillOwner} · Edit the anchor to make changes.`
      : `Selected: ${reference}`;
    syncFormatToolbar();
    updateFormulaTrace();
  }

  function syncFormatToolbar() {
    const reference = cellReference(state.activeRow, state.activeColumn);
    const projection = spillCells.get(reference);
    const child = projection && projection.spillOwner !== reference;
    const model = projection || cellData.get(reference);
    const override = cellFormatOverrides.get(reference);
    const numberFormat = model?.numberFormat || override?.type || "General";
    numberFormatSelect.value = numberFormat;
    currencyFormatButton.classList.toggle("active", numberFormat === "Currency");
    percentageFormatButton.classList.toggle("active", numberFormat === "Percentage");
    numberFormatButton.classList.toggle("active", numberFormat === "Number");
    const dateSelected = numberFormat === "Date";
    numberFormatSelect.disabled = Boolean(child);
    currencyFormatButton.disabled = Boolean(child);
    percentageFormatButton.disabled = Boolean(child);
    numberFormatButton.disabled = Boolean(child);
    decreaseDecimalButton.disabled = dateSelected || Boolean(child);
    increaseDecimalButton.disabled = dateSelected || Boolean(child);
  }

  function formatActiveCell(numberFormat, options = {}) {
    const reference = cellReference(state.activeRow, state.activeColumn);
    setCellNumberFormat(reference, numberFormat, options);
    syncFormatToolbar();
  }

  function adjustActiveDecimals(offset) {
    const reference = cellReference(state.activeRow, state.activeColumn);
    const model = cellData.get(reference);
    const override = cellFormatOverrides.get(reference) || null;
    const currentFormat = model?.numberFormat || override?.type || "General";
    if (currentFormat === "Date") return;
    const nextFormat = ["Number", "Currency", "Percentage"].includes(currentFormat)
      ? currentFormat
      : "Number";
    const currentOptions = model?.formatOptions
      || window.ExcelFormatting.normalizeFormatOptions(currentFormat, override || {});
    const currentDecimals = ["Number", "Currency", "Percentage"].includes(currentFormat)
      ? (currentOptions.decimals ?? 2)
      : (offset > 0 ? 1 : 1);
    formatActiveCell(nextFormat, {
      ...currentOptions,
      decimals: Math.max(0, currentDecimals + offset)
    });
  }

  function selectCell(row, column, options = {}) {
    const nextRow = Math.max(0, Math.min(ROW_COUNT - 1, row));
    const nextColumn = Math.max(0, Math.min(COLUMN_COUNT - 1, column));
    const currentCell = activeCellElement();
    const nextCell = cellElements.get(cellReference(nextRow, nextColumn));

    if (state.editingCell && state.editingCell !== nextCell) {
      finishCellEdit(true);
    }

    columnHeaders[state.activeColumn]?.classList.remove("selected-header");
    rowHeaders[state.activeRow]?.classList.remove("selected-header");
    currentCell?.classList.remove("active");
    currentCell?.setAttribute("aria-selected", "false");
    if (currentCell) currentCell.tabIndex = -1;

    state.activeRow = nextRow;
    state.activeColumn = nextColumn;
    nextCell.classList.add("active");
    nextCell.setAttribute("aria-selected", "true");
    nextCell.tabIndex = 0;
    columnHeaders[nextColumn]?.classList.add("selected-header");
    rowHeaders[nextRow]?.classList.add("selected-header");
    updateSelectionDisplay();

    if (options.focus !== false) {
      nextCell.focus({ preventScroll: true });
    }

    scrollCellIntoView(nextCell);
  }

  function placeCaretAtEnd(element) {
    const range = document.createRange();
    range.selectNodeContents(element);
    range.collapse(false);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
  }

  function startCellEdit(cell, replacementInput) {
    const row = Number(cell.dataset.row);
    const column = Number(cell.dataset.column);
    const reference = cellReference(row, column);
    const projection = spillCells.get(reference);
    if (projection && projection.spillOwner !== reference) {
      selectionStatus.textContent = `You can't change part of an array. Edit ${projection.spillOwner}.`;
      return false;
    }

    if (row !== state.activeRow || column !== state.activeColumn) {
      selectCell(row, column, { focus: false });
    }

    state.editingCell = cell;
    state.editStartInput = cellInput(row, column);
    cell.classList.add("editing");
    cell.setAttribute("contenteditable", "true");
    cell.textContent = replacementInput === undefined
      ? state.editStartInput
      : normalizeInput(replacementInput);
    formulaInput.value = cell.textContent;
    cell.focus({ preventScroll: true });
    placeCaretAtEnd(cell);
    return true;
  }

  function finishCellEdit(commit) {
    const cell = state.editingCell;
    if (!cell) return;

    const row = Number(cell.dataset.row);
    const column = Number(cell.dataset.column);
    const reference = cellReference(row, column);
    const nextInput = commit ? normalizeInput(cell.textContent) : state.editStartInput;

    state.editingCell = null;
    cell.classList.remove("editing");
    cell.setAttribute("contenteditable", "false");

    if (commit) {
      setCellInput(row, column, nextInput);
    } else {
      renderCell(reference);
    }

    updateSelectionDisplay();
  }

  function moveSelection(rowOffset, columnOffset) {
    selectCell(state.activeRow + rowOffset, state.activeColumn + columnOffset);
  }

  function isPrintableKey(event) {
    return event.key.length === 1 && !event.ctrlKey && !event.metaKey && !event.altKey;
  }

  grid.addEventListener("click", (event) => {
    const cell = event.target.closest(".sheet-cell");
    if (!cell || state.editingCell === cell) return;
    selectCell(Number(cell.dataset.row), Number(cell.dataset.column));
  });

  grid.addEventListener("dblclick", (event) => {
    const cell = event.target.closest(".sheet-cell");
    if (!cell) return;
    startCellEdit(cell);
  });

  grid.addEventListener("input", (event) => {
    const cell = event.target.closest(".sheet-cell.editing");
    if (!cell) return;
    formulaInput.value = normalizeInput(cell.textContent);
  });

  grid.addEventListener("blur", (event) => {
    if (event.target === state.editingCell) {
      finishCellEdit(true);
    }
  }, true);

  grid.addEventListener("keydown", (event) => {
    const cell = event.target.closest(".sheet-cell");
    if (!cell) return;

    if (state.editingCell === cell) {
      if (event.key === "Escape") {
        event.preventDefault();
        finishCellEdit(false);
        activeCellElement().focus({ preventScroll: true });
      } else if (event.key === "Enter") {
        event.preventDefault();
        finishCellEdit(true);
        moveSelection(1, 0);
      } else if (event.key === "Tab") {
        event.preventDefault();
        finishCellEdit(true);
        moveSelection(0, event.shiftKey ? -1 : 1);
      }
      return;
    }

    const navigation = {
      ArrowUp: [-1, 0],
      ArrowDown: [1, 0],
      ArrowLeft: [0, -1],
      ArrowRight: [0, 1]
    };

    if (navigation[event.key]) {
      event.preventDefault();
      moveSelection(...navigation[event.key]);
    } else if (event.key === "Enter") {
      event.preventDefault();
      moveSelection(event.shiftKey ? -1 : 1, 0);
    } else if (event.key === "Tab") {
      event.preventDefault();
      moveSelection(0, event.shiftKey ? -1 : 1);
    } else if (event.key === "F2") {
      event.preventDefault();
      startCellEdit(cell);
    } else if (event.key === "Backspace" || event.key === "Delete") {
      event.preventDefault();
      setCellInput(state.activeRow, state.activeColumn, "");
      updateSelectionDisplay();
    } else if (isPrintableKey(event)) {
      event.preventDefault();
      startCellEdit(cell, event.key);
    }
  });

  formulaInput.addEventListener("focus", () => {
    state.formulaStartInput = cellInput(state.activeRow, state.activeColumn);
  });

  formulaInput.addEventListener("input", () => {
    setCellInput(state.activeRow, state.activeColumn, formulaInput.value);
  });

  formulaInput.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      event.preventDefault();
      setCellInput(state.activeRow, state.activeColumn, state.formulaStartInput);
      updateSelectionDisplay();
      activeCellElement().focus({ preventScroll: true });
    } else if (event.key === "Enter") {
      event.preventDefault();
      moveSelection(1, 0);
    } else if (event.key === "Tab") {
      event.preventDefault();
      moveSelection(0, event.shiftKey ? -1 : 1);
    }
  });

  numberFormatSelect.addEventListener("change", () => {
    formatActiveCell(numberFormatSelect.value);
  });
  currencyFormatButton.addEventListener("click", () => formatActiveCell("Currency"));
  percentageFormatButton.addEventListener("click", () => formatActiveCell("Percentage"));
  numberFormatButton.addEventListener("click", () => formatActiveCell("Number"));
  decreaseDecimalButton.addEventListener("click", () => adjustActiveDecimals(-1));
  increaseDecimalButton.addEventListener("click", () => adjustActiveDecimals(1));

  seedSampleData();
  createGrid();
  selectCell(0, 0, { focus: false });

  window.ExcelSimulator = Object.freeze({
    clearRange: clearCellRange,
    formatValue(value, numberFormat = window.ExcelFormatting.NUMBER_FORMATS.GENERAL, options = {}) {
      return window.ExcelFormatting.formatValue(value, numberFormat, options);
    },
    getCell(reference) {
      coordinatesForReference(reference);
      const normalized = reference.toUpperCase();
      return cellData.get(normalized) || spillCells.get(normalized) || null;
    },
    getCellElement(reference) {
      coordinatesForReference(reference);
      return cellElements.get(reference.toUpperCase()) || null;
    },
    getCellFormat,
    getSpill(reference) {
      coordinatesForReference(reference);
      const normalized = reference.toUpperCase();
      const owner = spillCells.get(normalized)?.spillOwner || normalized;
      const spill = spillRanges.get(owner);
      return spill ? {
        ...spill,
        values: spill.values.map((row) => row.slice()),
        formats: spill.formats?.map((row) => row.slice()) || null,
        references: spill.references.slice()
      } : null;
    },
    getSpillOwner(reference) {
      coordinatesForReference(reference);
      return spillCells.get(reference.toUpperCase())?.spillOwner || null;
    },
    isSpillCell(reference) {
      coordinatesForReference(reference);
      const normalized = reference.toUpperCase();
      const projection = spillCells.get(normalized);
      return Boolean(projection && projection.spillOwner !== normalized);
    },
    selectCell(reference) {
      const { row, column } = coordinatesForReference(reference);
      selectCell(row, column);
    },
    setCell(reference, input) {
      const { row, column } = coordinatesForReference(reference);
      return setCellInput(row, column, input);
    },
    setCellNumberFormat,
    setCells: setCellInputs
  });
})();
