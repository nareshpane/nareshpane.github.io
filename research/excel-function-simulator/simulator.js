(() => {
  "use strict";

  const ROW_COUNT = 50;
  const COLUMN_COUNT = 26;
  const WORKBOOK_STORAGE_KEY = "excelFunctionSimulatorWorkbookV1";
  const WORKBOOK_STORAGE_VERSION = 1;
  const HISTORY_LIMIT = 100;
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
  let cornerHeader = null;

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
  const undoButton = document.querySelector("#undo-button");
  const redoButton = document.querySelector("#redo-button");
  const resetWorkbookButton = document.querySelector("#reset-workbook-button");
  const saveStatus = document.querySelector("#save-status");
  const highlightedReferences = new Set();
  const outlinedSpillReferences = new Set();
  const explorerNumberFormat = new Intl.NumberFormat("en-US", {
    maximumSignificantDigits: 12
  });

  const state = {
    activeRow: 0,
    activeColumn: 0,
    selectionAnchorRow: 0,
    selectionAnchorColumn: 0,
    selectionEndRow: 0,
    selectionEndColumn: 0,
    mouseSelecting: false,
    fillDragging: false,
    fillHoverRow: 0,
    fillHoverColumn: 0,
    editingCell: null,
    editStartInput: "",
    formulaStartInput: "",
    currentDateSerial: null,
    clipboard: null,
    clipboardMode: null,
    undoStack: [],
    redoStack: [],
    historyRestoring: false,
    pendingSelection: null
  };

  function workbookSnapshot() {
    const cells = [...cellData.entries()]
      .map(([reference, model]) => ({ reference, input: model.input }))
      .sort((a, b) => a.reference.localeCompare(b.reference, undefined, { numeric: true }));
    const formats = [...cellFormatOverrides.entries()]
      .map(([reference, format]) => ({ reference, format: { ...format } }))
      .sort((a, b) => a.reference.localeCompare(b.reference, undefined, { numeric: true }));
    return {
      version: WORKBOOK_STORAGE_VERSION,
      cells,
      formats,
      selection: {
        active: cellReference(state.activeRow, state.activeColumn),
        start: cellReference(state.selectionAnchorRow, state.selectionAnchorColumn),
        end: cellReference(state.selectionEndRow, state.selectionEndColumn)
      }
    };
  }

  function workbookContentSignature(snapshot) {
    return JSON.stringify({ cells: snapshot.cells, formats: snapshot.formats });
  }

  function updateHistoryControls() {
    if (undoButton) {
      undoButton.disabled = state.undoStack.length === 0;
      undoButton.title = state.undoStack.length
        ? `Undo ${state.undoStack[state.undoStack.length - 1].label || "change"} (Ctrl+Z)`
        : "Undo (Ctrl+Z)";
    }
    if (redoButton) {
      redoButton.disabled = state.redoStack.length === 0;
      redoButton.title = state.redoStack.length
        ? `Redo ${state.redoStack[state.redoStack.length - 1].label || "change"} (Ctrl+Y)`
        : "Redo (Ctrl+Y)";
    }
  }

  function updateSaveStatus(message = "Saved locally") {
    if (saveStatus) saveStatus.textContent = message;
  }

  function saveWorkbook() {
    try {
      localStorage.setItem(WORKBOOK_STORAGE_KEY, JSON.stringify(workbookSnapshot()));
      updateSaveStatus("Saved locally");
      return true;
    } catch (error) {
      updateSaveStatus("Local save unavailable");
      return false;
    }
  }

  function restoreWorkbookSnapshot(snapshot, options = {}) {
    if (!snapshot || snapshot.version !== WORKBOOK_STORAGE_VERSION
      || !Array.isArray(snapshot.cells) || !Array.isArray(snapshot.formats)) return false;
    const previousRestoring = state.historyRestoring;
    state.historyRestoring = true;
    try {
      cellData.clear();
      cellFormatOverrides.clear();
      spillCells = new Map();
      spillRanges = new Map();
      snapshot.cells.forEach((entry) => {
        try {
          const { row, column } = coordinatesForReference(entry.reference);
          storeCellInput(row, column, String(entry.input ?? ""));
        } catch (error) {
          // Ignore invalid or out-of-bounds persisted references.
        }
      });
      snapshot.formats.forEach((entry) => {
        try {
          coordinatesForReference(entry.reference);
          const type = entry.format?.type;
          if (!Object.values(window.ExcelFormatting.NUMBER_FORMATS).includes(type)) return;
          cellFormatOverrides.set(entry.reference.toUpperCase(), {
            ...entry.format,
            ...window.ExcelFormatting.normalizeFormatOptions(type, entry.format || {})
          });
        } catch (error) {
          // Ignore invalid persisted format entries.
        }
      });
      recalculateAll();
      const selection = snapshot.selection || { active: "A1", start: "A1", end: "A1" };
      state.pendingSelection = selection;
      if (cellElements.size) {
        try {
          selectRange(selection.start || selection.active || "A1", selection.end || selection.active || "A1", {
            active: selection.active || selection.start || "A1",
            focus: options.focus !== false
          });
        } catch (error) {
          selectCell(0, 0, { focus: options.focus !== false });
        }
        updateSelectionDisplay();
      }
      if (options.save !== false) saveWorkbook();
      return true;
    } finally {
      state.historyRestoring = previousRestoring;
    }
  }

  function loadPersistedWorkbook() {
    try {
      const raw = localStorage.getItem(WORKBOOK_STORAGE_KEY);
      if (!raw) return false;
      const snapshot = JSON.parse(raw);
      const restored = restoreWorkbookSnapshot(snapshot, { save: false, focus: false });
      if (restored) updateSaveStatus("Restored locally");
      return restored;
    } catch (error) {
      return false;
    }
  }

  function commitHistory(before, label = "change", coalesceKey = "") {
    if (state.historyRestoring || !before) return false;
    const after = workbookSnapshot();
    if (workbookContentSignature(before) === workbookContentSignature(after)) return false;
    const now = Date.now();
    const last = state.undoStack[state.undoStack.length - 1];
    if (coalesceKey && last?.coalesceKey === coalesceKey && now - last.timestamp < 1400) {
      last.after = after;
      last.timestamp = now;
      last.label = label;
      if (workbookContentSignature(last.before) === workbookContentSignature(last.after)) {
        state.undoStack.pop();
      }
    } else {
      state.undoStack.push({ before, after, label, coalesceKey, timestamp: now });
      if (state.undoStack.length > HISTORY_LIMIT) state.undoStack.shift();
    }
    state.redoStack = [];
    updateHistoryControls();
    saveWorkbook();
    return true;
  }

  function undoWorkbook() {
    const entry = state.undoStack.pop();
    if (!entry) return false;
    state.redoStack.push(entry);
    restoreWorkbookSnapshot(entry.before, { save: true, focus: true });
    updateHistoryControls();
    selectionStatus.textContent = `Undid: ${entry.label}`;
    return true;
  }

  function redoWorkbook() {
    const entry = state.redoStack.pop();
    if (!entry) return false;
    state.undoStack.push(entry);
    restoreWorkbookSnapshot(entry.after, { save: true, focus: true });
    updateHistoryControls();
    selectionStatus.textContent = `Redid: ${entry.label}`;
    return true;
  }

  function resetWorkbook() {
    if (typeof window !== "undefined" && typeof window.confirm === "function"
      && !window.confirm("Reset this workbook to the original employee sample? You can undo the reset.")) {
      return false;
    }
    const before = workbookSnapshot();
    cellData.clear();
    cellFormatOverrides.clear();
    spillCells = new Map();
    spillRanges = new Map();
    sampleRows.forEach((rowValues, row) => {
      rowValues.forEach((input, column) => {
        cellData.set(cellReference(row, column), createCellModel(input));
      });
    });
    recalculateAll();
    selectCell(0, 0, { focus: false });
    commitHistory(before, "reset workbook");
    updateSelectionDisplay();
    return true;
  }

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

  function calculatedSpillArray(reference) {
    const normalized = String(reference).toUpperCase();
    const descriptor = spillRanges.get(normalized);
    if (!descriptor) throw new window.FormulaEngine.FormulaError(window.FormulaEngine.ERROR_VALUES.REF);
    const referenceRows = [];
    for (let row = 0; row < descriptor.rows; row += 1) {
      referenceRows.push(descriptor.references.slice(
        row * descriptor.columns,
        (row + 1) * descriptor.columns
      ));
    }
    return window.FormulaEngine.makeArray(
      descriptor.rows,
      descriptor.columns,
      descriptor.values,
      { formats: descriptor.formats || undefined, references: referenceRows }
    );
  }

  function evaluateAstForExplanation(ast) {
    return window.FormulaEngine.evaluate(ast, {
      getCellValue: calculatedCellValue,
      getRangeValues(start, end) {
        return window.FormulaEngine.expandRange(start, end).map(calculatedCellValue);
      },
      getCellNumberFormat: calculatedCellNumberFormat,
      getSpillArray: calculatedSpillArray,
      getCurrentDateSerial: () => state.currentDateSerial
    });
  }

  function analyzeConditionalForExplanation(ast) {
    return window.FormulaEngine.analyzeConditionalAggregate(ast, {
      getCellValue: calculatedCellValue,
      getRangeValues(start, end) {
        return window.FormulaEngine.expandRange(start, end).map(calculatedCellValue);
      },
      getSpillArray: calculatedSpillArray
    });
  }

  function analyzeLookupForExplanation(ast) {
    return window.FormulaEngine.analyzeLookupExpression(ast, {
      getCellValue: calculatedCellValue,
      getRangeValues(start, end) {
        return window.FormulaEngine.expandRange(start, end).map(calculatedCellValue);
      },
      getSpillArray: calculatedSpillArray
    });
  }

  function analyzeTextForExplanation(ast) {
    return window.FormulaEngine.analyzeTextExpression(ast, {
      getCellValue: calculatedCellValue,
      getRangeValues(start, end) {
        return window.FormulaEngine.expandRange(start, end).map(calculatedCellValue);
      },
      getSpillArray: calculatedSpillArray
    });
  }

  function analyzeDateForExplanation(ast) {
    return window.FormulaEngine.analyzeDateExpression(ast, {
      getCellValue: calculatedCellValue,
      getRangeValues(start, end) {
        return window.FormulaEngine.expandRange(start, end).map(calculatedCellValue);
      },
      getCellNumberFormat: calculatedCellNumberFormat,
      getSpillArray: calculatedSpillArray,
      getCurrentDateSerial: () => state.currentDateSerial
    });
  }

  function analyzeMathForExplanation(ast) {
    return window.FormulaEngine.analyzeMathExpression(ast, {
      getCellValue: calculatedCellValue,
      getRangeValues(start, end) {
        return window.FormulaEngine.expandRange(start, end).map(calculatedCellValue);
      },
      getSpillArray: calculatedSpillArray
    });
  }

  function analyzeStatisticalForExplanation(ast) {
    return window.FormulaEngine.analyzeStatisticalExpression(ast, {
      getCellValue: calculatedCellValue,
      getRangeValues(start, end) {
        return window.FormulaEngine.expandRange(start, end).map(calculatedCellValue);
      },
      getCellNumberFormat: calculatedCellNumberFormat,
      getSpillArray: calculatedSpillArray
    });
  }

  function analyzeFinancialForExplanation(ast) {
    return window.FormulaEngine.analyzeFinancialExpression(ast, {
      getCellValue: calculatedCellValue,
      getRangeValues(start, end) {
        return window.FormulaEngine.expandRange(start, end).map(calculatedCellValue);
      },
      getCellNumberFormat: calculatedCellNumberFormat,
      getSpillArray: calculatedSpillArray
    });
  }

  function analyzeAdvancedForExplanation(ast) {
    return window.FormulaEngine.analyzeAdvancedExpression(ast, {
      getCellValue: calculatedCellValue,
      getRangeValues(start, end) {
        return window.FormulaEngine.expandRange(start, end).map(calculatedCellValue);
      },
      getCellNumberFormat: calculatedCellNumberFormat,
      getSpillArray: calculatedSpillArray,
      getCurrentDateSerial: () => state.currentDateSerial
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
      getSpillArray: calculatedSpillArray,
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

      function resolveSpillArray(reference) {
        const normalized = String(reference).toUpperCase();
        evaluateCell(normalized);
        const descriptor = nextSpillRanges.get(normalized);
        if (!descriptor) throw new window.FormulaEngine.FormulaError(window.FormulaEngine.ERROR_VALUES.REF);
        const referenceRows = [];
        for (let row = 0; row < descriptor.rows; row += 1) {
          referenceRows.push(descriptor.references.slice(
            row * descriptor.columns,
            (row + 1) * descriptor.columns
          ));
        }
        return window.FormulaEngine.makeArray(
          descriptor.rows,
          descriptor.columns,
          descriptor.values,
          { formats: descriptor.formats || undefined, references: referenceRows }
        );
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
              getSpillArray: resolveSpillArray,
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
    const before = state.historyRestoring ? null : workbookSnapshot();
    const projection = spillCells.get(reference);
    if (projection && projection.spillOwner !== reference) {
      selectionStatus.textContent = `You can't change part of an array. Edit ${projection.spillOwner}.`;
      renderCell(reference);
      return cellData.get(reference)?.input || "";
    }
    const normalizedInput = storeCellInput(row, column, input);

    recalculateAll();
    if (cellElements.size) updateFormulaTrace();
    commitHistory(before, `edit ${reference}`, `cell:${reference}`);
    return normalizedInput;
  }

  function setCellNumberFormat(reference, numberFormat, options = {}) {
    coordinatesForReference(reference);
    const before = state.historyRestoring ? null : workbookSnapshot();
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
    commitHistory(before, `format ${reference}`);
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
    const before = state.historyRestoring ? null : workbookSnapshot();
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
    commitHistory(before, "edit cells");
  }

  function clearCellRange(start, end) {
    const before = state.historyRestoring ? null : workbookSnapshot();
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
    commitHistory(before, `clear ${start}:${end}`);
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
    corner.dataset.selectAll = "true";
    corner.setAttribute("role", "button");
    corner.setAttribute("aria-label", "Select all cells");
    cornerHeader = corner;
    fragment.append(corner);

    COLUMN_LABELS.forEach((label) => {
      const header = document.createElement("div");
      header.className = "column-header";
      header.textContent = label;
      header.dataset.column = String(columnHeaders.length);
      header.setAttribute("role", "columnheader");
      columnHeaders.push(header);
      fragment.append(header);
    });

    for (let row = 0; row < ROW_COUNT; row += 1) {
      const rowHeader = document.createElement("div");
      rowHeader.className = "row-header";
      rowHeader.textContent = String(row + 1);
      rowHeader.dataset.row = String(row);
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

  function createWorkdayList(entries, limit = 20) {
    const list = createExplorerElement("div", "workday-list");
    entries.slice(0, limit).forEach((entry) => {
      const row = createExplorerElement("div", `workday-row${entry.workday ? " included" : " skipped"}`);
      row.append(createExplorerElement("span", "workday-date", entry.date));
      row.append(createExplorerElement("span", "workday-name", entry.dayName));
      row.append(createExplorerElement(
        "span",
        "workday-status",
        entry.workday ? "WORKDAY" : (entry.holiday ? "holiday" : "weekend")
      ));
      list.append(row);
    });
    if (entries.length > limit) {
      list.append(createExplorerElement("div", "workday-more", `… ${entries.length - limit} more day${entries.length - limit === 1 ? "" : "s"}`));
    }
    return list;
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
      return;
    }

    if (date.kind === "networkdays") {
      fragment.append(createExplorerSection("Start date", date.startDate, "start-date"));
      fragment.append(createExplorerSection("End date", date.endDate, "end-date"));
      fragment.append(createExplorerSection(
        "Calendar scan",
        createWorkdayList(date.days),
        "networkdays-scan"
      ));
      fragment.append(createExplorerSection(
        "Workdays counted",
        explorerValue(date.workdayCount),
        "networkdays-result"
      ));
      fragment.append(createExplorerSection(
        "Rule",
        "NETWORKDAYS counts Monday through Friday inclusively and excludes supplied holidays.",
        "networkdays-rule"
      ));
      return;
    }

    if (date.kind === "workday") {
      fragment.append(createExplorerSection("Start date", date.startDate, "start-date"));
      fragment.append(createExplorerSection(
        "Workdays to move",
        explorerValue(date.days),
        "workday-count"
      ));
      if (date.traversed.length) {
        fragment.append(createExplorerSection(
          "Days traversed",
          createWorkdayList(date.traversed),
          "workday-traversal"
        ));
      }
      fragment.append(createExplorerSection("Calendar result", date.resultDate, "calendar-result"));
      fragment.append(createExplorerSection(
        "Rule",
        "WORKDAY skips Saturdays, Sundays, and supplied holidays while moving through the calendar.",
        "workday-rule"
      ));
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

  function statisticalValueList(values) {
    const list = createExplorerElement("div", "statistical-value-list");
    values.forEach((value, index) => {
      const row = createExplorerElement("div", "statistical-value-row");
      row.append(createExplorerElement("span", "statistical-position", String(index + 1)));
      row.append(createExplorerElement("span", "statistical-value", explorerValue(value)));
      list.append(row);
    });
    return list;
  }

  function statisticalPairList(pairs) {
    const list = createExplorerElement("div", "statistical-pair-list");
    pairs.forEach((pair, index) => {
      const row = createExplorerElement("div", "statistical-pair-row");
      row.append(createExplorerElement("span", "statistical-position", String(index + 1)));
      row.append(createExplorerElement(
        "span",
        "statistical-pair-value",
        `${pair.xReference || "x"} ${explorerValue(pair.x)}`
      ));
      row.append(createExplorerElement(
        "span",
        "statistical-pair-value",
        `${pair.yReference || "y"} ${explorerValue(pair.y)}`
      ));
      list.append(row);
    });
    return list;
  }

  function renderStatisticalExplorer(fragment, trace) {
    if (trace.kind === "median") {
      fragment.append(createExplorerSection("Ordered values", statisticalValueList(trace.sorted), "stat-ordered-values"));
      fragment.append(createExplorerSection("Count", explorerValue(trace.sorted.length), "stat-count"));
      const positions = trace.lowerIndex === trace.upperIndex
        ? `Position ${trace.lowerIndex + 1}`
        : `Positions ${trace.lowerIndex + 1} and ${trace.upperIndex + 1}`;
      fragment.append(createExplorerSection("Middle position", positions, "stat-middle-position"));
      fragment.append(createExplorerSection("Median", explorerValue(trace.result), "stat-result"));
      return;
    }

    if (trace.kind === "mode") {
      const countText = trace.counts.length
        ? trace.counts.map(([value, count]) => `${explorerValue(value)} → ${count}`).join("\n")
        : "No numeric values";
      fragment.append(createExplorerSection(
        "Frequency table",
        createExplorerElement("pre", "explorer-value statistical-frequency", countText),
        "stat-frequency"
      ));
      fragment.append(createExplorerSection("Highest frequency", explorerValue(trace.frequency), "stat-highest-frequency"));
      fragment.append(createExplorerSection(
        "Mode",
        trace.result === window.FormulaEngine.ERROR_VALUES.NA ? "No repeated numeric value" : explorerValue(trace.result),
        "stat-result"
      ));
      return;
    }

    if (trace.kind === "dispersion") {
      fragment.append(createExplorerSection("Values", statisticalValueList(trace.values), "stat-values"));
      fragment.append(createExplorerSection("Mean", explorerValue(trace.mean), "stat-mean"));
      fragment.append(createExplorerSection("Sum of squared deviations", explorerValue(trace.sumSquared), "stat-squared-deviations"));
      fragment.append(createExplorerSection(
        "Denominator",
        `${trace.divisor} (${trace.sample ? "sample: n − 1" : "population: n"})`,
        "stat-denominator"
      ));
      fragment.append(createExplorerSection("Variance", explorerValue(trace.variance), "stat-variance"));
      fragment.append(createExplorerSection("Standard deviation", explorerValue(trace.standardDeviation), "stat-standard-deviation"));
      return;
    }

    if (trace.kind === "rank") {
      fragment.append(createExplorerSection("Number to rank", explorerValue(trace.number), "stat-rank-number"));
      fragment.append(createExplorerSection(
        "Order",
        trace.ascending ? "Ascending · smallest value ranks 1" : "Descending · largest value ranks 1",
        "stat-rank-order"
      ));
      fragment.append(createExplorerSection("Ordered reference values", statisticalValueList(trace.sorted), "stat-rank-values"));
      if (trace.tieCount > 1) {
        fragment.append(createExplorerSection(
          "Tie rule",
          `${trace.tieCount} equal values share the same rank; later ranks contain the usual gap.`,
          "stat-rank-tie"
        ));
      }
      fragment.append(createExplorerSection("Rank", explorerValue(trace.result), "stat-result"));
      return;
    }

    if (trace.kind === "percentile") {
      fragment.append(createExplorerSection("Ordered values", statisticalValueList(trace.sorted), "stat-ordered-values"));
      if (trace.functionName === "QUARTILE.INC") {
        fragment.append(createExplorerSection("Quartile", explorerValue(trace.quart), "stat-quartile"));
      }
      fragment.append(createExplorerSection("Percentile fraction", explorerValue(trace.k), "stat-percentile-fraction"));
      fragment.append(createExplorerSection(
        "Position in ordered data",
        `${explorerValue(trace.index + 1)} (one-based conceptual position)`,
        "stat-percentile-position"
      ));
      if (trace.lowerIndex !== trace.upperIndex) {
        fragment.append(createExplorerSection(
          "Interpolation",
          `${explorerValue(trace.lowerValue)} + (${explorerValue(trace.upperValue)} − ${explorerValue(trace.lowerValue)}) × ${explorerValue(trace.fraction)}`,
          "stat-interpolation"
        ));
      }
      fragment.append(createExplorerSection("Result", explorerValue(trace.result), "stat-result"));
      return;
    }

    if (trace.kind === "paired") {
      if (trace.leftLabel) fragment.append(createExplorerSection("First array", trace.leftLabel, "stat-first-array"));
      if (trace.rightLabel) fragment.append(createExplorerSection("Second array", trace.rightLabel, "stat-second-array"));
      fragment.append(createExplorerSection("Paired observations", statisticalPairList(trace.pairs), "stat-pairs"));
      fragment.append(createExplorerSection("Mean of first array", explorerValue(trace.meanX), "stat-mean-x"));
      fragment.append(createExplorerSection("Mean of second array", explorerValue(trace.meanY), "stat-mean-y"));
      if (trace.functionName === "CORREL") {
        fragment.append(createExplorerSection(
          "Interpretation",
          trace.result > 0
            ? "Positive values tend to move together; values nearer 1 indicate a stronger positive linear relationship."
            : (trace.result < 0
              ? "One value tends to rise as the other falls; values nearer −1 indicate a stronger negative linear relationship."
              : "No linear relationship is indicated by this data."),
          "stat-correlation-interpretation"
        ));
      } else {
        fragment.append(createExplorerSection(
          "Sample denominator",
          `${trace.pairs.length - 1} (n − 1)`,
          "stat-covariance-denominator"
        ));
      }
      fragment.append(createExplorerSection("Result", explorerValue(trace.result), "stat-result"));
    }
  }

  function financialPercent(value) {
    if (typeof value !== "number" || !Number.isFinite(value)) return explorerValue(value);
    return `${(value * 100).toFixed(4).replace(/0+$/, "").replace(/\.$/, "")}%`;
  }

  function financialFlowList(flows, dated = false, discounted = false) {
    const list = createExplorerElement("div", "financial-flow-list");
    flows.forEach((flow, index) => {
      const row = createExplorerElement("div", "financial-flow-row");
      const label = dated
        ? (flow.dateDisplay || `Date ${index + 1}`)
        : `Period ${flow.period ?? index}`;
      row.append(createExplorerElement("span", "financial-flow-period", label));
      row.append(createExplorerElement("span", "financial-flow-value", explorerValue(flow.value)));
      if (discounted && flow.presentValue !== null && flow.presentValue !== undefined) {
        row.append(createExplorerElement("span", "financial-flow-present", `PV ${explorerValue(flow.presentValue)}`));
      }
      list.append(row);
    });
    return list;
  }

  function renderFinancialExplorer(fragment, trace) {
    if (trace.kind === "time-value") {
      fragment.append(createExplorerSection("Rate per period", financialPercent(trace.rate), "financial-rate"));
      fragment.append(createExplorerSection("Number of periods", explorerValue(trace.nper), "financial-periods"));
      fragment.append(createExplorerSection(
        "Payment timing",
        trace.type === 1 ? "Beginning of each period" : "End of each period",
        "financial-payment-type"
      ));
      if (trace.functionName === "PV") {
        fragment.append(createExplorerSection("Periodic payment", explorerValue(trace.payment), "financial-payment"));
        fragment.append(createExplorerSection("Future value", explorerValue(trace.futureValue), "financial-fv"));
        fragment.append(createExplorerSection("Present value", explorerValue(trace.result), "financial-result"));
      } else if (trace.functionName === "FV") {
        fragment.append(createExplorerSection("Periodic payment", explorerValue(trace.payment), "financial-payment"));
        fragment.append(createExplorerSection("Present value", explorerValue(trace.presentValue), "financial-pv"));
        fragment.append(createExplorerSection("Future value", explorerValue(trace.result), "financial-result"));
      } else {
        fragment.append(createExplorerSection("Present value", explorerValue(trace.presentValue), "financial-pv"));
        fragment.append(createExplorerSection("Future value", explorerValue(trace.futureValue), "financial-fv"));
        fragment.append(createExplorerSection("Periodic payment", explorerValue(trace.result), "financial-result"));
      }
      fragment.append(createExplorerSection(
        "Cash-flow sign convention",
        "Money received and money paid should use opposite signs. A loan received today is positive, so its repayments are normally negative.",
        "financial-sign-rule"
      ));
      return;
    }

    if (trace.kind === "npv") {
      fragment.append(createExplorerSection("Discount rate", financialPercent(trace.rate), "financial-rate"));
      fragment.append(createExplorerSection("Discounted cash-flow timeline", financialFlowList(trace.flows, false, true), "financial-timeline"));
      fragment.append(createExplorerSection(
        "Timing rule",
        "NPV treats the first supplied cash flow as occurring one period from today. Add an initial time-0 investment separately when needed.",
        "financial-npv-rule"
      ));
      fragment.append(createExplorerSection("Net present value", explorerValue(trace.result), "financial-result"));
      return;
    }

    if (trace.kind === "irr") {
      fragment.append(createExplorerSection("Cash-flow timeline", financialFlowList(trace.flows), "financial-timeline"));
      fragment.append(createExplorerSection("Starting guess", financialPercent(trace.guess), "financial-guess"));
      fragment.append(createExplorerSection("Solved by", `${trace.method} · ${trace.iterations} iteration${trace.iterations === 1 ? "" : "s"}`, "financial-method"));
      fragment.append(createExplorerSection(
        "IRR condition",
        "At this periodic rate, the present value of all cash flows (including period 0) is approximately zero.",
        "financial-irr-rule"
      ));
      fragment.append(createExplorerSection("Internal rate of return", financialPercent(trace.result), "financial-result"));
      return;
    }

    if (trace.kind === "xnpv") {
      fragment.append(createExplorerSection("Annual discount rate", financialPercent(trace.rate), "financial-rate"));
      fragment.append(createExplorerSection("Dated cash-flow timeline", financialFlowList(trace.flows, true, true), "financial-timeline"));
      fragment.append(createExplorerSection(
        "Actual timing",
        "Each cash flow is discounted from the first date using its exact day difference divided by 365.",
        "financial-date-rule"
      ));
      fragment.append(createExplorerSection("XNPV", explorerValue(trace.result), "financial-result"));
      return;
    }

    if (trace.kind === "xirr") {
      fragment.append(createExplorerSection("Dated cash-flow timeline", financialFlowList(trace.flows, true), "financial-timeline"));
      fragment.append(createExplorerSection("Starting guess", financialPercent(trace.guess), "financial-guess"));
      fragment.append(createExplorerSection("Solved by", `${trace.method} · ${trace.iterations} iteration${trace.iterations === 1 ? "" : "s"}`, "financial-method"));
      fragment.append(createExplorerSection(
        "XIRR condition",
        "This annualized rate makes XNPV approximately zero using the actual cash-flow dates.",
        "financial-xirr-rule"
      ));
      fragment.append(createExplorerSection("Annualized return", financialPercent(trace.result), "financial-result"));
    }
  }

  function advancedBranchList(trace) {
    const list = createExplorerElement("div", "advanced-branch-list");
    trace.branches.forEach((branch) => {
      const row = createExplorerElement("div", `advanced-branch-row${branch.selected ? " selected" : ""}`);
      row.append(createExplorerElement("span", "advanced-branch-condition", `${branch.index}. ${branch.conditionExpression}`));
      row.append(createExplorerElement("span", "advanced-branch-status", branch.condition ? "TRUE" : "FALSE"));
      row.append(createExplorerElement("span", "advanced-branch-value", branch.valueExpression));
      list.append(row);
    });
    return list;
  }

  function switchCaseList(trace) {
    const list = createExplorerElement("div", "advanced-branch-list");
    trace.cases.forEach((entry) => {
      const row = createExplorerElement("div", `advanced-branch-row${entry.selected ? " selected" : ""}`);
      row.append(createExplorerElement("span", "advanced-branch-condition", explorerValue(entry.candidate)));
      row.append(createExplorerElement("span", "advanced-branch-status", entry.matched ? "MATCH" : "no match"));
      row.append(createExplorerElement("span", "advanced-branch-value", entry.resultExpression));
      list.append(row);
    });
    return list;
  }

  function renderAdvancedExplorer(fragment, trace) {
    if (trace.kind === "ifs") {
      fragment.append(createExplorerSection("Conditions checked in order", advancedBranchList(trace), "advanced-ifs-branches"));
      fragment.append(createExplorerSection(
        "Selected branch",
        trace.matchedBranch ? `Condition ${trace.matchedBranch}` : "No condition returned TRUE",
        "advanced-selected-branch"
      ));
      if (trace.matchedBranch) fragment.append(createExplorerSection("Returned value", explorerValue(trace.result), "advanced-result"));
      return;
    }

    if (trace.kind === "switch") {
      fragment.append(createExplorerSection("Expression", explorerValue(trace.expression), "advanced-switch-expression"));
      fragment.append(createExplorerSection("Cases", switchCaseList(trace), "advanced-switch-cases"));
      if (trace.defaultUsed) {
        fragment.append(createExplorerSection("Fallback", "No case matched, so the default value was returned.", "advanced-switch-default"));
      } else if (!trace.defaultProvided && String(trace.result) === "#N/A") {
        fragment.append(createExplorerSection("Fallback", "No case matched and no default value was supplied.", "advanced-switch-default"));
      }
      if (!String(trace.result).startsWith("#")) {
        fragment.append(createExplorerSection("Returned value", explorerValue(trace.result), "advanced-result"));
      }
      return;
    }

    if (trace.kind === "choose") {
      fragment.append(createExplorerSection("Index number", explorerValue(trace.index), "advanced-choose-index"));
      fragment.append(createExplorerSection("Available choices", explorerValue(trace.optionCount), "advanced-choose-count"));
      if (trace.selectedExpression) {
        fragment.append(createExplorerSection("Selected expression", trace.selectedExpression, "advanced-choose-expression"));
        fragment.append(createExplorerSection("Returned value", explorerValue(trace.result), "advanced-result"));
      }
      return;
    }

    if (trace.kind === "let") {
      const list = createExplorerElement("div", "advanced-let-list");
      trace.bindings.forEach((binding) => {
        const row = createExplorerElement("div", "advanced-let-row");
        row.append(createExplorerElement("strong", "advanced-let-name", binding.name));
        row.append(createExplorerElement("code", "advanced-let-expression", binding.expression));
        row.append(createExplorerElement("span", "advanced-let-value", explorerValue(binding.value)));
        list.append(row);
      });
      fragment.append(createExplorerSection("Local names", list, "advanced-let-bindings"));
      fragment.append(createExplorerSection("Final calculation", trace.calculationExpression, "advanced-let-calculation"));
      fragment.append(createExplorerSection("Returned value", explorerValue(trace.result), "advanced-result"));
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

    if (lookup.kind === "match" || lookup.kind === "xmatch") {
      fragment.append(createExplorerSection("Lookup value", explorerValue(lookup.lookupValue), "lookup-value"));
      fragment.append(createExplorerSection("Lookup range", lookup.lookupRange.label, "lookup-range"));
      if (lookup.kind === "xmatch") {
        fragment.append(createExplorerSection("Match mode", explorerValue(lookup.matchMode), "match-mode"));
        fragment.append(createExplorerSection("Search mode", explorerValue(lookup.searchMode), "search-mode"));
      }
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
        lookup.kind === "xmatch" ? "What XMATCH returns" : "What MATCH returns",
        createExplorerElement(
          "div",
          "explorer-value lookup-note",
          lookup.kind === "xmatch"
            ? "XMATCH returns a relative position and uses exact matching by default."
            : "MATCH returns a relative position, not the cell value."
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

    if (explanation.spillReference) {
      fragment.append(createExplorerSection("Spill anchor", explanation.spillReference.anchor, "spill-reference-anchor"));
      fragment.append(createExplorerSection("Source spill range", explanation.spillReference.range, "spill-reference-range"));
      fragment.append(createExplorerSection(
        "Array shape",
        `${explanation.spillReference.rows} × ${explanation.spillReference.columns}`,
        "spill-reference-shape"
      ));
      const preview = window.FormulaEngine.makeArray(
        explanation.spillReference.rows,
        explanation.spillReference.columns,
        explanation.spillReference.values,
        { formats: explanation.spillReference.formats || undefined }
      );
      fragment.append(createExplorerSection("Spill contents", createArrayPreview(preview), "spill-reference-preview"));
    }

    if (explanation.referenceLocks?.some((entry) => entry.columnAbsolute || entry.rowAbsolute)) {
      const lines = explanation.referenceLocks.map((entry) => {
        const column = entry.columnAbsolute ? "locked column" : "relative column";
        const row = entry.rowAbsolute ? "locked row" : "relative row";
        return `${entry.address}: ${column}, ${row}`;
      }).join("\n");
      const locking = createExplorerElement("pre", "explorer-value explorer-reference-locks", lines);
      fragment.append(createExplorerSection("Reference locking", locking, "reference-locking"));
    }

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

    if (explanation.statistical) {
      renderStatisticalExplorer(fragment, explanation.statistical);
    }

    if (explanation.financial) {
      renderFinancialExplorer(fragment, explanation.financial);
    }

    if (explanation.advanced) {
      renderAdvancedExplorer(fragment, explanation.advanced);
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
        "array-sort-key-reference",
        "statistical-x-reference",
        "statistical-y-reference"
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
      analyzeStatistics: analyzeStatisticalForExplanation,
      analyzeFinancial: analyzeFinancialForExplanation,
      analyzeAdvanced: analyzeAdvancedForExplanation,
      analyzeError: analyzeErrorForExplanation,
      analyzeDynamic: analyzeDynamicForExplanation,
      spill: spillRanges.get(selectedReference) || null,
      spillError: model.spillError || null,
      numberFormat: model.numberFormat,
      numberFormatOverride: model.numberFormatOverride,
      formatOptions: model.formatOptions,
      displayedResult: displayedValue(model),
      getCellNumberFormat: calculatedCellNumberFormat,
      getSpill(reference) {
        const descriptor = spillRanges.get(String(reference).toUpperCase());
        return descriptor ? {
          ...descriptor,
          values: descriptor.values.map((row) => row.slice()),
          formats: descriptor.formats?.map((row) => row.slice()) || null,
          references: descriptor.references.slice()
        } : null;
      },
      formatValue: window.ExcelFormatting.formatValue
    });

    model.dependencies.forEach((reference) => {
      const cell = cellElements.get(reference);
      if (!cell) return;
      cell.classList.add("formula-reference");
      highlightedReferences.add(reference);
    });

    if (explanation.spillReference?.references) {
      explanation.spillReference.references.forEach((reference) => {
        const cell = cellElements.get(reference);
        if (!cell) return;
        cell.classList.add("array-source-reference");
        highlightedReferences.add(reference);
      });
    }

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
      } else if (lookup.kind === "match" || lookup.kind === "xmatch") {
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

    if (explanation.statistical?.kind === "paired") {
      explanation.statistical.pairs.forEach((pair) => {
        if (pair.xReference) {
          cellElements.get(pair.xReference)?.classList.add("statistical-x-reference");
          highlightedReferences.add(pair.xReference);
        }
        if (pair.yReference) {
          cellElements.get(pair.yReference)?.classList.add("statistical-y-reference");
          highlightedReferences.add(pair.yReference);
        }
      });
    }

    if (explanation.financial) {
      const flows = explanation.financial.flows || [];
      flows.forEach((flow) => {
        const valueReference = flow.reference || flow.valueReference;
        if (valueReference) {
          cellElements.get(valueReference)?.classList.add("financial-value-reference");
          highlightedReferences.add(valueReference);
        }
        if (flow.dateReference) {
          cellElements.get(flow.dateReference)?.classList.add("financial-date-reference");
          highlightedReferences.add(flow.dateReference);
        }
      });
    }

    renderFormulaExplorer(explanation);
  }

  function selectionBounds() {
    return {
      top: Math.min(state.selectionAnchorRow, state.selectionEndRow),
      bottom: Math.max(state.selectionAnchorRow, state.selectionEndRow),
      left: Math.min(state.selectionAnchorColumn, state.selectionEndColumn),
      right: Math.max(state.selectionAnchorColumn, state.selectionEndColumn)
    };
  }

  function selectionIsSingle() {
    const bounds = selectionBounds();
    return bounds.top === bounds.bottom && bounds.left === bounds.right;
  }

  function selectionRangeReference() {
    const bounds = selectionBounds();
    const first = cellReference(bounds.top, bounds.left);
    const last = cellReference(bounds.bottom, bounds.right);
    return first === last ? first : `${first}:${last}`;
  }

  function selectionReferences() {
    const bounds = selectionBounds();
    const references = [];
    for (let row = bounds.top; row <= bounds.bottom; row += 1) {
      for (let column = bounds.left; column <= bounds.right; column += 1) {
        references.push(cellReference(row, column));
      }
    }
    return references;
  }

  function selectionContainsDynamicArray() {
    return selectionReferences().some((reference) => {
      const projection = spillCells.get(reference);
      return Boolean(projection || spillRanges.has(reference));
    });
  }

  function selectionSummary() {
    const references = selectionReferences();
    const numeric = references
      .map((reference) => calculatedCellValue(reference))
      .filter((value) => typeof value === "number" && Number.isFinite(value));
    if (!numeric.length) return { count: 0, sum: 0, average: null };
    const sum = numeric.reduce((total, value) => total + value, 0);
    return { count: numeric.length, sum, average: sum / numeric.length };
  }

  function clearSelectionVisuals() {
    cellElements.forEach((cell) => {
      cell.classList.remove(
        "active",
        "range-selected",
        "selection-top",
        "selection-right",
        "selection-bottom",
        "selection-left",
        "selection-fill-corner",
        "copy-source",
        "cut-source",
        "fill-preview"
      );
      cell.setAttribute("aria-selected", "false");
      if (cell.tabIndex === 0) cell.tabIndex = -1;
    });
    columnHeaders.forEach((header) => header.classList.remove("selected-header"));
    rowHeaders.forEach((header) => header.classList.remove("selected-header"));
    cornerHeader?.classList.remove("selected-header");
  }

  function renderClipboardOutline() {
    if (!state.clipboard || !state.clipboardMode) return;
    const { bounds } = state.clipboard;
    for (let row = bounds.top; row <= bounds.bottom; row += 1) {
      for (let column = bounds.left; column <= bounds.right; column += 1) {
        const cell = cellElements.get(cellReference(row, column));
        cell?.classList.add(state.clipboardMode === "cut" ? "cut-source" : "copy-source");
      }
    }
  }

  function renderSelectionVisuals() {
    clearSelectionVisuals();
    const bounds = selectionBounds();
    for (let row = bounds.top; row <= bounds.bottom; row += 1) {
      for (let column = bounds.left; column <= bounds.right; column += 1) {
        const cell = cellElements.get(cellReference(row, column));
        if (!cell) continue;
        cell.classList.add("range-selected");
        if (row === bounds.top) cell.classList.add("selection-top");
        if (row === bounds.bottom) cell.classList.add("selection-bottom");
        if (column === bounds.left) cell.classList.add("selection-left");
        if (column === bounds.right) cell.classList.add("selection-right");
        cell.setAttribute("aria-selected", "true");
      }
    }

    const active = activeCellElement();
    active?.classList.add("active");
    if (active) active.tabIndex = 0;

    const wholeSheet = bounds.top === 0 && bounds.bottom === ROW_COUNT - 1
      && bounds.left === 0 && bounds.right === COLUMN_COUNT - 1;
    if (wholeSheet) cornerHeader?.classList.add("selected-header");
    for (let column = bounds.left; column <= bounds.right; column += 1) {
      columnHeaders[column]?.classList.add("selected-header");
    }
    for (let row = bounds.top; row <= bounds.bottom; row += 1) {
      rowHeaders[row]?.classList.add("selected-header");
    }

    if (!selectionContainsDynamicArray()) {
      cellElements.get(cellReference(bounds.bottom, bounds.right))?.classList.add("selection-fill-corner");
    }
    renderClipboardOutline();
  }

  function renderSelectionExplorer() {
    clearReferenceHighlights();
    const summary = selectionSummary();
    explorerTitle.textContent = "Selection";
    explorerContent.replaceChildren();
    const range = createExplorerElement("div", "explorer-field");
    range.append(
      createExplorerElement("span", "explorer-field-label", "Selected range"),
      createExplorerElement("strong", "explorer-field-value", selectionRangeReference())
    );
    explorerContent.append(range);
    const totalCells = selectionReferences().length;
    const cells = createExplorerElement("div", "explorer-field");
    cells.append(
      createExplorerElement("span", "explorer-field-label", "Cells"),
      createExplorerElement("strong", "explorer-field-value", String(totalCells))
    );
    explorerContent.append(cells);
    if (summary.count) {
      [["Numeric cells", summary.count], ["Sum", summary.sum], ["Average", summary.average]].forEach(([label, value]) => {
        const field = createExplorerElement("div", "explorer-field");
        field.append(
          createExplorerElement("span", "explorer-field-label", label),
          createExplorerElement("strong", "explorer-field-value", explorerValue(value))
        );
        explorerContent.append(field);
      });
    }
  }

  function updateSelectionDisplay() {
    const reference = cellReference(state.activeRow, state.activeColumn);
    const projection = spillCells.get(reference);
    const child = projection && projection.spillOwner !== reference;
    const rangeReference = selectionRangeReference();
    nameBox.value = rangeReference;
    formulaInput.value = child
      ? (cellData.get(projection.spillOwner)?.input || "")
      : cellInput(state.activeRow, state.activeColumn);
    formulaInput.readOnly = Boolean(child);
    formulaFunctionButton.disabled = Boolean(child);
    formulaInput.classList.toggle("spill-formula-readonly", Boolean(child));

    if (selectionIsSingle()) {
      selectionStatus.textContent = child
        ? `Selected: ${reference} · Spilled from ${projection.spillOwner} · Edit the anchor to make changes.`
        : `Selected: ${reference}`;
    } else {
      const summary = selectionSummary();
      const parts = [`Selected: ${rangeReference}`];
      if (summary.count) {
        parts.push(`Count: ${summary.count}`);
        parts.push(`Sum: ${explorerValue(summary.sum)}`);
        parts.push(`Average: ${explorerValue(summary.average)}`);
      }
      selectionStatus.textContent = parts.join(" · ");
    }
    renderSelectionVisuals();
    syncFormatToolbar();
    if (selectionIsSingle()) updateFormulaTrace();
    else renderSelectionExplorer();
  }

  function selectedFormatState() {
    const formats = selectionReferences().map((reference) => {
      const projection = spillCells.get(reference);
      const model = projection || cellData.get(reference);
      const override = cellFormatOverrides.get(reference);
      return model?.numberFormat || override?.type || "General";
    });
    const unique = [...new Set(formats)];
    return unique.length === 1 ? unique[0] : "Mixed";
  }

  function syncFormatToolbar() {
    const reference = cellReference(state.activeRow, state.activeColumn);
    const projection = spillCells.get(reference);
    const child = projection && projection.spillOwner !== reference;
    const numberFormat = selectedFormatState();
    numberFormatSelect.value = numberFormat;
    currencyFormatButton.classList.toggle("active", numberFormat === "Currency");
    percentageFormatButton.classList.toggle("active", numberFormat === "Percentage");
    numberFormatButton.classList.toggle("active", numberFormat === "Number");
    const dateSelected = numberFormat === "Date";
    const allChildren = selectionReferences().every((cell) => {
      const spill = spillCells.get(cell);
      return spill && spill.spillOwner !== cell;
    });
    numberFormatSelect.disabled = allChildren;
    currencyFormatButton.disabled = allChildren;
    percentageFormatButton.disabled = allChildren;
    numberFormatButton.disabled = allChildren;
    decreaseDecimalButton.disabled = dateSelected || allChildren;
    increaseDecimalButton.disabled = dateSelected || allChildren;
  }

  function formatSelection(numberFormat, options = {}) {
    if (!Object.values(window.ExcelFormatting.NUMBER_FORMATS).includes(numberFormat)) return;
    const before = state.historyRestoring ? null : workbookSnapshot();
    selectionReferences().forEach((reference) => {
      const projection = spillCells.get(reference);
      if (projection && projection.spillOwner !== reference) return;
      cellFormatOverrides.set(reference, {
        type: numberFormat,
        ...window.ExcelFormatting.normalizeFormatOptions(numberFormat, options)
      });
    });
    recalculateAll();
    updateSelectionDisplay();
    commitHistory(before, `format ${selectionRangeReference()}`);
  }

  function formatActiveCell(numberFormat, options = {}) {
    formatSelection(numberFormat, options);
  }

  function adjustActiveDecimals(offset) {
    const before = state.historyRestoring ? null : workbookSnapshot();
    selectionReferences().forEach((reference) => {
      const projection = spillCells.get(reference);
      if (projection && projection.spillOwner !== reference) return;
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
        : 0;
      cellFormatOverrides.set(reference, {
        type: nextFormat,
        ...window.ExcelFormatting.normalizeFormatOptions(nextFormat, {
          ...currentOptions,
          decimals: Math.max(0, currentDecimals + offset)
        })
      });
    });
    recalculateAll();
    updateSelectionDisplay();
    commitHistory(before, `${offset > 0 ? "increase" : "decrease"} decimals`);
  }

  function selectCell(row, column, options = {}) {
    const nextRow = Math.max(0, Math.min(ROW_COUNT - 1, row));
    const nextColumn = Math.max(0, Math.min(COLUMN_COUNT - 1, column));
    const nextCell = cellElements.get(cellReference(nextRow, nextColumn));

    if (state.editingCell && state.editingCell !== nextCell) {
      finishCellEdit(true);
    }

    if (options.extend) {
      state.selectionEndRow = nextRow;
      state.selectionEndColumn = nextColumn;
    } else {
      state.activeRow = nextRow;
      state.activeColumn = nextColumn;
      state.selectionAnchorRow = nextRow;
      state.selectionAnchorColumn = nextColumn;
      state.selectionEndRow = nextRow;
      state.selectionEndColumn = nextColumn;
    }
    updateSelectionDisplay();

    const focusCell = options.extend ? activeCellElement() : nextCell;
    if (options.focus !== false) focusCell?.focus({ preventScroll: true });
    scrollCellIntoView(nextCell);
  }

  function selectRange(start, end = start, options = {}) {
    const first = coordinatesForReference(start);
    const last = coordinatesForReference(end);
    const active = options.active ? coordinatesForReference(options.active) : first;
    state.activeRow = active.row;
    state.activeColumn = active.column;
    state.selectionAnchorRow = first.row;
    state.selectionAnchorColumn = first.column;
    state.selectionEndRow = last.row;
    state.selectionEndColumn = last.column;
    updateSelectionDisplay();
    const endCell = cellElements.get(cellReference(last.row, last.column));
    if (options.focus !== false) activeCellElement()?.focus({ preventScroll: true });
    if (endCell) scrollCellIntoView(endCell);
  }

  function extendSelection(rowOffset, columnOffset) {
    const nextRow = Math.max(0, Math.min(ROW_COUNT - 1, state.selectionEndRow + rowOffset));
    const nextColumn = Math.max(0, Math.min(COLUMN_COUNT - 1, state.selectionEndColumn + columnOffset));
    selectCell(nextRow, nextColumn, { extend: true });
  }

  function moveSelection(rowOffset, columnOffset) {
    selectCell(state.activeRow + rowOffset, state.activeColumn + columnOffset);
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

  function isPrintableKey(event) {
    return event.key.length === 1 && !event.ctrlKey && !event.metaKey && !event.altKey;
  }

  function snapshotReference(reference, selectedSet = null) {
    const projection = spillCells.get(reference);
    if (projection && projection.spillOwner !== reference) {
      if (selectedSet?.has(projection.spillOwner)) return { skip: true, source: reference };
      const value = projection.value;
      return {
        source: reference,
        input: typeof value === "boolean" ? (value ? "TRUE" : "FALSE") : String(value ?? ""),
        valueOnly: true,
        format: {
          type: projection.numberFormat || "General",
          ...(projection.formatOptions || {})
        }
      };
    }
    const model = cellData.get(reference);
    const override = cellFormatOverrides.get(reference);
    return {
      source: reference,
      input: model?.input || "",
      valueOnly: false,
      format: override ? { ...override } : null
    };
  }

  function selectionSnapshot() {
    const bounds = selectionBounds();
    const selectedSet = new Set(selectionReferences());
    const cells = [];
    for (let row = bounds.top; row <= bounds.bottom; row += 1) {
      const outputRow = [];
      for (let column = bounds.left; column <= bounds.right; column += 1) {
        outputRow.push(snapshotReference(cellReference(row, column), selectedSet));
      }
      cells.push(outputRow);
    }
    return { bounds: { ...bounds }, rows: cells.length, columns: cells[0]?.length || 0, cells };
  }

  function systemClipboardText(snapshot) {
    return snapshot.cells.map((row) => row.map((entry) => {
      if (entry.skip) return "";
      const model = cellData.get(entry.source) || spillCells.get(entry.source);
      return displayedValue(model).replace(/\t/g, " ").replace(/[\r\n]+/g, " ");
    }).join("\t")).join("\n");
  }

  function copySelection(mode = "copy") {
    if (mode === "cut") {
      const refs = selectionReferences();
      const invalidChild = refs.some((reference) => {
        const projection = spillCells.get(reference);
        return projection && projection.spillOwner !== reference && !refs.includes(projection.spillOwner);
      });
      if (invalidChild) {
        selectionStatus.textContent = "Cut the dynamic-array anchor instead of a spill child.";
        return false;
      }
    }
    state.clipboard = selectionSnapshot();
    state.clipboardMode = mode;
    renderSelectionVisuals();
    const text = systemClipboardText(state.clipboard);
    navigator.clipboard?.writeText?.(text).catch(() => {});
    selectionStatus.textContent = `${mode === "cut" ? "Cut" : "Copied"}: ${selectionRangeReference()}`;
    return true;
  }

  function cancelClipboard() {
    state.clipboard = null;
    state.clipboardMode = null;
    renderSelectionVisuals();
  }

  function canWriteDestination(reference, destinationSet) {
    const projection = spillCells.get(reference);
    if (!projection || projection.spillOwner === reference) return true;
    return destinationSet.has(projection.spillOwner);
  }

  function clearAuthoredReferencesRaw(references) {
    const owners = new Set();
    references.forEach((reference) => {
      const projection = spillCells.get(reference);
      if (projection) owners.add(projection.spillOwner);
    });
    owners.forEach((owner) => {
      if (references.includes(owner)) {
        cellData.delete(owner);
        cellFormatOverrides.delete(owner);
      }
    });
    references.forEach((reference) => {
      const projection = spillCells.get(reference);
      if (projection && projection.spillOwner !== reference) return;
      cellData.delete(reference);
      cellFormatOverrides.delete(reference);
    });
  }

  function applyMutationBatch({ clears = [], writes = [] }, status = "") {
    const before = state.historyRestoring ? null : workbookSnapshot();
    clearAuthoredReferencesRaw([...new Set(clears)]);
    writes.forEach((write) => {
      const { row, column } = coordinatesForReference(write.reference);
      storeCellInput(row, column, write.input ?? "");
      if (write.format) {
        cellFormatOverrides.set(write.reference, {
          ...write.format,
          ...window.ExcelFormatting.normalizeFormatOptions(write.format.type || "General", write.format)
        });
      } else if (write.clearFormat) {
        cellFormatOverrides.delete(write.reference);
      }
    });
    recalculateAll();
    updateSelectionDisplay();
    commitHistory(before, status || "worksheet change");
    if (status) selectionStatus.textContent = status;
  }

  function buildPasteWrites(snapshot, targetRow, targetColumn, translate = true) {
    if (targetRow + snapshot.rows > ROW_COUNT || targetColumn + snapshot.columns > COLUMN_COUNT) {
      return { error: "Paste would extend beyond the worksheet." };
    }
    const writes = [];
    const destinationSet = new Set();
    for (let rowIndex = 0; rowIndex < snapshot.rows; rowIndex += 1) {
      for (let columnIndex = 0; columnIndex < snapshot.columns; columnIndex += 1) {
        destinationSet.add(cellReference(targetRow + rowIndex, targetColumn + columnIndex));
      }
    }
    for (const reference of destinationSet) {
      if (!canWriteDestination(reference, destinationSet)) {
        return { error: `You can't paste into spill cell ${reference}. Edit its anchor instead.` };
      }
    }

    snapshot.cells.forEach((row, rowIndex) => {
      row.forEach((entry, columnIndex) => {
        if (entry.skip) return;
        const destination = cellReference(targetRow + rowIndex, targetColumn + columnIndex);
        const sourceCoordinates = window.FormulaEngine.parseReference(entry.source);
        const destinationCoordinates = window.FormulaEngine.parseReference(destination);
        let input = entry.input;
        if (translate && typeof input === "string" && input.startsWith("=")) {
          input = window.FormulaEngine.translateFormula(
            input,
            destinationCoordinates.row - sourceCoordinates.row,
            destinationCoordinates.column - sourceCoordinates.column,
            { rowLimit: ROW_COUNT, columnLimit: COLUMN_COUNT }
          );
        }
        writes.push({ reference: destination, input, format: entry.format ? { ...entry.format } : null, clearFormat: !entry.format });
      });
    });
    return { writes, destinationSet };
  }

  function pasteInternal(targetReference = cellReference(state.activeRow, state.activeColumn)) {
    if (!state.clipboard) return false;
    const target = coordinatesForReference(targetReference);
    const result = buildPasteWrites(state.clipboard, target.row, target.column, state.clipboardMode !== "cut");
    if (result.error) {
      selectionStatus.textContent = result.error;
      return false;
    }
    const clears = state.clipboardMode === "cut"
      ? (() => {
        const refs = [];
        const { bounds } = state.clipboard;
        for (let row = bounds.top; row <= bounds.bottom; row += 1) {
          for (let column = bounds.left; column <= bounds.right; column += 1) refs.push(cellReference(row, column));
        }
        return refs;
      })()
      : [];
    applyMutationBatch({ clears, writes: result.writes });
    const end = cellReference(target.row + state.clipboard.rows - 1, target.column + state.clipboard.columns - 1);
    selectRange(targetReference, end, { focus: false });
    const wasCut = state.clipboardMode === "cut";
    if (wasCut) cancelClipboard();
    selectionStatus.textContent = wasCut ? `Moved to ${targetReference}` : `Pasted at ${targetReference}`;
    return true;
  }

  function parseExternalClipboard(text) {
    const normalized = String(text || "").replace(/\r\n/g, "\n").replace(/\r/g, "\n");
    const rows = normalized.split("\n");
    if (rows.length && rows[rows.length - 1] === "") rows.pop();
    return rows.map((row) => row.split("\t"));
  }

  function pasteExternal(text, targetReference = cellReference(state.activeRow, state.activeColumn)) {
    const rows = parseExternalClipboard(text);
    if (!rows.length) return false;
    const width = Math.max(...rows.map((row) => row.length));
    const target = coordinatesForReference(targetReference);
    if (target.row + rows.length > ROW_COUNT || target.column + width > COLUMN_COUNT) {
      selectionStatus.textContent = "Paste would extend beyond the worksheet.";
      return false;
    }
    const destinationSet = new Set();
    rows.forEach((row, r) => row.forEach((_, c) => destinationSet.add(cellReference(target.row + r, target.column + c))));
    for (const reference of destinationSet) {
      if (!canWriteDestination(reference, destinationSet)) {
        selectionStatus.textContent = `You can't paste into spill cell ${reference}.`;
        return false;
      }
    }
    const writes = [];
    rows.forEach((row, r) => row.forEach((input, c) => {
      writes.push({ reference: cellReference(target.row + r, target.column + c), input });
    }));
    applyMutationBatch({ writes });
    selectRange(targetReference, cellReference(target.row + rows.length - 1, target.column + width - 1), { focus: false });
    selectionStatus.textContent = `Pasted ${rows.length} × ${width} cells.`;
    return true;
  }

  function deleteSelection() {
    const refs = selectionReferences();
    const childWithoutAnchor = refs.some((reference) => {
      const projection = spillCells.get(reference);
      return projection && projection.spillOwner !== reference && !refs.includes(projection.spillOwner);
    });
    if (childWithoutAnchor) {
      selectionStatus.textContent = "You can't delete part of a spilled array. Include or edit its anchor.";
      return false;
    }
    applyMutationBatch({ clears: refs }, `Cleared ${selectionRangeReference()}`);
    return true;
  }

  function fillSourceEntries(bounds) {
    const selectedSet = new Set(selectionReferences());
    const cells = [];
    for (let row = bounds.top; row <= bounds.bottom; row += 1) {
      const output = [];
      for (let column = bounds.left; column <= bounds.right; column += 1) {
        output.push(snapshotReference(cellReference(row, column), selectedSet));
      }
      cells.push(output);
    }
    return cells;
  }

  function numericLiteral(entry) {
    if (!entry || entry.skip || typeof entry.input !== "string" || entry.input.startsWith("=")) return null;
    const trimmed = entry.input.trim();
    return /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$/.test(trimmed) ? Number(trimmed) : null;
  }

  function fillWritesForTarget(targetRow, targetColumn) {
    const bounds = selectionBounds();
    const source = fillSourceEntries(bounds);
    const height = bounds.bottom - bounds.top + 1;
    const width = bounds.right - bounds.left + 1;
    const verticalDistance = targetRow < bounds.top ? targetRow - bounds.top
      : targetRow > bounds.bottom ? targetRow - bounds.bottom : 0;
    const horizontalDistance = targetColumn < bounds.left ? targetColumn - bounds.left
      : targetColumn > bounds.right ? targetColumn - bounds.right : 0;
    if (!verticalDistance && !horizontalDistance) return { writes: [], range: bounds };
    const vertical = Math.abs(verticalDistance) >= Math.abs(horizontalDistance);
    const targetBounds = { ...bounds };
    if (vertical) {
      if (targetRow < bounds.top) targetBounds.top = targetRow;
      else targetBounds.bottom = targetRow;
    } else if (targetColumn < bounds.left) targetBounds.left = targetColumn;
    else targetBounds.right = targetColumn;

    if (targetBounds.top < 0 || targetBounds.left < 0
      || targetBounds.bottom >= ROW_COUNT || targetBounds.right >= COLUMN_COUNT) {
      return { error: "Fill would extend beyond the worksheet." };
    }

    const destinationSet = new Set();
    for (let row = targetBounds.top; row <= targetBounds.bottom; row += 1) {
      for (let column = targetBounds.left; column <= targetBounds.right; column += 1) {
        if (row >= bounds.top && row <= bounds.bottom && column >= bounds.left && column <= bounds.right) continue;
        destinationSet.add(cellReference(row, column));
      }
    }
    for (const reference of destinationSet) {
      if (!canWriteDestination(reference, destinationSet)) return { error: `Fill is blocked by spill cell ${reference}.` };
    }

    let series = null;
    if (vertical && width === 1 && height >= 2) {
      const values = source.map((row) => numericLiteral(row[0]));
      if (values.every((value) => value !== null)) series = { axis: "row", start: values[0], step: values[1] - values[0] };
    } else if (!vertical && height === 1 && width >= 2) {
      const values = source[0].map(numericLiteral);
      if (values.every((value) => value !== null)) series = { axis: "column", start: values[0], step: values[1] - values[0] };
    }

    const writes = [];
    destinationSet.forEach((destination) => {
      const coords = window.FormulaEngine.parseReference(destination);
      let entry;
      let input;
      if (series) {
        const offset = series.axis === "row" ? coords.row - bounds.top : coords.column - bounds.left;
        input = String(series.start + series.step * offset);
        const sourceRow = Math.max(0, Math.min(height - 1, ((coords.row - bounds.top) % height + height) % height));
        const sourceColumn = Math.max(0, Math.min(width - 1, ((coords.column - bounds.left) % width + width) % width));
        entry = source[sourceRow][sourceColumn];
      } else {
        const sourceRow = ((coords.row - bounds.top) % height + height) % height;
        const sourceColumn = ((coords.column - bounds.left) % width + width) % width;
        entry = source[sourceRow][sourceColumn];
        if (entry.skip) return;
        input = entry.input;
        if (typeof input === "string" && input.startsWith("=")) {
          const sourceCoords = window.FormulaEngine.parseReference(entry.source);
          input = window.FormulaEngine.translateFormula(
            input,
            coords.row - sourceCoords.row,
            coords.column - sourceCoords.column,
            { rowLimit: ROW_COUNT, columnLimit: COLUMN_COUNT }
          );
        }
      }
      writes.push({ reference: destination, input, format: entry?.format ? { ...entry.format } : null, clearFormat: !entry?.format });
    });
    return { writes, range: targetBounds };
  }

  function renderFillPreview(targetRow, targetColumn) {
    cellElements.forEach((cell) => cell.classList.remove("fill-preview"));
    const result = fillWritesForTarget(targetRow, targetColumn);
    if (result.error) return;
    result.writes.forEach((write) => cellElements.get(write.reference)?.classList.add("fill-preview"));
  }

  function performFillTo(targetRow, targetColumn) {
    cellElements.forEach((cell) => cell.classList.remove("fill-preview"));
    if (selectionContainsDynamicArray()) {
      selectionStatus.textContent = "Fill is disabled for dynamic-array spill ranges.";
      return false;
    }
    const result = fillWritesForTarget(targetRow, targetColumn);
    if (result.error) {
      selectionStatus.textContent = result.error;
      return false;
    }
    if (!result.writes.length) return false;
    applyMutationBatch({ writes: result.writes });
    selectRange(
      cellReference(result.range.top, result.range.left),
      cellReference(result.range.bottom, result.range.right),
      { active: cellReference(selectionBounds().top, selectionBounds().left), focus: false }
    );
    selectionStatus.textContent = `Filled ${selectionRangeReference()}`;
    return true;
  }

  function isFillHandleHit(event, cell) {
    if (!cell?.classList.contains("selection-fill-corner")) return false;
    const rect = cell.getBoundingClientRect();
    return event.clientX >= rect.right - 9 && event.clientY >= rect.bottom - 9;
  }

  grid.addEventListener("mousedown", (event) => {
    if (event.button !== 0 || state.editingCell) return;
    const cell = event.target.closest(".sheet-cell");
    const columnHeader = event.target.closest(".column-header");
    const rowHeader = event.target.closest(".row-header");
    const corner = event.target.closest(".corner-cell");

    if (cell) {
      const row = Number(cell.dataset.row);
      const column = Number(cell.dataset.column);
      if (isFillHandleHit(event, cell)) {
        event.preventDefault();
        state.fillDragging = true;
        state.fillHoverRow = row;
        state.fillHoverColumn = column;
        return;
      }
      event.preventDefault();
      if (event.shiftKey) {
        state.selectionEndRow = row;
        state.selectionEndColumn = column;
        updateSelectionDisplay();
      } else {
        selectCell(row, column, { focus: false });
      }
      state.mouseSelecting = true;
      return;
    }

    if (columnHeader) {
      event.preventDefault();
      const column = Number(columnHeader.dataset.column);
      state.activeRow = 0;
      state.activeColumn = column;
      state.selectionAnchorRow = 0;
      state.selectionAnchorColumn = column;
      state.selectionEndRow = ROW_COUNT - 1;
      state.selectionEndColumn = column;
      updateSelectionDisplay();
      return;
    }

    if (rowHeader) {
      event.preventDefault();
      const row = Number(rowHeader.dataset.row);
      state.activeRow = row;
      state.activeColumn = 0;
      state.selectionAnchorRow = row;
      state.selectionAnchorColumn = 0;
      state.selectionEndRow = row;
      state.selectionEndColumn = COLUMN_COUNT - 1;
      updateSelectionDisplay();
      return;
    }

    if (corner) {
      event.preventDefault();
      selectRange("A1", "Z50", { focus: false });
    }
  });

  grid.addEventListener("mouseover", (event) => {
    const cell = event.target.closest(".sheet-cell");
    if (!cell) return;
    const row = Number(cell.dataset.row);
    const column = Number(cell.dataset.column);
    if (state.fillDragging) {
      state.fillHoverRow = row;
      state.fillHoverColumn = column;
      renderFillPreview(row, column);
      return;
    }
    if (!state.mouseSelecting) return;
    state.selectionEndRow = row;
    state.selectionEndColumn = column;
    updateSelectionDisplay();
  });

  document.addEventListener("mouseup", () => {
    if (state.fillDragging) {
      const row = state.fillHoverRow;
      const column = state.fillHoverColumn;
      state.fillDragging = false;
      performFillTo(row, column);
    }
    state.mouseSelecting = false;
  });

  grid.addEventListener("click", (event) => {
    const cell = event.target.closest(".sheet-cell");
    if (!cell || state.editingCell === cell) return;
    activeCellElement()?.focus({ preventScroll: true });
  });

  grid.addEventListener("dblclick", (event) => {
    const cell = event.target.closest(".sheet-cell");
    if (!cell) return;
    selectCell(Number(cell.dataset.row), Number(cell.dataset.column), { focus: false });
    startCellEdit(cell);
  });

  grid.addEventListener("input", (event) => {
    const cell = event.target.closest(".sheet-cell.editing");
    if (!cell) return;
    formulaInput.value = normalizeInput(cell.textContent);
  });

  grid.addEventListener("blur", (event) => {
    const blurredCell = event.target;
    if (blurredCell !== state.editingCell) return;
    window.setTimeout(() => {
      if (blurredCell !== state.editingCell) return;
      const formulaAuthoring = document.querySelector(".formula-controls")?.classList.contains("formula-editing");
      if (!formulaAuthoring) finishCellEdit(true);
    }, 30);
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

    if ((event.ctrlKey || event.metaKey) && !event.altKey) {
      const key = event.key.toLowerCase();
      if (key === "z") {
        event.preventDefault();
        if (event.shiftKey) redoWorkbook();
        else undoWorkbook();
        return;
      }
      if (key === "y") {
        event.preventDefault();
        redoWorkbook();
        return;
      }
      if (key === "s") {
        event.preventDefault();
        saveWorkbook();
        selectionStatus.textContent = "Workbook saved locally.";
        return;
      }
      if (key === "c") {
        event.preventDefault();
        copySelection("copy");
        return;
      }
      if (key === "x") {
        event.preventDefault();
        copySelection("cut");
        return;
      }
      if (key === "a") {
        event.preventDefault();
        selectRange("A1", "Z50");
        return;
      }
    }

    const navigation = {
      ArrowUp: [-1, 0],
      ArrowDown: [1, 0],
      ArrowLeft: [0, -1],
      ArrowRight: [0, 1]
    };

    if (navigation[event.key]) {
      event.preventDefault();
      if (event.shiftKey) extendSelection(...navigation[event.key]);
      else moveSelection(...navigation[event.key]);
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
      deleteSelection();
    } else if (event.key === "Escape") {
      event.preventDefault();
      cancelClipboard();
      updateSelectionDisplay();
    } else if (isPrintableKey(event)) {
      event.preventDefault();
      if (!selectionIsSingle()) selectCell(state.activeRow, state.activeColumn, { focus: false });
      startCellEdit(activeCellElement(), event.key);
    }
  });

  document.addEventListener("copy", (event) => {
    if (!document.activeElement?.classList?.contains("sheet-cell") || state.editingCell) return;
    const text = systemClipboardText(selectionSnapshot());
    event.clipboardData?.setData("text/plain", text);
    event.preventDefault();
    copySelection("copy");
  });

  document.addEventListener("cut", (event) => {
    if (!document.activeElement?.classList?.contains("sheet-cell") || state.editingCell) return;
    const text = systemClipboardText(selectionSnapshot());
    event.clipboardData?.setData("text/plain", text);
    event.preventDefault();
    copySelection("cut");
  });

  document.addEventListener("paste", (event) => {
    if (!document.activeElement?.classList?.contains("sheet-cell") || state.editingCell) return;
    event.preventDefault();
    const text = event.clipboardData?.getData("text/plain") || "";
    if (state.clipboard) {
      const internalText = systemClipboardText(state.clipboard);
      if (!text || text === internalText) {
        pasteInternal();
        return;
      }
      cancelClipboard();
    }
    pasteExternal(text);
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

  nameBox.addEventListener("keydown", (event) => {
    if (event.key !== "Enter") return;
    event.preventDefault();
    const value = nameBox.value.trim().toUpperCase();
    const match = value.match(/^(\$?[A-Z]+\$?[1-9]\d*)(?:\s*:\s*(\$?[A-Z]+\$?[1-9]\d*))?$/);
    if (!match) {
      nameBox.value = selectionRangeReference();
      selectionStatus.textContent = "Enter a cell such as P25 or a range such as B2:D6.";
      return;
    }
    try {
      const start = window.FormulaEngine.formatReference(window.FormulaEngine.parseReference(match[1]), { absolute: false });
      const end = match[2]
        ? window.FormulaEngine.formatReference(window.FormulaEngine.parseReference(match[2]), { absolute: false })
        : start;
      selectRange(start, end);
    } catch (error) {
      nameBox.value = selectionRangeReference();
      selectionStatus.textContent = "That reference is outside this worksheet.";
    }
  });

  nameBox.addEventListener("blur", () => {
    if (!/^\$?[A-Z]+\$?[1-9]\d*(?:\s*:\s*\$?[A-Z]+\$?[1-9]\d*)?$/i.test(nameBox.value.trim())) {
      nameBox.value = selectionRangeReference();
    }
  });

  undoButton?.addEventListener("click", () => undoWorkbook());
  redoButton?.addEventListener("click", () => redoWorkbook());
  resetWorkbookButton?.addEventListener("click", () => resetWorkbook());

  numberFormatSelect.addEventListener("change", () => {
    if (numberFormatSelect.value !== "Mixed") formatActiveCell(numberFormatSelect.value);
  });
  currencyFormatButton.addEventListener("click", () => formatActiveCell("Currency"));
  percentageFormatButton.addEventListener("click", () => formatActiveCell("Percentage"));
  numberFormatButton.addEventListener("click", () => formatActiveCell("Number"));
  decreaseDecimalButton.addEventListener("click", () => adjustActiveDecimals(-1));
  increaseDecimalButton.addEventListener("click", () => adjustActiveDecimals(1));

  const restoredWorkbook = loadPersistedWorkbook();
  if (!restoredWorkbook) seedSampleData();
  createGrid();
  if (state.pendingSelection) {
    try {
      selectRange(
        state.pendingSelection.start || state.pendingSelection.active || "A1",
        state.pendingSelection.end || state.pendingSelection.active || "A1",
        { active: state.pendingSelection.active || state.pendingSelection.start || "A1", focus: false }
      );
    } catch (error) {
      selectCell(0, 0, { focus: false });
    }
  } else {
    selectCell(0, 0, { focus: false });
  }
  updateHistoryControls();
  if (!restoredWorkbook) saveWorkbook();

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
    getActiveReference() {
      return cellReference(state.activeRow, state.activeColumn);
    },
    getSelection() {
      return {
        range: selectionRangeReference(),
        bounds: { ...selectionBounds() },
        references: selectionReferences().slice(),
        active: cellReference(state.activeRow, state.activeColumn)
      };
    },
    selectCell(reference) {
      const { row, column } = coordinatesForReference(reference);
      selectCell(row, column);
    },
    selectRange(start, end = start) {
      selectRange(start, end);
    },
    copySelection(mode = "copy") {
      return copySelection(mode);
    },
    pasteInternal(reference) {
      return pasteInternal(reference || cellReference(state.activeRow, state.activeColumn));
    },
    pasteExternal(text, reference) {
      return pasteExternal(text, reference || cellReference(state.activeRow, state.activeColumn));
    },
    fillTo(reference) {
      const { row, column } = coordinatesForReference(reference);
      return performFillTo(row, column);
    },
    clearSelection: deleteSelection,
    undo: undoWorkbook,
    redo: redoWorkbook,
    saveWorkbook,
    resetWorkbook,
    getHistoryDepth() {
      return { undo: state.undoStack.length, redo: state.redoStack.length };
    },
    getWorkbookSnapshot() {
      return JSON.parse(JSON.stringify(workbookSnapshot()));
    },
    setCell(reference, input) {
      const { row, column } = coordinatesForReference(reference);
      return setCellInput(row, column, input);
    },
    setCellNumberFormat,
    setCells: setCellInputs
  });
})();
