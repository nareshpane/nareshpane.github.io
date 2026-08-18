(() => {
  "use strict";

  const STORAGE_KEY = "excelFunctionSimulatorProgressV1";
  const PRACTICE_START = "H1";
  const PRACTICE_END = "N25";
  const catalog = window.ExcelFunctionCatalog;
  const exercises = window.ExcelExercises;
  const simulator = window.ExcelSimulator;
  const learningPanel = document.querySelector("#learning-panel");
  const tracePanel = document.querySelector("#trace-panel");
  const sidePanel = document.querySelector("#formula-explorer");
  const modeButtons = [...document.querySelectorAll(".mode-button")];

  const state = {
    mode: "playground",
    selectedFunctionId: "sum",
    functionFilter: "",
    exerciseIndex: 0,
    hintsShown: 0,
    solutionVisible: false,
    feedback: null,
    practiceWorkspaceSnapshot: null,
    progress: loadProgress()
  };

  function emptyProgress() {
    return { version: 1, completed: {}, hints: {}, solutions: {} };
  }

  function loadProgress() {
    try {
      const saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
      if (saved?.version !== 1) return emptyProgress();
      return {
        version: 1,
        completed: saved.completed || {},
        hints: saved.hints || {},
        solutions: saved.solutions || {}
      };
    } catch (error) {
      return emptyProgress();
    }
  }

  function saveProgress() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state.progress));
  }

  function element(tagName, className, text) {
    const node = document.createElement(tagName);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function button(text, action, className = "learning-button") {
    const node = element("button", className, text);
    node.type = "button";
    node.dataset.action = action;
    return node;
  }

  function formattedValue(value, numberFormat = "General") {
    if (numberFormat !== "General") return simulator.formatValue(value, numberFormat);
    if (typeof value === "number") return value.toLocaleString("en-US", {
      maximumSignificantDigits: 12
    });
    if (typeof value === "boolean") return value ? "TRUE" : "FALSE";
    return String(value ?? "");
  }

  function markMode(mode) {
    modeButtons.forEach((modeButton) => {
      const active = modeButton.dataset.mode === mode;
      modeButton.classList.toggle("active", active);
      modeButton.setAttribute("aria-pressed", String(active));
    });
  }

  function clearTargetMarker() {
    document.querySelectorAll(".sheet-cell.practice-target").forEach((cell) => {
      cell.classList.remove("practice-target");
    });
  }

  function markTarget(reference) {
    clearTargetMarker();
    simulator.getCellElement(reference)?.classList.add("practice-target");
  }

  function applyWorkspace(setup = [], formula = null, targetCell = "H4") {
    simulator.clearRange(PRACTICE_START, PRACTICE_END);
    const updates = [{ cell: "H2", value: "Example workspace" }, ...setup];
    if (formula !== null) updates.push({ cell: targetCell, value: formula });
    simulator.setCells(updates);
    simulator.selectCell(targetCell);
  }

  function renderFunctionCatalog(container) {
    const query = state.functionFilter.trim().toLowerCase();
    const filtered = catalog.filter((entry) => (
      !query
      || entry.name.toLowerCase().includes(query)
      || entry.shortDescription.toLowerCase().includes(query)
    ));
    const categories = ["Basics", "Logic", "Conditional", "Lookup", "Text", "Date", "Math", "Error Handling", "Dynamic Array"];

    categories.forEach((category) => {
      const entries = filtered.filter((entry) => entry.category === category);
      if (!entries.length) return;
      const group = element("section", "function-catalog-group");
      group.append(element("h3", "function-category-title", category));
      const list = element("div", "function-catalog-list");
      entries.forEach((entry) => {
        const item = button(entry.name, "select-function", "function-catalog-button");
        item.dataset.functionId = entry.id;
        item.classList.toggle("selected", entry.id === state.selectedFunctionId);
        list.append(item);
      });
      group.append(list);
      container.append(group);
    });

    if (!filtered.length) {
      container.append(element("p", "learning-muted", "No supported functions match this filter."));
    }
  }

  function renderLesson(container, entry) {
    const lesson = element("section", "function-lesson");
    const heading = element("div", "learning-heading-row");
    heading.append(element("h3", "learning-title", entry.name));
    heading.append(element("span", "difficulty-label", entry.difficulty));
    lesson.append(heading);

    lesson.append(element("div", "learning-label", "What it does"));
    lesson.append(element("p", "learning-copy", entry.shortDescription));
    if (entry.lessonNote) {
      lesson.append(element("p", "learning-note", entry.lessonNote));
    }
    lesson.append(element("div", "learning-label", "Syntax"));
    lesson.append(element("code", "learning-formula", `=${entry.syntax}`));
    lesson.append(element("div", "learning-label", "Example"));
    lesson.append(element("code", "learning-formula", entry.exampleFormula));
    lesson.append(element("div", "learning-label", "Result"));
    if (Array.isArray(entry.exampleResult)) {
      const matrix = element("div", "learning-array-preview");
      matrix.style.setProperty("--learning-array-columns", String(entry.exampleResult[0]?.length || 1));
      entry.exampleResult.forEach((row) => row.forEach((value) => {
        matrix.append(element("span", "learning-array-cell", formattedValue(value)));
      }));
      lesson.append(matrix);
      lesson.append(element(
        "div",
        "learning-array-shape",
        `${entry.exampleShape?.[0] || entry.exampleResult.length} × ${entry.exampleShape?.[1] || entry.exampleResult[0]?.length || 1} spill from ${entry.exampleTargetCell || "H4"}`
      ));
    } else {
      lesson.append(element(
        "div",
        "learning-result",
        formattedValue(entry.exampleResult, entry.exampleNumberFormat)
      ));
    }
    if (entry.exampleNumberFormat === "Date" && typeof entry.exampleResult === "number") {
      lesson.append(element("div", "learning-label", "Underlying serial"));
      lesson.append(element("div", "learning-serial", String(entry.exampleResult)));
    }
    lesson.append(element("div", "learning-label", "Arguments"));

    const argumentsList = element("dl", "argument-list");
    entry.arguments.forEach((argument) => {
      argumentsList.append(element("dt", "argument-name", argument.name));
      argumentsList.append(element("dd", "argument-description", argument.description));
    });
    lesson.append(argumentsList);

    const actions = element("div", "learning-actions");
    actions.append(button("Load Example", "load-example", "learning-button primary"));
    if (firstExerciseForFunction(entry)) {
      actions.append(button("Practice This Function", "practice-function", "learning-button"));
    }
    lesson.append(actions);
    container.append(lesson);
  }

  function renderLearnPanel() {
    learningPanel.replaceChildren();
    learningPanel.append(element("h2", "", "Learn"));
    const searchLabel = element("label", "function-search-wrap");
    searchLabel.append(element("span", "visually-hidden", "Filter supported functions"));
    const search = element("input", "function-search");
    search.id = "function-search";
    search.type = "search";
    search.placeholder = "Filter functions";
    search.value = state.functionFilter;
    searchLabel.append(search);
    learningPanel.append(searchLabel);

    const catalogWrap = element("div", "function-catalog");
    renderFunctionCatalog(catalogWrap);
    learningPanel.append(catalogWrap);

    const formattingLesson = element("details", "formatting-mini-lesson");
    formattingLesson.append(element("summary", "", "Number Formatting"));
    formattingLesson.append(element(
      "p",
      "learning-copy",
      "Formatting changes appearance; ROUND changes the stored result."
    ));
    const comparison = element("div", "formatting-comparison");
    comparison.append(element("div", "", "10 / 3 underlying: 3.333333…"));
    comparison.append(element("div", "", "Number, 2 decimals: 3.33"));
    comparison.append(element("div", "", "ROUND(10/3,2) underlying: 3.33"));
    formattingLesson.append(comparison);
    learningPanel.append(formattingLesson);

    const spillLesson = element("details", "dynamic-array-mini-lesson");
    spillLesson.open = true;
    spillLesson.append(element("summary", "", "One formula can return many cells"));
    spillLesson.append(element(
      "p",
      "learning-copy",
      "The anchor cell stores the formula. Its result fills a spill range, and the remaining spill cells stay controlled by the anchor."
    ));
    const terms = element("div", "dynamic-array-terms");
    terms.append(element("div", "", "H15 · anchor cell · =SEQUENCE(5)"));
    terms.append(element("div", "", "H15:H19 · spill range"));
    terms.append(element("div", "", "H16:H19 · spill children"));
    spillLesson.append(terms);
    spillLesson.append(element(
      "p",
      "dynamic-array-warning",
      "If any destination is occupied, the anchor returns #SPILL! instead of overwriting the blocking cell."
    ));
    spillLesson.append(button("Load Blocked SEQUENCE", "load-blocked-spill", "learning-button"));
    learningPanel.append(spillLesson);

    const selected = catalog.find((entry) => entry.id === state.selectedFunctionId) || catalog[0];
    renderLesson(learningPanel, selected);
  }

  function currentExercise() {
    return exercises[state.exerciseIndex];
  }

  function completedCount() {
    return Object.keys(state.progress.completed).filter((id) => (
      exercises.some((exercise) => exercise.id === id)
    )).length;
  }

  function prepareExercise(exercise) {
    simulator.clearRange(PRACTICE_START, PRACTICE_END);
    if (exercise.setup?.length) simulator.setCells(exercise.setup);
    markTarget(exercise.targetCell);
    simulator.selectCell(exercise.targetCell);
  }

  function snapshotPracticeWorkspace() {
    return window.FormulaEngine.expandRange(PRACTICE_START, PRACTICE_END).map((cell) => {
      const format = simulator.getCellFormat(cell);
      return {
        cell,
        value: simulator.getCell(cell)?.input || "",
        numberFormat: format?.type,
        formatOptions: format
      };
    });
  }

  function restorePracticeWorkspace() {
    if (!state.practiceWorkspaceSnapshot) return;
    const updates = state.practiceWorkspaceSnapshot.filter((entry) => (
      entry.value !== "" || entry.numberFormat
    ));
    simulator.clearRange(PRACTICE_START, PRACTICE_END);
    if (updates.length) simulator.setCells(updates);
    state.practiceWorkspaceSnapshot = null;
  }

  function openExercise(index, options = {}) {
    state.exerciseIndex = Math.max(0, Math.min(exercises.length - 1, index));
    const exercise = currentExercise();
    state.hintsShown = state.progress.hints[exercise.id] || 0;
    state.solutionVisible = Boolean(state.progress.solutions[exercise.id]);
    state.feedback = null;
    if (options.prepare !== false) prepareExercise(exercise);
    renderPracticePanel();
  }

  function renderFeedback(container, feedback) {
    const box = element("div", `practice-feedback ${feedback.kind}`);
    box.dataset.feedbackKind = feedback.kind;
    box.append(element("strong", "practice-feedback-title", feedback.title));
    box.append(element("div", "practice-feedback-copy", feedback.message));
    if (feedback.expected !== undefined) {
      box.append(element(
        "div",
        "practice-feedback-detail",
        `Expected result: ${formattedValue(feedback.expected, feedback.expectedNumberFormat)}`
      ));
    }
    container.append(box);
  }

  function renderPracticePanel() {
    const exercise = currentExercise();
    learningPanel.replaceChildren();
    const heading = element("div", "learning-heading-row");
    heading.append(element("h2", "", "Practice"));
    heading.append(element(
      "span",
      "practice-position",
      `${state.exerciseIndex + 1} of ${exercises.length}`
    ));
    learningPanel.append(heading);
    learningPanel.append(element(
      "div",
      "practice-progress",
      `${completedCount()} / ${exercises.length} completed`
    ));
    learningPanel.append(element("div", "practice-function", exercise.functionCategory));
    learningPanel.append(element("h3", "learning-title", exercise.title));
    learningPanel.append(element("p", "learning-copy", exercise.prompt));
    learningPanel.append(element("div", "learning-label", "Target cell"));
    learningPanel.append(element("code", "practice-target-label", exercise.targetCell));
    learningPanel.append(element(
      "p",
      "practice-instruction",
      exercise.validationType === "format"
        ? `Use the number-format toolbar on ${exercise.targetCell} without changing its value.`
        : (exercise.validationType === "arrayFormula"
          ? `Enter one formula in ${exercise.targetCell}; its result should spill into the blank cells beside and below it.`
          : `Enter your formula directly into ${exercise.targetCell}.`)
    ));

    if (state.hintsShown) {
      const hints = element("div", "practice-hints");
      exercise.hints.slice(0, state.hintsShown).forEach((hint, index) => {
        const hintRow = element("div", "practice-hint");
        hintRow.append(element("strong", "", `Hint ${index + 1}`));
        hintRow.append(element("span", "", hint));
        hints.append(hintRow);
      });
      learningPanel.append(hints);
    }

    if (state.solutionVisible) {
      const solution = element("div", "practice-solution");
      solution.append(element("strong", "", "Solution viewed"));
      solution.append(element(
        "code",
        "learning-formula",
        exercise.validationType === "format" ? exercise.solutionFormat : exercise.solutionFormula
      ));
      learningPanel.append(solution);
    }

    if (state.feedback) renderFeedback(learningPanel, state.feedback);

    const actions = element("div", "learning-actions practice-actions");
    actions.append(button("Check Answer", "check-answer", "learning-button primary"));
    if (state.hintsShown < exercise.hints.length) {
      actions.append(button("Show Hint", "show-hint", "learning-button"));
    }
    actions.append(button("Reset Exercise", "reset-exercise", "learning-button"));
    actions.append(button("Show Solution", "show-solution", "learning-link-button"));
    if (state.feedback?.kind === "correct") {
      actions.append(button("Next Exercise", "next-exercise", "learning-button primary"));
    }
    learningPanel.append(actions);
    learningPanel.append(button("Reset Progress", "reset-progress", "reset-progress-button"));
  }

  function collectFunctions(ast, names = new Set()) {
    if (!ast) return names;
    if (ast.type === "function") {
      names.add(ast.name);
      ast.arguments.forEach((argument) => collectFunctions(argument, names));
    } else if (ast.type === "binary" || ast.type === "comparison") {
      collectFunctions(ast.left, names);
      collectFunctions(ast.right, names);
    } else if (ast.type === "unary") {
      collectFunctions(ast.operand, names);
    } else if (ast.type === "postfix") {
      collectFunctions(ast.operand, names);
    }
    return names;
  }

  function meetsFunctionRequirement(exercise, ast) {
    const usedFunctions = collectFunctions(ast);
    return exercise.acceptedFunctionSets.some((functionSet) => (
      functionSet.every((name) => usedFunctions.has(name))
    ));
  }

  function requiredFunctionText(exercise) {
    return exercise.acceptedFunctionSets
      .map((functionSet) => functionSet.join(" and "))
      .join(" or ");
  }

  function valuesEqual(actual, expected) {
    if (typeof actual === "string" && typeof expected === "string") {
      return actual === expected;
    }
    return window.LookupEngine.valuesEqual(actual, expected);
  }

  function errorMessage(value) {
    const messages = {
      "#N/A": "The lookup value was not found. Inspect the lookup range.",
      "#REF!": "The formula points outside a selected range.",
      "#VALUE!": "One or more arguments have incompatible values or dimensions.",
      "#NUM!": "A numeric date value or option is outside the supported range.",
      "#CALC!": "The array calculation returned no rows and no fallback was supplied.",
      "#SPILL!": "The result cannot spill because a destination is occupied or outside the worksheet.",
      "#DIV/0!": "The formula attempted to divide by zero or average no values.",
      "#NAME?": "The formula contains an unrecognized function or name.",
      "#ERROR!": "The formula could not be parsed or evaluated."
    };
    return messages[value] || "The formula returned an error.";
  }

  function checkAnswer() {
    const exercise = currentExercise();
    const model = simulator.getCell(exercise.targetCell);

    if (exercise.validationType === "format") {
      const valueCorrect = Boolean(model) && valuesEqual(model.value, exercise.expectedValue);
      const formatCorrect = model?.numberFormatOverride === exercise.expectedNumberFormat;
      const decimalsCorrect = model?.formatOptions?.decimals === exercise.expectedDecimals;
      if (!valueCorrect) {
        state.feedback = {
          kind: "incorrect",
          title: "Underlying value changed",
          message: `Keep the stored value at ${formattedValue(exercise.expectedValue)} while applying formatting.`
        };
      } else if (!formatCorrect || !decimalsCorrect) {
        state.feedback = {
          kind: "wrong-format",
          title: "Adjust the number format",
          message: `Apply ${exercise.expectedNumberFormat} with ${exercise.expectedDecimals} decimal places.`
        };
      } else {
        const solutionViewed = Boolean(state.progress.solutions[exercise.id]);
        const status = solutionViewed
          ? "after-solution"
          : (state.hintsShown ? "with-hints" : "independent");
        state.progress.completed[exercise.id] = { status, completedAt: Date.now() };
        saveProgress();
        state.feedback = {
          kind: "correct",
          title: "Correct",
          message: `Displayed result: ${formattedValue(model.value, model.numberFormat)}. The underlying value is unchanged.`
        };
      }
      renderPracticePanel();
      return state.feedback;
    }

    if (!model || model.type !== "formula") {
      state.feedback = {
        kind: "incorrect",
        title: "Enter a formula",
        message: `Use a formula in ${exercise.targetCell}, not a typed result.`
      };
      renderPracticePanel();
      return state.feedback;
    }

    if (typeof model.value === "string" && model.value.startsWith("#")) {
      state.feedback = {
        kind: "error",
        title: model.value,
        message: errorMessage(model.value)
      };
      renderPracticePanel();
      return state.feedback;
    }

    if (exercise.validationType === "arrayFormula") {
      const spill = simulator.getSpill(exercise.targetCell);
      const functionCorrect = meetsFunctionRequirement(exercise, model.ast);
      const shapeCorrect = Boolean(spill)
        && spill.rows === exercise.expectedShape[0]
        && spill.columns === exercise.expectedShape[1];
      let valuesCorrect = shapeCorrect;
      let mismatch = null;
      if (shapeCorrect) {
        for (let row = 0; row < spill.rows && valuesCorrect; row += 1) {
          for (let column = 0; column < spill.columns; column += 1) {
            if (!valuesEqual(spill.values[row][column], exercise.expectedValues[row][column])) {
              valuesCorrect = false;
              mismatch = { row, column };
              break;
            }
          }
        }
      }

      if (valuesCorrect && !functionCorrect) {
        const required = requiredFunctionText(exercise);
        state.feedback = {
          kind: "wrong-function",
          title: "Correct array, different method",
          message: `This exercise is practicing ${required}. Use that function in the anchor formula.`
        };
      } else if (!shapeCorrect) {
        state.feedback = {
          kind: "incorrect",
          title: "Check the spill shape",
          message: spill
            ? `Expected ${exercise.expectedShape[0]} × ${exercise.expectedShape[1]}, but received ${spill.rows} × ${spill.columns}.`
            : "The anchor did not produce a spilled array."
        };
      } else if (!valuesCorrect) {
        const reference = window.FormulaEngine.expandRange(
          exercise.targetCell,
          exercise.targetCell
        )[0];
        state.feedback = {
          kind: "incorrect",
          title: "Check the array values",
          message: `The first mismatch is at array row ${mismatch.row + 1}, column ${mismatch.column + 1} from ${reference}.`
        };
      } else {
        const solutionViewed = Boolean(state.progress.solutions[exercise.id]);
        const status = solutionViewed
          ? "after-solution"
          : (state.hintsShown ? "with-hints" : "independent");
        state.progress.completed[exercise.id] = { status, completedAt: Date.now() };
        saveProgress();
        state.feedback = {
          kind: "correct",
          title: "Correct",
          message: `The anchor produced the required ${spill.rows} × ${spill.columns} array in ${spill.range}.`
        };
      }
      renderPracticePanel();
      return state.feedback;
    }

    const valueCorrect = valuesEqual(model.value, exercise.expectedValue);
    const functionCorrect = meetsFunctionRequirement(exercise, model.ast);
    const formatCorrect = !exercise.expectedNumberFormat
      || model.numberFormat === exercise.expectedNumberFormat;

    if (valueCorrect && !functionCorrect) {
      const required = requiredFunctionText(exercise);
      state.feedback = {
        kind: "wrong-function",
        title: "Correct result, different method",
        message: `This exercise is practicing ${required}. Try solving it with ${required}.`
      };
    } else if (!valueCorrect) {
      state.feedback = {
        kind: "incorrect",
        title: "Not quite",
        message: `Your formula returned ${formattedValue(model.value, model.numberFormat)}. Inspect the referenced cells and try again.`,
        expected: exercise.expectedValue,
        expectedNumberFormat: exercise.expectedNumberFormat
      };
    } else if (!formatCorrect) {
      state.feedback = {
        kind: "wrong-format",
        title: "Correct serial, wrong display format",
        message: "The underlying value is correct, but this exercise expects a Date-formatted result."
      };
    } else {
      const solutionViewed = Boolean(state.progress.solutions[exercise.id]);
      const status = solutionViewed
        ? "after-solution"
        : (state.hintsShown ? "with-hints" : "independent");
      state.progress.completed[exercise.id] = { status, completedAt: Date.now() };
      saveProgress();
      state.feedback = {
        kind: "correct",
        title: "Correct",
        message: `Result: ${formattedValue(model.value, model.numberFormat)}. The formula used the required spreadsheet function.`
      };
    }

    renderPracticePanel();
    return state.feedback;
  }

  function firstExerciseForFunction(entry) {
    const candidates = exercises.filter((exercise) => (
      exercise.functionCategory === entry.name
      || exercise.acceptedFunctionSets.some((set) => set.includes(entry.name))
    ));
    return candidates.find((exercise) => !state.progress.completed[exercise.id]) || candidates[0];
  }

  function setMode(mode, options = {}) {
    if (!["playground", "learn", "practice"].includes(mode)) return;
    const previousMode = state.mode;
    if (mode === "practice" && previousMode !== "practice") {
      state.practiceWorkspaceSnapshot = snapshotPracticeWorkspace();
    } else if (mode !== "practice" && previousMode === "practice") {
      restorePracticeWorkspace();
    }
    state.mode = mode;
    markMode(mode);
    sidePanel.dataset.mode = mode;

    if (mode === "playground") {
      learningPanel.hidden = true;
      tracePanel.hidden = false;
      clearTargetMarker();
    } else if (mode === "learn") {
      learningPanel.hidden = false;
      tracePanel.hidden = true;
      clearTargetMarker();
      renderLearnPanel();
    } else {
      learningPanel.hidden = false;
      tracePanel.hidden = false;
      openExercise(options.exerciseIndex ?? state.exerciseIndex, {
        prepare: options.prepare !== false
      });
    }
  }

  modeButtons.forEach((modeButton) => {
    modeButton.addEventListener("click", () => setMode(modeButton.dataset.mode));
  });

  learningPanel.addEventListener("input", (event) => {
    if (event.target.id !== "function-search") return;
    state.functionFilter = event.target.value;
    renderLearnPanel();
    const search = document.querySelector("#function-search");
    search.focus();
    search.setSelectionRange(search.value.length, search.value.length);
  });

  learningPanel.addEventListener("click", (event) => {
    const actionButton = event.target.closest("[data-action]");
    if (!actionButton) return;
    const action = actionButton.dataset.action;

    if (action === "select-function") {
      state.selectedFunctionId = actionButton.dataset.functionId;
      renderLearnPanel();
    } else if (action === "load-example") {
      const entry = catalog.find((item) => item.id === state.selectedFunctionId);
      applyWorkspace(entry.exampleSetup || [], entry.exampleFormula, entry.exampleTargetCell || "H4");
    } else if (action === "load-blocked-spill") {
      applyWorkspace([{ cell: "H17", value: "Blocked" }], "=SEQUENCE(5)", "H15");
    } else if (action === "practice-function") {
      const entry = catalog.find((item) => item.id === state.selectedFunctionId);
      const exercise = firstExerciseForFunction(entry);
      const index = exercises.findIndex((item) => item.id === exercise?.id);
      if (index >= 0) setMode("practice", { exerciseIndex: index });
    } else if (action === "check-answer") {
      checkAnswer();
    } else if (action === "show-hint") {
      const exercise = currentExercise();
      state.hintsShown = Math.min(exercise.hints.length, state.hintsShown + 1);
      state.progress.hints[exercise.id] = state.hintsShown;
      saveProgress();
      renderPracticePanel();
    } else if (action === "show-solution") {
      const exercise = currentExercise();
      state.solutionVisible = true;
      state.progress.solutions[exercise.id] = true;
      saveProgress();
      renderPracticePanel();
    } else if (action === "reset-exercise") {
      state.feedback = null;
      prepareExercise(currentExercise());
      renderPracticePanel();
    } else if (action === "next-exercise") {
      openExercise((state.exerciseIndex + 1) % exercises.length);
    } else if (action === "reset-progress") {
      if (!window.confirm("Reset all saved practice progress?")) return;
      state.progress = emptyProgress();
      state.hintsShown = 0;
      state.solutionVisible = false;
      state.feedback = null;
      localStorage.removeItem(STORAGE_KEY);
      renderPracticePanel();
    }
  });

  window.ExcelLearningApp = Object.freeze({
    checkAnswer,
    getState() {
      return {
        mode: state.mode,
        selectedFunctionId: state.selectedFunctionId,
        exerciseId: currentExercise().id,
        exerciseIndex: state.exerciseIndex,
        feedback: state.feedback ? { ...state.feedback } : null,
        progress: JSON.parse(JSON.stringify(state.progress))
      };
    },
    openExercise(id) {
      const index = exercises.findIndex((exercise) => exercise.id === id);
      if (index >= 0) setMode("practice", { exerciseIndex: index });
    },
    setMode
  });
})();
