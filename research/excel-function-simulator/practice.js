(() => {
  "use strict";

  const STORAGE_KEY = "excelFunctionSimulatorProgressV1";
  const PRACTICE_START = "H1";
  const PRACTICE_END = "N25";
  const catalog = window.ExcelFunctionCatalog;
  const exercises = window.ExcelExercises;
  const curriculum = window.ExcelCurriculum || {
    levels: [
      { id: "Beginner", label: "Beginner" },
      { id: "Intermediate", label: "Intermediate" },
      { id: "Advanced", label: "Advanced" }
    ],
    specialTracks: [
      { id: "Mixed", label: "Mixed" },
      { id: "All", label: "All" }
    ]
  };
  const simulator = window.ExcelSimulator;
  const learningPanel = document.querySelector("#learning-panel");
  const tracePanel = document.querySelector("#trace-panel");
  const sidePanel = document.querySelector("#formula-explorer");
  const modeButtons = [...document.querySelectorAll(".mode-button")];

  const initialProgress = loadProgress();
  const state = {
    mode: "playground",
    selectedFunctionId: "sum",
    functionFilter: "",
    exerciseIndex: 0,
    practiceFilter: recommendedDifficulty(initialProgress),
    hintsShown: 0,
    solutionVisible: false,
    feedback: null,
    practiceWorkspaceSnapshot: null,
    progress: initialProgress
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

  function exerciseMatchesFilter(exercise, filter) {
    if (filter === "All") return true;
    if (filter === "Mixed") return exercise.curriculumTrack === "Mixed";
    return exercise.difficulty === filter;
  }

  function pathExercises(filter = state.practiceFilter) {
    return exercises.filter((exercise) => exerciseMatchesFilter(exercise, filter));
  }

  function completedCountFor(filter = "All", progress = state.progress) {
    return exercises.filter((exercise) => (
      exerciseMatchesFilter(exercise, filter) && progress.completed[exercise.id]
    )).length;
  }

  function totalCountFor(filter = "All") {
    return pathExercises(filter).length;
  }

  function recommendedDifficulty(progress) {
    for (const level of ["Beginner", "Intermediate", "Advanced"]) {
      const candidates = exercises.filter((exercise) => exercise.difficulty === level);
      if (candidates.some((exercise) => !progress.completed[exercise.id])) return level;
    }
    return "All";
  }

  function firstIncompleteIndex(filter = state.practiceFilter) {
    const candidate = pathExercises(filter).find((exercise) => !state.progress.completed[exercise.id])
      || pathExercises(filter)[0];
    return candidate ? exercises.findIndex((exercise) => exercise.id === candidate.id) : 0;
  }

  function activePathPosition(exercise = currentExercise()) {
    const path = pathExercises();
    const position = path.findIndex((item) => item.id === exercise.id);
    return { path, position };
  }

  function completionBreakdown(filter = state.practiceFilter) {
    const statuses = { independent: 0, "with-hints": 0, "after-solution": 0 };
    pathExercises(filter).forEach((exercise) => {
      const status = state.progress.completed[exercise.id]?.status;
      if (status && Object.prototype.hasOwnProperty.call(statuses, status)) statuses[status] += 1;
    });
    return statuses;
  }

  function setPracticeFilter(filter, options = {}) {
    const allowed = [
      ...curriculum.levels.map((entry) => entry.id),
      ...curriculum.specialTracks.map((entry) => entry.id)
    ];
    if (!allowed.includes(filter)) return;
    state.practiceFilter = filter;
    if (options.openExercise !== false) {
      openExercise(firstIncompleteIndex(filter), { prepare: options.prepare !== false });
    } else {
      renderPracticePanel();
    }
  }

  function adjacentExerciseIndex(direction) {
    const { path, position } = activePathPosition();
    if (!path.length) return state.exerciseIndex;
    const nextPosition = position < 0 ? 0 : position + direction;
    if (nextPosition >= 0 && nextPosition < path.length) {
      return exercises.findIndex((exercise) => exercise.id === path[nextPosition].id);
    }

    if (direction > 0 && ["Beginner", "Intermediate", "Advanced"].includes(state.practiceFilter)) {
      const order = ["Beginner", "Intermediate", "Advanced"];
      const current = order.indexOf(state.practiceFilter);
      if (current >= 0 && current < order.length - 1) {
        state.practiceFilter = order[current + 1];
        return firstIncompleteIndex(state.practiceFilter);
      }
    }

    const fallback = direction > 0 ? path[0] : path[path.length - 1];
    return exercises.findIndex((exercise) => exercise.id === fallback.id);
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
    const categories = ["Basics", "Logic", "Conditional", "Lookup", "Text", "Date", "Math", "Error Handling", "Statistics", "Financial", "Advanced", "Dynamic Array"];

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

    const pathLesson = element("details", "practice-path-mini-lesson");
    pathLesson.open = true;
    pathLesson.append(element("summary", "", "Practice Path · 100 exercises"));
    pathLesson.append(element(
      "p",
      "learning-copy",
      "Work from single-function confidence toward applied formulas and then multi-step analysis. Nothing is locked, so you can move between levels whenever you want."
    ));
    const pathTerms = element("div", "dynamic-array-terms");
    curriculum.levels.forEach((level) => {
      pathTerms.append(element(
        "div",
        "",
        `${level.label} · ${totalCountFor(level.id)} exercises · ${level.description}`
      ));
    });
    pathTerms.append(element(
      "div",
      "",
      `Mixed · ${totalCountFor("Mixed")} challenges · combine two or more spreadsheet skills.`
    ));
    pathLesson.append(pathTerms);
    learningPanel.append(pathLesson);

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

    const referenceLesson = element("details", "reference-mini-lesson");
    referenceLesson.open = true;
    referenceLesson.append(element("summary", "", "Cell References"));
    referenceLesson.append(element(
      "p",
      "learning-copy",
      "Relative references move when copied or filled. A dollar sign locks the row, the column, or both."
    ));
    const referenceTerms = element("div", "dynamic-array-terms");
    referenceTerms.append(element("div", "", "A1 · relative row, relative column"));
    referenceTerms.append(element("div", "", "$A$1 · locked row, locked column"));
    referenceTerms.append(element("div", "", "A$1 · locked row"));
    referenceTerms.append(element("div", "", "$A1 · locked column"));
    referenceLesson.append(referenceTerms);
    referenceLesson.append(element(
      "p",
      "learning-copy",
      "Example: =D2*$H$2 filled downward changes D2 to D3, D4, … while $H$2 stays fixed. Press F4 while editing a reference to cycle its locking."
    ));
    referenceLesson.append(button("Load Reference Example", "load-reference-example", "learning-button"));
    learningPanel.append(referenceLesson);

    const financeLesson = element("details", "financial-mini-lesson");
    financeLesson.append(element("summary", "", "Financial Cash Flows"));
    financeLesson.append(element(
      "p",
      "learning-copy",
      "Financial functions use a cash-flow sign convention: money received and money paid should have opposite signs."
    ));
    const financeTerms = element("div", "dynamic-array-terms");
    financeTerms.append(element("div", "", "Time 0 · initial investment or loan balance"));
    financeTerms.append(element("div", "", "NPV · first supplied value is one period in the future"));
    financeTerms.append(element("div", "", "IRR · periodic rate for equally spaced cash flows"));
    financeTerms.append(element("div", "", "XNPV / XIRR · use actual dates for irregular cash flows"));
    financeLesson.append(financeTerms);
    learningPanel.append(financeLesson);

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
    if (options.syncFilter && !exerciseMatchesFilter(exercise, state.practiceFilter)) {
      state.practiceFilter = exercise.difficulty;
    }
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
    const { path, position } = activePathPosition(exercise);
    const filterMetadata = [
      ...curriculum.levels,
      ...curriculum.specialTracks
    ].find((entry) => entry.id === state.practiceFilter);
    const breakdown = completionBreakdown();

    learningPanel.replaceChildren();
    const heading = element("div", "learning-heading-row");
    heading.append(element("h2", "", "Practice"));
    heading.append(element(
      "span",
      "practice-position",
      position >= 0 ? `${position + 1} of ${path.length}` : `${state.exerciseIndex + 1} of ${exercises.length}`
    ));
    learningPanel.append(heading);

    const pathWrap = element("section", "practice-path-controls");
    pathWrap.append(element("div", "learning-label", "Practice path"));
    const pathButtons = element("div", "practice-filter-buttons");
    [...curriculum.levels, ...curriculum.specialTracks].forEach((entry) => {
      const filterButton = button(entry.label, "set-practice-filter", "practice-filter-button");
      filterButton.dataset.practiceFilter = entry.id;
      filterButton.classList.toggle("active", entry.id === state.practiceFilter);
      filterButton.setAttribute("aria-pressed", String(entry.id === state.practiceFilter));
      pathButtons.append(filterButton);
    });
    pathWrap.append(pathButtons);
    if (filterMetadata?.description) {
      pathWrap.append(element("p", "practice-path-description", filterMetadata.description));
    }
    learningPanel.append(pathWrap);

    const levelSummary = element("div", "practice-level-summary");
    curriculum.levels.forEach((level) => {
      const completed = completedCountFor(level.id);
      const total = totalCountFor(level.id);
      const item = element("div", "practice-level-summary-item");
      item.append(element("span", "", level.label));
      item.append(element("strong", "", `${completed}/${total}`));
      levelSummary.append(item);
    });
    learningPanel.append(levelSummary);

    learningPanel.append(element(
      "div",
      "practice-progress",
      `${completedCountFor(state.practiceFilter)} / ${totalCountFor(state.practiceFilter)} completed in this path · ${completedCount()} / ${exercises.length} overall`
    ));
    learningPanel.append(element(
      "div",
      "practice-mastery",
      `Independent ${breakdown.independent} · With hints ${breakdown["with-hints"]} · Solution viewed ${breakdown["after-solution"]}`
    ));

    const exerciseMeta = element("div", "practice-exercise-meta");
    exerciseMeta.append(element("span", "practice-difficulty", exercise.difficulty));
    if (exercise.curriculumTrack === "Mixed") {
      exerciseMeta.append(element("span", "practice-track-label", "Mixed challenge"));
    }
    learningPanel.append(exerciseMeta);
    learningPanel.append(element("div", "practice-function", exercise.functionCategory));
    if (exercise.skills?.length) {
      const skillRow = element("div", "practice-skills");
      skillRow.append(element("span", "learning-label", "Skills"));
      exercise.skills.forEach((skill) => skillRow.append(element("span", "practice-skill", skill)));
      learningPanel.append(skillRow);
    }
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

    const navigation = element("div", "practice-navigation");
    navigation.append(button("← Previous", "previous-exercise", "learning-link-button"));
    navigation.append(button(
      state.feedback?.kind === "correct" ? "Continue next incomplete" : "Skip →",
      state.feedback?.kind === "correct" ? "continue-path" : "skip-exercise",
      "learning-link-button"
    ));
    learningPanel.append(navigation);
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
      "#NUM!": "A numeric argument or iterative calculation is outside the supported range or cannot produce a valid result.",
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

    const expectedValue = exercise.validationType === "todayFormula"
      ? window.ExcelFormatting.todaySerial()
      : exercise.expectedValue;
    const valueCorrect = valuesEqual(model.value, expectedValue);
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
        expected: expectedValue,
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
    const rank = { Beginner: 0, Intermediate: 1, Advanced: 2 };
    const candidates = exercises.filter((exercise) => (
      exercise.functionCategory === entry.name
      || exercise.acceptedFunctionSets.some((set) => set.includes(entry.name))
    )).sort((left, right) => (rank[left.difficulty] ?? 9) - (rank[right.difficulty] ?? 9));
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
      const requestedIndex = options.exerciseIndex ?? (
        previousMode !== "practice" ? firstIncompleteIndex(state.practiceFilter) : state.exerciseIndex
      );
      openExercise(requestedIndex, {
        prepare: options.prepare !== false,
        syncFilter: options.exerciseIndex !== undefined
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
    } else if (action === "load-reference-example") {
      applyWorkspace(
        [
          { cell: "H2", value: "Rate" },
          { cell: "I2", value: "0.05", numberFormat: "Percentage" },
          { cell: "H3", value: "Fill the formula downward" }
        ],
        "=D2*$I$2",
        "H4"
      );
    } else if (action === "practice-function") {
      const entry = catalog.find((item) => item.id === state.selectedFunctionId);
      const exercise = firstExerciseForFunction(entry);
      const index = exercises.findIndex((item) => item.id === exercise?.id);
      if (index >= 0) {
        state.practiceFilter = exercise.difficulty;
        setMode("practice", { exerciseIndex: index });
      }
    } else if (action === "set-practice-filter") {
      setPracticeFilter(actionButton.dataset.practiceFilter);
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
    } else if (action === "next-exercise" || action === "skip-exercise") {
      openExercise(adjacentExerciseIndex(1));
    } else if (action === "previous-exercise") {
      openExercise(adjacentExerciseIndex(-1));
    } else if (action === "continue-path") {
      openExercise(firstIncompleteIndex(state.practiceFilter));
    } else if (action === "reset-progress") {
      if (!window.confirm("Reset all saved practice progress?")) return;
      state.progress = emptyProgress();
      state.practiceFilter = "Beginner";
      state.hintsShown = 0;
      state.solutionVisible = false;
      state.feedback = null;
      localStorage.removeItem(STORAGE_KEY);
      openExercise(firstIncompleteIndex("Beginner"));
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
        practiceFilter: state.practiceFilter,
        feedback: state.feedback ? { ...state.feedback } : null,
        progress: JSON.parse(JSON.stringify(state.progress))
      };
    },
    openExercise(id) {
      const index = exercises.findIndex((exercise) => exercise.id === id);
      if (index >= 0) setMode("practice", { exerciseIndex: index });
    },
    setMode,
    setPracticeFilter
  });
})();
