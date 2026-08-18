((global) => {
  "use strict";

  const MILLISECONDS_PER_DAY = 86400000;
  const EXCEL_EPOCH_UTC = Date.UTC(1899, 11, 31);
  const MIN_SUPPORTED_SERIAL = 61;
  const MAX_SUPPORTED_SERIAL = 2958465;
  const NUMBER_FORMATS = Object.freeze({
    GENERAL: "General",
    NUMBER: "Number",
    CURRENCY: "Currency",
    PERCENTAGE: "Percentage",
    DATE: "Date"
  });
  const FORMAT_DEFAULTS = Object.freeze({
    General: Object.freeze({ decimals: null, useThousands: false, currencySymbol: "$" }),
    Number: Object.freeze({ decimals: 2, useThousands: true, currencySymbol: "$" }),
    Currency: Object.freeze({ decimals: 2, useThousands: true, currencySymbol: "$" }),
    Percentage: Object.freeze({ decimals: 2, useThousands: true, currencySymbol: "$" }),
    Date: Object.freeze({ decimals: null, useThousands: false, currencySymbol: "$" })
  });
  let dateProvider = () => new Date();

  function utcCalendarDate(year, month, day) {
    if (![year, month, day].every(Number.isInteger)) return null;
    if (Math.abs(year) > 10000 || Math.abs(month) > 120000 || Math.abs(day) > 1000000) {
      return null;
    }

    const date = new Date(0);
    date.setUTCHours(0, 0, 0, 0);
    date.setUTCFullYear(year, month - 1, day);
    return Number.isNaN(date.getTime()) ? null : date;
  }

  function calendarParts(date) {
    return {
      year: date.getUTCFullYear(),
      month: date.getUTCMonth() + 1,
      day: date.getUTCDate()
    };
  }

  function normalizeCalendar(year, month, day) {
    const date = utcCalendarDate(year, month, day);
    return date ? calendarParts(date) : null;
  }

  function calendarToSerial(year, month, day) {
    const date = utcCalendarDate(year, month, day);
    if (!date) return null;
    const parts = calendarParts(date);
    if (parts.year < 1900 || (parts.year === 1900 && parts.month < 3) || parts.year > 9999) {
      return null;
    }
    return Math.round((date.getTime() - EXCEL_EPOCH_UTC) / MILLISECONDS_PER_DAY) + 1;
  }

  function serialToCalendar(serial) {
    if (typeof serial !== "number" || !Number.isFinite(serial)) return null;
    const wholeSerial = Math.floor(serial);
    if (wholeSerial < MIN_SUPPORTED_SERIAL || wholeSerial > MAX_SUPPORTED_SERIAL) return null;
    const date = new Date(EXCEL_EPOCH_UTC + ((wholeSerial - 1) * MILLISECONDS_PER_DAY));
    return calendarParts(date);
  }

  function pad(value) {
    return String(value).padStart(2, "0");
  }

  function formatDateSerial(serial) {
    const calendar = serialToCalendar(serial);
    if (!calendar) return "#NUM!";
    return `${String(calendar.year).padStart(4, "0")}-${pad(calendar.month)}-${pad(calendar.day)}`;
  }

  function formatGeneral(value) {
    if (typeof value === "boolean") return value ? "TRUE" : "FALSE";
    if (typeof value === "number" && Number.isFinite(value)) {
      if (Object.is(value, -0)) return "0";
      if (Number.isInteger(value)) return String(value);
      return String(Number(value.toPrecision(12)));
    }
    return String(value ?? "");
  }

  function normalizeFormatOptions(numberFormat, options = {}) {
    const defaults = FORMAT_DEFAULTS[numberFormat] || FORMAT_DEFAULTS.General;
    const requestedDecimals = Number.isInteger(options.decimals)
      ? options.decimals
      : defaults.decimals;
    return {
      decimals: requestedDecimals === null
        ? null
        : Math.max(0, Math.min(10, requestedDecimals)),
      useThousands: options.useThousands === undefined
        ? defaults.useThousands
        : Boolean(options.useThousands),
      currencySymbol: typeof options.currencySymbol === "string"
        ? options.currencySymbol
        : defaults.currencySymbol
    };
  }

  function formatFixed(value, options) {
    if (typeof value !== "number" || !Number.isFinite(value)) return formatGeneral(value);
    return new Intl.NumberFormat("en-US", {
      minimumFractionDigits: options.decimals,
      maximumFractionDigits: options.decimals,
      useGrouping: options.useThousands
    }).format(value);
  }

  function formatNumber(value, options = {}) {
    return formatFixed(value, normalizeFormatOptions(NUMBER_FORMATS.NUMBER, options));
  }

  function formatCurrency(value, options = {}) {
    if (typeof value !== "number" || !Number.isFinite(value)) return formatGeneral(value);
    const normalized = normalizeFormatOptions(NUMBER_FORMATS.CURRENCY, options);
    const amount = formatFixed(Math.abs(value), normalized);
    return `${value < 0 ? "-" : ""}${normalized.currencySymbol}${amount}`;
  }

  function formatPercentage(value, options = {}) {
    if (typeof value !== "number" || !Number.isFinite(value)) return formatGeneral(value);
    const normalized = normalizeFormatOptions(NUMBER_FORMATS.PERCENTAGE, options);
    return `${formatFixed(value * 100, normalized)}%`;
  }

  function formatValue(value, numberFormat = NUMBER_FORMATS.GENERAL, options = {}) {
    if (numberFormat === NUMBER_FORMATS.DATE && typeof value === "number") {
      return formatDateSerial(value);
    }
    if (numberFormat === NUMBER_FORMATS.NUMBER) return formatNumber(value, options);
    if (numberFormat === NUMBER_FORMATS.CURRENCY) return formatCurrency(value, options);
    if (numberFormat === NUMBER_FORMATS.PERCENTAGE) return formatPercentage(value, options);
    return formatGeneral(value);
  }

  function formatSummary(numberFormat, options = {}) {
    const normalized = normalizeFormatOptions(numberFormat, options);
    if ([NUMBER_FORMATS.NUMBER, NUMBER_FORMATS.CURRENCY, NUMBER_FORMATS.PERCENTAGE].includes(numberFormat)) {
      return `${numberFormat}, ${normalized.decimals} decimal place${normalized.decimals === 1 ? "" : "s"}`;
    }
    return numberFormat;
  }

  function daysInMonth(year, month) {
    const lastDay = utcCalendarDate(year, month + 1, 0);
    return lastDay ? lastDay.getUTCDate() : null;
  }

  function addMonths(serial, months) {
    if (!Number.isInteger(months)) return null;
    const start = serialToCalendar(serial);
    if (!start) return null;
    const targetFirst = normalizeCalendar(start.year, start.month + months, 1);
    if (!targetFirst) return null;
    const targetDay = Math.min(start.day, daysInMonth(targetFirst.year, targetFirst.month));
    return calendarToSerial(targetFirst.year, targetFirst.month, targetDay);
  }

  function endOfMonth(serial, months) {
    if (!Number.isInteger(months)) return null;
    const start = serialToCalendar(serial);
    if (!start) return null;
    const target = normalizeCalendar(start.year, start.month + months + 1, 0);
    return target ? calendarToSerial(target.year, target.month, target.day) : null;
  }

  function weekday(serial, returnType = 1) {
    const calendar = serialToCalendar(serial);
    if (!calendar || ![1, 2].includes(returnType)) return null;
    const date = utcCalendarDate(calendar.year, calendar.month, calendar.day);
    const sundayBased = date.getUTCDay();
    return returnType === 1 ? sundayBased + 1 : ((sundayBased + 6) % 7) + 1;
  }

  function weekdayName(serial) {
    const calendar = serialToCalendar(serial);
    if (!calendar) return null;
    const names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
    return names[utcCalendarDate(calendar.year, calendar.month, calendar.day).getUTCDay()];
  }

  function todaySerial() {
    const provided = dateProvider();
    if (provided instanceof Date && !Number.isNaN(provided.getTime())) {
      return calendarToSerial(provided.getFullYear(), provided.getMonth() + 1, provided.getDate());
    }
    if (provided && typeof provided === "object") {
      return calendarToSerial(provided.year, provided.month, provided.day);
    }
    return null;
  }

  function setDateProvider(provider) {
    if (typeof provider !== "function") throw new TypeError("Date provider must be a function");
    dateProvider = provider;
  }

  function resetDateProvider() {
    dateProvider = () => new Date();
  }

  const api = {
    DATE_SYSTEM_NOTE: "Excel-compatible 1900 serials are supported from 1900-03-01 (serial 61) onward; the fictitious serial 60 date is intentionally excluded.",
    FORMAT_DEFAULTS,
    MAX_SUPPORTED_SERIAL,
    MIN_SUPPORTED_SERIAL,
    NUMBER_FORMATS,
    addMonths,
    calendarToSerial,
    endOfMonth,
    formatDateSerial,
    formatSummary,
    formatValue,
    normalizeFormatOptions,
    normalizeCalendar,
    resetDateProvider,
    serialToCalendar,
    setDateProvider,
    todaySerial,
    weekday,
    weekdayName
  };

  global.ExcelFormatting = api;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
})(typeof window === "undefined" ? globalThis : window);
