(() => {
  "use strict";

  function initializeMovie(figure) {
    const poster = figure.querySelector("canvas");
    const image = figure.querySelector("img");
    const button = figure.querySelector("button");
    const original = figure.querySelector('a[href$=".gif"]');
    const playLabel = button.getAttribute("aria-label");
    let playing = false;
    let posterRequested = false;

    function loadPoster() {
      if (posterRequested) return;
      posterRequested = true;
      const firstFrame = new Image();
      firstFrame.onload = () => {
        // Drawing an animated image to canvas uses its default (still) frame.
        poster.getContext("2d").drawImage(firstFrame, 0, 0, poster.width, poster.height);
        button.hidden = false;
      };
      firstFrame.src = original.href;
    }

    function stop() {
      if (!playing) return;
      playing = false;
      image.hidden = true;
      image.removeAttribute("src");
      poster.hidden = false;
      button.textContent = "Play GIF";
      button.setAttribute("aria-label", playLabel);
    }

    button.addEventListener("click", () => {
      if (playing) {
        stop();
        return;
      }
      image.src = original.href;
      image.hidden = false;
      poster.hidden = true;
      playing = true;
      button.textContent = "Stop GIF";
      button.setAttribute("aria-label", playLabel.replace(/^Play /, "Stop "));
    });

    if ("IntersectionObserver" in window) {
      new IntersectionObserver(([entry], observer) => {
        if (entry.isIntersecting) {
          loadPoster();
          observer.disconnect();
        }
      }, {rootMargin: "400px"}).observe(figure);
      new IntersectionObserver(([entry]) => {
        if (!entry.isIntersecting) stop();
      }).observe(figure);
    } else {
      loadPoster();
    }
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) stop();
    });
  }

  document.querySelectorAll("[data-original-gif]").forEach(initializeMovie);

  const copy = document.getElementById("archive-copy");
  const source = document.getElementById("archive-code");
  const status = document.getElementById("archive-copy-status");
  copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(source.textContent);
      status.textContent = "Code copied.";
    } catch {
      // Local documents may lack clipboard permission. Leave the source selected.
      const range = document.createRange();
      range.selectNodeContents(source);
      const selectedText = window.getSelection();
      selectedText.removeAllRanges();
      selectedText.addRange(range);
      status.textContent = document.execCommand("copy")
        ? "Code copied."
        : "Source selected. Use your keyboard’s Copy command.";
    }
  });
})();
