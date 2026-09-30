// Keyboard access to the Material for MkDocs navigation drawer (issue #283).
//
// At drawer widths the header menu button becomes a focusable button with
// aria-expanded. Enter or Space toggles the drawer. A closed drawer is inert,
// so Tab does not reach links that are off-screen. In an open drawer only the
// panel on show takes focus: the other nested panels are inert. Opening the
// drawer or a section moves focus to the first link of the panel on show.
// Escape closes the drawer and returns focus to the menu button. The header
// search button is focusable too, and Escape from search returns focus to it.
// Closed search is inert: at drawer-free widths only the results area, since
// the header input is on show there and focusing it opens search; below that
// the whole search panel, which is off-screen. Search is usable again before
// Material focuses the input (button, "/", "s" or "f").
// Toggles go through Material's own checkboxes so its state stays consistent;
// Material itself clicks a focused label on Enter. With the sidebar visible
// (desktop) nothing changes.
(function () {
  "use strict";

  // Material shows the primary sidebar as a drawer below 76.25em.
  var drawerWidth = window.matchMedia("(max-width: 76.234375em)");
  // Material shows the search input in the header from 60em.
  var searchWidth = window.matchMedia("(max-width: 59.984375em)");
  var focusable = 'a[href], [tabindex]:not([tabindex="-1"]), input, button';
  var INERT = "data-drawer-inert";
  var TABINDEX = "data-drawer-tabindex";

  function byId(id) {
    return document.getElementById(id);
  }

  function sidebar() {
    return document.querySelector(".md-sidebar--primary");
  }

  function headerButton(target) {
    return document.querySelector(
      'label.md-header__button[for="' + target + '"]',
    );
  }

  function child(el, selector) {
    return Array.prototype.find.call(el.children, function (c) {
      return c.matches(selector);
    });
  }

  function onScreen(el) {
    var r = el.getBoundingClientRect();
    return r.width > 0 && r.height > 0 && r.right > 0 && r.bottom > 0;
  }

  function makeButton(label, name) {
    if (!label) return;
    label.setAttribute("role", "button");
    label.setAttribute("tabindex", "0");
    if (!label.hasAttribute("aria-label")) label.setAttribute("aria-label", name);
  }

  function markInert(el) {
    if (el.inert) return;
    el.inert = true;
    el.setAttribute(INERT, "");
  }

  function markFocusable(el) {
    if (!el || el.getAttribute("tabindex") === "0") return;
    el.setAttribute(TABINDEX, el.getAttribute("tabindex") || "");
    el.setAttribute("tabindex", "0");
  }

  function clearMarks() {
    document.querySelectorAll("[" + INERT + "]").forEach(function (el) {
      el.inert = false;
      el.removeAttribute(INERT);
    });
    document.querySelectorAll("[" + TABINDEX + "]").forEach(function (el) {
      var old = el.getAttribute(TABINDEX);
      if (old) el.setAttribute("tabindex", old);
      else el.removeAttribute("tabindex");
      el.removeAttribute(TABINDEX);
    });
  }

  // The panels from the root to the one on show: follow checked sections.
  function panelPath() {
    var nav = sidebar();
    var panel = nav && nav.querySelector("nav.md-nav--primary");
    var path = [];
    while (panel) {
      path.push(panel);
      var list = child(panel, "ul");
      var open =
        list &&
        Array.prototype.find.call(list.children, function (li) {
          var toggle = child(li, "input.md-nav__toggle");
          return toggle && toggle.checked && child(li, "nav");
        });
      panel = open ? child(open, "nav") : null;
    }
    return path;
  }

  // Make every panel that is not on show inert; keep the one on show usable.
  function applyPanels() {
    clearMarks();
    var toggle = byId("__drawer");
    if (!drawerWidth.matches || !toggle || !toggle.checked) return;
    var path = panelPath();
    var shown = path[path.length - 1];
    if (!shown) return;
    path.slice(0, -1).forEach(function (panel, level) {
      var next = path[level + 1];
      Array.prototype.forEach.call(panel.children, function (c) {
        if (c.tagName !== "UL") return markInert(c);
        Array.prototype.forEach.call(c.children, function (li) {
          if (!li.contains(next)) return markInert(li);
          Array.prototype.forEach.call(li.children, function (part) {
            if (part !== next) markInert(part);
          });
        });
      });
    });
    if (path.length > 1) markFocusable(child(shown, "label.md-nav__title"));
    var list = child(shown, "ul");
    Array.prototype.forEach.call(list ? list.children : [], function (li) {
      var nested = child(li, "nav");
      if (!nested || !child(li, "input.md-nav__toggle")) return;
      markInert(nested);
      markFocusable(child(li, "label.md-nav__link"));
    });
  }

  function firstLink() {
    var shown = panelPath().pop();
    var list = shown && child(shown, "ul");
    var links = list ? list.querySelectorAll(focusable) : [];
    var first = Array.prototype.find.call(links, function (el) {
      return el.getClientRects().length > 0 && !el.closest("[inert]");
    });
    return first || (shown && child(shown, "label.md-nav__title"));
  }

  // The drawer stays hidden until its slide-in starts, and a hidden element
  // cannot take focus, so retry for a short time.
  function focusSoon(pick, tries) {
    window.setTimeout(function () {
      var el = pick();
      if (el) el.focus({ preventScroll: true });
      if (el && document.activeElement !== el && (tries || 0) < 10) {
        focusSoon(pick, (tries || 0) + 1);
      }
    }, 50);
  }

  // A closed search panel is inert; see the header comment for the widths.
  function syncSearch(open) {
    var search = byId("__search");
    var inner = document.querySelector(".md-search__inner");
    var output = document.querySelector(".md-search__output");
    var closed = !!search && !search.checked && !open;
    if (inner) inner.inert = closed && searchWidth.matches;
    if (output) output.inert = closed;
  }

  function onSearchChange(event) {
    if (event.target === byId("__search")) syncSearch();
  }

  // Material focuses the input during a click on the search button, before
  // the checkbox changes, so make search usable first and re-check after.
  function onSearchPress(event) {
    var search = byId("__search");
    var target = event.target;
    if (!search || search.checked || !(target instanceof Element)) return;
    if (!target.closest('label[for="__search"]')) return;
    syncSearch(true);
    window.setTimeout(syncSearch, 0);
  }

  // Sync inert, aria-expanded and panels with the checkboxes and the width.
  function sync() {
    var toggle = byId("__drawer");
    var nav = sidebar();
    var button = headerButton("__drawer");
    if (!toggle) return;
    if (button) button.setAttribute("aria-expanded", String(toggle.checked));
    if (nav) nav.inert = drawerWidth.matches && !toggle.checked;
    applyPanels();
    syncSearch();
  }

  function onChange(event) {
    var target = event.target;
    var toggle = byId("__drawer");
    var nav = sidebar();
    if (!toggle || !nav) return;
    var section =
      target.classList.contains("md-nav__toggle") && nav.contains(target);
    if (target !== toggle && !section) return;
    sync();
    if (!drawerWidth.matches) return;
    if (target === toggle && !toggle.checked) {
      if (nav.contains(document.activeElement)) headerButton("__drawer")?.focus();
    } else if (!toggle.checked) {
      return;
    } else if (section && !target.checked) {
      focusSoon(function () {
        return nav.querySelector('label.md-nav__link[for="' + target.id + '"]');
      });
    } else {
      focusSoon(firstLink);
    }
  }

  function focusQuery() {
    window.setTimeout(function () {
      var query = document.querySelector(".md-search__input");
      if (byId("__search").checked && query) query.focus();
    }, 0);
  }

  function onKeydown(event) {
    var toggle = byId("__drawer");
    var search = byId("__search");
    if (!toggle || event.metaKey || event.ctrlKey || event.altKey) return;
    var target = event.target;
    var isSearch = search && target === headerButton("__search");
    if (search && !search.checked && /^[/sf]$/.test(event.key)) {
      // Material's search shortcuts focus the input: allow it, then re-check.
      syncSearch(true);
      window.setTimeout(syncSearch, 0);
    }
    if (event.key === "Enter" || event.key === " ") {
      if (target !== headerButton("__drawer") && !isSearch) return;
      // Material clicks a focused label on Enter; click only for Space.
      if (event.key === " ") {
        event.preventDefault();
        (isSearch ? search : toggle).click();
      }
      if (isSearch) focusQuery();
      return;
    }
    if (event.key !== "Escape") return;
    if (search && search.checked) {
      // Material closes search on this key; restore focus after it runs.
      window.setTimeout(function () {
        var button = headerButton("__search");
        if (!search.checked && button && onScreen(button)) button.focus();
      }, 0);
    } else if (toggle.checked && drawerWidth.matches) {
      event.preventDefault();
      toggle.click();
      headerButton("__drawer")?.focus();
    }
  }

  // Header controls survive instant navigation; the sidebar is replaced.
  function init() {
    makeButton(headerButton("__drawer"), "Navigation menu");
    makeButton(headerButton("__search"), "Search");
    sync();
  }

  if (!window.__typevetDrawerKeyboard) {
    window.__typevetDrawerKeyboard = true;
    window.addEventListener("keydown", onKeydown, true);
    document.addEventListener("change", onChange);
    // Capture runs before Material's own listener focuses the input.
    document.addEventListener("change", onSearchChange, true);
    window.addEventListener("pointerdown", onSearchPress, true);
    window.addEventListener("click", onSearchPress, true);
    drawerWidth.addEventListener("change", sync);
    searchWidth.addEventListener("change", sync);
  }

  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(init);
  } else {
    init();
  }
})();
