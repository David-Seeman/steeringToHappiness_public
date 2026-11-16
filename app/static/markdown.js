"use strict";
// A small, dependency-free Markdown renderer for chat replies.
// Supports: # headings, **bold**, *italic*, ~~strike~~, `code`, ``` fenced code ```, > quotes, - / 1. lists (nested),
// tables, --- rules and [links](https://...). All text is HTML-escaped first, so model output can't inject markup.
// It is re-run on the whole reply for every streamed chunk, so half-finished syntax just shows as plain text until it closes.
(function (root) {
  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  function inline(text) {
    const codes = [];
    let s = esc(text).replace(/`([^`\n]+)`/g, (m, c) => { codes.push(`<code>${c}</code>`); return `\u0001${codes.length - 1}\u0001`; });
    s = s.replace(/\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
    s = s.replace(/\*\*([^\n]+?)\*\*/g, "<strong>$1</strong>").replace(/__([^\n]+?)__/g, "<strong>$1</strong>");
    s = s.replace(/(^|[^\w*])\*(?!\s)([^*\n]+?)\*(?!\w)/g, "$1<em>$2</em>").replace(/(^|[^\w_])_(?!\s)([^_\n]+?)_(?!\w)/g, "$1<em>$2</em>");
    s = s.replace(/~~([^\n]+?)~~/g, "<del>$1</del>");
    return s.replace(/\u0001(\d+)\u0001/g, (m, i) => codes[+i]);
  }

  const BLOCK_START = /^(\s{0,3}#{1,6}\s|\s{0,3}>|\s*([-*+]|\d+[.)])\s+|\s{0,3}([-*_])(\s*\3){2,}\s*$|\u0000B\d+\u0000)/;
  const LIST_ITEM = /^(\s*)([-*+]|\d+[.)])\s+(.*)$/;
  const TABLE_SEP = /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/;

  function renderList(lines) {
    // lines: consecutive list lines; indentation decides nesting
    let html = "";
    const stack = []; // {indent, tag}
    const close = (to) => { while (stack.length && stack[stack.length - 1].indent > to) { html += `</li></${stack.pop().tag}>`; } };
    for (const line of lines) {
      const m = LIST_ITEM.exec(line);
      if (!m) { html += "<br>" + inline(line.trim()); continue; }       // wrapped continuation line
      const indent = m[1].replace(/\t/g, "    ").length, tag = /\d/.test(m[2]) ? "ol" : "ul";
      if (!stack.length || indent > stack[stack.length - 1].indent) {
        html += `<${tag}>`; stack.push({ indent, tag });
      } else {
        close(indent);
        if (stack.length && stack[stack.length - 1].indent === indent) html += "</li>";
        else { html += `<${tag}>`; stack.push({ indent, tag }); }
      }
      html += `<li>${inline(m[3])}`;
    }
    close(-1);
    return html;
  }

  function cells(row) { return row.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim()); }

  function render(src) {
    const blocks = [];
    src = src.replace(/\r\n?/g, "\n").replace(/```[^\n]*\n([\s\S]*?)(```|$)/g, (m, code) => {
      blocks.push(`<pre><code>${esc(code.replace(/\n$/, ""))}</code></pre>`);
      return `\n\u0000B${blocks.length - 1}\u0000\n`;
    });
    const lines = src.split("\n");
    let out = "", i = 0;
    while (i < lines.length) {
      const line = lines[i];
      if (!line.trim()) { i++; continue; }
      let m;
      if ((m = /^\u0000B(\d+)\u0000$/.exec(line.trim()))) { out += blocks[+m[1]]; i++; continue; }
      if ((m = /^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$/.exec(line))) { out += `<h${m[1].length}>${inline(m[2])}</h${m[1].length}>`; i++; continue; }
      if (/^\s{0,3}([-*_])(\s*\1){2,}\s*$/.test(line)) { out += "<hr>"; i++; continue; }
      if (/^\s{0,3}>/.test(line)) {
        const q = [];
        while (i < lines.length && /^\s{0,3}>/.test(lines[i])) q.push(lines[i++].replace(/^\s{0,3}> ?/, ""));
        out += `<blockquote>${render(q.join("\n"))}</blockquote>`; continue;
      }
      if (line.includes("|") && i + 1 < lines.length && TABLE_SEP.test(lines[i + 1]) && lines[i + 1].includes("-")) {
        const head = cells(line); i += 2;
        const rows = [];
        while (i < lines.length && lines[i].trim() && lines[i].includes("|")) rows.push(cells(lines[i++]));
        out += `<div class="tablewrap"><table><thead><tr>${head.map((c) => `<th>${inline(c)}</th>`).join("")}</tr></thead><tbody>` +
          rows.map((r) => `<tr>${r.map((c) => `<td>${inline(c)}</td>`).join("")}</tr>`).join("") + "</tbody></table></div>";
        continue;
      }
      if (LIST_ITEM.test(line)) {
        const l = [];
        while (i < lines.length && lines[i].trim() && (LIST_ITEM.test(lines[i]) || (/^\s+\S/.test(lines[i]) && !BLOCK_START.test(lines[i])))) l.push(lines[i++]);
        out += renderList(l); continue;
      }
      const p = [];
      while (i < lines.length && lines[i].trim() && (p.length === 0 || !BLOCK_START.test(lines[i]))) p.push(lines[i++]);
      out += `<p>${p.map(inline).join("<br>")}</p>`;
    }
    return out;
  }

  root.renderMarkdown = render;
  if (typeof module !== "undefined") module.exports = render;
})(typeof window !== "undefined" ? window : globalThis);
