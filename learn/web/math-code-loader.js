/**
 * math-code-loader.js
 * Robust KaTeX mathematical formula rendering and latent LaTeX code inspector.
 * Guarantees math formulas render properly and provides on-demand inspection/copying of underlying LaTeX code.
 */

(function () {
  'use strict';

  // Walk text nodes and wrap any raw $$...$$ display equations in a display-math-block
  function wrapRawDisplayMath(root = document.body) {
    const walker = document.createTreeWalker(
      root,
      NodeFilter.SHOW_TEXT,
      {
        acceptNode(node) {
          if (!node.nodeValue.includes('$$')) return NodeFilter.FILTER_REJECT;
          const parent = node.parentElement;
          if (!parent) return NodeFilter.FILTER_REJECT;
          if (parent.closest('pre, code, script, style, .latex-source-panel, .display-math-block, .math-step-formula, .cell-inspector-math, .node-mpnn-math, .flowchart-stage, .flowchart-wrapper, .mermaid')) {
            return NodeFilter.FILTER_REJECT;
          }
          return NodeFilter.FILTER_ACCEPT;
        }
      }
    );

    const nodesToReplace = [];
    while (walker.nextNode()) {
      nodesToReplace.push(walker.currentNode);
    }

    nodesToReplace.forEach(textNode => {
      const parent = textNode.parentNode;
      if (!parent) return;
      const text = textNode.nodeValue;
      const parts = text.split(/(\$\$[\s\S]+?\$\$)/g);
      if (parts.length <= 1) return;

      const fragment = document.createDocumentFragment();
      parts.forEach(part => {
        if (!part) return;
        const match = part.match(/^\$\$([\s\S]+?)\$\$$/);
        if (match) {
          const div = document.createElement('div');
          div.className = 'display-math-block';
          div.dataset.latexCode = match[1].trim();
          div.dataset.latexProcessed = 'true';
          div.textContent = part;
          fragment.appendChild(div);
        } else {
          fragment.appendChild(document.createTextNode(part));
        }
      });
      parent.replaceChild(fragment, textNode);
    });
  }

  // Extract raw LaTeX source from element before KaTeX transforms it
  function preprocessMathBlocks() {
    wrapRawDisplayMath();

    const mathContainers = document.querySelectorAll('.math-step-formula, .cell-inspector-math, .node-mpnn-math, .display-math-block');
    mathContainers.forEach(container => {
      if (container.dataset.latexProcessed === 'true') return;
      
      const rawText = container.textContent.trim();
      let latexCode = rawText;

      const matchDisplay = rawText.match(/^\$\$([\s\S]+)\$\$$/) || rawText.match(/^\\\[([\s\S]+)\\\]$/);
      if (matchDisplay) {
        latexCode = matchDisplay[1].trim();
      }

      container.dataset.latexCode = latexCode;
      container.dataset.latexProcessed = 'true';
    });
  }

  // Attach latent code toggle badge and viewer panel to display math blocks
  function attachLatentCodeInspectors() {
    const mathContainers = document.querySelectorAll('.math-step-formula, .cell-inspector-math, .node-mpnn-math, .display-math-block');

    mathContainers.forEach(container => {
      if (container.dataset.inspectorAttached === 'true') return;
      if (!container.dataset.latexCode) return;

      container.dataset.inspectorAttached = 'true';
      container.classList.add('latex-inspectable-wrap');

      // Add "TeX" pill badge
      const badge = document.createElement('button');
      badge.type = 'button';
      badge.className = 'btn-latex-pill';
      badge.title = 'View / Copy Latent LaTeX Code';
      badge.setAttribute('aria-label', 'View LaTeX mathematical source code');
      badge.textContent = 'TeX';

      // Add collapsible LaTeX source panel
      const panel = document.createElement('div');
      panel.className = 'latex-source-panel';
      panel.hidden = true;
      panel.innerHTML = `
        <div class="latex-source-header">
          <span class="latex-source-title">Latent LaTeX Source Code</span>
          <div class="latex-source-actions">
            <button type="button" class="btn-copy-latex" aria-label="Copy LaTeX code to clipboard">Copy TeX</button>
            <button type="button" class="btn-close-latex" aria-label="Close LaTeX code viewer">&times;</button>
          </div>
        </div>
        <pre class="latex-source-code"><code>${escapeHtml(container.dataset.latexCode)}</code></pre>
      `;

      badge.addEventListener('click', (e) => {
        e.stopPropagation();
        const isHidden = panel.hidden;
        panel.hidden = !isHidden;
        badge.classList.toggle('active', !panel.hidden);
      });

      const closeBtn = panel.querySelector('.btn-close-latex');
      closeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        panel.hidden = true;
        badge.classList.remove('active');
      });

      const copyBtn = panel.querySelector('.btn-copy-latex');
      copyBtn.addEventListener('click', async (e) => {
        e.stopPropagation();
        try {
          await navigator.clipboard.writeText(container.dataset.latexCode);
          copyBtn.textContent = 'Copied!';
          copyBtn.classList.add('copied');
          setTimeout(() => {
            copyBtn.textContent = 'Copy TeX';
            copyBtn.classList.remove('copied');
          }, 1800);
        } catch (err) {
          // Fallback clipboard method
          const ta = document.createElement('textarea');
          ta.value = container.dataset.latexCode;
          document.body.appendChild(ta);
          ta.select();
          document.execCommand('copy');
          document.body.removeChild(ta);
          copyBtn.textContent = 'Copied!';
          setTimeout(() => {
            copyBtn.textContent = 'Copy TeX';
          }, 1800);
        }
      });

      container.appendChild(badge);
      container.appendChild(panel);
    });
  }

  function escapeHtml(str) {
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  // Render LaTeX math inside Mermaid flowchart nodes with auto-bbox adjustment
  function renderFlowchartMath(root = document.body) {
    if (!window.katex) return;

    // 1. Render all explicit .f-math elements
    root.querySelectorAll('.f-math').forEach(el => {
      if (el.dataset.katexRendered === 'true') return;
      let tex = (el.dataset.rawTex || el.textContent).trim();
      if (!el.dataset.rawTex) {
        el.dataset.rawTex = tex;
      }
      // Unescape doubled backslashes from Mermaid string literal escaping (e.g. \\alpha -> \alpha)
      tex = tex.split('\\\\').join('\\');
      try {
        katex.render(tex, el, { throwOnError: false, displayMode: false });
        el.dataset.katexRendered = 'true';
      } catch (e) {
        console.warn('Flowchart math render error:', e);
      }
    });

    // 2. Scan .nodeLabel for any $...$ or residual math expressions
    root.querySelectorAll('.nodeLabel').forEach(label => {
      if (label.dataset.mathProcessed === 'true') return;

      if (label.querySelector('.katex')) {
        label.dataset.mathProcessed = 'true';
        return;
      }

      let html = label.innerHTML;
      if (html.includes('$')) {
        html = html.replace(/\$([^$]+)\$/g, (match, tex) => {
          try {
            return katex.renderToString(tex.trim(), { throwOnError: false, displayMode: false });
          } catch(err) {
            return match;
          }
        });
        label.innerHTML = html;
        label.dataset.mathProcessed = 'true';
      }
    });

    // 3. Dynamic bbox adjustment on Mermaid node rects and foreignObjects
    root.querySelectorAll('.node').forEach(node => {
      const fo = node.querySelector('foreignObject');
      const rect = node.querySelector('rect.label-container');
      if (fo && rect) {
        const content = fo.firstElementChild;
        if (content) {
          const reqH = content.scrollHeight + 14;
          const reqW = content.scrollWidth + 24;
          const curH = parseFloat(rect.getAttribute('height') || '0');
          const curW = parseFloat(rect.getAttribute('width') || '0');
          if (reqH > curH) {
            rect.setAttribute('height', reqH);
            rect.setAttribute('y', -reqH / 2);
            fo.setAttribute('height', reqH);
          }
          if (reqW > curW) {
            rect.setAttribute('width', reqW);
            rect.setAttribute('x', -reqW / 2);
            fo.setAttribute('width', reqW);
          }
        }
      }
    });
  }

  // Resilient KaTeX renderer with retries and comprehensive delimiters
  function renderAllMath(target = document.body, attempts = 0) {
    preprocessMathBlocks();

    if (window.renderMathInElement) {
      try {
        window.renderMathInElement(target, {
          delimiters: [
            { left: '$$', right: '$$', display: true },
            { left: '\\[', right: '\\]', display: true },
            { left: '$', right: '$', display: false },
            { left: '\\(', right: '\\)', display: false }
          ],
          ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code', 'annotation', 'annotation-xml', 'svg', 'foreignObject'],
          ignoredClasses: ['mermaid', 'flowchart-wrapper', 'flowchart-stage', 'katex-math'],
          throwOnError: false
        });
        attachLatentCodeInspectors();
        renderFlowchartMath(target);
      } catch (err) {
        console.warn('KaTeX render warning:', err);
      }
    } else if (attempts < 30) {
      // Retry every 100ms up to 3 seconds for KaTeX scripts to finish loading
      setTimeout(() => renderAllMath(target, attempts + 1), 100);
    }
  }

  // Initialize on document ready and schedule flowchart math passes
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      renderAllMath();
      setTimeout(renderFlowchartMath, 250);
      setTimeout(renderFlowchartMath, 600);
      setTimeout(renderFlowchartMath, 1200);
    });
  } else {
    renderAllMath();
    setTimeout(renderFlowchartMath, 250);
    setTimeout(renderFlowchartMath, 600);
    setTimeout(renderFlowchartMath, 1200);
  }

  // Export globally for dynamic updates
  window.renderAllMath = renderAllMath;
  window.renderMath = renderAllMath; // Backwards compatible alias
  window.renderFlowchartMath = renderFlowchartMath;
})();
