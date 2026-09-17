/**
 * flowchart-zoom.js
 * Standalone, zero-dependency interactive zoom and pan engine for architectural flowcharts and SVGs.
 * Supports toolbar buttons (+, -, Reset), mouse wheel zoom, click-drag panning, and touch pinch.
 */

(function () {
  'use strict';

  function initZoomableFlowcharts() {
    const mermaidContainers = document.querySelectorAll('.mermaid, .zoomable-diagram');
    mermaidContainers.forEach((container, index) => {
      // Avoid double initialization
      if (container.dataset.zoomInitialized === 'true') return;

      // Check if SVG has been rendered inside container
      const svg = container.querySelector('svg');
      if (!svg) {
        // If mermaid is still rendering, retry shortly
        setTimeout(initZoomableFlowcharts, 120);
        return;
      }

      container.dataset.zoomInitialized = 'true';
      setupZoomPan(container, svg, index);
    });
  }

  function setupZoomPan(container, svg, id) {
    // Add wrapper classes and styling
    container.classList.add('flowchart-wrapper');

    // Create toolbar controls
    const toolbar = document.createElement('div');
    toolbar.className = 'flowchart-toolbar';
    toolbar.setAttribute('role', 'toolbar');
    toolbar.setAttribute('aria-label', 'Flowchart Zoom and Pan Controls');

    toolbar.innerHTML = `
      <button type="button" class="btn-flowchart-zoom btn-zoom-in" title="Zoom In (+)" aria-label="Zoom in flowchart">+</button>
      <button type="button" class="btn-flowchart-zoom btn-zoom-out" title="Zoom Out (-)" aria-label="Zoom out flowchart">&minus;</button>
      <button type="button" class="btn-flowchart-zoom btn-zoom-reset" title="Reset View (100%)" aria-label="Reset flowchart zoom">↺</button>
      <span class="flowchart-zoom-badge" aria-live="polite">100%</span>
    `;

    // Wrap SVG in a viewport stage
    const stage = document.createElement('div');
    stage.className = 'flowchart-stage';
    svg.parentNode.insertBefore(stage, svg);
    stage.appendChild(svg);

    container.insertBefore(toolbar, stage);

    // Zoom and Pan State
    let scale = 1;
    let translateX = 0;
    let translateY = 0;
    let isDragging = false;
    let startX = 0;
    let startY = 0;

    const MIN_SCALE = 0.5;
    const MAX_SCALE = 3.5;
    const badge = toolbar.querySelector('.flowchart-zoom-badge');

    function applyTransform(smooth = false) {
      if (smooth) {
        svg.style.transition = 'transform 0.18s ease-out';
      } else {
        svg.style.transition = 'none';
      }
      svg.style.transformOrigin = 'center center';
      svg.style.transform = `translate(${translateX}px, ${translateY}px) scale(${scale})`;
      if (badge) {
        badge.textContent = `${Math.round(scale * 100)}%`;
      }
    }

    function zoom(factor) {
      const newScale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, scale * factor));
      scale = newScale;
      applyTransform(true);
    }

    function reset() {
      scale = 1;
      translateX = 0;
      translateY = 0;
      applyTransform(true);
    }

    // Button event listeners
    toolbar.querySelector('.btn-zoom-in').addEventListener('click', (e) => {
      e.stopPropagation();
      zoom(1.25);
    });

    toolbar.querySelector('.btn-zoom-out').addEventListener('click', (e) => {
      e.stopPropagation();
      zoom(0.8);
    });

    toolbar.querySelector('.btn-zoom-reset').addEventListener('click', (e) => {
      e.stopPropagation();
      reset();
    });

    // Mouse Drag Panning
    stage.addEventListener('mousedown', (e) => {
      // Only left click
      if (e.button !== 0) return;
      isDragging = true;
      startX = e.clientX - translateX;
      startY = e.clientY - translateY;
      stage.classList.add('is-panning');
      e.preventDefault();
    });

    window.addEventListener('mousemove', (e) => {
      if (!isDragging) return;
      translateX = e.clientX - startX;
      translateY = e.clientY - startY;
      applyTransform(false);
    });

    window.addEventListener('mouseup', () => {
      if (isDragging) {
        isDragging = false;
        stage.classList.remove('is-panning');
      }
    });

    // Mouse Wheel Zoom
    stage.addEventListener('wheel', (e) => {
      e.preventDefault();
      const delta = e.deltaY < 0 ? 1.15 : 0.87;
      zoom(delta);
    }, { passive: false });

    // Touch Support for Mobile / Touchscreens
    let initialTouchDistance = null;
    stage.addEventListener('touchstart', (e) => {
      if (e.touches.length === 1) {
        isDragging = true;
        startX = e.touches[0].clientX - translateX;
        startY = e.touches[0].clientY - translateY;
      } else if (e.touches.length === 2) {
        isDragging = false;
        const dx = e.touches[0].clientX - e.touches[1].clientX;
        const dy = e.touches[0].clientY - e.touches[1].clientY;
        initialTouchDistance = Math.hypot(dx, dy);
      }
    }, { passive: true });

    stage.addEventListener('touchmove', (e) => {
      if (isDragging && e.touches.length === 1) {
        translateX = e.touches[0].clientX - startX;
        translateY = e.touches[0].clientY - startY;
        applyTransform(false);
      } else if (e.touches.length === 2 && initialTouchDistance) {
        const dx = e.touches[0].clientX - e.touches[1].clientX;
        const dy = e.touches[0].clientY - e.touches[1].clientY;
        const dist = Math.hypot(dx, dy);
        const factor = dist / initialTouchDistance;
        scale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, scale * factor));
        initialTouchDistance = dist;
        applyTransform(false);
      }
    }, { passive: true });

    stage.addEventListener('touchend', () => {
      isDragging = false;
      initialTouchDistance = null;
    });

    // Initialize clean view and trigger flowchart math rendering
    applyTransform(false);
    if (window.renderFlowchartMath) {
      window.renderFlowchartMath(container);
    }
  }

  // Hook into document load and Mermaid render cycle
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      setTimeout(initZoomableFlowcharts, 300);
    });
  } else {
    setTimeout(initZoomableFlowcharts, 300);
  }

  // Export globally for manual re-runs after dynamic renders
  window.initZoomableFlowcharts = initZoomableFlowcharts;
})();
