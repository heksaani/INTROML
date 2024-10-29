window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true,
    tags: 'ams',  // Automatic numbering for equations
    packages: { '[+]': ['ams'] }  // Load AMS package for referencing
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

document$.subscribe(() => { 
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});