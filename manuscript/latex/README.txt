Survival-Aware CAR-T Supply Chains — Methods section (LaTeX source)
===================================================================

CONTENTS
  main.tex             Standalone wrapper. Compile this.
  methods.tex          The Methods section itself. Self-contained; \input-able
                       into a host manuscript.
  preamble.tex         Packages and notation macros used by methods.tex.
  figures/
    Framework.png      Figure 1, the two-layer framework diagram.
  main.pdf             Pre-compiled output (12 pages), for reference.

BUILD
  pdflatex main
  pdflatex main        (run twice so \eqref and \ref resolve)

  Verified to compile with a minimal TeX Live installation
  (texlive-latex-base + texlive-latex-recommended): 12 pages, no errors, no
  warnings, no overfull boxes.

DROPPING THIS INTO AN EXISTING MANUSCRIPT
  1. Copy the notation macros from preamble.tex into your manuscript's
     preamble (the \newcommand block), together with amsmath, graphicx,
     booktabs and tabularx if you do not already load them.
  2. \input{methods} at the point the section belongs.
  3. Delete main.tex and preamble.tex; nothing in methods.tex depends on them.

  main.tex sets \setcounter{section}{2} so that the section is numbered 3.
  Remove that line if the Methods section sits elsewhere in your numbering.

EQUATION NUMBERING
  Numbering is automatic and sequential, (1)-(49) in order of appearance.
  Every cross-reference uses \eqref against a semantic label (eq:obj, eq:hold,
  eq:index, ...), so inserting or removing an equation renumbers the whole
  section consistently with no hand-editing.

  Table 2 in the section maps these numbers back to the block-structured
  numbering used in Survival_Aware_iSHIPMENT_Formulation.docx and in the code
  repository, so the manuscript, the specification and the implementation stay
  cross-referenceable.

NOTES
  - lmodern and microtype are loaded only if present (\IfFileExists), so the
    source compiles on minimal installations. On a full installation they load
    and improve typography.
  - The framework figure is labelled P1-P8 in the artwork while the manuscript
    numbers those equations (37)-(44). The caption names both. If you re-export
    the figure, consider relabelling it to match the manuscript.
  - The prose is identical to manuscript/methods.md in the repository; that
    file is the Markdown source of record for review, this is the typeset form.
