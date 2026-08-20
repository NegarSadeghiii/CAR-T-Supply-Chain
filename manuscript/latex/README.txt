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

REFERENCES
  main.tex carries a one-entry thebibliography so the wrapper compiles with
  pdflatex alone. references.bib holds the same entry for BibTeX/biblatex use
  in a host manuscript. The entry is INCOMPLETE by design: title, first author,
  journal and year are verified, but the full author list, volume, article
  number and DOI are left blank rather than guessed. Paste the complete record
  from Zotero before submission.

NOTES
  - lmodern and microtype are loaded only if present (\IfFileExists), so the
    source compiles on minimal installations. On a full installation they load
    and improve typography.
  - The framework figure's equation labels should be edited to match the
    manuscript numbering: the per-epoch problem is (37)-(44) and the
    re-collection gate is (45).
  - The prose is identical to manuscript/methods.md in the repository; that
    file is the Markdown source of record for review, this is the typeset form.
