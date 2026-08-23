# Release and archival checklist

1. Merge the SoftwareX preparation pull request after continuous integration passes.
2. Create the exact Git tag `v1.0.0` from the merged commit.
3. Create the GitHub release `FlowMLLab 1.0.0` from that tag.
4. Enable the repository in Zenodo and archive the GitHub release.
5. Add the returned version DOI to `CITATION.cff` and the manuscript metadata table.
6. Rebuild the article and preserve the archived DOI and commit together.

No DOI is inserted before Zenodo returns it; placeholders and guessed identifiers
must never enter the submission package.
