# Candidate figure-grounding demand rubrics — ADeLe/DeLeAn format

Three candidate demand rubrics (VL, GS, MA), rendered in the same style and structure as the
ADeLe/DeLeAn rubric appendix: a single definition paragraph carrying the range, the
scope note ("the challenge is not…but…"), the boundary discussion ("this differs from…"), and
a "Noteworthy" confound caveat; followed by **Levels** 0–5+, each with a name, a criterion,
and bulleted examples.

---

## Visual Localization and Grounding (VL)

This criterion assesses the level of visual search and grounding required to locate the specific
element a claim refers to, and to bind it to the term that names it, before any value,
relationship, or shape can be read from the figure. During this process there is the need to
connect an entity named in the claim to its corresponding mark in the figure—a mark that may be
directly labeled, distinguished only by position, or named by a semantic property the figure
never prints. The level represents how far the figure must be searched and how indirect the path
from claim term to visual referent is, ranging from a single labeled element already in view to
an exhaustive search for a referent that carries no label and must be identified through an
inferred external property. The challenge is not on deciding what to look for, nor on decoding
the target once found, but on deploying attention to find it and confirming that the mark located
is the one the claim names; the level reflects the minimum search the claim forces, since many
claims could be checked by scanning the whole figure but are settled cheaply once the right mark
is found. This differs from tasks where the need is to select which of several marks already
under consideration is the relevant one, and from the separate matter of a claim term that has no
counterpart in the figure at all. While these may overlap when a claim is at once indirect,
aggregative, and loosely grounded, localization specifically concerns finding and binding a
referent that does exist in the figure, rather than operating on it once located or detecting its
absence. Noteworthy, a mark that is easy to locate but rendered too small or rotated to read
cleanly is not high on this scale—resolution is a separate matter, and under the working
assumption that benchmark figures render effectively such cases are rare; likewise, binding a
referent the claim names to a label the figure prints remains low even in a busy figure, and the
upper levels are reached only when the binding rests on a property the figure does not print and
the reader must supply.

**Levels**

**Level 0 None.** No search is required. The claim's target is the only element present, or is
directly labeled and already in view. *Examples:*
- "The figure has a single bar and the claim says it exceeds 50%."
- "A scatter plot carries one labeled outlier and the claim is about that point."
- "A micrograph shows one stained cell and the claim concerns its shape."

**Level 1 Very low.** Minimal search is required. There is a single salient target and no need to
choose among like items—it is set apart by a clear direct label or a uniquely distinct position.
*Examples:*
- "The red line sits above the blue line and the claim compares red and blue."
- "Among bars of one colour, only one is hatched and marked 'ours', and the claim is about 'ours'."
- "A volcano plot labels one gene in the corner and the claim names that gene."

**Level 2 Low.** Some search is required. There are several like candidates and the target is
selected among them by reading an explicit distinguisher—a panel title, a legend entry, a column
or row header. The match is made by reading a label, not by inference. *Examples:*
- "In the middle panel of three titled panels, the orange series peaks first."
- "Read the 'val' column among the labeled columns of a results table."
- "Pick the 'IL-6' row from a labeled cytokine heatmap and read its top cell."

**Level 3 Intermediate.** Moderate search is required. Many like candidates must be read one by
one before the target can be matched, or the target is a region the figure cues through a guide it
does draw, such as a gridline or quadrant. The distinguisher is still printed; the cost is the
breadth of reading it. *Examples:*
- "In the dataset-D subplot of a labeled small-multiples grid, the green bar is tallest."
- "Locate panel (g) among a labeled twelve-panel composite (the letters are legible but numerous) and read its peak."
- "Find the cluster falling in the lower-left quadrant marked by the gridlines and judge its density."

**Level 4 High.** Sustained search is required. No printed cue distinguishes the target: it sits
among visually similar candidates with no label setting it apart, or it is a semantic region the
figure never names and that must be inferred from domain structure before reading. *Examples:*
- "Among many same-coloured curves, none labeled, identify the one the claim singles out only by its late-time plateau."
- "Read the value at the region the claim calls 'the jet–disk interface', which the figure never marks."
- "In a composite whose panels share a style and differ only subtly, find the one whose trend reverses."

**Level 5+ Very High.** Exhaustive search with no localization cue is required. The claim's entity
carries no figure label at all and its mark must be identified through an inferred external
property, possibly while scanning many dense candidates at once. *Examples:*
- "The model identifiable only by its parameter count lies on the Pareto front, though the figure labels none of the points."
- "Across all the spark-panels, locate the one whose unlabeled inset trend is non-monotonic."
- "Identify the spectral line the claim specifies only by its rest wavelength, among a dense forest of unlabeled lines."

---

## Gestalt and Shape Judgment (GS)

This criterion assesses the level of holistic, non-pointwise visual reasoning required to verify a
claim about the shape of the data rather than about any single value. It covers judging the
direction of a trend, the curvature or convexity of a function, conformance to a functional form
or a reference line, convergence to a plateau, and the number or separation of clusters. The level
represents how much the verification depends on integrating many marks into one qualitative
impression and how subtle that impression is, ranging from no shape judgment at all to
distinguishing a specific parametric form from a near alternative; the level reflects the shape
judgment the claim forces, not the richest one the figure could support, so finer structure that
happens to be visible does not raise it if a coarse impression settles the claim. Noteworthy, a
shape judgment produces no intermediate number—the verifier forms an impression of the whole
rather than reading and combining parts—and a claim that can be confirmed by reading one or two
specific values is not assessed here even when the figure shows a curve. This differs from tasks
that operate arithmetically on several extracted values. While a claim about shape can sometimes
also be checked by reading many points and computing over them, gestalt judgment specifically
concerns the qualitative impression the marks create together, rather than any quantity derived
from them—where each input is a discrete read value to be combined, the demand belongs to
aggregation, not to shape.

**Levels**

**Level 0 None.** No shape judgment is required; the claim concerns a single value or identity.
*Examples:*
- "Point A is at 0.8."
- "The bar for 2020 is blue."

**Level 1 Very low.** A coarse, obvious global impression. *Examples:*
- "The line increases overall."
- "The points form one visible group."

**Level 2 Low.** A clear directional or separation judgment read off a set of marks. *Examples:*
- "The curve rises then falls."
- "The two clusters are visibly separated rather than overlapping."

**Level 3 Intermediate.** A second-order judgment made against a reference the figure draws:
convexity, whether successive differences grow or shrink, or conformance to a line that is
explicitly plotted. The standard of comparison is present on the page. *Examples:*
- "The curve is concave, flattening as x increases."
- "The points track the drawn diagonal closely."

**Level 4 High.** A subtle judgment made against a reference that is implicit or absent:
distinguishing a claimed functional form from a near alternative, judging convergence to an
asymptote that is never drawn, or counting clusters in a dense field. The verifier must supply the
standard of comparison. *Examples:*
- "The decay is exponential rather than linear."
- "The training curve converges to a stable plateau after the spike."
- "The embedding separates into several distinct clusters."

**Level 5+ Very High.** A compound judgment combining several of the above across regions or
panels, with transients or noise that must be separated from the signal. *Examples:*
- "Both curves converge, but only past the change-point and to different asymptotes."
- "The distribution conforms to the unplotted diagonal in one panel but not the other."

---

## Multi-Element Visual Aggregation (MA)

This criterion assesses the level of computation required over multiple values extracted from the
figure to verify a claim, including summation, differencing, ratio, percentage change, ranking,
and extremum-finding across a set. The level represents how many marks must be read and how
complex the operation combining them is, ranging from a single direct readout to chained
computation or a full ranking over a large set; the level reflects the minimum computation the
claim forces, so a claim confirmable by one decisive read, or a comparison of two, stays low even
when the figure invites more arithmetic. Noteworthy, each input here is a discrete value the
verifier reads from the figure—the demand is to operate on those readings, not to perceive
them—and the level is set by the operation, not by how many marks happen to sit in the figure, so
reading one value from a crowded twenty-bar chart is still the lowest level. A useful test for the
boundary between bounded and whole-set operations is whether the marks needed can be named before
any are read: if the inputs are fixed and nameable up front, such as two endpoints or a stated set
of categories, the operation is bounded; if the whole set must be surveyed because any unread
element could change the answer, such as a mean, an interior rank, or a top-k membership, it is
whole-set. This differs from forming a holistic impression of shape, where the claim is settled by
the qualitative pattern many marks create together with no number computed; aggregation
specifically concerns arithmetic or ordinal operations over values already extracted.

**Levels**

**Level 0 None.** A single value answers the claim; nothing is combined, even when many marks
surround it. *Examples:*
- "Accuracy is 90%."
- "The peak of the loss curve is at epoch 10."
- "The brightest pixel in the heatmap is in the top row."

**Level 1 Very low.** A single comparison or difference between two read values. *Examples:*
- "Method A scores higher than method B."
- "The gap between the two curves at the final step is about 0.2."
- "The treated sample shows roughly twice the expression of the control."

**Level 2 Low.** An operation over a small handful of values whose inputs are obvious, or finding
the single maximum or minimum of a set in one pass. *Examples:*
- "Method C is the highest of the four plotted."
- "The two stacked segments together exceed the third bar."
- "The hottest cell in the labeled 3×3 block lies on the diagonal."

**Level 3 Intermediate.** Multi-step arithmetic whose inputs you can name before reading them—a
percentage change between two endpoints, a ratio of two differences, a total over a stated set of
categories—or an interior rank over a small set whose ordering is visually obvious. Reading
further marks would not change the result. *Examples:*
- "Accuracy rises by about two-fifths from the first epoch to the last."
- "The top two pie slices together make up over half."
- "Of five clearly separated bars, the third-tallest is the baseline."

**Level 4 High.** An operation that cannot name its inputs up front and so forces a survey of the
whole set, because any unread element could change the answer: a statistic over many marks (mean,
spread, count), an interior or k-th rank settled only by measuring and ruling each other element
in or out, or a top-k membership. *Examples:*
- "Method E is the second-lowest of the nine, none clearly separated."
- "The mean of the dozen plotted points sits near 0.5."
- "These four genes are the top four by fold-change across the whole volcano plot."

**Level 5+ Very High.** Chained aggregation combining several of the above: a whole-set operation
computed per category and then combined or ranked across categories, or a within-subset extremum
contrasted with a global one. *Examples:*
- "The per-model total, summed over its stacked components, is highest at model D among the eight."
- "The minimum within the low-dose subset differs from the global minimum across all doses."
- "Averaging each row of the heatmap, the highest row mean falls in the second cluster, not the first."
