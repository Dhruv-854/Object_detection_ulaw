# Technical Report Outline

1. **Abstract** — research question, method, verified headline results, limitations.
2. **Introduction** — motivation, objectives, contribution, and scope.
3. **Background and literature** — object detection, one-stage detectors, YOLOv5,
   current end-to-end YOLO models, metrics, related comparative work.
4. **Requirements and ethics** — inputs, outputs, non-functional requirements,
   privacy, approval status, licences, and exclusions.
5. **System design** — source pipeline, normalized detector interface, model-specific
   post-processing, output schemas, device selection, and reproducibility controls.
6. **Dataset** — collection protocol, annotation, audit, grouped split, balance,
   negatives, variation, bias, and dataset card.
7. **Experimental method** — four conditions, fixed settings, validation threshold
   selection, frozen test set, benchmark protocol, and hardware.
8. **Results** — regenerated numerical tables and figures only; no copied evidence.
9. **Critical analysis** — accuracy/speed tradeoffs, robustness, qualitative errors,
   threats to validity, environmental cost, and comparison with prior work.
10. **Conclusion and future work** — answer the research question without overstating
    generality; identify justified extensions.
11. **References and appendices** — Harvard-style sources, user guide, schemas,
    complete configs, checksums, audit, raw-result locations, and test evidence.

Never fill a placeholder with an estimated or inherited value. Every empirical claim
must trace to a saved run produced by this repository.

