# Ethics and Data Protocol

## Approval gate

The proposal records ethical approval as pending. Do not collect or retain footage
containing identifiable people until the university confirms whether approval and
consent are required. The custom research dataset is therefore restricted to five
non-person object classes.

## Collection rules

- Use objects owned by the researcher or used with permission.
- Frame scenes so faces, names, addresses, screens with personal information,
  documents, reflections, number plates, and bystanders are excluded.
- Use at least 20 capture sessions; a session is a distinct place/time/setup, not a
  short sequence of adjacent frames.
- Do not extract many near-identical frames from one video.
- Include controlled variation in lighting, distance, angle, background, occlusion,
  and motion blur, plus 100 genuinely negative images.
- Store raw and labelled data in access-controlled project storage. Do not publish
  it until every image has passed a privacy and licence review.

## Webcam behavior

Live webcam inference does not save frames by default. Saving requires the explicit
`--save` option and a lawful, consented testing context. The detector performs class
detection only; identity recognition, face recognition, and personal profiling are
out of scope.

## Reporting

Document data sources, permissions, deletion/retention dates, class imbalance,
known biases, failure cases, energy/hardware use, and model/dataset licences. Never
reuse the supplied reference report's screenshots, measurements, authorship, or
first-person claims as project evidence.

