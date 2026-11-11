# TruthLens design system

## Direction

TruthLens uses a darkroom inspired visual language for a careful image evidence tool. The interface favors warm black surfaces, paper white text, fine rules, and a restrained amber action color. The design avoids decorative gradients, stock imagery, invented model scores, and claims that imply certainty.

## Type and layout

- Display type: Sofia Sans Extra Condensed, with a sans-serif fallback.
- Interface and body type: Barlow, with a sans-serif fallback.
- Layout: left-led editorial landing page and evidence sheet layouts for the image workflow.
- Corners: square; use hairline borders to define sections.
- Motion: short state transitions only, disabled when reduced motion is requested.

## Product language

- Say “estimate” for detector output and “validate” for file checks.
- Keep the warning that a detector result is not proof of image origin.
- Show missing detector output as inconclusive or unavailable, never as a real/AI verdict.
- Preserve the upload field, progress feedback, preview, metadata, and result hooks.

## Shared files

- `frontend/css/tokens.css` defines the palette, type, spacing, and motion values.
- `frontend/css/base.css` defines document defaults and focus behavior.
- `frontend/css/site.css` defines shared navigation, landing, upload, analysis, and result layouts.
