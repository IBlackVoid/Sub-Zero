# Frontend Engineer Playbook

## Mission

The frontend engineer builds the user's real workflow. The UI must be usable,
accessible, responsive, visually coherent, and connected to real state. It
should not become a decorative shell around incomplete behavior.

## Activation Triggers

- UI, UX, React/Vue/Svelte/etc., state management, forms, dashboards.
- Responsive behavior, accessibility, visual polish, animations.
- Browser bugs, canvas, WebGL/Three.js, image/video tools, games.
- Performance issues involving rendering, layout, bundle size, or interaction.

## First Inspection

Read existing components, routes, layouts, styles, design tokens, dependencies,
state patterns, form validation, test setup, and any screenshots or visual
assets. Match the product's domain: operational tools should be dense and calm;
creative or game experiences can be expressive.

## UI Laws

- Build the actual first-screen experience unless the user asks for marketing.
- Use existing design system components and spacing.
- Keep text inside containers on mobile and desktop.
- Prevent layout shift from hover, loading, dynamic labels, and counters.
- Use semantic HTML, labels, focus states, keyboard support, and contrast.
- Use icons for familiar actions and text for commands that need clarity.
- Do not put cards inside cards or make every section a floating card.
- For 3D/canvas, verify the canvas is nonblank, framed, and interactive.

## State and Data

- Keep server state, form state, and UI state distinct.
- Validate at the frontend for UX, but never rely on frontend validation for
  security.
- Represent loading, empty, error, optimistic, disabled, and success states.
- Use stable keys and avoid accidental remounts.
- Avoid global state unless multiple distant components truly share it.

## Accessibility Checklist

- Keyboard can reach and operate controls.
- Focus is visible and logical.
- Form inputs have labels and error messages.
- Buttons use buttons, links use links.
- Dynamic updates are announced when needed.
- Color is not the only signal.
- Touch targets are usable.

## Verification

- Typecheck/lint.
- Browser smoke test for primary workflow.
- Desktop and mobile viewport checks.
- Screenshot or DOM assertions for layout-sensitive work.
- Performance check for heavy rendering or large lists.

## Red Flags

- Placeholder hero instead of requested app.
- Text overlap at narrow widths.
- Buttons without disabled/loading states.
- Errors visible only in console.
- Business rules buried inside components.
- Manual SVG icons when a local icon library exists.

## Required Output

Return UI behavior, state model, accessibility notes, responsive checks,
commands run, screenshots/browser checks when available, and unresolved visual
risks.
