# Visual direction

The service uses a restrained desktop workspace: a compact sidebar, a stable toolbar, quiet surface changes and large source images. Application actions and real results take visual precedence over instrumentation. The direction is an independent professional image workspace, not an imitation of an operating system.

## References

- [Untitled UI — free Figma UI kit](https://www.untitledui.com/free-figma-ui-kit) — reference for consistent typography, component proportions, spacing and sidebar layouts. Its paid components and source files are not included.
- [File manager — Untitled UI, Jordan Hughes](https://dribbble.com/shots/25583251-File-manager-Untitled-UI) — visual reference for navigation, content hierarchy and an orderly image/file workspace.
- [Figma dashboard templates](https://www.figma.com/templates/dashboard-designs/) — explored dashboard layout references; the product does not copy decorative analytics or irrelevant counters from these examples.
- [Apple Design Resources](https://developer.apple.com/design/resources/) — an additional reference for desktop hierarchy and interaction, not a required visual style.
- [Figma: Apple's UI kit](https://help.figma.com/hc/en-us/articles/24037833895831-Get-started-with-Apple-s-UI-kit) — reference for consistent component and typography systems.
- [Onest](https://github.com/google/fonts/tree/main/ofl/onest) — primary variable typeface with Cyrillic, SIL Open Font License.
- [IBM Plex Mono](https://github.com/google/fonts/tree/main/ofl/ibmplexmono) — compact identifiers and technical values, SIL Open Font License.

Font files and licenses are supplied under `frontend/public/fonts/`, so runtime does not contact Google Fonts or a CDN. SF Pro is not redistributed. The interface is an original implementation for this product; no claim is made that a Figma file was imported or that Apple endorses it.

Onest was chosen for the Russian-language interface and its distinctive but quiet shapes. IBM Plex Mono is limited to identifiers and technical values. Keeping two complementary families avoids a disconnected mixture of headline and interface styles.

## Product rules

- The source image remains visible during comparison. Images preserve their aspect ratios.
- Typographic hierarchy, alignment and whitespace do most of the work; a single accent identifies the primary action.
- Important states use text and icons as well as color. Focus is visible; dialogs can be closed with a keyboard; animations respect reduced motion.
- A model stub remains explicitly visible. Similarity values in this mode are not identity probabilities.
- Empty states guide the next real action. No invented counters, example vehicles or decorative model-performance charts.
