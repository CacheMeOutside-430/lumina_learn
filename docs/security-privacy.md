# Security And Privacy

- Raw screen frames are processed locally by default and are not persisted.
- Persisted events store OCR snippets, model outputs, timestamps, and analytics metadata.
- Cloud inference is disabled unless configured explicitly.
- Desktop capture is scoped to the active monitor through native OS APIs.
- WebSocket connections reject unexpected browser origins.
- HTTP CORS is configured through `CORS_ORIGINS` and does not allow credentials.
- Incoming frame payloads are bounded by `MAX_FRAME_BYTES`.
- Vector memory redacts common secrets, API tokens, email addresses, and card-like numbers before storage.
- Chroma telemetry is replaced by a no-op telemetry adapter for local privacy.
- The Tauri shell uses an explicit content security policy instead of disabling CSP.
- Future production releases should add encrypted SQLite, per-user model cache isolation, signed updates, and explicit data retention controls.

## Sensitive Data Controls

- Add app-level allow/deny lists before streaming frames.
- Disable memory writes for private browsing, password managers, banking, and auth flows.
- Expose a visible pause/resume capture control in the overlay.
