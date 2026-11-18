# Security notes

- Do not commit `.env` files, API keys, access tokens, passwords, private dataset links, or private model credentials.
- Use `.env.example` only as a template. Replace placeholders locally; do not put real values in the template.
- Treat public tunnel URLs and model-sharing links as public if you post them in issues, discussions, or commits.
- Before sharing a generated notebook, inspect its configuration and remove any private dataset paths or credentials.
- If a credential is accidentally committed, revoke or rotate it. Removing it in a later commit does not make the exposed credential safe again.
