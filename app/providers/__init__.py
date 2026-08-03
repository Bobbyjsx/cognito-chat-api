"""Provider abstraction layer.

The application talks to AI providers exclusively through the interfaces in
``app.providers.base``. SDK-specific logic (e.g. the Google GenAI SDK) is
encapsulated inside concrete providers under this package.
"""
