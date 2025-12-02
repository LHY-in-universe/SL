"""
Gradio theme configurations for Split Learning UI
"""

import gradio as gr


def get_theme(variant: str = "default") -> gr.Theme:
    """
    Get a Gradio theme for Split Learning UI

    Args:
        variant: Theme variant ("default", "dark", "light")

    Returns:
        Gradio theme object
    """
    if variant == "dark":
        return gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
            neutral_hue="slate",
        ).set(
            body_background_fill="#0f1419",
            body_background_fill_dark="#0f1419",
            button_primary_background_fill="#1e88e5",
            button_primary_background_fill_hover="#1976d2",
        )

    elif variant == "light":
        return gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="gray",
            neutral_hue="gray",
        ).set(
            body_background_fill="#ffffff",
            body_background_fill_dark="#f5f5f5",
        )

    else:  # default
        return gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
            neutral_hue="slate",
        )


# CSS customization for monospace fonts in specific components
DEFAULT_CSS = """
.status-box {
    font-family: monospace;
    font-size: 14px;
}

.stats-box {
    font-family: monospace;
    font-size: 12px;
}

.output-box {
    font-family: monospace;
    font-size: 14px;
    line-height: 1.5;
}
"""
