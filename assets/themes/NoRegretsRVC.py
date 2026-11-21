from __future__ import annotations
from typing import Iterable
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes


class NoRegretsRVCTheme(Base):
    """Locked deep crimson-red Gradio theme for NoRegretsRVC."""

    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.neutral,
        secondary_hue: colors.Color | str = colors.neutral,
        neutral_hue: colors.Color | str = colors.neutral,
        spacing_size: sizes.Size | str = sizes.spacing_md,
        radius_size: sizes.Size | str = sizes.radius_lg,
        text_size: sizes.Size | str = sizes.text_md,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Plus Jakarta Sans"),
            "ui-sans-serif",
            "system-ui",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("JetBrains Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        self.name = "NoRegretsRVC"
        super().set(
            body_background_fill="#0D0405",
            body_background_fill_dark="#0D0405",
            background_fill_primary="#160709",
            background_fill_primary_dark="#160709",
            background_fill_secondary="#200B0E",
            background_fill_secondary_dark="#200B0E",
            block_background_fill="#160709",
            block_background_fill_dark="#160709",
            block_border_color="#47161D",
            block_border_color_dark="#47161D",
            block_border_width="1px",
            block_border_width_dark="1px",
            block_label_background_fill="#200B0E",
            block_label_background_fill_dark="#200B0E",
            block_label_text_color="#DFBFC3",
            block_label_text_color_dark="#DFBFC3",
            block_title_text_color="#FDF7F7",
            block_title_text_color_dark="#FDF7F7",
            body_text_color="#FDF7F7",
            body_text_color_dark="#FDF7F7",
            body_text_color_subdued="#AD8288",
            body_text_color_subdued_dark="#AD8288",
            input_background_fill="#100506",
            input_background_fill_dark="#100506",
            input_border_color="#47161D",
            input_border_color_dark="#47161D",
            input_border_color_focus="#E51923",
            input_border_color_focus_dark="#E51923",
            button_primary_background_fill="#E51923",
            button_primary_background_fill_dark="#E51923",
            button_primary_background_fill_hover="#FF2B35",
            button_primary_background_fill_hover_dark="#FF2B35",
            button_primary_text_color="#FFFFFF",
            button_primary_text_color_dark="#FFFFFF",
            button_primary_border_color="#9F1018",
            button_primary_border_color_dark="#9F1018",
            button_secondary_background_fill="#200B0E",
            button_secondary_background_fill_dark="#200B0E",
            button_secondary_background_fill_hover="#331116",
            button_secondary_background_fill_hover_dark="#331116",
            button_secondary_text_color="#FDF7F7",
            button_secondary_text_color_dark="#FDF7F7",
            button_secondary_border_color="#47161D",
            button_secondary_border_color_dark="#47161D",
            slider_color="#E51923",
            slider_color_dark="#E51923",
            checkbox_background_color_selected="#E51923",
            checkbox_background_color_selected_dark="#E51923",
            checkbox_border_color_focus="#E51923",
            checkbox_border_color_focus_dark="#E51923",
            color_accent="#E51923",
            color_accent_soft="#5E0E14",
        )
