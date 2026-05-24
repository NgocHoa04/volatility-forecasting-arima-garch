"""Plotting utilities and configurations."""

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional

from ..config import (
    FIGURE_DPI,
    FIGURE_FACECOLOR,
    PLOT_STYLE,
    FONT_SIZE,
    FIGURES_DIR,
)


def setup_plot_style() -> None:
    """Configure global matplotlib and seaborn styles."""
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "figure.dpi": FIGURE_DPI,
        "figure.facecolor": FIGURE_FACECOLOR,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": FONT_SIZE,
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
    })


def save_figure(
    fig: plt.Figure,
    filename: str,
    output_dir: Optional[Path] = None,
    dpi: int = FIGURE_DPI,
    tight_layout: bool = True,
    **kwargs
) -> Path:
    """
    Save figure to disk.
    
    Parameters
    ----------
    fig : plt.Figure
        Figure object to save
    filename : str
        Filename (with or without extension)
    output_dir : Path, optional
        Output directory. If None, uses FIGURES_DIR
    dpi : int, optional
        DPI for saved figure
    tight_layout : bool, optional
        Apply tight_layout before saving
    **kwargs
        Additional arguments passed to fig.savefig()
        
    Returns
    -------
    Path
        Path where figure was saved
    """
    if output_dir is None:
        output_dir = FIGURES_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure filename has extension
    if not any(filename.endswith(ext) for ext in [".png", ".pdf", ".jpg", ".eps"]):
        filename = f"{filename}.png"
    
    filepath = output_dir / filename
    
    if tight_layout:
        fig.tight_layout()
    
    fig.savefig(filepath, dpi=dpi, **kwargs)
    
    return filepath
