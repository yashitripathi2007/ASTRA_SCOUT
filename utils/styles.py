"""
Centralized Styling & Responsive Layout Engine for ASTRA-SCOUT.
Provides unified typography, margin rhythm, line-height control,
and metric card padding to eliminate heading/text overlap across all viewports.
"""

import streamlit as st


def apply_shared_styles() -> None:
    """
    Injects global CSS rules into the active Streamlit page.
    Resolves:
    1. Title overlapping subtitle/caption text.
    2. Section headings (h3, h4) overlapping wrapped metric cards.
    3. Tight line-heights and collapsed margins on laptop/narrow viewports.
    """
    st.markdown("""
    <style>
    /* =========================================================================
       1. Global Page Headings & Typography Rhythm
       ========================================================================= */
    h1, [data-testid="stHeadingWithActionElements"] h1 {
        margin-top: 0.25rem !important;
        margin-bottom: 0.65rem !important;
        line-height: 1.35 !important;
        padding-bottom: 0.2rem !important;
        letter-spacing: -0.01em !important;
    }

    h2, [data-testid="stHeadingWithActionElements"] h2 {
        margin-top: 1.25rem !important;
        margin-bottom: 0.60rem !important;
        line-height: 1.35 !important;
        clear: both !important;
    }

    h3, [data-testid="stHeadingWithActionElements"] h3 {
        margin-top: 1.45rem !important;
        margin-bottom: 0.70rem !important;
        line-height: 1.35 !important;
        clear: both !important;
    }

    h4, [data-testid="stHeadingWithActionElements"] h4 {
        margin-top: 1.35rem !important;
        margin-bottom: 0.60rem !important;
        line-height: 1.35 !important;
        clear: both !important;
    }

    /* =========================================================================
       2. Captions & Subtitles (Prevents Title Descenders from Overlapping)
       ========================================================================= */
    [data-testid="stCaptionContainer"], .stCaption {
        margin-top: 0.35rem !important;
        margin-bottom: 1.15rem !important;
        line-height: 1.50 !important;
        font-size: 0.95rem !important;
        display: block !important;
    }

    /* Hero header subtitle in app.py */
    .hero-subtitle {
        font-size: 1.15rem !important;
        font-weight: 500 !important;
        margin-top: 0.35rem !important;
        margin-bottom: 0.35rem !important;
        line-height: 1.40 !important;
        color: #E2E8F0 !important;
    }

    /* =========================================================================
       3. Metric Cards (Structured Card Box + Guaranteed Bottom Margin)
       Prevents downstream headings or charts from colliding with wrapped metrics.
       ========================================================================= */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.035) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 8px !important;
        padding: 0.70rem 0.85rem !important;
        margin-bottom: 1.35rem !important;
        min-height: 98px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        box-sizing: border-box !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        line-height: 1.30 !important;
        margin-bottom: 0.25rem !important;
        white-space: normal !important;
        word-break: break-word !important;
        color: #94A3B8 !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.45rem !important;
        line-height: 1.25 !important;
        font-weight: 700 !important;
        word-break: break-word !important;
    }

    [data-testid="stMetricDelta"] {
        font-size: 0.80rem !important;
        line-height: 1.20 !important;
        margin-top: 0.20rem !important;
    }

    /* =========================================================================
       4. Column Grids & Vertical Alignment
       ========================================================================= */
    [data-testid="column"] {
        margin-bottom: 0.50rem !important;
    }

    /* =========================================================================
       5. Tabs, Alerts & Separator Rhythm
       ========================================================================= */
    .stTabs [data-baseweb="tab-list"] {
        margin-top: 0.75rem !important;
        margin-bottom: 1.0rem !important;
        gap: 0.5rem !important;
    }

    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 0.75rem !important;
    }

    [data-testid="stAlert"] {
        margin-top: 0.85rem !important;
        margin-bottom: 1.25rem !important;
        line-height: 1.45 !important;
    }

    hr {
        margin-top: 1.5rem !important;
        margin-bottom: 1.5rem !important;
        border: none !important;
        border-top: 1px solid rgba(255, 255, 255, 0.12) !important;
    }

    /* =========================================================================
       6. Responsive Scaling for Laptop Screens and Narrow Windows
       ========================================================================= */
    @media (max-width: 1200px) {
        h1, [data-testid="stHeadingWithActionElements"] h1 {
            font-size: 1.85rem !important;
            line-height: 1.30 !important;
            margin-bottom: 0.50rem !important;
        }
        h2, [data-testid="stHeadingWithActionElements"] h2 {
            font-size: 1.40rem !important;
        }
        h3, [data-testid="stHeadingWithActionElements"] h3 {
            font-size: 1.20rem !important;
            margin-top: 1.25rem !important;
        }
        h4, [data-testid="stHeadingWithActionElements"] h4 {
            font-size: 1.05rem !important;
            margin-top: 1.15rem !important;
        }
        [data-testid="stCaptionContainer"], .stCaption {
            font-size: 0.90rem !important;
            margin-bottom: 0.95rem !important;
        }
        [data-testid="stMetric"] {
            min-height: 88px !important;
            padding: 0.55rem 0.65rem !important;
            margin-bottom: 1.15rem !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.25rem !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.80rem !important;
        }
    }

    @media (max-width: 768px) {
        h1, [data-testid="stHeadingWithActionElements"] h1 {
            font-size: 1.60rem !important;
        }
        [data-testid="stMetric"] {
            margin-bottom: 0.85rem !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)
