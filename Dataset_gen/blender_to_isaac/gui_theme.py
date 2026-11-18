"""Frozen copy of tomo_vla_train_gui/gui_theme.py (2026-10-29), so the dataset generator window looks like the TOMO
Console without depending on the training folder."""
"""TOMO VLA Train's look = the TOMO Console's (tomo_console_gui/theme.py + bt_visualizer.STUDIO_STYLE on the robot
computer, copied 2026-10-24): the same colours, fonts, header, navigation, cards, log dock and red footer button.
Only the widgets the Console does not have (progress bars, tables, the architecture list, status pills) are added,
in the same colours."""

STYLE = """
QWidget {
    color: #182338;
    font-family: "Inter", "Segoe UI", "Ubuntu Sans", "Ubuntu", sans-serif;
    font-size: 13px;
}
QWidget#mainWindow, #centralStack, #consoleContent, #page { background: #eef2f7; }
#consoleHeader { background: #0d1625; border-bottom: 1px solid #2a3548; }
#consoleBrand { color: #f5f8fc; font-size: 17px; font-weight: 700; letter-spacing: 1px; }
#consoleSubtitle { color: #8f9db1; font-size: 11px; }
#currentPage {
    color: #dce5f2; font-size: 13px; font-weight: 600; padding: 7px 12px;
    background: #182438; border: 1px solid #2d3b50; border-radius: 6px;
}
#consoleMode {
    color: #72d8b5; background: #14372f; border-radius: 6px; padding: 6px 9px;
    font-size: 10px; font-weight: 700;
}
#consoleMode[state="busy"] { color: #ffe0a0; background: #4a3515; }
#consoleMode[state="error"] { color: #ff9aa7; background: #4a2028; }
#sidePanel { background: #131d2c; border-right: 1px solid #29364a; }
#sideLogoFrame { background: #182438; border: 1px solid #29384d; border-radius: 9px; }
#sideProduct { color: #eef4fb; font-size: 13px; font-weight: 700; }
#sideCaption { color: #7f8da2; font-size: 10px; }
#navigationLabel { color: #75849a; font-size: 10px; font-weight: 700; padding: 7px 4px 3px 4px; }
#navigationList { color: #aab6c8; background: transparent; border: none; outline: none; }
#navigationList::item { min-height: 38px; padding: 3px 11px; margin: 2px 0; border-radius: 6px; }
#navigationList::item:hover { color: white; background: #1c293b; }
#navigationList::item:selected { color: white; background: #2b4f98; border-left: 3px solid #79a0ff; }
#navigationList::item:disabled { color: #4b586c; }
#sideFooter { color: #6f7e93; font-size: 10px; padding: 7px 3px; }
#bottomPanel { background: #101927; border-top: 1px solid #29364a; }
#safetyLabel { color: #8492a6; font-size: 10px; font-weight: 700; }
#safetyHint { color: #a9b5c6; font-size: 11px; }
#emergencyStop {
    min-width: 72px; min-height: 48px; color: white; background: #c93f50;
    border: 3px solid #f07582; border-radius: 12px; font-size: 12px; font-weight: 800;
}
#emergencyStop:hover { background: #df495c; border-color: #ff9aa4; }
#emergencyStop:pressed { background: #a92f3f; }
#emergencyStop:disabled { color: #7d6a70; background: #2a2230; border-color: #3d3040; }
#navButton {
    min-width: 92px; min-height: 36px; color: #d8e0ec; background: #192538;
    border: 1px solid #344257; border-radius: 6px; font-weight: 600;
}
#navButton:hover { color: white; background: #223149; }
#navButton:disabled { color: #4f5b6d; background: #141e2d; border-color: #222e40; }
#nextButton {
    min-width: 92px; min-height: 36px; color: white; background: #2e63e6;
    border: 1px solid #2e63e6; border-radius: 6px; font-weight: 700;
}
#nextButton:hover { background: #2858ce; }
#nextButton:disabled { color: #4f5b6d; background: #141e2d; border-color: #222e40; }
#pageTitle { color: #223149; font-size: 19px; font-weight: 700; }
#pageSubtitle { color: #748196; font-size: 11px; }
#panelCard, QGroupBox { background: white; border: 1px solid #d7dee8; border-radius: 8px; }
QGroupBox { margin-top: 11px; padding: 14px 10px 10px 10px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #45546a; }
QGroupBox QLabel { font-weight: 400; }
QPushButton {
    min-height: 30px; padding: 0 12px; border-radius: 6px; border: 1px solid #cfd7e3; background: white;
}
QPushButton:hover { border-color: #8298bd; background: #f6f8fc; }
#primaryButton { color: white; background: #2e63e6; border-color: #2e63e6; font-weight: 600; }
#primaryButton:hover { background: #2858ce; }
#runButton { color: white; background: #268d69; border-color: #268d69; font-weight: 600; }
#runButton:hover { background: #217a5b; }
#stopButton { color: white; background: #cf4657; border-color: #cf4657; font-weight: 600; }
QPushButton:disabled, QToolButton:disabled { color: #9da8b7; background: #e4e9f0; border-color: #d5dce6; }
#primaryButton:disabled, #runButton:disabled, #stopButton:disabled {
    color: #9da8b7; background: #e4e9f0; border-color: #d5dce6;
}
QLineEdit, QComboBox {
    min-height: 30px; padding: 0 9px; background: white; border: 1px solid #ccd5e1; border-radius: 6px;
}
QLineEdit:focus, QComboBox:focus { border-color: #5e82e8; }
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled { color: #8a96a8; background: #f2f5f9; }
QSpinBox {
    min-height: 30px; padding: 0 7px; color: #28364b; background: white; border: 1px solid #ccd5e1; border-radius: 6px;
}
QSpinBox:focus { border-color: #5e82e8; }
QTextEdit, QPlainTextEdit, QTextBrowser {
    color: #28364b; background: white; border: 1px solid #d2dae5; border-radius: 6px;
    selection-background-color: #6689e6;
}
QCheckBox { spacing: 7px; }
QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #b9c4d3; border-radius: 4px; background: white; }
QCheckBox::indicator:checked { background: #2e63e6; border-color: #2e63e6; }
QProgressBar {
    min-height: 10px; max-height: 10px; background: #dfe5ee; border: none; border-radius: 5px;
}
QProgressBar::chunk { background: #2e63e6; border-radius: 5px; }
QProgressBar[state="ready"]::chunk { background: #268d69; }
QProgressBar[state="error"]::chunk { background: #cf4657; }
#statusPill {
    color: #758399; background: #f2f5f9; border-radius: 5px; padding: 7px 9px; font-size: 12px; font-weight: 600;
}
#statusPill[state="ready"] { color: #26785f; background: #e2f2ec; }
#statusPill[state="busy"] { color: #8a641f; background: #f8ebc9; }
#statusPill[state="error"] { color: #aa3544; background: #f8dfe3; }
#fieldLabel { color: #596a81; font-size: 11px; font-weight: 700; }
#detail { color: #536176; }
#archList {
    color: #334158; background: white; border: 1px solid #d7dee8; border-radius: 8px; outline: none; padding: 4px;
}
#archList::item { min-height: 54px; padding: 4px 10px; margin: 2px; border-radius: 6px; }
#archList::item:hover { background: #f3f6fa; }
#archList::item:selected { color: white; background: #2b4f98; border-left: 3px solid #79a0ff; }
QTableWidget {
    color: #334158; background: white; alternate-background-color: #f7f9fc;
    border: 1px solid #d7dee8; border-radius: 7px; gridline-color: #e8ecf2;
}
QHeaderView::section {
    color: #536176; background: #f3f6fa; border: none; border-bottom: 1px solid #d7dee8;
    padding: 6px 8px; font-weight: 600;
}
#logDock { background: #f8fafc; border-top: 1px solid #d7dee8; }
#logDockTitle { color: #536176; font-size: 11px; font-weight: 700; }
#logView { font-family: "JetBrains Mono", "DejaVu Sans Mono", monospace; font-size: 12px; }
QScrollBar:vertical { width: 10px; background: transparent; margin: 2px; }
QScrollBar::handle:vertical { min-height: 28px; background: #bec8d6; border-radius: 4px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QSplitter::handle { background: #d7dee8; }
"""
