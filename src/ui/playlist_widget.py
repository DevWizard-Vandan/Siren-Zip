"""VLC-Style Drag-and-Drop Playlist and Queue Dock Widget."""

from __future__ import annotations

import os
from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class PlaylistWidget(QDockWidget):
    """Playlist dock widget allowing file queues, next/previous tracks, and shuffle."""

    file_selected = Signal(str)  # Emits selected file path

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("📋 Playlist / File Queue", parent)
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)

        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # File List
        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet(
            """
            QListWidget {
                background-color: #161b22;
                border: 1px solid #30363d;
                border-radius: 6px;
                color: #e6edf3;
                font-size: 12px;
            }
            QListWidget::item {
                padding: 6px;
                border-bottom: 1px solid #21262d;
            }
            QListWidget::item:selected {
                background-color: #238636;
                color: #ffffff;
                font-weight: bold;
            }
            """
        )
        self.list_widget.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self.list_widget)

        # Action Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(6)

        self.btn_add = QPushButton("➕ Add File")
        self.btn_add.clicked.connect(self._on_add_file)
        btn_layout.addWidget(self.btn_add)

        self.btn_remove = QPushButton("➖ Remove")
        self.btn_remove.clicked.connect(self._on_remove_file)
        btn_layout.addWidget(self.btn_remove)

        self.btn_clear = QPushButton("🗑️ Clear")
        self.btn_clear.clicked.connect(self.clear_playlist)
        btn_layout.addWidget(self.btn_clear)

        layout.addLayout(btn_layout)
        self.setWidget(container)

        self.playlist_paths: List[str] = []
        self.current_index: int = -1

    def add_file(self, filepath: str) -> None:
        """Add file to playlist queue."""
        if not os.path.exists(filepath):
            return

        self.playlist_paths.append(filepath)
        name = os.path.basename(filepath)
        size_mb = os.path.getsize(filepath) / (1024.0 * 1024.0)
        item = QListWidgetItem(f"🎬 {name} ({size_mb:.1f} MB)")
        self.list_widget.addItem(item)

        if self.current_index == -1:
            self.current_index = 0
            self.list_widget.setCurrentRow(0)

    def _on_add_file(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Add Media to Playlist", "", "Media Files (*.neura *.mp4 *.mkv);;All Files (*)"
        )
        for f in files:
            self.add_file(f)

    def _on_remove_file(self) -> None:
        row = self.list_widget.currentRow()
        if 0 <= row < len(self.playlist_paths):
            del self.playlist_paths[row]
            self.list_widget.takeItem(row)

    def clear_playlist(self) -> None:
        self.playlist_paths.clear()
        self.list_widget.clear()
        self.current_index = -1

    def _on_item_double_clicked(self, item: QListWidgetItem) -> None:
        row = self.list_widget.row(item)
        if 0 <= row < len(self.playlist_paths):
            self.current_index = row
            self.file_selected.emit(self.playlist_paths[row])

    def get_next_file(self) -> Optional[str]:
        if not self.playlist_paths:
            return None
        self.current_index = (self.current_index + 1) % len(self.playlist_paths)
        self.list_widget.setCurrentRow(self.current_index)
        return self.playlist_paths[self.current_index]

    def get_previous_file(self) -> Optional[str]:
        if not self.playlist_paths:
            return None
        self.current_index = (self.current_index - 1) % len(self.playlist_paths)
        self.list_widget.setCurrentRow(self.current_index)
        return self.playlist_paths[self.current_index]
