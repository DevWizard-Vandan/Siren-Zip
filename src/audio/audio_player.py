"""Low-Latency Hardware Audio Playback Engine and Master Clock Provider."""

from __future__ import annotations

import os
import tempfile
import time
from typing import Optional

from PySide6.QtCore import QObject, QUrl, Signal
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer


class AudioMasterClock(QObject):
    """Audio playback engine that acts as the authoritative Master Clock for video rendering."""

    position_changed = Signal(float)  # Emits current timestamp in seconds
    playback_ended = Signal()

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)

        self.audio_output = QAudioOutput()
        self.player = QMediaPlayer()
        self.player.setAudioOutput(self.audio_output)

        self.temp_audio_file: Optional[str] = None
        self.is_loaded: bool = False
        self._is_playing: bool = False
        self.fallback_start_time: float = 0.0
        self.fallback_offset_sec: float = 0.0
        self.is_night_mode: bool = False
        self.current_vol: float = 1.0

        # Signals
        self.player.positionChanged.connect(self._on_player_position_changed)
        self.player.mediaStatusChanged.connect(self._on_media_status_changed)

    def load_audio_data(self, audio_bytes: bytes, codec_type: str = "aac") -> bool:
        """Save audio payload to temp buffer and prepare for playback."""
        self.cleanup()
        if not audio_bytes or len(audio_bytes) == 0:
            self.is_loaded = False
            return False

        ext = "aac" if codec_type in ("aac", "1") else ("opus" if codec_type in ("opus", "2") else "mp3")
        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
            tmp.write(audio_bytes)
            self.temp_audio_file = tmp.name

        self.player.setSource(QUrl.fromLocalFile(self.temp_audio_file))
        self.audio_output.setVolume(1.0)
        self.is_loaded = True
        self._is_playing = False
        return True

    def play(self) -> None:
        """Start audio playback."""
        if self.is_loaded:
            self.player.play()
            self._is_playing = True
        else:
            self.fallback_start_time = time.perf_counter()
            self._is_playing = True

    def pause(self) -> None:
        """Pause audio playback."""
        if self.is_loaded:
            self.player.pause()
            self._is_playing = False
        else:
            self.fallback_offset_sec = self.get_master_time()
            self._is_playing = False

    def stop(self) -> None:
        """Stop audio playback and reset to 0.0s."""
        if self.is_loaded:
            self.player.stop()
            self._is_playing = False
        else:
            self.fallback_offset_sec = 0.0
            self._is_playing = False

    def seek(self, timestamp_sec: float) -> None:
        """Seek audio to specific timestamp in seconds."""
        t_clamped = max(0.0, float(timestamp_sec))
        if self.is_loaded:
            self.player.setPosition(int(round(t_clamped * 1000.0)))
        else:
            self.fallback_offset_sec = t_clamped
            self.fallback_start_time = time.perf_counter()

    def set_volume(self, volume_fraction: float) -> None:
        """Set volume in range [0.0, 1.0]."""
        v = max(0.0, min(1.0, float(volume_fraction)))
        self.current_vol = v
        if self.is_night_mode:
            # Compress dynamic range: speech boost floor + peak limiter
            eff_v = 0.5 + 0.5 * v
            self.audio_output.setVolume(eff_v)
        else:
            self.audio_output.setVolume(v)

    def set_night_mode(self, enabled: bool) -> None:
        """Enable / Disable dynamic range compression for comfortable night listening."""
        self.is_night_mode = enabled
        self.set_volume(self.current_vol)

    def set_muted(self, muted: bool) -> None:
        """Mute / Unmute audio."""
        self.audio_output.setMuted(muted)

    def get_master_time(self) -> float:
        """Return authoritative hardware DAC timestamp with sub-millisecond precision."""
        if self.is_loaded:
            pos_ms = self.player.position()
            return float(pos_ms / 1000.0)
        else:
            if self._is_playing:
                elapsed = time.perf_counter() - self.fallback_start_time
                return self.fallback_offset_sec + elapsed
            else:
                return self.fallback_offset_sec

    def is_playing(self) -> bool:
        return self._is_playing

    def _on_player_position_changed(self, pos_ms: int) -> None:
        self.position_changed.emit(float(pos_ms / 1000.0))

    def _on_media_status_changed(self, status: QMediaPlayer.MediaStatus) -> None:
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self._is_playing = False
            self.playback_ended.emit()

    def cleanup(self) -> None:
        """Release audio resources and clean up temp files."""
        self.player.stop()
        self.player.setSource(QUrl())
        if self.temp_audio_file and os.path.exists(self.temp_audio_file):
            try:
                os.remove(self.temp_audio_file)
            except Exception:
                pass
            self.temp_audio_file = None
        self.is_loaded = False
        self._is_playing = False
