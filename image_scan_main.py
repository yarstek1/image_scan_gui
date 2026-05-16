import os
import sys
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.fft import fft, fftfreq, fftshift
from skimage import io as skio

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QColor, QIcon, QPixmap
    from PyQt6.QtWidgets import (
        QAbstractItemView,
        QApplication,
        QButtonGroup,
        QComboBox,
        QDialog,
        QColorDialog,
        QFileDialog,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSizePolicy,
        QSlider,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )

    os.environ.setdefault("QT_API", "pyqt6")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

    CHECKED = Qt.CheckState.Checked
    UNCHECKED = Qt.CheckState.Unchecked
    ORIENTATION_VERTICAL = Qt.Orientation.Vertical
    POLICY_FIXED = QSizePolicy.Policy.Fixed
    POLICY_EXPANDING = QSizePolicy.Policy.Expanding
    ITEM_USER_CHECKABLE = Qt.ItemFlag.ItemIsUserCheckable
    ITEM_EDITABLE = Qt.ItemFlag.ItemIsEditable
    ITEM_SELECTABLE = Qt.ItemFlag.ItemIsSelectable
    DIALOG_ACCEPTED = QDialog.DialogCode.Accepted
    COLOR_DIALOG_DONT_USE_NATIVE = QColorDialog.ColorDialogOption.DontUseNativeDialog
    COLOR_DIALOG_SHOW_ALPHA = QColorDialog.ColorDialogOption.ShowAlphaChannel
except ImportError:
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QColor, QIcon, QPixmap
    from PyQt5.QtWidgets import (
        QAbstractItemView,
        QApplication,
        QButtonGroup,
        QComboBox,
        QDialog,
        QColorDialog,
        QFileDialog,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSizePolicy,
        QSlider,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )

    os.environ.setdefault("QT_API", "pyqt5")
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

    CHECKED = Qt.Checked
    UNCHECKED = Qt.Unchecked
    ORIENTATION_VERTICAL = Qt.Vertical
    POLICY_FIXED = QSizePolicy.Fixed
    POLICY_EXPANDING = QSizePolicy.Expanding
    ITEM_USER_CHECKABLE = Qt.ItemIsUserCheckable
    ITEM_EDITABLE = Qt.ItemIsEditable
    ITEM_SELECTABLE = Qt.ItemIsSelectable
    DIALOG_ACCEPTED = QDialog.Accepted
    COLOR_DIALOG_DONT_USE_NATIVE = QColorDialog.DontUseNativeDialog
    COLOR_DIALOG_SHOW_ALPHA = QColorDialog.ShowAlphaChannel

from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector


FIXED_SIGNAL_COLORS = [
    QColor("#e53935"),  # красный
    QColor("#1e88e5"),  # синий
    QColor("#43a047"),  # зеленый
    QColor("#8e24aa"),  # фиолетовый
    QColor("#fb8c00"),  # оранжевый
    QColor("#6d4c41"),  # коричневый
    QColor("#00acc1"),  # голубой
]

STANDARD_PALETTE_COLORS = [
    QColor("#e53935"), QColor("#1e88e5"), QColor("#43a047"), QColor("#8e24aa"), QColor("#fb8c00"),
    QColor("#6d4c41"), QColor("#00acc1"), QColor("#fdd835"), QColor("#3949ab"), QColor("#00897b"),
    QColor("#c2185b"), QColor("#7cb342"), QColor("#5e35b1"), QColor("#546e7a"), QColor("#f4511e"),
    QColor("#039be5"), QColor("#8d6e63"), QColor("#d81b60"), QColor("#9e9d24"), QColor("#5d4037"),
]


@dataclass
class SignalItem:
    name: str
    color: QColor
    values: np.ndarray


def build_time_axis(t_half: float, n_points: int) -> np.ndarray:
    return np.linspace(-t_half, t_half, n_points, endpoint=True)


def reconstruct_signal_from_image(image: np.ndarray, pixcolor: np.ndarray) -> np.ndarray:
    if image.ndim < 3 or image.shape[2] < 3:
        raise ValueError("Изображение должно быть цветным (RGB/RGBA).")

    rgb = image[:, :, :3]
    mask = np.all(rgb == pixcolor.reshape(1, 1, 3), axis=2)

    h, w = mask.shape
    rows = np.arange(h, dtype=float)[:, None]
    row_values = np.where(mask, rows, np.nan)
    mean_rows = np.nanmean(row_values, axis=0)

    sig = (h - np.ceil(mean_rows) - np.floor(h / 2.0)) / h * 2.0
    sig[np.isnan(sig)] = 0.0
    return sig.astype(float)


def downsample_signal(sig: np.ndarray, n_points: int) -> np.ndarray:
    n_image = sig.shape[0]
    step = n_image // n_points
    if step < 1:
        raise ValueError("Недостаточная ширина изображения для указанного N.")

    idx = np.arange(n_points) * step
    idx = np.clip(idx, 0, n_image - 1)
    return sig[idx]


def random_qcolor() -> QColor:
    return QColor.fromHsv(random.randint(0, 359), 220, 220)


def default_signal_color(index: int) -> QColor:
    if 0 <= index < len(FIXED_SIGNAL_COLORS):
        return QColor(FIXED_SIGNAL_COLORS[index])
    return random_qcolor()


class ColorSelectDialog(QDialog):
    def __init__(self, colors: np.ndarray, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Выбор цвета")
        self.setModal(True)
        self.selected_color: Optional[np.ndarray] = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Выберите цвет линии сигнала на исходном изображении"))

        self.combo = QComboBox(self)
        for c in colors:
            qcolor = QColor(int(c[0]), int(c[1]), int(c[2]))
            pix = QPixmap(60, 18)
            pix.fill(qcolor)
            self.combo.addItem(QIcon(pix), f"RGB({c[0]}, {c[1]}, {c[2]})", c)
        layout.addWidget(self.combo)

        btn_ok = QPushButton("Подтвердить")
        btn_ok.clicked.connect(self.on_confirm)
        layout.addWidget(btn_ok)

    def on_confirm(self):
        data = self.combo.currentData()
        self.selected_color = np.array(data, dtype=np.uint8)
        self.accept()


class PreviewDialog(QDialog):
    def __init__(self, x: np.ndarray, y: np.ndarray, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Предпросмотр восстановленного сигнала")
        self.setModal(True)

        layout = QVBoxLayout(self)

        fig = Figure(figsize=(7, 3))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.plot(x, y, color="tab:blue", linewidth=1.5)
        ax.set_title("Восстановленный сигнал")
        ax.grid(True)

        layout.addWidget(canvas)

        btn_ok = QPushButton("Подтвердить")
        btn_ok.clicked.connect(self.accept)
        layout.addWidget(btn_ok)


class SignalPaletteDialog(QColorDialog):
    def __init__(self, initial_color: QColor, parent: Optional[QWidget] = None):
        super().__init__(initial_color, parent)
        self.setWindowTitle("Выбор цвета сигнала")
        self.setOption(COLOR_DIALOG_DONT_USE_NATIVE, True)
        self.setOption(COLOR_DIALOG_SHOW_ALPHA, False)

        for idx, color in enumerate(STANDARD_PALETTE_COLORS):
            self.setStandardColor(idx, color)


class MplView(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.figure = Figure(figsize=(5, 3))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

    def clear(self):
        self.ax.clear()
        self.canvas.draw_idle()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Сканирование и спектральный анализ")
        self.resize(1500, 900)

        self.signals: list[SignalItem] = []
        self.summed_signal: Optional[np.ndarray] = None

        self.current_edit_index: Optional[int] = None
        self.edit_values: Optional[np.ndarray] = None
        self.edit_values_baseline: Optional[np.ndarray] = None
        self.undo_stack: list[np.ndarray] = []
        self.is_table_refresh = False
        self.level_slider_active = False
        self.show_components_active = False

        self.spectrum_freq: Optional[np.ndarray] = None
        self.spectrum_amp: Optional[np.ndarray] = None
        self.spectrum_phase: Optional[np.ndarray] = None
        self.spectrum_real: Optional[np.ndarray] = None
        self.spectrum_imag: Optional[np.ndarray] = None

        self._build_ui()
        self._update_buttons_state()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        grid = QGridLayout(central)

        self.signals_box = self._build_signals_panel()
        self.edit_box = self._build_edit_panel()
        self.spectrum_box = self._build_spectrum_panel()
        self.sum_box = self._build_sum_panel()

        grid.addWidget(self.signals_box, 0, 0)
        grid.addWidget(self.edit_box, 0, 1)
        grid.addWidget(self.spectrum_box, 1, 0)
        grid.addWidget(self.sum_box, 1, 1)

        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

    def _build_signals_panel(self) -> QGroupBox:
        box = QGroupBox("Сигналы")
        root = QVBoxLayout(box)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Выбор", "Название", "Цвет", "Удалить", "Копировать"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.itemSelectionChanged.connect(self._update_buttons_state)
        self.table.itemChanged.connect(self._on_table_item_changed)
        self.table.cellClicked.connect(self._on_table_cell_clicked)
        root.addWidget(self.table)

        params_box = QGroupBox("Параметры развертки")
        params_layout = QVBoxLayout(params_box)

        row_t = QHBoxLayout()
        row_t.addWidget(QLabel("T/2, сек"))
        self.input_t_half = QLineEdit()
        self.input_t_half.textChanged.connect(self._on_params_changed)
        row_t.addWidget(self.input_t_half)
        params_layout.addLayout(row_t)

        row_n = QHBoxLayout()
        row_n.addWidget(QLabel("Количество отсчетов N"))
        self.input_n = QLineEdit()
        self.input_n.textChanged.connect(self._on_params_changed)
        row_n.addWidget(self.input_n)
        params_layout.addLayout(row_n)

        row_cmd = QHBoxLayout()
        self.btn_add = QPushButton("Добавить")
        self.btn_add.clicked.connect(self._add_signal_flow)
        row_cmd.addWidget(self.btn_add)

        self.btn_show = QPushButton("Показать")
        self.btn_show.clicked.connect(self._show_selected_signal)
        row_cmd.addWidget(self.btn_show)

        self.btn_show_sum = QPushButton("Показать сумму")
        self.btn_show_sum.clicked.connect(self._show_sum)
        row_cmd.addWidget(self.btn_show_sum)

        params_layout.addLayout(row_cmd)
        root.addWidget(params_box)

        return box

    def _build_edit_panel(self) -> QGroupBox:
        box = QGroupBox("Редактирование")
        root = QHBoxLayout(box)

        self.edit_plot = MplView()
        root.addWidget(self.edit_plot, 1)

        right = QVBoxLayout()

        self.btn_zero = QPushButton("Обнуление")
        self.btn_zero.setCheckable(True)
        self.btn_zero.clicked.connect(self._toggle_zero_mode)
        right.addWidget(self.btn_zero)

        self.btn_level = QPushButton("Уровень 0")
        self.btn_level.setCheckable(True)
        self.btn_level.clicked.connect(self._toggle_level_mode)
        right.addWidget(self.btn_level)

        self.btn_phase_shift = QPushButton("Редактировать сдвиг")
        self.btn_phase_shift.setCheckable(True)
        self.btn_phase_shift.clicked.connect(self._toggle_phase_shift)
        right.addWidget(self.btn_phase_shift)

        self.btn_undo = QPushButton("Отмена")
        self.btn_undo.clicked.connect(self._undo_last_edit)
        right.addWidget(self.btn_undo)

        self.btn_save = QPushButton("Сохранить")
        self.btn_save.clicked.connect(self._save_edit)
        right.addWidget(self.btn_save)

        self.level_slider = QSlider(ORIENTATION_VERTICAL)
        self.level_slider.setRange(-1000, 1000)
        self.level_slider.setValue(0)
        self.level_slider.hide()
        self.level_slider.sliderPressed.connect(self._on_level_slider_pressed)
        self.level_slider.sliderReleased.connect(self._on_level_slider_released)
        self.level_slider.valueChanged.connect(self._on_level_slider_changed)
        self.level_slider.setSizePolicy(POLICY_FIXED, POLICY_EXPANDING)
        right.addWidget(self.level_slider, 1)

        self.input_time_shift = QLineEdit()
        right.addWidget(self.input_time_shift)
        self.input_time_shift.hide()
        self.input_time_shift.textChanged.connect(self._on_phase_shift_changed)

        right.addStretch(1)
        root.addLayout(right)

        self.span_selector = SpanSelector(
            self.edit_plot.ax,
            self._on_span_selected,
            direction="horizontal",
            useblit=True,
            props=dict(alpha=0.25, facecolor="red"),
            interactive=True,
            drag_from_anywhere=True,
        )
        self.span_selector.set_active(False)

        return box

    def _build_sum_panel(self) -> QGroupBox:
        box = QGroupBox("Сумма")
        root = QVBoxLayout(box)

        self.sum_plot = MplView()
        root.addWidget(self.sum_plot)

        self.btn_show_components = QPushButton("Показать исходные сигналы")
        self.btn_show_components.setCheckable(True)
        self.btn_show_components.toggled.connect(self._toggle_show_components)
        root.addWidget(self.btn_show_components)

        return box

    def _build_spectrum_panel(self) -> QGroupBox:
        box = QGroupBox("Спектр")
        root = QVBoxLayout(box)

        self.btn_calc_spec = QPushButton("Рассчитать спектр")
        self.btn_calc_spec.clicked.connect(self._calculate_spectrum)
        root.addWidget(self.btn_calc_spec)

        modes = QHBoxLayout()
        self.btn_amp = QPushButton("Амплитуда")
        self.btn_phase = QPushButton("Фаза")
        self.btn_real = QPushButton("Действительная часть")
        self.btn_imag = QPushButton("Мнимая часть")

        for b in [self.btn_amp, self.btn_phase, self.btn_real, self.btn_imag]:
            b.setCheckable(True)
            modes.addWidget(b)

        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_group.addButton(self.btn_amp)
        self.mode_group.addButton(self.btn_phase)
        self.mode_group.addButton(self.btn_real)
        self.mode_group.addButton(self.btn_imag)

        self.btn_amp.clicked.connect(lambda: self._plot_spectrum_mode("amp"))
        self.btn_phase.clicked.connect(lambda: self._plot_spectrum_mode("phase"))
        self.btn_real.clicked.connect(lambda: self._plot_spectrum_mode("real"))
        self.btn_imag.clicked.connect(lambda: self._plot_spectrum_mode("imag"))

        root.addLayout(modes)

        self.spec_plot = MplView()
        root.addWidget(self.spec_plot)

        return box

    def _show_error(self, text: str):
        QMessageBox.warning(self, "Ошибка", text)

    def _on_params_changed(self):
        self._validate_params_ui()
        self._update_buttons_state()

    def _parse_params(self) -> tuple[Optional[float], Optional[int]]:
        t_half = None
        n_points = None

        try:
            val = float(self.input_t_half.text().strip())
            if val >= 0:
                t_half = val
        except Exception:
            pass

        try:
            val = int(self.input_n.text().strip())
            if val > 0:
                n_points = val
        except Exception:
            pass

        return t_half, n_points

    def _validate_params_ui(self):
        t_half, n_points = self._parse_params()

        self.input_t_half.setStyleSheet("" if t_half is not None else "border: 1px solid red;")
        self.input_n.setStyleSheet("" if n_points is not None else "border: 1px solid red;")

    def _update_buttons_state(self):
        t_half, n_points = self._parse_params()
        params_valid = t_half is not None and n_points is not None

        selected_row = self.table.currentRow()
        has_selected = selected_row >= 0 and selected_row < len(self.signals)

        checked_count = 0
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            if item is not None and item.checkState() == CHECKED:
                checked_count += 1

        self.btn_add.setEnabled(params_valid)
        self.btn_show.setEnabled(has_selected)
        self.btn_show_sum.setEnabled(len(self.signals) > 0 and checked_count > 0)

        has_edit = self.edit_values is not None
        self.btn_zero.setEnabled(has_edit)
        self.btn_level.setEnabled(has_edit)
        self.btn_phase_shift.setEnabled(has_edit)
        self.btn_undo.setEnabled(len(self.undo_stack) > 0)
        self.btn_save.setEnabled(has_edit and len(self.undo_stack) > 0)

        has_sum = self.summed_signal is not None
        if not has_sum:
            self.show_components_active = False
            self.btn_show_components.blockSignals(True)
            self.btn_show_components.setChecked(False)
            self.btn_show_components.blockSignals(False)

        self.btn_show_components.setEnabled(has_sum)
        self._update_show_components_button_style()
        self.btn_calc_spec.setEnabled(has_sum)

        spectrum_ready = self.spectrum_amp is not None
        for b in [self.btn_amp, self.btn_phase, self.btn_real, self.btn_imag]:
            b.setEnabled(spectrum_ready)

    def _update_show_components_button_style(self):
        if self.show_components_active and self.summed_signal is not None:
            self.btn_show_components.setStyleSheet("background-color: #87CEFA;")
        else:
            self.btn_show_components.setStyleSheet("")

    def _refresh_table(self):
        self.is_table_refresh = True
        prev_row = self.table.currentRow()

        self.table.setRowCount(len(self.signals))
        for i, sig in enumerate(self.signals):
            chk = self.table.item(i, 0)
            if chk is None:
                chk = QTableWidgetItem()
                chk.setFlags(chk.flags() | ITEM_USER_CHECKABLE)
                self.table.setItem(i, 0, chk)
            chk.setCheckState(chk.checkState() if chk.checkState() in (CHECKED, UNCHECKED) else UNCHECKED)

            name_item = QTableWidgetItem(sig.name)
            self.table.setItem(i, 1, name_item)

            color_item = QTableWidgetItem(" ")
            color_item.setFlags(color_item.flags() & ~(ITEM_EDITABLE | ITEM_SELECTABLE))
            color_item.setBackground(sig.color)
            self.table.setItem(i, 2, color_item)

            btn_delete = QPushButton("Удалить")
            btn_delete.clicked.connect(lambda _=False, row=i: self._delete_signal(row))
            self.table.setCellWidget(i, 3, btn_delete)

            btn_copy = QPushButton("Копировать")
            btn_copy.clicked.connect(lambda _=False, row=i: self._copy_signal(row))
            self.table.setCellWidget(i, 4, btn_copy)


        if 0 <= prev_row < len(self.signals):
            self.table.selectRow(prev_row)

        self.table.resizeColumnsToContents()
        self.is_table_refresh = False
        self._update_buttons_state()

    def _checked_rows(self) -> list[int]:
        rows = []
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 0)
            if it is not None and it.checkState() == CHECKED:
                rows.append(r)
        return rows

    def _on_table_item_changed(self, item: QTableWidgetItem):
        if self.is_table_refresh:
            return

        row = item.row()
        col = item.column()

        if 0 <= row < len(self.signals) and col == 1:
            self.signals[row].name = item.text().strip() or f"Сигнал {row + 1}"
        self._update_buttons_state()

    def _on_table_cell_clicked(self, row: int, col: int):
        if col != 2:
            return
        self._pick_signal_color(row)

    def _pick_signal_color(self, row: int):
        if row < 0 or row >= len(self.signals):
            return

        current = self.signals[row].color
        dialog = SignalPaletteDialog(current, self)
        if dialog.exec() != DIALOG_ACCEPTED:
            return

        selected = dialog.selectedColor()
        if not selected.isValid():
            return

        self.signals[row].color = selected
        self._refresh_table()

        if self.current_edit_index == row and self.edit_values is not None:
            self._plot_edit_signal()

        if self.summed_signal is not None:
            self._plot_sum()

    def _delete_signal(self, row: int):
        if row < 0 or row >= len(self.signals):
            return

        del self.signals[row]

        if self.current_edit_index == row:
            self.current_edit_index = None
            self.edit_values = None
            self.undo_stack.clear()
            self.edit_plot.clear()
        elif self.current_edit_index is not None and self.current_edit_index > row:
            self.current_edit_index -= 1

        self._refresh_table()
        self._update_buttons_state()

    def _copy_signal(self, row: int):
        if row < 0 or row >= len(self.signals):
            return
        
        copy_index = len(self.signals)
        name_copy = self.signals[row].name
        
        signal_copy = SignalItem(
            name=f"{name_copy} (Копия)",
            color=default_signal_color(copy_index),
            values=self.signals[row].values.copy(),
        )

        self.signals.append(signal_copy)
        self._refresh_table()

    def _extract_unique_colors(self, img: np.ndarray) -> np.ndarray:
        rgb = img[:, :, :3]
        flat = rgb.reshape(-1, 3)
        colors = np.unique(flat, axis=0)
        return colors

    def _add_signal_flow(self):
        t_half, n_points = self._parse_params()
        if t_half is None or n_points is None:
            self._show_error("Заполните корректно параметры T/2 и N.")
            return

        file_path, _ = QFileDialog.getOpenFileName(self, "Выбор png", "", "PNG files (*.png)")
        if not file_path:
            return

        if not file_path.lower().endswith(".png"):
            self._show_error("ошибка: некорректный формат")
            return

        try:
            image = skio.imread(file_path)
        except Exception as exc:
            self._show_error(f"Не удалось прочитать файл: {exc}")
            return

        if image.ndim < 3 or image.shape[2] < 3:
            self._show_error("Изображение должно быть цветным (RGB/RGBA).")
            return

        width = image.shape[1]
        if width < n_points:
            self._show_error("ошибка: слишком узкий файл")
            return

        colors = self._extract_unique_colors(image)
        color_dialog = ColorSelectDialog(colors, self)
        if color_dialog.exec() != DIALOG_ACCEPTED:
            return

        pixcolor = color_dialog.selected_color
        if pixcolor is None:
            return

        try:
            sig_full = reconstruct_signal_from_image(image, pixcolor)
            sig_resampled = downsample_signal(sig_full, n_points)
        except Exception as exc:
            self._show_error(f"Ошибка при сканировании: {exc}")
            return

        x = build_time_axis(t_half, n_points)
        preview = PreviewDialog(x, sig_resampled, self)
        if preview.exec() != DIALOG_ACCEPTED:
            return

        signal_index = len(self.signals)
        item = SignalItem(
            name=f"Сигнал {signal_index + 1}",
            color=default_signal_color(signal_index),
            values=sig_resampled.copy(),
        )
        self.signals.append(item)
        self._refresh_table()

    def _show_selected_signal(self):
        row = self.table.currentRow()
        if row < 0 or row >= len(self.signals):
            return

        self.current_edit_index = row
        self.edit_values = self.signals[row].values.copy()
        self.undo_stack.clear()

        self.btn_zero.setChecked(False)
        self.btn_level.setChecked(False)
        self.span_selector.set_active(False)
        self._clear_span_selection()
        self.level_slider.hide()
        self.level_slider.setValue(0)

        self._plot_edit_signal()
        self._update_buttons_state()

    def _plot_edit_signal(self):
        self.edit_plot.ax.clear()
        if self.current_edit_index is None or self.edit_values is None:
            self.edit_plot.canvas.draw_idle()
            return

        t_half, n_points = self._parse_params()
        if t_half is None or n_points is None:
            return

        x = build_time_axis(t_half, len(self.edit_values))
        color = self.signals[self.current_edit_index].color.name()

        self.edit_plot.ax.plot(x, self.edit_values, color=color, linewidth=1.6)
        self.edit_plot.ax.set_title(self.signals[self.current_edit_index].name)
        self.edit_plot.ax.grid(True)
        self.edit_plot.ax.set_xlabel("t, сек")
        self.edit_plot.canvas.draw_idle()

    def _clear_span_selection(self):
        if hasattr(self.span_selector, "clear"):
            self.span_selector.clear()
        elif hasattr(self.span_selector, "extents"):
            self.span_selector.extents = (0.0, 0.0)
        self.edit_plot.canvas.draw_idle()

    def _toggle_zero_mode(self):
        active = self.btn_zero.isChecked() and self.edit_values is not None
        if active:
            self.btn_level.setChecked(False)
            self.btn_phase_shift.setChecked(False)
            self.level_slider.hide()
            self.level_slider_active = False
            self.input_time_shift.hide()
        else:
            self._clear_span_selection()
        self.span_selector.set_active(active)


    def _toggle_level_mode(self):
        active = self.btn_level.isChecked() and self.edit_values is not None
        if active:
            self.btn_zero.setChecked(False)
            self.btn_phase_shift.setChecked(False)
            self.span_selector.set_active(False)
            self._clear_span_selection()
            self.level_slider_active = True
            self.level_slider.setValue(0)
            self.level_slider.show()
            self.edit_values_baseline = self.edit_values.copy() if self.edit_values is not None else None
            self.input_time_shift.hide()
        else:
            self.level_slider_active = False
            self.level_slider.hide()

    def _toggle_phase_shift(self):
        active = self.btn_phase_shift.isChecked() and self.edit_values is not None
        if active:
            self.btn_zero.setChecked(False)
            self.btn_level.setChecked(False)
            self.span_selector.set_active(False)
            self._clear_span_selection()
            self.level_slider_active = False
            self.level_slider.hide()
            self.input_time_shift.show()
            self.input_time_shift.setFocus()
        else:
            self.input_time_shift.hide()

    def _on_phase_shift_changed(self):
        self._validate_phase_shift()
        self._apply_phase_shift()

    def _parse_phase_shift(self) -> Optional[float]:
        text = self.input_time_shift.text().strip()
        if not text:
            return 0.0
        
        try:
            val = float(text)
            return val
        except Exception:
            return None
    
    def _validate_phase_shift(self):
        t_half, _ = self._parse_params()
        delta_t = self._parse_phase_shift()
        
        if t_half is None:
            self.input_time_shift.setStyleSheet("border: 1px solid red;")
        elif delta_t is None or delta_t < 0:
            self.input_time_shift.setStyleSheet("border: 1px solid red;")
        elif delta_t > t_half:
            self.input_time_shift.setStyleSheet("border: 1px solid red;")
        else:
            self.input_time_shift.setStyleSheet("")
    
    def _apply_phase_shift(self):
        delta_t = self._parse_phase_shift()
        t_half, _ = self._parse_params()
        
        if t_half is None:
            return

        # Пустая строка или 0 = нет сдвига
        if delta_t is None:
            return

        if delta_t < 0 or delta_t > t_half:
            return
        
        if self.edit_values is None:
            return

        n = len(self.edit_values)
        t = build_time_axis(t_half, n)
        delta_t_sample = abs(t[1] - t[0])
        n_steps = int(round(delta_t / delta_t_sample))
        
        signal_shifted = self.edit_values.copy()
        shifted_values = np.roll(signal_shifted, n_steps)
        
        # Сохраняем старое значение в undo stack
        if not np.array_equal(shifted_values, self.edit_values):
            self.undo_stack.append(self.edit_values.copy())
            self.edit_values = shifted_values
            self._plot_edit_signal()
            self._update_buttons_state()



    def _on_span_selected(self, x_min: float, x_max: float):
        if not self.btn_zero.isChecked() or self.edit_values is None:
            return

        t_half, _ = self._parse_params()
        if t_half is None:
            return

        x = build_time_axis(t_half, len(self.edit_values))
        i_min = int(np.searchsorted(x, min(x_min, x_max), side="left"))
        i_max = int(np.searchsorted(x, max(x_min, x_max), side="right"))

        i_min = max(0, min(i_min, len(self.edit_values)))
        i_max = max(0, min(i_max, len(self.edit_values)))
        if i_min >= i_max:
            return

        self.undo_stack.append(self.edit_values.copy())
        self.edit_values[i_min:i_max] = 0.0
        self._plot_edit_signal()
        self._update_buttons_state()

    def _on_level_slider_pressed(self):
        if self.edit_values is None:
            return
        self.edit_values_baseline = self.edit_values.copy()

    def _on_level_slider_changed(self, value: int):
        if not self.level_slider_active or self.edit_values_baseline is None:
            return

        scale = max(1.0, float(np.max(np.abs(self.edit_values_baseline))) * 2.0)
        delta = (value / 1000.0) * scale
        self.edit_values = self.edit_values_baseline + delta
        self._plot_edit_signal()

    def _on_level_slider_released(self):
        if self.edit_values is None or self.edit_values_baseline is None:
            return

        if not np.array_equal(self.edit_values, self.edit_values_baseline):
            self.undo_stack.append(self.edit_values_baseline.copy())
        self._update_buttons_state()

    def _undo_last_edit(self):
        if not self.undo_stack:
            return
        self.edit_values = self.undo_stack.pop()
        self._plot_edit_signal()
        self._update_buttons_state()

    def _save_edit(self):
        if self.current_edit_index is None or self.edit_values is None:
            return

        self.signals[self.current_edit_index].values = self.edit_values.copy()
        self.undo_stack.clear()
        self._update_buttons_state()

    def _show_sum(self):
        rows = self._checked_rows()
        if not rows:
            return

        base = np.zeros_like(self.signals[rows[0]].values, dtype=float)
        for r in rows:
            base = base + self.signals[r].values

        self.summed_signal = base
        self.show_components_active = False
        self.btn_show_components.blockSignals(True)
        self.btn_show_components.setChecked(False)
        self.btn_show_components.blockSignals(False)
        self._update_show_components_button_style()
        self._plot_sum()
        self._update_buttons_state()

    def _toggle_show_components(self, checked: bool):
        if self.summed_signal is None:
            return
        self.show_components_active = checked
        self._update_show_components_button_style()
        self._plot_sum()

    def _plot_sum(self):
        self.sum_plot.ax.clear()
        if self.summed_signal is None:
            self.sum_plot.canvas.draw_idle()
            return

        t_half, _ = self._parse_params()
        if t_half is None:
            return

        x = build_time_axis(t_half, len(self.summed_signal))

        if self.show_components_active:
            for r in self._checked_rows():
                sig = self.signals[r]
                self.sum_plot.ax.plot(x, sig.values, color=sig.color.name(), linewidth=1.2, alpha=0.8, label=sig.name)

        self.sum_plot.ax.plot(x, self.summed_signal, color="black", linewidth=2.0, label="Сумма")
        self.sum_plot.ax.grid(True)
        self.sum_plot.ax.set_title("Суммарный сигнал")
        self.sum_plot.ax.set_xlabel("t, сек")
        self.sum_plot.ax.legend(loc="best")
        self.sum_plot.canvas.draw_idle()


    def _calculate_spectrum(self):
        if self.summed_signal is None:
            return

        t_half, _ = self._parse_params()
        if t_half is None:
            self._show_error("Некорректное значение T/2.")
            return

        n = len(self.summed_signal)
        if n < 2:
            self._show_error("Для спектра требуется минимум 2 точки.")
            return

        t = build_time_axis(t_half, n)
        delta_t = abs(t[1] - t[0])

        spectrum = fftshift(fft(fftshift(self.summed_signal))) / n
        freq = fftshift(fftfreq(n, delta_t))

        self.spectrum_freq = freq
        self.spectrum_amp = np.abs(spectrum)
        self.spectrum_phase = np.angle(spectrum)
        self.spectrum_real = np.real(spectrum)
        self.spectrum_imag = np.imag(spectrum)

        self.btn_amp.setChecked(True)
        self._plot_spectrum_mode("amp")
        self._update_buttons_state()

    def _plot_spectrum_mode(self, mode: str):
        if self.spectrum_freq is None:
            return

        self.spec_plot.ax.clear()

        if mode == "amp":
            y = self.spectrum_amp
            title = "Амплитуда спектра"
        elif mode == "phase":
            y = self.spectrum_phase
            title = "Фаза спектра"
        elif mode == "real":
            y = self.spectrum_real
            title = "Действительная часть спектра"
        else:
            y = self.spectrum_imag
            title = "Мнимая часть спектра"

        self.spec_plot.ax.plot(self.spectrum_freq, y, color="tab:blue", linewidth=1.3)
        self.spec_plot.ax.set_title(title)
        self.spec_plot.ax.set_xlabel("f, Гц")
        self.spec_plot.ax.grid(True)
        self.spec_plot.canvas.draw_idle()


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
