import json
import math
import os
import sys
import random
from dataclasses import dataclass
from typing import Optional


import numpy as np
from scipy.fft import fft, fftfreq, fftshift
from skimage import io as skio

#Добавить возможность смотреть несколько спектров

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QAction, QColor, QIcon, QPixmap
    from PyQt6.QtWidgets import (

        QAbstractItemView,
        QApplication,
        QButtonGroup,
        QCheckBox,
        QComboBox,
        QDialog,
        QColorDialog,
        QFileDialog,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,

        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QDoubleSpinBox,
        QSizePolicy,
        QSlider,
        QTableWidget,
        QTableWidgetItem,
        QTextEdit,
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
    MESSAGEBOX_YES = QMessageBox.StandardButton.Yes
    MESSAGEBOX_NO = QMessageBox.StandardButton.No
except ImportError:
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QColor, QIcon, QPixmap
    from PyQt5.QtWidgets import (
        QAction,

        QAbstractItemView,
        QApplication,
        QButtonGroup,
        QCheckBox,
        QComboBox,
        QDialog,
        QColorDialog,
        QFileDialog,
        QFormLayout,
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
        QTextEdit,
        QVBoxLayout,
        QWidget,
        #QSpinBox,
        QDoubleSpinBox
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
    MESSAGEBOX_YES = QMessageBox.Yes
    MESSAGEBOX_NO = QMessageBox.No


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
    raw_values: np.ndarray



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


def calc_model_sigma(duration: float) -> Optional[float]:
    if duration <= 0.0:
        return None

    log_level = math.log(0.01)
    sigma_sq = -duration / log_level
    if sigma_sq <= 0.0:
        return None

    return math.sqrt(sigma_sq)


def calc_model_spectrum_width(duration: float) -> Optional[float]:
    sigma = calc_model_sigma(duration)
    if sigma is None or sigma <= 0.0:
        return None

    return 1.0 / (2.0 * math.pi * sigma)


class ModelSignalDialog(QDialog):
    def __init__(self, t_half: float, n_points: int, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Добавить модельный сигнал")
        self.setModal(True)

        self.t_half = t_half
        self.n_points = n_points

        self.duration_value: Optional[float] = None
        self.carrier_freq_value: Optional[float] = None
        self.amplitude_value: Optional[float] = None

        t_full = 2.0 * self.t_half
        self.delta_f = 1.0 / t_full if t_full > 0 else None
        self.f_max = self.n_points / (2.0 * t_full) if t_full > 0 else None

        root = QVBoxLayout(self)
        form = QFormLayout()

        self.input_duration = QDoubleSpinBox(self)
        self.input_duration.setDecimals(6)
        self.input_duration.setRange(0.0, max(0.0, t_full))
        self.input_duration.setSingleStep(max(0.001, t_full / 200.0 if t_full > 0 else 0.001))
        form.addRow("Длительность импульса, сек", self.input_duration)

        self.input_carrier = QDoubleSpinBox(self)
        self.input_carrier.setDecimals(6)
        self.input_carrier.setRange(0.0, max(0.0, self.f_max if self.f_max is not None else 0.0))
        self.input_carrier.setSingleStep(max(0.001, (self.f_max or 1.0) / 100.0))
        form.addRow("Несущая частота, Гц", self.input_carrier)

        self.input_amplitude = QDoubleSpinBox(self)
        self.input_amplitude.setDecimals(6)
        self.input_amplitude.setRange(0.0, 1e9)
        self.input_amplitude.setSingleStep(0.1)
        self.input_amplitude.setValue(1.0)
        form.addRow("Амплитуда, ед", self.input_amplitude)

        self.output_width = QLineEdit(self)
        self.output_width.setReadOnly(True)
        form.addRow("Ширина спектра на уровне 0.01, Гц", self.output_width)

        self.output_points = QLineEdit(self)
        self.output_points.setReadOnly(True)
        form.addRow("Количество точек на ширину спектра", self.output_points)

        self.output_periods = QLineEdit(self)
        self.output_periods.setReadOnly(True)
        form.addRow("Количество периодов на импульс", self.output_periods)

        root.addLayout(form)

        buttons = QHBoxLayout()
        self.btn_ok = QPushButton("Ок")
        self.btn_cancel = QPushButton("Отмена")
        buttons.addStretch(1)
        buttons.addWidget(self.btn_ok)
        buttons.addWidget(self.btn_cancel)
        root.addLayout(buttons)

        self.btn_ok.setEnabled(False)
        self.btn_ok.clicked.connect(self._accept_if_valid)
        self.btn_cancel.clicked.connect(self.reject)

        self.input_duration.valueChanged.connect(self._recalculate)
        self.input_carrier.valueChanged.connect(self._recalculate)
        self.input_amplitude.valueChanged.connect(self._recalculate)

        self._recalculate()

    def _set_field_style(self, widget: QWidget, ok: bool):
        widget.setStyleSheet("" if ok else "border: 1px solid red;")

    def _fmt(self, value: float) -> str:
        return f"{value:.6g}"

    def _recalculate(self):
        duration = float(self.input_duration.value())
        carrier = float(self.input_carrier.value())
        amplitude = float(self.input_amplitude.value())

        duration_ok = duration >= 0.0 and duration <= 2.0 * self.t_half
        carrier_ok = self.f_max is not None and carrier >= 0.0 and carrier <= self.f_max
        amplitude_ok = amplitude >= 0.0

        width_value = calc_model_spectrum_width(duration)
        width_ok = False
        if width_value is not None and self.delta_f is not None and self.f_max is not None:
            width_ok = self.delta_f <= width_value <= self.f_max

        if width_value is None:
            self.output_width.setText("—")
        else:
            self.output_width.setText(self._fmt(width_value))
        self._set_field_style(self.output_width, width_ok)

        points_value = None
        points_ok = False
        if width_value is not None and self.delta_f is not None and self.delta_f > 0.0:
            points_value = int(math.floor(width_value / self.delta_f))
            points_ok = points_value >= 10

        if points_value is None:
            self.output_points.setText("—")
        else:
            self.output_points.setText(str(points_value))
        self._set_field_style(self.output_points, points_ok)

        periods_value = duration * carrier
        self.output_periods.setText(self._fmt(periods_value))

        all_ok = duration_ok and carrier_ok and amplitude_ok and width_ok and points_ok
        self.btn_ok.setEnabled(all_ok)

    def _accept_if_valid(self):
        if not self.btn_ok.isEnabled():
            return

        self.duration_value = float(self.input_duration.value())
        self.carrier_freq_value = float(self.input_carrier.value())
        self.amplitude_value = float(self.input_amplitude.value())
        self.accept()


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


class HelpDialog(QDialog):
    def __init__(self, markdown_text: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Справка")
        self.resize(900, 700)

        layout = QVBoxLayout(self)
        viewer = QTextEdit(self)
        viewer.setReadOnly(True)
        viewer.setMarkdown(markdown_text)
        layout.addWidget(viewer)


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
        self.amplify_values_baseline: Optional[np.ndarray] = None
        self.amplify_undo_pushed = False
        self.phase_shift_values_baseline: Optional[np.ndarray] = None
        self.phase_shift_undo_pushed = False
        self.show_components_active = False


        self.spectrum_freq: Optional[np.ndarray] = None
        self.spectrum_amp: Optional[np.ndarray] = None
        self.spectrum_phase: Optional[np.ndarray] = None
        self.spectrum_real: Optional[np.ndarray] = None
        self.spectrum_imag: Optional[np.ndarray] = None
        self.spectrum_total_points: Optional[int] = None

        self.t_half: Optional[float] = None

        self.n_points: Optional[int] = None

        self.freq_limit_min: Optional[float] = None
        self.freq_limit_max: Optional[float] = None

        self.current_project_path: Optional[str] = None

        self._build_ui()

        self._update_fourier_params_ui()
        self._reset_frequency_limits(update_plot=False)
        self._update_buttons_state()


    def _build_ui(self):
        self._build_menu_bar()

        central = QWidget()
        self.setCentralWidget(central)

        grid = QGridLayout(central)


        self.sidebar_box = self._build_sidebar_panel()
        self.signals_box = self._build_signals_panel()
        self.edit_box = self._build_edit_panel()
        self.spectrum_box = self._build_spectrum_panel()
        self.sum_box = self._build_sum_panel()

        grid.addWidget(self.sidebar_box, 0, 0, 2, 1)
        grid.addWidget(self.signals_box, 0, 1)
        grid.addWidget(self.edit_box, 0, 2)
        grid.addWidget(self.spectrum_box, 1, 1)
        grid.addWidget(self.sum_box, 1, 2)

        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)

    def _build_menu_bar(self):
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("Файл")
        action_save = QAction("Сохранить проект", self)
        action_save_as = QAction("Сохранить проект как", self)
        action_open = QAction("Открыть проект из файла", self)

        action_save.triggered.connect(self._save_project)
        action_save_as.triggered.connect(self._save_project_as)
        action_open.triggered.connect(self._open_project)

        file_menu.addAction(action_save)
        file_menu.addAction(action_save_as)
        file_menu.addAction(action_open)

        help_menu = menu_bar.addMenu("Справка")
        action_show_help = QAction("Открыть документацию", self)
        action_show_help.triggered.connect(self._show_help)
        help_menu.addAction(action_show_help)

    def _build_sidebar_panel(self) -> QGroupBox:

        box = QGroupBox("Параметры")
        root = QVBoxLayout(box)

        scan_box = QGroupBox("Параметры развертки")
        scan_layout = QVBoxLayout(scan_box)

        row_t = QHBoxLayout()
        row_t.addWidget(QLabel("T/2, сек"))
        self.input_t_half = QLineEdit()
        self.input_t_half.textChanged.connect(self._on_params_changed)
        row_t.addWidget(self.input_t_half)
        scan_layout.addLayout(row_t)

        row_n = QHBoxLayout()
        row_n.addWidget(QLabel("Количество отсчетов N"))
        self.input_n = QLineEdit()
        self.input_n.textChanged.connect(self._on_params_changed)
        row_n.addWidget(self.input_n)
        scan_layout.addLayout(row_n)

        self.btn_apply_params = QPushButton("Применить параметры")
        self.btn_apply_params.clicked.connect(self._apply_scan_params)
        scan_layout.addWidget(self.btn_apply_params)

        self.btn_add = QPushButton("Добавить")
        self.btn_add.clicked.connect(self._add_signal_flow)
        scan_layout.addWidget(self.btn_add)

        self.btn_add_model = QPushButton("Добавить модельный сигнал")
        self.btn_add_model.clicked.connect(self._add_model_signal)
        scan_layout.addWidget(self.btn_add_model)

        root.addWidget(scan_box)

        fft_box = QGroupBox("Параметры Фурье")
        fft_layout = QVBoxLayout(fft_box)
        self.label_dt = QLabel("Шаг по времени (T / N), сек: —")
        self.label_df = QLabel("Шаг по частоте (1 / T), Гц: —")
        self.label_fs = QLabel("Частота дискретизации (N / T), Гц: —")
        self.label_fmax = QLabel("Максимальная частота спектра (N / 2T), Гц: —")

        fft_layout.addWidget(self.label_dt)
        fft_layout.addWidget(self.label_df)
        fft_layout.addWidget(self.label_fs)
        fft_layout.addWidget(self.label_fmax)

        root.addWidget(fft_box)

        limits_box = QGroupBox("Ограничение по частоте")
        limits_layout = QVBoxLayout(limits_box)

        row_min = QHBoxLayout()
        row_min.addWidget(QLabel("Нижняя, Гц"))
        self.input_freq_min = QLineEdit()
        self.input_freq_min.textChanged.connect(self._on_frequency_limits_changed)
        row_min.addWidget(self.input_freq_min)
        limits_layout.addLayout(row_min)

        row_max = QHBoxLayout()
        row_max.addWidget(QLabel("Верхняя, Гц"))
        self.input_freq_max = QLineEdit()
        self.input_freq_max.textChanged.connect(self._on_frequency_limits_changed)
        row_max.addWidget(self.input_freq_max)
        limits_layout.addLayout(row_max)

        row_samples = QHBoxLayout()
        row_samples.addWidget(QLabel("Количество отсчётов"))
        self.label_freq_samples_count = QLabel("—")
        row_samples.addWidget(self.label_freq_samples_count)
        limits_layout.addLayout(row_samples)

        self.btn_reset_freq_limits = QPushButton("Сбросить")

        self.btn_reset_freq_limits.clicked.connect(lambda: self._reset_frequency_limits(update_plot=True))
        limits_layout.addWidget(self.btn_reset_freq_limits)


        root.addWidget(limits_box)
        return box


    def _build_signals_panel(self) -> QGroupBox:
        box = QGroupBox("Сигналы")
        root = QVBoxLayout(box)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Выбор", "Название", "Цвет", "Удалить", "Копировать"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.itemSelectionChanged.connect(self._on_table_selection_changed)
        self.table.itemChanged.connect(self._on_table_item_changed)
        self.table.cellClicked.connect(self._on_table_cell_clicked)
        root.addWidget(self.table)

        self.btn_show_sum = QPushButton("Показать сумму")
        self.btn_show_sum.clicked.connect(self._show_sum)
        root.addWidget(self.btn_show_sum)

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

        self.btn_phase_shift = QPushButton("Cдвиг фазы")
        self.btn_phase_shift.setCheckable(True)
        self.btn_phase_shift.clicked.connect(self._toggle_phase_shift_mode)
        right.addWidget(self.btn_phase_shift)

        self.btn_amplify = QPushButton("Усиление")
        self.btn_amplify.setCheckable(True)
        self.btn_amplify.clicked.connect(self._toggle_amplify_mode)
        right.addWidget(self.btn_amplify)

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

        self.spinbox_time_shift = QDoubleSpinBox()
        self.spinbox_time_shift.setDecimals(6)
        self.spinbox_time_shift.setRange(-1.0, 1.0)
        self.spinbox_time_shift.setSingleStep(0.1)
        self.spinbox_time_shift.setValue(0.0)
        self.label_time_shift = QLabel("Введите сдвиг в сек:")
        right.addWidget(self.label_time_shift)
        right.addWidget(self.spinbox_time_shift)
        self.spinbox_time_shift.hide()
        self.label_time_shift.hide()
        self.spinbox_time_shift.valueChanged.connect(self._on_phase_shift_changed)


        self.spinbox_amplify = QDoubleSpinBox()
        self.spinbox_amplify.setRange(-10.0, 10.0)
        self.spinbox_amplify.setValue(1.0)

        self.spinbox_amplify.setSingleStep(0.1)
        self.label_amplify = QLabel("Выберите коэффициент усиления:")
        right.addWidget(self.label_amplify)
        right.addWidget(self.spinbox_amplify)
        self.spinbox_amplify.hide()
        self.label_amplify.hide() 
        self.spinbox_amplify.valueChanged.connect(self._on_amplify_value_changed)


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

        actions = QHBoxLayout()

        self.btn_show_components = QPushButton("Показать исходные сигналы")
        self.btn_show_components.setCheckable(True)
        self.btn_show_components.toggled.connect(self._toggle_show_components)
        actions.addWidget(self.btn_show_components)

        self.btn_save_sum = QPushButton("Сохранить сумму")
        self.btn_save_sum.clicked.connect(self._save_sum_plot)
        actions.addWidget(self.btn_save_sum)

        root.addLayout(actions)

        return box


    def _build_spectrum_panel(self) -> QGroupBox:
        box = QGroupBox("Спектр")
        root = QVBoxLayout(box)

        actions = QHBoxLayout()

        self.btn_calc_spec = QPushButton("Рассчитать спектр")
        self.btn_calc_spec.clicked.connect(self._calculate_spectrum)
        actions.addWidget(self.btn_calc_spec)

        self.btn_save_spectrum = QPushButton("Сохранить спектр")
        self.btn_save_spectrum.clicked.connect(self._save_spectrum_plot)
        actions.addWidget(self.btn_save_spectrum)

        root.addLayout(actions)

        self.checkbox_positive_freq_only = QCheckBox("только положительные")
        self.checkbox_positive_freq_only.setChecked(True)
        self.checkbox_positive_freq_only.toggled.connect(self._on_positive_freq_only_toggled)
        root.addWidget(self.checkbox_positive_freq_only)

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

        self.spec_span_selector = SpanSelector(
            self.spec_plot.ax,
            self._on_spectrum_span_selected,
            direction="horizontal",
            useblit=True,
            props=dict(alpha=0.2, facecolor="orange"),
            interactive=True,
            drag_from_anywhere=True,
        )
        self.spec_span_selector.set_active(True)

        return box


    def _show_error(self, text: str):
        QMessageBox.warning(self, "Ошибка", text)

    def _signal_to_dict(self, sig: SignalItem) -> dict:
        return {
            "name": sig.name,
            "color": sig.color.name(),
            "values": [float(v) for v in sig.values.tolist()],
        }

    def _serialize_project(self) -> dict:
        t_half, n_points = self._parse_params()
        return {
            "version": 1,
            "t_half": t_half,
            "n_points": n_points,
            "signals": [self._signal_to_dict(sig) for sig in self.signals],
        }

    def _validate_project_payload(self, payload: object) -> bool:
        if not isinstance(payload, dict):
            return False

        if "signals" not in payload:
            return False
        signals = payload.get("signals")
        if not isinstance(signals, list):
            return False

        t_half = payload.get("t_half")
        n_points = payload.get("n_points")

        if t_half is None or not isinstance(t_half, (int, float)) or float(t_half) < 0:
            return False
        if n_points is None or not isinstance(n_points, int) or n_points <= 0:
            return False

        for s in signals:
            if not isinstance(s, dict):
                return False
            name = s.get("name")
            color = s.get("color")
            values = s.get("values")

            if not isinstance(name, str):
                return False
            if not isinstance(color, str):
                return False
            if not isinstance(values, list):
                return False
            if len(values) != int(n_points):
                return False
            if not all(isinstance(v, (int, float)) for v in values):
                return False

            qcolor = QColor(color)
            if not qcolor.isValid():
                return False

        return True

    def _save_project(self):
        if self.current_project_path:
            self._save_project_to_path(self.current_project_path)
            return
        self._save_project_as()

    def _save_project_as(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить проект",
            "project.json",
            "JSON files (*.json)",
        )
        if not file_path:
            return

        if not file_path.lower().endswith(".json"):
            file_path += ".json"

        self._save_project_to_path(file_path)

    def _save_project_to_path(self, file_path: str):
        payload = self._serialize_project()
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            self._show_error(f"Не удалось сохранить проект: {exc}")
            return

        self.current_project_path = file_path

    def _open_project(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Открыть проект",
            "",
            "JSON files (*.json)",
        )
        if not file_path:
            return

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            self._show_error("некорректная структура  файла")
            return

        if not self._validate_project_payload(payload):
            self._show_error("некорректная структура  файла")
            return

        t_half = float(payload["t_half"])
        n_points = int(payload["n_points"])

        self.t_half = t_half
        self.n_points = n_points

        self.input_t_half.blockSignals(True)
        self.input_n.blockSignals(True)
        self.input_t_half.setText(self._format_param_value(t_half))
        self.input_n.setText(str(n_points))
        self.input_t_half.blockSignals(False)
        self.input_n.blockSignals(False)

        self.signals.clear()
        for s in payload["signals"]:
            values = np.array(s["values"], dtype=float)
            self.signals.append(
                SignalItem(
                    name=s["name"].strip() or f"Сигнал {len(self.signals) + 1}",
                    color=QColor(s["color"]),
                    values=values.copy(),
                    raw_values=values.copy(),
                )
            )

        self.current_project_path = file_path

        self.current_edit_index = None
        self.edit_values = None
        self.undo_stack.clear()
        self.edit_plot.clear()

        self._clear_sum_and_spectrum()
        self._validate_params_ui()
        self._update_fourier_params_ui()
        self._reset_frequency_limits(update_plot=False)
        self._refresh_table()
        self._update_buttons_state()

    def _show_help(self):
        docs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs.md")
        try:
            with open(docs_path, "r", encoding="utf-8") as f:
                markdown_text = f.read()
        except Exception as exc:
            self._show_error(f"Не удалось открыть справку: {exc}")
            return

        dialog = HelpDialog(markdown_text, self)
        dialog.exec()

    def _format_param_value(self, value: float) -> str:
        return f"{value:.6g}"


    def _parse_params_input(self) -> tuple[Optional[float], Optional[int]]:
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

    def _parse_params(self) -> tuple[Optional[float], Optional[int]]:
        return self.t_half, self.n_points

    def _validate_params_ui(self):
        t_half, n_points = self._parse_params_input()

        self.input_t_half.setStyleSheet("" if t_half is not None else "border: 1px solid red;")
        self.input_n.setStyleSheet("" if n_points is not None else "border: 1px solid red;")

    def _update_fourier_params_ui(self):
        t_half, n_points = self._parse_params_input()

        if t_half is None or n_points is None:
            self.label_dt.setText("Шаг по времени (T / N), сек: —")
            self.label_df.setText("Шаг по частоте (1 / T), Гц: —")
            self.label_fs.setText("Частота дискретизации (N / T), Гц: —")
            self.label_fmax.setText("Максимальная частота спектра (N / 2T), Гц: —")
            return

        t_full = 2.0 * t_half
        if t_full <= 0.0:
            self.label_dt.setText("Шаг по времени (T / N), сек: —")
            self.label_df.setText("Шаг по частоте (1 / T), Гц: —")
            self.label_fs.setText("Частота дискретизации (N / T), Гц: —")
            self.label_fmax.setText("Максимальная частота спектра (N / 2T), Гц: —")
            return

        dt = t_full / n_points
        df = 1.0 / t_full
        fs = n_points / t_full
        fmax = n_points / (2.0 * t_full)

        self.label_dt.setText(f"Шаг по времени (T / N), сек: {self._format_param_value(dt)}")
        self.label_df.setText(f"Шаг по частоте (1 / T), Гц: {self._format_param_value(df)}")
        self.label_fs.setText(f"Частота дискретизации (N / T), Гц: {self._format_param_value(fs)}")
        self.label_fmax.setText(f"Максимальная частота спектра (N / 2T), Гц: {self._format_param_value(fmax)}")


    def _frequency_bounds_defaults(self) -> tuple[Optional[float], Optional[float]]:
        t_half, n_points = self._parse_params_input()
        if t_half is None or n_points is None:
            return None, None

        t_full = 2.0 * t_half
        if t_full <= 0.0:
            return None, None

        f_max = n_points / (2.0 * t_full)
        f_min = 0.0 if self.checkbox_positive_freq_only.isChecked() else -f_max
        return f_min, f_max

    def _set_frequency_limits_inputs(self, f_min: float, f_max: float):
        self.input_freq_min.blockSignals(True)
        self.input_freq_max.blockSignals(True)
        self.input_freq_min.setText(self._format_param_value(f_min))
        self.input_freq_max.setText(self._format_param_value(f_max))
        self.input_freq_min.blockSignals(False)
        self.input_freq_max.blockSignals(False)

    def _clear_spectrum_span_selection(self):
        if hasattr(self.spec_span_selector, "clear"):
            self.spec_span_selector.clear()
        elif hasattr(self.spec_span_selector, "extents"):
            self.spec_span_selector.extents = (0.0, 0.0)

    def _update_frequency_samples_count(self):
        total_points = self.spectrum_total_points if self.spectrum_total_points is not None else self.n_points
        if total_points is None:
            self.label_freq_samples_count.setText("—")
            self.label_freq_samples_count.setStyleSheet("")
            return

        points = total_points
        if self.spectrum_freq is not None:
            x = self.spectrum_freq
            if self.checkbox_positive_freq_only.isChecked():
                x = x[x >= 0]

            f_min = self.freq_limit_min
            f_max = self.freq_limit_max
            if f_min is not None and f_max is not None and f_min < f_max:
                mask = (x >= f_min) & (x <= f_max)
                points = int(np.count_nonzero(mask))
            else:
                points = x.size

        self.label_freq_samples_count.setText(str(points))
        self.label_freq_samples_count.setStyleSheet("color: red;" if points < 10 else "")

    def _validate_frequency_limits_ui(self, f_min: Optional[float], f_max: Optional[float]) -> bool:

        valid = f_min is not None and f_max is not None and f_min < f_max
        min_style = "" if f_min is not None else "border: 1px solid red;"
        max_style = "" if f_max is not None else "border: 1px solid red;"

        if f_min is not None and f_max is not None and f_min >= f_max:
            min_style = "border: 1px solid red;"
            max_style = "border: 1px solid red;"
            valid = False

        bounds = self._frequency_bounds_defaults()
        if bounds[0] is not None and bounds[1] is not None and f_min is not None and f_max is not None:
            min_allowed, max_allowed = bounds
            if f_min < min_allowed:
                min_style = "border: 1px solid red;"
                valid = False
            if f_max > max_allowed:
                max_style = "border: 1px solid red;"
                valid = False

        self.input_freq_min.setStyleSheet(min_style)
        self.input_freq_max.setStyleSheet(max_style)
        return valid


    def _parse_frequency_limits_input(self) -> tuple[Optional[float], Optional[float]]:
        f_min = None
        f_max = None

        try:
            f_min = float(self.input_freq_min.text().strip())
        except Exception:
            pass

        try:
            f_max = float(self.input_freq_max.text().strip())
        except Exception:
            pass

        return f_min, f_max

    def _reset_frequency_limits(self, update_plot: bool = True):
        defaults = self._frequency_bounds_defaults()
        if defaults[0] is None or defaults[1] is None:
            self.freq_limit_min = None
            self.freq_limit_max = None
            self.input_freq_min.blockSignals(True)
            self.input_freq_max.blockSignals(True)
            self.input_freq_min.setText("")
            self.input_freq_max.setText("")
            self.input_freq_min.blockSignals(False)
            self.input_freq_max.blockSignals(False)
            self.input_freq_min.setStyleSheet("")
            self.input_freq_max.setStyleSheet("")
            self._clear_spectrum_span_selection()
            self._update_frequency_samples_count()
            return

        f_min, f_max = defaults
        self.freq_limit_min = f_min
        self.freq_limit_max = f_max
        self._set_frequency_limits_inputs(f_min, f_max)
        self._validate_frequency_limits_ui(f_min, f_max)
        self._clear_spectrum_span_selection()

        if update_plot and self.spectrum_freq is not None:
            mode = self._current_spectrum_mode()
            if mode is not None:
                self._plot_spectrum_mode(mode)

        self._update_frequency_samples_count()


    def _on_frequency_limits_changed(self):
        f_min, f_max = self._parse_frequency_limits_input()
        if not self._validate_frequency_limits_ui(f_min, f_max):
            self._update_frequency_samples_count()
            return

        self.freq_limit_min = f_min
        self.freq_limit_max = f_max

        if self.spectrum_freq is not None:
            mode = self._current_spectrum_mode()
            if mode is not None:
                self._plot_spectrum_mode(mode)

        self._update_frequency_samples_count()


    def _on_params_changed(self):
        self._validate_params_ui()
        self._update_fourier_params_ui()

        t_half, _ = self._parse_params_input()
        step = (2.0 * t_half) / 20.0 if t_half is not None and t_half > 0 else 0.1
        shift_limit = 2.0 * t_half if t_half is not None else 1.0
        self.spinbox_time_shift.setSingleStep(max(1e-6, step))
        self.spinbox_time_shift.setRange(-shift_limit, shift_limit)

        self._reset_frequency_limits(update_plot=False)
        self._update_buttons_state()



    def _clear_sum_and_spectrum(self):
        self.summed_signal = None
        self.sum_plot.clear()

        self.spectrum_freq = None
        self.spectrum_amp = None
        self.spectrum_phase = None
        self.spectrum_real = None
        self.spectrum_imag = None
        self.spectrum_total_points = None
        self.spec_plot.clear()
        self._update_frequency_samples_count()


    def _apply_scan_params(self):
        t_half_new, n_points_new = self._parse_params_input()
        if t_half_new is None or n_points_new is None:
            self._show_error("Заполните корректно параметры T/2 и N.")
            return

        n_changed = self.n_points is not None and self.n_points != n_points_new

        if n_changed and self.signals:
            reply = QMessageBox.question(
                self,
                "Подтверждение пересчёта",
                "Изменение N пересчитает все сигналы из исходных данных и сбросит внесённые правки. Продолжить?",
                MESSAGEBOX_YES | MESSAGEBOX_NO,
                MESSAGEBOX_NO,
            )
            if reply != MESSAGEBOX_YES:
                return

            try:
                new_values = [downsample_signal(sig.raw_values, n_points_new) for sig in self.signals]
            except Exception as exc:
                self._show_error(f"Не удалось применить параметры: {exc}")
                return

            for sig, vals in zip(self.signals, new_values):
                sig.values = vals

            self.current_edit_index = None
            self.edit_values = None
            self.undo_stack.clear()
            self.edit_plot.clear()

        self.t_half = t_half_new
        self.n_points = n_points_new

        self._clear_sum_and_spectrum()
        self._reset_frequency_limits(update_plot=False)
        self._update_buttons_state()


        if n_changed:
            selected_row = self.table.currentRow()
            if 0 <= selected_row < len(self.signals):
                self.current_edit_index = selected_row
                self._show_selected_signal()
            else:
                self.current_edit_index = None
                self.edit_values = None
                self.edit_plot.clear()
        elif self.current_edit_index is not None and self.edit_values is not None:
            self._plot_edit_signal()

    def _update_buttons_state(self):
        input_t_half, input_n_points = self._parse_params_input()
        params_input_valid = input_t_half is not None and input_n_points is not None
        params_applied = self.t_half is not None and self.n_points is not None

        params_changed = (
            not params_applied
            or (params_input_valid and (self.t_half != input_t_half or self.n_points != input_n_points))
        )
        self.btn_apply_params.setEnabled(params_input_valid and params_changed)

        checked_count = 0

        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            if item is not None and item.checkState() == CHECKED:
                checked_count += 1

        self.btn_add.setEnabled(params_applied and not params_changed)
        self.btn_add_model.setEnabled(params_applied and not params_changed)
        self.btn_show_sum.setEnabled(params_applied and len(self.signals) > 0 and checked_count > 0)




        has_edit = self.edit_values is not None
        self.btn_zero.setEnabled(has_edit)
        self.btn_level.setEnabled(has_edit)
        self.btn_phase_shift.setEnabled(has_edit)
        self.btn_amplify.setEnabled(has_edit)
        self.btn_undo.setEnabled(len(self.undo_stack) > 0)
        self.btn_save.setEnabled(has_edit and len(self.undo_stack) > 0)

        has_sum = self.summed_signal is not None
        if not has_sum:
            self.show_components_active = False
            self.btn_show_components.blockSignals(True)
            self.btn_show_components.setChecked(False)
            self.btn_show_components.blockSignals(False)

        self.btn_show_components.setEnabled(has_sum)
        self.btn_save_sum.setEnabled(has_sum)
        self._update_show_components_button_style()
        self.btn_calc_spec.setEnabled(has_sum)

        spectrum_ready = self.spectrum_amp is not None

        for b in [self.btn_amp, self.btn_phase, self.btn_real, self.btn_imag]:
            b.setEnabled(spectrum_ready)
        self.btn_save_spectrum.setEnabled(spectrum_ready)

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

    def _on_table_selection_changed(self):
        row = self.table.currentRow()
        if row < 0 or row >= len(self.signals):
            self._update_buttons_state()
            return

        if self.current_edit_index != row:
            self._show_selected_signal()
        else:
            self._update_buttons_state()

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

    def _build_copy_name(self, original_name: str) -> str:
        existing_names = {s.name for s in self.signals}

        first_variant = f"{original_name} (Копия)"
        if first_variant not in existing_names:
            return first_variant

        idx = 2
        while True:
            candidate = f"{original_name} (Копия {idx})"
            if candidate not in existing_names:
                return candidate
            idx += 1

    def _copy_signal(self, row: int):
        if row < 0 or row >= len(self.signals):
            return
        
        copy_index = len(self.signals)
        source = self.signals[row]

        signal_copy = SignalItem(
            name=self._build_copy_name(source.name),
            color=default_signal_color(copy_index),
            values=source.values.copy(),
            raw_values=source.raw_values.copy(),
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
            raw_values=sig_full.copy(),
        )
        self.signals.append(item)
        self._refresh_table()

    def _add_model_signal(self):
        t_half, n_points = self._parse_params()
        if t_half is None or n_points is None:
            self._show_error("Сначала примените корректные параметры T/2 и N.")
            return

        dialog = ModelSignalDialog(t_half, n_points, self)
        if dialog.exec() != DIALOG_ACCEPTED:
            return

        duration = dialog.duration_value
        carrier = dialog.carrier_freq_value
        amplitude = dialog.amplitude_value

        if duration is None or carrier is None or amplitude is None:
            return

        sigma = calc_model_sigma(duration)
        if sigma is None:
            self._show_error("Не удалось рассчитать sigma для модельного сигнала.")
            return

        x = build_time_axis(t_half, n_points)
        signal_values = amplitude * np.exp(-(x ** 2) / (2.0 * sigma ** 2)) * np.cos(2.0 * math.pi * carrier * x)

        signal_index = len(self.signals)
        item = SignalItem(
            name=f"Сигнал {signal_index + 1}",
            color=default_signal_color(signal_index),
            values=signal_values.astype(float),
            raw_values=signal_values.astype(float).copy(),
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
        self.btn_phase_shift.setChecked(False)
        self.btn_amplify.setChecked(False)
        self.span_selector.set_active(False)
        self._clear_span_selection()
        self.level_slider.hide()
        self.level_slider.setValue(0)
        self.spinbox_time_shift.hide()
        self.label_time_shift.hide()
        self.spinbox_time_shift.setValue(0.0)
        self.phase_shift_values_baseline = None
        self.phase_shift_undo_pushed = False
        self.spinbox_amplify.hide()
        self.label_amplify.hide()
        self.spinbox_amplify.setValue(1.0)
        self.amplify_values_baseline = None
        self.amplify_undo_pushed = False


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
            self.btn_amplify.setChecked(False)
            self.level_slider.hide()
            self.level_slider_active = False
            self.spinbox_time_shift.hide()
            self.label_time_shift.hide()
            self.spinbox_time_shift.setValue(0.0)
            self.phase_shift_values_baseline = None
            self.phase_shift_undo_pushed = False
            self.spinbox_amplify.hide()
            self.label_amplify.hide()
            self.amplify_values_baseline = None
            self.amplify_undo_pushed = False

        else:
            self._clear_span_selection()
        self.span_selector.set_active(active)


    def _toggle_level_mode(self):
        active = self.btn_level.isChecked() and self.edit_values is not None
        if active:
            self.btn_zero.setChecked(False)
            self.btn_phase_shift.setChecked(False)
            self.btn_amplify.setChecked(False)
            self.span_selector.set_active(False)
            self._clear_span_selection()
            self.level_slider_active = True
            self.level_slider.setValue(0)
            self.level_slider.show()
            self.edit_values_baseline = self.edit_values.copy() if self.edit_values is not None else None
            self.spinbox_time_shift.hide()
            self.label_time_shift.hide()
            self.spinbox_time_shift.setValue(0.0)
            self.phase_shift_values_baseline = None
            self.phase_shift_undo_pushed = False
            self.spinbox_amplify.hide()
            self.label_amplify.hide()
            self.amplify_values_baseline = None
            self.amplify_undo_pushed = False

        else:
            self.level_slider_active = False
            self.level_slider.hide()

    def _toggle_phase_shift_mode(self):
        active = self.btn_phase_shift.isChecked() and self.edit_values is not None
        if active:
            self.btn_zero.setChecked(False)
            self.btn_level.setChecked(False)
            self.btn_amplify.setChecked(False)
            self.span_selector.set_active(False)
            self._clear_span_selection()
            self.level_slider_active = False
            self.level_slider.hide()
            self.spinbox_amplify.hide()
            self.label_amplify.hide()
            self.amplify_values_baseline = None
            self.amplify_undo_pushed = False
            self.phase_shift_values_baseline = self.edit_values.copy()
            self.phase_shift_undo_pushed = False
            self.spinbox_time_shift.blockSignals(True)
            self.spinbox_time_shift.setValue(0.0)
            self.spinbox_time_shift.blockSignals(False)
            self.spinbox_time_shift.show()
            self.label_time_shift.show()
            self.spinbox_time_shift.setFocus()
            self._validate_phase_shift()
        else:
            self.spinbox_time_shift.hide()
            self.label_time_shift.hide()
            self.spinbox_time_shift.setValue(0.0)
            self.phase_shift_values_baseline = None
            self.phase_shift_undo_pushed = False

    def _on_phase_shift_changed(self):
        self._validate_phase_shift()
        self._apply_phase_shift()

    def _parse_phase_shift(self) -> Optional[float]:
        try:
            return float(self.spinbox_time_shift.value())
        except Exception:
            return None

    def _validate_phase_shift(self):
        t_half, _ = self._parse_params()
        delta_t = self._parse_phase_shift()

        if t_half is None or delta_t is None:
            self.spinbox_time_shift.setStyleSheet("border: 1px solid red;")
            return

        limit = 2.0 * t_half
        if abs(delta_t) > limit:
            self.spinbox_time_shift.setStyleSheet("border: 1px solid red;")
        else:
            self.spinbox_time_shift.setStyleSheet("")

    def _apply_phase_shift(self):
        delta_t = self._parse_phase_shift()
        t_half, _ = self._parse_params()

        if t_half is None or delta_t is None:
            return

        limit = 2.0 * t_half
        if abs(delta_t) > limit:
            return

        if self.edit_values is None or self.phase_shift_values_baseline is None:
            return

        n = len(self.phase_shift_values_baseline)
        if n < 2:
            return

        t = build_time_axis(t_half, n)
        delta_t_sample = abs(t[1] - t[0])
        n_steps = int(round(delta_t / delta_t_sample))
        shifted_values = np.roll(self.phase_shift_values_baseline, n_steps)

        if np.array_equal(shifted_values, self.edit_values):
            return

        if not self.phase_shift_undo_pushed:
            self.undo_stack.append(self.phase_shift_values_baseline.copy())
            self.phase_shift_undo_pushed = True

        self.edit_values = shifted_values
        self._plot_edit_signal()
        self._update_buttons_state()


    def _toggle_amplify_mode(self):
        active = self.btn_amplify.isChecked() and self.edit_values is not None
        if active:
            self.btn_zero.setChecked(False)
            self.btn_level.setChecked(False)
            self.btn_phase_shift.setChecked(False)

            self.span_selector.set_active(False)
            self._clear_span_selection()

            self.level_slider_active = False
            self.level_slider.hide()

            self.spinbox_time_shift.hide()
            self.label_time_shift.hide()
            self.spinbox_time_shift.setValue(0.0)
            self.phase_shift_values_baseline = None
            self.phase_shift_undo_pushed = False

            self.amplify_values_baseline = self.edit_values.copy()

            self.amplify_undo_pushed = False

            self.spinbox_amplify.blockSignals(True)
            self.spinbox_amplify.setValue(1.0)
            self.spinbox_amplify.blockSignals(False)

            self.label_amplify.show()
            self.spinbox_amplify.show()
            self.spinbox_amplify.setFocus()
        else:
            self.spinbox_amplify.hide()
            self.label_amplify.hide()
            self.amplify_values_baseline = None
            self.amplify_undo_pushed = False

    def _parse_amplitude(self) -> Optional[float]:
        try:
            return float(self.spinbox_amplify.value())
        except Exception:
            return None

    def _on_amplify_value_changed(self):
        if not self.btn_amplify.isChecked():
            return
        if self.edit_values is None or self.amplify_values_baseline is None:
            return

        amplify_coef = self._parse_amplitude()
        if amplify_coef is None:
            return

        amplified_signal = self.amplify_values_baseline * amplify_coef
        if np.array_equal(amplified_signal, self.edit_values):
            return

        if not self.amplify_undo_pushed:
            self.undo_stack.append(self.amplify_values_baseline.copy())
            self.amplify_undo_pushed = True

        self.edit_values = amplified_signal
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

    def _save_sum_plot(self):
        if self.summed_signal is None:
            self._show_error("Сначала сформируйте сумму сигналов.")
            return

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Сохранить график суммы",
            "sum_signal.png",
            "PNG files (*.png);;JPEG files (*.jpg *.jpeg);;PDF files (*.pdf);;SVG files (*.svg)",
        )
        if not file_path:
            return

        if not os.path.splitext(file_path)[1]:
            if "JPEG" in selected_filter:
                file_path += ".jpg"
            elif "PDF" in selected_filter:
                file_path += ".pdf"
            elif "SVG" in selected_filter:
                file_path += ".svg"
            else:
                file_path += ".png"

        try:
            self.sum_plot.figure.savefig(file_path, dpi=300, bbox_inches="tight")
        except Exception as exc:
            self._show_error(f"Не удалось сохранить сумму: {exc}")

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
        self.spectrum_total_points = int(freq.size)

        self.btn_amp.setChecked(True)
        self._plot_spectrum_mode("amp")
        self._update_frequency_samples_count()
        self._update_buttons_state()


    def _save_spectrum_plot(self):
        if self.spectrum_freq is None:
            self._show_error("Сначала рассчитайте спектр.")
            return

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Сохранить график спектра",
            "spectrum.png",
            "PNG files (*.png);;JPEG files (*.jpg *.jpeg);;PDF files (*.pdf);;SVG files (*.svg)",
        )
        if not file_path:
            return

        if not os.path.splitext(file_path)[1]:
            if "JPEG" in selected_filter:
                file_path += ".jpg"
            elif "PDF" in selected_filter:
                file_path += ".pdf"
            elif "SVG" in selected_filter:
                file_path += ".svg"
            else:
                file_path += ".png"

        try:
            self.spec_plot.figure.savefig(file_path, dpi=300, bbox_inches="tight")
        except Exception as exc:
            self._show_error(f"Не удалось сохранить спектр: {exc}")

    def _current_spectrum_mode(self) -> Optional[str]:
        if self.btn_amp.isChecked():
            return "amp"
        if self.btn_phase.isChecked():
            return "phase"
        if self.btn_real.isChecked():
            return "real"
        if self.btn_imag.isChecked():
            return "imag"
        return None

    def _on_positive_freq_only_toggled(self, _checked: bool):
        self._reset_frequency_limits(update_plot=True)

    def _on_spectrum_span_selected(self, x_min: float, x_max: float):
        if self.spectrum_freq is None:
            return

        f_left = float(min(x_min, x_max))
        f_right = float(max(x_min, x_max))
        if f_right <= f_left:
            return

        defaults = self._frequency_bounds_defaults()
        if defaults[0] is None or defaults[1] is None:
            return

        min_allowed, max_allowed = defaults
        f_left = max(min_allowed, f_left)
        f_right = min(max_allowed, f_right)

        if f_right <= f_left:
            return

        self.freq_limit_min = f_left
        self.freq_limit_max = f_right
        self._set_frequency_limits_inputs(f_left, f_right)
        self._validate_frequency_limits_ui(f_left, f_right)

        mode = self._current_spectrum_mode()
        if mode is not None:
            self._plot_spectrum_mode(mode)

        self._update_frequency_samples_count()


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

        x = self.spectrum_freq
        if self.checkbox_positive_freq_only.isChecked():
            positive_mask = x >= 0
            x = x[positive_mask]
            y = y[positive_mask]

        f_min = self.freq_limit_min
        f_max = self.freq_limit_max
        if f_min is None or f_max is None:
            defaults = self._frequency_bounds_defaults()
            f_min, f_max = defaults

        if f_min is not None and f_max is not None and f_min < f_max:
            limit_mask = (x >= f_min) & (x <= f_max)
            x = x[limit_mask]
            y = y[limit_mask]

        if x.size == 0:
            self.spec_plot.ax.set_title(title)
            self.spec_plot.ax.set_xlabel("f, Гц")
            self.spec_plot.ax.grid(True)
            self.spec_plot.canvas.draw_idle()
            return

        self.spec_plot.ax.plot(x, y, color="tab:blue", linewidth=1.3)
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
