import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, \
    QHBoxLayout, QWidget, QCheckBox, QInputDialog, QSlider, QLabel, QMessageBox, \
    QGridLayout, QFrame, QSizePolicy, QSpacerItem
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtGui import QAction
import numpy as np
from PyQt6.QtCore import Qt


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.data_len = 0
        self.y_true = []
        self.fields_widget = QWidget()
        self.slider_widget = QWidget()
        self.roc_info_widget = QWidget()
        self.matrix_widget = QWidget()
        self.all_info = QWidget()
        self.roc_data = []
        self.slider_label = None
        self.cur_dot = 0

        # Создание главного виджета и его компоновка
        main_widget = QWidget(self)
        self.layout = QVBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        # Создание горизонтального макета
        self.fig_widget = QWidget()

        # Создание графического виджета
        self.figure = Figure(figsize=(5, 5))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.canvas.setMinimumSize(200, 200)


        # Создание графического виджета
        self.figure2 = Figure(figsize=(5, 5))
        self.canvas2 = FigureCanvas(self.figure2)
        self.canvas2.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.canvas2.setMinimumSize(200, 200)

        # Создание кнопок открытия и закрытия файла
        bt_action_open_file = QAction("Set data length", self)
        # bt_action_open_file.triggered.connect(self.onOpenFileButtonClick)
        menu = self.menuBar()

        file_menu = menu.addMenu("Settings")
        file_menu.addAction(bt_action_open_file)
        bt_action_open_file.triggered.connect(self.set_data_len)
        file_menu.addSeparator()


    def fig_constructor(self):
        self.fig_widget.deleteLater()
        self.fig_widget = QWidget()

        # Добавление слайдера в макет
        self.layout.addWidget(self.fig_widget)

        fig_layout = QHBoxLayout(self.fig_widget)
        fig_layout.setContentsMargins(15, 15, 15, 15)
        fig_layout.setSpacing(10)
        fig_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        fig_layout.addWidget(self.canvas)
        fig_layout.addWidget(self.canvas2)

        self.plot_roc_curve()
        self.prec_rec_curve()

    def all_info_constructor(self):
        self.roc_info_constructor()
        self.matrix_constructor()
        self.all_info.deleteLater()
        self.all_info = QWidget()
        layout = QHBoxLayout()
        layout.addWidget(self.roc_info_widget)
        layout.addWidget(self.matrix_widget)
        self.all_info.setLayout(layout)
        self.layout.addWidget(self.all_info)

    def matrix_constructor(self):
        self.matrix_widget.deleteLater()
        self.matrix_widget = QWidget()

        # Создание макета сетки
        layout = QGridLayout()

        info = self.roc_data[self.data_len - self.cur_dot]
        # Матрица ошибок (confusion matrix)
        confusion_matrix = [
            [f'TN: {info["tn"]}', f'FP: {info["fp"]}'],
            [f'FN: {info["fn"]}', f'TP: {info["tp"]}']
        ]

        for i in range(len(confusion_matrix)):
            for j in range(len(confusion_matrix[i])):
                label = QLabel(confusion_matrix[i][j])
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)

                # # Создание границ для ячеек матрицы
                frame = QFrame()
                frame.setFrameShape(QFrame.Shape.Box)
                layout.addWidget(frame, i + 1, j + 1)

                # Добавление метки внутри ячейки
                layout.addWidget(label, i + 1, j + 1)

        self.matrix_widget.setLayout(layout)
        self.layout.addWidget(self.matrix_widget)

    def roc_info_constructor(self):
        self.roc_info_widget.deleteLater()
        self.roc_info_widget = QWidget()

        info = self.roc_data[self.data_len - self.cur_dot]
        print('INFO')
        print(info)
        print('ROC DATA')
        print(self.roc_data)

        layout = QVBoxLayout()

        # Добавление меток со значениями в макет
        label = QLabel(f'TRUE POSITIVE: {info["tp"]}  POSITIVE: {info["tp"] + info["fn"]}')
        layout.addWidget(label)
        label = QLabel(f'TRUE NEGATIVE: {info["tn"]}  NEGATIVE: {info["tn"] + info["fp"]}')
        layout.addWidget(label)
        label = QLabel(f'TPR: {info["tp"]} / {info["tp"] + info["fn"]} = {info["tpr"]:.3f} ')
        layout.addWidget(label)
        label = QLabel(f'FPR: {info["tn"]} / {info["tn"] + info["fp"]} = {info["fpr"]:.3f}')
        layout.addWidget(label)
        label = QLabel(f'PRECISION: {info["precision"]:.3f}')
        layout.addWidget(label)
        label = QLabel(f'RECALL: {info["recall"]:.3f}')
        layout.addWidget(label)

        self.roc_info_widget.setLayout(layout)
        # self.layout.addWidget(self.roc_info_widget)

    def checkbox_constructor(self):
        self.fields_widget.deleteLater()
        self.fields_widget = QWidget()

        # Создание виджета с полями
        fields_layout = QHBoxLayout(self.fields_widget)
        fields_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.fields_widget)

        # Создание флажков и привязка к функции обработки изменений
        self.checkboxes = []
        for i in range(len(self.y_true)):
            checkbox = QCheckBox()
            checkbox.setStyleSheet(
                """
                QCheckBox {
                    spacing: 5px;
                }
                QCheckBox::indicator {
                    width: 20px;
                    height: 20px;
                    border-radius: 10px;
                }
                QCheckBox::indicator:checked {
                    background-color: #FF0000;
                }
                QCheckBox::indicator:unchecked {
                    background-color: #0078D7;
                }
                """
            )
            checkbox.stateChanged.connect(lambda state, index=i: self.handle_checkbox_change(state, index))

            self.checkboxes.append(checkbox)
            fields_layout.addWidget(checkbox)
            fields_layout.setStretch(i, 1)

        self.plot_roc_curve()
        self.prec_rec_curve()

    def slider_constructor(self):
        self.slider_widget.deleteLater()
        self.slider_widget = QWidget()

        # Добавление слайдера в макет
        self.layout.addWidget(self.slider_widget)

        slider_layout = QVBoxLayout(self.slider_widget)

        # Создание метки для отображения номера деления
        self.slider_label = QLabel("1")

        # Создание слайдера
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        slider.setTickInterval(1)
        slider.setMinimum(1)
        slider.setMaximum(self.data_len + 1)
        slider.valueChanged.connect(self.slider_value_changed)
        slider_layout.addWidget(self.slider_label)
        slider_layout.addWidget(slider)

    def slider_value_changed(self, value):
        self.slider_label.setText(str(value))
        self.cur_dot = value - 1

        self.plot_roc_curve()
        self.prec_rec_curve()
        self.all_info_constructor()
        # self.roc_info_constructor()
        # self.matrix_constructor()

    def set_data_len(self):
        l = self.input_data_len()
        if l is None:
            return

        self.data_len = l
        self.y_true = [0 for _ in range(self.data_len)]
        self.fig_constructor()
        self.checkbox_constructor()
        self.slider_constructor()
        self.all_info_constructor()
        # self.roc_info_constructor()
        # self.matrix_constructor()

    def input_data_len(self):
        data_len, ok = QInputDialog.getText(self, 'Input', 'Enter data list size [1:30]:')

        try:
            width = int(data_len)
            if width < 1 or width > 30:
                self.invalid_data_len(data_len)
                return None
        except ValueError:
            self.invalid_data_len(data_len)
            return None

        if not ok:
            return None
        if ok:
            return width

    def invalid_data_len(self, l):
        error = QMessageBox()
        error.setIcon(QMessageBox.Icon.Critical)
        error.setText("Invalid data length")
        error.setInformativeText("Invalid data length: " + str(l))
        error.setWindowTitle("Invalid length")
        error.exec()

    def handle_checkbox_change(self, state, index):
        if state == 2:  # Флажок установлен
            self.y_true[index] = 1
        else:  # Флажок не установлен или поле пустое
            self.y_true[index] = 0
        print(f'TRUE {self.y_true}')

        self.plot_roc_curve()
        self.prec_rec_curve()
        self.all_info_constructor()
        # self.roc_info_constructor()
        # self.matrix_constructor()

    def true_false_positive(self, y_pred, y_real):
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        n = len(y_real)

        for i in range(n):
            if y_pred[i] == y_real[i] and y_real[i] == 1:
                true_positive += 1
            elif y_pred[i] == y_real[i] and y_real[i] == 0:
                true_negative += 1
            elif y_pred[i] == 1 and y_real[i] == 0:
                false_positive += 1
            elif y_pred[i] == 0 and y_real[i] == 1:
                false_negative += 1
            else:
                print('fail', y_pred[i], y_real[i])

        if true_positive == 0:
            precision = 0
        else:
            precision = true_positive / (true_positive + false_positive)

        if true_positive == 0:
            tpr = 0
        else:
            tpr = true_positive / (true_positive + false_negative)

        if false_positive == 0:
            fpr = 0
        else:
            fpr = false_positive / (false_positive + true_negative)

        recall = tpr

        return tpr, fpr, precision, recall, true_positive, true_negative, false_positive, false_negative

    def roc_from_scratch(self, y_real):
        self.roc_data = []
        n = len(y_real)
        probabilities = [1 for _ in range(n)]
        print(probabilities)

        roc = np.array([])
        for i in range(n + 1):

            if i != 0:
                probabilities[i - 1] = 0

            tpr, fpr, precision, recall, true_positive, true_negative, false_positive, false_negative = self.true_false_positive(probabilities, y_real)
            values = {
                'tpr': tpr,
                'fpr': fpr,
                'precision': precision,
                'recall': recall,
                'tp': true_positive,
                'tn': true_negative,
                'fp': false_positive,
                'fn': false_negative
            }
            self.roc_data.append(values)
            print('HERE', self.roc_data)

            roc = np.append(roc, [fpr, tpr])

        return roc.reshape(-1, 2)

    def plot_roc_curve(self):
        ROC = self.roc_from_scratch(self.y_true)

        # Очистка предыдущего графика (если был)
        self.figure.clear()
        # self.figure = Figure(figsize=(5, 5))

        # Построение графика ROC-AUC
        ax = self.figure.add_subplot(111, aspect='equal')
        ax.scatter(ROC[:, 0], ROC[:, 1], color='red')
        ax.scatter(ROC[self.data_len - self.cur_dot, 0], ROC[self.data_len - self.cur_dot, 1], color='blue')
        ax.plot(ROC[:, 0], ROC[:, 1], color='darkorange')
        ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc="lower right")

        # Обновление графического виджета
        self.canvas.draw()

    def prec_rec_data(self):
        prec = []
        rec = []
        for i in range(self.data_len + 1):
            prec.append(self.roc_data[self.data_len - i]['precision'])
            rec.append(self.roc_data[self.data_len - i]['recall'])

        return prec, rec

    def prec_rec_curve(self):
        prec, rec = self.prec_rec_data()

        # Очистка предыдущего графика (если был)
        self.figure2.clear()
        # self.figure2 = Figure(figsize=(5, 5))

        # Построение графика ROC-AUC
        ax = self.figure2.add_subplot(111, aspect='equal')
        ax.scatter(rec, prec, color='red')
        ax.plot(rec, prec, color='darkorange')
        ax.plot([1, 0], [0, 1], color='navy', linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.set_xlabel('recall')
        ax.set_ylabel('precision')
        ax.set_title('Precision recall graph')
        ax.legend(loc="lower right")

        # Обновление графического виджета
        self.canvas2.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.showFullScreen()  # Полноэкранный режим
    mainWindow.show()
    sys.exit(app.exec())