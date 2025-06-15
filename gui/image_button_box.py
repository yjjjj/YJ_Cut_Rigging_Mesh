# coding:utf-8


from .pyside import *


class ImageButtonBox(QPushButton):

    def __init__(self, image_path, image_size, parent=None):
        super(ImageButtonBox, self).__init__(parent)
        self.path = image_path
        self.size = image_size

        self.setFixedSize(self.size[0], self.size[1])
        self.setIconSize(QSize(self.size[0], self.size[1]))
        self.setStyleSheet("background:transparent; border-width:0; border-style:outset;")

        pix_normal = QPixmap(self.path)
        pix_over = pix_normal.copy()
        painter = QPainter(pix_over)
        painter.fillRect(pix_over.rect(), QBrush(QColor(0, 0, 0, 100)))

        pix_pressed = pix_normal.copy()
        painter = QPainter(pix_pressed)
        painter.fillRect(pix_pressed.rect(), QBrush(QColor(0, 255, 255, 125)))

        painter.end()

        self._icon_normal = QIcon(pix_normal)
        self._icon_over = QIcon(pix_over)
        self._icon_pressed = QIcon(pix_pressed)
        self.setIcon(self._icon_normal)
        self.clicked.connect(self.pressed_event)

        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.restore_icon)

    def enterEvent(self, event):
        self.setIcon(self._icon_over)
        return super(ImageButtonBox, self).enterEvent(event)

    def leaveEvent(self, event):
        self.setIcon(self._icon_normal)
        return super(ImageButtonBox, self).leaveEvent(event)

    def pressed_event(self):
        self.setIcon(self._icon_pressed)
        self.timer.start(100)

    def restore_icon(self):
        self.setIcon(self._icon_normal)
