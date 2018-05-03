import os
import sys

import cv2
import numpy as np
import qdarkstyle
from PyQt5 import QtCore, QtGui, uic, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileSystemModel

from algorithms.fcm import FCM
from algorithms.gk import GK
from file_path_manager import FilePathManager

FormClass = uic.loadUiType("ui.ui")[0]


class FilesTreeView(QtWidgets.QTreeView):
    def __init__(self, func, parent=None):
        super().__init__(parent)
        self.func = func
        self.setMinimumHeight(parent.height())
        self.setMinimumWidth(parent.width())

    def keyPressEvent(self, event):
        self.func(event)


class ImageWidget(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.raw_image = None
        self.image = None

    def setImage(self, image, raw_image):
        self.image = image
        self.raw_image = raw_image
        sz = image.size()
        self.setMinimumSize(sz)
        self.update()

    def paintEvent(self, event):
        qp = QtGui.QPainter()
        qp.begin(self)
        if self.image:
            qp.drawImage(QtCore.QPoint(0, 0), self.image)
        qp.end()


class Ui(QtWidgets.QMainWindow, FormClass):
    CMEANS = 0
    GK = 1

    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.setupUi(self)
        self.algorithm = "cmeans"
        self.root_path = FilePathManager.resolve("images")
        self.filesTreeView = FilesTreeView(self.keyPressEvent, self.filesTreeView)
        self.setup_events()

    def setup_events(self):
        model = QFileSystemModel()
        root = model.setRootPath(self.root_path)
        self.filesTreeView.setModel(model)
        self.filesTreeView.setRootIndex(root)
        self.filesTreeView.selectionModel().selectionChanged.connect(self.item_selection_changed_slot)
        self.segment_button.clicked.connect(self.segment)
        self.gk_radio.toggled.connect(self.select_gk)
        self.cmeans_radio.toggled.connect(self.select_cmeans)

    def select_gk(self, active):
        if not active:
            return
        self.algorithm = self.GK

    def select_cmeans(self, active):
        if not active:
            return
        self.algorithm = self.CMEANS

    def item_selection_changed_slot(self):
        index = self.filesTreeView.selectedIndexes()[0]
        item = self.filesTreeView.model().itemData(index)[0]
        image_path = "{}/{}".format(self.root_path, item)
        self.set_image(image_path)

    def set_image(self, image_path):
        pixmap = QtGui.QPixmap(image_path)
        scaled_pixmap = pixmap.scaled(self.imageLabel.size(), Qt.KeepAspectRatio)
        self.imageLabel.setPixmap(scaled_pixmap)

    def show_segmented_image(self):
        pixmap = QtGui.QPixmap("./temp.png")
        scaled_pixmap = pixmap.scaled(self.imageLabel.size(), Qt.KeepAspectRatio)
        self.output_image.setPixmap(scaled_pixmap)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Space:
            self.segment()

    def segment(self):
        index = self.filesTreeView.selectedIndexes()[0]
        item = self.filesTreeView.model().itemData(index)[0]
        image_path = "{}/{}".format(self.root_path, item)
        img = cv2.imread(image_path)
        x, y, z = img.shape
        img = img.reshape(x * y, z)

        m = int(self.m_text.text())
        n_clusters = int(self.n_clusters_text.text())
        iterations = int(self.iterations_text.text())

        if self.algorithm == self.CMEANS:
            algorithm = FCM(n_clusters=n_clusters, max_iter=iterations, m=m)
        else:
            algorithm = GK(n_clusters=n_clusters, max_iter=iterations, m=m)

        cluster_centers = algorithm.fit(img)
        output = algorithm.predict(img)
        img = cluster_centers[output].astype(np.int32).reshape(x, y, 3)
        image_path = f"./temp.png"
        cv2.imwrite(image_path, img)
        self.show_segmented_image()


if __name__ == '__main__':
    os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt5'
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_from_environment(is_pyqtgraph=True))
    ui = Ui()
    ui.setWindowTitle("Image Segmentation")
    ui.show()
    app.exec_()
