import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QPushButton, QFileDialog, QWidget, QHBoxLayout, \
    QMainWindow
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage


class ImageFusionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Fusion Application")
        self.resize(800, 600)

        # 初始化中央小部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 布局
        layout = QVBoxLayout(central_widget)

        # 图片选择布局
        img_layout = QHBoxLayout()

        # 左图框
        self.img1_label = QLabel("Click to Select Image 1")
        self.img1_label.setStyleSheet("border: 2px dashed gray;")
        self.img1_label.setFixedSize(300, 300)
        self.img1_label.setAlignment(Qt.AlignCenter)
        self.img1_label.mousePressEvent = self.select_image1
        img_layout.addWidget(self.img1_label)

        # 右图框
        self.img2_label = QLabel("Click to Select Image 2")
        self.img2_label.setStyleSheet("border: 2px dashed gray;")
        self.img2_label.setFixedSize(300, 300)
        self.img2_label.setAlignment(Qt.AlignCenter)
        self.img2_label.mousePressEvent = self.select_image2
        img_layout.addWidget(self.img2_label)

        # 运行按钮
        self.run_button = QPushButton("Run Fusion")
        self.run_button.clicked.connect(self.run_fusion)
        layout.addLayout(img_layout)
        layout.addWidget(self.run_button)

        # 结果显示
        self.result_label = QLabel()
        layout.addWidget(self.result_label)

    def select_image1(self, event):
        self.img1_path, _ = QFileDialog.getOpenFileName(self, "Select Image 1", "./", "Images (*.png *.jpg *.jpeg)")
        if self.img1_path:
            pixmap = QPixmap(self.img1_path)
            self.img1_label.setPixmap(pixmap.scaled(self.img1_label.size(), Qt.KeepAspectRatio))
            self.img1 = cv2.imread(self.img1_path)

    def select_image2(self, event):
        self.img2_path, _ = QFileDialog.getOpenFileName(self, "Select Image 2", "./", "Images (*.png *.jpg *.jpeg)")
        if self.img2_path:
            pixmap = QPixmap(self.img2_path)
            self.img2_label.setPixmap(pixmap.scaled(self.img2_label.size(), Qt.KeepAspectRatio))
            self.img2 = cv2.imread(self.img2_path)

    def run_fusion(self):
        if not hasattr(self, 'img1') or not hasattr(self, 'img2'):
            print("Please select both images")
            return

        # 图像融合逻辑
        fused_image = self.fuse_images_without_mask(self.img1, self.img2)

        # 显示结果
        self.show_image(fused_image, self.result_label)

    def fuse_images_without_mask(self, img1, img2):
        # 示例融合逻辑，您可以根据需要修改
        h, w, _ = img2.shape
        fused_image = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
        return fused_image

    def show_image(self, image, label):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = image.shape
        bytes_per_line = ch * w
        qimage = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio))




if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageFusionApp()
    window.show()
    sys.exit(app.exec_())
