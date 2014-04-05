import sys
import os
from PyQt4.QtGui import *
import PyQt4
import argparse
import cv2
import numpy as np

class MyLabelWidget(QLabel):
    changed = PyQt4.QtCore.pyqtSignal()

    def __init__(self, *args, **kwargs):
        super(MyLabelWidget, self).__init__(*args, **kwargs)
        self.start_x = -1
        self.start_y = -1
        self.end_x = -1
        self.end_y = -1
        self.left_button = False

    @property
    def rect(self):
        min_x = min(self.start_x, self.end_x)
        min_y = min(self.start_y, self.end_y)
        max_x = max(self.start_x, self.end_x)
        max_y = max(self.start_y, self.end_y)

        return (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)

    def paintEvent(self, event):
        super(MyLabelWidget, self).paintEvent(event)

        r = self.rect
        if r[0] < 0 and r[1] < 0:
            return

        paint = QPainter()
        paint.begin(self)
        paint.drawRect(r[0], r[1], r[2], r[3])
        paint.end()

    def mousePressEvent(self, ev):
        self.left_button = ev.button() == PyQt4.QtCore.Qt.LeftButton
        self.start_x = ev.x()
        self.start_y = ev.y()
        self.end_x = ev.x()
        self.end_y = ev.y()

    def mouseMoveEvent(self, ev):
        self.left_button = ev.button() == PyQt4.QtCore.Qt.LeftButton
        self.end_x = ev.x()
        self.end_y = ev.y()
        self.repaint()

    def mouseReleaseEvent(self, ev):
        self.left_button = ev.button() == PyQt4.QtCore.Qt.LeftButton
        self.end_x = ev.x()
        self.end_y = ev.y()
        self.changed.emit()

        self.start_x = -1
        self.start_y = -1
        self.end_x = -1
        self.end_y = -1
        self.repaint()


def rgb2qimage(rgb):
	"""Convert the 3D numpy array `rgb` into a 32-bit QImage.  `rgb` must
	have three dimensions with the vertical, horizontal and RGB image axes.

	ATTENTION: This QImage carries an attribute `ndimage` with a
	reference to the underlying numpy array that holds the data. On
	Windows, the conversion into a QPixmap does not copy the data, so
	that you have to take care that the QImage does not get garbage
	collected (otherwise PyQt will throw away the wrapper, effectively
	freeing the underlying memory - boom!)."""
	if len(rgb.shape) != 3:
		raise ValueError("rgb2QImage can only convert 3D arrays")
	if rgb.shape[2] not in (3, 4):
		raise ValueError("rgb2QImage can expects the last dimension to contain exactly three (R,G,B) or four (R,G,B,A) channels")

	h, w, channels = rgb.shape

	# Qt expects 32bit BGRA data for color images:
	bgra = np.empty((h, w, 4), np.uint8, 'C')
	bgra[...,0] = rgb[...,2]
	bgra[...,1] = rgb[...,1]
	bgra[...,2] = rgb[...,0]
	if rgb.shape[2] == 3:
		bgra[...,3].fill(255)
		fmt = QImage.Format_RGB32
	else:
		bgra[...,3] = rgb[...,3]
		fmt = QImage.Format_ARGB32

	result = QImage(bgra.data, w, h, fmt)
	result.ndarray = bgra
	return result

class BoxLabelWidget(QWidget):
    def __init__(self, input_path=None, output_path=None, input_image_path=None, parent=None):
        super(BoxLabelWidget, self).__init__(parent)
        self.input_path = input_path
        self.input_image_path = input_image_path
        self.output_path = output_path
        self.directories = self._parse_directories()
        self.images = self._parse_images(self.directories)

        self._create_widgets()

    def _parse_directories(self):
        result = [os.path.join(self.input_path, p) for p in os.listdir(self.input_path) \
                if p not in ['.', '..']]
        result.sort()
        return result

    def _parse_images(self, gt_dirs):
        gt_dirs = [p.split('/')[-1] for p in gt_dirs]
        result = [os.path.join(self.input_image_path, p) for p in \
                os.listdir(self.input_image_path) if p.split('.')[0] in gt_dirs]
        result.sort()
        return result

    def on_train_item_activated(self, list_item_widget):
        image_path = list_item_widget.text()
        imgno = image_path.split('/')[-1]
        imgno = int(imgno.split('.')[0])
        self.active_image_number = imgno

        self._load_image(imgno)

    def _load_gt_pixel_lists(self, gt_path):
        dirs = [os.path.join(gt_path, p, 'letters.txt') for p in os.listdir(gt_path)]
        dirs = [d for d in dirs if os.path.exists(d)]
        results = []
        for d in dirs:
            with open(d) as fp:
                nodes = [l.strip() for l in fp.readlines()]
                for node in nodes:
                    path = os.path.join(os.path.dirname(d), 'node%s_contour.csv' % node)
                    with open(path) as fp2:
                        pixel_list = [[int(y) for y in l.strip().split(',')] \
                                for l in fp2.readlines()]
                        results.append(pixel_list)
        return results

    def _get_bounding_boxes(self, pixel_lists):
        results = []
        for pixel_list in pixel_lists:
            xs = [x for (x,y) in pixel_list]
            ys = [y for (x,y) in pixel_list]
            min_x = min(xs)
            max_x = max(xs)
            min_y = min(ys)
            max_y = max(ys)

            results.append(
                (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1))
        return results

    def _load_boxes(self, path):
        if not os.path.exists(path):
            return []

        with open(path) as fp:
            boxes = [[float(p) for p in l.strip().split(',')] for l in fp.readlines()]
            return boxes

    def _load_image(self, imgno):
        self.train_img_path = os.path.join(self.input_image_path, str(imgno)) + ".jpg"
        gt_path = os.path.join(self.input_path, str(imgno))
        self.pixels = self._load_gt_pixel_lists(gt_path)
        self.bounding_boxes = self._get_bounding_boxes(self.pixels)
        self.gt_boxes = self._load_boxes(os.path.join(self.output_path, str(imgno) + ".txt"))

        self._redraw_image()

    def _redraw_image(self):
        if not hasattr(self, 'train_img_path'):
            return
        img = cv2.imread(self.train_img_path)
        for pixel_list in self.pixels:
            for (x,y) in pixel_list:
                img[y,x,:] = np.array([255,0,0])
        #for (x,y,w,h) in self.bounding_boxes:
        #    cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), [255,0,0], 2)

        for (x,y,w,h) in self.gt_boxes:
            cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), [0,255,0], 2)

        self.original_shape = img.shape
        qimg = rgb2qimage(img) 
        self.train_image_pixmap = QPixmap.fromImage(qimg.rgbSwapped())
        self.train_image_pixmap = self.train_image_pixmap.scaled(
            500,500, PyQt4.QtCore.Qt.KeepAspectRatio, PyQt4.QtCore.Qt.SmoothTransformation)
        self.train_image_widget.setPixmap(self.train_image_pixmap)
        
    def _create_image_list_widget(self):
        self.image_list_widget = QListWidget()
        for path in self.images:
            self.image_list_widget.addItem(QListWidgetItem(path))
        self.image_list_widget.itemActivated.connect(self.on_train_item_activated)
        return self.image_list_widget

    def on_save_clicked(self):
        if not hasattr(self, 'active_image_number'):
            return

        out_path = os.path.join(self.output_path, str(self.active_image_number) + ".txt")
        with open(out_path,'w') as fp:
            for box in self.gt_boxes:
                line = ','.join([str(b) for b in box])
                fp.write(line + '\n')

    def _create_buttons(self):
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.on_save_clicked)
        return self.save_button

    def _intersect_area(self, r1, r2):
        (l1,t1,w1,h1) = r1
        (l2,t2,w2,h2) = r2

        b1 = t1 + h1
        r1 = l1 + w1
        b2 = t2 + h2
        r2 = l2 + w2

        right = min(r1,r2)
        left = max(l1,l2)
        top = max(t1,t2)
        bottom = min(b1,b2)


        if right < left or bottom < top: 
            return 0

        return (right - left + 1) * (bottom - top + 1)


    def _union_rect(self, r1, r2):
        (l1,t1,w1,h1) = r1
        (l2,t2,w2,h2) = r2

        b1 = t1 + h1 - 1
        r1 = l1 + w1 - 1
        b2 = t2 + h2 - 1
        r2 = l2 + w2 - 1

        left = min(l1, l2)
        right = max(r1, r2)
        top = min(t1, t2)
        bottom = max(b1, b2)

        return (left, top, right-left+1, bottom-top+1)

    def _shrink_rect(self, rect):
        rects = [r for r in self.bounding_boxes if self._intersect_area(rect,r) > 0]
        if not rects:
            return (0,0,0,0)
        r = rects[0]
        for rect in rects[1:]:
            r = self._union_rect(r, rect)
        return r

    def on_rect_created(self):
        scale1 = self.original_shape[1] / float(self.train_image_pixmap.width())
        scale2 = self.original_shape[0] / float(self.train_image_pixmap.height())
        orig_rect = np.asarray(self.train_image_widget.rect) * scale1
        if self.train_image_widget.left_button:
            rect = self._shrink_rect(orig_rect)
            if all([r == 0 for r in rect]):
                return
            self.gt_boxes.append(self._shrink_rect(orig_rect))
            self._redraw_image()
        else:
            print self.gt_boxes
            print orig_rect
            print [self._intersect_area(orig_rect, b) for b in self.gt_boxes]
            self.gt_boxes = [b for b in self.gt_boxes if self._intersect_area(orig_rect, b) <= 0]
            self._redraw_image()

    def _create_train_image_widget(self):
        self.train_image_pixmap = QPixmap()
        self.train_image_widget = MyLabelWidget()
        self.train_image_widget.setAlignment(PyQt4.QtCore.Qt.AlignTop | PyQt4.QtCore.Qt.AlignLeft)
        self.train_image_widget.setPixmap(self.train_image_pixmap)

        self.train_image_widget.changed.connect(self.on_rect_created)

        return self.train_image_widget

    def _create_widgets(self):
        vbox_right = QVBoxLayout()
        vbox_right.addWidget(self._create_image_list_widget())
        vbox_right.addWidget(self._create_buttons())

        vbox_left = QVBoxLayout()
        vbox_left.addWidget(self._create_train_image_widget())

        hbox = QHBoxLayout()
        hbox.addStretch(2)
        hbox.addLayout(vbox_left)
        hbox.addLayout(vbox_right)
        self.setLayout(hbox)


parser = argparse.ArgumentParser(description="Label some GT Boxes")
parser.add_argument('--input-path', help='The input path')
parser.add_argument('--input-image-path', help='The input image path')
parser.add_argument('--output-path', help='The output path')

args = parser.parse_args()

keys = ['input_path', 'output_path', 'input_image_path']
if not all([getattr(args,p) for p in keys]):
    parser.print_help()
    sys.exit(1)

for path in keys:
    if not os.path.isdir(getattr(args, path)):
        print "directory: '%s' does not exist" % args.input_path
        sys.exit(1)

a = QApplication(sys.argv)
w = BoxLabelWidget(
    input_path=args.input_path,
    output_path=args.output_path,
    input_image_path=args.input_image_path)

w.resize(320,240)
w.setWindowTitle("Label Program")
w.show()
sys.exit(a.exec_())
