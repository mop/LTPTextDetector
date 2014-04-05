/**
 *  This file is part of ltp-text-detector.
 *  Copyright (C) 2013 Michael Opitz
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef LABELWIDGET_H

#define LABELWIDGET_H

#include <string>
#include <vector>

#include <QWidget>
#include <QLabel>
#include <QTreeWidget>
#include <QListWidget>
#include <QPushButton>
#include <QDialog>

#include "MserTree.h"

/**
 *  This class is responsible for relaying the click events for the training image to the other class
 */
class ImageLabel : public QLabel {
    Q_OBJECT
public:
    ImageLabel(QWidget *parent) : QLabel(parent), _is_left(false), _is_right(false) {}
    virtual ~ImageLabel() {}
    virtual void mousePressEvent(QMouseEvent *event);
    virtual void mouseReleaseEvent(QMouseEvent *event);
    virtual void paintEvent(QPaintEvent *ev);
    virtual void mouseMoveEvent(QMouseEvent *event);
signals:
    void clicked(QMouseEvent *event);
    void rectSelected(int x1, int y1, int x2, int y2);

private:
    //! Was the left button clicked
    bool _is_left;
    //! Was the right button clicked
    bool _is_right;
    //! Start and end coordinates for the rectangular selection
    int _start_x, _start_y;
    int _end_x, _end_y;
};

class TextLineDialog : public QDialog {
    Q_OBJECT
public:
    TextLineDialog(cv::Mat img);
    bool is_ok() const { return _ok_clicked; }
    int get_line_number() const;
public slots:
    void okClicked();
private:
    QLabel *_image;
    QPixmap _pixmap;
    QLineEdit *_input;
    QPushButton *_ok;
    QPushButton *_cancel;
    bool _ok_clicked;
};

class LabelWidget : public QWidget
{
    Q_OBJECT
public:
    LabelWidget(const std::string &input_directory, const std::string &gt_dir, const std::string &output_directory, QWidget *parent=0);
    ~LabelWidget();

private slots:
    void activate(const QModelIndex&);
    void mserItemActivated(QTreeWidgetItem *, int);
    void save(bool);
    void imageClicked(QMouseEvent *event);
    void mserItemChanged(QTreeWidgetItem *item, int col);
    void generateNegativeSamples(bool);
    void saveNegativeSamples(bool);
    void mserItemSelectionChanged();
    void mserParamsChanged(const QString &text);
    void textLineIdChanged(const QString &text);
    void recalculate(bool);
    void rectSelected(int,int,int,int);

private:
    //! Parses the training directories and collects the data
    void parse_directories();
    //! Extracts the MSERs w/ OpenCV
    void reload_msers();
    //! Extracts the MSERs w/ OpenCV
    void extract_msers(int id);
    //! Loads the GT labels for this particular training image
    void extract_labelled_gt(int id);
    //! Colors the checked MSERs in the training image
    void draw_msers();
    //! Returns the selected (checked!) children of the given item
    std::vector<std::string> get_selected_children(QTreeWidgetItem *item);
    //! Converts the MSER tree to a QT QTreeWidgetItem hierarchy
    QTreeWidgetItem* mser_tree_to_qt(std::shared_ptr<TextDetector::MserNode> node);
    //! Samples negative training data
    void sample(int root, std::shared_ptr<TextDetector::MserNode> node, cv::Mat mask);
    //! Returns true if the whole tree is unchecked
    bool is_unchecked(QTreeWidgetItem *item);
    //! Loads the mser parameters
    void load_mser_params(int idx);
    //! Saves the mser parameters
    void save_mser_params(int idx);
    //! Sets the GUI mser parameters
    void set_mser_widgets();
    //! loads the text line ids
    void load_text_line_ids(int idx);
    //! saves the text line ids
    void save_text_line_ids(int idx);
    //! Generates the positive training mask
    cv::Mat generate_positive_mask();
    //! Helper method for getting the label for the given id
    int getLabelForUid(int uid);

    //! Returns the selected children of the given node.
    std::vector<std::string> get_selected_children(int uid);

    //! The directory of the training data
    std::string _input_directory;
    //! The directory of the ground-truth
    std::string _gt_directory;
    //! The output directory in which the labeled data is stored
    std::string _output_directory;

    //! The training image
    QLabel *_train_image;
    //! The mser preview window
    QLabel *_mser_image;
    //! The mser tree widget
    QTreeWidget *_mser_tree;
    //! The widget which stores the training images
    QListWidget *_train_images_list;
    //! Maximum  area
    QSlider *_maximum_area;
    //! Maximum  area
    QSlider *_minimum_area;
    //! The save button
    QPushButton *_save_button;
    //! The reload button
    QPushButton *_refresh_button;
    //! The save button
    QPushButton *_sample_button;
    //! The save negative button
    QPushButton *_save_negative_button;
    //! The delta parameter
    QLineEdit * _mser_delta;
    //! The minimum area param
    QLineEdit * _mser_minimum_area;
    //! The maximum area param
    QLineEdit * _mser_maximum_area;
    //! The maximum variation param
    QLineEdit * _mser_maximum_variation;
    //! The minimum diversity param
    QLineEdit * _mser_minimum_diversity;
    //! The stable param
    QLineEdit * _mser_stable;
    //! The text line id of the connected component
    QLineEdit *_text_line_id;
    //! The training pixmap storing the (scaled) training image
    QPixmap _train_pixmap;
    //! The mser pixmap showing the active MSER
    QPixmap _mser_pixmap;

    //! Stores the list of training image
    std::vector<std::string> _train_images;

    //! Stores the current MserTree
    std::shared_ptr<TextDetector::MserTree> _tree;
    //! Stores the size of the current training image
    cv::Size _current_size;
    //! Stores the current id of the training image (the index in the _train_images vector).
    int _current_id;

    //! Flag which indicates whether the mser-changed events should be handled
    //! This is used to avoid too many redraws when loading objects
    bool _is_loading;

    //! Stores a list of negative mser samples
    std::vector<std::shared_ptr<TextDetector::MserNode> > _negative_samples;
    std::vector<std::vector<int> > _negative_samples_uids;
    //! Stores the list of text line ids
    //! This vector is as long as the number of connected components
    std::vector<int> _text_lines;

    //! Stores the MSER params
    int _delta;             // mser delta
    int _min_area;          // mser minimum area
    int _max_area;          // mser maximum area
    double _max_variation;  // mser maximum variation
    double _min_diversity;  // mser minimum diversity
    bool _stable;           // mser stable

};

#endif /* end of include guard: LABELWIDGET_H */
