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
#include <text_detector/LabelWidget.h>    

#include <text_detector/HierarchicalMSER.h>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QGroupBox>
#include <QPixmap>
#include <QMouseEvent>
#include <QScrollBar>
#include <QLineEdit>
#include <QPainter>

#include <boost/filesystem.hpp>
#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace fs = boost::filesystem;

static std::vector<cv::Rect> read_boxes(const fs::path &path)
{
    std::ifstream ifs(path.generic_string().c_str());
    std::string line;
    std::vector<cv::Rect> result;

    while (std::getline(ifs, line)) {
        std::string parts;
        std::stringstream strm(line);
        std::vector<double> vals;
        while (std::getline(strm, parts, ',')) {
            double val;
            std::stringstream(parts) >> val;
            vals.push_back(val);
        }

        result.push_back(cv::Rect(vals[0], vals[1], vals[2], vals[3]));
    }
    return result;
}


LabelWidget::LabelWidget(
    const std::string &input_directory,
    const std::string &gt_dir,
    const std::string &output_directory,
    QWidget *parent)
: QWidget(parent), 
  _input_directory(input_directory),
  _gt_directory(gt_dir),
  _output_directory(output_directory),
  _delta(1), _min_area(1), _max_area(14400),
  _max_variation(0.5), _min_diversity(0.1), _stable(true)
{
    parse_directories();

    QHBoxLayout *hbox = new QHBoxLayout(this);

    hbox->setSpacing(1);

    QGroupBox *rightBox = new QGroupBox(this);
    QVBoxLayout *vbox = new QVBoxLayout(rightBox);
    vbox->setSpacing(1);

    _train_image = new ImageLabel(this);
    _train_image->setText("Training Image");
    _train_image->setAlignment(Qt::AlignTop | Qt::AlignLeft);
    connect(_train_image, SIGNAL(clicked(QMouseEvent*)), this, SLOT(imageClicked(QMouseEvent *)));
    connect(_train_image, SIGNAL(rectSelected(int,int,int,int)), this, SLOT(rectSelected(int,int,int,int)));
    _mser_image = new QLabel(this);
    _train_images_list = new QListWidget(this);

    for (size_t i = 0; i < _train_images.size(); i++) {
        _train_images_list->addItem(_train_images[i].c_str());
    }

    _mser_tree = new QTreeWidget(this);
    _mser_tree->setColumnCount(2);
    QStringList qlist;
    qlist << tr("ID") << tr("isChar");
    _mser_tree->setHeaderLabels(qlist);

    QGroupBox *button_box = new QGroupBox(this);
    QHBoxLayout *button_layout = new QHBoxLayout(button_box);
    _refresh_button = new QPushButton(tr("Recalculate MSERs"));
    _save_button = new QPushButton(tr("Save"));
    _sample_button = new QPushButton(tr("Sample"));
    _save_negative_button = new QPushButton(tr("Save Negative"));

    button_layout->addWidget(_refresh_button);
    button_layout->addWidget(_save_button);
    button_layout->addWidget(_sample_button);
    button_layout->addWidget(_save_negative_button);

    _maximum_area = new QSlider(Qt::Horizontal, this);
    _minimum_area = new QSlider(Qt::Horizontal, this);

    _mser_delta = new QLineEdit("1", this);
    _mser_minimum_area = new QLineEdit("1", this);
    _mser_maximum_area = new QLineEdit("14400", this);
    _mser_maximum_variation = new QLineEdit("0.5", this);
    _mser_minimum_diversity = new QLineEdit("0.1", this);
    _mser_stable = new QLineEdit("1", this);

    _text_line_id = new QLineEdit("", this);

    QGroupBox *param_box = new QGroupBox(this);
    QHBoxLayout *param_box_layout = new QHBoxLayout(param_box);

    param_box_layout->addWidget(_mser_delta);
    param_box_layout->addWidget(_mser_minimum_area);
    param_box_layout->addWidget(_mser_maximum_area);
    param_box_layout->addWidget(_mser_maximum_variation);
    param_box_layout->addWidget(_mser_minimum_diversity);
    param_box_layout->addWidget(_mser_stable);

    hbox->addWidget(_train_image);
    hbox->addWidget(rightBox);

    vbox->addWidget(_train_images_list);
    vbox->addWidget(_mser_image);
    vbox->addWidget(_maximum_area);
    vbox->addWidget(_minimum_area);
    vbox->addWidget(_mser_tree);
    vbox->addWidget(param_box);
    vbox->addWidget(_text_line_id);
    vbox->addWidget(button_box);

    connect(_train_images_list, SIGNAL(activated(const QModelIndex &)), this, SLOT(activate(const QModelIndex&)));
    connect(_mser_tree, SIGNAL(itemActivated(QTreeWidgetItem *, int)), this, SLOT(mserItemActivated(QTreeWidgetItem*, int)));
    connect(_mser_tree, SIGNAL(itemSelectionChanged(void)), this, SLOT(mserItemSelectionChanged(void)));
    connect(_mser_tree, SIGNAL(itemChanged(QTreeWidgetItem *, int)), this, SLOT(mserItemChanged(QTreeWidgetItem *, int)));
    connect(_refresh_button, SIGNAL(clicked(bool)), this, SLOT(recalculate(bool)));
    connect(_save_button, SIGNAL(clicked(bool)), this, SLOT(save(bool)));
    connect(_sample_button, SIGNAL(clicked(bool)), this, SLOT(generateNegativeSamples(bool)));
    connect(_save_negative_button, SIGNAL(clicked(bool)), this, SLOT(saveNegativeSamples(bool)));
    connect(_mser_delta, SIGNAL(textChanged(const QString &)), this, SLOT(mserParamsChanged(const QString &)));
    connect(_mser_minimum_area, SIGNAL(textChanged(const QString &)), this, SLOT(mserParamsChanged(const QString &)));
    connect(_mser_maximum_area, SIGNAL(textChanged(const QString &)), this, SLOT(mserParamsChanged(const QString &)));
    connect(_mser_maximum_variation, SIGNAL(textChanged(const QString &)), this, SLOT(mserParamsChanged(const QString &)));
    connect(_mser_minimum_diversity, SIGNAL(textChanged(const QString &)), this, SLOT(mserParamsChanged(const QString &)));
    connect(_mser_stable, SIGNAL(textChanged(const QString &)), this, SLOT(mserParamsChanged(const QString &)));
    connect(_text_line_id, SIGNAL(textChanged(const QString &)), this, SLOT(textLineIdChanged(const QString &)));

    setLayout(hbox);
}

void LabelWidget::textLineIdChanged(const QString &str)
{
    if (_mser_tree->selectedItems().size() == 0)
        return;
    QTreeWidgetItem *item = _mser_tree->selectedItems().front();

    QString txt = item->text(0);
    std::string stxt = txt.toStdString();
    int uid;
    std::stringstream(stxt) >> uid;

    int line_id;
    std::stringstream(_text_line_id->text().toStdString()) >> line_id;

    _text_lines[uid] = line_id;
}

void LabelWidget::recalculate(bool)
{
    _is_loading = true;
    reload_msers();
    draw_msers();
    _is_loading = false;
}

void LabelWidget::mserParamsChanged(const QString &qs)
{
    std::stringstream(_mser_delta->text().toStdString()) >> _delta;
    std::stringstream(_mser_minimum_area->text().toStdString()) >> _min_area;
    std::stringstream(_mser_maximum_area->text().toStdString()) >> _max_area;
    std::stringstream(_mser_maximum_variation->text().toStdString()) >> _max_variation;
    std::stringstream(_mser_minimum_diversity->text().toStdString()) >> _min_diversity;
    std::stringstream(_mser_stable->text().toStdString()) >> _stable;
}

void LabelWidget::mserItemSelectionChanged()
{
    if (_mser_tree->selectedItems().size() > 0) 
        mserItemActivated(_mser_tree->selectedItems().front(), 0);
}

void LabelWidget::mserItemChanged(QTreeWidgetItem *item, int col)
{
    if (!_is_loading)
        draw_msers();
}

LabelWidget::~LabelWidget() {}

void LabelWidget::parse_directories()
{
    std::vector<fs::path> paths;
    std::copy(fs::directory_iterator(_input_directory), fs::directory_iterator(), std::back_inserter(paths));
    std::sort(paths.begin(), paths.end());
    for (auto it = paths.begin(); it != paths.end(); it++) {
        fs::path path = *it;

        if (path.extension() != ".jpg") {
            std::cout << "Skipping: " << path << std::endl;
            continue;
        }

        fs::path basename = path.stem();
        basename += ".txt";

        fs::path gt_path(_gt_directory);
        gt_path += "/";
        gt_path += basename;

        if (!fs::exists(gt_path)) {
            std::cout << "Skipping: " << path << std::endl;
            continue;
        }

        _train_images.push_back(path.generic_string());
    }
}

void LabelWidget::extract_msers(int id)
{
    // load the gt
    cv::Mat img = cv::imread(_train_images[id]);
    cv::cvtColor(img, img, CV_RGB2GRAY);
    _current_size = cv::Size(img.cols, img.rows);

    fs::path path(_train_images[id]);
    fs::path basename = path.stem();
    basename += ".txt";

    fs::path gt_path(_gt_directory);
    gt_path += "/";
    gt_path += basename;

    std::vector<std::vector<cv::Point> > msers;
    std::vector<cv::Vec4i> hierarchy;
    std::vector<double> vars;
    cv::HierarchicalMSER mser(
        _delta, _min_area, _max_area, _max_variation, 
        _min_diversity, _stable);
    mser(img, msers, vars, hierarchy);

    _text_lines = std::vector<int>(msers.size()+1, -1);

    _tree.reset(new TextDetector::MserTree(msers, vars, hierarchy));

    std::vector<cv::Rect> boxes = read_boxes(gt_path);
    cv::Mat mask(img.rows, img.cols, CV_8UC1, cv::Scalar(0));
    for (auto i = boxes.begin(); i != boxes.end(); ++i) {
        cv::Mat range = mask.rowRange(
            std::max(0, i->y),
            std::min(mask.rows, i->y + i->height)
        ).colRange(
            std::max(0, i->x),
            std::min(mask.cols, i->x + i->width)
        );
        range = cv::Scalar(255);
    }

    _tree->prune(mask);
    std::cout << "#MSERS: " << msers.size() << std::endl;
}

void LabelWidget::save_mser_params(int idx)
{
    fs::path p(_train_images[idx]);
    fs::path out_path(_output_directory);
    out_path += "/";
    out_path += p.stem();
    out_path += "/params.txt";

    std::ofstream ofs(out_path.generic_string());
    ofs << _delta << "," << _min_area << "," << _max_area << "," <<
           _max_variation << "," << _min_diversity << "," << _stable << std::endl;
}

void LabelWidget::set_mser_widgets()
{
    {
    std::stringstream strm;
    strm << _delta;
    std::string tmp(strm.str());
    _mser_delta->setText(QString(tmp.c_str()));
    }
    {
    std::stringstream strm;
    strm << _min_area;
    std::string tmp(strm.str());
    _mser_minimum_area->setText(QString(tmp.c_str()));
    }
    {
    std::stringstream strm;
    strm << _max_area;
    std::string tmp(strm.str());
    _mser_maximum_area->setText(QString(tmp.c_str()));
    }
    {
    std::stringstream strm;
    strm << _max_variation;
    std::string tmp(strm.str());
    _mser_maximum_variation->setText(QString(tmp.c_str()));
    }
    {
    std::stringstream strm;
    strm << _min_diversity;
    std::string tmp(strm.str());
    _mser_minimum_diversity->setText(QString(tmp.c_str()));
    }
    {
    std::stringstream strm;
    strm << int(_stable);
    std::string tmp(strm.str());
    _mser_stable->setText(QString(tmp.c_str()));
    }
}

void LabelWidget::load_mser_params(int id)
{
    fs::path p(_train_images[id]);
    fs::path out_path(_output_directory);
    out_path += "/";
    out_path += p.stem();
    out_path += "/params.txt";

    // check if the params.txt exists
    if (!fs::exists(out_path)) {
        // use default params
        _delta = 1;
        _min_area = 1;
        _max_area = 14400;
        _max_variation = 0.5;
        _min_diversity = 0.1;
        _stable = true;
        set_mser_widgets();
        return;
    }

    std::cout << "Loading the MSER Parameters: " << out_path << std::endl;
    {
    std::ifstream ifs(out_path.generic_string());

    std::string line;
    std::getline(ifs, line);
    std::stringstream ss(line);
    std::vector<double> parts;
    std::string part;
    while (std::getline(ss, part, ',')) {
        double tmp;
        std::stringstream(part) >> tmp;
        parts.push_back(tmp);
    }

    assert(parts.size() == 6);

    _delta = int (parts.at(0));
    _min_area = int (parts.at(1));
    _max_area = int (parts.at(2));
    _max_variation = parts.at(3);
    _min_diversity = parts.at(4);
    _stable = bool(parts.at(5));
    ifs.close();
    }

    set_mser_widgets();
}

void LabelWidget::reload_msers()
{
    extract_msers(_current_id);

    _negative_samples_uids.resize(_tree->_root->children.size());

    _train_pixmap.load(_train_images[_current_id].c_str());
    _train_pixmap = _train_pixmap.scaled(500, 500, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    _train_image->setPixmap(_train_pixmap);

    QTreeWidgetItem *root = mser_tree_to_qt(_tree->_root);
    _mser_tree->clear();
    _mser_tree->addTopLevelItem(root);
}

void LabelWidget::activate(const QModelIndex& idx)
{
    _negative_samples.clear();
    _negative_samples_uids.clear();
    _is_loading = true;
    _current_id = idx.row();

    load_mser_params(_current_id);
    reload_msers();

    load_text_line_ids(_current_id);
    extract_labelled_gt(_current_id);
    draw_msers();
    _is_loading = false;
}

void LabelWidget::load_text_line_ids(int id)
{
    fs::path p(_train_images[id]);
    fs::path out_path(_output_directory);
    out_path += "/";
    out_path += p.stem();
    out_path += "/text_lines.txt";

    if (!fs::exists(out_path)) 
        return;

    std::ifstream ifs(out_path.generic_string());
    std::string line;
    while (std::getline(ifs, line)) {
        int uid, line_id;

        std::stringstream ss(line);
        std::string str_uid, str_line_id;
        std::getline(ss, str_uid, ',');
        std::getline(ss, str_line_id, ',');

        std::stringstream(str_uid) >> uid;
        std::stringstream(str_line_id) >> line_id;

        _text_lines[uid] = line_id;
    }
}

void LabelWidget::save_text_line_ids(int id)
{
    fs::path p(_train_images[id]);
    fs::path out_path(_output_directory);
    out_path += "/";
    out_path += p.stem();
    out_path += "/text_lines.txt";

    if (!fs::is_directory(out_path.parent_path())) {
        if (!fs::create_directory(out_path.parent_path())) {
            std::cerr << "error: could not create " << out_path << std::endl;
            return;
        }
    }

    std::ofstream ofs(out_path.generic_string());

    for (int i = 0; i < _text_lines.size(); i++) {
        ofs << i << "," << _text_lines[i] << std::endl;
    }
    ofs.close();
}

QTreeWidgetItem* LabelWidget::mser_tree_to_qt(std::shared_ptr<TextDetector::MserNode> node)
{
    QTreeWidgetItem *item = new QTreeWidgetItem();
    std::stringstream uid; 
    uid << node->uid;
    item->setText(0, uid.str().c_str());
    item->setFlags(Qt::ItemIsUserCheckable | Qt::ItemIsSelectable | Qt::ItemIsEnabled);
    item->setCheckState(1, Qt::Unchecked);

    for (size_t i = 0; i < node->children.size(); i++) {
        QTreeWidgetItem *child = mser_tree_to_qt(node->children[i]);
        item->addChild(child);
    }
    return item;
}

void LabelWidget::mserItemActivated(QTreeWidgetItem *item, int)
{
    QString txt = item->text(0);
    std::string stxt = txt.toStdString();
    int uid;
    std::stringstream(stxt) >> uid;

    std::shared_ptr<TextDetector::MserNode> node = _tree->find(uid);
    cv::Mat img(_current_size.height, _current_size.width, CV_8UC3, cv::Scalar(0,0,0));
    node->draw_node(img);


    QImage qimg((unsigned char *)img.ptr<unsigned char*>(), _current_size.width, _current_size.height, img.step, QImage::Format_RGB888);
    int pos = _mser_tree->horizontalScrollBar()->value();
    _mser_pixmap = QPixmap::fromImage(qimg);
    _mser_pixmap = _mser_pixmap.scaled(400, 400, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    _mser_image->setPixmap(_mser_pixmap);

    std::stringstream ss;
    ss << _text_lines[uid];
    _text_line_id->setText(QString::fromStdString(ss.str()));
    std::cout << "ACTIVATED!: " << std::endl;
    _mser_tree->horizontalScrollBar()->setSliderPosition(pos);
}

void LabelWidget::save(bool)
{
    fs::path path(_train_images[_current_id]);
    fs::path out_path(_output_directory);
    out_path += "/";
    out_path += path.stem();
    out_path += "/";

    if (!fs::exists(out_path)) {
        if (!fs::create_directory(out_path)) {
            std::cout << "Error: " << out_path << " could not be created" << std::endl;
            return;
        }
    }

    std::cout << "Doing: " << out_path << std::endl;
    
    for (size_t i = 0; i < _tree->_root->children.size(); i++) {
        //cv::Mat result(img.rows, img.cols, CV_8UC3, cv::Scalar(0,0,0));
        //tree._root->children[i]->draw_tree(result);

        // the output path:

        // write a directory of images foreach node...
        fs::path out(out_path);
        std::stringstream strm;
        strm << "/" << i << "/";
        out += strm.str();

        if (!fs::exists(out)) {
            if (!fs::create_directory(out)) {
                std::cout << "Error creating: " << out << std::endl;
                continue;
            }
        }

        // dump the .png files
        _tree->_root->children[i]->dump_tree(
            out.generic_string(),
            _current_size);
        // dump the .csv file
        fs::path csv_path(out);
        csv_path += "data.csv";
        std::ofstream csv(csv_path.generic_string());
        _tree->_root->children[i]->dump_csv(csv);
        // dump the .gv file
        _tree->dump_child(out.generic_string(), i);

        std::vector<std::string> gt(get_selected_children(_tree->_root->children[i]->uid));

        if (!gt.empty()) {
            fs::path gt_path(out);
            gt_path += "letters.txt";
            std::ofstream gt_file(gt_path.generic_string());
            std::cout << gt_path << std::endl;
            for (size_t j = 0; j < gt.size(); j++) {
                gt_file << gt[j] << std::endl;
            }
        }
    }

    save_mser_params(_current_id);
    save_text_line_ids(_current_id);
}

std::vector<std::string> LabelWidget::get_selected_children(int uid)
{
    std::stringstream strm;
    strm << uid;
    std::string s = strm.str();
    QList<QTreeWidgetItem *> list = _mser_tree->findItems(
        QString(s.c_str()), Qt::MatchFixedString | Qt::MatchRecursive, 0); 
    if (list.empty()) {
        std::cout << "ERROR, could not find uid: " << uid << std::endl;
        return std::vector<std::string>();
    }

    return get_selected_children(list.front());
}

std::vector<std::string> LabelWidget::get_selected_children(QTreeWidgetItem *item)
{
    std::vector<std::string> result;
    if (item->checkState(1) == Qt::Checked) {
        result.push_back(item->text(0).toStdString());
    }

    for (int i = 0; i < item->childCount(); i++) {
        std::vector<std::string> tmp(get_selected_children(item->child(i)));
        if (!tmp.empty()) {
            result.insert(result.end(), tmp.begin(), tmp.end());
        }
    }

    return result;
}

static void expand_item(QTreeWidgetItem *item)
{
    item->setExpanded(true);
    QTreeWidgetItem *prev = NULL;
    QTreeWidgetItem *p = NULL;
    while ((p = item->parent()) != prev) {
        p->setExpanded(true);
        prev = p;
    }
}

void LabelWidget::imageClicked(QMouseEvent *event)
{
    // search for the mser-element which is clicked by the user
    int x = event->x();
    int y = event->y();

    float sy = float(_train_pixmap.height()) / float(_current_size.height);
    float sx = float(_train_pixmap.width()) / float(_current_size.width);
    x = x / sx;
    y = y / sy;

    std::cout << "MIN: " << _minimum_area->value() << std::endl;
    std::cout << "MAX: " << _maximum_area->value() << std::endl;
    float min_area = _minimum_area->value() / 100.0 * _current_size.height * _current_size.width;
    float max_area = _maximum_area->value() / 100.0 * _current_size.height * _current_size.width;
    int uid = _tree->find_by_coordinates(x, y, min_area, max_area);
    
    if (uid > 0) {
        std::stringstream ss;
        ss << uid;
        std::string s(ss.str());
        _mser_tree->clearSelection();
        QList<QTreeWidgetItem *> list = _mser_tree->findItems(
            QString(s.c_str()), Qt::MatchFixedString | Qt::MatchRecursive, 0); 
        for (QList<QTreeWidgetItem *>::iterator it = list.begin(); it != list.end(); ++it) {
            QTreeWidgetItem *item = *it;
            item->setSelected(true);
            mserItemActivated(item, 0);
            expand_item(item);
            _mser_tree->scrollToItem(item);
            _mser_tree->horizontalScrollBar()->setSliderPosition(_mser_tree->horizontalScrollBar()->maximum());
        }
    }
}

void LabelWidget::extract_labelled_gt(int id)
{
    fs::path p(_train_images[id]);
    fs::path out_path(_output_directory);
    out_path += "/";
    out_path += p.stem();
    out_path += "/";

    if (!fs::is_directory(out_path)) return;

    for (fs::directory_iterator it(out_path); it != fs::directory_iterator(); ++it) {
        fs::path tree_path(*it);
        tree_path += "/letters.txt";
        if (fs::exists(tree_path)) {
            // extract the letters
            std::ifstream ifs(tree_path.generic_string());
            std::string line;
            while (std::getline(ifs, line)) {
                QList<QTreeWidgetItem *> list = _mser_tree->findItems(
                    QString(line.c_str()), Qt::MatchFixedString | Qt::MatchRecursive, 0); 
                if (!list.empty()) {
                    QTreeWidgetItem *item = *list.begin();
                    item->setCheckState(1, Qt::Checked);
                    expand_item(item);
                }
            }
        }

        tree_path = *it;
        int child_id;
        std::stringstream(tree_path.filename().c_str()) >> child_id;
        tree_path += "/negatives.txt";
        if (fs::exists(tree_path)) {
            std::ifstream ifs(tree_path.generic_string());
            std::string line;
            while (std::getline(ifs, line)) {
                int idx;
                std::stringstream(line) >> idx;
                std::shared_ptr<TextDetector::MserNode> node(_tree->find(idx));
                if (node) {
                    _negative_samples_uids[child_id].push_back(idx);
                    _negative_samples.push_back(_tree->find(idx));
                }
            }
        }
    }
}

cv::Mat LabelWidget::generate_positive_mask()
{
    cv::Mat img(_current_size.height, _current_size.width, CV_8UC3, cv::Scalar(0,0,0));
    for (int i = 0; i < _tree->_root->children.size(); i++) {
        std::vector<std::string> selected_elements(
            get_selected_children(_tree->_root->children[i]->uid));
        for (size_t j = 0; j < selected_elements.size(); j++) {
            int uid;
            std::stringstream(selected_elements[j]) >> uid;

            std::shared_ptr<TextDetector::MserNode> node = _tree->find(uid);
            assert(node); 
            node->draw_node(img);
        }
    }

    cv::cvtColor(img, img, CV_RGB2GRAY);
    return img > 0;
}

void LabelWidget::draw_msers()
{
    cv::Mat img = cv::imread(_train_images[_current_id].c_str());

    // draw the training data
    for (size_t i = 0; i < _tree->_root->children.size(); i++) {
        std::vector<std::string> selected_elements(
            get_selected_children(_tree->_root->children[i]->uid));
        for (size_t j = 0; j < selected_elements.size(); j++) {
            int uid;
            std::stringstream(selected_elements[j]) >> uid;

            std::shared_ptr<TextDetector::MserNode> node = _tree->find(uid);
            assert(node); 
            node->draw_node(img);
        }
    }

    // draw the negative data
    for (size_t i = 0; i < _negative_samples.size(); i++) {
        _negative_samples[i]->draw_node(img, cv::Vec3b(0,0,255));
    }

    QImage qimg((unsigned char *)img.ptr<unsigned char*>(), img.cols, img.rows, img.step, QImage::Format_RGB888);
    _train_pixmap = QPixmap::fromImage(qimg.rgbSwapped());
    _train_pixmap = _train_pixmap.scaled(500, 500, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    _train_image->setPixmap(_train_pixmap);
}

void LabelWidget::generateNegativeSamples(bool)
{
    cv::Mat mask = generate_positive_mask();
    QTreeWidgetItem *item = _mser_tree->invisibleRootItem();
    item = item->child(0);
    _negative_samples.clear();
    for (int i = 0; i < item->childCount(); i++) {
        QTreeWidgetItem *child = item->child(i);
        // generate samples for this tree
        int uid;
        std::stringstream(child->text(0).toStdString()) >> uid;
        for (size_t j = 0; j < _tree->_root->children.size(); j++) {
            if (_tree->_root->children[j]->uid == uid) {
                sample(j, _tree->_root->children[j], mask);
            }
        }
    }
    draw_msers();
}

void LabelWidget::sample(int root, std::shared_ptr<TextDetector::MserNode> node, cv::Mat mask)
{
    // check if overlap
    bool overlap = false;
    for (int i = 0; i < node->contour.size(); i++) {
        if (mask.at<unsigned char>(node->contour[i].y, node->contour[i].x) > 0) {
            overlap = true; break;;
        }
    }
    if (!overlap) {
        _negative_samples.push_back(node);
        _negative_samples_uids[root].push_back(node->uid);
    }

    for (size_t i = 0; i < node->children.size(); i++) {
        sample(root, node->children[i], mask);
    }
}

bool LabelWidget::is_unchecked(QTreeWidgetItem *item)
{
    if (item->checkState(1) == Qt::Checked) {
        return false; // checked :(
    }
    for (int i = 0; i < item->childCount(); i++) {
        QTreeWidgetItem *n = item->child(i);
        if (is_unchecked(n) == false) {
            return false;
        }
    }
    return true; // yay
}

void LabelWidget::saveNegativeSamples(bool)
{
    fs::path p(_train_images[_current_id]);
    fs::path out_path(_output_directory);
    out_path += "/";
    out_path += p.stem();
    out_path += "/";

    for (size_t i = 0; i < _tree->_root->children.size(); i++) {
        fs::path out(out_path);
        std::stringstream strm;
        strm << "/" << i << "/";
        out += strm.str();

        if (!fs::exists(out)) {
            fs::create_directory(out);
        }

        if (!_negative_samples_uids[i].empty()) {

            fs::path neg_path(out);
            neg_path += "negatives.txt";
            std::ofstream neg_file(neg_path.generic_string());
            std::cout << neg_path << std::endl;
            for (size_t j = 0; j < _negative_samples_uids[i].size(); j++) {
                neg_file << _negative_samples_uids[i][j] << std::endl;
            }
        }
    }
}

int LabelWidget::getLabelForUid(int uid)
{
    std::stringstream ss;
    ss << uid;
    std::string str_uid(ss.str());

    QList<QTreeWidgetItem *> list = _mser_tree->findItems(
        QString(str_uid.c_str()), Qt::MatchFixedString | Qt::MatchRecursive, 0); 
    if (list.empty()) return -1;
    QTreeWidgetItem *item = list.front();
    return item->checkState(1) == Qt::Checked ? 1 : -1;
}

void LabelWidget::rectSelected(int x1, int y1, int x2, int y2)
{
    float sy = float(_train_pixmap.height()) / float(_current_size.height);
    float sx = float(_train_pixmap.width()) / float(_current_size.width);
    x1 = x1 / sx;
    y1 = y1 / sy;
    x2 = x2 / sx;
    y2 = y2 / sy;

    std::vector<std::shared_ptr<TextDetector::MserNode> > nodes = _tree->find_by_bounding_rect(cv::Rect(cv::Point(x1,y1), cv::Point(x2,y2)));
    std::vector<int> lbls(nodes.size(),-1);
    LabelWidget *widget = this;
    nodes.erase(std::remove_if(nodes.begin(), nodes.end(), [&widget] (std::shared_ptr<TextDetector::MserNode> node) -> int {
        return widget->getLabelForUid(node->uid) == -1;
    }), nodes.end());

    cv::Mat img(_current_size.height, _current_size.width, CV_8UC3, cv::Scalar(0));
    for (std::shared_ptr<TextDetector::MserNode> n : nodes) {
        n->draw_node(img);
    }

    TextLineDialog dlg(img);
    dlg.exec();
    if (dlg.is_ok()) {
        int number = dlg.get_line_number();

        for (std::shared_ptr<TextDetector::MserNode> n : nodes) {
            _text_lines[n->uid] = number;
        }
    }

}

void ImageLabel::mousePressEvent(QMouseEvent *event)
{
    _is_left = event->buttons() == Qt::LeftButton;
    _is_right = event->buttons() == Qt::RightButton;
    if (_is_right) {
        _start_x = event->x();
        _start_y = event->y();
        _end_x = event->x();
        _end_y = event->y();
    } else {
        _start_x = _start_y = _end_x = _end_y = -1;
    }
    //if (event->buttons() == Qt::LeftButton) {
    //    emit clicked(event);
    //}
}

void ImageLabel::mouseMoveEvent(QMouseEvent *event)
{
    if (_is_right) {
        _end_x = event->x();
        _end_y = event->y();
        repaint();
    }
}

void ImageLabel::mouseReleaseEvent(QMouseEvent *event) 
{
    if (_is_left) {
        emit clicked(event);
        _is_left = false;
    }
    if (_is_right) {
        int x1 = std::min(_start_x, _end_x);
        int x2 = std::max(_start_x, _end_x);
        int y1 = std::min(_start_y, _end_y);
        int y2 = std::max(_start_y, _end_y);
        emit rectSelected(x1,y1,x2,y2);

        _start_x = _start_y = _end_x = _end_y = -1;
        _is_right = false;
        repaint();
    }
}

void ImageLabel::paintEvent(QPaintEvent *event)
{
    // paint the parent
    QLabel::paintEvent(event);

    if (_is_right && _start_x >= 0 && _start_y >= 0) {
        int x1 = std::min(_start_x, _end_x);
        int x2 = std::max(_start_x, _end_x);
        int y1 = std::min(_start_y, _end_y);
        int y2 = std::max(_start_y, _end_y);

        QRectF rect(x1,y1,x2 - x1 + 1, y2 - y1 + 1);

        QPainter painter(this);
        painter.drawRect(rect);
        painter.end();
    }
}


TextLineDialog::TextLineDialog(cv::Mat img)
: _ok_clicked(false)
{
    // create the image
    _image = new QLabel(this);
    QImage qimg((unsigned char *)img.ptr<unsigned char*>(), img.cols, img.rows, img.step, QImage::Format_RGB888);
    _pixmap = QPixmap::fromImage(qimg);
    _pixmap = _pixmap.scaled(400, 400, Qt::KeepAspectRatio, Qt::SmoothTransformation);

    _image->setPixmap(_pixmap);

    _input = new QLineEdit(this);

    _ok = new QPushButton(tr("OK"));
    _ok->setDefault(true);
    _cancel = new QPushButton(tr("Cancel"));

    QHBoxLayout *hbox = new QHBoxLayout();
    hbox->addWidget(_ok);
    hbox->addWidget(_cancel);
    hbox->addStretch();

    QVBoxLayout *vbox = new QVBoxLayout();
    vbox->addWidget(_image);
    vbox->addWidget(_input);
    vbox->addLayout(hbox);

    setLayout(vbox);

    connect(_ok, SIGNAL(clicked()), this, SLOT(okClicked()));
    connect(_cancel, SIGNAL(clicked()), this, SLOT(close()));
}

void TextLineDialog::okClicked() { _ok_clicked = true; close(); }

int TextLineDialog::get_line_number() const 
{
    int no;
    std::stringstream(_input->text().toStdString()) >> no;
    return no;
}
