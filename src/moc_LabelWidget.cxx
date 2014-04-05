/****************************************************************************
** Meta object code from reading C++ file 'LabelWidget.h'
**
** Created by: The Qt Meta Object Compiler version 63 (Qt 4.8.5)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "LabelWidget.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'LabelWidget.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.5. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_ImageLabel[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       2,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       2,       // signalCount

 // signals: signature, parameters, type, tag, flags
      18,   12,   11,   11, 0x05,
      52,   40,   11,   11, 0x05,

       0        // eod
};

static const char qt_meta_stringdata_ImageLabel[] = {
    "ImageLabel\0\0event\0clicked(QMouseEvent*)\0"
    "x1,y1,x2,y2\0rectSelected(int,int,int,int)\0"
};

void ImageLabel::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        ImageLabel *_t = static_cast<ImageLabel *>(_o);
        switch (_id) {
        case 0: _t->clicked((*reinterpret_cast< QMouseEvent*(*)>(_a[1]))); break;
        case 1: _t->rectSelected((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])),(*reinterpret_cast< int(*)>(_a[3])),(*reinterpret_cast< int(*)>(_a[4]))); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData ImageLabel::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject ImageLabel::staticMetaObject = {
    { &QLabel::staticMetaObject, qt_meta_stringdata_ImageLabel,
      qt_meta_data_ImageLabel, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &ImageLabel::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *ImageLabel::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *ImageLabel::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_ImageLabel))
        return static_cast<void*>(const_cast< ImageLabel*>(this));
    return QLabel::qt_metacast(_clname);
}

int ImageLabel::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QLabel::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 2)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 2;
    }
    return _id;
}

// SIGNAL 0
void ImageLabel::clicked(QMouseEvent * _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void ImageLabel::rectSelected(int _t1, int _t2, int _t3, int _t4)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)), const_cast<void*>(reinterpret_cast<const void*>(&_t3)), const_cast<void*>(reinterpret_cast<const void*>(&_t4)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}
static const uint qt_meta_data_TextLineDialog[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       1,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      16,   15,   15,   15, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_TextLineDialog[] = {
    "TextLineDialog\0\0okClicked()\0"
};

void TextLineDialog::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        TextLineDialog *_t = static_cast<TextLineDialog *>(_o);
        switch (_id) {
        case 0: _t->okClicked(); break;
        default: ;
        }
    }
    Q_UNUSED(_a);
}

const QMetaObjectExtraData TextLineDialog::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject TextLineDialog::staticMetaObject = {
    { &QDialog::staticMetaObject, qt_meta_stringdata_TextLineDialog,
      qt_meta_data_TextLineDialog, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &TextLineDialog::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *TextLineDialog::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *TextLineDialog::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_TextLineDialog))
        return static_cast<void*>(const_cast< TextLineDialog*>(this));
    return QDialog::qt_metacast(_clname);
}

int TextLineDialog::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QDialog::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 1)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 1;
    }
    return _id;
}
static const uint qt_meta_data_LabelWidget[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
      12,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      13,   12,   12,   12, 0x08,
      37,   35,   12,   12, 0x08,
      77,   12,   12,   12, 0x08,
      94,   88,   12,   12, 0x08,
     130,  121,   12,   12, 0x08,
     168,   12,   12,   12, 0x08,
     198,   12,   12,   12, 0x08,
     224,   12,   12,   12, 0x08,
     256,  251,   12,   12, 0x08,
     283,  251,   12,   12, 0x08,
     310,   12,   12,   12, 0x08,
     332,  328,   12,   12, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_LabelWidget[] = {
    "LabelWidget\0\0activate(QModelIndex)\0,\0"
    "mserItemActivated(QTreeWidgetItem*,int)\0"
    "save(bool)\0event\0imageClicked(QMouseEvent*)\0"
    "item,col\0mserItemChanged(QTreeWidgetItem*,int)\0"
    "generateNegativeSamples(bool)\0"
    "saveNegativeSamples(bool)\0"
    "mserItemSelectionChanged()\0text\0"
    "mserParamsChanged(QString)\0"
    "textLineIdChanged(QString)\0recalculate(bool)\0"
    ",,,\0rectSelected(int,int,int,int)\0"
};

void LabelWidget::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        LabelWidget *_t = static_cast<LabelWidget *>(_o);
        switch (_id) {
        case 0: _t->activate((*reinterpret_cast< const QModelIndex(*)>(_a[1]))); break;
        case 1: _t->mserItemActivated((*reinterpret_cast< QTreeWidgetItem*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 2: _t->save((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 3: _t->imageClicked((*reinterpret_cast< QMouseEvent*(*)>(_a[1]))); break;
        case 4: _t->mserItemChanged((*reinterpret_cast< QTreeWidgetItem*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 5: _t->generateNegativeSamples((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 6: _t->saveNegativeSamples((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 7: _t->mserItemSelectionChanged(); break;
        case 8: _t->mserParamsChanged((*reinterpret_cast< const QString(*)>(_a[1]))); break;
        case 9: _t->textLineIdChanged((*reinterpret_cast< const QString(*)>(_a[1]))); break;
        case 10: _t->recalculate((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 11: _t->rectSelected((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])),(*reinterpret_cast< int(*)>(_a[3])),(*reinterpret_cast< int(*)>(_a[4]))); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData LabelWidget::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject LabelWidget::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_LabelWidget,
      qt_meta_data_LabelWidget, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &LabelWidget::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *LabelWidget::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *LabelWidget::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_LabelWidget))
        return static_cast<void*>(const_cast< LabelWidget*>(this));
    return QWidget::qt_metacast(_clname);
}

int LabelWidget::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 12)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 12;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
