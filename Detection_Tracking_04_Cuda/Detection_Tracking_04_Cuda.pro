TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

CUDA_LIBS = -lcuda -lcudart -lcublas -lcurand -L/usr/local/cuda-8.0/lib64

PKGCONFIG += opencv
CONFIG += link_pkgconfig

INCLUDEPATH += /home/elian/Documents/app/dlib/build/dlib


LIBS +=  -L/usr/lib \
         -lopenblas \
        -ljpeg \
        -fopenmp \
        -L /home/elian/Documents/app/dlib/build/dlib/ -ldlib

LIBS += $$CUDA_LIBS

QMAKE_CXXFLAGS += -std=c++11 -Wall -Wextra -pedantic -fopenmp -DDLIB_JPEG_SUPPORT

SOURCES += main.cpp \
    validation.cpp

HEADERS += \
    util.h \
    validation.h \
    compareD.h

