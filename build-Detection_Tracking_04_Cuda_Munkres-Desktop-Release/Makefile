#############################################################################
# Makefile for building: Detection_Tracking_04_Cuda_Munkres
# Generated by qmake (2.01a) (Qt 4.8.7) on: Fri Sep 29 18:20:53 2017
# Project:  ../Detection_Tracking_04_Cuda_Munkres/Detection_Tracking_04_Cuda_Munkres.pro
# Template: app
# Command: /usr/lib/x86_64-linux-gnu/qt4/bin/qmake -spec /usr/share/qt4/mkspecs/linux-g++-64 -o Makefile ../Detection_Tracking_04_Cuda_Munkres/Detection_Tracking_04_Cuda_Munkres.pro
#############################################################################

####### Compiler, tools and options

CC            = gcc
CXX           = g++
DEFINES       = 
CFLAGS        = -m64 -pipe -O2 -Wall -W $(DEFINES)
CXXFLAGS      = -m64 -pipe -std=c++11 -Wall -Wextra -pedantic -fopenmp -DDLIB_JPEG_SUPPORT -O2 -Wall -W $(DEFINES)
INCPATH       = -I/usr/share/qt4/mkspecs/linux-g++-64 -I../Detection_Tracking_04_Cuda_Munkres -I../../../app/dlib/build/dlib -I/usr/local/include/opencv -I/usr/local/include -I../Detection_Tracking_04_Cuda_Munkres -I.
LINK          = g++
LFLAGS        = -m64 -Wl,-O1
LIBS          = $(SUBLIBS)   -L/usr/lib -lopenblas -ljpeg -fopenmp -L /home/elian/Documents/app/dlib/build/dlib/ -ldlib -lcuda -lcudart -lcublas -lcurand -L/usr/local/cuda-8.0/lib64 -L/usr/local/lib -lopencv_cudabgsegm -lopencv_cudaobjdetect -lopencv_cudastereo -lopencv_stitching -lopencv_cudafeatures2d -lopencv_superres -lopencv_cudacodec -lopencv_videostab -lopencv_cudaoptflow -lopencv_cudalegacy -lopencv_cudawarping -lopencv_aruco -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_dpm -lopencv_freetype -lopencv_fuzzy -lopencv_hdf -lopencv_line_descriptor -lopencv_optflow -lopencv_reg -lopencv_saliency -lopencv_stereo -lopencv_structured_light -lopencv_phase_unwrapping -lopencv_rgbd -lopencv_viz -lopencv_surface_matching -lopencv_tracking -lopencv_datasets -lopencv_text -lopencv_face -lopencv_plot -lopencv_dnn -lopencv_xfeatures2d -lopencv_shape -lopencv_video -lopencv_ximgproc -lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_xobjdetect -lopencv_objdetect -lopencv_ml -lopencv_xphoto -lopencv_highgui -lopencv_videoio -lopencv_photo -lopencv_imgcodecs -lopencv_cudaimgproc -lopencv_cudafilters -lopencv_imgproc -lopencv_cudaarithm -lopencv_core -lopencv_cudev 
AR            = ar cqs
RANLIB        = 
QMAKE         = /usr/lib/x86_64-linux-gnu/qt4/bin/qmake
TAR           = tar -cf
COMPRESS      = gzip -9f
COPY          = cp -f
SED           = sed
COPY_FILE     = $(COPY)
COPY_DIR      = $(COPY) -r
STRIP         = strip
INSTALL_FILE  = install -m 644 -p
INSTALL_DIR   = $(COPY_DIR)
INSTALL_PROGRAM = install -m 755 -p
DEL_FILE      = rm -f
SYMLINK       = ln -f -s
DEL_DIR       = rmdir
MOVE          = mv -f
CHK_DIR_EXISTS= test -d
MKDIR         = mkdir -p

####### Output directory

OBJECTS_DIR   = ./

####### Files

SOURCES       = ../Detection_Tracking_04_Cuda_Munkres/main.cpp \
		../Detection_Tracking_04_Cuda_Munkres/munkres/munkres.cpp \
		../Detection_Tracking_04_Cuda_Munkres/validation.cpp 
OBJECTS       = main.o \
		munkres.o \
		validation.o
DIST          = /usr/share/qt4/mkspecs/common/unix.conf \
		/usr/share/qt4/mkspecs/common/linux.conf \
		/usr/share/qt4/mkspecs/common/gcc-base.conf \
		/usr/share/qt4/mkspecs/common/gcc-base-unix.conf \
		/usr/share/qt4/mkspecs/common/g++-base.conf \
		/usr/share/qt4/mkspecs/common/g++-unix.conf \
		/usr/share/qt4/mkspecs/qconfig.pri \
		/usr/share/qt4/mkspecs/features/qt_functions.prf \
		/usr/share/qt4/mkspecs/features/qt_config.prf \
		/usr/share/qt4/mkspecs/features/exclusive_builds.prf \
		/usr/share/qt4/mkspecs/features/default_pre.prf \
		/usr/share/qt4/mkspecs/features/release.prf \
		/usr/share/qt4/mkspecs/features/default_post.prf \
		/usr/share/qt4/mkspecs/features/link_pkgconfig.prf \
		/usr/share/qt4/mkspecs/features/shared.prf \
		/usr/share/qt4/mkspecs/features/unix/gdb_dwarf_index.prf \
		/usr/share/qt4/mkspecs/features/warn_on.prf \
		/usr/share/qt4/mkspecs/features/resources.prf \
		/usr/share/qt4/mkspecs/features/uic.prf \
		/usr/share/qt4/mkspecs/features/yacc.prf \
		/usr/share/qt4/mkspecs/features/lex.prf \
		/usr/share/qt4/mkspecs/features/include_source_dir.prf \
		../Detection_Tracking_04_Cuda_Munkres/Detection_Tracking_04_Cuda_Munkres.pro
QMAKE_TARGET  = Detection_Tracking_04_Cuda_Munkres
DESTDIR       = 
TARGET        = Detection_Tracking_04_Cuda_Munkres

first: all
####### Implicit rules

.SUFFIXES: .o .c .cpp .cc .cxx .C

.cpp.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

.cc.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

.cxx.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

.C.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

.c.o:
	$(CC) -c $(CFLAGS) $(INCPATH) -o "$@" "$<"

####### Build rules

all: Makefile $(TARGET)

$(TARGET):  $(OBJECTS)  
	$(LINK) $(LFLAGS) -o $(TARGET) $(OBJECTS) $(OBJCOMP) $(LIBS)

Makefile: ../Detection_Tracking_04_Cuda_Munkres/Detection_Tracking_04_Cuda_Munkres.pro  /usr/share/qt4/mkspecs/linux-g++-64/qmake.conf /usr/share/qt4/mkspecs/common/unix.conf \
		/usr/share/qt4/mkspecs/common/linux.conf \
		/usr/share/qt4/mkspecs/common/gcc-base.conf \
		/usr/share/qt4/mkspecs/common/gcc-base-unix.conf \
		/usr/share/qt4/mkspecs/common/g++-base.conf \
		/usr/share/qt4/mkspecs/common/g++-unix.conf \
		/usr/share/qt4/mkspecs/qconfig.pri \
		/usr/share/qt4/mkspecs/features/qt_functions.prf \
		/usr/share/qt4/mkspecs/features/qt_config.prf \
		/usr/share/qt4/mkspecs/features/exclusive_builds.prf \
		/usr/share/qt4/mkspecs/features/default_pre.prf \
		/usr/share/qt4/mkspecs/features/release.prf \
		/usr/share/qt4/mkspecs/features/default_post.prf \
		/usr/share/qt4/mkspecs/features/link_pkgconfig.prf \
		/usr/share/qt4/mkspecs/features/shared.prf \
		/usr/share/qt4/mkspecs/features/unix/gdb_dwarf_index.prf \
		/usr/share/qt4/mkspecs/features/warn_on.prf \
		/usr/share/qt4/mkspecs/features/resources.prf \
		/usr/share/qt4/mkspecs/features/uic.prf \
		/usr/share/qt4/mkspecs/features/yacc.prf \
		/usr/share/qt4/mkspecs/features/lex.prf \
		/usr/share/qt4/mkspecs/features/include_source_dir.prf
	$(QMAKE) -spec /usr/share/qt4/mkspecs/linux-g++-64 -o Makefile ../Detection_Tracking_04_Cuda_Munkres/Detection_Tracking_04_Cuda_Munkres.pro
/usr/share/qt4/mkspecs/common/unix.conf:
/usr/share/qt4/mkspecs/common/linux.conf:
/usr/share/qt4/mkspecs/common/gcc-base.conf:
/usr/share/qt4/mkspecs/common/gcc-base-unix.conf:
/usr/share/qt4/mkspecs/common/g++-base.conf:
/usr/share/qt4/mkspecs/common/g++-unix.conf:
/usr/share/qt4/mkspecs/qconfig.pri:
/usr/share/qt4/mkspecs/features/qt_functions.prf:
/usr/share/qt4/mkspecs/features/qt_config.prf:
/usr/share/qt4/mkspecs/features/exclusive_builds.prf:
/usr/share/qt4/mkspecs/features/default_pre.prf:
/usr/share/qt4/mkspecs/features/release.prf:
/usr/share/qt4/mkspecs/features/default_post.prf:
/usr/share/qt4/mkspecs/features/link_pkgconfig.prf:
/usr/share/qt4/mkspecs/features/shared.prf:
/usr/share/qt4/mkspecs/features/unix/gdb_dwarf_index.prf:
/usr/share/qt4/mkspecs/features/warn_on.prf:
/usr/share/qt4/mkspecs/features/resources.prf:
/usr/share/qt4/mkspecs/features/uic.prf:
/usr/share/qt4/mkspecs/features/yacc.prf:
/usr/share/qt4/mkspecs/features/lex.prf:
/usr/share/qt4/mkspecs/features/include_source_dir.prf:
qmake:  FORCE
	@$(QMAKE) -spec /usr/share/qt4/mkspecs/linux-g++-64 -o Makefile ../Detection_Tracking_04_Cuda_Munkres/Detection_Tracking_04_Cuda_Munkres.pro

dist: 
	@$(CHK_DIR_EXISTS) .tmp/Detection_Tracking_04_Cuda_Munkres1.0.0 || $(MKDIR) .tmp/Detection_Tracking_04_Cuda_Munkres1.0.0 
	$(COPY_FILE) --parents $(SOURCES) $(DIST) .tmp/Detection_Tracking_04_Cuda_Munkres1.0.0/ && (cd `dirname .tmp/Detection_Tracking_04_Cuda_Munkres1.0.0` && $(TAR) Detection_Tracking_04_Cuda_Munkres1.0.0.tar Detection_Tracking_04_Cuda_Munkres1.0.0 && $(COMPRESS) Detection_Tracking_04_Cuda_Munkres1.0.0.tar) && $(MOVE) `dirname .tmp/Detection_Tracking_04_Cuda_Munkres1.0.0`/Detection_Tracking_04_Cuda_Munkres1.0.0.tar.gz . && $(DEL_FILE) -r .tmp/Detection_Tracking_04_Cuda_Munkres1.0.0


clean:compiler_clean 
	-$(DEL_FILE) $(OBJECTS)
	-$(DEL_FILE) *~ core *.core


####### Sub-libraries

distclean: clean
	-$(DEL_FILE) $(TARGET) 
	-$(DEL_FILE) Makefile


check: first

compiler_rcc_make_all:
compiler_rcc_clean:
compiler_uic_make_all:
compiler_uic_clean:
compiler_image_collection_make_all: qmake_image_collection.cpp
compiler_image_collection_clean:
	-$(DEL_FILE) qmake_image_collection.cpp
compiler_yacc_decl_make_all:
compiler_yacc_decl_clean:
compiler_yacc_impl_make_all:
compiler_yacc_impl_clean:
compiler_lex_make_all:
compiler_lex_clean:
compiler_clean: 

####### Compile

main.o: ../Detection_Tracking_04_Cuda_Munkres/main.cpp ../Detection_Tracking_04_Cuda_Munkres/compareD.h \
		../Detection_Tracking_04_Cuda_Munkres/validation.h \
		../Detection_Tracking_04_Cuda_Munkres/munkres/munkres.h \
		../Detection_Tracking_04_Cuda_Munkres/munkres/matrix.h \
		../Detection_Tracking_04_Cuda_Munkres/munkres/matrix.cpp
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o main.o ../Detection_Tracking_04_Cuda_Munkres/main.cpp

munkres.o: ../Detection_Tracking_04_Cuda_Munkres/munkres/munkres.cpp ../Detection_Tracking_04_Cuda_Munkres/munkres/munkres.h \
		../Detection_Tracking_04_Cuda_Munkres/munkres/matrix.h \
		../Detection_Tracking_04_Cuda_Munkres/munkres/matrix.cpp
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o munkres.o ../Detection_Tracking_04_Cuda_Munkres/munkres/munkres.cpp

validation.o: ../Detection_Tracking_04_Cuda_Munkres/validation.cpp ../Detection_Tracking_04_Cuda_Munkres/validation.h \
		../Detection_Tracking_04_Cuda_Munkres/compareD.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o validation.o ../Detection_Tracking_04_Cuda_Munkres/validation.cpp

####### Install

install:   FORCE

uninstall:   FORCE

FORCE:

