#
# Makefile for optimizer
#
# src/backend/optimizer/Makefile
#

subdir = src/backend/optimizer
top_builddir = ../../..
include $(top_builddir)/src/Makefile.global

SUBDIRS     = geqo path plan prep util

include $(top_srcdir)/src/backend/common.mk

install-data:
	$(MAKE) -C path/gpukde install-data

uninstall-data:
	$(MAKE) -C path/gpukde uninstall-data