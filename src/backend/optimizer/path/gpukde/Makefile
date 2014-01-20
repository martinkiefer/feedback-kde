#-------------------------------------------------------------------------
#
# Makefile--
#    Makefile for optimizer/path
#
# IDENTIFICATION
#    src/backend/optimizer/path/Makefile
#
#-------------------------------------------------------------------------

subdir = src/backend/optimizer/path/gpukde
top_builddir = ../../../../..
include $(top_builddir)/src/Makefile.global

OBJS = ocl_utilities.o ocl_estimator.o   

include $(top_srcdir)/src/backend/common.mk

# Install kernel files
.PHONY: install-data

install: all installdirs
	$(INSTALL_DATA) $(srcdir)/kernel/sum.cl '$(DESTDIR)$(datadir)/kernel/sum.cl'
	$(INSTALL_DATA) $(srcdir)/kernel/kde.cl '$(DESTDIR)$(datadir)/kernel/kde.cl'

installdirs:
	$(MKDIR_P) '$(DESTDIR)$(datadir)/kernel'

uninstall:
	rm -f '$(DESTDIR)$(datadir)/kernel' 

