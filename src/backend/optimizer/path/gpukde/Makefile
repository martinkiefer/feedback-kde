#-------------------------------------------------------------------------
#
# Makefile--
#    Makefile for optimizer/path/gpukde
#
# IDENTIFICATION
#    src/backend/optimizer/path/gpukde/Makefile
#
#-------------------------------------------------------------------------

subdir = src/backend/optimizer/path/gpukde
top_builddir = ../../../../..
include $(top_builddir)/src/Makefile.global

OBJS = ocl_utilities.o ocl_estimator.o ocl_sample_maintenance.o ocl_model_maintenance.o

SUBDIRS = container

include $(top_srcdir)/src/backend/common.mk

# Install kernel files
.PHONY: install-data
install-data: all installdirs
	$(INSTALL_DATA) $(srcdir)/kernel/sum.cl '$(DESTDIR)$(datadir)/kernel/sum.cl'
	$(INSTALL_DATA) $(srcdir)/kernel/kde.cl '$(DESTDIR)$(datadir)/kernel/kde.cl'
	$(INSTALL_DATA) $(srcdir)/kernel/init.cl '$(DESTDIR)$(datadir)/kernel/init.cl'
	$(INSTALL_DATA) $(srcdir)/kernel/sample_maintenance.cl '$(DESTDIR)$(datadir)/kernel/sample_maintenance.cl'
	$(INSTALL_DATA) $(srcdir)/kernel/model_maintenance.cl '$(DESTDIR)$(datadir)/kernel/model_maintenance.cl'

installdirs:
	$(MKDIR_P) '$(DESTDIR)$(datadir)/kernel'

.PHONY: uninstall-data
uninstall-data:
	rm -f '$(DESTDIR)$(datadir)/kernel' 

